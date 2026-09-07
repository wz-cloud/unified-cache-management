"""Real UCM scheduler/GC and small CUDA round-trip diagnostics, no weights.

Requires Linux and installed UCM native libraries. No vLLM engine is started.
CPU mode never allocates CUDA tensors. Test files are retained for inspection.
"""

import argparse
import faulthandler
import gc
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager


@contextmanager
def stage(name):
    start = time.monotonic()
    print(f"BEGIN {name}", flush=True)
    try:
        yield
    except BaseException:
        print(f"FAIL {name}", flush=True)
        raise
    else:
        print(f"END {name}: {time.monotonic() - start:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--storage-parent", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=1,
                        help="KV head partition only; does not launch TP processes.")
    parser.add_argument("--observe-seconds", type=int, default=35)
    parser.add_argument("--trace-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.observe_seconds < 0 or args.trace_seconds <= 0:
        parser.error("observe-seconds must be >= 0; trace-seconds must be > 0")
    if args.mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.trace_seconds, repeat=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    worker = scheduler = None
    try:
        with stage("import native store"):
            from ucm.store.pipeline.connector import UcmPipelineStore

        args.storage_parent.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(prefix="ucm-qwen25-", dir=args.storage_parent)
        print(f"Isolated storage (retained): {directory}", flush=True)
        page = 64 * (8 // args.tp_size) * 256 * 2
        config = {
            "store_pipeline": "Cache|Posix",
            "storage_backends": [directory],
            "unique_id": secrets.token_hex(16),
            "device_id": -1,
            "block_size": 48 * page,
            "local_rank_size": 1,
            "share_buffer_enable": False,
            "cache_buffer_capacity_gb": 1,
            "cache_load_exclusive_buffer_number": 4,
            "cache_load_backend_only": True,
            "waiting_queue_depth": 128,
            "running_queue_depth": 128,
            "stream_number": 1,
            "timeout_ms": 30000,
            "io_direct": True,
            "posix_io_engine": "psync",
            "posix_data_trans_concurrency": 2,
            "posix_lookup_concurrency": 2,
            "posix_gc_enable": True,
            "posix_capacity_gb": 1,
            "posix_gc_concurrency": 2,
            "posix_gc_check_interval_sec": 5,
        }
        ids = [secrets.token_bytes(16) for _ in range(4)]
        if args.mode == "cuda":
            with stage("allocate shared CUDA KV backing"):
                import torch

                torch.cuda.set_device(args.device)
                # Physical LBNHC, exposed as logical BHNC views per layer.
                backing = torch.empty(
                    (48, 4, 64, 8 // args.tp_size, 256),
                    dtype=torch.bfloat16, device=f"cuda:{args.device}",
                )
                caches = [backing[i].permute(0, 2, 1, 3) for i in range(48)]
                for i, tensor in enumerate(caches):
                    for b in range(4):
                        tensor[b].fill_(i * 4 + b + 1)
                torch.cuda.synchronize()
                print(f"KV allocation: {backing.numel() * 2 / 2**20:.1f} MiB",
                      flush=True)
            with stage("create real worker Cache|Posix"):
                worker = UcmPipelineStore(config | {
                    "device_id": args.device,
                    "posix_gc_enable": False,
                    "tensor_size_list": [page],
                    "shard_size": page,
                    "gpu_kv_buffer_addrs": [backing.data_ptr()],
                    "gpu_kv_buffer_sizes": [backing.numel() * 2],
                })
        with stage("create real scheduler Cache|Posix (GC enabled)"):
            scheduler = UcmPipelineStore(config)
        with stage("scheduler lookup: fresh IDs must miss"):
            if any(scheduler.lookup(ids)):
                raise AssertionError("Fresh block IDs unexpectedly exist")
        if worker is not None:
            with stage("dump all 48 layer shards"):
                for i, tensor in enumerate(caches):
                    addresses = [[tensor[b].data_ptr()] for b in range(4)]
                    worker.wait(worker.dump_data(ids, [i] * 4, addresses))
            with stage("scheduler lookup: all blocks must exist"):
                if not all(scheduler.lookup(ids)):
                    raise AssertionError("Not all dumped blocks are visible")
            with stage("clear GPU cache and load from backend"):
                backing.zero_()
                torch.cuda.synchronize()
                for i, tensor in enumerate(caches):
                    addresses = [[tensor[b].data_ptr()] for b in range(4)]
                    worker.wait(worker.load_data(ids, [i] * 4, addresses))
                torch.cuda.synchronize()
            with stage("verify every restored element"):
                for i, tensor in enumerate(caches):
                    for b in range(4):
                        if not bool(torch.all(tensor[b] == i * 4 + b + 1)):
                            raise AssertionError(f"Corrupt layer={i}, block={b}")
        with stage("observe GC checks (not forced eviction)"):
            deadline = time.monotonic() + args.observe_seconds
            while time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
        print("Operations PASS; releasing stores next", flush=True)
    finally:
        with stage("destroy worker and scheduler stores"):
            worker = None
            scheduler = None
            gc.collect()
        faulthandler.cancel_dump_traceback_later()
    print("PASS: store lifecycle completed", flush=True)


if __name__ == "__main__":
    main()
