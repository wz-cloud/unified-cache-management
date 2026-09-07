"""Exercise Qwen2.5 KV-cache registration without loading model weights.

Run this inside the vLLM/UCM environment, for example:

    python test/debug_qwen25_register_kv_caches.py

The script creates small CPU tensors with the same logical shape and resolved
layout metadata as Qwen2.5-14B.  UCM store creation is replaced with a stub so
the test does not touch the Posix backend or start GC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from ucm.integration.vllm import ucm_connector as connector_module  # noqa: E402


NUM_LAYERS = 48
NUM_KV_HEADS = 8
HEAD_SIZE = 128
BLOCK_SIZE = 64
NUM_BLOCKS = 4
DTYPE = torch.bfloat16


def build_qwen25_config() -> KVCacheConfig:
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(NUM_LAYERS)]
    page_size = BLOCK_SIZE * NUM_KV_HEADS * (HEAD_SIZE * 2) * DTYPE.itemsize
    layer_stride = NUM_BLOCKS * page_size

    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=NUM_LAYERS * layer_stride,
                layers=layer_names,
                layer_stride=layer_stride,
                block_stride=page_size,
                offset=0,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=NUM_KV_HEADS,
                    head_size=HEAD_SIZE,
                    dtype=DTYPE,
                    head_size_v=HEAD_SIZE,
                ),
            )
        ],
        kv_cache_layout="LBNHC",
    )


def build_fake_vllm_config() -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_hidden_layers=NUM_LAYERS),
    )
    parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
    )


def build_fake_kv_caches() -> dict[str, torch.Tensor]:
    # This is the same logical per-layer shape as the real Qwen2.5 cache, but
    # with four blocks so the diagnostic remains small enough for CPU memory.
    shape = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE * 2)
    return {
        f"model.layers.{i}.self_attn.attn": torch.empty(shape, dtype=DTYPE)
        for i in range(NUM_LAYERS)
    }


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-addresses",
        action="store_true",
        help="Print the first block address for the first three layers.",
    )
    args = parser.parse_args()

    kv_cache_config = build_qwen25_config()
    kv_caches = build_fake_kv_caches()
    vllm_config = build_fake_vllm_config()

    connector = object.__new__(connector_module.UCMLayerWiseConnector)
    connector.kv_cache_dtype = None
    connector._vllm_config = vllm_config
    connector._kv_cache_config = kv_cache_config
    connector.launch_config = {"use_layerwise": True}
    connector.device_id = 0
    connector.device = None

    # Keep this test focused on register_kv_caches and layout construction.
    captured = {}

    def fake_create_store(*, kv_cache_layout, cpu_affinity_cores=None):
        captured["layout"] = kv_cache_layout
        captured["cpu_affinity_cores"] = cpu_affinity_cores
        return "STORE_CREATION_SKIPPED"

    connector._create_store = fake_create_store
    original_create_device = connector_module.create_device
    connector_module.create_device = lambda: object()
    try:
        connector.register_kv_caches(kv_caches)
    finally:
        connector_module.create_device = original_create_device

    layout = captured["layout"]
    expected_page_size = (
        BLOCK_SIZE * NUM_KV_HEADS * (HEAD_SIZE * 2) * DTYPE.itemsize
    )
    expected_layer_stride = NUM_BLOCKS * expected_page_size

    assert layout.block_size == expected_page_size * NUM_LAYERS
    assert layout.base_ptrs.shape == (NUM_LAYERS, 1)
    assert layout.tensor_size_lists.shape == (NUM_LAYERS, 1)
    assert layout.block_stride_lists.shape == (NUM_LAYERS, 1)
    assert all(
        int(value) == expected_page_size for value in layout.tensor_size_lists[:, 0]
    )
    assert all(
        int(value) == expected_page_size for value in layout.block_stride_lists[:, 0]
    )
    assert all(
        int(value) == expected_layer_stride for value in layout.buffer_sizes[:, 0]
    )

    print("register_kv_caches: PASS")
    print(f"kv cache entries: {len(kv_caches)}")
    print(f"per-layer shape: {tuple(next(iter(kv_caches.values())).shape)}")
    print(f"layout: {kv_cache_config.kv_cache_layout}")
    print(f"tensor_size_lists[0]: {int(layout.tensor_size_lists[0, 0])} bytes")
    print(f"block_stride_lists[0]: {int(layout.block_stride_lists[0, 0])} bytes")
    print(f"buffer_sizes[0]: {int(layout.buffer_sizes[0, 0])} bytes")
    print("UCM store creation: SKIPPED")

    if args.show_addresses:
        addresses = layout.extract_block_addrs([0, 1], layer_first=True)
        for layer_id in range(3):
            print(
                f"layer {layer_id}: block0={int(addresses[0, layer_id, 0])}, "
                f"block1={int(addresses[1, layer_id, 0])}"
            )


if __name__ == "__main__":
    run()
