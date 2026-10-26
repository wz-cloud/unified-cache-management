"""Test real multiprocess Prometheus export without native UCM, SGLang or GPUs.

Run with pytest --noconftest test/test_sglang_metrics.py. Each case uses a fresh
interpreter because Prometheus selects its storage mode at import time.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = r"""
import logging, os, sys, time, types
from pathlib import Path
import yaml

def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    sys.modules[name] = result

module("ucm", __path__=[str(Path.cwd() / "ucm")])
module("ucm.shared", __path__=[str(Path.cwd() / "ucm/shared")])
module("ucm.logger", init_logger=logging.getLogger)
snapshot = ({}, {}, {})
registered, events = [], []
drains = 0
def drain():
    global snapshot, drains
    result, snapshot = snapshot, ({}, {}, {})
    drains += 1
    return result
module("ucm.shared.metrics", ucmmetrics=types.SimpleNamespace(
    set_up=lambda: None, create_stats=lambda *a: registered.append(a),
    get_all_stats_and_clear=drain))
module("torch")
for name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
    module(name, __path__=[])
module("sglang.srt.distributed.parallel_state",
       get_world_group=lambda: types.SimpleNamespace(local_rank=0))
class Store:
    def close(self):
        events.append("store.close")
def create(*args):
    events.append("store.create")
    return Store()
module("ucm.store.factory_v1", UcmConnectorFactoryV1=types.SimpleNamespace(create_connector=create))
from ucm.integration.sglang import ucm_connector as c

config = {"log_interval": 0.02,
    "consumers": {"multiproc": True, "vllm_connector": False},
    "counter": [{"name": "posix_s2h_bytes_total"}],
    "gauge": [{"name": "posix_store_used_bytes", "multiprocess_mode": "livemostrecent"}],
    "histogram": [{"name": "posix_load_task_duration_ms", "buckets": [1, 10]}]}
config_path = Path(sys.argv[1]) / "metrics.yaml"
def write_config():
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
write_config()
launch = {"metrics_config_path": str(config_path), "ucm_connector_name": "FakeStore",
          "ucm_connector_config": {"storage_backends": "/tmp/fake-store"}}
storage = types.SimpleNamespace(model_name="test-model", is_mla_model=False,
    tp_rank=0, tp_size=2, pp_rank=0, pp_size=1, dp_rank=0, attn_cp_rank=0,
    enable_storage_metrics=True, extra_config={"kv_connector_extra_config": launch})
pool = types.SimpleNamespace(dtype="f16", page_size=1, kv_buffer=object(),
                             get_size_per_token=lambda: 128)
def start():
    return c.SglangUcmConnector.from_hicache(storage, pool)
def scrape():
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return generate_latest(registry).decode()
def value(name, **labels):
    from prometheus_client.parser import text_string_to_metric_families
    matches = [s.value for f in text_string_to_metric_families(scrape()) for s in f.samples
               if s.name == name and all(s.labels.get(k) == v for k, v in labels.items())]
    assert len(matches) == 1, (name, labels, matches)
    return matches[0]
"""


def run_child(tmp_path, code, *, multiprocess=True):
    env = os.environ.copy()
    env.pop("prometheus_multiproc_dir", None)
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    if multiprocess:
        env["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", BOOTSTRAP + "\n" + textwrap.dedent(code), str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_workers_existing_thread_and_asgi_endpoint(tmp_path):
    for rank, amount in [(0, 4096), (1, 8192)]:
        run_child(
            tmp_path,
            f"""
            storage.tp_rank = {rank}
            def create_with_metrics(*args):
                global snapshot
                assert registered, "Register metrics before creating stores"
                snapshot = ({{"posix_s2h_bytes_total": {amount}}},
                            {{"posix_store_used_bytes": {amount}}},
                            {{"posix_load_task_duration_ms": ([1, 1, 0], 6.0)}})
                return Store()
            c.UcmConnectorFactoryV1.create_connector = create_with_metrics
            connector = start()
            try:
                from prometheus_client import Counter
                Counter("sglang:generated_tokens_total", "Generated tokens").inc(3)
                deadline = time.monotonic() + 5
                while (snapshot[0] or drains < 3) and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert not snapshot[0] and drains >= 3
                assert value("ucm:posix_s2h_bytes_total", worker_id="dp0_tp{rank}_pp0_attn_cp0") == {amount}
                try:
                    start()
                except RuntimeError as error:
                    assert "only one" in str(error)
                else:
                    raise AssertionError("Duplicate logger accepted")
            finally:
                connector.close()
            assert not connector._metrics_logger.thread.is_alive()
        """,
        )
    assert len(list(tmp_path.glob("counter_*.db"))) == 2
    run_child(
        tmp_path,
        """
        import asyncio
        from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        responses = []
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}
        async def send(message):
            responses.append(message)
        asyncio.run(make_asgi_app(registry=registry)(
            {"type": "http", "method": "GET", "path": "/metrics", "headers": [],
             "query_string": b""}, receive, send))
        assert responses[0]["status"] == 200
        text = b"".join(r.get("body", b"") for r in responses).decode()
        assert "sglang:generated_tokens_total 6.0" in text
        assert "ucm:posix_s2h_bytes_total" in text
        worker = "dp0_tp1_pp0_attn_cp0"
        assert value("ucm:posix_s2h_bytes_total", worker_id=worker) == 8192
        assert value("ucm:posix_store_used_bytes", worker_id=worker) == 8192
        assert value("ucm:posix_load_task_duration_ms_count", worker_id=worker) == 2
        assert value("ucm:posix_load_task_duration_ms_sum", worker_id=worker) == 6
        assert value("ucm:posix_load_task_duration_ms_bucket", worker_id=worker, le="10.0") == 2
        assert scrape() == scrape()
    """,
    )


@pytest.mark.parametrize("disable", ["sglang", "ucm", "no_path"])
def test_disabled_does_not_import_prometheus(tmp_path, disable):
    run_child(
        tmp_path,
        f"""
        if {disable!r} == "sglang":
            storage.enable_storage_metrics = False
        elif {disable!r} == "ucm":
            launch["enable_metrics"] = False
        else:
            launch.pop("metrics_config_path")
        connector = start()
        assert connector._metrics_logger is None
        assert "prometheus_client" not in sys.modules
        connector.close()
    """,
        multiprocess=False,
    )


@pytest.mark.parametrize("early_import", [False, True])
def test_multiprocess_environment_errors(tmp_path, early_import):
    code = ""
    if early_import:
        code = f'from prometheus_client import Counter\nos.environ["PROMETHEUS_MULTIPROC_DIR"] = {str(tmp_path)!r}\n'
    run_child(
        tmp_path,
        code + """
try:
    start()
except RuntimeError as error:
    assert "PROMETHEUS_MULTIPROC_DIR" in str(error)
else:
    raise AssertionError("Invalid multiprocess environment accepted")
assert events == []
""",
        multiprocess=False,
    )


def test_vllm_only_yaml_rejected(tmp_path):
    run_child(
        tmp_path,
        """
        config["consumers"] = {"vllm_connector": True}
        write_config()
        try:
            start()
        except ValueError as error:
            assert "enable multiproc" in str(error)
        else:
            raise AssertionError("Wrong consumer config accepted")
        assert events == []
    """,
    )


def test_lifecycle_creation_failure_hybrid_pools_and_close_failure(tmp_path):
    run_child(
        tmp_path,
        """
        class Logger:
            def shutdown(self):
                events.append("shutdown")
        def create_logger(*args):
            events.append("logger.create")
            return Logger()
        c._create_metrics_logger = create_logger
        connector = start()
        assert events == ["logger.create", "store.create"]
        connector.pool_components = {"mamba": [(Store(), 1)], "swa": [(Store(), 1)]}
        connector.close()
        connector.close()
        assert events == ["logger.create", "store.create", "store.close",
                          "store.close", "store.close", "shutdown"]
        events.clear()
        def fail(*args):
            raise ValueError("store creation failed")
        c.UcmConnectorFactoryV1.create_connector = fail
        try:
            start()
        except ValueError:
            pass
        else:
            raise AssertionError("Expected store creation failure")
        assert events == ["logger.create", "shutdown"]
        events.clear()
        pool.kv_buffer = None
        connector = start()
        assert events == ["logger.create"]
        class BrokenStore:
            def close(self):
                events.append("broken.close")
                raise RuntimeError("close failed")
        connector.pool_components = {"pool": [(BrokenStore(), 1), (Store(), 1)]}
        try:
            connector.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected store close failure")
        assert events == ["logger.create", "broken.close", "store.close", "shutdown"]
    """,
    )
