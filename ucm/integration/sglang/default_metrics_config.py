"""Built-in SGLang metrics for HiCache and future Linker adapters.

Common definitions are copied from UCM defaults; vLLM configuration is never mutated.
Keep these additions in sync with examples/metrics/metrics_configs_sglang.yaml.
"""

from ucm.default_metrics_config import get_default_metrics_config


def get_default_sglang_metrics_config() -> dict:
    """Return independent definitions without reading any YAML at runtime."""
    config = get_default_metrics_config()
    config.pop("vllm_connector_prefix", None)
    config["multiproc_prefix"] = "ucm:"
    config["consumers"] = {"multiproc": True, "vllm_connector": False}
    config["counter"] = [
        {
            "name": "sglang_connector_load_requests_total",
            "documentation": "SGLang connector load: Physical storage transfer calls attempted, per worker and pool or layer",
        },
        {
            "name": "sglang_connector_load_bytes_total",
            "documentation": "SGLang connector load: Bytes transferred by successful storage calls, local to this worker",
        },
        {
            "name": "sglang_connector_load_submit_errors_total",
            "documentation": "SGLang connector load: Storage submission failures",
        },
        {
            "name": "sglang_connector_load_wait_errors_total",
            "documentation": "SGLang connector load: Storage completion wait failures",
        },
        {
            "name": "sglang_connector_dump_requests_total",
            "documentation": "SGLang connector dump: Physical storage transfer calls attempted, per worker and pool or layer",
        },
        {
            "name": "sglang_connector_dump_bytes_total",
            "documentation": "SGLang connector dump: Bytes transferred by successful storage calls, local to this worker",
        },
        {
            "name": "sglang_connector_dump_submit_errors_total",
            "documentation": "SGLang connector dump: Storage submission failures",
        },
        {
            "name": "sglang_connector_dump_wait_errors_total",
            "documentation": "SGLang connector dump: Storage completion wait failures",
        },
    ] + config["counter"]
    config["histogram"] = [
        {
            "name": "sglang_connector_load_pages",
            "documentation": "SGLang connector load: Successful storage batch size in pool pages; not deduplicated across TP/PP, pools or layers",
            "buckets": [1, 5, 10, 50, 100, 128, 256, 512],
        },
        {
            "name": "sglang_connector_load_duration_ms",
            "documentation": "SGLang connector load: Storage submission-to-completion duration including failed attempts (ms); excludes buffer preparation and restoration",
            "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        },
        {
            "name": "sglang_connector_load_bandwidth_gbps",
            "documentation": "SGLang connector load: Successful call bytes divided by storage submission-to-completion time (GB/s, decimal bytes); not device-to-storage end-to-end bandwidth",
            "buckets": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
        },
        {
            "name": "sglang_connector_dump_pages",
            "documentation": "SGLang connector dump: Successful storage batch size in pool pages; not deduplicated across TP/PP, pools or layers",
            "buckets": [1, 5, 10, 50, 100, 128, 256, 512],
        },
        {
            "name": "sglang_connector_dump_duration_ms",
            "documentation": "SGLang connector dump: Storage submission-to-completion duration including failed attempts (ms); excludes buffer preparation and restoration",
            "buckets": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        },
        {
            "name": "sglang_connector_dump_bandwidth_gbps",
            "documentation": "SGLang connector dump: Successful call bytes divided by storage submission-to-completion time (GB/s, decimal bytes); not device-to-storage end-to-end bandwidth",
            "buckets": [0.1, 0.5, 1, 5, 10, 20, 50, 100],
        },
    ] + config["histogram"]
    return config
