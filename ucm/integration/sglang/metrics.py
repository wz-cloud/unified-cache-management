"""Per-worker storage-call observations; export uses the existing UCM logger."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class IoObservation:
    submitted: bool = False
    failed: bool = False


@contextmanager
def measure_io(enabled: bool, is_set: bool, pages: int, byte_count: Callable[[], int]):
    """Observe one physical v1 call or v2 pool call, never an inference batch.

    Duration covers submission through wait, including failed attempts. Page and
    bandwidth histograms and byte counters contain successful transfers only.
    Callers mark caught wait failures so existing return/exception behavior is
    preserved. No native metrics import, clock read, or byte calculation occurs
    when disabled. No metric collection/draining happens on the business thread.
    """
    observation = IoObservation()
    if not enabled or pages == 0:
        yield observation
        return

    started = time.perf_counter()
    try:
        yield observation
    except BaseException:
        observation.failed = True
        raise
    finally:
        elapsed = max(time.perf_counter() - started, 0.0)
        prefix = "sglang_connector_dump" if is_set else "sglang_connector_load"
        values = {
            f"{prefix}_requests_total": 1,
            f"{prefix}_duration_ms": elapsed * 1000,
        }
        if observation.failed:
            phase = "wait" if observation.submitted else "submit"
            values[f"{prefix}_{phase}_errors_total"] = 1
        # Observability must not change the result of a storage operation.
        try:
            if not observation.failed:
                size = int(byte_count())
                if size < 0:
                    raise ValueError("negative storage transfer size")
                values[f"{prefix}_bytes_total"] = size
                values[f"{prefix}_pages"] = pages
                if elapsed > 0:
                    values[f"{prefix}_bandwidth_gbps"] = size / elapsed / 1e9
            from ucm.shared.metrics import ucmmetrics

            ucmmetrics.update_stats(values)
        except Exception:
            logger.warning(
                "Failed to record SGLang connector I/O metrics", exc_info=True
            )
