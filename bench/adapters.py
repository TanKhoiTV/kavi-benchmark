"""Candidate adapter base class + per-stage interfaces.

Each concrete candidate implements `_infer(item)` and declares its `stage` and
`id`. `run()` wraps timing + peak-RAM measurement and guarantees a `StageResult`
(even on failure) so one bad candidate never aborts the whole fan-out.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from .schema import EvalItem, StageResult


def peak_ram_mb() -> float:
    """Peak RSS in MB (process-wide, not per-candidate delta).

    Cross-platform via ``psutil``: ``peak_wset`` on Windows (peak working set),
    ``ru_maxrss`` via stdlib ``resource`` on Unix (peak resident set).
    """
    import sys

    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / (1024.0**2)

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return rss / divisor


class Candidate(ABC):
    stage: str = ""
    id: str = ""

    def run(self, item: EvalItem) -> StageResult:
        start = time.perf_counter()
        try:
            out_text, out_audio = self._infer(item)
            return StageResult(
                candidate_id=self.id,
                item_id=item.id,
                stage=self.stage,
                output_text=out_text,
                output_audio_path=out_audio,
                latency_s=round(time.perf_counter() - start, 4),
                peak_ram_mb=peak_ram_mb(),
            )
        except Exception as exc:  # noqa: BLE001 - one bad candidate must not kill the run
            return StageResult(
                candidate_id=self.id,
                item_id=item.id,
                stage=self.stage,
                latency_s=round(time.perf_counter() - start, 4),
                error=f"{type(exc).__name__}: {exc}",
            )

    def configure(self, config: dict) -> None:
        """Apply per-run config without reloading weights; ignore by default.

        Decode knobs such as ``beam_size`` are run configuration, not model
        identity, so a sweep reconfigures in place instead of paying a model load
        per setting (TanKhoiTV/kavi-benchmark#90 problem 4). Candidates whose
        knobs are fixed at construction simply keep the default no-op.
        """
        return None

    @abstractmethod
    def _infer(self, item: EvalItem) -> tuple[str | None, str | None]: ...
