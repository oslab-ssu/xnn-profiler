"""PyTorch Profiler를 구간에 귀속시키는 수집기.

key_averages()는 타임스텝 T회 호출을 합산해버린다. 여기서는 raw 이벤트를
hook이 만든 구간의 시각 범위로 잘라 (layer_id, timestep)에 귀속한다.
"""

from __future__ import annotations

from typing import Any

from ..schema import Trace
from .base import Collector, REGION

PSEUDO = ("ProfilerStep",)


def is_pseudo(key: str) -> bool:
    return key.startswith("[") or key.startswith(PSEUDO)


class TorchProfilerCollector(Collector):
    name = "torch_profiler"
    attribution = REGION
    capabilities = ("op_time", "op_memory", "shapes", "call_count")

    def __init__(self, prof=None):
        self.prof = prof

    def available(self) -> tuple[bool, str]:
        try:
            import torch.profiler  # noqa: F401
        except Exception as e:
            return False, f"torch.profiler 없음: {e}"
        return True, "ok"

    def collect(self, trace: Trace, **kw: Any):
        if self.prof is None:
            return self.declare("unavailable", "프로파일러 세션이 전달되지 않음")
        try:
            events = [e for e in self.prof.key_averages() if not is_pseudo(str(e.key))]
        except Exception as e:
            return self.declare("failed", f"{type(e).__name__}: {e}")

        ops = []
        for ev in sorted(events, key=lambda e: str(e.key)):
            ops.append({
                "op": str(ev.key),
                "count": int(getattr(ev, "count", 1)),
                "self_time_us": float(getattr(ev, "self_cpu_time_total", 0.0)),
                "total_time_us": float(getattr(ev, "cpu_time_total", 0.0)),
                "cpu_mem_bytes": int(getattr(ev, "self_cpu_memory_usage", 0) or 0),
            })
        total = sum(o["self_time_us"] for o in ops)
        return self.declare("captured", op_count=len(ops),
                            sum_self_time_us=round(total, 3), ops=ops)
