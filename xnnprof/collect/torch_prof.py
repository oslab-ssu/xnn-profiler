"""PyTorch Profiler 수집기.

⚠ 아직 구간(region)에 귀속하지 않는다. key_averages()는 T회 호출을 이미
합산한 값이라 레이어·타임스텝으로 되돌릴 수 없고, raw 이벤트를 쓰려면
프로파일러 타임베이스와 hook의 perf_counter 원점을 맞춰야 한다. 그 전까지는
attribution을 model-global로 선언한다 — 실제보다 세밀하게 선언하면
sources의 의미가 없어진다.
"""

from __future__ import annotations

from typing import Any

from ..schema import Trace
from .base import Collector, MODEL_GLOBAL

PSEUDO = ("ProfilerStep",)


def is_pseudo(key: str) -> bool:
    return key.startswith("[") or key.startswith(PSEUDO)


class TorchProfilerCollector(Collector):
    name = "torch_profiler"
    attribution = MODEL_GLOBAL
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
        return self.declare(
            "partial", "구간 귀속 미구현 — op 단위 집계만. 공통 시간 원점 필요",
            op_count=len(ops), sum_self_time_us=round(total, 3), ops=ops)
