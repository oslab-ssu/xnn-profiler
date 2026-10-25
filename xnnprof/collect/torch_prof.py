"""PyTorch Profiler 수집기 — 라벨 기반 통합.

프로파일러는 `record_function`으로 개시된 사용자 라벨을 이벤트 트리에 그대로
남긴다. hook이 통합 키 문자열 `fc1@t3`을 라벨로 개시하므로, 각 연산 이벤트의
부모를 거슬러 올라가 가장 가까운 라벨을 찾으면 그 연산이 어느 구간에 속하는지가
정해진다. 시계 원점을 맞추거나 시각 포함 여부를 판정할 필요가 없다.

구간별로 CPU self time뿐 아니라 GPU 커널 시간과 발행한 GPU op 수도 함께 모은다.
둘의 차이가 그 구간에서 GPU가 논 시간이다 — 연산량이 아니라 op 발행 횟수에
묶인 레이어가 여기서 드러난다.

`key_averages()`는 T회 호출을 이미 합산한 값이라 레이어·타임스텝으로 되돌릴 수
없으므로 쓰지 않는다. 라벨이 없는 실행(`emit_labels=False`, 또는 hook이 닿지
않는 함수형 호출)은 귀속되지 않은 몫으로 분리해 출처에 남긴다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

from ..schema import Trace
from .base import Collector, REGION
from .torch_hooks import INSTRUMENT_LABEL

LABEL = re.compile(r"^(?P<layer>.+)@t(?P<step>\d+)$")
PSEUDO = ("ProfilerStep",)
MAX_DEPTH = 64
INSTRUMENT = ("__instrument__", -1)


def is_pseudo(key: str) -> bool:
    return key.startswith("[") or key.startswith(PSEUDO)


def owner_of(ev) -> Optional[tuple[str, int]]:
    """이벤트가 속한 통합 키. 부모를 거슬러 올라가 가장 가까운 라벨을 찾는다.

    계측 자신의 라벨에 먼저 닿으면 측정 대상이 아니므로 `INSTRUMENT`를 돌려준다.
    """
    cur, depth = ev, 0
    while cur is not None and depth < MAX_DEPTH:
        nm = str(cur.name)
        if nm == INSTRUMENT_LABEL:
            return INSTRUMENT
        m = LABEL.match(nm)
        if m:
            return m.group("layer"), int(m.group("step"))
        cur = getattr(cur, "cpu_parent", None)
        depth += 1
    return None


class TorchProfilerCollector(Collector):
    name = "torch_profiler"
    attribution = REGION
    capabilities = ("op_time", "op_memory", "call_count", "label_join")

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
            events = list(self.prof.events())
        except Exception as e:
            return self.declare("failed", f"{type(e).__name__}: {e}")

        per_key: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        gpu_time: dict[tuple[str, int], float] = defaultdict(float)
        gpu_ops: dict[tuple[str, int], int] = defaultdict(int)
        orphan: dict[str, float] = defaultdict(float)
        orphan_count = 0
        instrument: dict[str, float] = defaultdict(float)

        for ev in events:
            op = str(ev.name)
            if is_pseudo(op) or LABEL.match(op) or op == INSTRUMENT_LABEL:
                continue                             # 라벨 자체는 연산이 아니다
            self_us = float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)
            dev_us = getattr(ev, "self_device_time_total", None)
            if dev_us is None:                       # torch 2.4 이전
                dev_us = getattr(ev, "self_cuda_time_total", 0.0)
            dev_us = float(dev_us or 0.0)
            n_kern = len(getattr(ev, "kernels", ()) or ())
            if self_us <= 0 and dev_us <= 0 and not n_kern:
                continue
            key = owner_of(ev)
            if key == INSTRUMENT:
                instrument[op] += self_us
            elif key is None:
                if self_us > 0:
                    orphan[op] += self_us
                    orphan_count += 1
            else:
                if self_us > 0:
                    per_key[key][op] += self_us
                gpu_time[key] += dev_us
                gpu_ops[key] += n_kern

        attributed = sum(sum(d.values()) for d in per_key.values())
        unattributed = sum(orphan.values())
        instrument_us = sum(instrument.values())
        total = attributed + unattributed          # 계측 비용은 모수에서 제외한다

        matched = 0
        for (lid, step), ops in per_key.items():
            r = trace.region(lid, step)
            if r is None:
                continue
            r.ops = {k: round(v, 3) for k, v in
                     sorted(ops.items(), key=lambda kv: -kv[1])}
            r.gpu_us = round(gpu_time[(lid, step)], 3)
            r.gpu_ops = gpu_ops[(lid, step)]
            matched += 1

        top_orphan = sorted(orphan.items(), key=lambda kv: -kv[1])[:10]
        top_inst = sorted(instrument.items(), key=lambda kv: -kv[1])[:10]
        status = "captured" if matched and not orphan else "partial"
        reason = "" if status == "captured" else (
            "라벨 밖에서 실행된 연산이 존재 — 모듈 트리에 나타나지 않는 "
            "함수형 호출(F.linear 등)과 프로파일러 자체 비용이 여기 포함된다")
        return self.declare(
            status, reason,
            keyed_regions=len(per_key),
            matched_regions=matched,
            attributed_us=round(attributed, 3),
            unattributed_us=round(unattributed, 3),
            attribution_rate=round(attributed / total, 6) if total else 0.0,
            unattributed_events=orphan_count,
            unattributed_top=[{"op": k, "self_time_us": round(v, 3)}
                              for k, v in top_orphan],
            gpu_us=round(sum(gpu_time.values()), 3),
            gpu_ops=sum(gpu_ops.values()),
            instrument_us=round(instrument_us, 3),
            instrument_top=[{"op": k, "self_time_us": round(v, 3)}
                            for k, v in top_inst])
