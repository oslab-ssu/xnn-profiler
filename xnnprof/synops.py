"""SynOps = 스파이크 수 x fan-out.

fan-out을 하드코딩하지 않고 그래프에서 읽는다. 기존 SNN-Specialized는
`spk1.sum() * 64`, `spk2.sum() * 10`처럼 상수로 박혀 있어 레이어 폭이 바뀌면
조용히 틀린 값이 나온다.

규칙: 뉴런 N개인 스파이크 레이어의 소비자는 '실행 순서상 그 뒤에 오면서
입력이 이진이고 in_features == N'인 레이어다. 그 레이어의 out_features가
fan-out이다. 소비자를 못 찾으면 None으로 두고 추정하지 않는다.
"""

from __future__ import annotations

from .schema import Trace


def _order(trace: Trace) -> list[str]:
    seen, out = set(), []
    for r in sorted(trace.regions, key=lambda r: r.ts_us):
        if r.layer_id not in seen:
            seen.add(r.layer_id)
            out.append(r.layer_id)
    return out


def annotate(trace: Trace) -> None:
    order = _order(trace)
    for sp in trace.spikes:
        try:
            start = order.index(sp.layer_id)
        except ValueError:
            start = -1
        consumer = None
        for lid in order[start + 1:]:
            l = trace.layer(lid)
            if l is None or not l.spike_driven:
                continue
            if (l.extra.get("fan") or {}).get("in") == sp.neurons:
                consumer = l
                break
        if consumer is None:
            sp.fanout_from = "output" if start >= 0 and start == len(order) - 1 else "unknown"
            continue
        fan = (consumer.extra.get("fan") or {}).get("out")
        if not fan:
            continue
        sp.fanout = int(fan)
        sp.fanout_from = consumer.layer_id
        sp.synops = int((sp.metrics or {}).get("spikes", 0)) * int(fan)
