"""신규 측정 항목 중 trace 전체를 보아야 산출되는 값.

레이어 단위 값은 수집기가, 스파이크 배열에서 나오는 값은 observations가 낸다.
여기서는 그 둘을 모아야 정해지는 두 항목만 산출한다.

  출력 발생 지연  출력 레이어의 스파이크율이 최초로 0이 아닌 타임스텝
  상태 메모리     타임스텝 사이에 유지되는 상태 텐서 크기의 합
"""

from __future__ import annotations

from .schema import Trace


def execution_order(trace: Trace) -> list[str]:
    """구간의 시작 시각 순으로 레이어가 처음 등장하는 순서."""
    seen, out = set(), []
    for r in sorted(trace.regions, key=lambda r: r.ts_us):
        if r.layer_id not in seen:
            seen.add(r.layer_id)
            out.append(r.layer_id)
    return out


def output_layer(trace: Trace) -> str:
    """실행 순서상 가장 마지막에 오는 스파이크 레이어를 출력 레이어로 본다."""
    order = execution_order(trace)
    spiking = {sp.layer_id for sp in trace.spikes}
    for lid in reversed(order):
        if lid in spiking:
            return lid
    return ""


def output_latency(trace: Trace):
    """스파이크 레이어가 없는 모델(순수 DNN)에서는 정의되지 않는다."""
    lid = output_layer(trace)
    sp = next((s for s in trace.spikes if s.layer_id == lid), None)
    if sp is None:
        return None
    series = (sp.metrics or {}).get("per_timestep_rate") or []
    first = next((i for i, r in enumerate(series) if r and r > 0), None)
    return {
        "layer_id": lid,
        "first_spike_step": first,
        "timesteps": len(series),
        "unit": "타임스텝 인덱스 (0부터)",
    }


def snn_metrics(trace: Trace) -> dict:
    """산출되지 않는 항목은 키를 두지 않는다. '0'과 '정의되지 않음'은 다르다."""
    out = {"executed_macs": trace.executed_macs()}
    state = sum(l.state_bytes for l in trace.layers)
    lat = output_latency(trace)
    if lat is not None:
        out["output_latency"] = lat
    if state:
        out["state_bytes"] = state
    syn = [sp.synops for sp in trace.spikes if sp.synops is not None]
    if syn:
        out["synops"] = sum(syn)
    return out
