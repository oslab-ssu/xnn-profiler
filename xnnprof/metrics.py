"""차별화 지표 — 절감 상한.

기존 프로파일러는 '지금 얼마 걸렸나'를 준다. 여기서는 '바꾸면 얼마 버나'를 낸다.
전부 상한(upper bound)이다. 실제 이득은 커널·런타임 구현에 달려 있다.
"""

from __future__ import annotations

from .schema import Trace


def headroom(trace: Trace) -> dict:
    exec_macs = trace.executed_macs()
    useful = trace.useful_macs()
    looped = trace.looped_macs()

    sparse_skippable = exec_macs - useful
    spike_part = sum((l.macs - (l.useful_macs if l.useful_macs is not None else l.macs))
                     for l in trace.layers if l.spike_driven)

    first_useful, out_layer = None, None
    for sp in trace.spikes:
        series = (sp.metrics or {}).get("per_timestep_rate") or []
        nz = [i for i, r in enumerate(series) if r and r > 0]
        if nz and (first_useful is None or nz[0] > first_useful):
            first_useful, out_layer = nz[0], sp.layer_id

    T = trace.timesteps or 0
    t_trim = (looped * (first_useful / T)) if (T and first_useful) else 0

    return {
        "executed_macs": exec_macs,
        "useful_macs": useful,
        "sparse_kernel": {
            "skippable_macs": sparse_skippable,
            "fraction_of_total": round(sparse_skippable / exec_macs, 6) if exec_macs else 0.0,
            "from_spike_driven_layers": spike_part,
            "note": "입력이 0인 MAC. sparse 커널이 전부 생략한다고 가정한 상한",
        },
        "time_loop": {
            "looped_macs": looped,
            "fraction_of_total": round(looped / exec_macs, 6) if exec_macs else 0.0,
            "note": "시간축 루프 안의 연산. T 단축·이벤트 구동이 영향을 주는 범위",
        },
        "timestep_trim": {
            "first_output_step": first_useful,
            "output_layer": out_layer,
            "skippable_macs": int(t_trim),
            "fraction_of_total": round(t_trim / exec_macs, 6) if exec_macs else 0.0,
            "note": "출력 레이어가 발화하기 전 스텝. 조기 종료 여지의 상한",
        },
        "persistent_state_bytes": sum(l.state_bytes for l in trace.layers),
    }
