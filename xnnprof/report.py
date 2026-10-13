"""표준출력 요약."""

from __future__ import annotations

from .schema import Trace


def summary(trace: Trace) -> str:
    out = []
    e = trace.executed_macs() or 1
    out.append(f"model={trace.model_id}  T={trace.timesteps}  batch={trace.batch}")
    out.append("")
    out.append(f"{'레이어':18s} {'도메인':12s} {'호출':>4s} {'MAC':>12s} {'비중':>6s} "
               f"{'nonzero':>8s} {'유효 MAC':>12s}")
    for l in trace.layers:
        dom = "spike-driven" if l.spike_driven else l.domain
        nz = "—" if l.input_nonzero_fraction is None else f"{l.input_nonzero_fraction:.1%}"
        um = "—" if l.useful_macs is None else f"{l.useful_macs:,}"
        out.append(f"{l.layer_id:18s} {dom:12s} {l.calls:4d} {l.macs:12,} "
                   f"{l.macs/e:6.1%} {nz:>8s} {um:>12s}")
    h = trace.headroom
    if h:
        out.append("")
        out.append(f"실행 MAC            {h['executed_macs']:,}")
        out.append(f"유효 MAC            {h['useful_macs']:,}")
        out.append(f"sparse 커널 상한    {h['sparse_kernel']['skippable_macs']:,}"
                   f" ({h['sparse_kernel']['fraction_of_total']:.2%})"
                   f"  스파이크 구동분 {h['sparse_kernel']['from_spike_driven_layers']:,}")
        out.append(f"시간축 루프 안      {h['time_loop']['looped_macs']:,}"
                   f" ({h['time_loop']['fraction_of_total']:.1%})")
        tt = h["timestep_trim"]
        out.append(f"출력 첫 발화 스텝   {tt['first_output_step']} ({tt['output_layer']})"
                   f"  조기종료 상한 {tt['fraction_of_total']:.2%}")
        out.append(f"지속 상태 메모리    {h['persistent_state_bytes']:,} bytes")
    out.append("")
    out.append("출처")
    for s in trace.sources:
        out.append(f"  {s.collector:16s} {s.status:12s} 귀속={s.attribution:15s} "
                   f"{s.reason}")
    for sp in trace.spikes:
        m = sp.metrics
        if sp.synops is not None:
            syn = f"SynOps {sp.synops:,} (fan-out {sp.fanout} <- {sp.fanout_from})"
        elif sp.fanout_from == "output":
            syn = "SynOps — (출력층, fan-out 없음)"
        else:
            syn = "SynOps — (소비자 미식별)"
        out.append(f"  spike {sp.layer_id:10s} 발화율 {m['mean_rate']:.4f}  "
                   f"dead {m['dead_fraction']:.4f}  sat {m['saturated_fraction']:.4f}  {syn}")
        top = m.get("top_neurons") or []
        if top:
            s = "  ".join(f"#{d['neuron']}:{d['spikes']}" for d in top[:5])
            out.append(f"      상위 뉴런  {s}")
    return "\n".join(out)
