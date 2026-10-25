"""표준출력 요약."""

from __future__ import annotations

from .schema import Trace


def summary(trace: Trace) -> str:
    out = []
    e = trace.executed_macs() or 1
    dev = (trace.env or {}).get("device", "cpu")
    rt = (trace.env or {}).get("region_timing", "")
    out.append(f"model={trace.model_id}  T={trace.timesteps}  batch={trace.batch}  "
               f"device={dev}" + (f"  구간시간={rt}" if rt else ""))
    if rt == "cpu-launch-window" and dev.startswith("cuda"):
        out.append("  주의: 구간 시간은 CPU 쪽 실행 요청 구간이다. "
                   "실제 GPU 시간은 --sync-regions 또는 Nsight Systems로 확인한다")
    out.append("")
    out.append(f"{'레이어':22s} {'도메인':12s} {'호출':>4s} {'MAC':>14s} {'비중':>6s} "
               f"{'hook 시간(us)':>14s} {'상태(B)':>9s}")
    for l in trace.layers:
        dom = "spike-driven" if l.spike_driven else l.domain
        st = "—" if l.self_time_us is None else f"{l.self_time_us:,.1f}"
        out.append(f"{l.layer_id:22s} {dom:12s} {l.calls:4d} {l.macs:14,} "
                   f"{l.macs/e:6.1%} {st:>14s} {l.state_bytes:9,}")

    m = trace.metrics
    if m:
        out.append("")
        out.append(f"실행 MAC            {m['executed_macs']:,}")
        ol = m.get("output_latency")
        if ol:
            out.append(f"출력 발생 지연      {ol['first_spike_step']} / T={ol['timesteps']}"
                       f"  (출력 레이어 {ol['layer_id']})")
        if m.get("state_bytes"):
            out.append(f"상태 메모리         {m['state_bytes']:,} bytes")
        if m.get("synops") is not None:
            out.append(f"SynOps              {m['synops']:,}")

    out.append("")
    out.append("출처")
    for s in trace.sources:
        out.append(f"  {s.collector:16s} {s.status:12s} 귀속={s.attribution:15s} "
                   f"{s.reason}")
        rate = (s.detail or {}).get("attribution_rate")
        if rate is not None:
            d = s.detail
            out.append(f"      귀속률 {rate:.1%}  "
                       f"귀속 {d['attributed_us']:,.0f}us / "
                       f"미귀속 {d['unattributed_us']:,.0f}us"
                       f"  구간 {d['matched_regions']}/{d['keyed_regions']}")
            for o in (d.get("unattributed_top") or [])[:3]:
                out.append(f"        미귀속 {o['op']:32s} {o['self_time_us']:>10,.1f}us")
            if d.get("instrument_us"):
                out.append(f"      계측 자신의 비용 {d['instrument_us']:,.0f}us "
                           f"(귀속 모수에서 제외)")

    for sp in trace.spikes:
        m = sp.metrics
        if sp.synops is not None:
            syn = f"SynOps {sp.synops:,} (fan-out {sp.fanout} <- {sp.fanout_from})"
        elif sp.fanout_from == "output":
            syn = "SynOps — (출력층, fan-out 없음)"
        else:
            syn = "SynOps — (소비자 미식별)"
        out.append(f"  spike {sp.layer_id:14s} 발화율 {m['mean_rate']:.4f}  "
                   f"dead {m['dead_fraction']:.4f}  sat {m['saturated_fraction']:.4f}  {syn}")
        first = m.get("first_spike_step_mean")
        if first is not None:
            out.append(f"      첫 스파이크 지연 {first:.2f} 스텝 "
                       f"(미발화 {m['never_fired_sample_neuron_fraction']:.1%})")
        top = m.get("top_neurons") or []
        if top:
            s = "  ".join(f"#{d['neuron']}:{d['spikes']}" for d in top[:5])
            out.append(f"      상위 뉴런  {s}")
    return "\n".join(out)
