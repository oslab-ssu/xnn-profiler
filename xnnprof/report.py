"""Stdout summary."""

from __future__ import annotations

from .schema import Trace


def summary(trace: Trace) -> str:
    out = []
    e = trace.executed_macs() or 1
    dev = (trace.env or {}).get("device", "cpu")
    rt = (trace.env or {}).get("region_timing", "")
    out.append(f"model={trace.model_id}  T={trace.timesteps}  batch={trace.batch}  "
               f"device={dev}" + (f"  region_timing={rt}" if rt else ""))
    if rt == "cpu-launch-window" and dev.startswith("cuda"):
        out.append("  note: region times are CPU-side launch windows. "
                   "Use --sync-regions or Nsight Systems for actual GPU time")
    out.append("")
    out.append(f"{'layer':22s} {'domain':12s} {'calls':>5s} {'MAC':>14s} {'share':>6s} "
               f"{'hook time(us)':>14s} {'state(B)':>9s}")
    for l in trace.layers:
        dom = "spike-driven" if l.spike_driven else l.domain
        st = "—" if l.self_time_us is None else f"{l.self_time_us:,.1f}"
        out.append(f"{l.layer_id:22s} {dom:12s} {l.calls:5d} {l.macs:14,} "
                   f"{l.macs/e:6.1%} {st:>14s} {l.state_bytes:9,}")

    from .collect.torch_prof import SYNC_OPS

    sync: dict[str, list[float]] = {}
    for r in trace.regions:
        hit = sum(v for k, v in (r.ops or {}).items() if k in SYNC_OPS)
        if not hit:
            continue
        a = sync.setdefault(r.layer_id, [0.0, 0.0, 0.0])
        a[0] += hit
        a[1] += r.dur_us
        a[2] += 1
    if sync:
        out.append("")
        out.append("host sync: time the CPU waited on the GPU to pull values into Python")
        out.append(f"{'layer':22s} {'regions':>7s} {'sync(us)':>11s} "
                   f"{'region(us)':>11s} {'share':>12s} {'causing op':>26s}")
        for lid, (hit, dur, n) in sorted(sync.items(), key=lambda kv: -kv[1][0]):
            tops: dict[str, float] = {}
            for r in trace.regions:
                if r.layer_id != lid:
                    continue
                for k, v in (r.ops or {}).items():
                    if k in SYNC_OPS:
                        tops[k] = tops.get(k, 0.0) + v
            top = max(tops, key=tops.get) if tops else "-"
            out.append(f"{lid:22s} {int(n):7d} {hit:11,.1f} {dur:11,.1f} "
                       f"{hit / dur if dur else 0:11.1%} {top:>26s}")

    m = trace.metrics
    if m:
        out.append("")
        out.append(f"executed MACs       {m['executed_macs']:,}")
        ol = m.get("output_latency")
        if ol:
            out.append(f"output latency      {ol['first_spike_step']} / T={ol['timesteps']}"
                       f"  (output layer {ol['layer_id']})")
        if m.get("state_bytes"):
            out.append(f"state memory        {m['state_bytes']:,} bytes")
        if m.get("synops") is not None:
            out.append(f"SynOps              {m['synops']:,}")
        for lid, ss in (m.get("steady_state") or {}).items():
            step = ss.get("reached_step")
            rng = (f"c_t {ss['c_t_min']:.3f}~{ss['c_t_max']:.3f}"
                   if "c_t_min" in ss else ss.get("reason", ""))
            got = f"{step} / T={trace.timesteps}" if step is not None else "not reached"
            out.append(f"steady-state step   {got:14s} {lid:10s} {rng}")
        for lid, mb in (m.get("membrane_distribution") or {}).items():
            near = mb.get("near_threshold_fraction_mean")
            if near is None:
                continue
            lo, hi = (mb.get("near_bins") or [0, 0])
            out.append(f"membrane near θ     {near:.2%} (mean)   {lid:10s} "
                       f"θ={mb['threshold']} bins {lo}~{hi}/{mb['bins']}")

    out.append("")
    out.append("sources")
    for s in trace.sources:
        out.append(f"  {s.collector:16s} {s.status:12s} attribution={s.attribution:15s} "
                   f"{s.reason}")
        rate = (s.detail or {}).get("attribution_rate")
        if rate is not None:
            d = s.detail
            out.append(f"      attribution rate {rate:.1%}  "
                       f"attributed {d['attributed_us']:,.0f}us / "
                       f"unattributed {d['unattributed_us']:,.0f}us"
                       f"  regions {d['matched_regions']}/{d['keyed_regions']}")
            for o in (d.get("unattributed_top") or [])[:3]:
                out.append(f"        unattributed {o['op']:32s} {o['self_time_us']:>10,.1f}us")
            if d.get("instrument_us"):
                out.append(f"      instrumentation cost {d['instrument_us']:,.0f}us "
                           f"(excluded from the attribution base)")
        if s.collector == "perf" and s.status in ("captured", "partial"):
            d = s.detail
            per = d.get("per_forward") or {}
            out.append(f"      per forward (x{d.get('reps')}): "
                       f"cycles {per.get('cycles') or 0:,.0f}  "
                       f"instructions {per.get('instructions') or 0:,.0f}  "
                       f"IPC {d.get('ipc')}  cache-miss {d.get('cache_miss_rate')}  "
                       f"branch-miss {d.get('branch_miss_rate')}")

    for sp in trace.spikes:
        m = sp.metrics
        if sp.synops is not None:
            syn = f"SynOps {sp.synops:,} (fan-out {sp.fanout} <- {sp.fanout_from})"
        elif sp.fanout_from == "output":
            syn = "SynOps — (output layer, no fan-out)"
        else:
            syn = "SynOps — (consumer not identified)"
        out.append(f"  spike {sp.layer_id:14s} rate {m['mean_rate']:.4f}  "
                   f"dead {m['dead_fraction']:.4f}  sat {m['saturated_fraction']:.4f}  {syn}")
        first = m.get("first_spike_step_mean")
        if first is not None:
            out.append(f"      first-spike latency {first:.2f} steps "
                       f"(never fired {m['never_fired_sample_neuron_fraction']:.1%})")
        top = m.get("top_neurons") or []
        if top:
            s = "  ".join(f"#{d['neuron']}:{d['spikes']}" for d in top[:5])
            out.append(f"      top neurons  {s}")
    return "\n".join(out)
