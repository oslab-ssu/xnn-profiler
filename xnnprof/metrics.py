"""Metrics that can only be derived from the trace as a whole.

Per-layer values come from the collectors and spike-array values come from
`observations`; this module derives only what needs both.

  output latency        first timestep at which the output layer's spike rate
                        is non-zero
  state memory          total size of the state tensors carried across timesteps
  steady-state step     gathers the per-layer results (formula in `steady.py`)
  membrane distribution gathers the per-layer histogram summaries
"""

from __future__ import annotations

from .schema import Trace


def execution_order(trace: Trace) -> list[str]:
    """Order in which layers first appear, sorted by region start time."""
    seen, out = set(), []
    for r in sorted(trace.regions, key=lambda r: r.ts_us):
        if r.layer_id not in seen:
            seen.add(r.layer_id)
            out.append(r.layer_id)
    return out


def output_layer(trace: Trace) -> str:
    """Treat the last spiking layer in execution order as the output layer."""
    order = execution_order(trace)
    spiking = {sp.layer_id for sp in trace.spikes}
    for lid in reversed(order):
        if lid in spiking:
            return lid
    return ""


def output_latency(trace: Trace):
    """Undefined for a model with no spiking layer (a pure DNN)."""
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
        "unit": "timestep index (0-based)",
    }


def steady_state(trace: Trace) -> dict:
    """Per-layer steady-state convergence step. Formula in `steady.py`."""
    out = {}
    for sp in trace.spikes:
        ss = sp.steady_state or {}
        if not ss:
            continue
        row = {k: ss[k] for k in ("reached_step", "c_t_min", "c_t_max",
                                  "mean_rate", "threshold", "reason")
               if k in ss}
        out[sp.layer_id] = row
    return out


def membrane_distribution(trace: Trace) -> dict:
    """Per-layer membrane distribution summary. Full arrays live in the
    observation output files.
    """
    out = {}
    for mb in trace.membranes:
        m = mb.metrics or {}
        out[mb.layer_id] = {
            "threshold": mb.threshold,
            "bins": mb.bins,
            "near_bins": m.get("near_bins"),
            "near_threshold_fraction_mean": m.get("near_threshold_fraction_mean"),
            "path": mb.path,
            "sha256": mb.sha256,
        }
    return out


def snn_metrics(trace: Trace) -> dict:
    """Omit the key for anything not derived. '0' and 'undefined' differ."""
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
    ss = steady_state(trace)
    if ss:
        out["steady_state"] = ss
    mem = membrane_distribution(trace)
    if mem:
        out["membrane_distribution"] = mem
    return out
