"""SynOps = spike count x fan-out.

Fan-out is read from the graph, not hard-coded. The old SNN-Specialized code used
constants such as `spk1.sum() * 64` and `spk2.sum() * 10`, which silently gave wrong
values whenever a layer width changed.

Rule: the consumer of a spiking layer with N neurons is the layer that comes after it
in execution order, takes binary input, and has in_features == N. That layer's
out_features is the fan-out. If no consumer is found, leave it None; never guess.
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
