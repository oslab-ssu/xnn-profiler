"""PyTorch Profiler collector: label-based correlation.

The profiler keeps user labels opened with `record_function` in its event
tree. The hooks open the unified key string `fc1@t3` as such a label, so
walking an operator event's parents up to the nearest label determines which
region that operator belongs to. No clock origin has to be aligned and no
timestamp containment has to be tested.

Operators that force a host synchronization are tallied per region as well.
A call like `.item()`, which pulls a GPU value into Python, stalls the CPU
until the GPU queue has drained, and therefore dominates that region's time
regardless of how much arithmetic it does. Which layer and which timestep it
happened in only becomes visible once it is grouped by the unified key.

`key_averages()` is not used: it has already summed the T calls together and
cannot be taken back apart by layer and timestep. Execution without a label
(`emit_labels=False`, or a functional call the hooks cannot reach) is kept
separate as the unattributed share and recorded in the provenance.
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
PYFRAME = re.compile(r"^.+\(\d+\): .+$")      # torch/nn/modules/module.py(1736): _call_impl
MAX_DEPTH = 64
INSTRUMENT = ("__instrument__", -1)

# Operators that force a host sync: calls that pull a GPU value into Python.
SYNC_OPS = frozenset({
    "aten::item", "aten::_local_scalar_dense", "aten::is_nonzero",
    "aten::nonzero", "aten::equal", "aten::allclose", "aten::_assert_async",
    "cudaStreamSynchronize", "cudaDeviceSynchronize",
})


def is_pseudo(key: str) -> bool:
    return key.startswith("[") or key.startswith(PSEUDO)


def is_python_frame(key: str) -> bool:
    """A Python tracing event, produced when `with_stack` is enabled.

    Excluded from both the attribution tally and `op_tree`. Including them
    would make the attribution rate of the same measurement depend on a single
    flag, which breaks comparison across runs. To see Python frames, the
    execution-wide track (`emit/full_trace.py`) carries the profiler export
    verbatim.
    """
    return key.startswith("<") or PYFRAME.match(key) is not None


def locate(ev):
    """The unified key an event belongs to, its depth below that label, and
    the label event itself.

    Walks up the parents to the nearest label. If the instrumentation's own
    label is hit first, the event is not part of the measurement and
    `INSTRUMENT` is returned.

    Returns `(key, depth, label_event)`. Depth counts levels below the label,
    with a direct child at 1; it is used to draw the nested slices.
    """
    cur, depth = ev, 0
    while cur is not None and depth < MAX_DEPTH:
        nm = str(cur.name)
        if nm == INSTRUMENT_LABEL:
            return INSTRUMENT, depth, cur
        m = LABEL.match(nm)
        if m:
            return (m.group("layer"), int(m.group("step"))), depth, cur
        cur = getattr(cur, "cpu_parent", None)
        depth += 1
    return None, depth, None


def owner_of(ev) -> Optional[tuple[str, int]]:
    return locate(ev)[0]


def _span(ev):
    """An event's `(start, duration)` in us.

    The attribute name differs between profiler versions.
    """
    tr = getattr(ev, "time_range", None) or getattr(ev, "cpu_interval", None)
    if tr is None:
        return None
    try:
        return float(tr.start), float(tr.end) - float(tr.start)
    except Exception:
        return None


def nest(calls: list[dict], dur_us: float, cap: int) -> list[dict]:
    """Clip slices into the region and drop the deepest levels to meet the cap.

    Offsets are relative to the label's start, so the two clocks never have to
    share an origin. Region length is measured with `perf_counter` while
    offsets come from the profiler's clock, so a slice can overhang the end by
    a few us. Viewers reject slices that leave their parent, so they are cut at
    the region boundary and the cut is recorded as `clamped`.
    """
    out = []
    for c in sorted(calls, key=lambda c: (c["depth"], c["off_us"])):
        off = max(0.0, min(c["off_us"], dur_us))
        end = min(c["off_us"] + c["dur_us"], dur_us)
        if end - off <= 0:
            continue
        item = {"name": c["name"], "depth": c["depth"],
                "off_us": round(off, 3), "dur_us": round(end - off, 3)}
        if end < c["off_us"] + c["dur_us"] - 1e-9:
            item["clamped"] = True
        out.append(item)
    if len(out) <= cap:
        return out
    keep, total = [], 0                  # cut whole depth levels -- dropping
    for d in sorted({c["depth"] for c in out}):      # a middle one would break
        level = [c for c in out if c["depth"] == d]  # the nesting
        if keep and total + len(level) > cap:
            break
        keep += level                    # keep the first level even past the
        total += len(level)              # cap, or the region would come out empty
    return keep


class TorchProfilerCollector(Collector):
    name = "torch_profiler"
    attribution = REGION
    capabilities = ("op_time", "op_memory", "call_count", "label_join")

    def __init__(self, prof=None, op_tree: bool = True,
                 op_tree_cap: int = 400):
        self.prof = prof
        self.op_tree = op_tree
        self.op_tree_cap = op_tree_cap

    def available(self) -> tuple[bool, str]:
        try:
            import torch.profiler  # noqa: F401
        except Exception as e:
            return False, f"torch.profiler unavailable: {e}"
        return True, "ok"

    def collect(self, trace: Trace, **kw: Any):
        if self.prof is None:
            return self.declare("unavailable", "no profiler session was passed in")
        try:
            events = list(self.prof.events())
        except Exception as e:
            return self.declare("failed", f"{type(e).__name__}: {e}")

        per_key: dict[tuple[str, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        tree: dict[tuple[str, int], list[dict]] = defaultdict(list)
        orphan: dict[str, float] = defaultdict(float)
        orphan_count = 0
        instrument: dict[str, float] = defaultdict(float)

        for ev in events:
            op = str(ev.name)
            if (is_pseudo(op) or LABEL.match(op) or op == INSTRUMENT_LABEL
                    or is_python_frame(op)):
                continue                             # a label is not an operator
            key, depth, label_ev = locate(ev)
            if self.op_tree and key is not None and key != INSTRUMENT:
                me, base = _span(ev), _span(label_ev)
                if me and base:
                    tree[key].append({"name": op, "depth": depth,
                                      "off_us": me[0] - base[0],
                                      "dur_us": me[1]})
            self_us = float(getattr(ev, "self_cpu_time_total", 0.0) or 0.0)
            if self_us <= 0:
                continue
            if key == INSTRUMENT:
                instrument[op] += self_us
            elif key is None:
                orphan[op] += self_us
                orphan_count += 1
            else:
                per_key[key][op] += self_us

        attributed = sum(sum(d.values()) for d in per_key.values())
        unattributed = sum(orphan.values())
        instrument_us = sum(instrument.values())
        total = attributed + unattributed          # instrumentation cost is excluded

        matched, nested = 0, 0
        for (lid, step), ops in per_key.items():
            r = trace.region(lid, step)
            if r is None:
                continue
            r.ops = {k: round(v, 3) for k, v in
                     sorted(ops.items(), key=lambda kv: -kv[1])}
            matched += 1
        for (lid, step), calls in tree.items():
            r = trace.region(lid, step)
            if r is None:
                continue
            r.op_tree = nest(calls, r.dur_us, self.op_tree_cap)
            nested += len(r.op_tree)

        top_orphan = sorted(orphan.items(), key=lambda kv: -kv[1])[:10]
        top_inst = sorted(instrument.items(), key=lambda kv: -kv[1])[:10]
        status = "captured" if matched and not orphan else "partial"
        reason = "" if status == "captured" else (
            "operations ran outside any label -- this covers functional calls "
            "that do not appear in the module tree (F.linear and the like) "
            "and the profiler's own cost")
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
            sync_us=round(sum(v for ops in per_key.values()
                               for k, v in ops.items() if k in SYNC_OPS), 3),
            instrument_us=round(instrument_us, 3),
            instrument_top=[{"op": k, "self_time_us": round(v, 3)}
                            for k, v in top_inst],
            op_tree_calls=nested if self.op_tree else 0,
            op_tree="offsets relative to the label start" if self.op_tree else "off")
