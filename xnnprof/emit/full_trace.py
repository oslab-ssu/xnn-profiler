"""Op hierarchy of the whole run: overlays PyTorch Profiler's Chrome Trace export.

Per-region `op_tree` holds only ops attributed to registered regions. Layer tracks are
therefore filled only while a region is open, and whatever ran in between
(instrumentation cost, functional calls outside labels, ops outside the loop) shows up
nowhere. Seeing what ran when, without gaps, needs that share too.

The profiler export contains the hook labels as `user_annotation` slices. The same
names exist in our region list, so the clock origin offset is taken from the label
pairs and added once to place every event on our timeline. The residual problem of
design doc section 2.2 does not arise here: the offset is taken from all labels rather
than a single anchor, and the remaining residual is recorded in sources.

Overlaid events go on a separate pid. Layer tracks are a view keyed by the unified key;
this is a view in execution order. Mixing both on one track makes neither readable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..collect.torch_prof import LABEL
from ..schema import Trace

FULL_PID = 2
PROCESS_NAME = "Full run (PyTorch Profiler)"


def _label_events(events: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if LABEL.match(name) and name not in out:
            out[name] = e
    return out


def align(trace: Trace, events: list[dict]) -> tuple[Optional[float], dict[str, Any]]:
    """Clock origin offset, taken from all labels; the residual is returned with it."""
    labels = _label_events(events)
    diffs = []
    for r in trace.regions:
        if r.timestep is None:
            continue
        ev = labels.get(f"{r.layer_id}@t{r.timestep}")
        if ev is None:
            continue
        diffs.append(r.ts_us - float(ev["ts"]))
    if not diffs:
        return None, {"reason": "no label slices found in the export"}
    diffs.sort()
    offset = diffs[len(diffs) // 2]
    resid = [d - offset for d in diffs]
    return offset, {
        "matched_labels": len(diffs),
        "offset_us": round(offset, 3),
        "residual_max_us": round(max(abs(x) for x in resid), 3),
    }


def extract(trace_path: str) -> list[dict]:
    """Pull the full-run track out of an already written trace file.

    These events exist only in `traceEvents`, not in the `xnn` extension key. Paths
    that reread and rewrite the file, such as merging, would drop them, so they are
    taken out and put back.
    """
    try:
        with open(trace_path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:                                        # noqa: BLE001
        return []
    return [e for e in raw.get("traceEvents", []) if e.get("pid") == FULL_PID]


def perfectly_nest(slices: list[dict]) -> tuple[list[dict], int]:
    """Clip slice ends so children never extend past their parent.

    With `with_stack`, Python trace events and op events are timestamped by different
    paths and boundaries drift by a few µs. Viewers reject slices that do not nest
    exactly, so containment is enforced. The clip count is recorded in sources.
    """
    groups: dict[tuple, list[dict]] = {}
    for s in slices:
        groups.setdefault((s["pid"], s["tid"]), []).append(s)
    out, clamped = [], 0
    for items in groups.values():
        items.sort(key=lambda s: (s["ts"], -s["dur"]))
        stack: list[float] = []
        for s in items:
            start, end = s["ts"], s["ts"] + s["dur"]
            while stack and stack[-1] <= start + 1e-9:
                stack.pop()
            if stack and end > stack[-1]:
                end = stack[-1]
                clamped += 1
            if end - start <= 0:
                continue
            s["dur"] = round(end - start, 3)
            stack.append(end)
            out.append(s)
    return out, clamped


def events_for(trace: Trace, export_path: str) -> tuple[list[dict], dict[str, Any]]:
    """Read the export file and build the event list to overlay on our timeline."""
    if not export_path or not os.path.exists(export_path):
        return [], {"reason": "export file missing"}
    try:
        with open(export_path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:                                   # noqa: BLE001
        return [], {"reason": f"{type(e).__name__}: {e}"}

    src = raw.get("traceEvents", raw) if isinstance(raw, dict) else raw
    offset, detail = align(trace, src)
    if offset is None:
        return [], detail

    tids: dict[Any, int] = {}
    out: list[dict] = []
    for e in src:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        tid = tids.setdefault(e.get("tid"), len(tids) + 1)
        out.append({
            "ph": "X", "name": e.get("name"),
            "cat": e.get("cat", "op"),
            "ts": round(float(e["ts"]) + offset, 3),
            "dur": round(float(e["dur"]), 3),
            "pid": FULL_PID, "tid": tid,
        })
    out, clamped = perfectly_nest(out)
    detail["slices"] = len(out)
    detail["threads"] = len(tids)
    if clamped:
        detail["clamped_slices"] = clamped

    meta = [{"ph": "M", "name": "process_name", "pid": FULL_PID, "tid": 0,
             "args": {"name": PROCESS_NAME}}]
    for raw_tid, tid in tids.items():
        meta.append({"ph": "M", "name": "thread_name", "pid": FULL_PID, "tid": tid,
                     "args": {"name": f"thread {raw_tid}"}})
    return meta + out, detail
