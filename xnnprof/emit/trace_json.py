"""통일 포맷 = Chrome Trace Event JSON.

Perfetto · Speedscope · Tracy · 기존 VS Code extension이 모두 이 포맷을 읽는다.
별도 포맷을 두지 않고, 우리 추가 정보는 이 포맷의 확장 지점에 싣는다.

  tid        레이어 1개 = 트랙 1개. 시간축 루프가 그 트랙의 반복 슬라이스로 보인다
  cat        dense | spike | state — 뷰어에서 도메인 필터
  args       통합 키(timestep) · 레이어별 MAC · 스파이크 구동 여부 ·
             그 구간에 귀속된 PyTorch Profiler op별 self time
  ph="C"     타임스텝별 스파이크 발화율 카운터 (Perfetto에서 그래프로 뜬다)
  "xnn"      최상위 확장 키. 뷰어는 traceEvents만 읽으므로 무시하고 지나간다
"""

from __future__ import annotations

import json
from typing import Any

from ..schema import Trace


def _tracks(trace: Trace) -> dict[str, int]:
    order = [l.layer_id for l in sorted(trace.layers, key=lambda l: -l.macs)]
    for r in trace.regions:
        if r.layer_id not in order:
            order.append(r.layer_id)
    return {lid: i + 1 for i, lid in enumerate(order)}


def build(trace: Trace, pid: int = 1) -> dict[str, Any]:
    tid = _tracks(trace)
    ev: list[dict] = [{
        "ph": "M", "name": "process_name", "pid": pid, "tid": 0,
        "args": {"name": f"xnn {trace.model_id} T={trace.timesteps} batch={trace.batch}"},
    }]

    for layer in trace.layers:
        t = tid[layer.layer_id]
        tag = "spike-driven" if layer.spike_driven else layer.domain
        ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": t,
                   "args": {"name": f"{layer.layer_id} [{tag}]"}})
        ev.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": t,
                   "args": {"sort_index": t}})

    for r in trace.regions:
        layer = trace.layer(r.layer_id)
        args: dict[str, Any] = {"timestep": r.timestep}
        if layer is not None:
            args.update({
                "op_type": layer.op_type,
                "macs_total": layer.macs,
                "spike_driven": layer.spike_driven,
                "in_loop": layer.in_loop,
                "state_bytes": layer.state_bytes,
            })
        if r.ops:
            args["ops_self_time_us"] = r.ops
        name = (f"{r.layer_id}@t{r.timestep}" if r.timestep is not None else r.layer_id)
        ev.append({"ph": "X", "name": name, "cat": r.domain,
                   "ts": round(r.ts_us, 3), "dur": round(r.dur_us, 3),
                   "pid": pid, "tid": tid.get(r.layer_id, 0), "args": args})

    # 타임스텝별 발화율을 카운터로 — 시간축에 걸쳐 그래프가 된다.
    for sp in trace.spikes:
        series = (sp.metrics or {}).get("per_timestep_rate") or []
        base = [r for r in trace.regions if r.layer_id == sp.layer_id]
        base.sort(key=lambda r: (r.timestep if r.timestep is not None else 0))
        for i, rate in enumerate(series):
            ts = base[i].ts_us if i < len(base) else float(i)
            ev.append({"ph": "C", "name": f"spike_rate {sp.layer_id}", "cat": "spike",
                       "ts": round(ts, 3), "pid": pid,
                       "args": {"rate": round(float(rate), 6)}})

    ev.sort(key=lambda e: (e.get("ts", -1), 0 if e["ph"] == "M" else 1))
    return {
        "traceEvents": ev,
        "displayTimeUnit": "ms",
        "otherData": {"schema": "xnn.trace", "model": trace.model_id},
        "xnn": trace.to_dict(),
    }


def write(trace: Trace, path: str, pid: int = 1) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build(trace, pid), f, ensure_ascii=False, indent=2)
    return path
