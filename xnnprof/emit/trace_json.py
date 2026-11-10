"""통일 포맷 = Chrome Trace Event JSON.

Perfetto · Speedscope · Tracy · 기존 VS Code extension이 모두 이 포맷을 읽는다.
별도 포맷을 두지 않고, 우리 추가 정보는 이 포맷의 확장 지점에 싣는다.

  tid        레이어 1개 = 트랙 1개. 시간축 루프가 그 트랙의 반복 슬라이스로 보인다
  cat        dense | spike | state — 뷰어에서 도메인 필터
  args       통합 키(timestep) · 레이어별 MAC · 스파이크 구동 여부 ·
             그 구간에 귀속된 PyTorch Profiler op별 self time ·
             스파이크 레이어면 신규 측정 항목(args.snn)
  ph="i"     모델 단위 신규 측정 항목을 담은 요약 마커 1개
  ph="C"     타임스텝별 스파이크 발화율 카운터 (Perfetto에서 그래프로 뜬다)
  "xnn"      최상위 확장 키. 뷰어는 traceEvents만 읽으므로 무시하고 지나간다
"""

from __future__ import annotations

import json
from typing import Any

from ..collect.torch_prof import SYNC_OPS
from ..schema import Trace


def _tracks(trace: Trace) -> dict[str, int]:
    order = [l.layer_id for l in sorted(trace.layers, key=lambda l: -l.macs)]
    for r in trace.regions:
        if r.layer_id not in order:
            order.append(r.layer_id)
    return {lid: i + 1 for i, lid in enumerate(order)}


def _snn_args(sp, timestep) -> dict[str, Any]:
    """스파이크 레이어 구간에 붙는 신규 측정 항목."""
    m = sp.metrics or {}
    series = m.get("per_timestep_rate") or []
    out: dict[str, Any] = {"neurons": sp.neurons}
    if timestep is not None and timestep < len(series):
        out["spike_rate_this_step"] = round(float(series[timestep]), 6)
    for key, src in (("spike_rate_mean", "mean_rate"),
                     ("dead_fraction", "dead_fraction"),
                     ("saturated_fraction", "saturated_fraction"),
                     ("first_spike_step_mean", "first_spike_step_mean"),
                     ("never_fired_fraction", "never_fired_sample_neuron_fraction")):
        if m.get(src) is not None:
            out[key] = round(float(m[src]), 6)
    if sp.synops is not None:
        out["synops"] = sp.synops
        out["fanout"] = sp.fanout
        out["fanout_from"] = sp.fanout_from
    elif sp.fanout_from:
        out["synops"] = f"정의되지 않음 ({sp.fanout_from})"
    ss = sp.steady_state or {}
    if ss:
        out["steady_state_step"] = ss.get("reached_step")
        if ss.get("reached_step") is None and ss.get("reason"):
            out["steady_state"] = f"미도달 ({ss['reason']})"
        c = ss.get("c_t") or []
        # c_t[i] 는 스텝 i+1 의 값이다. 스텝 0 에는 직전 스텝이 없다.
        if timestep is not None and 1 <= timestep <= len(c):
            out["pattern_change_c_t"] = c[timestep - 1]
    return out


def _mem_args(mb, timestep) -> dict[str, Any]:
    """막전위 분포. 구간에는 그 스텝의 값만 싣고 전체 배열은 파일로 남긴다."""
    m = mb.metrics or {}
    out: dict[str, Any] = {"threshold": mb.threshold, "bins": mb.bins,
                           "path": mb.path}
    for key, src in (("near_threshold_fraction", "near_threshold_fraction"),
                     ("at_or_above_threshold_fraction",
                      "at_or_above_threshold_fraction"),
                     ("below_zero_fraction", "below_zero_fraction")):
        series = m.get(src) or []
        if timestep is not None and timestep < len(series):
            out[key] = series[timestep]
    return out


def _summary(trace: Trace) -> dict[str, Any]:
    """모델 단위 측정값. 구간 하나에 귀속되지 않아 별도 마커로 내보낸다."""
    env = trace.env or {}
    out: dict[str, Any] = dict(trace.metrics or {})
    for k in ("device", "region_timing", "weights_sha256", "batch_accuracy", "gpu"):
        if env.get(k) is not None:
            out[k] = env[k]
    src = next((s for s in trace.sources if s.collector == "torch_profiler"), None)
    if src is not None and (src.detail or {}).get("attribution_rate") is not None:
        out["attribution_rate"] = src.detail["attribution_rate"]
        out["instrument_us"] = src.detail.get("instrument_us")
    return out


def build(trace: Trace, pid: int = 1) -> dict[str, Any]:
    tid = _tracks(trace)
    spikes = {sp.layer_id: sp for sp in trace.spikes}
    membranes = {mb.layer_id: mb for mb in trace.membranes}
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
        sp = spikes.get(r.layer_id)
        if sp is not None:
            args["snn"] = _snn_args(sp, r.timestep)
        mb = membranes.get(r.layer_id)
        if mb is not None:
            args["membrane"] = _mem_args(mb, r.timestep)
        if r.gpu:
            args["gpu"] = r.gpu
        if r.ops:
            args["ops_self_time_us"] = r.ops
            sync = sum(v for k, v in r.ops.items() if k in SYNC_OPS)
            if sync:
                args["host_sync_us"] = round(sync, 3)
                args["host_sync_fraction"] = round(sync / r.dur_us, 4) if r.dur_us else None
        name = (f"{r.layer_id}@t{r.timestep}" if r.timestep is not None else r.layer_id)
        t = tid.get(r.layer_id, 0)
        ev.append({"ph": "X", "name": name, "cat": r.domain,
                   "ts": round(r.ts_us, 3), "dur": round(r.dur_us, 3),
                   "pid": pid, "tid": t, "args": args})
        # 구간 안의 연산자 호출 계층. 같은 트랙에 중첩 슬라이스로 쌓으면
        # Flame Graph의 뿌리가 통합 키가 된다.
        for c in r.op_tree:
            ev.append({"ph": "X", "name": c["name"], "cat": "op",
                       "ts": round(r.ts_us + c["off_us"], 3),
                       "dur": round(c["dur_us"], 3),
                       "pid": pid, "tid": t,
                       "args": {"depth": c["depth"], "region": name,
                                **({"clamped": True} if c.get("clamped") else {})}})

    # 모델 단위 측정값은 구간에 귀속되지 않으므로 마커 하나로 내보낸다.
    summary = _summary(trace)
    if summary:
        t0 = min((r.ts_us for r in trace.regions), default=0.0)
        ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": 0,
                   "args": {"name": "xnn 요약"}})
        ev.append({"ph": "i", "name": "xnn 신규 측정 항목", "cat": "xnn", "s": "p",
                   "ts": round(t0, 3), "pid": pid, "tid": 0, "args": summary})

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

    # 같은 시각이면 긴 슬라이스가 먼저 와야 중첩 관계가 유지된다.
    ev.sort(key=lambda e: (e.get("ts", -1), 0 if e["ph"] == "M" else 1,
                           -e.get("dur", 0)))
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
