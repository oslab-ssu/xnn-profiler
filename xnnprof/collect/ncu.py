"""Nsight Compute collector: per-kernel hardware counters grouped into regions.

Hooks push the unified key `fc1@t3` as an NVTX range. ncu records the NVTX stack of
every profiled kernel, so a kernel belongs to each `layer@tN` range on its stack, the
same launch-in-range rule as Nsight Systems. The NVTX filter limits profiling to those
ranges, which leaves out warm-up, perf repetitions and `xnn::instrument`.

ncu serializes kernels and replays each one several times with caches flushed and
clocks locked to base. Counter values are per-kernel facts; durations are not the
durations of a normal run and must not be compared with Nsight Systems timings.

The report (`.ncu-rep`) is read with `ncu_report`, which ships with Nsight Compute.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections import defaultdict
from typing import Any, Iterable, Optional

from ..schema import Trace
from .base import Collector, REGION
from .torch_prof import LABEL

PROJECTION = "launch-in-range"
NVTX_FILTER = "regex:.*@t[0-9]+/"
CACHE_CONTROL = "all"
CLOCK_CONTROL = "base"

DURATION = "gpu__time_duration.sum"
METRICS = (
    DURATION,
    "sm__cycles_elapsed.avg",
    "gpc__cycles_elapsed.avg.per_second",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
)
SUMMED = {
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "sm__inst_executed.sum": "inst_executed",
}
# Duration-weighted, since a share of peak only means something over the time it covers.
WEIGHTED = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy_pct",
}
LAUNCH = ("launch__grid_size", "launch__block_size", "launch__registers_per_thread")


def ncu_bin() -> str:
    """The `ncu` executable. If not on PATH, set the `NCU` environment variable."""
    env = os.environ.get("NCU", "").strip('"')
    if env and os.path.exists(env):
        return env
    return shutil.which("ncu") or ""


def profile_cmd(ncu: str, report: str) -> list[str]:
    return [ncu, "--nvtx", "--nvtx-include", NVTX_FILTER,
            "--metrics", ",".join(METRICS),
            "--cache-control", CACHE_CONTROL, "--clock-control", CLOCK_CONTROL,
            "--target-processes", "all", "-f", "-o", report]


def _module():
    """`ncu_report` lives in `extras/python` next to the ncu binary, not on sys.path."""
    try:
        import ncu_report                                   # type: ignore
        return ncu_report, ""
    except ImportError:
        pass
    ncu = ncu_bin()
    if not ncu:
        return None, ("ncu_report not importable and ncu not found; "
                      "put ncu on PATH or set the NCU environment variable")
    path = os.path.join(os.path.dirname(os.path.realpath(ncu)), "extras", "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        import ncu_report                                   # type: ignore
        return ncu_report, ""
    except ImportError as e:
        return None, f"ncu_report not importable from {path}: {e}"


def read_report(path: str) -> tuple[list[dict[str, Any]], str]:
    """One row per profiled kernel: name, NVTX push/pop stack, metric values."""
    mod, why = _module()
    if mod is None:
        return [], why
    try:
        ctx = mod.load_report(path)
    except Exception as e:                                   # noqa: BLE001
        return [], f"cannot load {path}: {type(e).__name__}: {e}"
    rows = []
    for ri in range(ctx.num_ranges()):
        rg = ctx.range_by_idx(ri)
        for ai in range(rg.num_actions()):
            a = rg.action_by_idx(ai)
            ranges: list[str] = []
            st = a.nvtx_state()
            if st is not None:
                for did in st.domains():
                    ranges.extend(str(x) for x in st.domain_by_id(did).push_pop_ranges())
            m = {}
            for name in METRICS:
                v = a.metric_by_name(name)
                m[name] = None if v is None else v.value()
            rows.append({"kernel": a.name(), "ranges": ranges, "metrics": m})
    return rows, ""


def _num(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None


def aggregate(trace: Trace, rows: Iterable[dict[str, Any]], top: int = 3) -> dict[str, Any]:
    """Attach per-region counters to `Region.ncu`. Returns totals for the source."""
    per: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "n": 0, "ns": 0.0, "sum": defaultdict(float), "w": defaultdict(float),
        "wt": defaultdict(float), "flops": 0.0, "k": defaultdict(lambda: [0.0, None])})
    total_n, total_ns, outside, missing = 0, 0.0, 0, set()
    clocks = []
    for row in rows:
        m = row["metrics"]
        total_n += 1
        ns = _num(m.get(DURATION)) or 0.0
        total_ns += ns
        if _num(m.get("gpc__cycles_elapsed.avg.per_second")):
            clocks.append(m["gpc__cycles_elapsed.avg.per_second"])
        missing.update(k for k, v in m.items() if v is None)
        keys = [(g.group("layer"), int(g.group("step")))
                for g in (LABEL.match(r) for r in row["ranges"]) if g]
        if not keys:
            outside += 1
        for key in keys:
            d = per[key]
            d["n"] += 1
            d["ns"] += ns
            for name, out in SUMMED.items():
                v = _num(m.get(name))
                if v is not None:
                    d["sum"][out] += v
            for name, out in WEIGHTED.items():
                v = _num(m.get(name))
                if v is not None:
                    d["w"][out] += v * ns
                    d["wt"][out] += ns
            f = [_num(m.get(f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum"))
                 for op in ("ffma", "fadd", "fmul")]
            if all(v is not None for v in f):
                d["flops"] += 2 * f[0] + f[1] + f[2]
            k = d["k"][row["kernel"]]
            k[0] += ns
            if k[1] is None:
                k[1] = {n.split("__")[1]: m.get(n) for n in LAUNCH}

    matched, in_ns, unmatched = 0, 0.0, []
    for (lid, step), d in per.items():
        r = trace.region(lid, step)
        if r is None:
            unmatched.append(f"{lid}@t{step}")
            continue
        nbytes = d["sum"].get("dram_read_bytes", 0.0) + d["sum"].get("dram_write_bytes", 0.0)
        out: dict[str, Any] = {
            "kernels": d["n"],
            "kernel_us": round(d["ns"] / 1000.0, 3),
            "projection": PROJECTION,
            **{k: int(v) for k, v in d["sum"].items()},
            "flops_fp32": int(d["flops"]),
            "flops_per_dram_byte": round(d["flops"] / nbytes, 4) if nbytes else None,
            **{k: round(d["w"][k] / d["wt"][k], 3) for k in d["w"] if d["wt"][k]},
            "top_kernels": [{"name": n, "us": round(v[0] / 1000.0, 3), **(v[1] or {})}
                            for n, v in sorted(d["k"].items(), key=lambda kv: -kv[1][0])[:top]],
        }
        r.ncu = out
        matched += 1
        in_ns += d["ns"]
    return {
        "kernels_profiled": total_n, "kernels_outside_ranges": outside,
        "nvtx_ranges": len(per), "matched_regions": matched,
        "unmatched_ranges": sorted(unmatched)[:20],
        "kernel_us_in_ranges": round(in_ns / 1000.0, 3),
        "kernel_us_profiled": round(total_ns / 1000.0, 3),
        "clock_hz": round(sum(clocks) / len(clocks)) if clocks else None,
        "metrics_missing": sorted(missing),
    }


class NcuCollector(Collector):
    name = "nsight_compute"
    attribution = REGION
    capabilities = ("kernel_counters", "occupancy", "memory_bandwidth", "flops")

    def __init__(self, report: str = "", top: int = 3):
        self.report = report
        self.top = top

    def available(self) -> tuple[bool, str]:
        if not self.report:
            return False, ("no report given; the report is written after the process "
                           "exits, so use --ncu or tools/ncu_merge")
        if not os.path.exists(self.report):
            return False, f"{self.report} missing"
        mod, why = _module()
        return (True, "ok") if mod else (False, why)

    def collect(self, trace: Trace, **kw: Any):
        # Values from an earlier merge must not outlive a failed one.
        for r in trace.regions:
            r.ncu = {}
        ok, why = self.available()
        if not ok:
            return self.declare("unavailable", why)
        rows, why = read_report(self.report)
        if why:
            return self.declare("failed", why)
        d = aggregate(trace, rows, self.top)
        status = ("captured" if d["matched_regions"] and not d["metrics_missing"]
                  else "partial")
        reason = ("" if d["matched_regions"] else
                  "NVTX ranges in the report do not match the region list")
        if d["metrics_missing"]:
            reason = (reason + "; " if reason else "") + "some metrics not collected"
        return self.declare(
            status, reason, report=self.report, projection=PROJECTION,
            nvtx_filter=NVTX_FILTER, metrics=list(METRICS),
            replay="kernel", cache_control=CACHE_CONTROL, clock_control=CLOCK_CONTROL,
            note=("kernels are serialized and replayed with caches flushed and clocks "
                  "locked to base; kernel_us is not a normal-run duration and is not "
                  "comparable with Nsight Systems kernel_us"),
            **d)
