"""Nsight Systems collector: projects GPU kernels onto NVTX ranges so they
land in the right region.

The hooks leave the unified key string `fc1@t3` as an NVTX range, so finding
that name in the report immediately identifies the region. No clock origin
has to be aligned.

The projection rule matches nsys's own `nvtx_gpu_proj_sum`: a kernel counts
toward the range whose CUDA runtime API call launched it. Kernels run
asynchronously, so the decision must not be made from the kernel's execution
timestamp -- a kernel running after the range has closed is normal.

    NVTX_EVENTS                      range name, [start, end), thread
    CUPTI_ACTIVITY_KIND_RUNTIME      runtime API calls, carrying correlationId
    CUPTI_ACTIVITY_KIND_KERNEL       kernel executions, joined on correlationId

The report (`.nsys-rep`) is exported to SQLite and read directly. The tables
printed by `nsys stats` are not parsed because their column names and units
change between versions. If a `.sqlite` already exists, the merge works
without nsys installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from typing import Any

from ..schema import Trace
from .base import Collector, REGION
from .torch_prof import LABEL

PROJECTION = "launch-in-range"      # a kernel counts toward the range that launched it

QUERY = """
SELECT n.text, s.value, COUNT(*), SUM(k.end - k.start),
       MIN(k.start), MAX(k.end)
FROM NVTX_EVENTS n
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.globalTid = n.globalTid AND r.start >= n.start AND r.start <= n.end
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON k.correlationId = r.correlationId
LEFT JOIN StringIds s ON s.id = k.shortName
WHERE n.text LIKE '%@t%'
GROUP BY n.text, s.value
"""

TOTAL_KERNEL = "SELECT SUM(end - start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"


def nsys_bin() -> str:
    """The `nsys` executable. If it is not on PATH, set the `NSYS` env var.

    The Windows installer does not put it on PATH, and the identically named
    binary shipped with Nsight Compute cannot produce a report, so there has
    to be a way to point at the right path directly.
    """
    env = os.environ.get("NSYS", "").strip('"')
    if env and os.path.exists(env):
        return env
    return shutil.which("nsys") or ""


def sqlite_for(report: str, export: bool = True) -> tuple[str, str]:
    """Path of the `.sqlite` matching the report, created via `nsys export`
    if missing.
    """
    if report.endswith(".sqlite"):
        return (report, "") if os.path.exists(report) else ("", f"{report} not found")
    if not os.path.exists(report):
        return "", f"{report} not found"
    db = os.path.splitext(report)[0] + ".sqlite"
    # If the report was regenerated but an old sqlite is still around, the
    # merge would silently use the previous measurement. Re-export any db
    # older than the report.
    if os.path.exists(db) and os.path.getmtime(db) >= os.path.getmtime(report):
        return db, ""
    if not export:
        stale = "older than the report" if os.path.exists(db) else "not found"
        return "", f"{db} {stale} (export skipped)"
    nsys = nsys_bin()
    if not nsys:
        return "", (f"{db} is missing and nsys could not be found, so it "
                    "cannot be exported -- put nsys on PATH or set the NSYS "
                    "env var to its path")
    cmd = [nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
           "-o", db, report]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:                                   # noqa: BLE001
        return "", f"nsys export failed: {type(e).__name__}: {e}"
    if p.returncode != 0 or not os.path.exists(db):
        return "", f"nsys export failed (rc={p.returncode}): {p.stderr.strip()[:200]}"
    return db, ""


class NsysCollector(Collector):
    name = "nsight_systems"
    attribution = REGION
    capabilities = ("cuda_api", "kernel_timeline", "nvtx_ranges", "gpu_projection")

    def __init__(self, report: str = "", export: bool = True, top: int = 3):
        self.report = report
        self.export = export
        self.top = top

    def available(self) -> tuple[bool, str]:
        if not self.report:
            return False, "no report path given (--nsys-report)"
        db, why = sqlite_for(self.report, export=False)
        if db:
            return True, "ok"
        if nsys_bin():
            return True, "ok (export required)"
        return False, why

    def collect(self, trace: Trace, **kw: Any):
        import sqlite3

        if not self.report:
            return self.declare("unavailable", "no report path given (--nsys-report)")
        db, why = sqlite_for(self.report, self.export)
        if not db:
            return self.declare("unavailable", why)

        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = list(con.execute(QUERY))
            total_us, total_n = con.execute(TOTAL_KERNEL).fetchone()
        except Exception as e:                               # noqa: BLE001
            return self.declare("failed", f"{type(e).__name__}: {e}")
        finally:
            try:
                con.close()
            except Exception:
                pass

        per: dict[tuple[str, int], dict] = defaultdict(
            lambda: {"ns": 0, "n": 0, "lo": None, "hi": None, "k": defaultdict(int)})
        for text, kname, n, ns, lo, hi in rows:
            m = LABEL.match(str(text))
            if not m:
                continue
            key = (m.group("layer"), int(m.group("step")))
            d = per[key]
            d["ns"] += int(ns or 0)
            d["n"] += int(n or 0)
            d["lo"] = int(lo) if d["lo"] is None else min(d["lo"], int(lo))
            d["hi"] = int(hi) if d["hi"] is None else max(d["hi"], int(hi))
            d["k"][str(kname)] += int(ns or 0)

        matched, in_ranges = 0, 0
        for (lid, step), d in per.items():
            r = trace.region(lid, step)
            if r is None:
                continue
            top = sorted(d["k"].items(), key=lambda kv: -kv[1])[:self.top]
            r.gpu = {
                "kernel_us": round(d["ns"] / 1000.0, 3),
                "kernels": d["n"],
                "span_us": round((d["hi"] - d["lo"]) / 1000.0, 3),
                "projection": PROJECTION,
                "top_kernels": [{"name": k, "us": round(v / 1000.0, 3)}
                                for k, v in top],
            }
            in_ranges += d["ns"]
            matched += 1

        total_us = float(total_us or 0) / 1000.0
        in_us = in_ranges / 1000.0
        return self.declare(
            "captured" if matched else "partial",
            "" if matched else "the report's NVTX ranges do not match the region list",
            sqlite=db, projection=PROJECTION,
            nvtx_ranges=len(per), matched_regions=matched,
            kernel_us_in_ranges=round(in_us, 3),
            kernel_us_total=round(total_us, 3),
            kernels_total=int(total_n or 0),
            in_range_fraction=round(in_us / total_us, 6) if total_us else 0.0,
            note=("kernels outside any range belong to execution that is not "
                  "part of the measurement -- warm-up, training, accuracy "
                  "evaluation. The capture wraps the whole process"))
