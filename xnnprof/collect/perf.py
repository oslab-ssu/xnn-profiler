"""perf stat collector. Counts hardware events process-wide.

perf counts per process (thread), so values cannot be attributed to layers or regions.
The window is a loop of forwards with hooks and profiler detached; values are also
reported per forward. The window is opened and closed through a `--control` FIFO so
perf start-up latency is not included.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from .base import PROCESS_GLOBAL, Collector
from ..schema import Source, Trace

EVENTS = ("task-clock", "cycles", "instructions", "cache-references",
          "cache-misses", "branches", "branch-misses", "page-faults",
          "context-switches")


def paranoid() -> Optional[int]:
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def parse_csv(text: str) -> dict[str, dict[str, Any]]:
    """Parse `perf stat -x,` output. Fields: value,unit,event,run time,enabled %,...

    `<not counted>` / `<not supported>` become None, which must stay distinct from 0.
    """
    out: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) < 3:
            continue
        raw, unit, event = f[0], f[1], f[2]
        try:
            value: Optional[float] = float(raw)
        except ValueError:
            value = None
        enabled = None
        if len(f) > 4:
            try:
                enabled = float(f[4])
            except ValueError:
                pass
        out[event] = {"value": value, "unit": unit, "raw": raw,
                      "enabled_pct": enabled}
    return out


def _canon(event: str) -> str:
    return event.split(":")[0].replace("cpu/", "").rstrip("/")


def derive(counts: dict[str, dict[str, Any]], reps: int) -> dict[str, Any]:
    v = {_canon(k): c["value"] for k, c in counts.items()}
    per = {k: (x / reps if x is not None else None) for k, x in v.items()}
    d: dict[str, Any] = {"per_forward": per}
    if v.get("cycles") and v.get("instructions") is not None:
        d["ipc"] = round(v["instructions"] / v["cycles"], 4)
    if v.get("cache-references") and v.get("cache-misses") is not None:
        d["cache_miss_rate"] = round(v["cache-misses"] / v["cache-references"], 6)
    if v.get("branches") and v.get("branch-misses") is not None:
        d["branch_miss_rate"] = round(v["branch-misses"] / v["branches"], 6)
    mux = [c["enabled_pct"] for c in counts.values()
           if c.get("enabled_pct") is not None]
    d["min_enabled_pct"] = min(mux) if mux else None
    d["not_counted"] = sorted(k for k, c in counts.items() if c["value"] is None)
    return d


class PerfCollector(Collector):
    name = "perf"
    attribution = PROCESS_GLOBAL
    capabilities = ("cycles", "instructions", "ipc", "cache_misses", "page_faults")

    def __init__(self, events: tuple[str, ...] = EVENTS):
        self.events = events
        self._counts: Optional[dict] = None
        self._reps = 0
        self._wall_us = 0.0
        self._err = ""

    def available(self) -> tuple[bool, str]:
        if os.name != "posix":
            return False, "perf is Linux-only"
        if not shutil.which("perf"):
            return False, "perf not installed"
        p = paranoid()
        if p is not None and p > 2 and os.geteuid() != 0:
            return False, (f"kernel.perf_event_paranoid={p}; user-space counters "
                           f"need 2 or lower")
        return True, "ok"

    @contextmanager
    def window(self, reps: int) -> Iterator[None]:
        """Count only what runs inside this block. The block must run `reps` forwards."""
        ok, why = self.available()
        if not ok:
            self._err = why
            yield
            return
        tmp = tempfile.mkdtemp(prefix="xnnperf-")
        ctl, ack, out = (os.path.join(tmp, n) for n in ("ctl", "ack", "out.csv"))
        os.mkfifo(ctl), os.mkfifo(ack)
        cmd = ["perf", "stat", "-x,", "-e", ",".join(self.events),
               "-p", str(os.getpid()), "-D", "-1",
               "--control", f"fifo:{ctl},{ack}", "-o", out]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        cfd = afd = -1
        try:
            cfd = os.open(ctl, os.O_WRONLY)
            afd = os.open(ack, os.O_RDONLY)
            self._send(cfd, afd, "enable")
            t0 = time.perf_counter()
            yield
            self._wall_us = (time.perf_counter() - t0) * 1e6
            self._send(cfd, afd, "disable")
        finally:
            for fd in (cfd, afd):
                if fd >= 0:
                    os.close(fd)
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            try:
                _, err = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, err = proc.communicate()
            if os.path.exists(out):
                with open(out) as f:
                    self._counts = parse_csv(f.read())
            if not self._counts:
                self._err = (err or "").strip()[-400:] or f"perf rc={proc.returncode}"
            shutil.rmtree(tmp, ignore_errors=True)
        self._reps = reps

    @staticmethod
    def _send(cfd: int, afd: int, cmd: str) -> None:
        os.write(cfd, (cmd + "\n").encode())
        got = os.read(afd, 16)
        if not got.startswith(b"ack"):
            raise RuntimeError(f"unexpected perf --control reply: {got!r}")

    def collect(self, trace: Trace, **kw: Any) -> Source:
        if self._counts is None:
            ok, why = self.available()
            if not ok:
                return self.declare("unavailable", why)
            return self.declare("unavailable", self._err or "no measurement window opened (--perf not given)")
        d = derive(self._counts, max(self._reps, 1))
        status = "captured" if not d["not_counted"] else "partial"
        return self.declare(
            status, "process-wide counters; not attributable to layers or regions",
            window="uninstrumented-forward", reps=self._reps,
            wall_us_per_forward=round(self._wall_us / max(self._reps, 1), 3),
            events=list(self.events), counts=self._counts, **d)
