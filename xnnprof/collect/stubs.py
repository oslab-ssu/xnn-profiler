"""Collectors not wired up yet. They declare their interface contract and
attribution limit up front.

They do not drop out silently; they appear in `sources` as 'unavailable', so
that a tool's inherent limit can be told apart from an actual omission.
"""

from __future__ import annotations

import shutil

from .base import PROCESS_GLOBAL, REGION, STATIC_NAME, Unimplemented


class PerfCollector(Unimplemented):
    name = "perf"
    attribution = PROCESS_GLOBAL
    capabilities = ("cycles", "instructions", "ipc", "cache_misses", "page_faults")

    def available(self):
        return (True, "ok") if shutil.which("perf") else (False, "perf is not installed")


class NcuCollector(Unimplemented):
    name = "nsight_compute"
    attribution = REGION
    capabilities = ("kernel_counters", "occupancy", "memory_bandwidth")

    def available(self):
        return (True, "ok") if shutil.which("ncu") else (False, "ncu is not installed")


class MlirStaticCollector(Unimplemented):
    name = "mlir_static"
    attribution = STATIC_NAME
    capabilities = ("ir_op_counts", "static_memory", "lowering_expansion")

    def available(self):
        return False, "no path given for the compiler artifacts (.mlir / .ll)"


ALL = (PerfCollector, NcuCollector, MlirStaticCollector)
