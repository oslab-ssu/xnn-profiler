"""Collectors not yet wired in. They declare the interface contract and attribution limit up front.

They never drop out silently; each is recorded in `sources` as 'unavailable' so that
a tool limitation can be told apart from a genuinely missing measurement.
"""

from __future__ import annotations

import shutil

from .base import REGION, STATIC_NAME, Unimplemented


class NcuCollector(Unimplemented):
    name = "nsight_compute"
    attribution = REGION
    capabilities = ("kernel_counters", "occupancy", "memory_bandwidth")

    def available(self):
        return (True, "ok") if shutil.which("ncu") else (False, "ncu not installed")


class MlirStaticCollector(Unimplemented):
    name = "mlir_static"
    attribution = STATIC_NAME
    capabilities = ("ir_op_counts", "static_memory", "lowering_expansion")

    def available(self):
        return False, "no compiler output path given (.mlir / .ll)"


ALL = (NcuCollector, MlirStaticCollector)
