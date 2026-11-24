"""Collectors not yet wired in. They declare the interface contract and attribution limit up front.

They never drop out silently; each is recorded in `sources` as 'unavailable' so that
a tool limitation can be told apart from a genuinely missing measurement.
"""

from __future__ import annotations

from .base import STATIC_NAME, Unimplemented


class MlirStaticCollector(Unimplemented):
    name = "mlir_static"
    attribution = STATIC_NAME
    capabilities = ("ir_op_counts", "static_memory", "lowering_expansion")

    def available(self):
        return False, "no compiler output path given (.mlir / .ll)"


ALL = (MlirStaticCollector,)
