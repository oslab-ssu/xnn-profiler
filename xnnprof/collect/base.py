"""Collector interface.

Every tool sits behind it, and only the unified format is exposed outward.
"""

from __future__ import annotations

from typing import Any

from ..schema import Source, Trace

REGION = "region"
LAYER = "layer"
MODEL_GLOBAL = "model-global"
PROCESS_GLOBAL = "process-global"
STATIC_NAME = "static-name"


class Collector:
    name = "base"
    attribution = "none"
    capabilities: tuple[str, ...] = ()

    def available(self) -> tuple[bool, str]:
        return False, "not implemented"

    def collect(self, trace: Trace, **kw: Any) -> Source:
        raise NotImplementedError

    def declare(self, status: str, reason: str = "", **detail: Any) -> Source:
        return Source(collector=self.name, status=status, attribution=self.attribution,
                      capabilities=list(self.capabilities), reason=reason, detail=detail)


class Unimplemented(Collector):
    """A collector that declares only the interface contract.

    It is never skipped silently; it is recorded in the provenance.
    """

    def collect(self, trace: Trace, **kw: Any) -> Source:
        ok, why = self.available()
        return self.declare("unavailable",
                            "adapter not implemented (the tool itself is available)"
                            if ok else why)
