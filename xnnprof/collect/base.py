"""수집기 인터페이스. 모든 도구는 이 뒤에 두고, 통일 포맷만 바깥에 노출한다."""

from __future__ import annotations

from typing import Any

from ..schema import Source, Trace

REGION = "region"
LAYER = "layer"
PROCESS_GLOBAL = "process-global"
STATIC_NAME = "static-name"


class Collector:
    name = "base"
    attribution = "none"
    capabilities: tuple[str, ...] = ()

    def available(self) -> tuple[bool, str]:
        return False, "미구현"

    def collect(self, trace: Trace, **kw: Any) -> Source:
        raise NotImplementedError

    def declare(self, status: str, reason: str = "", **detail: Any) -> Source:
        return Source(collector=self.name, status=status, attribution=self.attribution,
                      capabilities=list(self.capabilities), reason=reason, detail=detail)


class Unimplemented(Collector):
    """인터페이스 계약만 선언된 수집기. 조용히 건너뛰지 않고 출처에 남는다."""

    def collect(self, trace: Trace, **kw: Any) -> Source:
        ok, why = self.available()
        return self.declare("unavailable",
                            "어댑터 미구현 (도구는 사용 가능)" if ok else why)
