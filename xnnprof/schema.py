"""통일 프로파일 포맷 xnn.trace. 모든 수집기가 이 스키마로 정규화해 반환한다."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

SCHEMA = "xnn.trace"
VERSION = 1

DENSE, SPIKE, STATE = "dense", "spike", "state"


@dataclass
class Region:
    """(layer_id, timestep) 시간 구간. 도구 간 상관 키다."""
    region_id: int
    layer_id: str
    timestep: Optional[int]
    ts_us: float
    dur_us: float
    domain: str = DENSE
    parent_id: Optional[int] = None


@dataclass
class Layer:
    layer_id: str
    op_type: str
    calls: int = 0
    in_loop: bool = False
    domain: str = DENSE
    spike_driven: bool = False
    macs: int = 0
    useful_macs: Optional[int] = None
    input_nonzero_fraction: Optional[float] = None
    self_time_us: Optional[float] = None
    state_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeRef:
    layer_id: str
    path: str
    sha256: str = ""
    timesteps: int = 0
    batch: int = 0
    neurons: int = 0
    fanout: Optional[int] = None
    fanout_from: str = ""
    synops: Optional[int] = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    """수집기 1종의 출처 선언.

    attribution은 그 도구가 '원리적으로' 어느 단위까지 귀속 가능한지다.
    perf는 프로세스 전역이므로 레이어 귀속이 불가하고, 그걸 숨기면 거짓 수치가 된다.
    """
    collector: str
    status: str = "unavailable"           # captured | partial | unavailable | failed
    attribution: str = "none"            # region | layer | process-global | static-name
    capabilities: list[str] = field(default_factory=list)
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class IrLevel:
    level: str
    total_operations: int = 0
    operation_details: dict[str, int] = field(default_factory=dict)
    static_memory_kb: float = 0.0


@dataclass
class Trace:
    run_id: str
    model_id: str
    framework: str = "pytorch"
    timesteps: Optional[int] = None
    batch: Optional[int] = None
    mode: str = "inference"
    env: dict[str, Any] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    spikes: list[SpikeRef] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    ir_levels: list[IrLevel] = field(default_factory=list)
    headroom: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def layer(self, layer_id: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.layer_id == layer_id), None)

    def executed_macs(self) -> int:
        return sum(l.macs for l in self.layers)

    def useful_macs(self) -> int:
        return sum(l.useful_macs if l.useful_macs is not None else l.macs
                   for l in self.layers)

    def looped_macs(self) -> int:
        return sum(l.macs for l in self.layers if l.in_loop)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"], d["version"] = SCHEMA, VERSION
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "Trace":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("schema", None), d.pop("version", None)
        return Trace(
            regions=[Region(**r) for r in d.pop("regions", [])],
            layers=[Layer(**l) for l in d.pop("layers", [])],
            spikes=[SpikeRef(**s) for s in d.pop("spikes", [])],
            sources=[Source(**s) for s in d.pop("sources", [])],
            ir_levels=[IrLevel(**i) for i in d.pop("ir_levels", [])],
            **d)
