"""Unified profile format xnn.trace. Every collector normalizes its output to this schema."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

SCHEMA = "xnn.trace"
VERSION = 1

DENSE, SPIKE, STATE = "dense", "spike", "state"


@dataclass
class Region:
    """(layer_id, timestep) time region. The correlation key across tools."""
    region_id: int
    layer_id: str
    timestep: Optional[int]
    ts_us: float
    dur_us: float
    domain: str = DENSE
    parent_id: Optional[int] = None
    ops: dict[str, float] = field(default_factory=dict)   # op name -> self time (us)
    # Operator call hierarchy inside the region. Offsets are relative to the label
    # start, so no clock alignment is needed. {name, depth, off_us, dur_us}
    op_tree: list[dict[str, Any]] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)     # projected from Nsight Systems
    ncu: dict[str, Any] = field(default_factory=dict)     # counters from Nsight Compute
    ir: dict[str, Any] = field(default_factory=dict)      # static op counts from MLIR


@dataclass
class Layer:
    layer_id: str
    op_type: str
    calls: int = 0
    in_loop: bool = False
    domain: str = DENSE
    spike_driven: bool = False
    macs: int = 0
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
    steady_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class MembraneRef:
    """Reference to the membrane-potential observation file. `[T, 66]` histogram bin counts."""
    layer_id: str
    path: str
    sha256: str = ""
    timesteps: int = 0
    bins: int = 0
    threshold: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    """Provenance declaration for one collector.

    `attribution` is the finest unit the tool can attribute to *in principle*.
    perf is process-global, so it cannot be attributed to layers; hiding that
    would produce false numbers.
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
    membranes: list[MembraneRef] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    ir_levels: list[IrLevel] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def layer(self, layer_id: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.layer_id == layer_id), None)

    def executed_macs(self) -> int:
        return sum(l.macs for l in self.layers)

    def region(self, layer_id: str, timestep: Optional[int]) -> Optional[Region]:
        return next((r for r in self.regions
                     if r.layer_id == layer_id and r.timestep == timestep), None)

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
        # The only output file is a Chrome Trace Event JSON; our content lives in
        # its `xnn` extension key. Accept either form.
        if "traceEvents" in d and isinstance(d.get("xnn"), dict):
            d = d["xnn"]
        d.pop("schema", None), d.pop("version", None)
        return Trace(
            regions=[Region(**r) for r in d.pop("regions", [])],
            layers=[Layer(**l) for l in d.pop("layers", [])],
            spikes=[SpikeRef(**s) for s in d.pop("spikes", [])],
            membranes=[MembraneRef(**m) for m in d.pop("membranes", [])],
            sources=[Source(**s) for s in d.pop("sources", [])],
            ir_levels=[IrLevel(**i) for i in d.pop("ir_levels", [])],
            **d)
