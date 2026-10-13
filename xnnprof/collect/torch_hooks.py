"""PyTorch hook 수집기 — 레이어·타임스텝 구간, 희소성, 스파이크, 지속 상태.

모델 코드를 수정하지 않는다. 레이어 타입을 훑어 hook을 붙이므로 DNN·SNN·혼합이
같은 경로로 측정된다. 구간(region)이 다른 도구와의 상관 키가 된다.
"""

from __future__ import annotations

import time
from typing import Any

from ..schema import DENSE, Layer, Region, SpikeRef, STATE, Trace
from .base import Collector, REGION


def is_snn(module) -> bool:
    return type(module).__module__.startswith(("snntorch", "spikingjelly"))


def macs_of(module, x, out) -> int:
    import torch.nn as nn
    if isinstance(module, nn.Linear):
        return int(x.numel() // x.shape[-1]) * module.in_features * module.out_features
    if isinstance(module, nn.Conv2d):
        o = out[0] if isinstance(out, tuple) else out
        k = module.kernel_size[0] * module.kernel_size[1]
        return (int(o.shape[0]) * module.out_channels
                * int(o.shape[-2] * o.shape[-1])
                * (module.in_channels // module.groups) * k)
    return 0


def _fan(module) -> dict:
    import torch.nn as nn
    if isinstance(module, nn.Linear):
        return {"in": module.in_features, "out": module.out_features}
    if isinstance(module, nn.Conv2d):
        return {"in": module.in_channels, "out": module.out_channels}
    return {}


def looks_binary(x) -> bool:
    import torch
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return False
    return bool(((x == 0) | (x == 1)).all())


class TorchHookCollector(Collector):
    name = "torch_hooks"
    attribution = REGION
    capabilities = ("layer", "timestep", "macs", "sparsity", "spikes",
                    "persistent_state", "domain")

    def __init__(self, emit_nvtx: bool = False):
        self.emit_nvtx = emit_nvtx
        self._handles: list = []
        self._acc: dict[str, dict] = {}
        self._spikes: dict[str, list] = {}
        self._regions: list[Region] = []
        self._t0 = 0.0

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as e:
            return False, f"torch import 실패: {e}"
        return True, "ok"

    # ---------------- hook 부착 ----------------
    def attach(self, model) -> None:
        import torch.nn as nn
        self._t0 = time.perf_counter()
        for name, m in model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) or is_snn(m):
                self._handles.append(m.register_forward_pre_hook(self._pre(name)))
                self._handles.append(m.register_forward_hook(self._post(name)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pre(self, name):
        def pre(mod, inputs):
            a = self._acc.setdefault(name, {"open": None})
            a["open"] = time.perf_counter()
            if self.emit_nvtx:
                try:
                    import torch
                    step = self._acc[name].get("calls", 0)
                    torch.cuda.nvtx.range_push(f"{name}@t{step}")
                except Exception:
                    pass
        return pre

    def _post(self, name):
        def post(mod, inputs, output):
            import torch
            now = time.perf_counter()
            a = self._acc.setdefault(name, {})
            opened = a.pop("open", None)
            step = a.get("calls", 0)
            a["calls"] = step + 1
            a.setdefault("op_type", type(mod).__name__)
            a.setdefault("is_snn", is_snn(mod))
            a.setdefault("macs", 0)
            a.setdefault("in_elems", 0)
            a.setdefault("in_nonzero", 0)
            a.setdefault("self_time_us", 0.0)
            a.setdefault("spike_driven", False)
            a.setdefault("state_bytes", 0)
            if "fan" not in a:
                a["fan"] = _fan(mod)

            x = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if x is not None:
                a["macs"] += macs_of(mod, x, output)
                a["in_elems"] += int(x.numel())
                a["in_nonzero"] += int((x != 0).sum().item())
                if step == 0:
                    a["spike_driven"] = looks_binary(x)
            if opened is not None:
                dur = (now - opened) * 1e6
                a["self_time_us"] += dur
                self._regions.append(Region(
                    region_id=len(self._regions), layer_id=name, timestep=step,
                    ts_us=(opened - self._t0) * 1e6, dur_us=dur,
                    domain=STATE if a["is_snn"] else DENSE))

            if a["is_snn"]:
                spk = output[0] if isinstance(output, tuple) else output
                mem = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if torch.is_tensor(spk):
                    self._spikes.setdefault(name, []).append(
                        spk.detach().to(torch.uint8).cpu())
                if mem is not None and torch.is_tensor(mem):
                    a["state_bytes"] = int(mem.numel() * mem.element_size())
            if self.emit_nvtx:
                try:
                    import torch
                    torch.cuda.nvtx.range_pop()
                except Exception:
                    pass
        return post

    # ---------------- 통일 포맷으로 ----------------
    def spike_arrays(self) -> dict:
        import numpy as np
        import torch
        return {k: np.ascontiguousarray(torch.stack(v).numpy(), dtype=np.uint8)
                for k, v in self._spikes.items() if v}

    def collect(self, trace: Trace, spike_dir: str = "", **kw: Any):
        from ..observations import write_spike_observation

        for name, a in self._acc.items():
            nz = (a["in_nonzero"] / a["in_elems"]) if a.get("in_elems") else None
            trace.layers.append(Layer(
                layer_id=name, op_type=a.get("op_type", "?"), calls=a.get("calls", 0),
                in_loop=a.get("calls", 0) > 1,
                domain=STATE if a.get("is_snn") else DENSE,
                spike_driven=bool(a.get("spike_driven")),
                macs=a.get("macs", 0),
                useful_macs=int(a.get("macs", 0) * nz) if nz is not None else None,
                input_nonzero_fraction=nz,
                self_time_us=round(a.get("self_time_us", 0.0), 3),
                state_bytes=a.get("state_bytes", 0),
                extra={"fan": a.get("fan", {})}))
        trace.layers.sort(key=lambda l: -l.macs)
        trace.regions.extend(self._regions)

        if spike_dir:
            for lid, arr in self.spike_arrays().items():
                o = write_spike_observation(
                    arr, out_dir=spike_dir, layer_id=lid,
                    run_tag=f"T{trace.timesteps}", state_reset="subtract",
                    encoding="direct")
                trace.spikes.append(SpikeRef(
                    layer_id=o.layer_id, path=o.path, sha256=o.sha256,
                    timesteps=o.timesteps, batch=o.batch, neurons=o.neurons,
                    metrics=o.metrics))
        return self.declare("captured", layers=len(self._acc), regions=len(self._regions))
