"""PyTorch hook collector: builds unified keys, times regions, opens labels
and captures SNN metrics.

Model code is never modified. Hooks are registered by walking the model tree
and matching on layer type, so DNN, SNN and mixed models all go through the
same path. A region denotes one execution of the unified key
`(layer_id, timestep)` and is the correlation basis for every other tool.

The hooks do two things.

  build unified keys     take timestamps immediately before and after the
  and regions            call to get `[start, end)`, assign the timestep from
                         the call count, and open/close the key string
                         `fc1@t3` as a record_function and NVTX label
  capture SNN metrics    MACs, spike arrays, state tensor size -- values no
                         existing tool observes
"""

from __future__ import annotations

import time
from typing import Any

from ..observations import MEM_BINS, MEM_IN_RANGE
from ..schema import DENSE, Layer, MembraneRef, Region, SpikeRef, STATE, Trace
from .base import Collector, REGION

INSTRUMENT_LABEL = "xnn::instrument"   # label holding instrumentation's own cost


def is_snn(module) -> bool:
    return type(module).__module__.startswith(("snntorch", "spikingjelly"))


def theta_of(module):
    """Firing threshold theta, read once at `attach()` and cached.

    snntorch keeps `threshold` as a tensor buffer, so calling `float()` on it
    inside a hook pulls the value off the GPU and forces a host sync -- which
    makes the instrumentation reorder the execution it is measuring. If the
    value cannot be read, the metric is simply not produced.
    """
    t = getattr(module, "threshold", None)
    if t is None:
        return None
    try:
        v = float(t)
    except Exception:
        return None
    return v if v > 0 else None


def layer_id_of(name: str, module) -> str:
    """The `layer_id` of the unified key; numeric names get their type appended.

    Children of an `nn.Sequential` come out of `named_modules()` as indices
    such as `1` or `dnn1.0`. Left as-is they are unreadable in the result
    file, so they are normalized to `1:Linear`, `dnn1.0:Conv2d`.
    """
    head, _, last = name.rpartition(".")
    if last.isdigit():
        return f"{name}:{type(module).__name__}"
    return name


def targets(model, cover_leaves: bool = False):
    """Modules to instrument. Walks the tree without assuming a model shape.

    The default rule covers modules that perform MACs (`nn.Linear`,
    `nn.Conv2d`) and SNN neuron modules, which hold state and are called once
    per timestep. Containers such as `nn.Sequential` are excluded: they would
    create an outer region enclosing their children and count the same time
    twice.

    With `cover_leaves` the rule widens to every module without children.
    That gives a region for the time actually spent in modules with no MACs
    (`nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`), at the cost of more regions and
    more instrumentation overhead. It exists so the attribution rate of the
    two rules can be compared.
    """
    import torch.nn as nn
    for name, m in model.named_modules():
        if not name:                                  # the root module
            continue
        if isinstance(m, (nn.Linear, nn.Conv2d)) or is_snn(m):
            yield name, m
        elif cover_leaves and not next(m.children(), None):
            yield name, m


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
    """Decide whether a layer is spike-driven, i.e. whether its input is binary.

    Must never be called inside a measured region: it adds a kernel that scans
    the whole tensor, mixing instrumentation cost into the measurement. The
    first call's input is kept by reference only and tested in `collect()`
    once measurement is done.
    """
    import torch
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return False
    return bool(((x == 0) | (x == 1)).all())


class TorchHookCollector(Collector):
    name = "torch_hooks"
    attribution = REGION
    capabilities = ("layer", "timestep", "macs", "spikes",
                    "persistent_state", "membrane_histogram", "domain", "label")

    def __init__(self, emit_labels: bool = True, emit_nvtx: bool = False,
                 cover_leaves: bool = False, sync_regions: bool = False,
                 membrane: bool = True):
        self.emit_labels = emit_labels
        self.cover_leaves = cover_leaves
        self.sync_regions = sync_regions
        self.membrane = membrane
        self._cuda: bool | None = None
        self.emit_nvtx = emit_nvtx
        self._handles: list = []
        self._acc: dict[str, dict] = {}
        self._spikes: dict[str, list] = {}
        self._hist: dict[str, list] = {}
        self._theta: dict[str, float] = {}
        self._mem_skip: dict[str, str] = {}
        self._regions: list[Region] = []
        self._open_labels: dict[str, list] = {}
        self._t0 = 0.0

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as e:
            return False, f"torch import failed: {e}"
        return True, "ok"

    def _sync(self) -> None:
        """Wait for the GPU at region boundaries.

        The GPU runs asynchronously, so what the hooks time is the launch
        window, not kernel execution. A layer called while the queue is full
        absorbs the wait caused by earlier layers, which makes a neuron module
        with no arithmetic look slower than a matrix multiply. Synchronizing at
        every boundary makes each region time that layer's real GPU time, but
        removes the overlap between layers, so the total exceeds real wall time.
        """
        if not self.sync_regions:
            return
        import torch
        if self._cuda is None:
            self._cuda = torch.cuda.is_available()
        if self._cuda:
            torch.cuda.synchronize()

    # ---------------- attaching hooks ----------------
    def attach(self, model) -> None:
        for name, m in targets(model, self.cover_leaves):
            lid = layer_id_of(name, m)
            if self.membrane and is_snn(m):
                theta = theta_of(m)
                if theta is None:
                    self._mem_skip[lid] = "could not read the threshold attribute"
                else:
                    self._theta[lid] = theta
        self._t0 = time.perf_counter()        # set the origin after reading all thetas
        for name, m in targets(model, self.cover_leaves):
            lid = layer_id_of(name, m)
            self._handles.append(m.register_forward_pre_hook(self._pre(lid)))
            self._handles.append(m.register_forward_hook(self._post(lid)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for stack in self._open_labels.values():          # close labels left by an exception
            while stack:
                self._close_label(stack.pop())
        self._open_labels.clear()

    # ---------------- opening and closing labels ----------------
    def _open_label(self, key: str):
        """Stamp the unified key string into each tool's output.

        Returns the handle used to close it.
        """
        rf = nvtx = None
        if self.emit_labels:
            try:
                from torch.profiler import record_function
                rf = record_function(key)
                rf.__enter__()
            except Exception:
                rf = None
        if self.emit_nvtx:
            try:
                import torch
                torch.cuda.nvtx.range_push(key)
                nvtx = True
            except Exception:
                nvtx = None
        return rf, nvtx

    @staticmethod
    def _close_label(handle) -> None:
        rf, nvtx = handle
        if nvtx:
            try:
                import torch
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
        if rf is not None:
            try:
                rf.__exit__(None, None, None)
            except Exception:
                pass

    def _pre(self, lid: str):
        def pre(mod, inputs):
            self._sync()                  # drain the previous layer's kernels first
            a = self._acc.setdefault(lid, {})
            step = a.get("calls", 0)
            self._open_labels.setdefault(lid, []).append(
                self._open_label(f"{lid}@t{step}"))
            a["open"] = time.perf_counter()
        return pre

    def _post(self, lid: str):
        def post(mod, inputs, output):
            import torch
            self._sync()                  # wait for this layer's kernels to finish
            now = time.perf_counter()
            # Close the label first. The tensor checks and copies below are
            # instrumentation cost; left inside the label they would be
            # attributed to this layer's compute time.
            stack = self._open_labels.get(lid)
            if stack:
                self._close_label(stack.pop())
            inst = self._open_label(INSTRUMENT_LABEL) if self.emit_labels else None
            a = self._acc.setdefault(lid, {})
            opened = a.pop("open", None)
            step = a.get("calls", 0)
            a["calls"] = step + 1
            a.setdefault("op_type", type(mod).__name__)
            a.setdefault("is_snn", is_snn(mod))
            a.setdefault("macs", 0)
            a.setdefault("self_time_us", 0.0)
            a.setdefault("spike_driven", False)
            a.setdefault("state_bytes", 0)
            if "fan" not in a:
                a["fan"] = _fan(mod)

            x = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if x is not None:
                a["macs"] += macs_of(mod, x, output)
                if step == 0:
                    a["probe"] = x.detach()   # tested after measurement (looks_binary)
            if opened is not None:
                dur = (now - opened) * 1e6
                a["self_time_us"] += dur
                self._regions.append(Region(
                    region_id=len(self._regions), layer_id=lid, timestep=step,
                    ts_us=(opened - self._t0) * 1e6, dur_us=dur,
                    domain=STATE if a["is_snn"] else DENSE))

            if a["is_snn"]:
                spk = output[0] if isinstance(output, tuple) else output
                # The number of state tensors varies by neuron type: Leaky is
                # (spk, mem), Synaptic is (spk, syn, mem), Alpha is
                # (spk, syn_exc, syn_inh, mem). Only the first item is the
                # spike; everything after it is persistent state.
                states = [t for t in (output[1:] if isinstance(output, tuple) else ())
                          if torch.is_tensor(t)]
                if torch.is_tensor(spk):
                    # Accumulate on-device. Calling .cpu() each step would
                    # force a sync per step and let the instrumentation
                    # reorder execution.
                    self._spikes.setdefault(lid, []).append(
                        spk.detach().to(torch.uint8))
                if states:
                    a["state_bytes"] = sum(int(t.numel() * t.element_size())
                                           for t in states)
                    a["state_tensors"] = len(states)
                    # The membrane potential is the last state tensor of the
                    # output tuple: Leaky (spk, mem), Synaptic (spk, syn, mem),
                    # Alpha (spk, syn_exc, syn_inh, mem).
                    if self.membrane:
                        self._bucket(lid, states[-1])
            if inst is not None:
                self._close_label(inst)
        return post

    def _bucket(self, lid: str, mem) -> None:
        """Convert the membrane potential to a 66-bin histogram on-device.

        It costs two kernels and never pulls a value to the host, so there is
        no host sync. The raw tensor is not retained: memory is one of the
        things being measured, and holding that much per step would change the
        measurement.
        """
        import torch
        theta = self._theta.get(lid)
        if theta is None:
            return
        # Bins are equal width, so the bin index is arithmetic, not a search.
        #   mem < 0 -> 0 | [0, theta) -> 1~64 | mem >= theta -> 65
        # Identical to `bucketize` but without a per-element binary search.
        # On CPU that matters at 1,000 neurons x batch 64 x T=10.
        scale = MEM_IN_RANGE / theta
        idx = (mem.detach().reshape(-1) * scale).floor_().add_(1).clamp_(
            0, MEM_BINS - 1).to(torch.int64)
        # Counting differs by device; both paths give identical results
        # (verified by checksum).
        #   CUDA  `bincount` pulls the input min/max to the host to size its
        #         output (aten::_local_scalar_dense), forcing a full GPU sync
        #         on every call. The bin count is fixed, so preallocate the
        #         output and accumulate into it to remove that round trip.
        #   CPU   there is no sync to speak of, and `bincount` beats scatter_add_
        if idx.is_cuda:
            out = torch.zeros(MEM_BINS, dtype=torch.int64, device=idx.device)
            out.scatter_add_(0, idx, torch.ones_like(idx))
        else:
            out = torch.bincount(idx, minlength=MEM_BINS)
        self._hist.setdefault(lid, []).append(out)

    # ---------------- into the unified format ----------------
    def membrane_arrays(self) -> dict:
        """`[T, 66]` integer arrays, pulled to the host once after measurement."""
        import numpy as np
        import torch
        return {k: np.ascontiguousarray(torch.stack(v).cpu().numpy(), dtype=np.int64)
                for k, v in self._hist.items() if v}

    def spike_arrays(self) -> dict:
        import numpy as np
        import torch
        return {k: np.ascontiguousarray(torch.stack(v).cpu().numpy(), dtype=np.uint8)
                for k, v in self._spikes.items() if v}

    def collect(self, trace: Trace, spike_dir: str = "", **kw: Any):
        from ..observations import (write_membrane_observation,
                                    write_spike_observation)
        from ..steady import derive as steady_derive

        for lid, a in self._acc.items():
            a["spike_driven"] = looks_binary(a.pop("probe", None))
            trace.layers.append(Layer(
                layer_id=lid, op_type=a.get("op_type", "?"), calls=a.get("calls", 0),
                in_loop=a.get("calls", 0) > 1,
                domain=STATE if a.get("is_snn") else DENSE,
                spike_driven=bool(a.get("spike_driven")),
                macs=a.get("macs", 0),
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
                    metrics=o.metrics, steady_state=steady_derive(arr)))
            for lid, counts in self.membrane_arrays().items():
                o = write_membrane_observation(
                    counts, out_dir=spike_dir, layer_id=lid,
                    run_tag=f"T{trace.timesteps}",
                    threshold=self._theta.get(lid), state_reset="subtract")
                trace.membranes.append(MembraneRef(
                    layer_id=o.layer_id, path=o.path, sha256=o.sha256,
                    timesteps=o.timesteps, bins=o.bins,
                    threshold=o.threshold, metrics=o.metrics))

        detail: dict[str, Any] = dict(
            layers=len(self._acc), regions=len(self._regions),
            registration="leaf-modules" if self.cover_leaves else "mac-and-neuron",
            region_timing="gpu-synchronized" if self.sync_regions
                          else "cpu-launch-window",
            labels="record_function" if self.emit_labels else "none",
            membrane=("off" if not self.membrane else
                      f"{len(self._hist)} layers" if self._hist else "nothing collected"))
        if self._mem_skip:
            detail["membrane_skipped"] = dict(self._mem_skip)
        return self.declare("captured", **detail)
