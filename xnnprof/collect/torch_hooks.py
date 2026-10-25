"""PyTorch hook 수집기 — 통합 키 생성, 구간 측정, 라벨 개시, SNN 측정값 취득.

모델 코드를 수정하지 않는다. 모델 트리를 순회하여 레이어 타입을 기준으로 hook을
등록하므로 DNN·SNN·혼합이 같은 경로로 측정된다. 구간(region)이 통합 키
`(layer_id, timestep)`의 실행 1회를 가리키며, 다른 도구와의 상관 기준이 된다.

hook이 수행하는 기능은 두 가지다.

  통합 키와 구간의 생성   호출 직전·직후 시각으로 `[시작, 종료)`를 확보하고,
                          호출 계수로 타임스텝을 부여하며, 키 문자열
                          `fc1@t3`을 record_function·NVTX 라벨로 개시·종료한다
  SNN 측정값의 취득       MAC, 스파이크 배열, 상태 텐서 크기. 기존 도구가
                          관측하지 않는 값이다
"""

from __future__ import annotations

import time
from typing import Any

from ..schema import DENSE, Layer, Region, SpikeRef, STATE, Trace
from .base import Collector, REGION

INSTRUMENT_LABEL = "xnn::instrument"   # 계측 자신의 비용을 담는 라벨


def is_snn(module) -> bool:
    return type(module).__module__.startswith(("snntorch", "spikingjelly"))


def layer_id_of(name: str, module) -> str:
    """통합 키의 `layer_id`. 숫자 이름은 타입을 덧붙여 식별 가능하게 만든다.

    `nn.Sequential`의 자식은 `named_modules()`에서 `1`, `dnn1.0`처럼 인덱스로
    나온다. 그대로 쓰면 결과 파일에서 어느 연산인지 판독되지 않으므로
    `1:Linear`, `dnn1.0:Conv2d` 형태로 정규화한다.
    """
    head, _, last = name.rpartition(".")
    if last.isdigit():
        return f"{name}:{type(module).__name__}"
    return name


def targets(model, cover_leaves: bool = False):
    """계측 지점을 삽입할 모듈. 모델 구조를 가정하지 않고 트리를 순회한다.

    기본 규칙은 MAC이 발생하는 모듈(`nn.Linear`·`nn.Conv2d`)과 상태를 보유하고
    타임스텝마다 호출되는 SNN 뉴런 모듈이다. 컨테이너(`nn.Sequential` 등)는
    자식 구간을 포괄하는 상위 구간이 생겨 동일 시간이 이중 계상되므로 제외한다.

    `cover_leaves`를 켜면 자식이 없는 모든 모듈까지 넓힌다. MAC이 없는 모듈
    (`nn.ReLU`·`nn.MaxPool2d`·`nn.Flatten`)이 실제로 차지하는 시간을
    구간으로 확보할 수 있으나, 구간 수가 늘고 계측 비용도 함께 늘어난다.
    두 규칙의 귀속률 차이를 측정하기 위한 선택지다.
    """
    import torch.nn as nn
    for name, m in model.named_modules():
        if not name:                                  # 루트 모듈
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
    """스파이크 구동 레이어 판정 — 입력이 이진 텐서인지.

    측정 구간 안에서 호출하면 안 된다. 텐서 전체를 훑는 커널이 추가되어
    계측 비용이 측정 대상에 섞인다. 첫 호출의 입력을 참조만 보관해 두고
    측정이 끝난 뒤 `collect()`에서 판정한다.
    """
    import torch
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return False
    return bool(((x == 0) | (x == 1)).all())


class TorchHookCollector(Collector):
    name = "torch_hooks"
    attribution = REGION
    capabilities = ("layer", "timestep", "macs", "spikes",
                    "persistent_state", "domain", "label")

    def __init__(self, emit_labels: bool = True, emit_nvtx: bool = False,
                 cover_leaves: bool = False):
        self.emit_labels = emit_labels
        self.cover_leaves = cover_leaves
        self.emit_nvtx = emit_nvtx
        self._handles: list = []
        self._acc: dict[str, dict] = {}
        self._spikes: dict[str, list] = {}
        self._regions: list[Region] = []
        self._open_labels: dict[str, list] = {}
        self._t0 = 0.0

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as e:
            return False, f"torch import 실패: {e}"
        return True, "ok"

    # ---------------- hook 부착 ----------------
    def attach(self, model) -> None:
        self._t0 = time.perf_counter()
        for name, m in targets(model, self.cover_leaves):
            lid = layer_id_of(name, m)
            self._handles.append(m.register_forward_pre_hook(self._pre(lid)))
            self._handles.append(m.register_forward_hook(self._post(lid)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for stack in self._open_labels.values():          # 예외로 남은 라벨 정리
            while stack:
                self._close_label(stack.pop())
        self._open_labels.clear()

    # ---------------- 라벨 개시·종료 ----------------
    def _open_label(self, key: str):
        """통합 키 문자열을 도구 결과에 남긴다. 반환값은 종료에 쓸 핸들이다."""
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
            a = self._acc.setdefault(lid, {})
            step = a.get("calls", 0)
            self._open_labels.setdefault(lid, []).append(
                self._open_label(f"{lid}@t{step}"))
            a["open"] = time.perf_counter()
        return pre

    def _post(self, lid: str):
        def post(mod, inputs, output):
            import torch
            now = time.perf_counter()
            # 라벨을 먼저 닫는다. 아래의 텐서 검사·복사는 계측 자신의 비용이므로
            # 라벨 안에 남으면 그 레이어의 연산 시간으로 잘못 귀속된다.
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
                    a["probe"] = x.detach()   # 판정은 측정 종료 후 (looks_binary)
            if opened is not None:
                dur = (now - opened) * 1e6
                a["self_time_us"] += dur
                self._regions.append(Region(
                    region_id=len(self._regions), layer_id=lid, timestep=step,
                    ts_us=(opened - self._t0) * 1e6, dur_us=dur,
                    domain=STATE if a["is_snn"] else DENSE))

            if a["is_snn"]:
                spk = output[0] if isinstance(output, tuple) else output
                mem = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if torch.is_tensor(spk):
                    self._spikes.setdefault(lid, []).append(
                        spk.detach().to(torch.uint8).cpu())
                if mem is not None and torch.is_tensor(mem):
                    a["state_bytes"] = int(mem.numel() * mem.element_size())
            if inst is not None:
                self._close_label(inst)
        return post

    # ---------------- 통일 포맷으로 ----------------
    def spike_arrays(self) -> dict:
        import numpy as np
        import torch
        return {k: np.ascontiguousarray(torch.stack(v).numpy(), dtype=np.uint8)
                for k, v in self._spikes.items() if v}

    def collect(self, trace: Trace, spike_dir: str = "", **kw: Any):
        from ..observations import write_spike_observation

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
                    metrics=o.metrics))
        return self.declare("captured", layers=len(self._acc),
                            regions=len(self._regions),
                            registration="leaf-modules" if self.cover_leaves
                                         else "mac-and-neuron",
                            labels="record_function" if self.emit_labels else "none")
