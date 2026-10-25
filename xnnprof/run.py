"""측정 1회 실행 — hook 부착, forward, 통일 포맷 기록."""

from __future__ import annotations

import os
import platform
import time
from typing import Any, Optional

from .collect.base import Collector
from .collect.stubs import ALL as STUB_COLLECTORS
from .collect.torch_hooks import TorchHookCollector
from .collect.torch_prof import TorchProfilerCollector
from .emit import trace_json
from .metrics import snn_metrics
from .synops import annotate as annotate_synops
from .schema import Trace


def env_snapshot(device: str = "cpu") -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "device": device}
    for mod in ("torch", "snntorch", "numpy"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            env[mod] = None
    if device.startswith("cuda"):
        try:
            import torch
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
            env["driver"] = getattr(torch.cuda, "driver_version", lambda: None)()
        except Exception:
            pass
    return env


def profile(model, inputs, *, model_id: str, timesteps: Optional[int] = None,
            batch: Optional[int] = None, out_dir: str = "out",
            with_torch_profiler: bool = True, device: str = "cpu",
            emit_nvtx: bool = False, cover_leaves: bool = False,
            extra: tuple[Collector, ...] = (), **forward_kw: Any) -> Trace:
    """측정 1회.

    GPU에서 hook이 재는 시간은 커널 실행 시간이 아니라 CPU 쪽 실행 요청 구간이다.
    비동기 실행이므로 둘은 일치하지 않는다. 구간은 NVTX 라벨로 그대로 내보내고,
    커널을 그 구간에 연결하는 일은 Nsight Systems가 맡는다.
    """
    import torch

    os.makedirs(out_dir, exist_ok=True)
    cuda = device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없다. nvidia-smi와 torch.version.cuda를 확인한다")

    trace = Trace(
        run_id=f"{model_id}-{device}-T{timesteps}-{int(time.time())}",
        model_id=model_id, timesteps=timesteps, batch=batch,
        env=env_snapshot(device))

    model = model.to(device)
    hooks = TorchHookCollector(emit_nvtx=emit_nvtx or cuda,
                               cover_leaves=cover_leaves)
    ok, why = hooks.available()
    if not ok:
        trace.sources.append(hooks.declare("unavailable", why))
        return trace

    args = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    args = tuple(a.to(device) if torch.is_tensor(a) else a for a in args)
    model.eval()
    with torch.no_grad():
        model(*args, **forward_kw)          # 워밍업: 지연 초기화·커널 선택 제외
    if cuda:
        torch.cuda.synchronize()

    prof = None
    hooks.attach(model)
    if with_torch_profiler:
        from torch.profiler import ProfilerActivity, profile as tprofile
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with tprofile(activities=acts, profile_memory=True,
                      record_shapes=False) as prof:
            with torch.no_grad():
                model(*args, **forward_kw)
            if cuda:
                torch.cuda.synchronize()
    else:
        with torch.no_grad():
            model(*args, **forward_kw)
        if cuda:
            torch.cuda.synchronize()
    hooks.detach()

    trace.sources.append(hooks.collect(trace, spike_dir=os.path.join(out_dir, "spikes")))
    trace.sources.append(TorchProfilerCollector(prof).collect(trace))
    for cls in STUB_COLLECTORS:
        trace.sources.append(cls().collect(trace))
    for c in extra:
        trace.sources.append(c.collect(trace))

    annotate_synops(trace)
    trace.metrics = snn_metrics(trace)
    trace_json.write(trace, os.path.join(out_dir, "xnn-trace.json"))
    return trace
