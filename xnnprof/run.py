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
from .metrics import headroom
from .schema import Trace


def env_snapshot() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "snntorch", "numpy"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            env[mod] = None
    return env


def profile(model, inputs, *, model_id: str, timesteps: Optional[int] = None,
            batch: Optional[int] = None, out_dir: str = "out",
            with_torch_profiler: bool = True, extra: tuple[Collector, ...] = (),
            **forward_kw: Any) -> Trace:
    import torch

    os.makedirs(out_dir, exist_ok=True)
    trace = Trace(
        run_id=f"{model_id}-T{timesteps}-{int(time.time())}",
        model_id=model_id, timesteps=timesteps, batch=batch, env=env_snapshot())

    hooks = TorchHookCollector()
    ok, why = hooks.available()
    if not ok:
        trace.sources.append(hooks.declare("unavailable", why))
        return trace

    args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
    model.eval()
    with torch.no_grad():
        model(*args, **forward_kw)          # 워밍업: 지연 초기화 제외

    prof = None
    hooks.attach(model)
    if with_torch_profiler:
        from torch.profiler import ProfilerActivity, profile as tprofile
        with tprofile(activities=[ProfilerActivity.CPU], profile_memory=True,
                      record_shapes=False) as prof:
            with torch.no_grad():
                model(*args, **forward_kw)
    else:
        with torch.no_grad():
            model(*args, **forward_kw)
    hooks.detach()

    trace.sources.append(hooks.collect(trace, spike_dir=os.path.join(out_dir, "spikes")))
    trace.sources.append(TorchProfilerCollector(prof).collect(trace))
    for cls in STUB_COLLECTORS:
        trace.sources.append(cls().collect(trace))
    for c in extra:
        trace.sources.append(c.collect(trace))

    trace.headroom = headroom(trace)
    trace_json.write(trace, os.path.join(out_dir, "xnn-trace.json"))
    return trace
