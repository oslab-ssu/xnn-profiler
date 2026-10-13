"""스파이크 관측 산출물. [T, batch, neurons] 이진 배열을 uint8 .npy로 내보낸다."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

OBSERVATION_SCHEMA = "ssu.snn.observation"
OBSERVATION_VERSION = 1


@dataclass
class SpikeObservation:
    layer_id: str
    timesteps: int
    batch: int
    neurons: int
    path: str = ""
    sha256: str = ""
    quantity: str = "spikes"
    axes: tuple[str, ...] = ("time", "batch", "neurons")
    dt: Optional[float] = None
    dt_units: str = "abstract-step"
    state_reset: str = ""
    encoding: str = ""
    threshold: Optional[float] = None
    beta: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axes"] = list(self.axes)
        d["schema"] = OBSERVATION_SCHEMA
        d["version"] = OBSERVATION_VERSION
        return d


def derive_spike_metrics(array) -> dict[str, Any]:
    """RTST checks.py의 spikes kind와 식을 일치시킨다. 바꾸면 인계 후 대조 불가."""
    import numpy as np

    a = np.asarray(array)
    if a.ndim < 3:
        raise ValueError(f"[time, batch, neurons...] 3차원 이상 필요: {a.shape}")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"0/1만 허용 (예: {float(a[(a != 0) & (a != 1)].ravel()[0])})")

    rates = a.mean(axis=(0, 1))
    first = np.where(a.any(axis=0), a.argmax(axis=0), -1)
    fired = first[first >= 0]
    per_neuron = a.sum(axis=(0, 1)).reshape(-1).astype(int)
    top = np.argsort(-per_neuron)[:10]

    return {
        "timesteps": int(a.shape[0]),
        "batch": int(a.shape[1]),
        "neurons": int(rates.size),
        "spikes": int(a.sum()),
        "mean_rate": float(a.mean()),
        "per_timestep_rate": a.reshape(len(a), -1).mean(axis=1).tolist(),
        "dead_fraction": float((rates == 0).mean()),
        "saturated_fraction": float((rates == 1).mean()),
        "first_spike_step_mean": float(fired.mean()) if fired.size else None,
        "never_fired_fraction": float((first < 0).mean()),
        "spikes_per_neuron": (per_neuron.tolist() if per_neuron.size <= 4096
                              else None),
        "spikes_per_neuron_truncated": bool(per_neuron.size > 4096),
        "top_neurons": [{"neuron": int(i), "spikes": int(per_neuron[i])} for i in top],
    }


def write_spike_observation(
    array, *, out_dir: str, layer_id: str, run_tag: str = "",
    dt: Optional[float] = None, dt_units: str = "abstract-step",
    state_reset: str = "", encoding: str = "",
    threshold: Optional[float] = None, beta: Optional[float] = None,
    notes: Optional[list[str]] = None,
) -> SpikeObservation:
    import numpy as np

    a = np.asarray(array)
    metrics = derive_spike_metrics(a)
    a = a.astype(np.uint8, copy=False)

    os.makedirs(out_dir, exist_ok=True)
    stem = (f"spikes_{run_tag}_{layer_id}" if run_tag else f"spikes_{layer_id}")
    path = os.path.join(out_dir, stem.replace("/", "_").replace(" ", "_") + ".npy")
    np.save(path, a, allow_pickle=False)

    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    obs = SpikeObservation(
        layer_id=layer_id, timesteps=metrics["timesteps"],
        batch=metrics["batch"], neurons=metrics["neurons"],
        path=path, sha256=digest, dt=dt, dt_units=dt_units,
        state_reset=state_reset, encoding=encoding,
        threshold=threshold, beta=beta, metrics=metrics, notes=list(notes or []),
    )
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(obs.to_dict(), f, ensure_ascii=False, indent=2)
    return obs


def artifact_entries(observations: list[SpikeObservation]) -> dict[str, dict]:
    return {
        f"spikes::{o.layer_id}": {"path": o.path, "sha256": o.sha256,
                                  "kind": OBSERVATION_SCHEMA,
                                  "version": OBSERVATION_VERSION}
        for o in observations
    }
