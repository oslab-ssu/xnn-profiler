"""Observation files: raw data is exported, not aggregates.

  spikes     `[T, batch, neurons]` binary array as uint8 `.npy`
  membrane   `[T, 66]` histogram bin counts as int64 `.npy`

Raw membrane tensors are not accumulated. `[batch, neurons]` float32 for `fcsnn`'s
`lif1` is 256 KB per step, 2.56 MB at T=10. Memory is one of the things being
measured, and instrumentation holding that much would change the subject, so each
tensor is converted to a fixed-bin histogram as soon as it arrives.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

OBSERVATION_SCHEMA = "ssu.snn.observation"
OBSERVATION_VERSION = 1

# Membrane histogram bin layout. Matches `torch.bucketize(..., right=True)`.
#   0        mem < 0             subtractive reset can push values below zero
#   1 ~ 64   [0, θ) in 64 bins   each bin is θ/64 wide
#   65       mem >= θ            the share that fired at this step
MEM_BINS = 66
MEM_IN_RANGE = 64
MEM_NEAR_BINS = (57, 64)        # top 12.5% of [0, θ): candidates to fire next step


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
    """Formulas match the spikes kind in RTST checks.py. Changing them breaks cross-checking after handover."""
    import numpy as np

    a = np.asarray(array)
    if a.ndim < 3:
        raise ValueError(f"need at least 3 dims [time, batch, neurons...]: {a.shape}")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"only 0/1 allowed (e.g. {float(a[(a != 0) & (a != 1)].ravel()[0])})")

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
        "first_spike_step_unit": "mean over (sample, neuron) pairs that fired",
        "never_fired_sample_neuron_fraction": float((first < 0).mean()),
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


@dataclass
class MembraneObservation:
    layer_id: str
    timesteps: int
    bins: int = MEM_BINS
    path: str = ""
    sha256: str = ""
    quantity: str = "membrane_histogram"
    axes: tuple[str, ...] = ("time", "bin")
    threshold: Optional[float] = None
    bin_layout: str = ("0: mem < 0 | 1~64: [0, θ) in 64 bins | 65: mem >= θ")
    near_bins: tuple[int, int] = MEM_NEAR_BINS
    state_reset: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axes"] = list(self.axes)
        d["near_bins"] = list(self.near_bins)
        d["schema"] = OBSERVATION_SCHEMA
        d["version"] = OBSERVATION_VERSION
        return d


def derive_membrane_metrics(counts) -> dict[str, Any]:
    """Compute from `[T, 66]` bin counts without rereading raw tensors.

    `near_threshold_fraction` is the share of the top 8 bins of `[0, θ)`: neurons
    sitting just below threshold. It indicates how much the output may wobble when
    timesteps are reduced or values are quantized.

    The histogram covers the membrane potential returned by the neuron module, with
    reset and decay already applied, so `at_or_above_threshold_fraction` must not be
    read as that step's firing rate. Firing rate is computed from the spike array.
    """
    import numpy as np

    a = np.asarray(counts)
    if a.ndim != 2 or a.shape[1] != MEM_BINS:
        raise ValueError(f"need [time, {MEM_BINS}]: {a.shape}")
    total = a.sum(axis=1).astype(float)
    total[total == 0] = np.nan
    lo, hi = MEM_NEAR_BINS
    near = a[:, lo:hi + 1].sum(axis=1) / total
    fired = a[:, MEM_BINS - 1] / total
    below = a[:, 0] / total

    def ser(x):
        return [None if np.isnan(v) else round(float(v), 6) for v in x]

    return {
        "timesteps": int(a.shape[0]),
        "bins": MEM_BINS,
        "near_bins": list(MEM_NEAR_BINS),
        "values_per_step": int(np.nanmax(a.sum(axis=1))) if a.size else 0,
        "near_threshold_fraction": ser(near),
        "at_or_above_threshold_fraction": ser(fired),
        "below_zero_fraction": ser(below),
        "near_threshold_fraction_mean": (None if np.all(np.isnan(near))
                                         else round(float(np.nanmean(near)), 6)),
    }


def write_membrane_observation(
    counts, *, out_dir: str, layer_id: str, run_tag: str = "",
    threshold: Optional[float] = None, state_reset: str = "",
    notes: Optional[list[str]] = None,
) -> MembraneObservation:
    import numpy as np

    a = np.asarray(counts)
    metrics = derive_membrane_metrics(a)
    a = np.ascontiguousarray(a, dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    stem = (f"membrane_{run_tag}_{layer_id}" if run_tag else f"membrane_{layer_id}")
    path = os.path.join(out_dir, stem.replace("/", "_").replace(" ", "_") + ".npy")
    np.save(path, a, allow_pickle=False)

    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    obs = MembraneObservation(
        layer_id=layer_id, timesteps=metrics["timesteps"], path=path,
        sha256=digest, threshold=threshold, state_reset=state_reset,
        metrics=metrics, notes=list(notes or []))
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
