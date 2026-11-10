"""관측 산출물 — 집계값이 아니라 원본을 파일로 내보낸다.

  스파이크   `[T, batch, neurons]` 이진 배열을 uint8 `.npy`로
  막전위     `[T, 66]` 히스토그램 칸 개수를 int64 `.npy`로

막전위는 원 텐서를 쌓지 않는다. `[batch, neurons]` float32는 `fcsnn`의 `lif1`
기준 스텝당 256 KB, T=10이면 2.56 MB다. 재려는 것 자체가 메모리인데 계측이
그만큼 잡으면 측정 대상이 바뀌므로, 받자마자 고정 칸 히스토그램으로 환산한다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

OBSERVATION_SCHEMA = "ssu.snn.observation"
OBSERVATION_VERSION = 1

# 막전위 히스토그램의 칸 구성. `torch.bucketize(..., right=True)` 결과와 일치한다.
#   0        mem < 0            리셋이 빼기 방식이라 음수로 내려갈 수 있다
#   1 ~ 64   [0, θ) 를 64등분   칸 하나의 폭은 θ/64
#   65       mem >= θ           이 스텝에 발화한 몫
MEM_BINS = 66
MEM_IN_RANGE = 64
MEM_NEAR_BINS = (57, 64)        # [0, θ) 의 상위 12.5% — 다음 스텝에 발화할 후보


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
        "first_spike_step_unit": "발화한 (sample, neuron) 쌍 평균",
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
    bin_layout: str = ("0: mem < 0 | 1~64: [0, θ) 64등분 | 65: mem >= θ")
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
    """`[T, 66]` 칸 개수에서 산출한다. 원 텐서를 다시 읽지 않는다.

    `near_threshold_fraction`은 `[0, θ)`의 상위 8칸이 차지하는 비율이다. 임계
    바로 아래에 몰린 뉴런의 몫이며, 타임스텝을 줄이거나 양자화할 때 출력이
    얼마나 흔들릴지를 가늠하는 값이다.

    히스토그램의 대상은 뉴런 모듈이 반환한 막전위다. 리셋과 감쇠가 이미 적용된
    값이므로 `at_or_above_threshold_fraction`을 그 스텝의 발화율로 읽으면 안
    된다. 발화율은 스파이크 배열에서 따로 산출한다.
    """
    import numpy as np

    a = np.asarray(counts)
    if a.ndim != 2 or a.shape[1] != MEM_BINS:
        raise ValueError(f"[time, {MEM_BINS}] 필요: {a.shape}")
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
