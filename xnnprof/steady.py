"""Steady-state step: the timestep after which the spike pattern stops changing.

Mean firing rate cannot show this. Even at a constant rate, the set of firing neurons
can change every step, so the patterns themselves must be compared.

    d_t = mean( s_t != s_{t-1} ),        t = 1 .. T-1
    c_t = d_t / (2·r̄·(1-r̄))              normalized by the expectation for independent patterns
    reached step = min { t : c_u <= θ for all u >= t },   θ = 0.1

Read `c_t` in three bands: near 0 the pattern is fixed, near 1 it is no different from
drawing independently each step, and above 1 consecutive steps alternate.

The denominator `2·r̄·(1-r̄)` assumes a homogeneous Bernoulli model where every neuron
fires with the same probability r̄. When per-neuron probabilities differ, this exceeds
the true independence baseline: x(1-x) is concave, so mean(2·p_i·(1-p_i)) <= 2·r̄·(1-r̄).
The gap is large for layers with many dead or saturated neurons, and the verdict can
depend on the baseline model. The per-neuron-normalized value is therefore reported too.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_THRESHOLD = 0.1


def steady_state(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Compute from a `[T, batch, neurons...]` binary array.

    If r̄ is 0 or 1 the denominator is 0 and the value is undefined (a layer that never
    fires or always fires). In that case `reached_step` is omitted and a reason is
    recorded. A value of 0 and an undefined value are different.
    """
    import numpy as np

    a = np.asarray(array)
    if a.ndim < 2:
        raise ValueError(f"need at least 2 dims [time, ...]: {a.shape}")
    flat = a.reshape(len(a), -1)
    T = int(flat.shape[0])

    out: dict[str, Any] = {
        "timesteps": T,
        "threshold": float(threshold),
        "unit": "timestep index (0-based)",
    }
    if T < 2:
        out["reason"] = "fewer than 2 timesteps; nothing to compare consecutively"
        return out

    rbar = float(flat.mean())
    out["mean_rate"] = round(rbar, 6)
    denom = 2.0 * rbar * (1.0 - rbar)
    if denom <= 0.0:
        out["reason"] = ("firing rate is 0 or 1, so the denominator is 0; "
                         "pattern change cannot be compared to the independence expectation")
        return out

    d = (flat[1:] != flat[:-1]).mean(axis=1)
    out["d_t"] = [round(float(v), 6) for v in d]
    out["c_t_from_step"] = 1
    out["baseline"] = "homogeneous Bernoulli 2·r̄·(1-r̄)"

    def judge(c) -> Optional[int]:
        """Scan backwards for the last value above θ; the step after it is the reached step."""
        step = 1
        for i in range(len(c) - 1, -1, -1):
            if c[i] > threshold:
                step = i + 2         # c[i] is c_{i+1}, so take the next step
                break
        return step if step <= T - 1 else None

    c = d / denom
    out["c_t"] = [round(float(v), 6) for v in c]
    out["c_t_min"] = round(float(c.min()), 6)
    out["c_t_max"] = round(float(c.max()), 6)
    out["reached_step"] = judge(c)
    if out["reached_step"] is None:
        out["reason"] = f"c_t > {threshold} through the last step; not reached"

    # Per-neuron denominator, to see whether the verdict depends on the baseline model.
    if a.ndim >= 3:
        p = a.mean(axis=(0, 1))
        den_n = float((2.0 * p * (1.0 - p)).mean())
        out["baseline_per_neuron"] = round(den_n, 6)
        if den_n > 0.0:
            cn = d / den_n
            out["c_t_per_neuron_min"] = round(float(cn.min()), 6)
            out["c_t_per_neuron_max"] = round(float(cn.max()), 6)
            out["reached_step_per_neuron"] = judge(cn)
            out["baseline_sensitive"] = (
                out["reached_step_per_neuron"] != out["reached_step"])
    return out


def derive(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Do not let a computation failure turn into a measurement failure."""
    try:
        return steady_state(array, threshold)
    except Exception as e:                                   # noqa: BLE001
        return {"reason": f"{type(e).__name__}: {e}"}
