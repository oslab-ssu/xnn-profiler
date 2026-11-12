"""Steady-state convergence timestep -- the timestep past which the spike
pattern no longer changes.

The mean firing rate cannot tell you this. Even at a constant firing rate the
*set* of neurons that fire may differ at every timestep, so the patterns
themselves have to be compared directly.

    d_t = mean( s_t != s_{t-1} ),        t = 1 .. T-1
    c_t = d_t / (2*r_bar*(1-r_bar))      normalized by the expectation for
                                         two independent patterns
    converged step = min { t : c_u <= theta for all u >= t },   theta = 0.1

Read `c_t` in three bands. Near 0 the pattern is fixed; near 1 it is
indistinguishable from drawing each step independently; above 1 consecutive
steps move in opposite directions, i.e. an alternating pattern.

The normalizing denominator `2*r_bar*(1-r_bar)` assumes a homogeneous
Bernoulli model in which every neuron fires with the same probability r_bar.
When per-neuron firing probabilities differ, that value exceeds the true
independence baseline, because x(1-x) is concave and therefore
mean(2*p_i*(1-p_i)) <= 2*r_bar*(1-r_bar). The gap is large in layers with many
dead or saturated neurons, and the verdict can flip depending on which
baseline is used. The per-neuron normalization is therefore reported as well.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_THRESHOLD = 0.1


def steady_state(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Derive the metric from a `[T, batch, neurons...]` binary array.

    When r_bar is 0 or 1 the normalizing denominator is 0 and the metric is
    undefined. That happens for a layer that never fires and for one that
    always fires; in those cases no `reached_step` is reported and the reason
    is recorded instead. A value of 0 and an undefined value are not the same
    thing.
    """
    import numpy as np

    a = np.asarray(array)
    if a.ndim < 2:
        raise ValueError(f"expected [time, ...] with 2 or more dims: {a.shape}")
    flat = a.reshape(len(a), -1)
    T = int(flat.shape[0])

    out: dict[str, Any] = {
        "timesteps": T,
        "threshold": float(threshold),
        "unit": "timestep index (0-based)",
    }
    if T < 2:
        out["reason"] = "fewer than 2 timesteps, nothing to compare"
        return out

    rbar = float(flat.mean())
    out["mean_rate"] = round(rbar, 6)
    denom = 2.0 * rbar * (1.0 - rbar)
    if denom <= 0.0:
        out["reason"] = ("firing rate is 0 or 1, so the normalizing "
                         "denominator is 0 -- the pattern change cannot be "
                         "compared against the independent expectation")
        return out

    d = (flat[1:] != flat[:-1]).mean(axis=1)
    out["d_t"] = [round(float(v), 6) for v in d]
    out["c_t_from_step"] = 1
    out["baseline"] = "homogeneous Bernoulli 2*r_bar*(1-r_bar)"

    def judge(c) -> Optional[int]:
        """Scan backwards for the last point above theta.

        The step after it is the convergence step.
        """
        step = 1
        for i in range(len(c) - 1, -1, -1):
            if c[i] > threshold:
                step = i + 2         # c[i] is c_{i+1}, so the next step
                break
        return step if step <= T - 1 else None

    c = d / denom
    out["c_t"] = [round(float(v), 6) for v in c]
    out["c_t_min"] = round(float(c.min()), 6)
    out["c_t_max"] = round(float(c.max()), 6)
    out["reached_step"] = judge(c)
    if out["reached_step"] is None:
        out["reason"] = f"c_t > {threshold} through the last step -- not reached"

    # Denominator based on per-neuron firing rates, to show whether the
    # verdict depends on the choice of baseline model.
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
    """Never let a derivation failure escalate into a measurement failure."""
    try:
        return steady_state(array, threshold)
    except Exception as e:                                   # noqa: BLE001
        return {"reason": f"{type(e).__name__}: {e}"}
