"""Steady-state step: the timestep after which the spike pattern stops changing.

Mean firing rate cannot show this. Even at a constant rate, the set of firing neurons
can change every step, so the patterns themselves must be compared.

    d_t = mean( s_t != s_{t-1} ),            t = 1 .. T-1
    c_t = d_t / mean_i 2·p_i·(1-p_i)          normalized by the expectation for independent patterns
    reached step = min { t : c_u <= θ for all u >= t },   θ = 0.1

p_i is neuron i's firing probability over all steps and samples. Read `c_t` in three
bands: near 0 the pattern is fixed, near 1 it is no different from each neuron drawing
independently at its own rate every step, and above 1 consecutive steps alternate.

The per-neuron baseline is the verdict (decided 2026-11-24). The homogeneous baseline
`2·r̄·(1-r̄)` is still reported as `*_homogeneous` for comparison with reports made before
then. It assumes every neuron fires with the same probability; since x(1-x) is concave
it is never smaller than the per-neuron baseline, so dead and saturated neurons inflate
it and make a fixed weight structure look like temporal convergence.

p_i pools samples, so a neuron that is always on for one input and always off for
another still counts as p=0.5. The per-neuron baseline therefore remains an upper bound.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_THRESHOLD = 0.1


def steady_state(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Compute from a `[T, batch, neurons...]` binary array.

    If the denominator is 0 the value is undefined (homogeneous: r̄ is 0 or 1; per-neuron:
    every neuron is dead or saturated). `reached_step` is then omitted and a reason is
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
    # neurons are the last axes; time and (if present) batch are pooled into p_i
    p = (a.reshape(T, a.shape[1], -1).mean(axis=(0, 1)) if a.ndim >= 3
         else flat.mean(axis=0))
    den = float((2.0 * p * (1.0 - p)).mean())
    den_h = 2.0 * rbar * (1.0 - rbar)
    out["baseline"] = "per-neuron Bernoulli mean_i 2·p_i·(1-p_i)"
    out["baseline_value"] = round(den, 6)
    out["baseline_homogeneous"] = round(den_h, 6)

    d = (flat[1:] != flat[:-1]).mean(axis=1)
    out["d_t"] = [round(float(v), 6) for v in d]
    out["c_t_from_step"] = 1

    def judge(c) -> Optional[int]:
        """Scan backwards for the last value above θ; the step after it is the reached step."""
        step = 1
        for i in range(len(c) - 1, -1, -1):
            if c[i] > threshold:
                step = i + 2         # c[i] is c_{i+1}, so take the next step
                break
        return step if step <= T - 1 else None

    def fill(denom: float, suffix: str) -> None:
        if denom <= 0.0:
            out["reason" + suffix] = (
                "every neuron fires never or always, so the denominator is 0; "
                "pattern change cannot be compared to the independence expectation"
                if not suffix else "firing rate is 0 or 1, so the denominator is 0")
            return
        c = d / denom
        out["c_t" + suffix] = [round(float(v), 6) for v in c]
        out["c_t_min" + suffix] = round(float(c.min()), 6)
        out["c_t_max" + suffix] = round(float(c.max()), 6)
        out["reached_step" + suffix] = judge(c)
        if out["reached_step" + suffix] is None:
            out["reason" + suffix] = f"c_t > {threshold} through the last step; not reached"

    fill(den, "")
    fill(den_h, "_homogeneous")
    if "reached_step" in out and "reached_step_homogeneous" in out:
        out["baseline_sensitive"] = out["reached_step"] != out["reached_step_homogeneous"]
    return out


def derive(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Do not let a computation failure turn into a measurement failure."""
    try:
        return steady_state(array, threshold)
    except Exception as e:                                   # noqa: BLE001
        return {"reason": f"{type(e).__name__}: {e}"}
