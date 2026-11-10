"""정상상태 도달 스텝 — 스파이크 패턴이 더 이상 바뀌지 않는 타임스텝.

발화율 평균으로는 알 수 없는 값이다. 발화율이 일정해도 발화하는 뉴런의 조합은
매 스텝 달라질 수 있으므로, 패턴 자체를 직접 비교해야 한다.

    d_t = mean( s_t != s_{t-1} ),        t = 1 .. T-1
    c_t = d_t / (2·r̄·(1-r̄))              두 패턴이 독립일 때의 기댓값으로 정규화
    도달 스텝 = min { t : 모든 u >= t 에 대하여 c_u <= θ },   θ = 0.1

`c_t`는 세 구간으로 읽는다. 0 부근은 패턴이 고정된 것, 1 부근은 매 스텝 독립으로
뽑은 것과 다름이 없는 것, 1 초과는 연속한 스텝이 서로 반대로 가는 교대 패턴이다.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_THRESHOLD = 0.1


def steady_state(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """`[T, batch, neurons...]` 이진 배열에서 산출한다.

    r̄이 0 또는 1이면 정규화 분모가 0이 되어 정의되지 않는다. 한 번도 발화하지
    않은 레이어와 항상 발화한 레이어가 그렇고, 그때는 `reached_step`을 두지 않고
    사유를 남긴다. 값이 0인 것과 정의되지 않는 것은 다르다.
    """
    import numpy as np

    a = np.asarray(array)
    if a.ndim < 2:
        raise ValueError(f"[time, ...] 2차원 이상 필요: {a.shape}")
    flat = a.reshape(len(a), -1)
    T = int(flat.shape[0])

    out: dict[str, Any] = {
        "timesteps": T,
        "threshold": float(threshold),
        "unit": "타임스텝 인덱스 (0부터)",
    }
    if T < 2:
        out["reason"] = "타임스텝이 2개 미만이라 연속 비교 대상이 없음"
        return out

    rbar = float(flat.mean())
    out["mean_rate"] = round(rbar, 6)
    denom = 2.0 * rbar * (1.0 - rbar)
    if denom <= 0.0:
        out["reason"] = ("발화율이 0 또는 1이라 정규화 분모가 0 — "
                         "패턴 변화량을 독립 기댓값과 비교할 수 없음")
        return out

    d = (flat[1:] != flat[:-1]).mean(axis=1)
    c = d / denom
    out["c_t"] = [round(float(v), 6) for v in c]
    out["c_t_from_step"] = 1
    out["c_t_min"] = round(float(c.min()), 6)
    out["c_t_max"] = round(float(c.max()), 6)

    # 뒤에서부터 θ를 넘는 지점을 찾는다. 그 다음 스텝이 도달 스텝이다.
    reached: Optional[int] = 1
    for i in range(len(c) - 1, -1, -1):
        if c[i] > threshold:
            reached = i + 2          # c[i]는 c_{i+1} 이므로 그 다음 스텝
            break
    out["reached_step"] = reached if reached <= T - 1 else None
    if out["reached_step"] is None:
        out["reason"] = f"마지막 스텝까지 c_t > {threshold} — 미도달"
    return out


def derive(array, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """산출 실패를 측정 실패로 번지게 하지 않는다."""
    try:
        return steady_state(array, threshold)
    except Exception as e:                                   # noqa: BLE001
        return {"reason": f"{type(e).__name__}: {e}"}
