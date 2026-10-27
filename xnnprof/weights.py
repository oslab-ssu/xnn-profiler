"""가중치의 저장·적재와 지문.

장치를 바꿔 가며 같은 모델을 측정하려면 같은 가중치를 써야 한다. 같은 시드를
주어도 CPU와 CUDA는 커널이 다르므로 학습 결과가 갈리고, 그러면 스파이크 통계가
장치 때문에 달라진 것인지 모델이 달라서 달라진 것인지 구분되지 않는다.

지문(sha256)을 결과 파일에 남겨, 두 측정이 같은 가중치를 쓴 것임을 사후에
확인할 수 있게 한다.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional


def fingerprint(model) -> str:
    """state_dict의 내용으로 만든 지문. 키 이름과 값의 바이트를 모두 반영한다."""
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def load_or_train(model, name: str, weights_dir: str, device: str,
                  train: Optional[Callable[[], None]]) -> tuple[str, str]:
    """(출처, 지문)을 돌려준다.

    출처는 loaded · trained · random 중 하나다. `weights_dir`에 파일이 있으면
    학습을 건너뛰고 불러 쓴다. 없으면 학습한 뒤 저장한다.
    """
    import torch

    path = os.path.join(weights_dir, f"{name}.pt") if weights_dir else ""
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return "loaded", fingerprint(model)

    if train is None:
        return "random", fingerprint(model)

    train()
    if path:
        os.makedirs(weights_dir, exist_ok=True)
        torch.save(model.state_dict(), path)
    return "trained", fingerprint(model)
