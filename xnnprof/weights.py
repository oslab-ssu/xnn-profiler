"""Saving, loading and fingerprinting weights.

Measuring the same model across devices requires the same weights. Even with
the same seed, CPU and CUDA use different kernels, so training diverges -- and
then it is impossible to tell whether the spike statistics differ because of
the device or because the model differs.

The sha256 fingerprint is recorded in the result file so that two measurements
can be confirmed after the fact to have used the same weights.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional


def fingerprint(model) -> str:
    """Fingerprint built from the state_dict contents.

    Covers both the key names and the bytes of the values.
    """
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
    """Return `(origin, fingerprint)`.

    The origin is one of loaded, trained or random. If a file exists in
    `weights_dir`, training is skipped and the file is loaded; otherwise the
    model is trained and then saved.
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
