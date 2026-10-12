"""측정 대상 — 순수 DNN · 순수 SNN · 혼합. 하나의 프로파일러로 셋 다 측정한다."""

from __future__ import annotations


def mnist_mlp(**_):
    """순수 DNN 기준선. SNN과 레이어 구성을 맞춘 784-1000-10 MLP."""
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 1000), nn.ReLU(),
                         nn.Linear(1000, 10)), {}


def fcsnn(num_steps: int = 10, **_):
    """순수 SNN. snnTorch Tutorial-5 구성 (784-1000-10, Leaky beta=0.95)."""
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate

    class FcSnn(nn.Module):
        def __init__(self):
            super().__init__()
            g = surrogate.fast_sigmoid(slope=25)
            self.fc1 = nn.Linear(784, 1000)
            self.lif1 = snn.Leaky(beta=0.95, spike_grad=g, reset_delay=False)
            self.fc2 = nn.Linear(1000, 10)
            self.lif2 = snn.Leaky(beta=0.95, spike_grad=g, reset_delay=False)

        def forward(self, x, num_steps: int = num_steps):
            x = x.flatten(1)
            m1, m2 = self.lif1.init_leaky(), self.lif2.init_leaky()
            acc = 0
            for _ in range(num_steps):
                s1, m1 = self.lif1(self.fc1(x), m1)
                s2, m2 = self.lif2(self.fc2(s1), m2)
                acc = acc + s2
            return acc

    return FcSnn(), {"num_steps": num_steps}


def repeated_dnn_snn(num_steps: int = 10, **_):
    """혼합. 인수인계 문서 p128~129 / snn_profiling/snn_profile.py 구조."""
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            g = surrogate.fast_sigmoid()
            self.dnn1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(2))
            self.snn1_fc = nn.Linear(16 * 14 * 14, 64)
            self.lif1 = snn.Leaky(beta=0.9, spike_grad=g)
            self.dnn2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
            self.snn2_fc = nn.Linear(32, 10)
            self.lif2 = snn.Leaky(beta=0.9, spike_grad=g)
            self.num_steps = num_steps

        def forward(self, x):
            b = x.size(0)
            cur1 = self.snn1_fc(self.dnn1(x).view(b, -1))
            m1 = self.lif1.init_leaky()
            spk1 = []
            for _ in range(self.num_steps):
                s1, m1 = self.lif1(cur1, m1)
                spk1.append(s1)
            series = [self.snn2_fc(self.dnn2(s)) for s in spk1]
            m2 = self.lif2.init_leaky()
            acc = 0
            for s in series:
                s2, m2 = self.lif2(s, m2)
                acc = acc + s2
            return acc

    return Mixed(), {}


REGISTRY = {"mnist_mlp": mnist_mlp, "fcsnn": fcsnn, "repeated_dnn_snn": repeated_dnn_snn}


def build(name: str, **kw):
    if name not in REGISTRY:
        raise KeyError(f"알 수 없는 모델 '{name}'. known={sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
