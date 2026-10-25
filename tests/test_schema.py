"""torch 없이 돌아가는 계약 테스트."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xnnprof.collect.torch_hooks import layer_id_of
from xnnprof.collect.torch_hooks import INSTRUMENT_LABEL
from xnnprof.collect.torch_prof import INSTRUMENT, owner_of
from xnnprof.emit import trace_json
from xnnprof.metrics import snn_metrics
from xnnprof.schema import DENSE, Layer, Region, SpikeRef, STATE, Trace


def sample() -> Trace:
    t = Trace(run_id="r", model_id="m", timesteps=4, batch=2)
    t.layers = [
        Layer("fc1", "Linear", calls=1, macs=1000),
        Layer("fc2", "Linear", calls=4, in_loop=True, spike_driven=True, macs=400),
        Layer("lif1", "Leaky", calls=4, in_loop=True, domain=STATE, state_bytes=64),
    ]
    t.regions = [Region(i, "fc2", i, ts_us=i * 10.0, dur_us=5.0) for i in range(4)]
    t.regions += [Region(4 + i, "lif1", i, ts_us=i * 10.0 + 5.0, dur_us=3.0,
                         domain=STATE) for i in range(4)]
    t.spikes = [SpikeRef("lif1", "spikes/lif1.npy", timesteps=4, batch=2, neurons=8,
                         metrics={"mean_rate": 0.25,
                                  "per_timestep_rate": [0.0, 0.0, 0.5, 0.5]})]
    t.metrics = snn_metrics(t)
    return t


def test_snn_metrics():
    m = sample().metrics
    assert m["executed_macs"] == 1400
    assert m["state_bytes"] == 64
    ol = m["output_latency"]  # noqa
    # 실행 순서상 마지막 스파이크 레이어가 출력 레이어
    assert ol["layer_id"] == "lif1"
    # per_timestep_rate가 0,0,0.5,0.5 -> 출력은 스텝 2부터 발생한다
    assert ol["first_spike_step"] == 2 and ol["timesteps"] == 4
    assert "synops" not in m, "산출되지 않은 항목은 키를 두지 않는다"

    dnn = Trace(run_id="r", model_id="dnn", timesteps=1)
    dnn.layers = [Layer("fc1", "Linear", calls=1, macs=10)]
    dnn.regions = [Region(0, "fc1", 0, ts_us=0.0, dur_us=1.0)]
    d = snn_metrics(dnn)
    assert "output_latency" not in d and "state_bytes" not in d, \
        "순수 DNN에는 SNN 항목이 정의되지 않는다"
    print("  신규 측정 항목 OK")


class FakeEvent:
    """FunctionEvent의 name·cpu_parent·self_cpu_time_total만 흉내낸다."""

    def __init__(self, name, parent=None, self_us=0.0):
        self.name = name
        self.cpu_parent = parent
        self.self_cpu_time_total = self_us


def test_label_join():
    label = FakeEvent("fc1@t3")
    linear = FakeEvent("aten::linear", label, 10.0)
    addmm = FakeEvent("aten::addmm", linear, 90.0)
    loose = FakeEvent("aten::empty", None, 7.0)
    assert owner_of(addmm) == ("fc1", 3), "손자 이벤트도 라벨까지 거슬러 올라간다"
    assert owner_of(linear) == ("fc1", 3)
    assert owner_of(loose) is None, "라벨 밖은 귀속되지 않는다"
    # 정규화된 layer_id도 그대로 복원된다
    assert owner_of(FakeEvent("aten::addmm", FakeEvent("1:Linear@t0"))) \
        == ("1:Linear", 0)
    # 계측 자신의 연산은 가까운 라벨이 계측 라벨이므로 구간에 섞이지 않는다
    inst = FakeEvent(INSTRUMENT_LABEL, FakeEvent("fc1@t0"))
    assert owner_of(FakeEvent("aten::all", inst, 5.0)) == INSTRUMENT
    print("  라벨 기반 통합 OK")


class Linear:
    pass


def test_layer_id_normalization():
    m = Linear()
    assert layer_id_of("fc1", m) == "fc1"
    assert layer_id_of("1", m) == "1:Linear", "nn.Sequential 자식의 숫자 이름"
    assert layer_id_of("dnn1.0", m) == "dnn1.0:Linear"
    assert layer_id_of("block.sub1", m) == "block.sub1"
    print("  layer_id 정규화 OK")


def test_trace_event_format():
    t = sample()
    t.regions[0].ops = {"aten::addmm": 90.0, "aten::linear": 10.0}
    d = trace_json.build(t)
    ev = d["traceEvents"]
    assert d["displayTimeUnit"] == "ms"
    x = [e for e in ev if e["ph"] == "X"]
    assert len(x) == 8
    for e in x:
        assert all(k in e for k in ("name", "ts", "dur", "pid", "tid", "cat", "args"))
    fc2 = [e for e in x if e["name"].startswith("fc2@")]
    assert len(fc2) == 4 and all(e["args"]["macs_total"] == 400 for e in fc2)
    assert all(e["args"]["spike_driven"] is True for e in fc2)
    with_ops = [e for e in x if "ops_self_time_us" in e["args"]]
    assert len(with_ops) == 1 and with_ops[0]["args"]["ops_self_time_us"]["aten::addmm"] == 90.0
    counters = [e for e in ev if e["ph"] == "C"]
    assert len(counters) == 4 and counters[0]["args"]["rate"] == 0.0
    names = {e["args"]["name"] for e in ev if e.get("name") == "thread_name"}
    assert any("spike-driven" in n for n in names)
    assert d["xnn"]["schema"] == "xnn.trace"
    json.dumps(d, allow_nan=False)
    print(f"  Chrome Trace Event OK (X {len(x)}, C {len(counters)})")


def test_roundtrip():
    t = sample()
    t.regions[0].ops = {"aten::addmm": 90.0}
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.json")
        t.save(p)
        back = Trace.load(p)
    assert back.executed_macs() == t.executed_macs()
    assert [l.layer_id for l in back.layers] == [l.layer_id for l in t.layers]
    assert back.regions[2].timestep == 2
    assert back.regions[0].ops == {"aten::addmm": 90.0}
    assert back.region("fc2", 2) is not None
    assert back.spikes[0].metrics["mean_rate"] == 0.25
    assert back.metrics["output_latency"]["first_spike_step"] == 2
    print("  무손실 왕복 OK")


def test_sources_declare_limits():
    from xnnprof.collect.stubs import ALL
    for cls in ALL:
        s = cls().collect(Trace(run_id="r", model_id="m"))
        assert s.status == "unavailable" and s.reason
        assert s.attribution in ("region", "process-global", "static-name")
    perf = next(c for c in ALL if c.name == "perf")
    assert perf.attribution == "process-global", "perf는 레이어 귀속이 불가하다"
    print("  수집기 귀속 한계 선언 OK")


if __name__ == "__main__":
    for fn in (test_snn_metrics, test_label_join, test_layer_id_normalization,
               test_trace_event_format, test_roundtrip, test_sources_declare_limits):
        fn()
    print("\n계약 테스트 전부 통과")
