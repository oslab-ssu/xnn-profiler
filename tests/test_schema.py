"""torch 없이 돌아가는 계약 테스트."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xnnprof.emit import trace_json
from xnnprof.metrics import headroom
from xnnprof.schema import DENSE, Layer, Region, SpikeRef, STATE, Trace


def sample() -> Trace:
    t = Trace(run_id="r", model_id="m", timesteps=4, batch=2)
    t.layers = [
        Layer("fc1", "Linear", calls=1, macs=1000, useful_macs=1000,
              input_nonzero_fraction=1.0),
        Layer("fc2", "Linear", calls=4, in_loop=True, spike_driven=True,
              macs=400, useful_macs=100, input_nonzero_fraction=0.25),
        Layer("lif1", "Leaky", calls=4, in_loop=True, domain=STATE, state_bytes=64),
    ]
    t.regions = [Region(i, "fc2", i, ts_us=i * 10.0, dur_us=5.0) for i in range(4)]
    t.spikes = [SpikeRef("lif1", "spikes/lif1.npy", timesteps=4, batch=2, neurons=8,
                         metrics={"mean_rate": 0.25,
                                  "per_timestep_rate": [0.0, 0.0, 0.5, 0.5]})]
    t.headroom = headroom(t)
    return t


def test_headroom():
    h = sample().headroom
    assert h["executed_macs"] == 1400
    assert h["useful_macs"] == 1100
    assert h["sparse_kernel"]["skippable_macs"] == 300
    assert h["sparse_kernel"]["from_spike_driven_layers"] == 300
    assert h["time_loop"]["looped_macs"] == 400
    # per_timestep_rate가 0,0,0.5,0.5 -> 출력은 스텝 2부터. T=4의 절반이 낭비
    assert h["timestep_trim"]["first_output_step"] == 2
    assert h["timestep_trim"]["skippable_macs"] == 200
    assert h["persistent_state_bytes"] == 64
    print("  headroom OK")


def test_trace_event_format():
    d = trace_json.build(sample())
    ev = d["traceEvents"]
    assert d["displayTimeUnit"] == "ms"
    x = [e for e in ev if e["ph"] == "X"]
    assert len(x) == 4
    for e in x:
        assert all(k in e for k in ("name", "ts", "dur", "pid", "tid", "cat", "args"))
        assert e["args"]["macs_total"] == 400
        assert e["args"]["spike_driven"] is True
    counters = [e for e in ev if e["ph"] == "C"]
    assert len(counters) == 4 and counters[0]["args"]["rate"] == 0.0
    names = {e["args"]["name"] for e in ev if e.get("name") == "thread_name"}
    assert any("spike-driven" in n for n in names)
    assert d["xnn"]["schema"] == "xnn.trace"
    json.dumps(d, allow_nan=False)
    print(f"  Chrome Trace Event OK (X {len(x)}, C {len(counters)})")


def test_roundtrip():
    t = sample()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "t.json")
        t.save(p)
        back = Trace.load(p)
    assert back.executed_macs() == t.executed_macs()
    assert [l.layer_id for l in back.layers] == [l.layer_id for l in t.layers]
    assert back.regions[2].timestep == 2
    assert back.spikes[0].metrics["mean_rate"] == 0.25
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
    for fn in (test_headroom, test_trace_event_format, test_roundtrip,
               test_sources_declare_limits):
        fn()
    print("\n계약 테스트 전부 통과")
