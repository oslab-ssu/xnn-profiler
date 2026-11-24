"""Contract tests that run without torch."""

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
    # the last spiking layer in execution order is the output layer
    assert ol["layer_id"] == "lif1"
    # per_timestep_rate is 0,0,0.5,0.5 -> output starts at step 2
    assert ol["first_spike_step"] == 2 and ol["timesteps"] == 4
    assert "synops" not in m, "items that are not computed get no key"

    dnn = Trace(run_id="r", model_id="dnn", timesteps=1)
    dnn.layers = [Layer("fc1", "Linear", calls=1, macs=10)]
    dnn.regions = [Region(0, "fc1", 0, ts_us=0.0, dur_us=1.0)]
    d = snn_metrics(dnn)
    assert "output_latency" not in d and "state_bytes" not in d, \
        "SNN metrics are undefined for a pure DNN"
    print("  new metrics OK")


class FakeEvent:
    """Mimics only FunctionEvent name, cpu_parent, and self_cpu_time_total."""

    def __init__(self, name, parent=None, self_us=0.0):
        self.name = name
        self.cpu_parent = parent
        self.self_cpu_time_total = self_us


def test_label_join():
    label = FakeEvent("fc1@t3")
    linear = FakeEvent("aten::linear", label, 10.0)
    addmm = FakeEvent("aten::addmm", linear, 90.0)
    loose = FakeEvent("aten::empty", None, 7.0)
    assert owner_of(addmm) == ("fc1", 3), "grandchild events walk up to the label too"
    assert owner_of(linear) == ("fc1", 3)
    assert owner_of(loose) is None, "events outside labels are not attributed"
    # normalized layer_ids are recovered as-is
    assert owner_of(FakeEvent("aten::addmm", FakeEvent("1:Linear@t0"))) \
        == ("1:Linear", 0)
    # instrumentation ops have the instrument label as nearest label, so they stay out of regions
    inst = FakeEvent(INSTRUMENT_LABEL, FakeEvent("fc1@t0"))
    assert owner_of(FakeEvent("aten::all", inst, 5.0)) == INSTRUMENT
    print("  label-based integration OK")


class Linear:
    pass


def test_layer_id_normalization():
    m = Linear()
    assert layer_id_of("fc1", m) == "fc1"
    assert layer_id_of("1", m) == "1:Linear", "numeric name of an nn.Sequential child"
    assert layer_id_of("dnn1.0", m) == "dnn1.0:Linear"
    assert layer_id_of("block.sub1", m) == "block.sub1"
    print("  layer_id normalization OK")


def test_trace_event_format():
    t = sample()
    t.regions[0].ops = {"aten::addmm": 90.0, "aten::linear": 10.0}
    t.regions[1].ops = {"aten::_local_scalar_dense": 4.0, "aten::mul": 1.0}
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
    assert len(with_ops) == 2
    assert with_ops[0]["args"]["ops_self_time_us"]["aten::addmm"] == 90.0
    syncd = [e for e in x if "host_sync_us" in e["args"]]
    assert len(syncd) == 1, "only regions with sync ops are flagged"
    assert syncd[0]["args"]["host_sync_us"] == 4.0
    assert syncd[0]["args"]["host_sync_fraction"] == 0.8, "4us of a 5us region is sync"
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
    print("  lossless round trip OK")


def test_sources_declare_limits():
    from xnnprof.collect.mlir_static import MlirStaticCollector
    s = MlirStaticCollector().collect(Trace(run_id="r", model_id="m"))
    assert s.status == "unavailable" and s.reason and s.attribution == "static-name"
    from xnnprof.collect.perf import PerfCollector
    assert PerfCollector.attribution == "process-global", "perf cannot attribute to layers"
    s = PerfCollector().collect(Trace(run_id="r", model_id="m"))
    assert s.status == "unavailable" and s.reason, "recorded in sources when no window was opened"
    from xnnprof.collect.ncu import NcuCollector
    s = NcuCollector().collect(Trace(run_id="r", model_id="m"))
    assert s.status == "unavailable" and s.reason and s.attribution == "region"
    print("  collector attribution limits declared OK")


def test_steady_state():
    from xnnprof.steady import steady_state
    # every neuron dead or saturated: the per-neuron expectation is 0 change too, so the
    # verdict is undefined; the homogeneous baseline still calls it reached at step 1
    ss = steady_state([[0, 1, 0, 1]] * 4)
    assert "reached_step" not in ss and ss["reason"], ss
    assert ss["reached_step_homogeneous"] == 1 and ss["c_t_homogeneous"] == [0.0] * 3
    # changes only at step 0->1 and fixed after: reached at step 2 under both
    ss = steady_state([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]])
    assert ss["reached_step"] == 2 and ss["reached_step_homogeneous"] == 2, ss
    assert ss["baseline_value"] == 0.375 and ss["baseline_homogeneous"] == 0.5
    # fully flips every step: alternating, so c_t > 1, not reached
    flip = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    ss = steady_state(flip)
    assert ss["reached_step"] is None and ss["c_t_min"] > 1.0, ss
    # [T, batch, neurons]: half the neurons fixed, half flip at random -> the
    # homogeneous baseline dilutes the change, the per-neuron one does not
    import numpy as np
    rng = np.random.default_rng(0)
    arr = np.zeros((10, 8, 20), dtype=np.uint8)
    arr[:, :, :5] = 1
    arr[:, :, 10:] = rng.integers(0, 2, (10, 8, 10))
    ss = steady_state(arr)
    assert ss["baseline_value"] < ss["baseline_homogeneous"]
    assert ss["c_t_min"] > ss["c_t_min_homogeneous"]
    # firing rate 0: denominator is 0, so undefined under both. Never store 0
    ss = steady_state([[0, 0], [0, 0], [0, 0]])
    assert "reached_step" not in ss and ss["reason"], ss
    assert "reached_step_homogeneous" not in ss and ss["reason_homogeneous"]
    print("  steady-state step OK")


def test_membrane_histogram():
    from xnnprof.observations import MEM_BINS, derive_membrane_metrics
    counts = [[0] * MEM_BINS for _ in range(2)]
    counts[0][0] = 50          # mem < 0
    counts[0][60] = 25         # near threshold (bins 57~64)
    counts[0][65] = 25         # mem >= θ
    counts[1][30] = 100        # middle
    m = derive_membrane_metrics(counts)
    assert m["bins"] == MEM_BINS and m["near_bins"] == [57, 64]
    assert m["near_threshold_fraction"] == [0.25, 0.0], m
    assert m["at_or_above_threshold_fraction"] == [0.25, 0.0]
    assert m["below_zero_fraction"] == [0.5, 0.0]
    print("  membrane histogram OK")


def test_op_tree_nesting():
    from xnnprof.collect.torch_prof import nest
    calls = [{"name": "aten::linear", "depth": 1, "off_us": 0.0, "dur_us": 90.0},
             {"name": "aten::addmm", "depth": 2, "off_us": 5.0, "dur_us": 200.0},
             {"name": "aten::t", "depth": 2, "off_us": 1.0, "dur_us": 2.0}]
    out = nest(calls, dur_us=100.0, cap=10)
    assert [c["depth"] for c in out] == [1, 2, 2], out
    deep = next(c for c in out if c["name"] == "aten::addmm")
    assert deep["off_us"] + deep["dur_us"] <= 100.0 and deep["clamped"]
    # the count cap cuts by whole depth levels; dropping only middle levels breaks nesting
    assert {c["depth"] for c in nest(calls, 100.0, cap=1)} == {1}
    print("  operator call hierarchy OK")


def test_nsys_projection():
    import os
    import sqlite3
    import tempfile
    from xnnprof.collect.nsys import NsysCollector

    t = sample()
    db = os.path.join(tempfile.mkdtemp(), "x.sqlite")
    c = sqlite3.connect(db)
    c.executescript("""
    CREATE TABLE NVTX_EVENTS(start INT, end INT, eventType INT, text TEXT,
                             globalTid INT);
    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT, end INT, globalTid INT,
                                             correlationId INT);
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT, end INT,
                                            correlationId INT, shortName INT);
    CREATE TABLE StringIds(id INT, value TEXT);
    """)
    c.execute("INSERT INTO NVTX_EVENTS VALUES(100,200,59,'fc2@t0',7)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES(200,300,59,'xnn::instrument',7)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(150,160,7,1)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(250,260,7,2)")
    # kernels run after the range closes; they must be grouped by launch time
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(9000,12000,1,1)")
    c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(20000,21000,2,1)")
    c.execute("INSERT INTO StringIds VALUES(1,'sgemm')")
    c.commit()
    c.close()

    src = NsysCollector(db).collect(t)
    assert src.status == "captured", src
    g = t.region("fc2", 0).gpu
    assert g["kernels"] == 1 and g["kernel_us"] == 3.0, g
    assert g["top_kernels"][0]["name"] == "sgemm"
    # kernels launched inside the instrument label do not belong to the region
    assert src.detail["kernel_us_in_ranges"] == 3.0
    assert src.detail["kernel_us_total"] == 4.0
    print("  NVTX GPU projection OK")


def test_nsys_stale_sqlite():
    """If the report was re-captured but an old sqlite remains, it must be re-exported."""
    import os
    import tempfile
    import time
    from xnnprof.collect.nsys import sqlite_for

    d = tempfile.mkdtemp()
    rep, db = os.path.join(d, "r.nsys-rep"), os.path.join(d, "r.sqlite")
    open(db, "w").write("old")
    time.sleep(0.02)
    open(rep, "w").write("new")
    got, why = sqlite_for(rep, export=False)
    assert got == "" and "older" in why, (got, why)
    os.utime(db, None)                       # make the db newest
    got, why = sqlite_for(rep, export=False)
    assert got == db, (got, why)
    print("  nsys report/sqlite ordering OK")


def test_op_tree_cap_keeps_first_level():
    """Even when the cap is below the first level, the region must not end up empty."""
    from xnnprof.collect.torch_prof import nest
    calls = [{"name": f"op{i}", "depth": 1, "off_us": float(i), "dur_us": 1.0}
             for i in range(5)]
    calls.append({"name": "deep", "depth": 2, "off_us": 0.1, "dur_us": 0.5})
    out = nest(calls, dur_us=100.0, cap=2)
    assert out and {c["depth"] for c in out} == {1}, out
    print("  op_tree cap handling OK")


def test_summary_gpu_key_distinct():
    """The summary marker device name and region kernel stats must not share a key."""
    t = sample()
    t.env = dict(t.env or {}, gpu="NVIDIA GeForce RTX 2060 SUPER")
    t.regions[0].gpu = {"kernel_us": 1.0, "kernels": 1}
    ev = trace_json.build(t)["traceEvents"]
    marker = [e for e in ev if e["ph"] == "i" and e.get("cat") == "xnn"]
    assert len(marker) == 1
    assert marker[0]["args"]["gpu_name"] == "NVIDIA GeForce RTX 2060 SUPER"
    assert "gpu" not in marker[0]["args"]
    gx = [e for e in ev if e["ph"] == "X" and "gpu" in e["args"]]
    assert gx and all(isinstance(e["args"]["gpu"], dict) for e in gx)
    print("  summary gpu_name / region gpu kept apart OK")


def test_perf_csv_parse():
    """Parse perf stat -x, output. Events that could not be counted are None, not 0."""
    from xnnprof.collect.perf import derive, parse_csv
    text = ("# started on Mon Nov 23 18:00:00 2026\n\n"
            "12.50,msec,task-clock,12500000,100.00,0.998,CPUs utilized\n"
            "40000000,,cycles,12500000,100.00,3.200,GHz\n"
            "80000000,,instructions,12500000,100.00,2.00,insn per cycle\n"
            "1000000,,cache-references,12500000,100.00,,\n"
            "250000,,cache-misses,12500000,100.00,25.00,of all cache refs\n"
            "<not supported>,,branch-misses,0,100.00,,\n")
    c = parse_csv(text)
    assert c["cycles"]["value"] == 40e6 and c["branch-misses"]["value"] is None
    d = derive(c, reps=10)
    assert d["ipc"] == 2.0 and d["cache_miss_rate"] == 0.25
    assert d["per_forward"]["cycles"] == 4e6
    assert d["not_counted"] == ["branch-misses"]
    assert d["privilege"] == "user+kernel"
    # Real perf 6.11 output at perf_event_paranoid=2 (2026-11-24): events carry ":u".
    real = ("540.56,msec,task-clock:u,540558841,100.00,0.692,CPUs utilized\n"
            "82668138,,cycles:u,540558841,100.00,0.153,GHz\n"
            "73770367,,instructions:u,540558841,100.00,0.89,insn per cycle\n")
    d = derive(parse_csv(real), reps=1)
    assert d["per_forward"]["cycles"] == 82668138 and d["ipc"] == round(73770367 / 82668138, 4)
    assert d["privilege"] == "user-only"
    print("  perf stat CSV parsing OK")


def test_ncu_aggregate():
    """Kernels group by every `layer@tN` on their NVTX stack; ratios are duration-weighted."""
    from xnnprof.collect.ncu import aggregate

    def row(kernel, ranges, ns, ffma=0, rd=0, sm=None, **extra):
        m = {"gpu__time_duration.sum": ns,
             "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": ffma,
             "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": 0,
             "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": 0,
             "dram__bytes_read.sum": rd, "dram__bytes_write.sum": 0,
             "sm__throughput.avg.pct_of_peak_sustained_elapsed": sm,
             "launch__grid_size": 8, "launch__block_size": 128,
             "launch__registers_per_thread": 32, **extra}
        return {"kernel": kernel, "ranges": ranges, "metrics": m}

    t = sample()
    rows = [row("sgemm", ["fc2@t0"], 3000, ffma=100, rd=50, sm=40.0),
            row("bias", ["fc2@t0"], 1000, rd=50, sm=80.0),
            row("stray", ["xnn::instrument"], 500),
            row("ghost", ["nope@t7"], 200)]
    d = aggregate(t, rows)
    g = t.region("fc2", 0).ncu
    assert g["kernels"] == 2 and g["kernel_us"] == 4.0, g
    assert g["flops_fp32"] == 200 and g["dram_read_bytes"] == 100
    assert g["flops_per_dram_byte"] == 2.0
    assert g["sm_throughput_pct"] == 50.0, "duration-weighted: (40*3+80*1)/4"
    assert g["top_kernels"][0]["name"] == "sgemm" and g["top_kernels"][0]["grid_size"] == 8
    assert d["kernels_profiled"] == 4 and d["kernels_outside_ranges"] == 1
    assert d["matched_regions"] == 1 and d["unmatched_ranges"] == ["nope@t7"]
    # a metric that was not collected stays None and is reported, never read as 0
    t2 = sample()
    d2 = aggregate(t2, [row("k", ["fc2@t1"], 10, sm=None)])
    assert "sm_throughput_pct" not in t2.region("fc2", 1).ncu
    assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in d2["metrics_missing"]
    from xnnprof.collect.ncu import NcuCollector
    src = NcuCollector("/nonexistent.ncu-rep").collect(t)
    assert src.status == "unavailable" and not t.region("fc2", 0).ncu, "stale counters cleared"
    print("  ncu region aggregation OK")


MLIR_SAMPLE = """#loc1 = loc("/x/linear.py":134:0)
#loc2 = loc("fc2@t0"(#loc1))
#loc3 = loc("fc2@t1"(#loc1))
#loc4 = loc("lif1@t0"(#loc1))
#loc5 = loc(fused[#loc2, #loc4])
module attributes {xnn.level = "bufferized", xnn.provenance = "{\\22timesteps\\22: 4, \\22batch\\22: 2}"} {
  memref.global "private" constant @w : memref<10x8xf32> = dense<0.0> loc(#loc1)
  func.func @main(%arg0: memref<2x8xf32> loc(#loc1)) -> memref<2x10xf32> attributes {llvm.emit_c_interface} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x10xf32> loc(#loc2)
    linalg.matmul ins(%arg0, %arg0 : memref<2x8xf32>, memref<2x8xf32>) outs(%alloc : memref<2x10xf32>) loc(#loc2)
    linalg.generic {indexing_maps = [], iterator_types = ["parallel"]} ins(%alloc : memref<2x10xf32>) outs(%alloc : memref<2x10xf32>) {
    ^bb0(%in: f32 loc("fc2@t1"(#loc1)), %out: f32 loc("fc2@t1"(#loc1))):
      %0 = arith.addf %in, %in : f32 loc(#loc3)
      linalg.yield %0 : f32 loc(#loc3)
    } loc(#loc3)
    %c = arith.constant 0.0 : f32 loc(#loc5)
    return %alloc : memref<2x10xf32> loc(#loc1)
  } loc(#loc1)
} loc(#loc1)
{-#
  dialect_resources: { builtin: { w: "0x0400" } }
#-}
"""


def test_mlir_static_parse():
    """NameLocs through aliases and multi-line ops; fused keys count under each; absent
    runtime keys are reported as eliminated."""
    import os
    import tempfile
    from xnnprof.collect.mlir_static import MlirStaticCollector, level_stats, parse_mlir

    p = parse_mlir(MLIR_SAMPLE)
    names = [n for n, _, _ in p["ops"]]
    assert names.count("linalg.generic") == 1 and "builtin.module" not in names
    generic = next(k for n, k, _ in p["ops"] if n == "linalg.generic")
    assert generic == {"fc2@t1"}, "location after the closing brace belongs to the op"
    st = level_stats(p, "bufferized")
    assert st["per_key"]["fc2@t0"]["linalg.matmul"] == 1
    assert st["alloc_bytes"]["fc2@t0"] == 80 and st["global_bytes"] == 320
    assert st["shared"]["lif1@t0"]["arith.constant"] == 1, "fused location"
    assert p["attrs"]["provenance"]["timesteps"] == 4

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "m.bufferized.mlir"), "w") as f:
        f.write(MLIR_SAMPLE)
    t = sample()
    src = MlirStaticCollector(d).collect(t)
    assert src.status == "partial" and "LLVM IR" in src.reason, src
    assert t.region("fc2", 0).ir["bufferized"] == {"ops": 3, "shared_ops": 1}
    assert t.region("fc2", 2).ir["bufferized"] == {"ops": 0, "eliminated": True}
    assert "fc2@t3" in src.detail["eliminated"]["bufferized"]["keys"]
    assert t.layer("fc2").extra["ir"]["bufferized"]["eliminated_steps"] == 2
    assert t.ir_levels[0].level == "bufferized" and t.ir_levels[0].static_memory_kb == 0.31
    print("  MLIR static attribution OK")


if __name__ == "__main__":
    for fn in (test_snn_metrics, test_label_join, test_layer_id_normalization,
               test_trace_event_format, test_roundtrip, test_sources_declare_limits,
               test_steady_state, test_membrane_histogram, test_op_tree_nesting,
               test_nsys_projection, test_nsys_stale_sqlite,
               test_op_tree_cap_keeps_first_level,
               test_summary_gpu_key_distinct, test_perf_csv_parse,
               test_ncu_aggregate, test_mlir_static_parse):
        fn()
    print("\nall contract tests passed")
