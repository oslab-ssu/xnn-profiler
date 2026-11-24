"""MLIR static collector: op counts and static memory per lowering level, per unified key.

Reads the artifacts of `tools/mlir_export` (`<model>.{linalg,bufferized,llvm_dialect}.mlir`,
optionally `<model>.llvm_ir.ll`). Every op there carries a NameLoc `layer@tN`, so ops are
attributed by name, not by time (`attribution: static-name`).

A region that ran but has no ops at a level was removed by the compiler: CSE keeps the
first op and its location, so a deduplicated step leaves no trace other than its absence
(`fcsnn` fc1@t1..t9, whose input is the same image every step). Such keys are listed as
`eliminated`. An op whose location fuses several keys is counted under each and in
`shared_ops`, never split.

The text is parsed directly so this runs without torch-mlir. A multi-line op's location
follows its closing brace, so op starts and closing lines are paired with a stack.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Optional

from ..schema import IrLevel, Trace
from .base import Collector, STATIC_NAME
from .torch_prof import LABEL

LEVELS = ("linalg", "bufferized", "llvm_dialect")
LOC_ALIAS = re.compile(r"^(#loc\d*) = loc\((.*)\)\s*$")
TRAILING_LOC = re.compile(r"loc\((.*)\)\s*$")
OP = re.compile(r"^\s*(?:%[\w#:, %]+ = )?\"?([a-z_][\w]*\.[\w.]+)\"?[\s({<]")
NAME = re.compile(r'"([^"]+)"\(')            # NameLoc "name"(child); file locs are "f":l:c
REF = re.compile(r"#loc\d*")
ATTR = re.compile(r'xnn\.(\w+) = "((?:[^"\\]|\\.)*)"')
MEMREF = re.compile(r"memref<([^>]*)>")
ELEM_BYTES = {"f64": 8, "i64": 8, "f32": 4, "i32": 4, "f16": 2, "bf16": 2,
              "i16": 2, "i8": 1, "i1": 1}
SKIP_OPS = ("builtin.module",)
SHAPE = re.compile(r"(?:tensor|memref)<([0-9x]+)x[a-z]+\d*[,>]")
INS_OUTS = re.compile(r"ins\([^:]*:([^)]*)\)\s*outs\([^:]*:([^)]*)\)")


def contraction_macs(name: str, line: str) -> Optional[int]:
    """Multiply-accumulates of a named linalg contraction, from its operand shapes."""
    m = INS_OUTS.search(line)
    if not m:
        return None
    ins = [list(map(int, s.split("x"))) for s in SHAPE.findall(m.group(1) + ">")]
    outs = [list(map(int, s.split("x"))) for s in SHAPE.findall(m.group(2) + ">")]
    if not outs or len(ins) < 2:
        return None
    a, b, o = ins[0], ins[1], outs[0]
    if name == "linalg.matmul" and len(a) == 2 and len(b) == 2:
        return a[0] * a[1] * b[1]
    if name == "linalg.batch_matmul" and len(a) == 3 and len(b) == 3:
        return a[0] * a[1] * a[2] * b[2]
    if name == "linalg.conv_2d_nchw_fchw" and len(b) == 4 and len(o) == 4:
        return o[0] * o[1] * o[2] * o[3] * b[1] * b[2] * b[3]
    return None


def memref_bytes(spec: str) -> Optional[int]:
    """`64x1000xf32` -> bytes. None when a dimension is dynamic."""
    parts = spec.split(",")[0].strip().split("x")
    n = 1
    for d in parts[:-1]:
        if not d.isdigit():
            return None
        n *= int(d)
    return n * ELEM_BYTES.get(parts[-1], 4)


class LocTable:
    def __init__(self) -> None:
        self.alias: dict[str, str] = {}
        self._keys: dict[str, frozenset] = {}

    def keys(self, expr: str) -> frozenset:
        """All `layer@tN` NameLocs reachable from a location expression."""
        found = {n for n in NAME.findall(expr) if LABEL.match(n)}
        for ref in REF.findall(expr):
            if ref not in self._keys:
                self._keys[ref] = frozenset()              # cycle guard
                self._keys[ref] = self.keys(self.alias.get(ref, ""))
            found |= self._keys[ref]
        return frozenset(found)


def parse_mlir(text: str) -> dict[str, Any]:
    """Per-op (name, keys) list plus module attributes and memref sizes."""
    locs = LocTable()
    body = text.split("{-#", 1)[0]                   # drop dialect_resources blobs
    lines = body.splitlines()
    for ln in lines:
        m = LOC_ALIAS.match(ln)
        if m:
            locs.alias[m.group(1)] = m.group(2)
    ops: list[tuple[str, frozenset, str]] = []
    stack: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(("#", "//", "^")):
            continue
        loc = TRAILING_LOC.search(s)
        if s.startswith("}"):
            if loc and stack:
                ops.append((stack.pop(), locs.keys(loc.group(1)), ln))
            continue
        m = OP.match(ln)
        if not m:
            continue
        name = m.group(1)
        if s.endswith("{") and not loc:
            stack.append(name)
        else:
            ops.append((name, locs.keys(loc.group(1)) if loc else frozenset(), ln))
    attrs = {}
    for k, v in ATTR.findall(body[:200000]):
        raw = v.replace('\\22', '"').replace('\\"', '"')
        try:
            attrs[k] = json.loads(raw)
        except ValueError:
            attrs[k] = raw
    return {"ops": [o for o in ops if o[0] not in SKIP_OPS], "attrs": attrs}


def level_stats(parsed: dict[str, Any], level: str) -> dict[str, Any]:
    ops = parsed["ops"]
    total = Counter(n for n, _, _ in ops)
    per_key: dict[str, Counter] = defaultdict(Counter)
    shared: dict[str, Counter] = defaultdict(Counter)
    alloc: dict[str, int] = defaultdict(int)
    macs: dict[str, int] = defaultdict(int)
    global_bytes = 0
    for name, keys, ln in ops:
        if name == "memref.global":
            m = MEMREF.search(ln)
            b = memref_bytes(m.group(1)) if m else None
            global_bytes += b or 0
        mac = contraction_macs(name, ln) if name.startswith("linalg.") else None
        for k in keys:
            per_key[k][name] += 1
            if mac:
                macs[k] += mac
            if len(keys) > 1:
                shared[k][name] += 1
            if name == "memref.alloc":
                m = MEMREF.search(ln)
                b = memref_bytes(m.group(1)) if m else None
                alloc[k] += b or 0
    unkeyed = Counter(n for n, k, _ in ops if not k)
    return {"level": level, "total": total, "per_key": per_key, "shared": shared,
            "alloc_bytes": alloc, "macs": macs, "global_bytes": global_bytes,
            "unkeyed": unkeyed}


def count_llvm_ir(text: str) -> Counter:
    """Instruction counts in a `.ll` file; no debug locations, so totals only."""
    skip = {"br", "phi", "ret", "declare", "define", "attributes", "source_filename",
            "target", "module"}
    c: Counter = Counter()
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s[0] in ";!\"}" or re.match(r"^[\w.]+:", s):
            continue
        p = s.split()
        op = p[2] if p[0].startswith("%") and len(p) > 2 and p[1] == "=" else p[0]
        if op[0].isalpha() and op not in skip:
            c[op] += 1
    return c


def _split(key: str) -> tuple[str, int]:
    m = LABEL.match(key)
    return m.group("layer"), int(m.group("step"))


class MlirStaticCollector(Collector):
    name = "mlir_static"
    attribution = STATIC_NAME
    capabilities = ("ir_op_counts", "static_memory", "lowering_expansion", "shared_ops")

    def __init__(self, path: str = "", model_id: str = ""):
        self.path = path
        self.model_id = model_id

    def files(self, model_id: str) -> dict[str, str]:
        out = {}
        for lv in LEVELS:
            p = os.path.join(self.path, f"{model_id}.{lv}.mlir")
            if os.path.exists(p):
                out[lv] = p
        ll = os.path.join(self.path, f"{model_id}.llvm_ir.ll")
        if os.path.exists(ll):
            out["llvm_ir"] = ll
        return out

    def available(self) -> tuple[bool, str]:
        if not self.path:
            return False, "no compiler output path given (--mlir DIR from tools/mlir_export)"
        if not os.path.isdir(self.path):
            return False, f"{self.path} is not a directory"
        return True, "ok"

    def collect(self, trace: Trace, **kw: Any):
        ok, why = self.available()
        if not ok:
            return self.declare("unavailable", why)
        model = self.model_id or trace.model_id
        files = self.files(model)
        if not files:
            return self.declare("unavailable",
                                f"no {model}.<level>.mlir in {self.path}")
        trace.ir_levels = []
        for layer in trace.layers:
            layer.extra.pop("ir", None)
        for r in trace.regions:
            r.ir = {}

        prov, check, warn = {}, {}, []
        known = {(r.layer_id, r.timestep) for r in trace.regions}
        lids = {l.layer_id for l in trace.layers}
        unmatched: set[str] = set()
        expansion, eliminated, macs_cmp = {}, {}, {}
        for lv in LEVELS:
            if lv not in files:
                continue
            with open(files[lv], encoding="utf-8") as f:
                parsed = parse_mlir(f.read())
            prov = parsed["attrs"].get("provenance", prov)
            check = parsed["attrs"].get("execution_check", check)
            st = level_stats(parsed, lv)
            trace.ir_levels.append(IrLevel(
                level=lv, total_operations=sum(st["total"].values()),
                operation_details=dict(st["total"].most_common()),
                static_memory_kb=round(st["global_bytes"] / 1024, 2)))
            per_layer: dict[str, Counter] = defaultdict(Counter)
            shared_layer: dict[str, Counter] = defaultdict(Counter)
            for key, c in st["per_key"].items():
                lid, step = _split(key)
                if lid not in lids:
                    unmatched.add(key)
                    continue
                per_layer[lid].update(c)
                shared_layer[lid].update(st["shared"].get(key, {}))
                r = trace.region(lid, step) if (lid, step) in known else None
                if r is not None:
                    r.ir[lv] = {"ops": sum(c.values()),
                                "shared_ops": sum(st["shared"].get(key, {}).values())}
                    if lv == "linalg" and st["macs"].get(key):
                        r.ir[lv]["macs"] = st["macs"][key]
                    if lv == "bufferized" and st["alloc_bytes"].get(key):
                        r.ir["alloc_bytes"] = st["alloc_bytes"][key]
            layer_macs: dict[str, int] = defaultdict(int)
            for key, v in st["macs"].items():
                layer_macs[_split(key)[0]] += v
            for lid, c in per_layer.items():
                layer = trace.layer(lid)
                d = layer.extra.setdefault("ir", {})
                d[lv] = {"ops": sum(c.values()), "op_types": dict(c.most_common(8)),
                         "shared_ops": sum(shared_layer[lid].values())}
                if lv == "linalg" and layer_macs.get(lid):
                    d[lv]["static_macs"] = layer_macs[lid]
                    d[lv]["executed_macs"] = layer.macs
            if lv == "linalg":
                total_static = sum(st["macs"].values())
                macs_cmp = {"static": total_static, "executed": trace.executed_macs(),
                            "static_over_executed": (round(total_static / trace.executed_macs(), 6)
                                                     if trace.executed_macs() else None)}
            present = {_split(k) for k in st["per_key"]}
            gone = sorted(f"{l}@t{s}" for l, s in known - present if l in lids)
            eliminated[lv] = gone
            for g in gone:
                trace.region(*_split(g)).ir[lv] = {"ops": 0, "eliminated": True}
            for lid in lids:
                n = sum(1 for g in gone if _split(g)[0] == lid)
                if n and trace.layer(lid) is not None:
                    trace.layer(lid).extra.setdefault("ir", {}).setdefault(
                        lv, {"ops": 0})["eliminated_steps"] = n
            expansion[lv] = sum(st["total"].values())
            if st["unkeyed"]:
                d = expansion.setdefault("unkeyed", {})
                d[lv] = sum(st["unkeyed"].values())
        if "llvm_ir" in files:
            with open(files["llvm_ir"], encoding="utf-8") as f:
                c = count_llvm_ir(f.read())
            trace.ir_levels.append(IrLevel(level="llvm_ir", total_operations=sum(c.values()),
                                           operation_details=dict(c.most_common())))

        for k, want in (("timesteps", trace.timesteps), ("batch", trace.batch)):
            if prov.get(k) is not None and want is not None and prov[k] != want:
                warn.append(f"compiled with {k}={prov[k]}, measured with {want}")
        status = "partial" if warn or "llvm_ir" not in files else "captured"
        reason = "; ".join(warn + ([] if "llvm_ir" in files else
                                   ["no LLVM IR (.ll); op counts stop at the LLVM dialect"]))
        return self.declare(
            status, reason, path=self.path, files=sorted(files), provenance=prov,
            execution_check=check, ops_per_level=expansion, macs=macs_cmp,
            eliminated={lv: {"count": len(v), "keys": v[:20]} for lv, v in eliminated.items()},
            unmatched_keys=sorted(unmatched)[:20],
            note=("ops are attributed by the NameLoc layer@tN set at export; a region that "
                  "ran but has no ops at a level was removed by the compiler (eliminated)"))
