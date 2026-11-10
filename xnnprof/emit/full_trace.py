"""실행 전체의 연산 계층 — PyTorch Profiler의 Chrome Trace export를 얹는다.

구간별 `op_tree`는 '등록된 구간에 귀속된 연산'만 담는다. 그래서 레이어 트랙은
구간이 열린 동안만 차고, 그 사이에 실행된 것(계측 자신의 비용, 라벨 밖 함수형
호출, 루프 바깥 연산)은 어디에도 보이지 않는다. 무엇이 언제 돌았는지를 끊김
없이 보려면 그 몫까지 있어야 한다.

프로파일러의 export에는 hook이 건 라벨이 `user_annotation` 슬라이스로 그대로
들어 있다. 같은 이름이 우리 구간 목록에도 있으므로, 두 시계의 원점 차이를
라벨 쌍에서 구해 한 번만 더하면 전체 이벤트가 우리 타임라인에 얹힌다.
설계서 2.2절의 잔차 문제는 여기서는 생기지 않는다 — 대응점을 하나가 아니라
라벨 전부에서 잡고, 남은 잔차를 출처에 기록하기 때문이다.

얹은 이벤트는 별도 pid에 둔다. 레이어별 트랙은 통합 키로 읽는 뷰이고, 이쪽은
실행 순서로 읽는 뷰다. 둘을 같은 트랙에 섞으면 어느 쪽도 읽히지 않는다.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..collect.torch_prof import LABEL
from ..schema import Trace

FULL_PID = 2
PROCESS_NAME = "실행 전체 (PyTorch Profiler)"


def _label_events(events: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        name = str(e.get("name", ""))
        if LABEL.match(name) and name not in out:
            out[name] = e
    return out


def align(trace: Trace, events: list[dict]) -> tuple[Optional[float], dict[str, Any]]:
    """두 시계의 원점 차이. 라벨 전부에서 구하고 잔차를 함께 돌려준다."""
    labels = _label_events(events)
    diffs = []
    for r in trace.regions:
        if r.timestep is None:
            continue
        ev = labels.get(f"{r.layer_id}@t{r.timestep}")
        if ev is None:
            continue
        diffs.append(r.ts_us - float(ev["ts"]))
    if not diffs:
        return None, {"reason": "export에서 라벨 슬라이스를 찾지 못함"}
    diffs.sort()
    offset = diffs[len(diffs) // 2]
    resid = [d - offset for d in diffs]
    return offset, {
        "matched_labels": len(diffs),
        "offset_us": round(offset, 3),
        "residual_max_us": round(max(abs(x) for x in resid), 3),
    }


def perfectly_nest(slices: list[dict]) -> tuple[list[dict], int]:
    """자식이 부모 밖으로 나가지 않게 끝을 자른다.

    `with_stack`을 켜면 파이썬 추적 이벤트와 연산 이벤트의 시각이 서로 다른
    경로로 찍혀 경계가 몇 µs씩 어긋난다. 뷰어는 완전히 포개지지 않는 슬라이스를
    거부하므로 포함 관계를 강제한다. 자른 개수는 출처에 남긴다.
    """
    groups: dict[tuple, list[dict]] = {}
    for s in slices:
        groups.setdefault((s["pid"], s["tid"]), []).append(s)
    out, clamped = [], 0
    for items in groups.values():
        items.sort(key=lambda s: (s["ts"], -s["dur"]))
        stack: list[float] = []
        for s in items:
            start, end = s["ts"], s["ts"] + s["dur"]
            while stack and stack[-1] <= start + 1e-9:
                stack.pop()
            if stack and end > stack[-1]:
                end = stack[-1]
                clamped += 1
            if end - start <= 0:
                continue
            s["dur"] = round(end - start, 3)
            stack.append(end)
            out.append(s)
    return out, clamped


def events_for(trace: Trace, export_path: str) -> tuple[list[dict], dict[str, Any]]:
    """export 파일을 읽어 우리 타임라인에 얹을 이벤트 목록을 만든다."""
    if not export_path or not os.path.exists(export_path):
        return [], {"reason": "export 파일 없음"}
    try:
        with open(export_path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:                                   # noqa: BLE001
        return [], {"reason": f"{type(e).__name__}: {e}"}

    src = raw.get("traceEvents", raw) if isinstance(raw, dict) else raw
    offset, detail = align(trace, src)
    if offset is None:
        return [], detail

    tids: dict[Any, int] = {}
    out: list[dict] = []
    for e in src:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        tid = tids.setdefault(e.get("tid"), len(tids) + 1)
        out.append({
            "ph": "X", "name": e.get("name"),
            "cat": e.get("cat", "op"),
            "ts": round(float(e["ts"]) + offset, 3),
            "dur": round(float(e["dur"]), 3),
            "pid": FULL_PID, "tid": tid,
        })
    out, clamped = perfectly_nest(out)
    detail["slices"] = len(out)
    detail["threads"] = len(tids)
    if clamped:
        detail["clamped_slices"] = clamped

    meta = [{"ph": "M", "name": "process_name", "pid": FULL_PID, "tid": 0,
             "args": {"name": PROCESS_NAME}}]
    for raw_tid, tid in tids.items():
        meta.append({"ph": "M", "name": "thread_name", "pid": FULL_PID, "tid": tid,
                     "args": {"name": f"thread {raw_tid}"}})
    return meta + out, detail
