"""Nsight Systems 수집기 — NVTX 범위에 GPU 커널을 투영해 구간에 넣는다.

hook이 통합 키 문자열 `fc1@t3`을 NVTX 범위로 남기므로, 리포트에서 그 이름을
찾으면 어느 구간인지 바로 정해진다. 시계 원점을 맞출 필요가 없다.

투영 규칙은 nsys의 `nvtx_gpu_proj_sum`과 같다. NVTX 범위 안에서 호출된 CUDA
런타임 API가 발행한 커널을 그 범위의 몫으로 본다. 커널은 비동기로 실행되므로
커널의 실행 시각이 범위 안에 있는지로 판정하면 안 된다 — 범위가 끝난 뒤에
실행되는 커널이 정상이다.

    NVTX_EVENTS                      범위 이름과 [start, end), 스레드
    CUPTI_ACTIVITY_KIND_RUNTIME      런타임 API 호출. correlationId를 들고 있다
    CUPTI_ACTIVITY_KIND_KERNEL       커널 실행. 같은 correlationId로 이어진다

리포트(`.nsys-rep`)는 SQLite로 내보낸 뒤 직접 읽는다. `nsys stats`의 표를
파싱하지 않는 이유는 열 이름과 단위가 버전마다 달라지기 때문이다. 이미
`.sqlite`가 있으면 nsys 없이도 병합된다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from typing import Any

from ..schema import Trace
from .base import Collector, REGION
from .torch_prof import LABEL

PROJECTION = "launch-in-range"      # 범위 안에서 발행된 커널을 그 범위의 몫으로

QUERY = """
SELECT n.text, s.value, COUNT(*), SUM(k.end - k.start),
       MIN(k.start), MAX(k.end)
FROM NVTX_EVENTS n
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.globalTid = n.globalTid AND r.start >= n.start AND r.start <= n.end
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON k.correlationId = r.correlationId
LEFT JOIN StringIds s ON s.id = k.shortName
WHERE n.text LIKE '%@t%'
GROUP BY n.text, s.value
"""

TOTAL_KERNEL = "SELECT SUM(end - start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"


def nsys_bin() -> str:
    """`nsys` 실행 파일. PATH에 없으면 환경변수 `NSYS`로 지정한다.

    Windows 설치본은 PATH에 등록되지 않고, Nsight Compute에 딸려 오는 동명의
    바이너리는 리포트를 만들지 못한다. 경로를 직접 줄 수단이 필요하다.
    """
    env = os.environ.get("NSYS", "").strip('"')
    if env and os.path.exists(env):
        return env
    return shutil.which("nsys") or ""


def sqlite_for(report: str, export: bool = True) -> tuple[str, str]:
    """리포트에 대응하는 `.sqlite` 경로. 없으면 `nsys export`로 만든다."""
    if report.endswith(".sqlite"):
        return (report, "") if os.path.exists(report) else ("", f"{report} 없음")
    if not os.path.exists(report):
        return "", f"{report} 없음"
    db = os.path.splitext(report)[0] + ".sqlite"
    # 리포트를 다시 떴는데 옛 sqlite 가 남아 있으면 지난 측정을 병합하게 된다.
    # 조용히 틀린 수치가 나오므로 더 오래된 db 는 다시 내보낸다.
    if os.path.exists(db) and os.path.getmtime(db) >= os.path.getmtime(report):
        return db, ""
    if not export:
        stale = "리포트보다 오래됨" if os.path.exists(db) else "없음"
        return "", f"{db} {stale} (export 생략)"
    nsys = nsys_bin()
    if not nsys:
        return "", (f"{db} 가 없고 nsys 도 찾지 못해 export 불가 — "
                    "PATH에 넣거나 환경변수 NSYS로 경로를 준다")
    cmd = [nsys, "export", "--type", "sqlite", "--force-overwrite", "true",
           "-o", db, report]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:                                   # noqa: BLE001
        return "", f"nsys export 실패: {type(e).__name__}: {e}"
    if p.returncode != 0 or not os.path.exists(db):
        return "", f"nsys export 실패 (rc={p.returncode}): {p.stderr.strip()[:200]}"
    return db, ""


class NsysCollector(Collector):
    name = "nsight_systems"
    attribution = REGION
    capabilities = ("cuda_api", "kernel_timeline", "nvtx_ranges", "gpu_projection")

    def __init__(self, report: str = "", export: bool = True, top: int = 3):
        self.report = report
        self.export = export
        self.top = top

    def available(self) -> tuple[bool, str]:
        if not self.report:
            return False, "리포트 경로 미지정 (--nsys-report)"
        db, why = sqlite_for(self.report, export=False)
        if db:
            return True, "ok"
        if nsys_bin():
            return True, "ok (export 필요)"
        return False, why

    def collect(self, trace: Trace, **kw: Any):
        import sqlite3

        if not self.report:
            return self.declare("unavailable", "리포트 경로 미지정 (--nsys-report)")
        db, why = sqlite_for(self.report, self.export)
        if not db:
            return self.declare("unavailable", why)

        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = list(con.execute(QUERY))
            total_us, total_n = con.execute(TOTAL_KERNEL).fetchone()
        except Exception as e:                               # noqa: BLE001
            return self.declare("failed", f"{type(e).__name__}: {e}")
        finally:
            try:
                con.close()
            except Exception:
                pass

        per: dict[tuple[str, int], dict] = defaultdict(
            lambda: {"ns": 0, "n": 0, "lo": None, "hi": None, "k": defaultdict(int)})
        for text, kname, n, ns, lo, hi in rows:
            m = LABEL.match(str(text))
            if not m:
                continue
            key = (m.group("layer"), int(m.group("step")))
            d = per[key]
            d["ns"] += int(ns or 0)
            d["n"] += int(n or 0)
            d["lo"] = int(lo) if d["lo"] is None else min(d["lo"], int(lo))
            d["hi"] = int(hi) if d["hi"] is None else max(d["hi"], int(hi))
            d["k"][str(kname)] += int(ns or 0)

        matched, in_ranges = 0, 0
        for (lid, step), d in per.items():
            r = trace.region(lid, step)
            if r is None:
                continue
            top = sorted(d["k"].items(), key=lambda kv: -kv[1])[:self.top]
            r.gpu = {
                "kernel_us": round(d["ns"] / 1000.0, 3),
                "kernels": d["n"],
                "span_us": round((d["hi"] - d["lo"]) / 1000.0, 3),
                "projection": PROJECTION,
                "top_kernels": [{"name": k, "us": round(v / 1000.0, 3)}
                                for k, v in top],
            }
            in_ranges += d["ns"]
            matched += 1

        total_us = float(total_us or 0) / 1000.0
        in_us = in_ranges / 1000.0
        return self.declare(
            "captured" if matched else "partial",
            "" if matched else "리포트의 NVTX 범위가 구간 목록과 맞지 않음",
            sqlite=db, projection=PROJECTION,
            nvtx_ranges=len(per), matched_regions=matched,
            kernel_us_in_ranges=round(in_us, 3),
            kernel_us_total=round(total_us, 3),
            kernels_total=int(total_n or 0),
            in_range_fraction=round(in_us / total_us, 6) if total_us else 0.0,
            note=("범위 밖 커널은 워밍업·학습·정확도 평가 등 측정 구간이 아닌 "
                  "실행의 몫이다. 캡처는 프로세스 전체를 감싼다"))
