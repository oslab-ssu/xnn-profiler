"""아직 연결되지 않은 수집기들. 인터페이스 계약과 귀속 한계를 미리 선언한다.

조용히 빠지지 않고 출처(sources)에 'unavailable'로 남는다 — 도구 한계와
실제 누락을 구분하기 위한 것이다.
"""

from __future__ import annotations

import shutil

from .base import PROCESS_GLOBAL, REGION, STATIC_NAME, Unimplemented


class PerfCollector(Unimplemented):
    name = "perf"
    attribution = PROCESS_GLOBAL
    capabilities = ("cycles", "instructions", "ipc", "cache_misses", "page_faults")

    def available(self):
        return (True, "ok") if shutil.which("perf") else (False, "perf 미설치")


class NsysCollector(Unimplemented):
    name = "nsight_systems"
    attribution = REGION
    capabilities = ("cuda_api", "kernel_timeline", "nvtx_ranges")

    def available(self):
        return (True, "ok") if shutil.which("nsys") else (False, "nsys 미설치")


class NcuCollector(Unimplemented):
    name = "nsight_compute"
    attribution = REGION
    capabilities = ("kernel_counters", "occupancy", "memory_bandwidth")

    def available(self):
        return (True, "ok") if shutil.which("ncu") else (False, "ncu 미설치")


class MlirStaticCollector(Unimplemented):
    name = "mlir_static"
    attribution = STATIC_NAME
    capabilities = ("ir_op_counts", "static_memory", "lowering_expansion")

    def available(self):
        return False, "컴파일 산출물 경로 미지정 (.mlir / .ll)"


ALL = (PerfCollector, NsysCollector, NcuCollector, MlirStaticCollector)
