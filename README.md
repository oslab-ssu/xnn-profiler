# xnn-profiler

DNN · SNN · 혼합(xNN) 모델을 **하나의 축에서** 측정하고, 시각화 도구가 바로 읽는
**통일 포맷** 한 벌로 내보내는 프로파일러.

국가 R&D 과제 「xNN 지원 범용 통합 컴파일러 프레임워크」 2차년도 숭실대 수행분.

## 왜

기존에는 도구 5종(PyTorch Profiler · perf · Nsight Systems · Nsight Compute ·
MLIR 정적 분석)이 각각 다른 형식의 결과를 내고, SNN 지표는 모델 코드에 수집 코드를
삽입해 얻었다. 결과를 합칠 공통 키가 없고, 모델이 바뀌면 수집 코드를 다시 썼다.

이 저장소는 두 가지를 한다.

1. **통합** — 레이어·타임스텝 구간을 공통 상관 키로 삼아 도구들의 관측을 한 레코드로 수렴
2. **차별화** — 기존 프로파일러가 주지 않는 "바꾸면 얼마 버나"(절감 상한)를 산출

## 통일 포맷

출력은 **Chrome Trace Event JSON**(`xnn-trace.json`) 하나다. Perfetto · Speedscope ·
Tracy · VS Code extension이 변환 없이 읽는다. 추가 정보는 포맷의 확장 지점에 싣는다.

| 자리 | 담는 것 |
|---|---|
| `tid` | 레이어 1개 = 트랙 1개. 시간축 루프가 그 트랙의 반복 슬라이스로 보인다 |
| `cat` | `dense` · `spike` · `state` — 뷰어에서 도메인 필터 |
| `args` | 레이어별 MAC · 입력 nonzero 비율 · 유효 MAC · 스파이크 구동 여부 · 타임스텝 |
| `ph:"C"` | 타임스텝별 스파이크 발화율 카운터 (Perfetto에서 그래프) |
| `"xnn"` | 최상위 확장 키 — 레이어 집계, 출처 선언, 절감 상한, 스파이크 참조. 뷰어는 `traceEvents`만 읽으므로 무시한다 |

스파이크 원본은 `out/<model>/spikes/*.npy`에 `[T, batch, neurons]` uint8로 따로 나간다
(RTST 독립 검증기가 파일을 직접 읽고 해시를 확인하는 경로).

## 통합 방식 — 상관 키

도구마다 관측 단위가 다르고 서로를 잇는 식별자가 없다. 그래서 **이름 붙은 시간 구간**을
공통 축으로 쓴다. hook이 모든 레이어에 걸려 있으므로 `(layer_id, timestep)` 구간이
공짜로 생기고, 각 도구의 샘플을 그 구간의 시각 범위로 잘라 귀속한다.

귀속 한계는 숨기지 않고 선언한다 (`xnn.sources[].attribution`).

| 수집기 | 귀속 단위 | 상태 |
|---|---|---|
| `torch_hooks` | `region` | 동작 — 레이어·타임스텝·MAC·희소성·스파이크·지속 상태 |
| `torch_profiler` | `region` | 동작 — op 시간·메모리 |
| `perf` | `process-global` | 미구현. **레이어 귀속이 원리적으로 불가** (프로세스 전체 샘플링) |
| `nsight_systems` | `region` | 미구현. NVTX 범위로 귀속 (hook이 이미 방출 가능) |
| `nsight_compute` | `region` | 미구현 |
| `mlir_static` | `static-name` | 미구현. 시간이 없으므로 함수 이름으로 조인 |

perf의 cache-miss를 특정 레이어에 붙이면 거짓 수치가 된다. 그래서 `process-global`로
선언하고, 미구현 수집기도 조용히 빠지지 않고 출처에 남는다.

## 쓰기

```bash
pip install torch snntorch torchvision          # CPU로 충분
tools/xnnprof --model all --steps 10 --batch 64 --train 6000
```

| 모델 | 구성 |
|---|---|
| `mnist_mlp` | 순수 DNN. 784-1000-10 |
| `fcsnn` | 순수 SNN. 같은 784-1000-10에 LIF (snnTorch Tutorial-5 구성) |
| `repeated_dnn_snn` | 혼합. DNN→SNN→DNN→SNN (인수인계 문서 p128~129) |

`data/MNIST`가 필요하다 (`torchvision.datasets.MNIST`가 받는 위치).

## 측정 예 — 같은 구조를 DNN과 SNN으로

T=10, batch 64, MNIST, CPU 단일 스레드.

| | `mnist_mlp` (DNN) | `fcsnn` (SNN) | `repeated_dnn_snn` (혼합) |
|---|---|---|---|
| 실행 MAC | 50,816,000 | **508,160,000** | 21,585,920 |
| 시간축 루프 안 | 0% | 100% | 7.0% |
| sparse 커널 절감 상한 | 0.59% | **0.92%** | 36.4% |
| 지속 상태 메모리 | 0 B | 258,560 B | 18,944 B |

`fcsnn`은 `mnist_mlp`와 레이어 구성이 같고 **정확히 10배(=T)** 의 MAC을 실행한다.
그런데 절감 상한은 0.92%다 — `fc1`이 연산의 98.7%인데 그 입력이 매 스텝 동일한
dense 이미지라 희소성이 0이기 때문이다. **SNN을 쓰는 이유가 거의 실현되지 않는다.**

이 한 줄이 컴파일러 쪽에 주는 지시다: 커널을 sparse로 바꾸는 것보다 **입력 인코딩**을
먼저 봐야 한다. 기존 도구로는 나오지 않던 수치다.

## 구조

```
xnnprof/
├── schema.py          통일 포맷 모델 (Region · Layer · SpikeRef · Source · Trace)
├── run.py             측정 1회 실행
├── metrics.py         절감 상한 (sparse 커널 · 시간축 · 조기 종료)
├── report.py          표준출력 요약
├── observations.py    스파이크 .npy 내보내기
├── collect/           수집기 — base · torch_hooks · torch_prof · stubs
├── emit/trace_json.py Chrome Trace Event 생성
└── models/            mnist_mlp · fcsnn · repeated_dnn_snn
tools/xnnprof          CLI
```

## 다음

1. `perf` · Nsight Systems · Nsight Compute 어댑터를 구간 축에 연결
2. MLIR 정적 분석 이름 조인 (IR 단계별 연산 수 · `memref.alloc` 정적 메모리)
3. SynOps(fan-out 자동 산출) · 뉴런별 스파이크 횟수 추가
4. VS Code extension에 타임스텝 트랙 표현 반영
