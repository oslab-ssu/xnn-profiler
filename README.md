# xnn-profiler

DNN · SNN · 혼합 모델을 하나의 축에서 측정하고 Chrome Trace Event JSON으로 내보냄.
모델 코드를 수정하지 않고 forward hook으로만 수집함.

설계 문서는 노션 `통합 프로파일링 정보 추출 기법 설계` 참고.

## 환경

```
pip install torch torchvision snntorch numpy
```

GPU 측정은 CUDA 빌드 torch가 필요함. 현재 검증 환경은 torch 2.6.0+cu124 · snntorch 0.9.4 · Python 3.12.

MNIST는 저장소의 `data/`에서 찾고, 없으면 옆 폴더 `../profiler-verification/data`를 봄. 둘 다 없으면 `--download`를 줌.

## tools/xnnprof — 측정

```
tools/xnnprof --model fcsnn --device cuda --weights weights --out out-gpu
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--model` | `repeated_dnn_snn` | `fcsnn` · `mnist_mlp` · `repeated_dnn_snn` · `all` |
| `--steps` | `10` | 타임스텝 수 T |
| `--batch` | `64` | 배치 크기 |
| `--train` | `6000` | 학습 샘플 수. `0`이면 학습하지 않음 |
| `--data` | 저장소 `data/` → `../profiler-verification/data` | MNIST 위치 |
| `--download` | off | `--data`에 없으면 내려받음 |
| `--raw-input` | off | 정규화 없이 원본 입력을 씀 |
| `--out` | `out` | 출력 디렉터리 |
| `--weights` | 없음 | 가중치 디렉터리. 파일이 있으면 불러 쓰고 없으면 학습 후 저장 |
| `--device` | `cpu` | `cpu` · `cuda`. cuda면 NVTX 라벨도 내보냄 |
| `--no-torch-profiler` | off | hook만으로 측정. 레이어별 시간을 보거나 nsys로 쌀 때 켬 |
| `--sync-regions` | off | GPU 구간 경계마다 동기화해 레이어별 실제 GPU 시간을 잼 |
| `--cover-leaves` | off | 자식이 없는 모든 모듈까지 계측 지점을 삽입 |

### 장치를 바꿔 가며 같은 모델을 측정할 때

같은 시드를 줘도 CPU와 CUDA는 커널이 달라 학습 결과가 갈림. `--weights`로 가중치를 공유해야 장치 간 비교가 성립함.

```
tools/xnnprof --model all --device cpu  --weights weights --out out-cpu   # 학습 후 저장
tools/xnnprof --model all --device cuda --weights weights --out out-gpu   # 불러 씀
```

`state_dict`의 sha256이 결과 파일 `env.weights_sha256`에 기록됨.

### nsys로 쌀 때

이중 계측을 피하려면 PyTorch Profiler를 끔. 경로에 한글이 있으면 nsys가 실패하므로 ASCII 경로에서 실행함.

```
nsys profile -t cuda,nvtx -o out/fcsnn/nsys --force-overwrite true \
    python tools/xnnprof --model fcsnn --device cuda --no-torch-profiler
nsys stats --report nvtx_gpu_proj_sum out/fcsnn/nsys.nsys-rep
```

## 출력물

```
out/<model>/
├── xnn-trace.json          Chrome Trace Event. Perfetto · Speedscope가 변환 없이 읽음
└── spikes/
    ├── spikes_T10_lif1.npy   [T, batch, neurons] uint8 이진 배열
    └── spikes_T10_lif1.json  체크섬과 산출 지표
```

`xnn-trace.json`에서 보는 위치

| 위치 | 내용 |
| --- | --- |
| 슬라이스 이름 | 통합 키의 문자열 표현 `fc1@t3` |
| `args` | 타임스텝 · MAC · `ops_self_time_us`(구간에 귀속된 연산자별 시간) · `host_sync_us` |
| `args.snn` | 스파이크 레이어만. dead/saturated 비율 · 발화율 · 첫 스파이크 지연 · SynOps |
| `xnn 요약` 마커 | 모델 단위 값. 출력 발생 지연 · 상태 메모리 · 귀속률 · 가중치 지문 |
| 최상위 `xnn` 키 | 구간 전체 · 레이어 집계 · 스파이크 참조 · 수집기별 출처. 뷰어는 읽지 않음 |

## tools/overhead — 계측 비용

같은 입력을 네 조건으로 반복 측정해 hook · 라벨 · 프로파일러의 비용을 분리함.

```
tools/overhead --model all --device cuda --repeat 60
```

| 옵션 | 기본값 |
| --- | --- |
| `--model` · `--steps` · `--batch` | `all` · `10` · `64` |
| `--repeat` | `20` |
| `--device` | `cpu` |

## tools/coverage — 계측 지점 누락

등록 규칙 두 가지의 귀속률을 비교함. 미귀속 연산 상위 3건을 함께 출력함.

```
tools/coverage --repeat 5
```

| 옵션 | 기본값 |
| --- | --- |
| `--model` · `--steps` · `--batch` | `all` · `10` · `64` |
| `--repeat` | `3` |
| `--profile-memory` | off |

## tools/evidence — 장치 간 대조

측정값을 구조 불변량 · 모델 의존 · 장치 의존 세 부류로 나눠 대조함. 구조 불변량이 하나라도 어긋나면 0이 아닌 값으로 종료함.

```
tools/evidence out-cpu out-gpu
```

| 인자 | 설명 |
| --- | --- |
| `dirs` | 비교할 출력 디렉터리 2개 이상 |
| `--models` | 기본 `fcsnn,mnist_mlp,repeated_dnn_snn` |

## tools/static_scan — 소스 정적 검사

모델 소스를 AST로 파싱해 `forward`의 호출 지점을 분류하고, 트레이스와 대조해 계측되지 않은 지점을 줄 번호와 함께 보고함.

```
tools/static_scan --trace out-cpu/repeated_dnn_snn/xnn-trace.json --model Mixed
```

| 옵션 | 기본값 |
| --- | --- |
| `--file` | `xnnprof/models/__init__.py` |
| `--trace` | 없음. 주면 실제 구간과 대조 |
| `--model` | 없음. 주면 그 클래스만 봄 |

## 임의의 모델에 적용

레지스트리에 등록하지 않아도 Python API로 바로 씀.

```py
from xnnprof.run import profile
from xnnprof.report import summary

trace = profile(my_model, inputs, model_id="mynet", timesteps=6, batch=8,
                out_dir="out/mynet", device="cuda")
print(summary(trace))
```

## 테스트

torch 없이 도는 계약 테스트.

```
python tests/test_schema.py
```
