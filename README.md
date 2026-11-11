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
| `--no-membrane` | off | 막전위 분포를 산출하지 않음. 호스트 동기화 비중을 잴 때는 켜야 함(아래 주의) |
| `--no-full-trace` | off | 실행 전체의 연산 계층을 얹지 않음. 등록된 구간 밖에서 무엇이 돌았는지가 빠짐 |
| `--nsys` | off | 측정을 `nsys profile`로 감싸고 끝난 뒤 리포트를 구간에 병합. 한 명령으로 끝남 |
| `--with-stack` | off | 파이썬 호출 스택까지 기록. 실행 전체 트랙 깊이가 6 → 19로 늘고 파일이 3배, 실행은 2.5~3.4배 느려짐 |
| `--no-op-tree` | off | 구간 안의 연산자 호출 계층을 내보내지 않음. 빼면 Flame Graph가 그려지지 않음 |

### 장치를 바꿔 가며 같은 모델을 측정할 때

같은 시드를 줘도 CPU와 CUDA는 커널이 달라 학습 결과가 갈림. `--weights`로 가중치를 공유해야 장치 간 비교가 성립함.

```
tools/xnnprof --model all --device cpu  --weights weights --out out-cpu   # 학습 후 저장
tools/xnnprof --model all --device cuda --weights weights --out out-gpu   # 불러 씀
```

`state_dict`의 sha256이 결과 파일 `env.weights_sha256`에 기록됨.

### 호스트 동기화를 잴 때는 --no-membrane

막전위 히스토그램은 계측 라벨 안에서 돌지만, 스텝마다 GPU 큐를 건드려 **다음 구간에서
snntorch가 기다리는 시간을 바꿈**. 같은 모델의 `lif1` 동기화 비중이 막전위를 켜면 37.4%,
끄면 그보다 높게 나옴. 구간 시간이나 동기화 비중을 보고할 때는 끄고 잼.

### nsys로 쌀 때

이중 계측을 피하려면 PyTorch Profiler를 끔. 경로에 한글이 있으면 nsys가 실패하므로 ASCII 경로에서 실행함.

한 명령으로 끝내려면 `--nsys`를 줌.

```
tools/xnnprof --model fcsnn --device cuda --nsys --out out
```

자기 자신을 `nsys profile`로 다시 띄우고, 자식이 끝나 리포트가 생긴 뒤에 병합함.
리포트는 감싼 프로세스가 종료된 뒤에 쓰이므로 측정하는 쪽이 자기 리포트를 읽는 것은
원리적으로 불가능하고, 그래서 바깥에 한 겹을 둠. 결과는 파일 하나에 구간별 GPU 커널
시간·연산자 호출 계층·실행 전체 트랙이 모두 들어감.

NVTX 범위 이름이 모델 사이에서 겹치므로(`lif1@t0` 등) `--model all`과는 같이 못 씀.

나눠서 해도 됨.

```
nsys profile -t cuda,nvtx -o out/fcsnn/nsys --force-overwrite true \
    python tools/xnnprof --model fcsnn --device cuda --out out
tools/nsys_merge out/fcsnn/xnn-trace.json out/fcsnn/nsys.nsys-rep
```

`--no-torch-profiler`를 같이 주면 이중 계측이 빠져 시간이 정확해지는 대신 연산자
호출 계층과 실행 전체 트랙이 안 들어감.

Windows 설치본은 `nsys`가 PATH에 없음. Nsight **Compute**에 딸려 오는 동명의 바이너리는
리포트를 만들지 못하므로 Nsight **Systems** 쪽 경로를 써야 함.

```
set NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe
```

`tools/nsys_merge`가 `nsys export`를 부를 때 이 값을 봄. PATH에 올려도 됨.

리포트는 프로세스가 끝난 뒤에 쓰이므로 측정과 같은 실행에서는 병합할 수 없음.
`tools/nsys_merge`가 2단계로 병합함.

## 출력물

```
out/<model>/
├── xnn-trace.json          Chrome Trace Event. Perfetto · Speedscope가 변환 없이 읽음
└── spikes/
    ├── spikes_T10_lif1.npy     [T, batch, neurons] uint8 이진 배열
    ├── spikes_T10_lif1.json    체크섬과 산출 지표
    ├── membrane_T10_lif1.npy   [T, 66] 막전위 히스토그램 칸 개수 (int64)
    └── membrane_T10_lif1.json  체크섬 · θ · 칸 구성
```

막전위 히스토그램의 칸 구성. 전부 더하면 `batch × neurons`임.

| 칸 | 범위 |
| --- | --- |
| 0 | `mem < 0`. 리셋이 빼기 방식이라 음수로 내려감 |
| 1 ~ 64 | `[0, θ)`를 64등분. 칸 하나의 폭은 `θ/64` |
| 65 | `mem >= θ` |

θ는 `attach()` 때 한 번 읽어 캐시함. hook 안에서 `float(module.threshold)`를 부르면
GPU에서 값을 내리느라 호스트 동기화가 걸려 계측이 실행 순서를 바꿈. 히스토그램의
대상은 뉴런 모듈이 반환한 막전위이고 리셋·감쇠가 이미 적용된 값이므로, 칸 65의
비율을 그 스텝의 발화율로 읽으면 안 됨. 발화율은 스파이크 배열에서 따로 산출함.

`xnn-trace.json`에서 보는 위치

| 위치 | 내용 |
| --- | --- |
| 슬라이스 이름 | 통합 키의 문자열 표현 `fc1@t3` |
| `args` | 타임스텝 · MAC · `ops_self_time_us`(구간에 귀속된 연산자별 시간) · `host_sync_us` |
| `args.snn` | 스파이크 레이어만. dead/saturated 비율 · 발화율 · 첫 스파이크 지연 · SynOps · 정상상태 도달 스텝 · 그 스텝의 `c_t` |
| `args.membrane` | 스파이크 레이어만. θ · 그 스텝의 임계 근처 비율 |
| `args.gpu` | `tools/nsys_merge`로 병합한 뒤에만. 구간별 GPU 커널 시간 · 커널 수 · 상위 커널 |
| 중첩 슬라이스 | 구간 아래에 연산자 호출 계층이 쌓임. Flame Graph의 뿌리가 통합 키가 됨 |
| `실행 전체` 프로세스 | 등록된 구간 밖까지 포함한 전체 연산 계층. 레이어 트랙과 섞이지 않게 별도 프로세스에 둠 |
| `xnn 요약` 마커 | 모델 단위 값. 출력 발생 지연 · 상태 메모리 · 귀속률 · 가중치 지문 |
| 최상위 `xnn` 키 | 구간 전체 · 레이어 집계 · 스파이크 참조 · 수집기별 출처. 뷰어는 읽지 않음 |

## tools/nsys_merge — NVTX 범위 병합

Nsight Systems 리포트의 GPU 커널 시간을 구간에 병합함. NVTX 범위 안에서 발행된
커널을 그 범위의 몫으로 봄(`launch-in-range`) — 커널은 비동기로 실행되므로 커널의
실행 시각으로 판정하면 안 됨.

```
tools/nsys_merge out/fcsnn/xnn-trace.json out/fcsnn/nsys.nsys-rep
```

| 인자 · 옵션 | 설명 |
| --- | --- |
| `trace` | `xnn-trace.json`. 기본은 제자리에서 갱신 |
| `report` | `.nsys-rep` 또는 `.sqlite` |
| `--no-export` | `.sqlite`가 없을 때 `nsys export`를 시도하지 않음 |
| `--out` | 따로 쓸 경로 |
| `--top` | 구간별 상위 커널 수. 기본 `3` |

`.nsys-rep`을 주면 옆에 `.sqlite`를 만들어 읽음. 이미 `.sqlite`가 있으면 nsys 없이도
병합되므로 측정한 기계가 아니어도 됨.

## tools/overhead — 계측 비용

같은 입력을 다섯 조건으로 반복 측정해 hook · 라벨 · 막전위 분포 · 프로파일러의
비용을 분리함.

```
tools/overhead --model all --device cuda --repeat 60
```

| 옵션 | 기본값 |
| --- | --- |
| `--model` · `--steps` · `--batch` | `all` · `10` · `64` |
| `--repeat` | `20` |
| `--device` | `cpu` |

조건은 `base`(계측 없음) · `hook` · `label` · `mem`(막전위 분포) · `prof` · `stack`(`--with-stack`)임.
GPU(RTX 2060 SUPER) · batch 64 · T=10 · repeat 200 기준 실측값.

| 모델 | base | hook | label | mem | prof | stack |
| --- | --- | --- | --- | --- | --- | --- |
| `fcsnn` | 13.4 ms | 1.11x | 1.28x | **1.46x** | 3.68x | **11.05x** |
| `mnist_mlp` | 0.19 ms | 1.09x | 1.37x | 1.34x | 7.17x | 17.80x |
| `repeated_dnn_snn` | 5.97 ms | 1.09x | 1.41x | **1.80x** | 4.96x | **17.06x** |

막전위 분포는 label 대비 `fcsnn` +14%, `repeated_dnn_snn` +28%임. 칸을 세는 데
`torch.bincount`를 쓰면 CUDA 구현이 출력 크기를 정하려고 입력 min·max를 호스트로 꺼내
호출마다 GPU 전체 동기화가 걸림. `scatter_add_`로 바꾼 뒤의 값이고, 바꾸기 전에는
각각 +55% / +106%였음.

파이썬 스택은 prof 대비 2.5~3.4배임. 구조를 볼 때만 켬.

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
