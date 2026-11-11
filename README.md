# xnn-profiler

A unified profiling tool for **Deep Neural Networks (DNNs), Spiking Neural Networks (SNNs), and hybrid DNN-SNN models**.

The profiler collects execution data through PyTorch forward hooks without requiring modifications to model implementations. All profiling results are exported in the **Chrome Trace Event JSON format**, enabling visualization and analysis with tools such as Perfetto and Speedscope.

## Requirements

```bash
pip install torch torchvision snntorch numpy
```

GPU profiling requires a CUDA-enabled PyTorch build.

The currently verified environment is:

- Python 3.12
- PyTorch 2.6.0+cu124
- snntorch 0.9.4

MNIST data is searched in the repository's `data/` directory first, followed by `../profiler-verification/data`. If the dataset is not found in either location, use `--download` to download it automatically.

## Profiling with `tools/xnnprof`

Run the profiler using:

```bash
tools/xnnprof --model fcsnn --device cuda --weights weights --out out-gpu
```

### Command-line Options

| Option | Default | Description |
|---|---|---|
| `--model` | `repeated_dnn_snn` | Model to profile: `fcsnn`, `mnist_mlp`, `repeated_dnn_snn`, or `all` |
| `--steps` | `10` | Number of simulation timesteps (T) |
| `--batch` | `64` | Batch size |
| `--train` | `6000` | Number of training samples. Set to `0` to skip training |
| `--data` | `data/` or `../profiler-verification/data` | Path to the MNIST dataset |
| `--download` | Disabled | Download the dataset if it is not available locally |
| `--raw-input` | Disabled | Use raw input without normalization |
| `--out` | `out` | Output directory |
| `--weights` | None | Directory for model weights. Existing weights are loaded; otherwise, the model is trained and the weights are saved |
| `--device` | `cpu` | Execution device: `cpu` or `cuda`. CUDA mode also exports NVTX annotations |
| `--no-torch-profiler` | Disabled | Disable PyTorch Profiler and collect data using hooks only. Useful for measuring layer-level timing or running with Nsight Systems |
| `--sync-regions` | Disabled | Synchronize at GPU region boundaries to measure actual GPU execution time for each layer |
| `--cover-leaves` | Disabled | Instrument all leaf modules, including modules without child modules |
| `--no-membrane` | Disabled | Disable membrane potential distribution collection. Enable this option when measuring host synchronization overhead (see notes below) |
| `--no-full-trace` | Disabled | Disable the execution-wide operator trace. Operations outside registered profiling regions will not be captured |
| `--nsys` | Disabled | Run profiling under `nsys profile` and automatically merge the generated report into the trace |
| `--with-stack` | Disabled | Capture Python call stacks. Increases execution-wide track depth from 6 to 19, output file size by approximately 3x, and execution time by 2.5–3.4x |
| `--no-op-tree` | Disabled | Disable the operator call hierarchy within profiling regions. The operator hierarchy is required for Flame Graph visualization |

### Comparing CPU and CUDA Results

Using the same random seed does not guarantee identical training results across CPU and CUDA because the underlying computational kernels differ.

To ensure a valid comparison, both devices must use the same model weights. Use `--weights` to share the trained weights across runs.

```bash
# Train on CPU and save the weights
tools/xnnprof --model all --device cpu --weights weights --out out-cpu

# Load the same weights and profile on CUDA
tools/xnnprof --model all --device cuda --weights weights --out out-gpu
```

The SHA-256 hash of the model's `state_dict` is recorded in `env.weights_sha256` in the output trace.

### Measuring Host Synchronization Overhead

Use `--no-membrane` when measuring host synchronization overhead.

Although membrane potential histograms are collected within instrumented regions, their computation accesses the GPU queue at every timestep. This can affect subsequent execution by introducing synchronization delays in snntorch.

For example, the measured synchronization overhead of the `lif1` layer in the same model was 37.4% with membrane profiling enabled, compared to a higher value when it was disabled.

For accurate measurements of region execution time and synchronization overhead, disable membrane profiling using `--no-membrane`.

### Profiling with Nsight Systems

Disable PyTorch Profiler when using Nsight Systems to avoid overlapping instrumentation.

Additionally, Nsight Systems may fail when output paths contain non-ASCII characters. Use ASCII-only paths for profiling.

#### Automatic Integration

The `--nsys` option handles the entire profiling and report-merging process in a single command:

```bash
tools/xnnprof --model fcsnn --device cuda --nsys --out out
```

The profiler launches itself inside `nsys profile`, waits for the child process to finish, and merges the generated report into the trace.

This requires an outer process because Nsight Systems writes its report only after the profiled process terminates. Consequently, a process cannot read its own report while it is still running.

The resulting trace contains:

- GPU kernel execution times for each profiling region
- Operator call hierarchies
- Execution-wide operator traces

**Note:** NVTX range names may overlap across models (e.g., `lif1@t0`). Therefore, `--nsys` cannot be used with `--model all`. Profile each model separately instead.

#### Manual Integration

Nsight Systems can also be run separately:

```bash
nsys profile -t cuda,nvtx -o out/fcsnn/nsys --force-overwrite true \
    python tools/xnnprof --model fcsnn --device cuda --out out

tools/nsys_merge out/fcsnn/xnn-trace.json out/fcsnn/nsys.nsys-rep
```

Using `--no-torch-profiler` avoids overlapping instrumentation and improves timing accuracy. However, the operator call hierarchy and execution-wide trace will not be available.

On Windows, the Nsight Systems executable may not be included in `PATH` by default. Make sure to use the executable bundled with Nsight Systems, rather than the similarly named binary included with Nsight Compute, which cannot generate the required report.

For example:

```cmd
set NSYS=C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe
```

`tools/nsys_merge` uses this environment variable when invoking `nsys export`. Alternatively, add the executable to `PATH`.

Report merging must be performed after the profiled process exits. The `tools/nsys_merge` utility handles the required two-stage merging process.

## Output Structure

Each model produces the following output:

```text
out/<model>/
├── xnn-trace.json
└── spikes/
    ├── spikes_T10_lif1.npy
    ├── spikes_T10_lif1.json
    ├── membrane_T10_lif1.npy
    └── membrane_T10_lif1.json
```

| File | Description |
|---|---|
| `xnn-trace.json` | Chrome Trace Event JSON. Can be opened directly in Perfetto and Speedscope |
| `spikes_*.npy` | Binary spike array with shape `[T, batch, neurons]` and `uint8` dtype |
| `spikes_*.json` | Checksums and derived spike metrics |
| `membrane_*.npy` | Membrane potential histogram with shape `[T, 66]` and `int64` dtype |
| `membrane_*.json` | Histogram metadata, threshold (θ), and bin configuration |

### Membrane Potential Histogram

The histogram contains 66 bins. The sum of all bins at each timestep equals `batch × neurons`.

| Bin | Range | Description |
|---|---|---|
| `0` | `mem < 0` | Negative membrane potentials. The reset operation uses subtraction, which can produce negative values |
| `1–64` | `[0, θ)` | 64 equally sized bins, each with a width of `θ/64` |
| `65` | `mem >= θ` | Membrane potentials at or above the firing threshold |

The threshold (θ) is read once and cached during `attach()`.

Calling `float(module.threshold)` inside a profiling hook would transfer the value from the GPU to the host, introducing synchronization and potentially changing the execution order being measured.

The histogram is computed from the membrane potentials returned by the neuron module, after reset and decay have already been applied. Therefore, the proportion of values in bin 65 must not be interpreted as the firing rate of that timestep.

The actual firing rate is calculated separately from the spike array.

### Understanding `xnn-trace.json`

The trace contains the following information:

| Trace Element | Description |
|---|---|
| Slice name | String representation of the unified profiling key, e.g., `fc1@t3` |
| `args` | Timestep, MAC count, `ops_self_time_us` (operator execution time attributed to the region), and `host_sync_us` |
| `args.snn` | SNN-specific metrics for spiking layers, including dead/saturated ratios, firing rate, first-spike latency, SynOps, steady-state convergence timestep, and the corresponding `c_t` |
| `args.membrane` | Membrane potential metrics for spiking layers, including threshold (θ) and the proportion of values near the threshold |
| `args.gpu` | GPU kernel execution time, kernel count, and top kernels for each region. Available only after merging with `tools/nsys_merge` |
| Nested slices | Operator call hierarchy within each profiling region. The unified profiling key serves as the Flame Graph root |
| `Execution-wide` process | Full operator execution hierarchy, including operations outside registered profiling regions. Stored in a separate process to avoid mixing with layer-level tracks |
| `xnn summary` markers | Model-level metrics, including output latency, state memory, attribution ratio, and weight fingerprint |
| Top-level `xnn` key | Aggregated region statistics, layer summaries, spike references, and data collector metadata. Not directly consumed by visualization tools |

## Nsight Systems Report Merging

### `tools/nsys_merge`

Merges GPU kernel execution times from Nsight Systems reports into the Chrome Trace Event JSON.

GPU kernels are attributed to the NVTX range in which they were launched (`launch-in-range`).

This is important because GPU kernels execute asynchronously. Attribution based on kernel execution timestamps would produce incorrect results.

```bash
tools/nsys_merge out/fcsnn/xnn-trace.json out/fcsnn/nsys.nsys-rep
```

#### Command-line Options

| Argument / Option | Description |
|---|---|
| `trace` | Path to `xnn-trace.json`. Updated in place by default |
| `report` | Path to an `.nsys-rep` or `.sqlite` report |
| `--no-export` | Skip `nsys export` if the corresponding `.sqlite` file does not exist |
| `--out` | Write the merged trace to a separate output path |
| `--top` | Number of top GPU kernels to include per profiling region. Default: `3` |

When an `.nsys-rep` file is provided, a corresponding `.sqlite` file is generated alongside it.

If the `.sqlite` file already exists, report merging can be performed without Nsight Systems installed. This allows trace processing on a different machine from the one used for profiling.

## Profiling Overhead

### `tools/overhead`

Measures the overhead introduced by different profiling components by running the same input under five profiling configurations.

```bash
tools/overhead --model all --device cuda --repeat 60
```

#### Command-line Options

| Option | Default |
|---|---|
| `--model` | `all` |
| `--steps` | `10` |
| `--batch` | `64` |
| `--repeat` | `20` |
| `--device` | `cpu` |

The evaluated configurations are:

- `base`: No instrumentation
- `hook`: Forward hooks
- `label`: Profiling region labels
- `mem`: Membrane potential distribution collection
- `prof`: PyTorch Profiler
- `stack`: Python call stack collection (`--with-stack`)

The following measurements were obtained on an NVIDIA RTX 2060 SUPER with batch size 64, T=10, and 200 repetitions.

| Model | Base | Hook | Label | Mem | Prof | Stack |
|---|---:|---:|---:|---:|---:|---:|
| `fcsnn` | 13.4 ms | 1.11x | 1.28x | 1.46x | 3.68x | 11.05x |
| `mnist_mlp` | 0.19 ms | 1.09x | 1.37x | 1.34x | 7.17x | 17.80x |
| `repeated_dnn_snn` | 5.97 ms | 1.09x | 1.41x | 1.80x | 4.96x | 17.06x |

Membrane profiling introduces an additional overhead of 14% for `fcsnn` and 28% for `repeated_dnn_snn`, relative to the `label` configuration.

The initial implementation used `torch.bincount` to compute histogram bins. However, its CUDA implementation retrieves the input minimum and maximum from the host to determine the output size, causing a full GPU synchronization on every invocation.

Replacing `torch.bincount` with `scatter_add_` eliminated this synchronization bottleneck. Before this change, the additional overhead was 55% and 106%, respectively.

Python stack collection incurs 2.5–3.4x the overhead of PyTorch Profiler. Enable it only when analyzing execution structure.

## Instrumentation Coverage

### `tools/coverage`

Compares the attribution ratios of two instrumentation rules and reports the top three unattributed operations.

```bash
tools/coverage --repeat 5
```

#### Command-line Options

| Option | Default |
|---|---|
| `--model` | `all` |
| `--steps` | `10` |
| `--batch` | `64` |
| `--repeat` | `3` |
| `--profile-memory` | Disabled |

## Cross-Device Consistency

### `tools/evidence`

Compares profiling results across devices by classifying metrics into three categories:

- **Structural invariants:** Metrics expected to remain unchanged across devices
- **Model-dependent metrics:** Metrics determined by model behavior
- **Device-dependent metrics:** Metrics affected by the underlying hardware

The tool exits with a non-zero status if any structural invariant is violated.

```bash
tools/evidence out-cpu out-gpu
```

#### Command-line Options

| Argument / Option | Description |
|---|---|
| `dirs` | Two or more output directories to compare |
| `--models` | Models to compare. Default: `fcsnn,mnist_mlp,repeated_dnn_snn` |

## Static Source Analysis

### `tools/static_scan`

Parses model source code using Python's AST and classifies call sites within `forward()` methods.

The results are compared against the execution trace to identify potentially uninstrumented call sites, including their source line numbers.

```bash
tools/static_scan --trace out-cpu/repeated_dnn_snn/xnn-trace.json --model Mixed
```

#### Command-line Options

| Option | Default | Description |
|---|---|---|
| `--file` | `xnnprof/models/__init__.py` | Python source file to analyze |
| `--trace` | None | Optional trace file for comparison against actual profiling regions |
| `--model` | None | Optional model class name to analyze exclusively |

## Profiling Custom Models

Models do not need to be registered in the built-in model registry. The profiler can also be used directly through its Python API.

```python
from xnnprof.run import profile
from xnnprof.report import summary

trace = profile(
    my_model,
    inputs,
    model_id="mynet",
    timesteps=6,
    batch=8,
    out_dir="out/mynet",
    device="cuda",
)

print(summary(trace))
```

## Tests

The repository includes contract tests that can run without PyTorch installed.

```bash
python tests/test_schema.py
```
