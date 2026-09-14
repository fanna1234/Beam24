# Reproduction

Use `./reproduce.sh` from the repository root. The original
`./artifact/reproduce.sh` entry point remains supported. CPU checks, saved
evidence, real-data quality, and new GPU measurements are distinct targets.

## Environment

The pinned Python dependencies require **Python 3.12 or newer**. A clean CPU
environment can be created with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh smoke
```

If the environment is not activated, set `BEAM24_PYTHON` to its interpreter.
The existing `PYTHON_BIN` setting is also accepted; conflicting values fail.
No private data or prebuilt binary is required for the CPU smoke.

GPU execution requires Linux, an SM120a GPU, a C++ toolchain, CUDA 13.0–13.3,
CMake, and `flock`. The measured configuration uses an RTX PRO 6000 Blackwell
Workstation Edition, CUDA 13.2.51, and driver 610.43.02. CMake 3.25+ and Ninja
are needed for the supplied presets. Compute Sanitizer is used by GPU smoke
when installed; its absence is not a sanitizer pass.

```bash
export CUDACXX=/usr/local/cuda-13.2/bin/nvcc
./reproduce.sh doctor
./reproduce.sh build
./reproduce.sh gpu-smoke
```

`NVCC` or `CUDA_HOME` can also select the compiler. A requested CUDA build
fails if the compiler is missing; CPU-only configuration is explicit:

```bash
cmake --preset cpu-check
```

## Primary external comparison

```bash
./reproduce.sh external-hierarchy --dry-run
./reproduce.sh external-hierarchy
```

This target reproduces the main **3.87988×** same-hierarchy comparison. It:

- fetches ccglib and its transitive dependencies at the evaluated revisions;
- builds Beam24 and the native ccglib hierarchy from source;
- checks candidate coverage, indexing, gathered operands, and output on the
  generated 481-angle correctness case;
- measures six independent process pairs with alternating order, using direct
  launches for both paths;
- preserves every log and rejects incomplete pairs, invalid latency, foreign
  GPU compute activity, or a result below the existing 97%-of-reference gate.

The ccglib baseline uses `basic` at both stages because Stage-2 weights are
dynamic. It includes the centered input gather, dense weight gather, complex
outputs, power reduction, and top-1. It is not the static-A optimized operator
control. Production Beam24 does not link against ccglib.

The source fetcher refuses a different revision, origin, or dirty checkout.
`BEAM24_BASELINE_ROOT` can point to an existing **clean, pinned** dependency
cache. The wrapper, xtl, and xtensor revisions match the original evaluated
build; a moving upstream `main` is not used. No third-party source is vendored.

## Timing contract

| Item | Setting |
|---|---|
| Shape | batch 256, 1024 beams, 1024 snapshots, 512 sensors |
| Input / accumulation | Planar complex FP16 / FP32 |
| Algorithm | Centered K128 subarray, 128 coarse beams, top-8 sectors, 128 padded K512 refinement rows |
| Output | Maximum beam power and full-grid beam index per batch |
| Timer | CUDA events around the complete device-resident pipeline |
| Repetition | 20 warmups, 100 iterations, six fresh AB/BA process pairs |
| External launch mode | Direct on both sides |
| Excluded | Host transfers and static codebook preparation |

The timed Beam24 pipeline includes dynamic gathers, metadata repacking,
power-buffer initialization, both sparse stages, and mapped top-1. CUDA Graph
instantiation is excluded for Graph-mode internal ablations. Nsight timing is
diagnostic rather than the reported latency.

A local GPU lock only coordinates cooperating jobs. The maintained timing
entry points also check for unrelated compute activity; do not terminate
someone else's process to obtain a result.
On a multi-GPU server, select one device with `CUDA_VISIBLE_DEVICES`, preferably
using its GPU UUID. The occupancy guard checks that same device rather than
rejecting unrelated jobs on other cards. This is still single-GPU execution.

To admit the built external and Beam24 implementations without collecting a
timing campaign, run `python scripts/run_external_hierarchy.py --check-only`.
Results from other device models or software versions are portability checks,
not replacements for the primary performance record.

## Real-array quality

```bash
./artifact/scripts/get_data.sh spib
export BEAM24_DATA_ROOT="$PWD/work/datasets"
./reproduce.sh quality
```

The downloader validates the archive and sensor-file checksums. Partial
transfers remain `.part` files and are never treated as completed downloads.
The ten extracted recordings are checked against their source archive before
analysis. `quality` evaluates all representation and pruning rows and then
checks the Local-F4 claim on all ten files. It verifies correlation and peak
shift directly; the 3% **performance** tolerance is not applied to quality.

The target reproduces the A2601 ten-recording Local-F4 screen. Held-out SPIB,
Acoular, and LOCATA evidence remains separately identified in
[Results](RESULTS.md) and [Data](DATASETS.md); this command does not silently
claim to rerun every historical dataset cell.

## Internal controls and optional campaigns

| Target | Purpose | Requirement |
|---|---|---|
| `smoke` | Algebra, support, source contracts, and regression tests | CPU |
| `evidence` | Verify recorded measurements and source identities | CPU; no new measurements |
| `doctor` | Check interpreter, compiler, and GPU class | CUDA environment |
| `build` | Build the maintained CUDA targets | CUDA compiler |
| `gpu-smoke` | High-entropy correctness and available sanitizer checks | SM120a |
| `quality` | Ten-recording real VLA Local-F4 screen | SPIB data, CPU |
| `external-hierarchy` | Main external identical-hierarchy comparison | SM120a, pinned ccglib |
| `hierarchy` | Exhaustive-to-hierarchical algorithmic ablation | SM120a; Graph on both sides |
| `system` | Exhaustive sparse-versus-dense-fused attribution | SM120a |
| `finite-gpu-quality` | Full 33,792-trial generated finite-snapshot quality campaign | SM120a |
| `robustness` | Perturbation and top-L coverage/stress sweeps | CPU |
| `fourier-control` | Maintained streamed L4096 Fourier control | SM120a |

`hierarchy` and `system` reproduce internal ablations, not external speedups.
The finite-snapshot campaign is a quality experiment, not a latency run.
`all` runs every available target and therefore needs both GPU resources and
SPIB data. Every target accepts `--dry-run`; use it to inspect the plan first.

## Outputs and failure handling

Each campaign creates a fresh directory under `artifact/runs/`; existing runs
are not overwritten. Raw process logs, environment details, admission checks,
and summaries stay together. The external summary records binary identities
and six paired ratios, with a process-bootstrap confidence interval.

Timing status is `[OK >=reference]`, `[~within3%]`, or `[LOW]` using the
reference in `artifact/expected/reference_anchors.json`. Low results return a
nonzero status and remain in the run directory. They must not be dropped from
an aggregate or replaced by a more favorable denominator.

| Setting | Purpose |
|---|---|
| `BEAM24_PYTHON` / legacy `PYTHON_BIN` | Interpreter for all reproduction scripts |
| `CUDACXX`, `NVCC`, or `CUDA_HOME` | CUDA compiler selection |
| `BEAM24_BUILD_DIR` | Production build directory |
| `BEAM24_EXTERNAL_BUILD_DIR` | Separate external-control build directory |
| `BEAM24_BASELINE_ROOT` | Pinned upstream dependency cache |
| `BEAM24_DATA_ROOT` | External dataset tree |
| `BEAM24_GPU_LOCK` | Cooperative GPU lock |
| `BEAM24_BUILD_JOBS` | CPU compilation parallelism |

See [Validation status](VALIDATION.md) for the checks actually completed during
this repository update. Frozen results and fresh reproduction evidence remain
separate.
