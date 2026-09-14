# Beam24

**Regular-array beam search on 2:4 Sparse Tensor Cores.**

[Quick start](#quick-start) · [Reproduction](docs/REPRODUCIBILITY.md) · [Design](docs/DESIGN.md) · [Results](docs/RESULTS.md) · [Data](docs/DATASETS.md)

Beam24 uses the phase structure of regularly sampled line arrays to make
beamforming efficient on NVIDIA Sparse Tensor Cores. A local four-point
Fourier transform concentrates steering energy into two modes, producing one
2:4 support shared by real and imaginary coefficients. Coarse-to-fine search
then restricts full-aperture computation to selected beam sectors.

The GPU implementation carries both structures through sparse complex
computation, beam-power reduction, and top-1 selection. It returns the strongest
beam and its power without materializing the full complex response matrix.

![Local phase structure supplies shared complex 2:4 support; a coarse-to-fine GPU pipeline refines selected sectors and returns top-1.](docs/assets/beam24-overview.svg)

## Results at a glance

| Evaluation | Reference result |
|---|---|
| Same-algorithm external comparison | **3.88×** over the ccglib implementation of the identical hierarchy |
| Complete beam search | **0.470 ms** at the primary K512 workload |
| Dense-GPU top-1 agreement | **99.93%** across held-out finite-snapshot K512 trials; all differences are one grid cell |

The primary workload uses an RTX PRO 6000 Blackwell, batch 256, 1024 beams,
1024 snapshots, 512 sensors, complex FP16 input, and FP32 accumulation.
Timings include both search stages, dynamic candidate gathers, metadata
repacking, power reduction, and final selection. Inputs are device-resident;
static codebook preparation and transfers are outside the timed region.

The external same-hierarchy result is the main comparison. Algorithmic work
reduction and sparse-executor attribution are separate internal ablations,
not additional external baselines. Exact values, intervals, other shapes,
and retained failures are in [Results](docs/RESULTS.md).

## Quick start

```bash
git clone https://github.com/fanna1234/Beam24.git
cd Beam24
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./reproduce.sh smoke
```

The CPU smoke checks the beamforming algebra, support structure, frozen
contracts, and reproduction guards. To verify only the saved evidence:

```bash
./reproduce.sh evidence
```

This verifies recorded results and source identities; it does not execute new
GPU measurements.

For the primary external comparison on SM120a, configure the CUDA compiler
and use the same entry point:

```bash
export CUDACXX=/usr/local/cuda-13.2/bin/nvcc
./reproduce.sh doctor
./reproduce.sh gpu-smoke
./reproduce.sh external-hierarchy
```

`external-hierarchy` fetches pinned ccglib sources and dependencies, builds
both implementations, checks their outputs, and runs six alternating-order
process pairs with direct launches. It rejects incomplete records, other GPU
compute activity, and results below the existing reproduction tolerance.
See [Reproduction](docs/REPRODUCIBILITY.md) for requirements and all commands.

## Signal quality and scope

```bash
./artifact/scripts/get_data.sh spib
export BEAM24_DATA_ROOT="$PWD/work/datasets"
./reproduce.sh quality
```

Real SPIB48 recordings support the **Local-F4 representation**. They do not
establish the complete K512 hierarchy on small real arrays: the proportional
K48 hierarchy fails on held-out sessions. The full hierarchy's strongest
quality evidence is the generated K512 finite-snapshot GPU campaign.

Beam24 is approximate: retaining two Fourier modes and a bounded sector set
can change the selected beam. The coordinate transform itself is exact, and
every emitted sparse group is hardware-legal. Close coherent sources and
nonregular geometry remain explicit counterexamples, not omitted cases.
[Design](docs/DESIGN.md) explains the two approximation boundaries.

## Code organization

| Location | Responsibility |
|---|---|
| [`src/cuda/`](src/cuda/) | Sparse operator, fused power/top-1, hierarchy, and dense control |
| [`src/quality/`](src/quality/) | Synthetic and real-data signal-quality evaluators |
| [`baselines/`](baselines/) | External comparison drivers and exact source revisions |
| [`artifact/`](artifact/) | Reproduction groups, data access, and fixed reference values |
| [`evidence/results/`](evidence/results/) | Frozen measurements and quality boundaries |
| [`tests/`](tests/) | Algebra, contracts, and failure-path regression tests |

The ccglib library is an external comparator, not part of Beam24's production
execution. Historical Fourier and backend variants stay outside default
reproduction paths. [Validation status](docs/VALIDATION.md) distinguishes fresh
checks from retained reference evidence.

## License

Project code uses the [MIT License](LICENSE). Datasets and external libraries
retain their upstream licenses and are not redistributed in the repository.
