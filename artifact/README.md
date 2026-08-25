# Beam24 Artifact

## Reproduction entrypoint

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
./artifact/reproduce.sh smoke
./artifact/reproduce.sh --dry-run all
```

Targets:

- `smoke`: CPU-only anchors, source checks, tests, and bounded synthetic ULA quality.
- `gpu-smoke`: fresh SM120 build and small high-entropy correctness checks.
- `quality`: checksum-pinned real-data quality rows under `BEAM24_DATA_ROOT`.
- `robustness`: three-seed synthetic K512 regular-ULA perturbation gate; no GPU
  or external dataset is required.
- `fft-lower-bound`: six-process materialized FP32 cuFFT/Beam24 screen on
  SM120; the target excludes uniform-angle interpolation and expects FP16
  correctness to fail.
- `system`: six-process same-output internal D1/S2 attribution campaign on the
  validated GPU class.
- `hierarchy`: six-process exhaustive-versus-hierarchical Beam24 ablation at
  batch256 M=N=1024 K512.
- `all`: every target; missing hardware or data is a hard failure.

Outputs use a new directory under `artifact/runs/`; no result is overwritten.
Evaluation targets print the reference value beside the measured value and preserve
`[LOW]` outcomes.

Environment variables: `BEAM24_DATA_ROOT`, `BEAM24_BUILD_DIR`, `CUDA_HOME`,
`BEAM24_GPU_LOCK`, and `PYTHON_BIN`.

The robustness target validates only bounded synthetic top-1 evidence; it does
not claim full-spectrum, real-calibration, top-k, or general-array robustness.

The system and hierarchy targets are validated for CUDA 13.0–13.3 on an SM120
GPU. They validate the 1.50462x dense-fused attribution control and the 4.56517x
hierarchical ablation. External ccglib comparisons remain separately recorded
baseline campaigns because ccglib is not bundled into the artifact.

Data hooks:

```bash
./artifact/scripts/get_data.sh spib
./artifact/scripts/get_data.sh acoular
./artifact/scripts/get_data.sh locata
```

LOCATA is large and is not downloaded by the default smoke.
