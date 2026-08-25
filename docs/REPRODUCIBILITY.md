# Reproducibility Policy

## Profiles

- `smoke`: CPU-only contract checks plus a bounded ideal-ULA quality run.
- `gpu-smoke`: fresh SM120 build, high-entropy K64 correctness, and sanitizer
  admission where Compute Sanitizer is available.
- `quality`: external real-data screens using checksum-pinned datasets.
- `robustness`: generated K512 regular-ULA perturbation sweep with three
  held-out seeds and a per-condition top-1 gate.
- `fft-lower-bound`: six-process generated-data screen of the optimistic FP32
  cuFFT path, exhaustive Beam24, and hierarchical Beam24 on SM120.
- `system`: the six-process direction-balanced internal D1/S2 attribution
  campaign. The primary external ccglib comparison is recorded separately.
- `hierarchy`: the six-process direction-balanced exhaustive-versus-hierarchy
  ablation at batch256 M=N=1024 K512.
- `all`: every available target; missing required hardware or data is a hard
  failure with a precise message.

## Measurement contract

The system attribution row uses batch256 M=N=1024 K=512, planar complex FP16 input,
FP32 accumulation, device-resident inputs, 20 warmups, 100 timed iterations,
and six independent AB/BA processes under one GPU lock. Static weight
preparation is amortized. The final output is one maximum power and beam index
per batch.

The hierarchy row uses the same device-resident precision, shape, warmup,
iteration, process, and GPU-lock contract. Its timed candidate includes both
Beam24 stages, top-8 selection, packed-A gather, metadata repacking, power
zero-fills, and the mapped final argmax. Static codebook preparation and Graph
instantiation are amortized. The separately recorded external comparison uses
direct launch mode for both hierarchy and ccglib.

CUDA-event timing is public latency. NCU replay is diagnostic only. The
reproduce scripts retain every process log and print `[OK >=reference]`,
`[~within3%]`, or `[LOW]` against the canonical published anchor.

## Cold-checkout rule

The acceptance test is a fresh checkout, from-scratch environment, build,
smoke, and full target on the validated hardware class. Failures are preserved and
never converted into a success banner.
