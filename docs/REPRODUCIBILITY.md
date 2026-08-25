# Reproducibility Policy

## Profiles

- `smoke`: CPU-only contract checks plus a bounded ideal-ULA quality run.
- `gpu-smoke`: fresh SM120 build, high-entropy K64 correctness, and sanitizer
  admission where Compute Sanitizer is available.
- `quality`: external real-data screens using checksum-pinned datasets.
- `robustness`: generated K512 regular-ULA perturbation sweep plus a disjoint
  three-seed top-L margin/coverage audit with admitted and stress rows.
- `finite-gpu-quality`: 33,792 finite-snapshot K512 trials through the dense
  GPU oracle, exhaustive Local-F4 GPU, and complete hierarchical Beam24 GPU.
- `fourier-control`: six-process generated-data comparison of the streamed
  L4096 uniform-angle cuFFT top-1 control and Beam24 on SM120.
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

The finite-GPU quality row uses batch256 M=N=1024 K512, eleven perturbation
conditions, three held-out seeds, and four chunks per seed/condition. It is a
quality campaign rather than a latency comparison; all three routes consume
identical generated planar FP16 snapshots.

CUDA-event timing is public latency. NCU replay is diagnostic only. The
reproduce scripts retain every process log and print `[OK >=reference]`,
`[~within3%]`, or `[LOW]` against the canonical published anchor.
The Fourier target fails before timing when `nvidia-smi pmon` reports an
external non-display process; the local lock cannot serialize unrelated jobs.

## Cold-checkout rule

The acceptance test is a fresh checkout, from-scratch environment, build,
smoke, and full target on the validated hardware class. Failures are preserved and
never converted into a success banner.
