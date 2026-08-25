# Historical Fourier Controls

These controls preserve the route by which the active streamed Fourier
comparison was selected. They are not active denominators.

| Control | Result | Reason superseded | Evidence |
|---|---:|---|---|
| Interpolation-free FP32 cuFFT | 6.584 ms | Does not recover the uniform-angle output grid. | `evidence/history/system_fft_baseline_sm120.json` |
| Materialized L4096 cuFFT | 27.301 ms | Uses the correct grid but allocates the complete spectrum. | `evidence/history/system_fourier_uniform_angle_sm120.json` |
| Prototype streamed L4096 cuFFT | 22.211x over Beam24 | Replaced by a fresh build of the maintained public source. | Local immutable experiment raw data |
| Pre-cleanup maintained wrapper | 21.991x over Beam24 | Still compiled superseded entrypoint and unused kernels. | Local immutable experiment raw data |

The cold reproduction of the interpolation-free control is retained at
`evidence/history/fft_lower_bound_reproduce_sm120_2026-08-24.json`; its source
is retained at `baselines/history/cufft_top1_lower_bound.cu`.

The accepted control is `CUFFT_STREAMED_UNIFORM_ANGLE_TOP1`: L4096 FP32 cuFFT,
uniform-angle interpolation, direct power/top-1, and a 64 MiB bounded spectrum
buffer. The generic cuFINUFFT type-2 screen is retained separately as a
quality-passing but noncompetitive implementation.
