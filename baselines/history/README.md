# Historical Baseline Implementations

This directory preserves superseded controls that are no longer built or used
as active denominators. The maintained baseline inventory is defined by
`baselines/manifest.json`.

- `cufft_top1_lower_bound.cu`: interpolation-free materialized cuFFT control
  and the rejected FP16 numerical path. It was replaced by the quality-passing
  bounded-workspace uniform-angle implementation in `baselines/cuda/`.
- `beam24_rep_cusparselt_backend.cu`: vendor-backend ablation for the Beam24
  representation. It was slower than ccglib at every measured aperture and is
  excluded from the active baseline suite. Its result remains under
  `evidence/history/`.
