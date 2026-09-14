# Design

**Beam24: Exploiting Regular-Array Phase Structure for 2:4 Sparse Tensor-Core Beamforming**

## Why dense beamforming does not map to 2:4

For $M$ beams, $K$ sensors, and $N$ snapshots, multi-beam processing
computes

$$
\mathbf{Y}=\overline{\mathbf{W}}\mathbf{X},
\qquad
\mathbf{W}\in\mathbb{C}^{M\times K},
\quad
\mathbf{X}\in\mathbb{C}^{K\times N}.
$$

Here $\mathbf{W}$ contains nonconjugated steering rows. Regular-array steering
weights are dense along the reduction dimension, while Sparse Tensor Cores
require two zeros in every aligned group of four values in the sparse operand.
Directly removing antenna-domain coefficients changes coherent accumulation and
degrades the spatial response.

On ten real 48-sensor VLA recordings, direct joint-complex 2:4 selection gives
only **0.97075** mean spatial-spectrum correlation and matches the dense peak
on **10%** of recordings. The hardware format is useful; the naive
representation is not.

## Signal-to-hardware co-design

Adjacent sensors in a regular line array follow a predictable local phase
progression. Let $\mathbf{F}$ be block diagonal with unitary four-point
transforms. Beamforming admits the exact change of basis

$$
\overline{\mathbf{W}}\mathbf{X}
=\overline{(\mathbf{W}\mathbf{F})}(\mathbf{F}\mathbf{X})
=\overline{\mathbf{U}}\mathbf{Z}.
$$

The change of basis is exact. Within each four-sensor block, the transformed
steering progression concentrates in two adjacent modes. Beam24 retains one
two-of-four support for the complete complex coefficient, so the real and
imaginary planes share one metadata pattern across all four real products.

Across beams, a centered subarray selects eight sectors that expand into one
fixed 128-row full-aperture tile. This avoids an irregular candidate list while
evaluating 15.6% of exhaustive beam--sensor work at the primary shape. The GPU
producer fuses $\mathbf{F}\mathbf{X}$, the sparse executor retains FP32 complex
accumulators, and the epilogue reduces $|Y|^2$ and selects top-1 without writing
the 2-GiB complex output.

Approximation begins only when two transformed modes and a bounded set of beam
sectors are retained; the coordinate change and every emitted 2:4 group remain
exactly legal by construction.

## Cross-layer design

### 1. Signal-structured support

Beam24 chooses a single two-of-four support for each complex group. Sharing the
support between real and imaginary weights preserves complex semantics and
allows one sparse metadata stream to drive all four real products.

Mask quality is evaluated in the spatial-response domain rather than by
coefficient error alone. This is why the local-F4 representation recovers
**0.999975** mean spectrum correlation on the real VLA recordings while direct
antenna-domain 2:4 does not.

### 2. Full-complex Sparse Tensor-Core operator

For $\widehat{\mathbf{Y}}=\overline{\mathbf{S}}\mathbf{Z}$, each complex tile
is assembled from

$$
\widehat{\mathbf{Y}}_{r}=\mathbf{S}_{r}\mathbf{Z}_{r}
                 +\mathbf{S}_{i}\mathbf{Z}_{i},
\qquad
\widehat{\mathbf{Y}}_{i}=\mathbf{S}_{r}\mathbf{Z}_{i}
                 -\mathbf{S}_{i}\mathbf{Z}_{r}.
$$

Beam24 maps these four terms to sparse MMA instructions while retaining the
real and imaginary accumulator planes across the reduction loop. The SM120a
kernel uses warp-specialized producer/consumer roles, a two-stage TMA pipeline,
shared-memory local transforms, `ldmatrix` fragment loads, and FP32
accumulation.

### 3. Same-output system fusion

An operator-only comparison can hide the cost of downstream materialization.
Beam24 therefore includes an internal dense Tensor-Core attribution control
with the same final output contract: one FP32 maximum beam power and one beam
index per batch. Both paths perform beamforming, power reduction, and top-1
selection; only the representation and matrix engine differ.


## Search and representation have different boundaries

The local transform exposes approximate two-mode structure without removing
physical sensors. The global search bounds the candidate set; a fixed top-8
budget is not a worst-case recovery guarantee. These choices are evaluated
separately, including close coherent-source failures and the held-out K48
hierarchy rejection. See [Results](RESULTS.md) for the associated evidence.

The GPU dispatch retains exhaustive Beam24 below `batch × snapshots = 3072`
on the validated K512 profile. Changing a shape or architecture requires its
own correctness and performance checks rather than copying this crossover.

## Overview source

```bash
python scripts/render_overview.py
```

The diagram uses a declared four-sensor phase progression to illustrate the
transform and its shared support. It is not a timing trace or a measured array
recording. Its generated manifest records the values and the source hash.
