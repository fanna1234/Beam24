# Reference results

The values below belong to their original measured campaigns. New repository
validation does not replace the frozen anchors. The primary result is the
external identical-hierarchy comparison; internal controls have separate roles.


### Signal quality

| Configuration | Dataset / geometry | Mean spectrum correlation | Dense-peak agreement | Mean peak shift |
|---|---|---:|---:|---:|
| Dense reference | SPIB/SACLANT, 48-sensor VLA, 10 recordings | 1.000000 | 100% | 0.000° |
| Direct joint 2:4 | Same data and beamformer | 0.970747 | 10% | 0.225° |
| **Beam24 local-F4 joint 2:4** | Same data and beamformer | **0.999975** | **90%** | **0.025°** |
| Beam24 local-F4 joint 2:4 | LOCATA Eigenmike, spherical geometry | 0.930901 | 11.1% | unsupported |

The Eigenmike row is a deliberate negative boundary: the current construction
is for regularly sampled line arrays, not arbitrary array geometry.

The early hierarchy screens remain in the historical evidence. The maintained
full-hierarchy quality claim is anchored to the K512 finite-snapshot GPU
campaign below. Real K48 results establish Local-F4 quality separately; the
held-out proportional hierarchy failure prevents extending the global-search
claim to those recordings.

The finite-snapshot GPU campaign runs the complete FP16-input/FP32-power path
on 33,792 held-out K512 trials. Beam24 matches dense GPU top-1 in 99.929% of
trials; all 24 differences are one grid cell, and hierarchy adds no miss over
exhaustive Local-F4. Five repeated replays are index-stable, with zero packed-A,
metadata, or nonfinite errors.

On 15 held-out SPIB P2701 moving-source and A2601_2 recordings, Local-F4
matches dense top-1 in every file with zero grid shift and mean spectrum
correlations of 0.999965 and 0.998332. The proportional K48 hierarchy fails on
the same sessions, so they extend the local representation evidence but not
the global search claim.

### GPU performance

At the primary K512 system shape, the integrated hierarchy includes Stage 1,
device top-8 selection, packed-weight gather, sparse-metadata repacking, Stage
2, and mapped top-1 in the timed region.

| Scope | Comparator and role | Speedup | 95% paired bootstrap CI | Wins |
|---|---|---:|---:|---:|
| End-to-end top-1 | **External:** ccglib materialized-top1 | **10.837x** | [10.829, 10.843] | 6/6 |
| Same hierarchy | **External:** ccglib dynamic-A hierarchy | **3.880x** | [3.879, 3.881] | 6/6 |
| Same sparse executor | **Ablation:** exhaustive Beam24 | **4.565x** | [4.561, 4.569] | 6/6 |
| Same hierarchy | **Attribution:** internal dense-fused control | **1.160x** | [1.159, 1.161] | 6/6 |

The 10.837x row is the complete application result and intentionally includes
hierarchical work reduction, sparse execution, and avoidance of materialized
complex output. The 3.880x row applies the identical hierarchy to the fastest
measured external dynamic-A path, closing the same-algorithm fairness
comparison. The exhaustive and dense-fused rows isolate the algorithmic and
sparse-executor contributions. Hierarchy uses 36,056,064 bytes of dynamic
candidate workspace and retains the exhaustive route when
`batch × snapshots < 3072`, where its fixed multi-launch cost is slower on the
measured SM120a K512 route.

An independent 46,080-trial hierarchy audit separates 39,936 admitted K512
cases from 6,144 close-source stress cases. Every admitted exhaustive Local-F4
winner appears within the first four Stage-1 sectors: top-1/top-2/top-4 recall
is 97.539%/99.679%/100%. Fixed top-8 therefore adds hardware-aligned slack for
the 128-row Stage-2 tile. Close coherent stress produces two top-8 misses and a
maximum coverage rank of 37, so no arbitrary-signal recall guarantee is made.

A second measured point at batch64, M1024, N512, K512 retains 2.386x over the
external identical-hierarchy path, 3.886x from global hierarchy, and 1.106x
from local sparse execution, with 6/6 wins for every comparison. This is a
shape-local replication on the same GPU, not cross-GPU evidence.

The strongest same-grid Fourier control uses a quality-passing 4096-point FP32
cuFFT, linear complex interpolation, and direct power/top-1. It streams two
batches at a time, reducing the full-spectrum workspace from 8 GiB to 64 MiB.
The maintained implementation takes 7.769 ms versus 0.4708 ms for Beam24, a
paired 16.498x ratio with 95% CI [16.484, 16.512] and 6/6 wins. These results do
not bound every specialized NUFFT or CZT. Superseded and rejected Fourier
routes are separated under `baselines/history/` and
`docs/HISTORICAL_FOURIER_CONTROLS.md`.

The exhaustive results below remain the representation/executor controls with
no hierarchical work reduction.

All rows below use an NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
$B=256$, $M=N=1024$, $K=512$, complex FP16 inputs, and FP32
accumulation. Each speedup is the paired geometric mean of six independent
direction-balanced processes.

| Scope | Comparator and role | Beam24 speedup | 95% paired bootstrap CI | Wins |
|---|---|---:|---:|---:|
| Full complex output | ccglib opt-static-A | **1.47097x** | [1.46888, 1.47308] | 6/6 |
| Beamforming + power + top-1 | **External:** materialized ccglib pipeline | **2.37665x** | [2.37546, 2.37791] | 6/6 |
| Beamforming + power + top-1 | **Attribution:** internal dense-fused control | **1.50462x** | [1.50362, 1.50582] | 6/6 |

The 2.37665x row is the exhaustive-path external system control and includes
the generally applicable benefit of avoiding complex-output materialization.
The 1.50462x row is its matched internal attribution after applying the same
output fusion to dense Tensor Cores. The primary hierarchical result remains the
hierarchical 10.837x complete result paired with the 3.880x same-algorithm
external comparison above.

The completed multi-K campaign additionally measures Beam24 over the fastest
ccglib configuration at **1.107x / 1.143x / 1.465x** for K64/K128/K512. The
same-output dense-fused attribution ratios are **1.630x / 1.593x / 1.505x**.
These are independent replication rows, not replacements for the canonical
K512 campaigns. See [the evaluation suite](../baselines/README.md) for contract
separation.

A fresh-checkout validation rebuilt the SM120a targets, passed numerical
checks and Compute Sanitizer with zero errors, and reproduced the internal
dense-fused attribution at **1.50431x**, with wins in all six processes.

## Performance contract

| Item | Validated setting |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0 |
| CUDA | 13.0–13.3; validation run used 13.2.51 |
| Input / accumulation | Planar complex FP16 / FP32 |
| Primary shape | batch=256, M=1024, N=1024, K=512 |
| Timing | CUDA events, device-resident inputs, 20 warmups, 100 iterations |
| Repetition | Six independent AB/BA processes under one GPU lock |
| System output | One FP32 maximum power and one beam index per batch |
| Complex output workspace | 0 B in production `check=0`; allocated only for correctness oracle |
| Static preparation | Steering-weight transform and compression amortized |

The repository does not extrapolate these numbers to unmeasured shapes,
precisions, GPU generations, host-transfer-inclusive execution, or multi-GPU
systems.
