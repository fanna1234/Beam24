# Beam24 Evaluation Suite

The primary tables compare Beam24 with external implementations. Beam24
representation variants, backend substitutions, and fusion choices are kept in
separate ablation tables.

Machine-readable contracts:

- [manifest.json](manifest.json): exact external revisions and implementation ownership;
- [evaluation_matrix.json](evaluation_matrix.json): selected tests and cell status.

## Final selected tests

### 1. Main signal-quality table

| Row | Role |
|---|---|
| Dense reference | Accuracy oracle |
| cuSPARSELt-Prune-Complex | Off-the-shelf independent real/imaginary 2:4 pruning and execution |
| Beam24 | Proposed regular-array representation and fused execution |

Datasets:

| Dataset family | Purpose |
|---|---|
| Synthetic regular arrays | K/look-angle dense-response sweep |
| SPIB48 | Primary real regular-line-array evidence |
| Acoular64 | Multi-source non-line-array stress test |
| LOCATA32 Eigenmike | Real spherical-array negative boundary |

Every row is evaluated on every dataset family. A geometry failure is retained,
not dropped. Dense GPU libraries share the same mathematical dense reference;
their actual-data numerical parity is a correctness gate, not duplicated as
multiple quality rows.

### 2. Main full-complex operator table

| Row | Exact role |
|---|---|
| ccglib best | Fastest measured external ccglib configuration at each shape |
| cuBLASLt complex 4-GEMM | Vendor dense-library sanity baseline |
| cuSPARSELt-Prune-Complex | True off-the-shelf sparse path from antenna-domain weights |
| Beam24 | Proposed full-complex sparse operator |

Physical execution profiles are K64, K128, and K512 with batch=256 and
M=N=1024. SPIB48, Acoular64, LOCATA32, and synthetic K64 share one K64 timing
campaign; their quality cells remain independent.

`ccglib best` is selected from basic, optimized static-A, and full-pipeline
variants under the same contract. These configuration variants are not separate
headline rows.

### 3. Main beamforming-to-top1 system table

| Row | Final output |
|---|---|
| ccglib materialized-top1 | One maximum power and beam index per batch |
| ccglib hierarchical-top1 | Same hierarchy and output, with dynamic subarray-input and Stage-2 A gathers |
| streamed cuFFT uniform-angle top1 | L4096 interpolation, bounded spectrum buffer, and the same final output |
| Beam24 | Same output without materializing the complex matrix |

K512 is the primary hierarchical result. The ccglib same-hierarchy row is
measured only at K512; standard exhaustive K64/K128/K512 rows remain separate.

## Ablations

### Representation

```text
dense
-> direct antenna-domain 2:4
-> add local F4
-> shared joint-complex support
-> response-aware support refinement
```

### Backend

`Beam24 representation + cuSPARSELt backend` is a hybrid ablation. It uses the
Beam24 F4 and joint 2:4 representation but replaces the custom fused executor
with four cuSPARSELt real GEMMs and FP32 output promotion. It is not an external
method and never appears in the main baseline table.

### Consumer fusion

```text
materialized complex output
-> separate power reduction
-> accumulator-resident power
-> GPU top-1
```

The dense fused same-output implementation is an internal attribution control,
not an external baseline.

## External methods excluded from the numeric main tables

Sparse-array ADMM and reduced-dimensional beamspace methods change physical
sensor selection, the SINR/beampattern objective, or output dimension. They are
discussed as related methods and receive explicit `not_applicable` matrix cells
unless a same-contract implementation and dataset definition are established.

## Frozen performance contract

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition, SM120.
- CUDA 13.2.51.
- Planar complex FP16 input and FP32 public output/accumulation.
- Device-resident timing; 20 warmups; 100 iterations.
- Six independent direction-balanced processes under one GPU lock.
- Static weight preparation measured separately and amortized.
- CUDA-event latency is public; Nsight replay is diagnostic only.

## Pinned external sources

| Source | Revision | License |
|---|---|---|
| [ccglib](https://github.com/nlesc-recruit/ccglib) | `756cc1436136387087d9f5f4778c90f347658cd0` | Apache-2.0 |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | `v4.7.0` / `dcf215af68a2d08d305076c152a06f201728cd53` | BSD-3-Clause for selected C++ sources |
| [CUDA Library Samples](https://github.com/NVIDIA/CUDALibrarySamples) | `34377293fb148e90aad16fd762fa2445f941be55` | Apache-2.0 |
| [cuSPARSELt](https://docs.nvidia.com/cuda/cusparselt/) | 0.9.1.1 | NVIDIA component terms |

```bash
python3 artifact/scripts/get_baselines.py all
```

The fetcher refuses to overwrite a checkout at a different revision.

## Current completion boundary

Completed:

- external ccglib and cuBLASLt K64/K128/K512 operator profiles;
- ccglib materialized-top1 and Beam24 K512 system row;
- quality-passing streamed L4096 cuFFT with uniform-angle interpolation,
  64 MiB bounded workspace, and direct power/top-1;
- Beam24 quality on all four dataset families, including the LOCATA rejection;
- internal dense-fused attribution under the same final-output contract.

The selected external quality, operator, and system matrices are complete.
Additional SNR/SIR, calibration, position-error, and multipath sweeps are
optional robustness extensions rather than missing main-table cells.
Superseded controls and slower backend experiments are retained only in
`baselines/history/` and `evidence/history/`.

## Completed matrix summary

| Operator K | Beam24 over ccglib best | Beam24 over cuSPARSELt-Prune | Wins |
|---:|---:|---:|---:|
| 64 | 1.107x | 3.003x | 6/6 |
| 128 | 1.143x | 2.721x | 6/6 |
| 512 | 1.465x | 2.107x | 6/6 |

| Same-output system K | Beam24 over ccglib materialized-top1 | Beam24 over dense fused control |
|---:|---:|---:|
| 64 | 9.490x | 1.630x |
| 128 | 6.174x | 1.593x |
| 512 | 2.367x | 1.505x |

The external-system ratios include the generally applicable benefit of avoiding
complex-output materialization and form the main external table. The
dense-fused column is the internal attribution control for the sparse
representation and executor.
