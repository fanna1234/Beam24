# Performance Claim Taxonomy

Beam24 separates external comparisons from internal attribution controls.  The
roles below are part of each claim's contract; a speedup cannot move between
roles even when the final output is identical.

## Canonical K512 exhaustive-control rows

| Claim ID | Role | Comparator | Beam24 / comparator median | Paired speedup | Evidence |
|---|---|---|---:|---:|---|
| `OPERATOR-EXTERNAL-K512` | Primary external operator comparison | ccglib opt-static-A, full planar complex FP32 output | 2.58311 / 3.79739 ms | 1.47097x | `evidence/results/operator_sm120_batch256_m1024_n1024_k512.json` |
| `SYSTEM-EXTERNAL-K512` | Exhaustive-path external system control | ccglib materialized complex output plus power and top-1 | 2.14762 / 5.10418 ms | 2.37665x | `evidence/results/system_fair_dense_fused.json` |
| `SYSTEM-ATTRIBUTION-K512` | Exhaustive-path internal attribution | Self-developed dense Tensor-Core path with identical accumulator-resident power and top-1 fusion | 2.14762 / 3.23090 ms | 1.50462x | `evidence/results/system_fair_dense_fused.json` |
| `SYSTEM-WORKSPACE-K512` | Capacity/API result | Production planar complex FP32 workspace | 0 / 2,147,483,648 B | not a latency claim | `evidence/results/system_workspace_elision.json` |

The primary system result is the hierarchical top-1 path. The 2.37665x row
remains an exhaustive-path external control and includes the generally
applicable benefit of avoiding complex-output materialization. The 1.50462x
row is its matched internal attribution: it isolates the sparse representation
and executor after giving dense Tensor Cores the same fusion. Neither row may
be relabeled as a hierarchical result.

## Primary hierarchical K512 rows

The regular-array top-1 route adds an algorithmic hierarchy. Its claims use a
separate decomposition from the exhaustive rows above.

| Claim ID | Role | Comparator | Candidate / comparator median | Paired speedup | Evidence |
|---|---|---|---:|---:|---|
| `HIERARCHY-EXTERNAL-K512` | Primary hierarchical external system comparison | ccglib materialized complex output plus power and top-1 | 0.46956 / 5.09019 ms | 10.83733x | `evidence/results/system_hierarchical_sm120.json` |
| `HIERARCHY-EXTERNAL-SAME-ALGORITHM-K512` | Primary external same-algorithm comparison | ccglib basic with dynamic subarray input and dynamic Stage-2 A gathers | 0.46960 / 1.82195 ms | 3.87988x | `evidence/results/system_external_same_hierarchy.json` |
| `HIERARCHY-ABLATION-K512` | Internal algorithmic ablation | Exhaustive Beam24 with the same sparse executor | 0.46965 / 2.14392 ms | 4.56517x | `evidence/results/system_hierarchical_sm120.json` |
| `HIERARCHY-ATTRIBUTION-K512` | Internal sparse-executor attribution | Dense-fused implementation of the same two-stage hierarchy | 0.46943 / 0.54477 ms | 1.16019x | `evidence/results/system_hierarchical_sm120.json` |

The 10.83733x row is the complete hierarchical application result. It includes
algorithmic beam reduction, sparse execution, and output fusion. The 3.87988x
row applies the same hierarchy to an external dynamic-A implementation and is
the primary fairness denominator. Both must be reported together with the
4.56517x algorithmic ablation and the 1.16019x same-hierarchy executor
attribution; none is a 10.83733x kernel claim. The hierarchical route reserves
36,056,064 bytes of candidate workspace while continuing to avoid the 2 GiB
materialized complex output.

## Independent multi-K replication

The later multi-K matrix is a separate six-process campaign.  It reports
K64/K128/K512 external operator speedups of 1.107x/1.143x/1.465x, external
system speedups of 9.490x/6.174x/2.367x, and internal attribution ratios of
1.630x/1.593x/1.505x.  These rows establish shape coverage.  Their K512 values
must not silently replace or be averaged with the canonical K512 campaigns
above.

## Fourier-family lower bounds and controls

The maintained bounded-workspace control streams two batches through the same
L4096 FP32 cuFFT, interpolation, power, and top-1 path. It uses a 64 MiB
spectrum buffer instead of the full 8 GiB output and measures 7.769 ms versus
0.4708 ms for Beam24. The paired ratio is 16.498x with 95% CI
[16.484, 16.512] and 6/6 wins. A direct cuFINUFFT 2.6 type-2 implementation
passes the 481-angle gate only at `eps=1e-6` but takes 1.633 s, so it is retained
as a rejected implementation. Neither result is a universal lower bound for
specialized NUFFT or CZT systems. Evidence:
`evidence/results/system_fourier_streamed_top1_sm120.json`.

Superseded materialized and interpolation-free controls remain traceable in
[`HISTORICAL_FOURIER_CONTROLS.md`](HISTORICAL_FOURIER_CONTROLS.md); they are not
active denominators.

## New replication and negative boundaries

The finite-snapshot GPU quality campaign executes the actual FP16-input,
FP32-accumulation/atomic-power, complete K512 hierarchy on 33,792 trials across
the frozen eleven-condition envelope. Complete Beam24 matches dense GPU top-1
in 99.929% of trials; all 24 differences are one grid cell. Exhaustive Local-F4
has the same 24 differences and the hierarchy adds none. This replaces the
expected-power result as the paper headline while preserving the older oracle
as independent evidence. Evidence:
`evidence/results/quality_gpu_finite_snapshot_k512.json`.

A disjoint hierarchy audit spans 39,936 admitted K512 trials and 6,144
close-source stress trials. Admitted recall is 97.539%/99.679%/100% at
top-1/top-2/top-4; fixed top-8 preserves 100% recall while matching the 128-row
Stage-2 tile. Close coherent stress contains two top-8 misses and a maximum
coverage rank of 37. This supports hardware-aligned slack inside the admitted
envelope, not an arbitrary-signal guarantee. Evidence:
`evidence/results/quality_hierarchy_margin_coverage_heldout.json`.

At batch64 M1024 N512 K512 on the same SM120a GPU, the external
identical-hierarchy, global-hierarchy, and local-sparse gains remain 2.386x,
3.886x, and 1.106x. This is a second work point, not cross-GPU evidence. The
Stage-1 static-A hybrid is rejected because it is 1.429x slower than the basic
dynamic-A route. Evidence:
`evidence/results/system_hierarchy_secondshape_sm120.json`.

On 15 held-out SPIB K48 recordings, Local-F4 matches dense top-1 in every file,
but the proportional K48 hierarchy fails. These sessions promote only the
local representation claim. Any K48 hierarchy repaired after inspecting these
sessions is diagnostic until validated on a third independent session.
Evidence: `evidence/results/quality_spib48_heldout_sessions.json`.

## Wording rules

- `ccglib`, cuBLASLt, and cuSPARSELt rows are external baselines when their
  manifest ownership is `external`.
- The dense-fused same-output implementation is an internal attribution
  control, not an external baseline and not the headline comparator.
- `2.37665x` must name the materialized ccglib pipeline and disclose the
  output-fusion benefit.
- `1.50462x` must be labeled internal attribution and must name the identical
  fused power/top-1 contract.
- `1.47097x` is a separate full-complex-output operator claim.
- Workspace elimination is reported in bytes and must not be converted into a
  speedup without a fresh timing campaign.
- `10.83733x` must be labeled as the complete hierarchical system result.
- `3.87988x` must name the external ccglib dynamic-A same-hierarchy path.
- `4.56517x` is an internal exhaustive-to-hierarchical algorithmic ablation.
- `1.16019x` is the internal sparse-versus-dense executor attribution after
  both paths use the same hierarchy.
- Hierarchy quality wording is limited to top-1 on the admitted regular-array
  screens; it does not imply top-k recovery or general-array support.
- Margin/coverage wording must report both the admitted top-4 saturation and
  the close coherent top-8 counterexample; top-8 is hardware-aligned slack, not
  a universal recall certificate.
- Streamed cuFFT wording must report its 64 MiB bounded spectrum buffer and
  must not be rewritten as a universal NUFFT/CZT lower bound.
- Held-out K48 data support Local-F4 only; they are counterevidence for a
  scale-free global hierarchy.
- Finite-snapshot K512 GPU wording must report 99.929% dense exact agreement,
  one-grid maximum shift, and zero hierarchy-added misses; it does not imply
  full-spectrum, top-k, cross-K, cross-GPU, or general-array equivalence.
