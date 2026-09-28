# Repository validation

This record separates the source-delivery check from the original measurements.
The CUDA implementations, baseline drivers, and frozen result anchors are
unchanged from the repository's previous revision.

## Completed source and data checks

- The pinned Python environment installs on Python 3.12.
- All 32 algebra, documentation, source-contract, and failure-path tests pass.
- Production and pinned ccglib builds succeed independently with CUDA 13.2.51.
- All ten SPIB recordings match the source archive, and the Local-F4 quality
  screen reproduces its correlation and peak-shift criteria.
- The external-hierarchy runner now builds its dependencies, admits outputs,
  validates the exact direct-launch schedule, and preserves six AB/BA pairs.
- Supplemental CUDA 13.1 execution on an idle SM120a Server Edition passes the
  GPU correctness smoke, both available memcheck checks, and the 481-angle
  external/Beam24 hierarchy admission. No paired latency campaign was run there.

The changes address delivery and validation, not the beamforming algorithm.
The previous environment advertised Python 3.10 although the pinned NumPy and
SciPy packages require Python 3.12. A CUDA build could also configure as CPU-only
when its compiler was missing. Both boundaries now fail clearly.

The real-quality checker no longer reuses a 3% latency tolerance for correlation.
It checks ten distinct records, finite correlations, the peak-shift bound, and
the rounded correlation reference. The downloader keeps partial transfers
separate and verifies extracted recordings before analysis.

## Performance validation

The same-hierarchy external comparison reproduces on the RTX PRO 6000 Blackwell
Workstation Edition with CUDA 13.2.51 and driver 610.43.02. A source rebuild
passes the 481-angle correctness checks and measures **3.86076x** speedup
over ccglib across six alternating-order process pairs (95% CI
**[3.85819, 3.86346]**, **6/6 wins**). This is within 0.5% of the 3.87988x
reference and passes the documented 3% reproduction tolerance.

[Recorded samples](../evidence/validation/external_hierarchy_20260928/samples.jsonl)
and [measurement details](../evidence/validation/repository_20260914.json)
support this comparison. The original paper reference values are retained;
the Server Edition check above covers functional portability only.

The full finite-snapshot, perturbation/coverage, and Fourier campaigns retain
their original evidence and explicit reproduction targets. They are not
implicitly rerun by the CPU smoke or the ten-recording real-data target.
