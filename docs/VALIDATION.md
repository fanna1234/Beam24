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

## Measurement boundary

Fresh GPU execution and timing are recorded separately from build success.
The primary workstation was occupied by unrelated work during this update;
the existing performance evidence is not relabeled as a new measurement.
The Server Edition / CUDA 13.1 run is a functional portability check, not a
replacement for the Workstation Edition / CUDA 13.2 performance reference.
The [machine-readable validation record](../evidence/validation/repository_20260914.json)
records this distinction. The primary-card paired timing run remains pending
because of unrelated GPU work; no process was stopped or preempted.

The full finite-snapshot, perturbation/coverage, and Fourier campaigns retain
their original evidence and explicit reproduction targets. They are not
implicitly rerun by the CPU smoke or the ten-recording real-data target.
