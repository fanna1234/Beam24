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

## Primary-card timing follow-up

The primary workstation was occupied during the September 14 source update.
On September 28, a clean export of revision `6c37b7555c85c20afd7f66cbf550788f60d00ade`
completed `./reproduce.sh external-hierarchy` on the RTX PRO 6000 Blackwell
Workstation Edition with CUDA 13.2.51 and driver 610.43.02. Both implementations
were rebuilt from source using the clean, pinned dependency cache. The
481-angle output admissions passed before six fresh, alternating-order process
pairs were measured under the existing GPU lock and idle checks.

The paired geometric-mean speedup was **3.86076x**, with 95% process-bootstrap
CI **[3.85819, 3.86346]** and **6/6 wins**. Median complete-pipeline latencies
were 1.816874 ms for ccglib and 0.470576 ms for Beam24. The speedup is 0.493%
below the frozen 3.87988x reference and passes the existing 3% tolerance gate
with status `[~within3%]`.

The [raw process samples](../evidence/validation/external_hierarchy_20260928/samples.jsonl),
[summary](../evidence/validation/external_hierarchy_20260928/summary.json),
[output admission and binary identities](../evidence/validation/external_hierarchy_20260928/admission.json),
and [environment](../evidence/validation/external_hierarchy_20260928/environment.txt)
are retained alongside the updated
[validation record](../evidence/validation/repository_20260914.json).

This completes the primary-card timing check for the current same-hierarchy
external-comparison runner. The frozen paper results remain unchanged; the
10.83733x exhaustive-external comparison and internal ablations were not
remeasured in this follow-up. The earlier Server Edition / CUDA 13.1 run remains
a separate functional portability check.

The full finite-snapshot, perturbation/coverage, and Fourier campaigns retain
their original evidence and explicit reproduction targets. They are not
implicitly rerun by the CPU smoke or the ten-recording real-data target.
