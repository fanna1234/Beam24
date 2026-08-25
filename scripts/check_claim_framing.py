#!/usr/bin/env python3
"""Check Beam24 performance numbers and comparator roles across maintained text."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text())


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"{label}: missing required text: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise SystemExit(f"{label}: stale framing remains: {needle}")


def close(actual: float, expected: float, label: str, tolerance: float = 5e-5) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise SystemExit(f"{label}: {actual} != {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    anchors = load("artifact/expected/reference_anchors.json")
    operator = load("evidence/results/operator_sm120_batch256_m1024_n1024_k512.json")
    system = load("evidence/results/system_fair_dense_fused.json")
    hierarchical = load("evidence/results/system_hierarchical_sm120.json")
    same_hierarchy = load("evidence/results/system_external_same_hierarchy.json")
    multik = load("evidence/results/system_multik_sm120.json")
    workspace = load("evidence/results/system_workspace_elision.json")
    fourier = load("evidence/results/system_fourier_streamed_top1_sm120.json")
    margin = load("evidence/results/quality_hierarchy_margin_coverage_heldout.json")
    manifest = load("baselines/manifest.json")

    expected_roles = {
        "primary_external_operator": "operator.sm120_batch256_m1024_n1024_k512_speedup",
        "exhaustive_external_system_control": "system.materialized_ccglib_pipeline_speedup",
        "exhaustive_internal_attribution": "system.same_output_dense_fused_speedup",
        "primary_hierarchical_external_system": "system.hierarchical_external_ccglib_speedup",
        "primary_hierarchical_same_algorithm_external": "system.hierarchical_external_same_algorithm_speedup",
        "hierarchy_algorithm_ablation": "system.hierarchical_exhaustive_ablation_speedup",
        "hierarchy_sparse_attribution": "system.hierarchical_dense_attribution_speedup",
    }
    if anchors.get("claim_roles") != expected_roles:
        raise SystemExit("reference-anchor claim roles do not match the frozen taxonomy")

    close(
        operator["direct_dense_campaign"]["paired_speedup"],
        anchors["operator"]["sm120_batch256_m1024_n1024_k512_speedup"],
        "primary external operator",
    )
    close(
        system["comparisons"]["dense_materialized_over_beam24"]["paired_geomean_speedup"],
        anchors["system"]["materialized_ccglib_pipeline_speedup"],
        "primary external system",
    )
    close(
        system["comparisons"]["dense_fused_over_beam24"]["paired_geomean_speedup"],
        anchors["system"]["same_output_dense_fused_speedup"],
        "internal dense-fused attribution",
    )
    close(
        hierarchical["performance"]["external_ccglib_materialized_top1_direct"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_external_ccglib_speedup"],
        "hierarchical external system",
    )
    close(
        same_hierarchy["performance"]["external_ccglib_same_hierarchy"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_external_same_algorithm_speedup"],
        "hierarchical external same-algorithm system",
    )
    close(
        hierarchical["performance"]["exhaustive_beam24_ablation"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_exhaustive_ablation_speedup"],
        "hierarchy algorithm ablation",
    )
    close(
        hierarchical["performance"]["dense_fused_same_hierarchy_attribution"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_dense_attribution_speedup"],
        "hierarchy sparse attribution",
    )
    close(
        fourier["performance"]["paired_geomean_cufft_over_beam24"],
        anchors["system"]["streamed_cufft_over_beam24"],
        "streamed Fourier control",
        tolerance=1e-12,
    )
    close(
        margin["admitted"]["recall"]["top4"],
        anchors["quality"]["hierarchy_margin_admitted_top4_recall"],
        "hierarchy margin admitted top4 recall",
        tolerance=1e-12,
    )
    close(
        multik["external_system"]["shapes"]["512"]["paired_speedup"],
        2.3666631961301055,
        "multi-K external system replication",
        tolerance=1e-12,
    )
    if workspace["workspace"]["current_complex_workspace_bytes"] != 0:
        raise SystemExit("production complex workspace must be zero")
    if workspace["workspace"]["previous_complex_workspace_bytes"] != 2_147_483_648:
        raise SystemExit("unexpected historical complex workspace size")
    source_pairs = {
        "beam24": ROOT / "src/cuda/beam24_system.cu",
        "dense_fused": ROOT / "src/cuda/dense_fused_baseline.cu",
    }
    for key, path in source_pairs.items():
        recorded = workspace[key]["source_sha256"]
        current = sha256(path)
        if current != recorded:
            raise SystemExit(f"{key} source hash drift: {current} != {recorded}")
    hierarchy_source = ROOT / "src/cuda/beam24_hierarchical_system.cu"
    hierarchy_recorded = hierarchical["provenance"]["hierarchical_source_sha256"]
    hierarchy_current = sha256(hierarchy_source)
    if hierarchy_current != hierarchy_recorded:
        raise SystemExit(
            "hierarchical source hash drift: "
            f"{hierarchy_current} != {hierarchy_recorded}"
        )
    external_hierarchy_source = (
        ROOT / "baselines/cuda/ccglib_hierarchical_top1.cu"
    )
    external_hierarchy_recorded = same_hierarchy["provenance"][
        "baseline_source_sha256"
    ]
    external_hierarchy_current = sha256(external_hierarchy_source)
    if external_hierarchy_current != external_hierarchy_recorded:
        raise SystemExit(
            "external hierarchy source hash drift: "
            f"{external_hierarchy_current} != "
            f"{external_hierarchy_recorded}"
        )
    fourier_source = ROOT / "baselines/cuda/cufft_streamed_uniform_angle_top1.cu"
    fourier_header = ROOT / "baselines/cuda/cufft_top1_common.cuh"
    if sha256(fourier_source) != fourier["provenance"]["streamed_source_sha256"]:
        raise SystemExit("streamed Fourier source hash drift")
    if sha256(fourier_header) != fourier["provenance"]["streamed_common_header_sha256"]:
        raise SystemExit("streamed Fourier common-header hash drift")
    margin_source = ROOT / "src/quality/hierarchy_margin_coverage.py"
    if sha256(margin_source) != margin["provenance"]["maintained_source_sha256"]:
        raise SystemExit("hierarchy margin source hash drift")

    by_id = {entry["id"]: entry for entry in manifest["baselines"]}
    external = by_id["CCGLIB_MATERIALIZED_TOP1"]
    external_hierarchy = by_id["CCGLIB_HIERARCHICAL_TOP1"]
    external_fourier = by_id["CUFFT_STREAMED_UNIFORM_ANGLE_TOP1"]
    attribution = by_id["DENSE_FUSED_TOP1_CONTROL"]
    if (external["ownership"], external["table_role"]) != ("external", "main"):
        raise SystemExit("ccglib materialized-top1 must remain an external main row")
    if (
        external_hierarchy["ownership"],
        external_hierarchy["table_role"],
    ) != ("external", "main"):
        raise SystemExit(
            "ccglib hierarchical-top1 must remain an external main row"
        )
    if (external_fourier["ownership"], external_fourier["table_role"]) != (
        "external",
        "main",
    ):
        raise SystemExit("streamed Fourier top-1 must remain an external main row")
    if (attribution["ownership"], attribution["table_role"]) != (
        "internal_control",
        "attribution_control",
    ):
        raise SystemExit("dense fused must remain an internal attribution control")
    hierarchy = system["decision"]
    if hierarchy.get("primary_external_comparison") != "dense_materialized_over_beam24":
        raise SystemExit("system evidence lost the primary external comparison role")
    if hierarchy.get("internal_attribution_control") != "dense_fused_over_beam24":
        raise SystemExit("system evidence lost the internal attribution role")

    required_text = {
        "README.md": (
            "**External:** materialized ccglib pipeline",
            "**Attribution:** internal dense-fused control",
            "**2.37665x**",
            "**1.50462x**",
            "**10.837x**",
            "**3.880x**",
            "**4.565x**",
            "**1.160x**",
            "16.498x",
            "64 MiB",
            "97.539%/99.679%/100%",
        ),
        "docs/CLAIMS.md": (
            "SYSTEM-EXTERNAL",
            "measured exhaustive-path external control",
            "SYSTEM-ATTRIBUTION",
            "measured exhaustive-path internal ablation",
            "HIERARCHY-EXTERNAL",
            "HIERARCHY-EXTERNAL-SAME-ALGORITHM",
            "HIERARCHY-ABLATION",
            "HIERARCHY-ATTRIBUTION",
            "HIER-QUALITY-ROBUST-TOP1",
            "HIER-MARGIN-COVERAGE",
            "The parent joint-spectrum gate was rejected",
            "HIERARCHY-HEADLINE-REPRO",
            "FFT-STREAMED-TOP1-K512",
            "HIERARCHY-SECOND-SHAPE",
            "QUALITY-HELDOUT-REAL-LOCAL",
            "QUALITY-HELDOUT-REAL-HIERARCHY",
        ),
        "docs/CLAIM_TAXONOMY.md": (
            "The primary system result is the hierarchical top-1 path.",
            "remains an exhaustive-path external control",
            "2.58311 / 3.79739 ms",
            "2.14762 / 5.10418 ms",
            "2.14762 / 3.23090 ms",
            "0 / 2,147,483,648 B",
            "Fourier-family lower bound",
            "16.498x",
            "64 MiB",
            "coverage rank of 37",
            "proportional K48 hierarchy fails",
        ),
        "baselines/README.md": (
            "internal attribution control",
            "not an external baseline",
            "generally applicable benefit of avoiding",
        ),
    }
    for path, needles in required_text.items():
        text = (ROOT / path).read_text()
        for needle in needles:
            require(text, needle, path)

    stale_phrases = (
        "primary same-output system claim",
        "论文系统headline应使用1.505x",
        "前者是primary same-output system claim",
        "2.377x materialized-pipeline comparison is secondary",
    )
    maintained = [
        ROOT / "README.md",
        ROOT / "docs/CLAIMS.md",
        ROOT / "docs/CLAIM_TAXONOMY.md",
        ROOT / "artifact/README.md",
    ]
    for candidate in maintained:
        text = candidate.read_text()
        for stale in stale_phrases:
            forbid(text, stale, str(candidate.relative_to(ROOT)))

    print(
        "claim framing: exhaustive external control=2.37665x, external operator=1.47097x, "
        "exhaustive internal attribution=1.50462x, hierarchical external=10.83733x, "
        "hierarchical same-algorithm external=3.87988x, "
        "hierarchy ablation=4.56517x, hierarchy sparse attribution=1.16019x, "
        "streamed Fourier=16.49824x, workspace=0 B [OK]"
    )


if __name__ == "__main__":
    main()
