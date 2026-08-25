#!/usr/bin/env python3
"""Verify canonical reference anchors against the checked-in result records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str):
    return json.loads((ROOT / path).read_text())


def check(label: str, measured: float, anchor: float, tolerance: float = 5e-5) -> None:
    if abs(measured - anchor) > tolerance:
        raise SystemExit(f"{label}: evidence={measured} anchor={anchor}")
    print(f"{label}: {measured:.6f} [OK]")


def check_reproduction(label: str, measured: float, anchor: float) -> None:
    ratio = measured / anchor
    if ratio < 0.97 or ratio > 1.03:
        raise SystemExit(
            f"{label}: reproduction={measured} anchor={anchor} ratio={ratio}"
        )
    print(f"{label}: {measured:.6f} vs {anchor:.6f} [within 3%]")


def main() -> None:
    anchors = load("artifact/expected/reference_anchors.json")
    operator = load("evidence/results/operator_sm120_batch256_m1024_n1024_k512.json")
    system = load("evidence/results/system_fair_dense_fused.json")
    hierarchy = load("evidence/results/system_hierarchical_sm120.json")
    same_hierarchy = load("evidence/results/system_external_same_hierarchy.json")
    quality = load("evidence/results/quality_spib48_vla.json")
    robustness = load("evidence/results/quality_regular_ula_robustness_top1.json")
    fft = load("evidence/results/system_fft_baseline_sm120.json")
    fft_same_grid = load("evidence/results/system_fourier_uniform_angle_sm120.json")
    second_shape = load("evidence/results/system_hierarchy_secondshape_sm120.json")
    heldout_real = load("evidence/results/quality_spib48_heldout_sessions.json")
    fft_repro = load(
        "evidence/validation/fft_lower_bound_reproduce_sm120_2026-08-24.json"
    )
    cold = load(
        "evidence/validation/hierarchy_headline_cold_reproduce_sm120_2026-08-24.json"
    )
    check(
        "operator speedup",
        operator["direct_dense_campaign"]["paired_speedup"],
        anchors["operator"]["sm120_batch256_m1024_n1024_k512_speedup"],
    )
    check(
        "internal dense-fused attribution",
        system["comparisons"]["dense_fused_over_beam24"]["paired_geomean_speedup"],
        anchors["system"]["same_output_dense_fused_speedup"],
    )
    check(
        "materialized pipeline speedup",
        system["comparisons"]["dense_materialized_over_beam24"]["paired_geomean_speedup"],
        anchors["system"]["materialized_ccglib_pipeline_speedup"],
    )
    check(
        "real VLA map correlation",
        quality["summary"]["Q2_localf4_joint_top2"]["mean_map_correlation"],
        anchors["quality"]["spib48_vla_mean_map_correlation"],
    )
    check(
        "hierarchical external system speedup",
        hierarchy["performance"]["external_ccglib_materialized_top1_direct"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_external_ccglib_speedup"],
    )
    check(
        "hierarchical external same-algorithm speedup",
        same_hierarchy["performance"]["external_ccglib_same_hierarchy"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_external_same_algorithm_speedup"],
    )
    check(
        "hierarchical exhaustive ablation",
        hierarchy["performance"]["exhaustive_beam24_ablation"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_exhaustive_ablation_speedup"],
    )
    check(
        "hierarchical dense attribution",
        hierarchy["performance"]["dense_fused_same_hierarchy_attribution"]
        ["paired_geomean_speedup"],
        anchors["system"]["hierarchical_dense_attribution_speedup"],
    )
    check(
        "regular ULA robustness local top1 minimum",
        robustness["practical_envelope"]["local_f4_exact_top1_min"],
        anchors["quality"]["regular_ula_robustness_local_top1_min"],
        tolerance=1e-12,
    )
    check(
        "regular ULA robustness local maximum shift",
        robustness["practical_envelope"]["local_f4_max_shift_deg"],
        anchors["quality"]["regular_ula_robustness_local_max_shift_deg"],
        tolerance=1e-12,
    )
    check(
        "regular ULA robustness hierarchy top1 minimum",
        robustness["practical_envelope"]["hierarchy_exact_top1_min"],
        anchors["quality"]["regular_ula_robustness_hierarchy_top1_min"],
        tolerance=1e-12,
    )
    check(
        "regular ULA robustness correlation advantage",
        robustness["practical_envelope"]["local_minus_direct_mean_correlation_min"],
        anchors["quality"]["regular_ula_robustness_local_minus_direct_correlation_min"],
        tolerance=1e-12,
    )
    check_reproduction(
        "cold hierarchy ablation reproduction",
        cold["performance"]["internal_hierarchy_ablation"]["paired_geomean_speedup"],
        anchors["system"]["hierarchical_exhaustive_ablation_speedup"],
    )
    check_reproduction(
        "cold external same-hierarchy reproduction",
        cold["performance"]["external_same_hierarchy"]["paired_geomean_speedup"],
        anchors["system"]["hierarchical_external_same_algorithm_speedup"],
    )
    check_reproduction(
        "cold external standard reproduction",
        cold["performance"]["external_standard_exhaustive"]["paired_geomean_speedup"],
        anchors["system"]["hierarchical_external_ccglib_speedup"],
    )
    check(
        "interpolated FFT SPIB correlation",
        fft["quality"]["cpu_interpolated_fft_spib48"]["mean_map_correlation"],
        anchors["quality"]["interpolated_fft_spib_mean_map_correlation"],
        tolerance=5e-9,
    )
    check(
        "optimistic cuFFT FP32 latency",
        fft["performance"]["cufft_fp32"]["median_ms"],
        anchors["system"]["optimistic_cufft_fp32_ms"],
        tolerance=5e-5,
    )
    if fft["quality"]["fp16_fft_transform"]["status"] != "rejected":
        raise SystemExit("FP16 cuFFT negative gate must remain rejected")
    check_reproduction(
        "cold cuFFT lower-bound reproduction",
        fft_repro["performance"]["cufft_fp32_median_ms"],
        anchors["system"]["optimistic_cufft_fp32_ms"],
    )
    check(
        "same-grid cuFFT latency",
        fft_same_grid["performance"]["cufft_uniform_angle_median_ms"],
        anchors["system"]["same_grid_cufft_fp32_ms"],
        tolerance=1e-9,
    )
    check(
        "same-grid cuFFT over Beam24",
        fft_same_grid["performance"]["paired_geomean_cufft_over_beam24"],
        anchors["system"]["same_grid_cufft_over_beam24"],
        tolerance=1e-12,
    )
    check(
        "Stage-1 static over basic",
        second_shape["hybrid_stage1_screen"]["paired_geomean_static_over_basic"],
        anchors["system"]["stage1_static_over_basic"],
        tolerance=1e-12,
    )
    for label, evidence_key, anchor_key in (
        ("second-shape external", "external_same_hierarchy_speedup", "second_shape_external_same_hierarchy_speedup"),
        ("second-shape global", "global_hierarchy_speedup", "second_shape_global_hierarchy_speedup"),
        ("second-shape local", "local_sparse_speedup", "second_shape_local_sparse_speedup"),
    ):
        check(
            label,
            second_shape["second_shape"][evidence_key],
            anchors["system"][anchor_key],
            tolerance=1e-12,
        )
    check(
        "held-out P2701 Local-F4 correlation",
        heldout_real["datasets"]["P2701_moving_170Hz"]["local_f4_mean_spectrum_correlation"],
        anchors["quality"]["heldout_p2701_local_map_correlation"],
        tolerance=1e-12,
    )
    check(
        "held-out A2601_2 Local-F4 correlation",
        heldout_real["datasets"]["A2601_2_350Hz"]["local_f4_mean_spectrum_correlation"],
        anchors["quality"]["heldout_a2601_2_local_map_correlation"],
        tolerance=1e-12,
    )
    check(
        "held-out real Local-F4 exact top1 minimum",
        min(
            heldout_real["datasets"]["P2701_moving_170Hz"]["local_f4_dense_exact_top1"],
            heldout_real["datasets"]["A2601_2_350Hz"]["local_f4_dense_exact_top1"],
        ),
        anchors["quality"]["heldout_real_local_exact_top1_min"],
        tolerance=1e-12,
    )
    check(
        "held-out real hierarchy exact top1 minimum",
        min(
            heldout_real["datasets"]["P2701_moving_170Hz"]["proportional_hierarchy_dense_exact_top1"],
            heldout_real["datasets"]["A2601_2_350Hz"]["proportional_hierarchy_dense_exact_top1"],
        ),
        anchors["quality"]["heldout_real_hierarchy_exact_top1_min"],
        tolerance=1e-12,
    )
    if fft_repro["correctness"]["fp16_status"] != "rejected":
        raise SystemExit("cold FP16 cuFFT rejection must remain preserved")


if __name__ == "__main__":
    main()
