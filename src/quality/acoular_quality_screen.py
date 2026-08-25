#!/usr/bin/env python3
"""Run the frozen Acoular-64 dense and sparsification baseline screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from acoular_dense_oracle import SOURCE_XY, greedy_match, load_mics, select_peaks, steering_weights, stft
from cusparselt_prune import complex_matmul_fp16_output, prune_complex


def dft4() -> np.ndarray:
    row = np.arange(4)[:, None]
    col = np.arange(4)[None, :]
    return np.exp(-2j * np.pi * row * col / 4.0) / 2.0


def keep_topk(values: np.ndarray, score: np.ndarray, count: int) -> np.ndarray:
    indices = np.argpartition(score, kth=score.shape[-1] - count, axis=-1)[..., -count:]
    mask = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    return np.where(mask, values, 0.0)


def sparsify(weights: np.ndarray, variant: str) -> np.ndarray:
    groups = weights.reshape(weights.shape[0], -1, 4)
    transform = dft4()
    if variant == "D0_dense":
        return weights.copy()
    if variant == "CUSPARSELT_PRUNE_STRIP":
        return prune_complex(weights, "strip")
    if variant == "CUSPARSELT_PRUNE_TILE":
        return prune_complex(weights, "tile")
    if variant == "Q1_direct_joint_2of4":
        return keep_topk(groups, np.abs(groups), 2).reshape(weights.shape)

    transformed = groups @ transform.conj().T
    if variant == "Q2_localf4_joint_top2":
        sparse = keep_topk(transformed, np.abs(transformed), 2)
    elif variant == "Q4_localf4_unstructured50":
        flat = transformed.reshape(weights.shape)
        sparse = keep_topk(
            flat.reshape(flat.shape[0], 1, flat.shape[1]),
            np.abs(flat).reshape(flat.shape[0], 1, flat.shape[1]),
            flat.shape[1] // 2,
        ).reshape(transformed.shape)
    elif variant == "Q5_localf4_independent_ri_2of4":
        sparse_real = keep_topk(transformed.real, np.abs(transformed.real), 2)
        sparse_imag = keep_topk(transformed.imag, np.abs(transformed.imag), 2)
        sparse = sparse_real + 1j * sparse_imag
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return (sparse @ transform).reshape(weights.shape)


def distortionless_normalize(weights: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    gain = np.sum(weights * transfer, axis=1)
    return weights / gain[:, None]


def metric_summary(
    name: str,
    power: np.ndarray,
    dense_power: np.ndarray,
    grid_xy: np.ndarray,
    off_source: np.ndarray,
    noise_gain_db: list[float],
) -> dict:
    peak_indices = select_peaks(power, grid_xy, count=3, radius=0.04)
    peaks = grid_xy[peak_indices]
    assignment, truth_errors = greedy_match(peaks, SOURCE_XY)
    dense_peaks = select_peaks(dense_power, grid_xy, count=3, radius=0.04)
    dense_peak_xy = grid_xy[dense_peaks]
    _, dense_shift = greedy_match(peaks, dense_peak_xy)

    truth_indices = [int(np.argmin(np.linalg.norm(grid_xy - point, axis=1))) for point in SOURCE_XY]
    peak_loss_db = 10.0 * np.log10(
        np.maximum(power[truth_indices], 1e-300) / np.maximum(dense_power[truth_indices], 1e-300)
    )
    dense_psll = 10.0 * np.log10(np.max(dense_power[off_source]) / np.max(dense_power))
    candidate_psll = 10.0 * np.log10(np.max(power[off_source]) / np.max(power))
    dense_norm = dense_power / np.max(dense_power)
    candidate_norm = power / np.max(power)
    correlation = float(np.corrcoef(dense_norm, candidate_norm)[0, 1])
    nmse = float(np.sum((candidate_norm - dense_norm) ** 2) / np.sum(dense_norm**2))

    return {
        "variant": name,
        "peak_xy_m": peaks.tolist(),
        "peak_power_db_relative": (10.0 * np.log10(power[peak_indices] / np.max(power))).tolist(),
        "truth_to_peak_assignment": assignment,
        "truth_location_error_m": truth_errors,
        "max_truth_location_error_m": max(truth_errors),
        "max_peak_shift_vs_dense_m": max(dense_shift),
        "source_peak_delta_db_vs_dense": peak_loss_db.tolist(),
        "worst_source_peak_loss_db": float(np.min(peak_loss_db)),
        "psll_db": float(candidate_psll),
        "psll_delta_db_vs_dense": float(candidate_psll - dense_psll),
        "map_correlation": correlation,
        "map_nmse": nmse,
        "mean_noise_gain_delta_db": float(np.mean(noise_gain_db)),
        "worst_noise_gain_delta_db": float(np.max(noise_gain_db)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.data, "r") as handle:
        waveform = np.asarray(handle["time_data"], dtype=np.float64)
        sample_rate = float(handle["time_data"].attrs["sample_freq"])
    mic_xyz = load_mics(args.geometry)

    nfft = 128
    hop = 64
    spectrum = stft(waveform, nfft=nfft, hop=hop)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    lower = 8000.0 * 2.0 ** (-1.0 / 6.0)
    upper = 8000.0 * 2.0 ** (+1.0 / 6.0)
    bins = np.flatnonzero((frequencies >= lower) & (frequencies < upper))

    axis = np.arange(-0.20, 0.2001, 0.01)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_xyz = np.column_stack([grid_xy, np.full(grid_xy.shape[0], -0.30)])
    off_source = np.ones(grid_xy.shape[0], dtype=bool)
    for point in SOURCE_XY:
        off_source &= np.linalg.norm(grid_xy - point, axis=1) > 0.04

    variants = [
        "D0_dense",
        "CUSPARSELT_PRUNE_STRIP",
        "CUSPARSELT_PRUNE_TILE",
        "Q1_direct_joint_2of4",
        "Q2_localf4_joint_top2",
        "Q4_localf4_unstructured50",
        "Q5_localf4_independent_ri_2of4",
    ]
    maps = {name: np.zeros(grid_xy.shape[0], dtype=np.float64) for name in variants}
    noise_gain = {name: [] for name in variants}

    for freq_bin in bins:
        frequency = frequencies[freq_bin]
        transfer_weights = steering_weights(grid_xyz, mic_xyz, frequency, 343.0)

        array_center = mic_xyz.mean(axis=0)
        r0 = np.linalg.norm(grid_xyz - array_center[None, :], axis=1)
        rm = np.linalg.norm(grid_xyz[:, None, :] - mic_xyz[None, :, :], axis=2)
        wave_number = 2.0 * np.pi * frequency / 343.0
        transfer = (r0[:, None] / rm) * np.exp(-1j * wave_number * (rm - r0[:, None]))
        snapshots = spectrum[:, freq_bin, :].T
        dense_noise_norm = np.sum(np.abs(transfer_weights) ** 2, axis=1)

        for name in variants:
            candidate = sparsify(transfer_weights, name)
            candidate = distortionless_normalize(candidate, transfer)
            output = (
                complex_matmul_fp16_output(candidate, snapshots)
                if name.startswith("CUSPARSELT_")
                else candidate @ snapshots
            )
            maps[name] += np.sum(np.abs(output) ** 2, axis=1)
            ratio = np.sum(np.abs(candidate) ** 2, axis=1) / dense_noise_norm
            noise_gain[name].extend((10.0 * np.log10(ratio)).tolist())

    dense_power = maps["D0_dense"]
    summaries = [
        metric_summary(name, maps[name], dense_power, grid_xy, off_source, noise_gain[name])
        for name in variants
    ]
    result = {
        "experiment": "beam24_20260823_acoular64_multisource_quality",
        "stage": "baseline_screen_without_q3",
        "input_kind": "synthetic_acoustic_waveform",
        "frequency_bins_hz": [float(frequencies[index]) for index in bins],
        "variants": summaries,
        "note": "Q3 greedy-dither3 is not included in this screen and remains unknown.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.maps, grid_xy=grid_xy, **maps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
