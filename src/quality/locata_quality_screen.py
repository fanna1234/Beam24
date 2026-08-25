#!/usr/bin/env python3
"""Matched dense/sparse delay-and-sum screen on LOCATA Eigenmike recordings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

from cusparselt_prune import complex_matmul_fp16_output, prune_complex

GLOBAL_WEIGHT_CACHE: dict[bytes, dict[str, list[np.ndarray]]] = {}
LEGAL_MASKS = tuple((i, j) for i in range(4) for j in range(i + 1, 4))

def read_tsv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def absolute_seconds(table: dict[str, np.ndarray]) -> np.ndarray:
    result = np.empty(len(table["second"]), dtype=np.float64)
    for index in range(len(result)):
        day_index = date(
            int(table["year"][index]), int(table["month"][index]), int(table["day"][index])
        ).toordinal()
        result[index] = (
            day_index * 86400.0
            + table["hour"][index] * 3600.0
            + table["minute"][index] * 60.0
            + table["second"][index]
        )
    return result


def grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(np.arange(-180.0, 180.1, 5.0))
    elevation = np.deg2rad(np.arange(0.0, 180.1, 10.0))
    az_grid, el_grid = np.meshgrid(azimuth, elevation, indexing="ij")
    eta = np.column_stack(
        [
            -np.sin(el_grid.ravel()) * np.sin(az_grid.ravel()),
            np.sin(el_grid.ravel()) * np.cos(az_grid.ravel()),
            np.cos(el_grid.ravel()),
        ]
    )
    return az_grid.ravel(), el_grid.ravel(), eta


def dft4() -> np.ndarray:
    row = np.arange(4)[:, None]
    col = np.arange(4)[None, :]
    return np.exp(-2j * np.pi * row * col / 4.0) / 2.0


def keep_topk(values: np.ndarray, score: np.ndarray, count: int) -> np.ndarray:
    indices = np.argpartition(score, score.shape[-1] - count, axis=-1)[..., -count:]
    mask = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    return np.where(mask, values, 0.0)


def sparsify(weights: np.ndarray, variant: str) -> np.ndarray:
    groups = weights.reshape(weights.shape[0], -1, 4)
    if variant == "D0_dense":
        return weights.copy()
    if variant == "CUSPARSELT_PRUNE_STRIP":
        candidate = prune_complex(weights, "strip")
    elif variant == "CUSPARSELT_PRUNE_TILE":
        candidate = prune_complex(weights, "tile")
    elif variant == "Q1_direct_joint_2of4":
        candidate = keep_topk(groups, np.abs(groups), 2).reshape(weights.shape)
    else:
        transform = dft4()
        transformed = groups @ transform.conj().T
        if variant == "Q2_localf4_joint_top2":
            sparse = keep_topk(transformed, np.abs(transformed), 2)
        elif variant == "Q4_localf4_unstructured50":
            flat = transformed.reshape(weights.shape)
            sparse = keep_topk(flat[:, None, :], np.abs(flat)[:, None, :], flat.shape[1] // 2)
            sparse = sparse.reshape(transformed.shape)
        elif variant == "Q5_localf4_independent_ri_2of4":
            real = keep_topk(transformed.real, np.abs(transformed.real), 2)
            imag = keep_topk(transformed.imag, np.abs(transformed.imag), 2)
            sparse = real + 1j * imag
        else:
            raise ValueError(variant)
        candidate = (sparse @ transform).reshape(weights.shape)
    # Preserve distortionless gain for the candidate's own scan direction.
    gain = np.sum(candidate * weights.conj(), axis=1)
    dense_gain = np.sum(weights * weights.conj(), axis=1)
    return candidate * (dense_gain / gain)[:, None]


def response_dither3(weights: np.ndarray, steering: np.ndarray) -> np.ndarray:
    """Greedily repair poor transformed 2:4 beampatterns without using recordings."""
    transform = dft4()
    result = np.empty_like(weights)
    groups_count = weights.shape[1] // 4
    steering_groups = steering.reshape(steering.shape[0], groups_count, 4)
    min_snr_ratio = 10.0 ** (-1.0 / 10.0)
    for focus in range(weights.shape[0]):
        dense = weights[focus]
        dense_response = dense @ steering.T
        dense_magnitude = np.abs(dense_response)
        dense_magnitude /= max(dense_magnitude[focus], 1e-30)
        coeff = dense.reshape(groups_count, 4) @ transform.conj().T
        blocks = np.zeros((groups_count, len(LEGAL_MASKS), 4), dtype=np.complex128)
        energy = np.zeros((groups_count, len(LEGAL_MASKS)), dtype=np.float64)
        for option, pair in enumerate(LEGAL_MASKS):
            sparse = np.zeros_like(coeff)
            sparse[:, pair] = coeff[:, pair]
            blocks[:, option] = sparse @ transform
            energy[:, option] = np.sum(np.abs(sparse) ** 2, axis=1)
        option_response = np.einsum("gok,pgk->gop", blocks, steering_groups, optimize=True)
        assignment = np.argmax(energy, axis=1)
        group_index = np.arange(groups_count)
        response = np.sum(option_response[group_index, assignment], axis=0)
        raw_norm = float(np.sum(np.abs(blocks[group_index, assignment]) ** 2))
        dense_norm = float(np.sum(np.abs(dense) ** 2))
        changed = set()

        def metrics(value: np.ndarray, norm: float) -> tuple[float, float, complex]:
            gain = value[focus]
            magnitude = np.abs(value / gain)
            nmse = float(np.sum((magnitude - dense_magnitude) ** 2) / np.sum(dense_magnitude**2))
            snr = dense_norm * float(np.abs(gain) ** 2) / norm
            return nmse, snr, gain

        current_nmse, current_snr, gain = metrics(response, raw_norm)
        for _ in range(3):
            if current_nmse <= 0.01 and current_snr >= min_snr_ratio:
                break
            candidates = []
            for group in range(groups_count):
                if group in changed:
                    continue
                old = int(assignment[group])
                for option in range(len(LEGAL_MASKS)):
                    if option == old:
                        continue
                    candidate_response = response - option_response[group, old] + option_response[group, option]
                    candidate_norm = (
                        raw_norm
                        - float(np.sum(np.abs(blocks[group, old]) ** 2))
                        + float(np.sum(np.abs(blocks[group, option]) ** 2))
                    )
                    nmse, snr, candidate_gain = metrics(candidate_response, candidate_norm)
                    if snr >= min_snr_ratio:
                        candidates.append((nmse, -snr, group, option, candidate_response, candidate_norm, candidate_gain))
            if not candidates:
                break
            best = min(candidates, key=lambda item: item[:4])
            if best[0] >= current_nmse - 1e-12:
                break
            current_nmse, negative_snr, group, option, response, raw_norm, gain = best
            current_snr = -negative_snr
            assignment[group] = option
            changed.add(group)
        result[focus] = blocks[group_index, assignment].reshape(-1) / gain
    return result


def stft_official(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nfft = 1024
    window_length = int(round(0.03 * sample_rate))
    hop = window_length // 4
    prepad = window_length - hop
    postpad = hop - (signal.shape[0] % hop) + prepad
    padded = np.pad(signal, ((prepad, postpad), (0, 0)))
    starts = np.arange(0, padded.shape[0] - window_length + 1, hop)
    window = np.hamming(window_length)
    frames = np.stack([padded[start : start + window_length] * window[:, None] for start in starts])
    spectrum = np.fft.rfft(frames, n=nfft, axis=1)
    timestamps = (starts - prepad + nfft / 2.0) / sample_rate
    frequencies = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    return spectrum, timestamps, frequencies


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def process_recording(directory: Path, max_blocks: int) -> list[dict]:
    audio_path = directory / "audio_array_eigenmike.wav"
    signal, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if signal.shape[1] != 32:
        raise ValueError(f"Expected 32 channels, got {signal.shape}")
    spectrum, frame_times, frequencies = stft_official(signal, sample_rate)
    freq_indices = np.flatnonzero((frequencies > 800.0) & (frequencies < 1400.0))

    vad_path = next(directory.glob("VAD_eigenmike_*.txt"))
    vad = np.loadtxt(vad_path, skiprows=1, dtype=np.float64)
    audio_time = read_tsv(directory / "audio_array_timestamps_eigenmike.txt")
    audio_start = absolute_seconds(audio_time)[0]
    array_table = read_tsv(directory / "position_array_eigenmike.txt")
    source_table = read_tsv(next(directory.glob("position_source_*.txt")))
    array_times = absolute_seconds(array_table)
    source_times = absolute_seconds(source_table)

    azimuth_grid, elevation_grid, eta_grid = grid()
    variants = [
        "D0_dense",
        "CUSPARSELT_PRUNE_STRIP",
        "CUSPARSELT_PRUNE_TILE",
        "Q1_direct_joint_2of4",
        "Q2_localf4_joint_top2",
        "Q3_response_dither3",
        "Q4_localf4_unstructured50",
        "Q5_localf4_independent_ri_2of4",
    ]

    frame_starts = np.arange(0, len(frame_times), 10)
    frame_ends = np.minimum(frame_starts + 99, len(frame_times) - 1)
    candidates = []
    for start, end in zip(frame_starts, frame_ends):
        sample_start = max(0, int(frame_times[start] * sample_rate))
        sample_end = min(len(vad), int(frame_times[end] * sample_rate))
        active = float(vad[sample_start:sample_end].mean()) if sample_end > sample_start else 0.0
        if active >= 0.5:
            candidates.append((int(start), int(end), active))
    if not candidates:
        return []
    selected_indices = np.linspace(0, len(candidates) - 1, min(max_blocks, len(candidates)), dtype=int)
    selected = [candidates[index] for index in selected_indices]

    records = []
    for block_number, (start, end, active) in enumerate(selected):
        relative_time = 0.5 * (frame_times[start] + frame_times[end])
        absolute_time = audio_start + relative_time
        array_index = int(np.argmin(np.abs(array_times - absolute_time)))
        source_index = int(np.argmin(np.abs(source_times - absolute_time)))
        rotation = np.asarray(
            [[array_table[f"rotation_{row}{col}"][array_index] for col in range(1, 4)] for row in range(1, 4)]
        )
        array_position = np.asarray([array_table[axis][array_index] for axis in ("x", "y", "z")])
        source_position = np.asarray([source_table[axis][source_index] for axis in ("x", "y", "z")])
        microphones = np.asarray(
            [[array_table[f"mic{mic}_{axis}"][array_index] for axis in ("x", "y", "z")] for mic in range(1, 33)]
        )
        local_source = rotation.T @ (source_position - array_position)
        radius = np.linalg.norm(local_source)
        truth_elevation = math.acos(float(local_source[2] / radius))
        truth_azimuth = wrap_angle(math.atan2(float(local_source[1]), float(local_source[0])) - math.pi / 2.0)
        truth_eta = np.asarray(
            [
                -math.sin(truth_elevation) * math.sin(truth_azimuth),
                math.sin(truth_elevation) * math.cos(truth_azimuth),
                math.cos(truth_elevation),
            ]
        )

        local_mics = (rotation.T @ (microphones - microphones[0]).T).T
        cache_key = np.round(local_mics, 6).tobytes()
        if cache_key not in GLOBAL_WEIGHT_CACHE:
            tau = eta_grid @ local_mics.T / 340.0
            cache = {variant: [] for variant in variants}
            for fi in freq_indices:
                steering = np.exp(1j * 2.0 * np.pi * frequencies[fi] * tau)
                dense_weights = steering.conj() / 32.0
                for variant in variants:
                    if variant == "Q3_response_dither3":
                        cache[variant].append(response_dither3(dense_weights, steering))
                    else:
                        cache[variant].append(sparsify(dense_weights, variant))
            GLOBAL_WEIGHT_CACHE[cache_key] = cache
        weights = GLOBAL_WEIGHT_CACHE[cache_key]

        maps = {variant: np.zeros(len(eta_grid), dtype=np.float64) for variant in variants}
        for local_fi, fi in enumerate(freq_indices):
            snapshots = spectrum[start : end + 1, fi, :].T
            for variant in variants:
                output = (
                    complex_matmul_fp16_output(weights[variant][local_fi], snapshots)
                    if variant.startswith("CUSPARSELT_")
                    else weights[variant][local_fi] @ snapshots
                )
                maps[variant] += np.sum(np.abs(output) ** 2, axis=1)

        dense_index = int(np.argmax(maps["D0_dense"]))
        dense_error = angular_error_deg(eta_grid[dense_index], truth_eta)
        admitted = dense_error <= 15.0
        dense_norm = maps["D0_dense"] / np.max(maps["D0_dense"])
        for variant in variants:
            peak = int(np.argmax(maps[variant]))
            candidate_norm = maps[variant] / np.max(maps[variant])
            records.append(
                {
                    "task": directory.parents[1].name,
                    "recording": directory.parent.name,
                    "block": block_number,
                    "relative_time_s": relative_time,
                    "vad_fraction": active,
                    "variant": variant,
                    "admitted": admitted,
                    "dense_error_deg": dense_error,
                    "error_deg": angular_error_deg(eta_grid[peak], truth_eta),
                    "delta_error_deg": angular_error_deg(eta_grid[peak], truth_eta) - dense_error,
                    "dense_agreement": peak == dense_index,
                    "shift_from_dense_deg": angular_error_deg(eta_grid[peak], eta_grid[dense_index]),
                    "peak_delta_db_at_dense": 10.0
                    * math.log10(max(maps[variant][dense_index], 1e-300) / max(maps["D0_dense"][dense_index], 1e-300)),
                    "map_correlation": float(np.corrcoef(dense_norm, candidate_norm)[0, 1]),
                    "truth_azimuth_deg": math.degrees(truth_azimuth),
                    "truth_elevation_deg": math.degrees(truth_elevation),
                    "estimated_azimuth_deg": math.degrees(azimuth_grid[peak]),
                    "estimated_elevation_deg": math.degrees(elevation_grid[peak]),
                }
            )
    return records


def summarize(records: list[dict]) -> dict:
    admitted = [row for row in records if row["admitted"]]
    variants = sorted({row["variant"] for row in records})
    result = {}
    for variant in variants:
        rows = [row for row in admitted if row["variant"] == variant]
        result[variant] = {
            "blocks": len(rows),
            "dense_agreement_rate": float(np.mean([row["dense_agreement"] for row in rows])) if rows else None,
            "mean_error_deg": float(np.mean([row["error_deg"] for row in rows])) if rows else None,
            "mean_delta_error_deg": float(np.mean([row["delta_error_deg"] for row in rows])) if rows else None,
            "max_shift_from_dense_deg": float(np.max([row["shift_from_dense_deg"] for row in rows])) if rows else None,
            "mean_map_correlation": float(np.mean([row["map_correlation"] for row in rows])) if rows else None,
            "worst_peak_delta_db_at_dense": float(np.min([row["peak_delta_db_at_dense"] for row in rows])) if rows else None,
        }
    unique_blocks = {(row["task"], row["recording"], row["block"]) for row in records}
    admitted_blocks = {(row["task"], row["recording"], row["block"]) for row in admitted}
    return {
        "input_kind": "real_reverberant_locata",
        "blocks_total": len(unique_blocks),
        "blocks_dense_admitted": len(admitted_blocks),
        "dense_admission_rate": len(admitted_blocks) / len(unique_blocks) if unique_blocks else 0.0,
        "variants": result,
        "note": "Q3 uses at most three signal-independent K4 mask substitutions to reduce steering-response magnitude NMSE under a 1 dB noise-gain constraint.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-blocks-per-recording", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    directories = []
    for task in ("task1", "task3"):
        directories.extend(sorted((args.data_root / task).glob("recording*/eigenmike")))
    records = []
    for directory in directories:
        print(f"processing {directory}", flush=True)
        records.extend(process_recording(directory, args.max_blocks_per_recording))
    result = {
        "experiment": "beam24_20260823_locata_eigenmike_realdata",
        "contract": {
            "tasks": [1, 3],
            "recordings": len(directories),
            "max_blocks_per_recording": args.max_blocks_per_recording,
            "nfft": 1024,
            "window_ms": 30,
            "hop_fraction": 0.25,
            "frequency_hz": [800, 1400],
            "azimuth_step_deg": 5,
            "elevation_step_deg": 10,
            "dense_admission_error_deg": 15,
        },
        "summary": summarize(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.details.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
