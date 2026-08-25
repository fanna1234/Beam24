#!/usr/bin/env python3
"""Validate the frozen dense Acoular-64 beamforming oracle."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


SOURCE_XY = np.array([[-0.10, -0.10], [0.15, 0.00], [0.00, 0.10]])


def load_mics(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    positions = []
    for node in root.findall("pos"):
        positions.append([float(node.attrib[axis]) for axis in ("x", "y", "z")])
    result = np.asarray(positions, dtype=np.float64)
    if result.shape != (64, 3):
        raise ValueError(f"Expected 64 microphone positions, got {result.shape}")
    return result


def stft(data: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    nframes = 1 + (data.shape[0] - nfft) // hop
    starts = np.arange(nframes) * hop
    frames = np.stack([data[start : start + nfft] for start in starts], axis=0)
    frames *= np.hanning(nfft)[None, :, None]
    return np.fft.rfft(frames, axis=1)


def steering_weights(
    grid_xyz: np.ndarray, mic_xyz: np.ndarray, frequency: float, sound_speed: float
) -> np.ndarray:
    array_center = mic_xyz.mean(axis=0)
    r0 = np.linalg.norm(grid_xyz - array_center[None, :], axis=1)
    rm = np.linalg.norm(grid_xyz[:, None, :] - mic_xyz[None, :, :], axis=2)
    wave_number = 2.0 * np.pi * frequency / sound_speed
    transfer = (r0[:, None] / rm) * np.exp(-1j * wave_number * (rm - r0[:, None]))
    steer = transfer / np.sum(np.abs(transfer) ** 2, axis=1, keepdims=True)
    return steer.conj()


def select_peaks(power: np.ndarray, grid_xy: np.ndarray, count: int, radius: float) -> np.ndarray:
    available = np.ones(power.size, dtype=bool)
    selected = []
    for _ in range(count):
        masked = np.where(available, power, -np.inf)
        idx = int(np.argmax(masked))
        selected.append(idx)
        distance = np.linalg.norm(grid_xy - grid_xy[idx], axis=1)
        available &= distance > radius
    return np.asarray(selected, dtype=np.int64)


def greedy_match(estimated: np.ndarray, truth: np.ndarray) -> tuple[list[int], list[float]]:
    remaining = list(range(len(estimated)))
    assignment = []
    errors = []
    for target in truth:
        best = min(remaining, key=lambda idx: np.linalg.norm(estimated[idx] - target))
        assignment.append(best)
        errors.append(float(np.linalg.norm(estimated[best] - target)))
        remaining.remove(best)
    return assignment, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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

    power = np.zeros(grid_xy.shape[0], dtype=np.float64)
    for freq_bin in bins:
        weights = steering_weights(grid_xyz, mic_xyz, frequencies[freq_bin], 343.0)
        snapshots = spectrum[:, freq_bin, :].T
        output = weights @ snapshots
        power += np.sum(np.abs(output) ** 2, axis=1)

    peak_indices = select_peaks(power, grid_xy, count=3, radius=0.04)
    peaks = grid_xy[peak_indices]
    assignment, errors = greedy_match(peaks, SOURCE_XY)
    passed = max(errors) <= 0.02

    result = {
        "experiment": "beam24_20260823_acoular64_multisource_quality",
        "stage": "dense_oracle",
        "input_kind": "synthetic_acoustic_waveform",
        "channels": int(waveform.shape[1]),
        "samples": int(waveform.shape[0]),
        "sample_rate_hz": sample_rate,
        "nfft": nfft,
        "hop": hop,
        "frequency_bins_hz": [float(frequencies[index]) for index in bins],
        "grid_points": int(grid_xy.shape[0]),
        "peak_xy_m": peaks.tolist(),
        "peak_power_db_relative": (10.0 * np.log10(power[peak_indices] / power.max())).tolist(),
        "truth_to_peak_assignment": assignment,
        "truth_location_error_m": errors,
        "max_truth_location_error_m": max(errors),
        "gate_max_error_m": 0.02,
        "dense_oracle_pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
