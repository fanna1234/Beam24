"""CPU oracle for NVIDIA cuSPARSELt FP16 strip/tile pruning."""

from __future__ import annotations

import itertools

import numpy as np


PAIRS = tuple(itertools.combinations(range(4), 2))
TILE_MASKS = np.asarray(
    [
        [[column in PAIRS[choice] for column in range(4)] for choice in choices]
        for choices in itertools.product(range(len(PAIRS)), repeat=4)
        if all(sum(column in PAIRS[choice] for choice in choices) == 2 for column in range(4))
    ],
    dtype=bool,
)


def prune_strip_real(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    if columns % 4:
        raise ValueError("strip pruning requires K divisible by four")
    groups = values.reshape(rows, columns // 4, 4)
    indices = np.argpartition(np.abs(groups), kth=2, axis=-1)[..., 2:]
    mask = np.zeros(groups.shape, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    return np.where(mask, groups, 0.0).reshape(values.shape)


def prune_tile_real(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    if columns % 4:
        raise ValueError("tile pruning requires K divisible by four")
    padded_rows = (rows + 3) // 4 * 4
    padded = np.zeros((padded_rows, columns), dtype=values.dtype)
    padded[:rows] = values
    tiles = padded.reshape(padded_rows // 4, 4, columns // 4, 4).transpose(0, 2, 1, 3)
    scores = np.einsum("rcij,pij->rcp", np.abs(tiles), TILE_MASKS, optimize=True)
    choices = np.argmax(scores, axis=-1)
    selected = TILE_MASKS[choices]
    pruned = np.where(selected, tiles, 0.0).transpose(0, 2, 1, 3).reshape(padded.shape)
    return pruned[:rows]


def prune_complex(weights: np.ndarray, algorithm: str) -> np.ndarray:
    if weights.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got {weights.shape}")
    if algorithm == "strip":
        prune = prune_strip_real
    elif algorithm == "tile":
        prune = prune_tile_real
    else:
        raise ValueError(f"unknown cuSPARSELt pruning algorithm: {algorithm}")
    return prune(weights.real) + 1j * prune(weights.imag)


def complex_matmul_fp16_output(weights: np.ndarray, snapshots: np.ndarray) -> np.ndarray:
    """Emulate four FP16-input/FP32-compute cuSPARSELt calls with FP16 C/D."""
    wr = weights.real.astype(np.float16).astype(np.float32)
    wi = weights.imag.astype(np.float16).astype(np.float32)
    max_input = float(max(np.max(np.abs(snapshots.real)), np.max(np.abs(snapshots.imag))))
    scale = 1.0 if max_input <= 1.0 else float(2.0 ** np.ceil(np.log2(max_input)))
    xr = (snapshots.real / scale).astype(np.float16).astype(np.float32)
    xi = (snapshots.imag / scale).astype(np.float16).astype(np.float32)
    real_first = (wr @ xr).astype(np.float16).astype(np.float32)
    real = (real_first - wi @ xi).astype(np.float16).astype(np.float32)
    imag_first = (wr @ xi).astype(np.float16).astype(np.float32)
    imag = (imag_first + wi @ xr).astype(np.float16).astype(np.float32)
    return scale * (real + 1j * imag)
