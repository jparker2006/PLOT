"""LOGO fold helpers shared by the LightGBM baseline and the torch sequence model.

These are pure polars/numpy utilities with NO model dependency. Keeping them out of
``crossval.py`` (which imports LightGBM) lets the sequence pipeline reuse them WITHOUT pulling
LightGBM into a process that also loads torch — on macOS, loading both LightGBM's OpenMP runtime
and torch's bundled OpenMP in one process can segfault, so the two model backends must not meet.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from plot.features.eval_bar import GROUP_COLUMN, WEIGHT_COLUMN


def inner_val_game(train_ids: list[str], held: str) -> str:
    """Deterministic grouped watch game for early stopping: the next game after ``held`` (rotating)."""
    ordered = sorted(set(train_ids) | {held})
    i = ordered.index(held)
    for step in range(1, len(ordered)):
        cand = ordered[(i + step) % len(ordered)]
        if cand != held and cand in train_ids:
            return cand
    return train_ids[0]


def oof_arrays(oof: pl.DataFrame) -> dict:
    """Unpack a pooled OOF DataFrame into numpy arrays for the calibration metrics.

    Expects columns: epv, p0..p{K-1}, y, weight, game_id, possession_id (the shape both the
    baseline and the sequence model emit), so calibration scores them identically.
    """
    k = sum(c.startswith("p") and c[1:].isdigit() for c in oof.columns)
    probs = np.column_stack([oof[f"p{j}"].to_numpy() for j in range(k)])
    return {
        "epv": oof["epv"].to_numpy(),
        "probs": probs,
        "y": oof["y"].to_numpy(),
        "weight": oof[WEIGHT_COLUMN].to_numpy(),
        "game_id": oof[GROUP_COLUMN].to_numpy(),
        "possession_id": oof["possession_id"].to_numpy(),
    }
