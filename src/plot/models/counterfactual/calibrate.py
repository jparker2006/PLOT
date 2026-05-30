"""Fold-safe isotonic recalibration of the per-frame EPV trace, for counterfactual use.

The sequence eval bar is calibrated in-the-large and on the inverse-possession-weighted full frame
set (Gate G1), but on the specific dynamic states the counterfactual reads — a receiver catching
after a pass — it is slightly OVER-spread (slope ~0.83: extreme EPVs a touch too extreme). Regret
compares this EPV against xPoints, so a tail miscalibration would distort it. We fix it with a
monotone isotonic map EPV -> expected points, fit leave-one-game-out (so a game's frames are
recalibrated by a map that never saw them) — it preserves the EPV ORDERING (and thus G1's ranking)
while pulling the tails onto the diagonal.
"""

from __future__ import annotations

import polars as pl
from sklearn.isotonic import IsotonicRegression


def recalibrate_epv_oof(
    trace: pl.DataFrame,
    realized: pl.DataFrame,
    *,
    game_col: str = "game_id",
    epv_col: str = "epv",
    out_col: str = "epv_cal",
    min_fit: int = 500,
) -> pl.DataFrame:
    """Add ``out_col`` = isotonic-recalibrated EPV, fit fold-safe (per held-out game).

    ``trace``: per-frame OOF EPV (game_id, possession_id, ..., epv). ``realized``: one row per
    (game_id, possession_id) with a ``realized`` expected-points target. Frames whose possession
    has no realized target pass through uncalibrated.
    """
    tr = trace.join(realized, on=[game_col, "possession_id"], how="left")
    games = sorted(tr[game_col].unique().to_list())
    parts = []
    for held in games:
        fit = tr.filter((pl.col(game_col) != held) & pl.col("realized").is_not_null())
        held_df = tr.filter(pl.col(game_col) == held)
        if held_df.height == 0:
            continue
        if fit.height < min_fit:
            parts.append(held_df.with_columns(pl.col(epv_col).alias(out_col)))
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            fit[epv_col].to_numpy(), fit["realized"].to_numpy()
        )
        parts.append(held_df.with_columns(pl.Series(out_col, iso.predict(held_df[epv_col].to_numpy()))))
    return pl.concat(parts).drop("realized")
