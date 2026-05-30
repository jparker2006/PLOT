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
    n_folds: int = 5,
    max_fit_points: int = 1_000_000,
    seed: int = 1729,
) -> pl.DataFrame:
    """Add ``out_col`` = isotonic-recalibrated EPV, fit leakage-free via group k-fold.

    ``trace``: per-frame OOF EPV (game_id, possession_id, ..., epv). ``realized``: one row per
    (game_id, possession_id) with a ``realized`` expected-points target. Frames whose possession
    has no realized target pass through uncalibrated.

    Games are partitioned into ``n_folds`` round-robin groups; each group's frames are recalibrated
    by an isotonic map fit on the OTHER groups' frames (so no game is calibrated by a map that saw
    it). This is O(n_folds) isotonic fits — the earlier leave-one-GAME-out form was O(games), i.e.
    one fit over the whole multi-million-row trace per game, which is quadratic and OOMs at corpus
    scale. The fit set is subsampled to ``max_fit_points`` (isotonic is a low-capacity monotone map,
    so a large sample is unnecessary and the extra points only cost memory). With one game (n_folds
    collapses to 1) every frame passes through uncalibrated, exactly as a single held-out fold should.
    """
    tr = trace.join(realized, on=[game_col, "possession_id"], how="left")
    games = sorted(tr[game_col].unique().to_list())
    k = max(1, min(n_folds, len(games)))
    fold_of = {g: i % k for i, g in enumerate(games)}
    parts = []
    for fi in range(k):
        held_games = [g for g in games if fold_of[g] == fi]
        held_df = tr.filter(pl.col(game_col).is_in(held_games))
        if held_df.height == 0:
            continue
        fit = tr.filter((~pl.col(game_col).is_in(held_games)) & pl.col("realized").is_not_null())
        if fit.height < min_fit:
            parts.append(held_df.with_columns(pl.col(epv_col).alias(out_col)))
            continue
        if fit.height > max_fit_points:
            fit = fit.sample(n=max_fit_points, seed=seed)
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            fit[epv_col].to_numpy(), fit["realized"].to_numpy()
        )
        parts.append(held_df.with_columns(pl.Series(out_col, iso.predict(held_df[epv_col].to_numpy()))))
    return pl.concat(parts).drop("realized")
