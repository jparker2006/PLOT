"""Leave-one-game-out cross-validation producing pooled out-of-fold EPV predictions.

The held-out unit is the whole GAME (moments within a game/possession share labels and are
massively autocorrelated, so any finer split leaks). Each fold trains on all-but-one game,
carving one further whole game as a grouped inner watch set for early stopping (never the
held-out game), and predicts on the held-out game. Pooling every fold's held-out predictions
gives an OOF set the size of the full corpus where each game is genuinely out-of-sample — the
lowest-variance unbiased held-out calibration estimate at n~10 games, spending no permanent holdout.
"""

from __future__ import annotations

import polars as pl

from plot.features.eval_bar import GROUP_COLUMN, LABEL_COLUMN, WEIGHT_COLUMN
from plot.models.eval_bar.dataset import logo_folds
from plot.models.eval_bar.folds import inner_val_game, oof_arrays  # noqa: F401  (re-exported)
from plot.models.eval_bar.model import N_CLASSES, predict_epv, train_booster

# back-compat alias for the private name used in the baseline script/tests
_inner_val_game = inner_val_game


def run_logo(
    corpus: pl.DataFrame,
    game_ids: list[str] | None = None,
    *,
    params: dict | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
    use_inner_val: bool = True,
) -> dict:
    """Run LOGO CV; return {oof: DataFrame, folds: [...]}. OOF has epv, p0..pK, y, game_id, possession_id, weight."""
    games = game_ids or sorted(corpus[GROUP_COLUMN].unique().to_list())
    oof_parts, folds = [], []
    for train_ids, held in logo_folds(games):
        train = corpus.filter(pl.col(GROUP_COLUMN).is_in(train_ids))
        held_df = corpus.filter(pl.col(GROUP_COLUMN) == held)
        if use_inner_val and len(train_ids) > 1:
            val_id = _inner_val_game(train_ids, held)
            valid = train.filter(pl.col(GROUP_COLUMN) == val_id)
            fit = train.filter(pl.col(GROUP_COLUMN) != val_id)
        else:
            val_id, valid, fit = None, None, train
        booster = train_booster(fit, valid, params=params,
                                num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds)
        epv, probs = predict_epv(booster, held_df)
        part = held_df.select(GROUP_COLUMN, "possession_id", WEIGHT_COLUMN,
                              pl.col(LABEL_COLUMN).clip(0, N_CLASSES - 1).alias("y")).with_columns(
            pl.Series("epv", epv),
            *[pl.Series(f"p{k}", probs[:, k]) for k in range(probs.shape[1])],
        )
        oof_parts.append(part)
        folds.append({"held_out": held, "inner_val": val_id, "n_train": fit.height,
                      "n_test": held_df.height, "best_iteration": int(booster.best_iteration or num_boost_round)})
    oof = pl.concat(oof_parts) if oof_parts else pl.DataFrame()
    return {"oof": oof, "folds": folds, "games": games}
