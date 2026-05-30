"""xPoints — the expected points of taking a shot from a given state.

The well-identified counterfactual at the core of Gate G2: a calibrated *shoot-now* value, fit on
REAL observed shots (``plot.possessions.shots``), so it answers "what is an open shot from here
worth?" without a generative rollout. We model P(make | distance, openness, 2-vs-3) with a
regularized logistic regression — calibrated and hard to overfit on a sparse, well-understood
signal — and read xPoints = P(make) x points_if_made. The make-probability bowl in distance is
captured with a quadratic term; openness (nearest-defender distance) and the 3-point indicator
enter linearly.

LOGO (leave-one-game-out) cross-validation produces pooled out-of-fold predictions, so xPoints'
calibration is measured on held-out games — exactly the G2 question ("when the open shot WAS
taken, does xPoints match the realized points?").
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

XPOINTS_FEATURES = ["dist_to_rim", "dist_to_rim_sq", "nearest_def_dist", "three_pt"]


def _design(shots: pl.DataFrame) -> np.ndarray:
    d = shots.with_columns(
        (pl.col("dist_to_rim") ** 2).alias("dist_to_rim_sq"),
        pl.col("three_pt").cast(pl.Float64),
    )
    return np.column_stack([d[c].to_numpy().astype(float) for c in XPOINTS_FEATURES])


def train_xpoints(shots: pl.DataFrame, *, c: float = 0.1, seed: int = 1729):
    """Fit P(make | features) as a standardized logistic regression.

    ``c`` defaults to 0.1 (strong L2): on this sparse, noisy-openness data a softly-regularized fit
    is over-spread (slope ~0.79, ECE ~0.11); c=0.1 pulls the slope toward 1 and ECE under 0.10.
    """
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
    model.fit(_design(shots), shots["made"].to_numpy().astype(int))
    return model


def predict_xpoints(model, shots: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (P(make), xPoints = P(make) * points_if_made)."""
    p = model.predict_proba(_design(shots))[:, 1]
    return p, p * shots["pts_if_made"].to_numpy().astype(float)


def predict_xpoints_open(model, shots: pl.DataFrame, *, openness_ft: float = 6.0) -> np.ndarray:
    """xPoints for the OPEN-shot counterfactual: the same shots evaluated at wide-open spacing.

    Used by the regret layer — "best available open shot" assumes the look is uncontested, so we
    substitute a wide-open nearest-defender distance regardless of the actual contest.
    """
    open_shots = shots.with_columns(pl.lit(openness_ft).alias("nearest_def_dist"))
    return predict_xpoints(model, open_shots)[1]


def logo_xpoints(shots: pl.DataFrame, *, game_col: str = "game_id", min_train: int = 50) -> pl.DataFrame:
    """Leave-one-game-out OOF predictions: per held-out shot, p_make + xpoints from a model that
    never saw its game. Returns the held-out shots with p_make / xpoints attached."""
    games = sorted(shots[game_col].unique().to_list())
    parts = []
    for held in games:
        train = shots.filter(pl.col(game_col) != held)
        test = shots.filter(pl.col(game_col) == held)
        if test.height == 0 or train.height < min_train:
            continue
        model = train_xpoints(train)
        p, xp = predict_xpoints(model, test)
        parts.append(test.with_columns(pl.Series("p_make", p), pl.Series("xpoints", xp)))
    return pl.concat(parts) if parts else pl.DataFrame()
