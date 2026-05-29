"""LightGBM multiclass outcome head for the baseline EPV, with the EPV = sum_k k*P_k reduction.

The plan's CPU baseline. We model the possession-points outcome as a categorical {0..K-1} with
multiclass logloss (a strictly-proper loss that drives the probabilities G1 measures) and read
the scalar eval bar off it as EPV = sum_k k * P(points=k). Rare 4-point possessions are clamped
into the top class. Hyperparameters are fixed and regularized for autocorrelated small data
(high ``min_child_samples`` because 10 fps frames within a possession share one label); the same
``predict_epv`` API is reused by Stage 3 and the demo export.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from plot.eval.calibration import epv_from_proba
from plot.features.court import REGIONS
from plot.features.eval_bar import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, LABEL_COLUMN, WEIGHT_COLUMN

N_CLASSES = 4  # points classes {0,1,2,3}; rare 4 clamped to 3

# fixed category vocabularies so train/val/predict encode identically
_CATEGORIES = {
    "court_region": list(REGIONS),
    "prev_possession_end_reason": [
        "none", "made_fg", "made_ft", "defensive_rebound", "turnover",
        "end_period", "jump_ball", "control_change",
    ],
}

DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "min_child_samples": 200,
    "min_data_in_bin": 50,
    "verbosity": -1,
    "seed": 1729,
    "num_threads": 0,
}


def features_to_pandas(X: pl.DataFrame) -> pd.DataFrame:
    """Convert the polars feature matrix to a LightGBM-ready pandas frame (fixed categoricals)."""
    pdf = X.select(FEATURE_COLUMNS).to_pandas()
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            pdf[col] = pd.Categorical(pdf[col].fillna("none"), categories=_CATEGORIES[col])
        elif pdf[col].dtype == bool:
            pdf[col] = pdf[col].astype("float32")
        else:
            pdf[col] = pdf[col].astype("float32")
    return pdf


def _labels(X: pl.DataFrame) -> np.ndarray:
    return np.clip(X[LABEL_COLUMN].to_numpy(), 0, N_CLASSES - 1).astype(int)


def train_booster(
    train: pl.DataFrame,
    valid: pl.DataFrame | None = None,
    *,
    params: dict | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    """Train the multiclass booster; if ``valid`` is given, early-stop on its multi_logloss."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    dtrain = lgb.Dataset(
        features_to_pandas(train), label=_labels(train),
        weight=train[WEIGHT_COLUMN].to_numpy(),
        categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    if valid is not None and valid.height > 0:
        dvalid = lgb.Dataset(
            features_to_pandas(valid), label=_labels(valid),
            weight=valid[WEIGHT_COLUMN].to_numpy(),
            categorical_feature=CATEGORICAL_COLUMNS, reference=dtrain, free_raw_data=False,
        )
        valid_sets = [dvalid]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    return lgb.train(p, dtrain, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)


def predict_proba(booster: lgb.Booster, X: pl.DataFrame) -> np.ndarray:
    """Class probabilities (n, K) over points {0..K-1}."""
    probs = booster.predict(features_to_pandas(X), num_iteration=booster.best_iteration or None)
    return np.asarray(probs)


def predict_epv(booster: lgb.Booster, X: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (EPV per frame, class probabilities)."""
    probs = predict_proba(booster, X)
    return epv_from_proba(probs), probs


def save_booster(booster: lgb.Booster, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(p))


def load_booster(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
