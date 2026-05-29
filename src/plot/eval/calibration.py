"""Calibration metrics for the EPV eval bar (Gate G1) — pure numpy, no plotting.

G1 asks: is the EPV forecast calibrated on held-out games? EPV is a CONTINUOUS expected-points
value (in [0, K-1]) read off a multiclass outcome head (points in {0..K-1}), so we measure both
value-level calibration of the scalar EPV and probabilistic calibration of the head:

- reliability_bins / ece: continuous-target reliability (bin by predicted EPV; compare the bin's
  mean prediction to its mean realized points) and the binned ECE/MCE in POINTS units.
- calibration_slope: OLS of realized points on predicted EPV (slope 1, intercept 0 = perfect).
- proper scores: multiclass logloss, Brier, and the ordinal RPS, each vs a constant class-prior
  baseline the model must beat (a trivially-calibrated constant must NOT pass G1).

Every metric takes optional per-row weights (inverse possession size, so each possession counts
equally) and a game id for the game-block bootstrap. Possession-level variants collapse each
possession to (mean EPV, its single label) — the least-autocorrelated, headline estimate.
"""

from __future__ import annotations

import numpy as np


def epv_from_proba(probs: np.ndarray) -> np.ndarray:
    """EPV = sum_k k * P(points=k). ``probs`` is (n, K) over points {0..K-1}."""
    points = np.arange(probs.shape[1], dtype=float)
    return probs @ points


def _norm_weights(w: np.ndarray | None, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n)
    w = np.asarray(w, dtype=float)
    return np.where(np.isfinite(w) & (w > 0), w, 0.0)


def _bin_edges(pred: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(pred, qs)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([pred.min(), pred.max() + 1e-9])
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        return edges
    return np.linspace(pred.min() - 1e-9, pred.max() + 1e-9, n_bins + 1)


def reliability_bins(
    pred: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None,
    n_bins: int = 15, strategy: str = "quantile",
) -> list[dict]:
    """Binned conditional-mean calibration: per bin, weighted mean prediction vs mean realized."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    w = _norm_weights(weight, pred.size)
    edges = _bin_edges(pred, n_bins, strategy)
    idx = np.clip(np.digitize(pred, edges) - 1, 0, edges.size - 2)
    out = []
    for b in range(edges.size - 1):
        m = idx == b
        wb = w[m].sum()
        if wb <= 0:
            continue
        out.append({
            "bin": b,
            "pred_mean": float(np.average(pred[m], weights=w[m])),
            "obs_mean": float(np.average(y[m], weights=w[m])),
            "weight": float(wb),
            "count": int(m.sum()),
        })
    return out


def ece(pred, y, weight=None, n_bins: int = 15, strategy: str = "quantile") -> dict:
    """Expected/Max calibration error of EPV in POINTS units, over reliability bins."""
    bins = reliability_bins(pred, y, weight, n_bins, strategy)
    if not bins:
        return {"ece": float("nan"), "mce": float("nan"), "n_bins": 0}
    total = sum(b["weight"] for b in bins)
    gaps = [abs(b["obs_mean"] - b["pred_mean"]) for b in bins]
    e = sum(b["weight"] * g for b, g in zip(bins, gaps, strict=True)) / total
    return {"ece": float(e), "mce": float(max(gaps)), "n_bins": len(bins)}


def calibration_slope(pred, y, weight=None) -> dict:
    """Weighted OLS of realized points on predicted EPV: slope, intercept, calibration-in-large."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    w = _norm_weights(weight, pred.size)
    sw = w.sum()
    mx, my = np.average(pred, weights=w), np.average(y, weights=w)
    var = np.average((pred - mx) ** 2, weights=w)
    cov = np.average((pred - mx) * (y - my), weights=w)
    slope = float(cov / var) if var > 0 else float("nan")
    intercept = float(my - slope * mx) if var > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "calibration_in_the_large": float(my - mx),
            "mean_pred": float(mx), "mean_obs": float(my), "n_eff": float(sw)}


def theil_sen_slope(pred, y) -> dict:
    """Robust (Theil-Sen) slope/intercept of y on pred; resistant to the discrete-y spikes."""
    from scipy.stats import theilslopes

    pred, y = np.asarray(pred, float), np.asarray(y, float)
    if pred.size < 2 or np.ptp(pred) == 0:
        return {"slope": float("nan"), "intercept": float("nan")}
    slope, intercept, _, _ = theilslopes(y, pred)
    return {"slope": float(slope), "intercept": float(intercept)}


def _onehot(y: np.ndarray, K: int) -> np.ndarray:
    oh = np.zeros((y.size, K))
    oh[np.arange(y.size), np.asarray(y, int)] = 1.0
    return oh


def multiclass_logloss(probs, y, weight=None, eps: float = 1e-12) -> float:
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    w = _norm_weights(weight, y.size)
    p = np.clip(probs[np.arange(y.size), y], eps, 1.0)
    return float(np.average(-np.log(p), weights=w))


def brier(probs, y, weight=None) -> float:
    probs = np.asarray(probs, float)
    oh = _onehot(np.asarray(y, int), probs.shape[1])
    w = _norm_weights(weight, probs.shape[0])
    return float(np.average(((probs - oh) ** 2).sum(axis=1), weights=w))


def rps(probs, y, weight=None) -> float:
    """Ranked Probability Score — penalizes by ORDINAL distance (a '3' predicted for a '0' hurts more)."""
    probs = np.asarray(probs, float)
    oh = _onehot(np.asarray(y, int), probs.shape[1])
    w = _norm_weights(weight, probs.shape[0])
    cdf_p, cdf_o = np.cumsum(probs, axis=1), np.cumsum(oh, axis=1)
    return float(np.average(((cdf_p - cdf_o) ** 2).sum(axis=1), weights=w))


def proper_scores(probs, y, weight=None) -> dict:
    return {"logloss": multiclass_logloss(probs, y, weight),
            "brier": brier(probs, y, weight), "rps": rps(probs, y, weight)}


def constant_baseline(y, weight=None, K: int | None = None) -> dict:
    """Scores of the constant predictor = weighted class marginals (must be BEATEN to pass G1)."""
    y = np.asarray(y, int)
    w = _norm_weights(weight, y.size)
    K = K or int(y.max() + 1)
    marg = np.array([w[y == k].sum() for k in range(K)])
    marg = marg / marg.sum()
    probs = np.tile(marg, (y.size, 1))
    out = proper_scores(probs, y, weight)
    out["epv"] = float(epv_from_proba(marg[None, :])[0])
    return out


def per_class_reliability(probs, y, weight=None, n_bins: int = 10) -> dict:
    """One-vs-rest reliability + classwise ECE for each points class."""
    probs, y = np.asarray(probs, float), np.asarray(y, int)
    K = probs.shape[1]
    classwise, eces = {}, []
    for k in range(K):
        b = reliability_bins(probs[:, k], (y == k).astype(float), weight, n_bins, "quantile")
        classwise[k] = b
        if b:
            tot = sum(x["weight"] for x in b)
            eces.append(sum(x["weight"] * abs(x["obs_mean"] - x["pred_mean"]) for x in b) / tot)
    return {"classwise_bins": classwise, "macro_ece": float(np.mean(eces)) if eces else float("nan")}


def collapse_to_possessions(pred, y, game_id, possession_id) -> dict:
    """Collapse to one (mean pred, single label, game) row per possession (least-autocorrelated)."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    keys = np.array([f"{g}|{p}" for g, p in zip(np.asarray(game_id), np.asarray(possession_id), strict=True)])
    uniq = np.unique(keys)
    mp = np.array([pred[keys == k].mean() for k in uniq])
    my = np.array([y[keys == k][0] for k in uniq])
    mg = np.array([str(game_id[np.where(keys == k)[0][0]]) for k in uniq])
    return {"pred": mp, "y": my, "game_id": mg}


def game_block_bootstrap(metric_fn, game_id, *, B: int = 1000, seed: int = 1729) -> dict:
    """Resample whole GAMES with replacement; return (lo, hi) 95% CI of ``metric_fn(mask)->float``.

    ``metric_fn`` takes a boolean row mask (or index array) and returns a scalar; we pass the row
    indices of the resampled games so within-game correlation is honored.
    """
    rng = np.random.default_rng(seed)
    game_id = np.asarray(game_id)
    games = np.unique(game_id)
    by_game = {g: np.where(game_id == g)[0] for g in games}
    vals = []
    for _ in range(B):
        pick = rng.choice(games, size=games.size, replace=True)
        idx = np.concatenate([by_game[g] for g in pick])
        v = metric_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan")}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "mean": float(np.mean(vals))}
