"""Gate G5 — *validity* of the PLOT regret metric: is high regret really value left behind by a
worse decision, or an artifact of how the inputs are valued?

G3/G4 established that PLOT is real, stable, and box-score-orthogonal. They did *not* establish that
the points it says were "left on the table" were genuinely available. Three concrete worries, and a
test for each (all runnable on local data — they are properties of the *method*, scale-invariant):

* **G5(a) — the inputs are calibrated where the metric actually lives.** Regret = ``xPoints(open
  shot) − post-pass EPV``. If, *restricted to the open pass-up decision set* (not all passes), the
  post-pass EPV matches realized possession points and xPoints matches realized makes — including
  the high-value / near-rim slice where the leaderboard's bigs sit — then positive regret means the
  possession genuinely yielded less than an open shot was worth. (G2 showed this in aggregate; G5(a)
  checks the specific subspace.)
* **G5(b) — selection-on-observables is bounded.** ``xPoints`` is a *population* value for the
  passed-up shot; we never see what the player would have made. If players pass up the observably
  *worse* open looks, population xPoints overstates them. We bound this by comparing taken vs passed
  open looks on observable difficulty (standardized mean differences, a propensity AUC) and anchor
  on the empirical make rate of *taken* open shots by value bin. It cannot eliminate selection on
  unobservables — reported as a bound, not a pass.
* **G5(c) — the ordering is robust to finishing skill.** Rebuild regret with a *shooter-aware*
  xPoints (each shot re-valued at the shooter's own season make rate, empirical-Bayes shrunk) and
  check the per-player ordering barely moves. If it does, the leaderboard is not a finishing-skill
  artifact of the population make model (a stronger form of G3(b-ii)).

Pure over its array / DataFrame inputs (no I/O, no model training) so every statistic is testable.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _num(d: dict, key: str, default: float) -> float:
    """None-safe numeric read (a legitimate 0.0 survives; ``d.get(k) or default`` would not)."""
    v = d.get(key, default)
    return default if v is None else float(v)


def calibration(pred: np.ndarray, realized: np.ndarray, *, n_bins: int = 10) -> dict:
    """Calibration of a continuous-or-binary prediction against its realized target: bias
    (cal-in-the-large), OLS slope of realized-on-pred, and a quantile-binned ECE. Works uniformly
    for P(make) vs made (binary) and post-pass EPV vs realized points (0..3)."""
    pred = np.asarray(pred, float)
    realized = np.asarray(realized, float)
    n = pred.size
    if n < 5:
        return {"n": int(n), "note": "too few"}
    cal_in_large = float(pred.mean() - realized.mean())
    # OLS realized ~ a + b*pred
    b, a = np.polyfit(pred, realized, 1) if np.ptp(pred) > 0 else (float("nan"), float("nan"))
    # quantile-binned ECE
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2) if len(edges) > 2 else np.zeros(n, int)
    ece = 0.0
    for bb in np.unique(idx):
        m = idx == bb
        ece += (m.sum() / n) * abs(pred[m].mean() - realized[m].mean())
    return {
        "n": int(n),
        "mean_pred": round(float(pred.mean()), 4),
        "mean_realized": round(float(realized.mean()), 4),
        "cal_in_large": round(cal_in_large, 4),
        "slope": round(float(b), 4),
        "ece": round(float(ece), 4),
    }


def calibration_gate(legs: dict, *, cal_max: float = 0.05, ece_max: float = 0.10) -> dict:
    """G5(a): every named calibration leg is unbiased (|cal-in-large| < ``cal_max``) and sharp
    (ECE < ``ece_max``)."""
    ok = {
        name: abs(_num(c, "cal_in_large", 1.0)) < cal_max and _num(c, "ece", 1.0) < ece_max
        for name, c in legs.items()
    }
    return {"per_leg": ok, "pass": all(ok.values()) if ok else False,
            "cal_max": cal_max, "ece_max": ece_max}


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (taken − passed), pooled-SD scaled. |SMD|<0.2 ≈ negligible."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if (a.size > 1 and b.size > 1) else float("nan")
    return float((a.mean() - b.mean()) / sd) if sd and sd > 0 else float("nan")


def selection_diagnostic(
    taken: pl.DataFrame, passed: pl.DataFrame, *, feature_cols: list[str],
    value_col: str = "open_xpoints", n_value_bins: int = 5,
) -> dict:
    """G5(b): how observably different are the open shots a player *takes* from the ones they *pass*?

    Returns per-feature standardized mean differences, a propensity AUC (how well observable
    features predict take-vs-pass — high ⇒ strong selection on observables), and the empirical make
    rate of *taken* open shots by value bin (the anchor xPoints is fit to)."""
    smd = {c: round(_smd(taken[c].to_numpy(), passed[c].to_numpy()), 4) for c in feature_cols}

    y = np.concatenate([np.ones(taken.height), np.zeros(passed.height)])
    X = np.vstack([
        np.column_stack([taken[c].to_numpy().astype(float) for c in feature_cols]),
        np.column_stack([passed[c].to_numpy().astype(float) for c in feature_cols]),
    ])
    auc = float("nan")
    if taken.height >= 10 and passed.height >= 10:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X, y)
        auc = round(float(roc_auc_score(y, clf.predict_proba(X)[:, 1])), 4)

    # empirical make rate of taken open shots by value (open-xpoints) bin
    by_bin = []
    if value_col in taken.columns and "made" in taken.columns and taken.height >= n_value_bins:
        v = taken[value_col].to_numpy().astype(float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_value_bins + 1)))
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
        made = taken["made"].to_numpy().astype(float)
        for bb in np.unique(idx):
            m = idx == bb
            by_bin.append({"bin": int(bb), "n": int(m.sum()),
                           "mean_open_xpoints": round(float(v[m].mean()), 4),
                           "empirical_make_rate": round(float(made[m].mean()), 4)})

    return {
        "n_taken": int(taken.height), "n_passed": int(passed.height),
        "feature_smd_taken_minus_passed": smd,
        "propensity_auc": auc,
        "taken_open_make_rate_by_value_bin": by_bin,
    }


def per_player_shot_rates(pbp_norm: pl.DataFrame) -> pl.DataFrame:
    """Season-long per-player 2P / 3P makes & attempts from normalized PBP (for the shooter-aware
    re-valuation). Players keyed by id; team entities (huge ids) dropped, as in box_stats."""
    made = pl.col("msg_type") == 1
    att = pl.col("msg_type").is_in([1, 2])
    is3 = pl.col("desc_upper").str.contains("3PT")
    df = pbp_norm.filter(
        pl.col("player_id").is_not_null() & (pl.col("player_id") > 0)
        & (pl.col("player_id") < 1_000_000_000) & att
    )
    return df.group_by("player_id").agg(
        (made & ~is3).cast(pl.Int64).sum().alias("fg2m"),
        (att & ~is3).cast(pl.Int64).sum().alias("fg2a"),
        (made & is3).cast(pl.Int64).sum().alias("fg3m"),
        (att & is3).cast(pl.Int64).sum().alias("fg3a"),
    )


def shooter_skill_ratios(
    rates: pl.DataFrame, *, k2: float = 200.0, k3: float = 100.0
) -> tuple[pl.DataFrame, float, float]:
    """Empirical-Bayes shrunk per-player make rate ÷ league make rate, per shot type. Returns
    (ratios_df[player_id, ratio_2p, ratio_3p], league_2p, league_3p). K is the shrinkage strength
    (attempts of league-average prior); low-volume shooters pull toward 1.0."""
    lg2 = float(rates["fg2m"].sum() / max(rates["fg2a"].sum(), 1))
    lg3 = float(rates["fg3m"].sum() / max(rates["fg3a"].sum(), 1))
    out = rates.with_columns(
        (((pl.col("fg2m") + k2 * lg2) / (pl.col("fg2a") + k2)) / lg2).alias("ratio_2p"),
        (((pl.col("fg3m") + k3 * lg3) / (pl.col("fg3a") + k3)) / lg3).alias("ratio_3p"),
    ).select("player_id", "ratio_2p", "ratio_3p")
    return out, lg2, lg3


def shooter_adjusted_regret(decisions: pl.DataFrame, ratios: pl.DataFrame) -> pl.DataFrame:
    """Re-value each passed-up open shot at the shooter's own make rate: ``p_make_pop`` (=
    best_available / pts_if_made) × the shooter's type skill ratio, clipped, × points. Adds
    ``best_available_sh`` and ``regret_signed_sh`` / ``regret_clipped_sh``. Players with no rate
    fall back to ratio 1.0 (population)."""
    pts = pl.when(pl.col("three_pt")).then(3.0).otherwise(2.0)
    d = decisions.join(ratios, on="player_id", how="left").with_columns(
        pl.col("ratio_2p").fill_null(1.0), pl.col("ratio_3p").fill_null(1.0)
    ).with_columns(
        (pl.col("best_available") / pts).alias("_pmake_pop"),
        pl.when(pl.col("three_pt")).then(pl.col("ratio_3p")).otherwise(pl.col("ratio_2p")).alias("_ratio"),
    ).with_columns(
        (pl.min_horizontal(pl.col("_pmake_pop") * pl.col("_ratio"), pl.lit(0.99)) * pts).alias("best_available_sh")
    ).with_columns(
        (pl.col("best_available_sh") - pl.col("post_epv")).alias("regret_signed_sh"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.col("regret_signed_sh")).alias("regret_clipped_sh"),
    )
    return d.drop("_pmake_pop", "_ratio")


def ordering_robustness(
    pop_per_player: pl.DataFrame, adj_per_player: pl.DataFrame, *,
    value_col: str = "mean_regret_clipped", top_k: int = 30,
) -> dict:
    """G5(c): does per-player PLOT survive swapping population xPoints for shooter-aware xPoints?
    Spearman + Pearson of the two per-player metrics, and top-K leaderboard overlap."""
    j = pop_per_player.select("player_id", pl.col(value_col).alias("pop")).join(
        adj_per_player.select("player_id", pl.col(value_col).alias("adj")), on="player_id", how="inner"
    )
    if j.height < 5:
        return {"n_players": int(j.height), "note": "too few overlapping players"}
    a, b = j["pop"].to_numpy(), j["adj"].to_numpy()
    pr, _ = stats.pearsonr(a, b)
    sr, _ = stats.spearmanr(a, b)
    k = min(top_k, j.height)
    top_pop = set(j.sort("pop", descending=True).head(k)["player_id"].to_list())
    top_adj = set(j.sort("adj", descending=True).head(k)["player_id"].to_list())
    overlap = len(top_pop & top_adj) / k
    return {
        "n_players": int(j.height), "value_col": value_col,
        "pearson_r": round(float(pr), 4), "spearman_r": round(float(sr), 4),
        "top_k": int(k), "top_k_overlap": round(float(overlap), 4),
    }


def g5_gate(cal_gate: dict, robustness: dict, *, spearman_min: float = 0.80) -> dict:
    """G5 PASS = inputs calibrated in-subspace (G5a) AND per-player ordering robust to finishing
    skill (G5c). The selection bound (G5b) is reported, not gated — it bounds, not certifies."""
    gate = {
        "inputs_calibrated_in_subspace": bool(cal_gate.get("pass", False)),
        "ordering_robust_to_finishing": _num(robustness, "spearman_r", 0.0) >= spearman_min,
    }
    return {"gate": gate, "spearman_min": spearman_min, "pass": all(gate.values())}
