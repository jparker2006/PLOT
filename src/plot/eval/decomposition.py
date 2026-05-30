"""Gate G6, step 1 — role-of-touch vs within-role decision quality.

The PLOT regret metric is stable (G3a) and box-orthogonal (G4), but every variant we built (v1, v1.5,
v2) sorts players by the *kind of touches they get* — a center's rim catches, a guard's perimeter
pass-ups — more than by how well they decide *within* those touches. Before spending the
outcome-validity budget (G6 step 2) on a signal that might be entirely role, this module puts a
number on the founding question: *decision IQ, or just role?*

Method. Fit a **context** model ``regret ~ touch-context`` (location, openness, decision kind) with
**no player identity**, out-of-fold by game so no decision is scored by a model that saw its own
game. With no player feature the model can only learn the *average* regret a situation type carries,
so:

* the out-of-fold prediction is the **role component** — the regret anyone faces in that spot;
* the residual is the **within-role component** — how much *this* player beat or trailed the
  situation's expectation.

Then two readouts:

* **variance split** — of the between-player variance in mean regret, how much the role component
  reconstructs (``role_share``) vs the within-role residual (``within_role_share``);
* **reliability** — split-half (Spearman-Brown) of the within-role residual. Raw regret is reliable;
  the real question is whether anything reliable *survives* removing role. If residual reliability
  collapses, the reliable signal was all role-of-touch; if it holds, within-role decision quality is
  itself a stable skill — the thing G6 step 2 then validates against real outcomes.

NOTE on "position": the public tracking feed carries no listed positions, so touch-context (where a
player gets the ball, how open, which decision) *is* the operational role here — and a per-decision
context is a finer role control than a five-bucket position label would be. The residual is what
player identity adds on top of the situation.

Pure over its DataFrame inputs (the context fit is deterministic given a fixed seed) so the split is
unit-testable; the heavy regret build lives in ``plot.models.regret.pipeline``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from plot.models.regret.plot_metric import _num

# touch-context: WHERE the decision happens (spatial role) + how open + 3-pt. Deliberately NO player
# identity — that is exactly the signal we want to land in the residual, not absorb into "role".
CONTEXT_FEATURES = ["sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist"]


# context-model capacity grid for the leakage robustness check (weak → strong role absorption). If a
# STRONGER role model still leaves the within-role residual reliable, the residual is genuine skill,
# not leftover unabsorbed role; if it collapses as capacity rises, it was leakage.
CONTEXT_MODEL_GRID = {
    "weak": dict(max_depth=2, max_iter=100, learning_rate=0.05, min_samples_leaf=80),
    "default": dict(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=40),
    "strong": dict(max_depth=5, max_iter=400, learning_rate=0.05, min_samples_leaf=15),
}


def residualize_on_context(
    regret: pl.DataFrame,
    *,
    value_col: str = "regret_clipped",
    features: list[str] | None = None,
    group_col: str = "game_id",
    n_splits: int = 5,
    seed: int = 0,
    model_kwargs: dict | None = None,
) -> pl.DataFrame:
    """Add ``<value>_role`` (out-of-fold context prediction) and ``<value>_resid`` (= value − role).

    The context model is gradient-boosted on touch-context only and fit out-of-fold grouped by
    ``group_col`` (game), so no decision is scored by a model that saw its own game. Because the
    model has no player feature, a player's systematic over/under-performance of a situation's
    average regret lands entirely in the residual — which is what makes the residual a clean
    *within-role* signal. ``model_kwargs`` overrides the booster capacity (see ``CONTEXT_MODEL_GRID``
    for the leakage sensitivity). Deterministic given ``seed``.
    """
    features = features or CONTEXT_FEATURES
    if regret.height == 0:
        return regret
    df = regret.with_columns(pl.col("three_pt").cast(pl.Float64))
    feat = list(features)
    if "decision_kind" in df.columns:  # let the model know pass-up vs shot-selection
        df = df.with_columns((pl.col("decision_kind") == "shot_selection").cast(pl.Float64).alias("_is_shotsel"))
        feat = feat + ["_is_shotsel"]

    X = np.column_stack([df[c].to_numpy().astype(float) for c in feat])
    y = df[value_col].to_numpy().astype(float)
    groups = df[group_col].to_numpy()
    n_groups = len({*groups.tolist()})
    pred = np.full(y.shape, np.nan)
    kw = model_kwargs or CONTEXT_MODEL_GRID["default"]

    def _fit(seed_):
        return HistGradientBoostingRegressor(random_state=seed_, **kw)

    splits = min(n_splits, n_groups)
    if splits < 2:  # too few games to cross-fit — in-sample fallback (residual still defined)
        pred = _fit(seed).fit(X, y).predict(X)
    else:
        for tr, te in GroupKFold(n_splits=splits).split(X, y, groups):
            pred[te] = _fit(seed).fit(X[tr], y[tr]).predict(X[te])

    return df.with_columns(
        pl.Series(f"{value_col}_role", pred),
        pl.Series(f"{value_col}_resid", y - pred),
    )


def variance_decomposition(
    residualized: pl.DataFrame, *, value_col: str = "regret_clipped", min_decisions: int = 30
) -> dict:
    """Split the between-player variance of mean regret into role vs within-role.

    Aggregates the raw value, its role component, and its residual per player (≥ ``min_decisions``),
    then uses the exact identity ``Var(raw) = Var(role) + Var(resid) + 2·Cov(role, resid)`` on the
    per-player means. ``role_share``/``within_role_share`` are the first two terms over Var(raw); the
    cross term is reported so the three sum to 1 (sign-honest — role and within-role can correlate).
    ``r2_role_reconstructs_leaderboard`` is the squared correlation of the role component with the raw
    per-player metric: how much of *who leaves points* is explained by touch profile alone.
    """
    role_col, resid_col = f"{value_col}_role", f"{value_col}_resid"
    agg = (
        residualized.group_by("player_id")
        .agg(
            pl.len().alias("n"),
            pl.col(value_col).mean().alias("m_raw"),
            pl.col(role_col).mean().alias("m_role"),
            pl.col(resid_col).mean().alias("m_resid"),
        )
        .filter(pl.col("n") >= min_decisions)
    )
    if agg.height < 3:
        return {"n_players": int(agg.height), "value_col": value_col, "note": "too few players"}
    raw, role, resid = (agg[c].to_numpy() for c in ("m_raw", "m_role", "m_resid"))
    var_raw = float(np.var(raw, ddof=1))
    var_role = float(np.var(role, ddof=1))
    var_resid = float(np.var(resid, ddof=1))
    cov = float(np.cov(role, resid, ddof=1)[0, 1])
    r_role = float(np.corrcoef(role, raw)[0, 1])
    return {
        "n_players": int(agg.height),
        "value_col": value_col,
        "var_player_mean_raw": round(var_raw, 6),
        "var_role_component": round(var_role, 6),
        "var_within_role_component": round(var_resid, 6),
        "cov_role_within": round(cov, 6),
        "role_share": round(var_role / var_raw, 4) if var_raw > 0 else None,
        "within_role_share": round(var_resid / var_raw, 4) if var_raw > 0 else None,
        "cross_share": round(2 * cov / var_raw, 4) if var_raw > 0 else None,
        "r2_role_reconstructs_leaderboard": round(r_role**2, 4),
    }


def player_components(
    residualized: pl.DataFrame, *, value_col: str = "regret_clipped", per: int = 100, min_decisions: int = 30
) -> pl.DataFrame:
    """Per-player table: raw PLOT, its role component, and the within-role residual (each per ``per``
    decisions), sorted by the residual. The residual column is the candidate 'decision-quality'
    metric G6 step 2 validates against outcomes — role-of-touch removed."""
    role_col, resid_col = f"{value_col}_role", f"{value_col}_resid"
    return (
        residualized.group_by("player_id")
        .agg(
            pl.len().alias("n_decisions"),
            (pl.col(value_col).mean() * per).alias(f"plot_per{per}"),
            (pl.col(role_col).mean() * per).alias(f"role_per{per}"),
            (pl.col(resid_col).mean() * per).alias(f"within_role_per{per}"),
        )
        .filter(pl.col("n_decisions") >= min_decisions)
        .sort(f"within_role_per{per}", descending=True)
    )


def context_capacity_sensitivity(
    regret: pl.DataFrame, *, value_col: str = "regret_clipped",
    min_decisions: int = 30, min_decisions_per_half: int = 15,
) -> dict:
    """Leakage robustness: re-run the split across the ``CONTEXT_MODEL_GRID`` capacities and report
    role_share + within-role residual reliability at each. A within-role signal that is *real* should
    be roughly **stable** as the role model gets stronger; one that *shrinks toward zero* as capacity
    rises was unabsorbed role leaking into the residual. Imported lazily by the build script."""
    from plot.models.regret.plot_metric import split_half_stability  # noqa: PLC0415
    rows = {}
    for name, kw in CONTEXT_MODEL_GRID.items():
        res = residualize_on_context(regret, value_col=value_col, model_kwargs=kw)
        decomp = variance_decomposition(res, value_col=value_col, min_decisions=min_decisions)
        rel = split_half_stability(res, value_col=f"{value_col}_resid",
                                   min_decisions_per_half=min_decisions_per_half)
        rows[name] = {
            "role_share": decomp.get("role_share"),
            "within_role_share": decomp.get("within_role_share"),
            "resid_reliability_sb": rel.get("spearman_brown_full"),
            "resid_reliability_p": rel.get("pearson_p"),
        }
    return rows


def g6_step1_gate(
    decomp: dict, reliability_raw: dict, reliability_resid: dict, *,
    role_share_dominant: float = 0.50, resid_reliability_min: float = 0.30,
) -> dict:
    """Summarize step 1: is the metric role-dominated, and does a reliable within-role signal survive?

    Two independent booleans:
    * ``role_dominated`` — the role component explains ≥ ``role_share_dominant`` of between-player
      variance (confirms the worry quantitatively).
    * ``within_role_reliable`` — the residual's Spearman-Brown split-half reliability is positive,
      significant, and ≥ ``resid_reliability_min``: a within-role decision signal worth validating.

    They are NOT mutually exclusive. Role-dominated + reliable residual ⇒ step 2 validates the
    residual (player-level decision quality exists, just swamped by role in the raw number).
    Role-dominated + NO reliable residual ⇒ the honest finding is "PLOT measures role-of-touch", and
    step 2 must validate at the *decision* level, not the player level.
    """
    role_share = _num(decomp, "role_share", 0.0)
    sb_resid = _num(reliability_resid, "spearman_brown_full", 0.0)
    gate = {
        "role_dominated": role_share >= role_share_dominant,
        "within_role_reliable": (
            sb_resid >= resid_reliability_min
            and _num(reliability_resid, "pearson_p", 1.0) < 0.05
            and _num(reliability_resid, "pearson_r", 0.0) > 0
        ),
    }
    return {
        "gate": gate,
        "role_share": round(role_share, 4),
        "reliability_raw_sb": round(_num(reliability_raw, "spearman_brown_full", 0.0), 4),
        "reliability_within_role_sb": round(sb_resid, 4),
        "verdict": (
            "within-role decision quality is a reliable, validate-able signal"
            if gate["within_role_reliable"]
            else "no reliable within-role signal — PLOT measures role-of-touch"
        ),
    }
