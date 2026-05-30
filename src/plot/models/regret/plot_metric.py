"""The PLOT metric — per-player decision regret aggregated to "points left on the table" / 100, and
the Gate G3 statistics that decide whether it is a *stable skill* and *distinct from finishing*.

The per-decision regret (``plot.models.counterfactual.regret``) is the raw signal: at each open
pass-up decision, ``regret_signed = xPoints(open shot here) − post-pass EPV`` and its clipped
headline ``max(0, ·)``. This module rolls those up per player (per 100 decisions) and runs the two
G3 checks:

* **G3(a) — stable.** Split the season's games into two independent halves (odd/even by sorted game
  id, so neither half is a contiguous date block), aggregate per player within each, and correlate.
  A skill repeats across independent samples; noise does not. We report the raw split-half
  correlation and its Spearman–Brown up-correction to the full-data reliability.
* **G3(b) — decision, not finishing.** Two complementary cuts. (i) A decision-level OLS regressing
  regret on shot-location controls, then an F-test for whether *player* fixed effects add
  explanatory power beyond location — i.e. players differ in regret for reasons location alone can't
  explain. (ii) A player-level check that mean regret is ~uncorrelated with finishing skill (how
  much a player out/under-shoots the population make model on their *real* shots). By construction
  regret uses a population xPoints (no shooter identity), so this should hold; (ii) confirms it
  empirically rather than asserting it.

Pure over its DataFrame inputs (no I/O, no model training) so every statistic is unit-testable.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

# location controls for the G3(b) regression (same geometry xPoints is built on, plus openness)
_LOC_CONTROLS = ["dist_to_rim", "dist_to_rim_sq", "three_pt", "nearest_def_dist"]


def aggregate_per_player(
    regret: pl.DataFrame, *, per: int = 100, min_decisions: int = 30
) -> pl.DataFrame:
    """Per-player PLOT metric: mean regret scaled to ``per`` decisions, for players with ≥
    ``min_decisions`` evaluated decisions. Sorted by the clipped headline (most points left first)."""
    if regret.height == 0:
        return pl.DataFrame()
    agg = (
        regret.group_by("player_id")
        .agg(
            pl.len().alias("n_decisions"),
            pl.col("regret_signed").mean().alias("mean_regret_signed"),
            pl.col("regret_clipped").mean().alias("mean_regret_clipped"),
        )
        .filter(pl.col("n_decisions") >= min_decisions)
        .with_columns(
            (pl.col("mean_regret_signed") * per).alias(f"plot_per{per}_signed"),
            (pl.col("mean_regret_clipped") * per).alias(f"plot_per{per}_clipped"),
        )
        .sort(f"plot_per{per}_clipped", descending=True)
    )
    return agg


def _half_assignment(games: list[str]) -> dict[str, int]:
    """Odd/even split on sorted game ids (balanced, not a contiguous date block)."""
    return {g: (i % 2) for i, g in enumerate(sorted(games))}


def split_half_stability(
    regret: pl.DataFrame, *, value_col: str = "regret_clipped", min_decisions_per_half: int = 15
) -> dict:
    """G3(a): correlate per-player mean regret across two independent halves of the season.

    Players must clear ``min_decisions_per_half`` in BOTH halves to enter. Returns Pearson and
    Spearman correlations plus the Spearman–Brown up-correction (split-half halves the per-player
    sample, so the full-data reliability is ``2r / (1 + r)``).
    """
    if regret.height == 0:
        return {"n_players": 0, "value_col": value_col}
    halves = _half_assignment(regret["game_id"].unique().to_list())
    tagged = regret.with_columns(
        pl.col("game_id").replace_strict(halves, default=0).alias("_half")
    )
    by = (
        tagged.group_by("player_id", "_half")
        .agg(pl.len().alias("n"), pl.col(value_col).mean().alias("m"))
        .filter(pl.col("n") >= min_decisions_per_half)
    )
    wide = (
        by.pivot(values=["n", "m"], index="player_id", on="_half")
        .drop_nulls()
    )
    # pivot names the value columns m_<half>; resolve them defensively (order/suffix vary by version)
    m_cols = sorted(c for c in wide.columns if c.startswith("m_"))
    if len(m_cols) < 2 or wide.height < 3:
        return {"n_players": int(wide.height), "value_col": value_col,
                "note": "too few players in both halves for a stable correlation"}
    a = wide[m_cols[0]].to_numpy()
    b = wide[m_cols[1]].to_numpy()
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    sb = (2 * pr / (1 + pr)) if (1 + pr) != 0 else float("nan")
    return {
        "n_players": int(wide.height),
        "value_col": value_col,
        "min_decisions_per_half": min_decisions_per_half,
        "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6),
        "spearman_brown_full": round(float(sb), 4),
    }


def _ols_rss(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """RSS and rank of an OLS fit (intercept assumed already in X)."""
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), int(rank)


def player_fixed_effect_test(regret: pl.DataFrame, *, value_col: str = "regret_signed",
                             min_decisions: int = 30) -> dict:
    """G3(b-i): does PLAYER identity explain regret beyond shot location?

    Nested OLS F-test. Reduced model: ``value ~ location_controls``. Full model: adds player dummies
    (restricted to players with ≥ ``min_decisions`` so the dummies are estimable; others fold into
    the reference pool). A significant F means between-player regret differences survive controlling
    for *where* the shot would come from — decision tendency, not just court geometry.
    """
    df = regret.with_columns((pl.col("dist_to_rim") ** 2).alias("dist_to_rim_sq"),
                             pl.col("three_pt").cast(pl.Float64))
    counts = df.group_by("player_id").len()
    keep = set(counts.filter(pl.col("len") >= min_decisions)["player_id"].to_list())
    df = df.with_columns(
        pl.when(pl.col("player_id").is_in(list(keep))).then(pl.col("player_id"))
        .otherwise(pl.lit(-1)).alias("_pid")
    )
    n = df.height
    y = df[value_col].to_numpy().astype(float)
    loc = np.column_stack([df[c].to_numpy().astype(float) for c in _LOC_CONTROLS])
    intercept = np.ones((n, 1))
    X_reduced = np.column_stack([intercept, loc])

    pids = df["_pid"].to_numpy()
    levels = sorted(set(pids.tolist()))
    if len(levels) > 1:
        # drop the first level as the reference category
        dummies = np.column_stack([(pids == lv).astype(float) for lv in levels[1:]])
        X_full = np.column_stack([X_reduced, dummies])
    else:
        X_full = X_reduced

    rss_r, k_r = _ols_rss(X_reduced, y)
    rss_f, k_f = _ols_rss(X_full, y)
    q = k_f - k_r  # added (player) parameters
    df_resid = n - k_f
    if q <= 0 or df_resid <= 0:
        return {"n_decisions": n, "n_player_levels": len(levels), "note": "no estimable player effects"}
    f_stat = ((rss_r - rss_f) / q) / (rss_f / df_resid)
    p_value = float(stats.f.sf(f_stat, q, df_resid))
    tss = float(((y - y.mean()) ** 2).sum())
    return {
        "value_col": value_col, "n_decisions": n,
        "n_players_with_fe": len(levels) - 1 if len(levels) > 1 else 0,
        "r2_location_only": round(1 - rss_r / tss, 4),
        "r2_location_plus_player": round(1 - rss_f / tss, 4),
        "incremental_r2_player": round((rss_r - rss_f) / tss, 4),
        "f_stat": round(float(f_stat), 4), "f_df": [int(q), int(df_resid)],
        "p_value": round(p_value, 8),
    }


def finishing_residual_by_player(shots_with_pmake: pl.DataFrame, *, min_shots: int = 10) -> pl.DataFrame:
    """Per-player finishing skill = mean(made − P(make)) on their REAL shots: how much they
    out/under-perform the population make model. The yardstick G3(b-ii) checks regret against."""
    if shots_with_pmake.height == 0:
        return pl.DataFrame()
    return (
        shots_with_pmake.with_columns(
            (pl.col("made").cast(pl.Float64) - pl.col("p_make")).alias("_resid")
        )
        .group_by(pl.col("shooter_id").alias("player_id"))
        .agg(pl.len().alias("n_shots"), pl.col("_resid").mean().alias("finishing_resid"))
        .filter(pl.col("n_shots") >= min_shots)
    )


def _num(d: dict, key: str, default: float) -> float:
    """Read a numeric stat None-safely. NOT ``d.get(k) or default``: a legitimate 0.0 (an F-test
    p-value below machine epsilon, a zero correlation) is falsy and ``or`` would swap in the
    default — silently turning a decisive pass into a fail. Missing/None falls back; 0.0 survives."""
    v = d.get(key, default)
    return default if v is None else float(v)


def g3_gate(stab_clipped: dict, fe: dict, indep: dict, *, finishing_corr_max: float = 0.40) -> dict:
    """Combine the three G3 statistics into the gate booleans + overall pass.

    PASS = stable (split-half correlation positive AND significant) AND decision-not-location
    (player fixed effects significant beyond shot location) AND decision-not-finishing (per-player
    regret ~uncorrelated with finishing skill).
    """
    gate = {
        "stable_split_half": _num(stab_clipped, "pearson_r", 0.0) > 0
        and _num(stab_clipped, "pearson_p", 1.0) < 0.05,
        "decision_not_location": _num(fe, "p_value", 1.0) < 0.05,
        "decision_not_finishing": abs(_num(indep, "pearson_r", 1.0)) < finishing_corr_max,
    }
    return {"gate": gate, "pass": all(gate.values())}


def finishing_independence(player_regret: pl.DataFrame, finishing: pl.DataFrame,
                           *, regret_col: str = "mean_regret_signed") -> dict:
    """G3(b-ii): correlation of per-player regret with per-player finishing skill.

    Near-zero ⇒ regret is not a relabeled finishing metric (it shouldn't be — xPoints uses a
    population make model with no shooter identity — and this verifies the construction held)."""
    j = player_regret.join(finishing, on="player_id", how="inner")
    if j.height < 3:
        return {"n_players": int(j.height), "note": "too few overlapping players"}
    a = j[regret_col].to_numpy().astype(float)
    b = j["finishing_resid"].to_numpy().astype(float)
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {
        "n_players": int(j.height), "regret_col": regret_col,
        "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6),
    }
