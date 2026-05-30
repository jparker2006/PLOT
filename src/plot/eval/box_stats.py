"""Per-player box-score stats from play-by-play, and the Gate G4 test: is the PLOT metric *new*
information, or just box-score efficiency / usage relabeled?

G4 asks the question a skeptic asks of any new basketball number — "isn't this just shooting
efficiency (or volume, or scoring) with extra steps?" It has two parts:

* **Incremental information.** Per-player PLOT (points left on the table / 100 open pass-up
  decisions) should be only weakly correlated with the box-score quantities it could be confused
  for — true-shooting % (efficiency), scoring possessions per game (usage / role), points per game
  (output), field-goal attempts per game (shot volume). Low ``|corr|`` ⇒ PLOT carries something the
  box score does not.
* **Reliability.** The metric must repeat across independent samples — already established by
  G3(a) (split-half 0.51, Spearman–Brown 0.68). G4 *reuses* that number rather than recomputing it
  (the per-decision regret for the 208-game run is not local), and pairs it with incremental info:
  a reliable signal that is also distinct from the box score is, by definition, new information.

The box stats are **season-long** (the conventional unit for a player's efficiency/usage
reputation) and **SCORE-column-independent** in exactly the way segmentation is: points come from
event semantics (made FG = 2, +1 if the description says "3PT"; made FT = 1), never the
orientation-inconsistent SCORE column (see ``possessions.segment``).

Pure over its DataFrame inputs (no I/O, no model training) so every statistic is unit-testable.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

# EVENTMSGTYPE codes (shared with possessions.segment)
_MADE_FG, _MISSED_FG, _FREE_THROW, _TURNOVER = 1, 2, 3, 5
# NBA player ids are small (≤ ~1.7e6); team entities (team rebounds / team turnovers) carry the
# ~1.61e9 team id in PLAYER1_ID — this bound drops them without trusting PERSON1TYPE.
_MAX_PLAYER_ID = 1_000_000_000


def normalize_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Tidy the columns box stats need from raw stats.nba.com PBP (coalesced upper-case description,
    PLAYER1 as the acting player). Mirrors ``loaders.events_table`` but season-wide and lighter."""
    return pbp.select(
        pl.col("GAME_ID").alias("game_id"),
        pl.col("EVENTMSGTYPE").alias("msg_type"),
        pl.col("PLAYER1_ID").alias("player_id"),
        pl.col("PLAYER1_NAME").alias("player_name"),
        pl.coalesce("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION").alias("description"),
    ).with_columns(
        pl.col("description").fill_null("").str.to_uppercase().alias("desc_upper")
    )


def per_player_box_stats(pbp_norm: pl.DataFrame, *, min_fga_fta: int = 50) -> pl.DataFrame:
    """Season-long per-player box stats from normalized PBP, for players with ≥ ``min_fga_fta``
    true-shooting attempts (FGA + FTA) so the rate stats are stable.

    Columns: ``player_id, player_name, gp, fgm, fga, fg3m, fta, ftm, tov, pts, ts_pct,
    usg_proxy_pg, pts_pg, fga_pg``. ``ts_pct`` is null for players with no true-shooting attempts.
    """
    if pbp_norm.height == 0:
        return pl.DataFrame()
    made_fg = pl.col("msg_type") == _MADE_FG
    fga = pl.col("msg_type").is_in([_MADE_FG, _MISSED_FG])
    fg3m = made_fg & pl.col("desc_upper").str.contains("3PT")
    ft = pl.col("msg_type") == _FREE_THROW
    ftm = ft & ~pl.col("desc_upper").str.contains("MISS")
    tov = pl.col("msg_type") == _TURNOVER

    players = pbp_norm.filter(
        pl.col("player_id").is_not_null()
        & (pl.col("player_id") > 0)
        & (pl.col("player_id") < _MAX_PLAYER_ID)
    )
    agg = (
        players.group_by("player_id")
        .agg(
            pl.col("player_name").drop_nulls().first().alias("player_name"),
            pl.col("game_id").n_unique().alias("gp"),
            made_fg.cast(pl.Int64).sum().alias("fgm"),
            fga.cast(pl.Int64).sum().alias("fga"),
            fg3m.cast(pl.Int64).sum().alias("fg3m"),
            ft.cast(pl.Int64).sum().alias("fta"),
            ftm.cast(pl.Int64).sum().alias("ftm"),
            tov.cast(pl.Int64).sum().alias("tov"),
        )
        # PTS from event semantics: 2 per made FG, +1 for each made 3, +1 per made FT.
        .with_columns((2 * pl.col("fgm") + pl.col("fg3m") + pl.col("ftm")).alias("pts"))
        .with_columns(
            (pl.col("fga") + 0.44 * pl.col("fta")).alias("_tsa"),
            (pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov")).alias("_scoring_poss"),
        )
        .with_columns(
            pl.when(pl.col("_tsa") > 0)
            .then(pl.col("pts") / (2 * pl.col("_tsa")))
            .otherwise(None)
            .alias("ts_pct"),
            (pl.col("_scoring_poss") / pl.col("gp")).alias("usg_proxy_pg"),
            (pl.col("pts") / pl.col("gp")).alias("pts_pg"),
            (pl.col("fga") / pl.col("gp")).alias("fga_pg"),
        )
        .drop("_tsa", "_scoring_poss")
        .filter((pl.col("fga") + pl.col("fta")) >= min_fga_fta)
        .sort("ts_pct", descending=True, nulls_last=True)
    )
    return agg


def _num(d: dict, key: str, default: float) -> float:
    """Read a numeric stat None-safely. NOT ``d.get(k) or default``: a legitimately-falsy 0.0 (a
    zero correlation, a p-value below machine epsilon) would be swapped for the default by ``or``,
    silently flipping a verdict. Missing/None falls back; 0.0 survives. (Same lesson as the G3 gate.)
    """
    v = d.get(key, default)
    return default if v is None else float(v)


def _corr(a: np.ndarray, b: np.ndarray) -> dict:
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {
        "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
        "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6),
    }


def correlate_plot_with_box(joined: pl.DataFrame, *, target: str, stat_cols: list[str]) -> dict:
    """Pearson + Spearman of the per-player PLOT ``target`` against each box-stat column, on the
    rows where both are present. Returns ``{stat: {n, pearson_r, ...}}``."""
    out: dict[str, dict] = {}
    for c in stat_cols:
        sub = joined.select([target, c]).drop_nulls()
        if sub.height < 3:
            out[c] = {"n": int(sub.height), "note": "too few overlapping players"}
            continue
        a = sub[target].to_numpy().astype(float)
        b = sub[c].to_numpy().astype(float)
        out[c] = {"n": int(sub.height), **_corr(a, b)}
    return out


def variance_explained(joined: pl.DataFrame, *, target: str, predictors: list[str]) -> dict:
    """How much of per-player PLOT variance a *joint* OLS on the box-stat ``predictors`` explains —
    the strongest single "is PLOT redundant?" number. Small R² ⇒ the box score barely reconstructs
    PLOT. Reports R² and the sample-size-corrected adjusted R²."""
    sub = joined.select([target, *predictors]).drop_nulls()
    k = len(predictors) + 1
    if sub.height <= k + 1:
        return {"n": int(sub.height), "predictors": predictors, "note": "too few rows for the fit"}
    y = sub[target].to_numpy().astype(float)
    X = np.column_stack([np.ones(sub.height), *[sub[p].to_numpy().astype(float) for p in predictors]])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    n = sub.height
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else float("nan")
    return {"n": int(n), "predictors": predictors, "r2": round(float(r2), 4), "adj_r2": round(float(adj), 4)}


def g4_gate(
    corrs: dict,
    reliability_sb: float | None,
    *,
    corr_max: float = 0.50,
    reliability_min: float = 0.60,
    gate_stats: list[str] | None = None,
) -> dict:
    """Combine the incremental-info correlations + the (reused) split-half reliability into the gate.

    PASS = every gated ``|Pearson(PLOT, box stat)|`` is below ``corr_max`` (incremental information)
    AND the Spearman–Brown reliability is at least ``reliability_min`` (the metric repeats).
    """
    gate_stats = gate_stats if gate_stats is not None else list(corrs)
    per_stat = {
        s: abs(_num(corrs.get(s, {}), "pearson_r", 1.0)) < corr_max for s in gate_stats
    }
    gate = {
        "incremental_info": all(per_stat.values()) if per_stat else False,
        "reliable": reliability_sb is not None and float(reliability_sb) >= reliability_min,
    }
    return {
        "per_stat_below_corr_max": per_stat,
        "corr_max": corr_max,
        "reliability_sb": None if reliability_sb is None else round(float(reliability_sb), 4),
        "reliability_min": reliability_min,
        "gate": gate,
        "pass": all(gate.values()),
    }
