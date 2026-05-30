"""The narrow counterfactual: regret = best available open shot − chosen action value.

This is the bridge from the value layer to the PLOT metric. At each on-ball decision the ball-handler
*could* take an open shot; regret asks how that compares to what they actually did:

    regret = xPoints(open shot from here) − value(chosen action)

following the locked decisions: the available set v1 is the OPEN SHOT ONLY (so ``best_available`` is
the ball-handler's own open look, valued by xPoints at wide-open spacing), and the chosen action is
scored by its model EXPECTED value (decision quality, not luck) — a pass is worth the post-pass EPV
read from the leakage-free OOF trace; a shot is worth its own xPoints. Reported both clipped
(``max(0, ·)``, the headline "points left on the table") and signed (a diagnostic that also credits
good decisions). Regret is only defined at decisions where the handler actually HAD an open look
(nearest defender ≥ OPEN_FT); elsewhere "shoot now" is not in the available set.

Per-player aggregation + the stability gate (G3) come in Stage 5; this module is the per-decision
machinery and is validated on its calibrated inputs (xPoints + post-pass EPV) in Gate G2.
"""

from __future__ import annotations

import polars as pl

from plot.features.court import RIM_X, RIM_Y, region_expr
from plot.models.counterfactual.xpoints import predict_xpoints_open
from plot.possessions.shots import _MAX_DEF_DIST, OPEN_FT, WIDE_OPEN_FT

_THREE_REGIONS = ["corner_3", "above_break_3"]


def decision_states(cdf: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """Ball-handler location + openness at each action's start frame (the decision moment)."""
    keyed = actions.select(
        "possession_id", "action_idx", "action_type", "is_terminal",
        pl.col("actor_player_id").alias("actor"), pl.col("start_wall_ms").alias("wall_clock_ms"),
    ).filter(pl.col("actor").is_not_null())
    players = cdf.filter(pl.col("entity") == "player").select(
        "possession_id", "wall_clock_ms", "player_id", "team_id", "x_canon", "y_canon",
        "offense_team_id", "defense_team_id",
    )
    handler = (
        players.join(keyed, on=["possession_id", "wall_clock_ms"], how="inner")
        .filter(pl.col("player_id") == pl.col("actor"))
        .select("possession_id", "action_idx", "action_type", "is_terminal", "wall_clock_ms",
                pl.col("actor"), pl.col("x_canon").alias("sx"), pl.col("y_canon").alias("sy"),
                "defense_team_id")
    )
    nearest_def = (
        players.drop("defense_team_id", "offense_team_id").join(
            handler.select("possession_id", "action_idx", "wall_clock_ms", "sx", "sy",
                           pl.col("defense_team_id").alias("def_team")),
            on=["possession_id", "wall_clock_ms"], how="inner")
        .filter(pl.col("team_id") == pl.col("def_team"))
        .with_columns((((pl.col("x_canon") - pl.col("sx")) ** 2
                        + (pl.col("y_canon") - pl.col("sy")) ** 2).sqrt()).alias("_d"))
        .group_by("possession_id", "action_idx").agg(pl.col("_d").min().alias("nearest_def_dist"))
    )
    return (
        handler.join(nearest_def, on=["possession_id", "action_idx"], how="left")
        .with_columns(
            pl.col("nearest_def_dist").fill_null(_MAX_DEF_DIST).clip(0.0, _MAX_DEF_DIST).alias("nearest_def_dist"),
            (((pl.col("sx") - RIM_X) ** 2 + (pl.col("sy") - RIM_Y) ** 2).sqrt()).alias("dist_to_rim"),
            region_expr(pl.col("sx"), pl.col("sy")).is_in(_THREE_REGIONS).alias("three_pt"),
        )
        .with_columns(pl.when(pl.col("three_pt")).then(3).otherwise(2).alias("pts_if_made"))
    )


def compute_regret(
    actions: pl.DataFrame,
    cdf: pl.DataFrame,
    xpoints_model,
    epv_trace: pl.DataFrame,
    game_id: str,
) -> pl.DataFrame:
    """Per-decision regret for the v1 narrow set (convenience wrapper: states then core)."""
    return regret_from_states(decision_states(cdf, actions), actions, xpoints_model, epv_trace, game_id)


def regret_from_states(
    ds: pl.DataFrame,
    actions: pl.DataFrame,
    xpoints_model,
    epv_trace: pl.DataFrame,
    game_id: str,
) -> pl.DataFrame:
    """Per-decision regret from precomputed decision states (lets the caller drop the heavy cdf).

    Evaluated at PASS decisions (the ball-handler kept it moving rather than shooting) where the
    handler had an open look. best_available = xPoints of that open shot; chosen = post-pass EPV
    (next action's start EPV from the trace). Returns one row per evaluated decision.
    """
    if ds.height == 0:
        return pl.DataFrame()
    best = predict_xpoints_open(xpoints_model, ds, openness_ft=WIDE_OPEN_FT)
    ds = ds.with_columns(pl.Series("best_available", best),
                         (pl.col("nearest_def_dist") >= OPEN_FT).alias("handler_open"))

    # post-action continuation value = EPV at the NEXT action's start frame
    nxt = actions.select("possession_id", "action_idx", "start_wall_ms").with_columns(
        (pl.col("action_idx") - 1).alias("_prev")
    )
    trace = epv_trace.filter(pl.col("game_id") == game_id).select(
        "possession_id", pl.col("wall_clock_ms").alias("start_wall_ms"), pl.col("epv").alias("post_epv")
    )
    nxt = nxt.join(trace, on=["possession_id", "start_wall_ms"], how="left").select(
        "possession_id", pl.col("_prev").alias("action_idx"), "post_epv"
    )
    ds = ds.join(nxt, on=["possession_id", "action_idx"], how="left")

    out = ds.filter(
        pl.col("handler_open") & (~pl.col("is_terminal")) & (pl.col("action_type") == "pass")
        & pl.col("post_epv").is_not_null()
    ).with_columns(
        (pl.col("best_available") - pl.col("post_epv")).alias("regret_signed"),
        pl.lit(game_id).alias("game_id"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.col("regret_signed")).alias("regret_clipped"),
    )
    return out.select(
        "game_id", "possession_id", "action_idx", pl.col("actor").alias("player_id"),
        "sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist",
        "best_available", "post_epv", "regret_signed", "regret_clipped",
    )


def regret_summary(regret: pl.DataFrame) -> dict:
    """Headline distribution of the per-decision regret (Stage 5 will aggregate per player)."""
    if regret.height == 0:
        return {"n_decisions": 0}
    rs, rc = regret["regret_signed"], regret["regret_clipped"]
    return {
        "n_decisions": regret.height,
        "n_players": regret["player_id"].n_unique(),
        "mean_regret_clipped": round(float(rc.mean()), 4),
        "mean_regret_signed": round(float(rs.mean()), 4),
        "median_regret_signed": round(float(rs.median()), 4),
        "frac_positive_regret": round(float((rs > 0).mean()), 4),
        "p90_regret_clipped": round(float(rc.quantile(0.90)), 4),
        "max_regret_clipped": round(float(rc.max()), 4),
    }
