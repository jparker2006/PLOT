"""v2 regret — the *full available set* at every ball-handler decision (Stage 8).

v1 (``regret.py``) scored one decision type: an open handler who *passed*, against the value of the
open shot they passed up. That is blind to the decision basketball actually argues about — taking a
contested shot when an open teammate was right there — and it can't tell a decisive scorer from an
over-shooter (a player who never passes up an open look is simply never evaluated).

v2 fixes both by widening two knobs:

* **Which decisions:** every ball-handler **shot** and **pass** (not just open pass-ups).
* **The available set:** ``{shoot now (at the handler's ACTUAL contest), pass to each OPEN
  frontcourt teammate (their shot, at their actual contest)}``. ``best_available`` is the max.

Then ``regret = best_available − chosen``, where chosen is the action's model value (a shot → its
own xPoints at the real contest; a pass → the post-pass EPV from the leakage-free trace). A
contested shot taken over an open teammate now scores positive regret — the canonical bad read.

NEW assumption vs v1 (flag, validate G5-style): valuing a pass to an open teammate assumes the pass
arrives and they shoot from there — shakier than the handler's own immediate look. Drives and
"did the pass find the *best* teammate" remain off-distribution future work. Pure over its inputs.
"""

from __future__ import annotations

import polars as pl

from plot.features.court import HALF_COURT_X, RIM_X, RIM_Y, region_expr
from plot.models.counterfactual.xpoints import predict_xpoints
from plot.possessions.shots import _MAX_DEF_DIST, OPEN_FT, WIDE_OPEN_FT

_THREE_REGIONS = ["corner_3", "above_break_3"]
_SHOT_TYPES = ["shot_make", "shot_miss"]
_SCORE_TYPES = ["pass", "shot_make", "shot_miss"]


def offensive_options(cdf: pl.DataFrame, actions: pl.DataFrame) -> pl.DataFrame:
    """One row per (decision, offensive player) at each action's start frame: location, openness,
    3-pt, and whether that player is the ball-handler. The raw material for the available set."""
    keyed = actions.select(
        "possession_id", "action_idx", "action_type", "is_terminal",
        pl.col("actor_player_id").alias("actor"), pl.col("start_wall_ms").alias("wall_clock_ms"),
    ).filter(pl.col("actor").is_not_null())
    players = cdf.filter(pl.col("entity") == "player").select(
        "possession_id", "wall_clock_ms", "player_id", "team_id", "x_canon", "y_canon",
        "offense_team_id", "defense_team_id",
    )
    # offensive players present at each decision frame
    off = players.join(keyed, on=["possession_id", "wall_clock_ms"], how="inner").filter(
        pl.col("team_id") == pl.col("offense_team_id")
    )
    # nearest defender to each offensive player at that frame
    defs = players.select(
        "possession_id", "wall_clock_ms",
        pl.col("x_canon").alias("dx"), pl.col("y_canon").alias("dy"), pl.col("team_id").alias("dteam"),
    )
    near = (
        off.join(defs, on=["possession_id", "wall_clock_ms"], how="inner")
        .filter(pl.col("dteam") == pl.col("defense_team_id"))
        .with_columns((((pl.col("x_canon") - pl.col("dx")) ** 2
                        + (pl.col("y_canon") - pl.col("dy")) ** 2).sqrt()).alias("_d"))
        .group_by("possession_id", "action_idx", "player_id").agg(pl.col("_d").min().alias("nearest_def_dist"))
    )
    return (
        off.join(near, on=["possession_id", "action_idx", "player_id"], how="left")
        .with_columns(
            pl.col("nearest_def_dist").fill_null(_MAX_DEF_DIST).clip(0.0, _MAX_DEF_DIST).alias("nearest_def_dist"),
            (((pl.col("x_canon") - RIM_X) ** 2 + (pl.col("y_canon") - RIM_Y) ** 2).sqrt()).alias("dist_to_rim"),
            region_expr(pl.col("x_canon"), pl.col("y_canon")).is_in(_THREE_REGIONS).alias("three_pt"),
            (pl.col("player_id") == pl.col("actor")).alias("is_handler"),
        )
        .with_columns(pl.when(pl.col("three_pt")).then(3).otherwise(2).alias("pts_if_made"))
    )


def regret_full(
    opts: pl.DataFrame,
    actions: pl.DataFrame,
    xpoints_model,
    epv_trace: pl.DataFrame,
    game_id: str,
    *,
    open_ft: float = OPEN_FT,
    teammate_min_x: float = HALF_COURT_X,
) -> pl.DataFrame:
    """Per-decision v2 regret over shots + passes (see module docstring). One row per scored decision."""
    if opts.height == 0:
        return pl.DataFrame()
    _, xp = predict_xpoints(xpoints_model, opts)
    opts = opts.with_columns(pl.Series("xpoints", xp))

    handler = opts.filter(pl.col("is_handler")).select(
        "possession_id", "action_idx", "action_type", "is_terminal", "actor", "wall_clock_ms",
        pl.col("xpoints").alias("own_shot_xp"), "dist_to_rim", "three_pt", "nearest_def_dist",
        pl.col("x_canon").alias("sx"), pl.col("y_canon").alias("sy"),
    )
    teammates = (
        opts.filter((~pl.col("is_handler")) & (pl.col("nearest_def_dist") >= open_ft)
                    & (pl.col("x_canon") >= teammate_min_x))
        .group_by("possession_id", "action_idx")
        .agg(pl.col("xpoints").max().alias("best_teammate_xp"), pl.len().alias("open_teammates"))
    )
    d = handler.join(teammates, on=["possession_id", "action_idx"], how="left").with_columns(
        pl.col("best_teammate_xp").fill_null(0.0), pl.col("open_teammates").fill_null(0)
    )

    # post-pass EPV for pass decisions: EPV at the NEXT action's start frame (same as v1)
    nxt = actions.select("possession_id", "action_idx", "start_wall_ms").with_columns(
        (pl.col("action_idx") - 1).alias("_prev")
    )
    trace = epv_trace.filter(pl.col("game_id") == game_id).select(
        "possession_id", pl.col("wall_clock_ms").alias("start_wall_ms"), pl.col("epv").alias("post_epv")
    )
    nxt = nxt.join(trace, on=["possession_id", "start_wall_ms"], how="left").select(
        "possession_id", pl.col("_prev").alias("action_idx"), "post_epv"
    )
    d = d.join(nxt, on=["possession_id", "action_idx"], how="left")

    is_shot = pl.col("action_type").is_in(_SHOT_TYPES)
    is_pass = pl.col("action_type") == "pass"
    d = d.filter(pl.col("action_type").is_in(_SCORE_TYPES)).with_columns(
        pl.max_horizontal(pl.col("own_shot_xp"), pl.col("best_teammate_xp")).alias("best_available"),
        pl.when(is_shot).then(pl.col("own_shot_xp")).when(is_pass).then(pl.col("post_epv")).alias("chosen"),
        pl.when(is_shot).then(pl.lit("shot")).otherwise(pl.lit("pass")).alias("decision_kind"),
    ).filter(pl.col("chosen").is_not_null())  # passes need a valid post-pass EPV

    d = d.with_columns(
        (pl.col("best_available") - pl.col("chosen")).alias("regret_signed"),
        pl.lit(game_id).alias("game_id"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.col("regret_signed")).alias("regret_clipped")
    )
    return d.select(
        "game_id", "possession_id", "action_idx", pl.col("actor").alias("player_id"), "decision_kind",
        "sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist",
        "own_shot_xp", "best_teammate_xp", "open_teammates", "best_available", "chosen",
        "regret_signed", "regret_clipped",
    )


def shot_selection_regret(
    opts: pl.DataFrame,
    xpoints_model,
    game_id: str,
    *,
    wide_open_ft: float = WIDE_OPEN_FT,
    max_pass_ft: float = 28.0,
    completion: float = 0.80,
    teammate_min_x: float = HALF_COURT_X,
) -> pl.DataFrame:
    """v1.5 *shot-selection* regret — the surgical, conservative half of v2 (the v2 pass-decision
    scoring is dropped; the v2 "any open teammate" optimism is tightened).

    At each ball-handler SHOT, the only alternative considered is the **best WIDE-OPEN
    (≥ ``wide_open_ft``), frontcourt, reachable (≤ ``max_pass_ft``) teammate**, valued at their shot
    and discounted by a flat pass-``completion`` factor (the pass must arrive and they must shoot).
    ``regret_signed = completion·alt − own_shot`` (negative ⇒ the shot was the right call); clipped is
    the positive part — points left by shooting over a clearly-better open look. Schema matches
    ``regret.regret_from_states`` (``post_epv`` holds the chosen shot's value) so the two concatenate.
    """
    if opts.height == 0:
        return pl.DataFrame()
    _, xp = predict_xpoints(xpoints_model, opts)
    opts = opts.with_columns(pl.Series("xpoints", xp))
    handler = opts.filter(pl.col("is_handler") & pl.col("action_type").is_in(_SHOT_TYPES)).select(
        "possession_id", "action_idx", "actor",
        pl.col("xpoints").alias("own_shot_xp"),
        pl.col("x_canon").alias("sx"), pl.col("y_canon").alias("sy"),
        "dist_to_rim", "three_pt", "nearest_def_dist",
    )
    if handler.height == 0:
        return pl.DataFrame()
    tm = (
        opts.filter((~pl.col("is_handler")) & pl.col("action_type").is_in(_SHOT_TYPES)
                    & (pl.col("nearest_def_dist") >= wide_open_ft) & (pl.col("x_canon") >= teammate_min_x))
        .join(handler.select("possession_id", "action_idx", "sx", "sy"),
              on=["possession_id", "action_idx"], how="inner")
        .with_columns((((pl.col("x_canon") - pl.col("sx")) ** 2
                        + (pl.col("y_canon") - pl.col("sy")) ** 2).sqrt()).alias("_pd"))
        .filter(pl.col("_pd") <= max_pass_ft)
        .group_by("possession_id", "action_idx").agg(pl.col("xpoints").max().alias("best_tm"))
    )
    d = handler.join(tm, on=["possession_id", "action_idx"], how="left").with_columns(
        pl.col("best_tm").fill_null(0.0)
    ).with_columns(
        (completion * pl.col("best_tm")).alias("alt")
    ).with_columns(
        (pl.col("alt") - pl.col("own_shot_xp")).alias("regret_signed"),
        pl.max_horizontal(pl.col("own_shot_xp"), pl.col("alt")).alias("best_available"),
        pl.col("own_shot_xp").alias("post_epv"),  # the chosen shot's value, in v1's schema slot
        pl.lit(game_id).alias("game_id"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.col("regret_signed")).alias("regret_clipped")
    )
    return d.select(
        "game_id", "possession_id", "action_idx", pl.col("actor").alias("player_id"),
        "sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist",
        "best_available", "post_epv", "regret_signed", "regret_clipped",
    )
