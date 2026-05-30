"""Observed shots with their location, openness, and outcome — training data for xPoints.

xPoints ("expected points of shooting now") is the well-identified counterfactual at the heart of
Gate G2: it is fit on REAL shots from similar states, so it answers "if this open shot were taken,
what would it be worth?" without a generative rollout. This module extracts those real shots.

Localizing the shot from this bleedy 10 fps tracking is the delicate part. The robust anchor is the
**PBP shot event** itself (msg_type 1 = made FG, 2 = missed FG): the central-event de-bleed labels
every kept frame with the event it sits most centrally inside, and a shot is the central action of
its OWN event — so ``cdf.filter(event_id == shot_event)`` returns the shot's core frames, with the
bled neighbours stripped to other events. Within those, the **release** is the last frontcourt
frame where the PBP shooter (PLAYER1) is holding the ball (<= 2.5 ft); location and openness are
read there. Made and 2-vs-3 come straight from the PBP (authoritative), not from noisy geometry.

Needs the events table (``plot.io.loaders.events_table``) alongside the canonical frames. Pure polars.
"""

from __future__ import annotations

import polars as pl

from plot.features.court import HALF_COURT_X, RIM_X, RIM_Y, region_expr

OPEN_FT = 4.0          # nearest defender >= 4 ft -> "open"  (locked config)
WIDE_OPEN_FT = 6.0     # nearest defender >= 6 ft -> "wide open"
_MAX_DEF_DIST = 15.0   # cap a missing / far nearest-defender distance (uncontested)
_HOLD_FT = 2.5         # offensive player within this of the ball is holding it (release proxy)
_MADE_FG, _MISSED_FG = 1, 2

_OUT_COLS = ["game_id", "possession_id", "event_id", "shooter_id", "shot_wall_ms", "sx", "sy",
             "dist_to_rim", "angle_sin", "angle_cos", "three_pt", "nearest_def_dist", "is_open",
             "is_wide_open", "made", "pts_if_made", "fg_points"]


def extract_shots(cdf: pl.DataFrame, events: pl.DataFrame, game_id: str) -> pl.DataFrame:
    """Return one row per locatable observed field-goal attempt (see module docstring for columns)."""
    shot_ev = events.filter(pl.col("msg_type").is_in([_MADE_FG, _MISSED_FG])).select(
        "event_id",
        pl.col("PLAYER1_ID").cast(pl.Int64).alias("shooter_id"),
        pl.col("PLAYER1_TEAM_ID").cast(pl.Int64).alias("shooter_team"),
        (pl.col("msg_type") == _MADE_FG).alias("made"),
        pl.col("description").str.contains("3PT").fill_null(False).alias("three_pt"),
    )
    if shot_ev.height == 0:
        return pl.DataFrame(schema={c: pl.Float64 for c in _OUT_COLS})

    ball = cdf.filter(pl.col("entity") == "ball").select(
        "event_id", "possession_id", "wall_clock_ms",
        pl.col("x_canon").alias("bx"), pl.col("y_canon").alias("by"),
    ).filter(pl.col("bx") >= HALF_COURT_X)
    players = cdf.filter(pl.col("entity") == "player").select(
        "event_id", "possession_id", "wall_clock_ms", "player_id", "team_id", "x_canon", "y_canon",
    )
    # the shooter holding the ball, frame by frame within the shot's own event
    held = (
        players.join(shot_ev, on="event_id", how="inner")
        .filter(pl.col("player_id") == pl.col("shooter_id"))
        .join(ball.select("event_id", "wall_clock_ms", "bx", "by"),
              on=["event_id", "wall_clock_ms"], how="inner")
        .with_columns((((pl.col("x_canon") - pl.col("bx")) ** 2
                        + (pl.col("y_canon") - pl.col("by")) ** 2).sqrt()).alias("d_ball"))
        .filter(pl.col("d_ball") <= _HOLD_FT)
    )
    rel = held.group_by("event_id").agg(pl.col("wall_clock_ms").max().alias("rel_wall"))
    shooter = (
        held.join(rel, on="event_id", how="inner")
        .filter(pl.col("wall_clock_ms") == pl.col("rel_wall"))
        .group_by("event_id").first()
        .select("event_id", "possession_id", "shooter_id", "shooter_team", "made", "three_pt",
                pl.col("rel_wall").alias("shot_wall_ms"),
                pl.col("x_canon").alias("sx"), pl.col("y_canon").alias("sy"))
    )
    nearest_def = (
        players.join(
            shooter.select("event_id", pl.col("shot_wall_ms").alias("wall_clock_ms"),
                           "shooter_team", "sx", "sy"),
            on=["event_id", "wall_clock_ms"], how="inner",
        )
        .filter(pl.col("team_id") != pl.col("shooter_team"))
        .with_columns((((pl.col("x_canon") - pl.col("sx")) ** 2
                        + (pl.col("y_canon") - pl.col("sy")) ** 2).sqrt()).alias("_d"))
        .group_by("event_id").agg(pl.col("_d").min().alias("nearest_def_dist"))
    )

    angle = pl.arctan2(pl.col("sy") - RIM_Y, RIM_X - pl.col("sx"))
    shots = (
        shooter.join(nearest_def, on="event_id", how="left")
        .with_columns(
            pl.col("nearest_def_dist").fill_null(_MAX_DEF_DIST).clip(0.0, _MAX_DEF_DIST).alias("nearest_def_dist"),
            (((pl.col("sx") - RIM_X) ** 2 + (pl.col("sy") - RIM_Y) ** 2).sqrt()).alias("dist_to_rim"),
            angle.sin().alias("angle_sin"), angle.cos().alias("angle_cos"),
            pl.lit(game_id).alias("game_id"),
            pl.when(pl.col("three_pt")).then(3).otherwise(2).alias("pts_if_made"),
        )
        .with_columns(
            (pl.col("nearest_def_dist") >= OPEN_FT).alias("is_open"),
            (pl.col("nearest_def_dist") >= WIDE_OPEN_FT).alias("is_wide_open"),
            (pl.when(pl.col("made")).then(pl.col("pts_if_made")).otherwise(0)).alias("fg_points"),
        )
    )
    return shots.select(_OUT_COLS)


def shot_region(shots: pl.DataFrame) -> pl.DataFrame:
    """Attach the coarse court region of each shot (for reporting / sensibility breakdowns)."""
    return shots.with_columns(region_expr(pl.col("sx"), pl.col("sy")).alias("region"))
