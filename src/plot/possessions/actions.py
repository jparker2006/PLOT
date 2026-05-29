"""Per-possession on-ball ACTIONS from tracking + segmentation — the SPADL-analog for Stage 3.

The per-action value layer needs to partition each possession into contiguous on-ball actions so
that ``value(action) = EPV(end) - EPV(start)`` telescopes to ``realized - initial`` over the whole
possession. We ground the partition in the one signal tracking gives reliably: **who has the ball**.

Per ~10 fps frame, the ball-handler is the offensive player nearest the ball within
``HANDLER_BALL_MAX_FT`` (the same rule the eval-bar features use). The handler id is then
forward-filled within a possession (so the brief in-flight frames of a pass stay credited to the
*passer*) and back-filled at the start (pre-catch inbound frames). A new action begins whenever the
(filled) handler id changes; the run of frames a single player holds the ball is one action.

Action typing: a non-terminal segment ends because the ball left for a teammate -> ``pass``. The
LAST segment of a possession is the **terminal** action, typed from the segmenter's ``end_reason``
(made shot / missed shot / turnover / drawn foul / ...). This is a deliberately coarse v1: it does
not yet split an intermediate missed-shot-then-offensive-rebound (it reads as a ``pass`` to the
rebounder), and it attributes the terminal to the tracking ball-handler rather than the PBP
shooter. Both are noted refinements; neither affects the telescoping decomposition.

Pure polars — no model, no EPV. The value layer (``plot.models.action_value``) attaches an EPV
trace to these actions; the action table is model-agnostic and fully unit-testable.
"""

from __future__ import annotations

import polars as pl

from plot.features.eval_bar import HANDLER_BALL_MAX_FT

# terminal action type from the possession's end_reason (see plot.possessions.segment)
_TERMINAL_TYPE = {
    "made_fg": "shot_make",
    "defensive_rebound": "shot_miss",
    "turnover": "turnover",
    "made_ft": "foul_drawn",
    "end_period": "period_end",
    "jump_ball": "jump_ball",
    "control_change": "control_change",
}
_NO_HANDLER = -1  # sentinel for frames with no offensive player within reach of the ball


def ball_handler_trace(cdf: pl.DataFrame) -> pl.DataFrame:
    """Per-frame ball-handler: the offensive player nearest the ball within HANDLER_BALL_MAX_FT.

    ``cdf`` is canonicalized long moments (``plot.models.eval_bar.seq_dataset.build_game_canonical``).
    Returns one row per (possession_id, wall_clock_ms): handler_player_id (null if none in reach).
    """
    ball = cdf.filter(pl.col("entity") == "ball").select(
        "possession_id", "wall_clock_ms",
        pl.col("x_canon").alias("_bx"), pl.col("y_canon").alias("_by"),
    )
    frames = ball.select("possession_id", "wall_clock_ms")
    players = (
        cdf.filter(pl.col("entity") == "player")
        .join(ball, on=["possession_id", "wall_clock_ms"], how="inner")
        .filter(pl.col("team_id") == pl.col("offense_team_id"))
        .with_columns(
            (((pl.col("x_canon") - pl.col("_bx")) ** 2 + (pl.col("y_canon") - pl.col("_by")) ** 2)
             .sqrt()).alias("_dist")
        )
        .filter(pl.col("_dist") <= HANDLER_BALL_MAX_FT)
        .sort("_dist")
        .group_by("possession_id", "wall_clock_ms")
        .first()
        .select("possession_id", "wall_clock_ms", pl.col("player_id").alias("handler_player_id"))
    )
    return frames.join(players, on=["possession_id", "wall_clock_ms"], how="left")


def extract_actions(cdf: pl.DataFrame, possessions: pl.DataFrame, game_id: str) -> pl.DataFrame:
    """Segment each possession into contiguous on-ball actions; return the action table.

    Columns: game_id, possession_id, action_idx, actor_player_id, action_type, is_terminal,
    start_wall_ms, end_wall_ms, n_frames, offense_team_id, end_reason, points.
    """
    trace = ball_handler_trace(cdf)
    # forward/back-fill the handler within a possession, then segment on handler changes
    h = (
        trace.sort("possession_id", "wall_clock_ms")
        .with_columns(
            pl.col("handler_player_id").forward_fill().over("possession_id").alias("_h")
        )
        .with_columns(pl.col("_h").backward_fill().over("possession_id").alias("_h"))
        .with_columns(pl.col("_h").fill_null(_NO_HANDLER).alias("_h"))
    )
    h = h.with_columns(
        (pl.col("_h") != pl.col("_h").shift(1).over("possession_id")).fill_null(True).alias("_change")
    ).with_columns(
        (pl.col("_change").cast(pl.Int64).cum_sum().over("possession_id") - 1).alias("action_idx")
    )

    segs = (
        h.group_by("possession_id", "action_idx", maintain_order=True)
        .agg(
            pl.col("_h").first().alias("actor_player_id"),
            pl.col("wall_clock_ms").min().alias("start_wall_ms"),
            pl.col("wall_clock_ms").max().alias("end_wall_ms"),
            pl.len().alias("n_frames"),
        )
    )
    last_idx = segs.group_by("possession_id").agg(pl.col("action_idx").max().alias("_last"))
    poss = possessions.select(
        "possession_id", "offense_team_id", "end_reason",
        pl.col("points").alias("points"),
    )
    out = (
        segs.join(last_idx, on="possession_id", how="left")
        .join(poss, on="possession_id", how="left")
        .with_columns((pl.col("action_idx") == pl.col("_last")).alias("is_terminal"))
        .with_columns(
            pl.when(pl.col("is_terminal"))
            .then(pl.col("end_reason").replace_strict(_TERMINAL_TYPE, default="other"))
            .otherwise(pl.lit("pass"))
            .alias("action_type"),
            pl.when(pl.col("actor_player_id") == _NO_HANDLER)
            .then(None).otherwise(pl.col("actor_player_id")).alias("actor_player_id"),
            pl.lit(game_id).alias("game_id"),
        )
    )
    return out.select(
        "game_id", "possession_id", "action_idx", "actor_player_id", "action_type", "is_terminal",
        "start_wall_ms", "end_wall_ms", "n_frames", "offense_team_id", "end_reason", "points",
    ).sort("possession_id", "action_idx")
