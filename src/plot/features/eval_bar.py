"""Per-moment feature matrix for the baseline EPV ("eval bar").

Single source of truth for the eval-bar features (so training, evaluation, and the demo export
all share identical columns). Built on CLEANED + CANONICALIZED moments: each kept ~10 fps frame
becomes one wide row of coarse, DECISION-INSENSITIVE, identity-free context — ball geometry to
the attacked rim, a region prior, clocks/period, coarse spacing/pressure counts, and a single
ball-handler->nearest-defender pressure scalar. Label = the points the moment's possession
ultimately yields (a shared {0,1,2,3} target). No player ids, no shot-type / chosen-action /
shot-made columns: this is a location/context prior that validates calibration plumbing; the
decision-sensitive signal and regret arrive in Stage 3+.

Every feature is instantaneous state at frame t (no future leakage). The two careful features —
``score_diff`` (strictly pre-possession, SCORE-column-independent) and ``ball_speed`` (a backward
difference over past frames only) — are flagged in their construction and guarded by tests.
"""

from __future__ import annotations

import polars as pl

from plot.features.court import (
    COURT_LENGTH,
    COURT_WIDTH,
    RIM_X,
    RIM_Y,
    THREE_POINT_RADIUS,
    in_paint_expr,
    in_restricted_area_expr,
    region_expr,
)
from plot.features.orientation import attacked_rim_expr, canonicalize_xy

PERIOD_SECONDS = 720.0          # 12-minute quarters
HANDLER_BALL_MAX_FT = 6.0       # an offensive player within this of the ball is the "ball-handler"
OOB_DROP_FT = 5.0               # ball observed out of bounds by more than this -> drop the frame

# The frozen feature contract (single source of truth). NO player/team ids; NO action/outcome.
FEATURE_COLUMNS = [
    "ball_x_canon", "ball_y_canon", "ball_z",
    "ball_dist_to_rim", "ball_angle_sin", "ball_angle_cos",
    "court_region", "in_paint", "in_restricted_area", "beyond_arc", "in_corner",
    "game_clock_sec", "shot_clock_sec", "shot_clock_missing", "seconds_remaining_in_half",
    "period", "is_overtime",
    "score_diff", "is_clutch",
    "n_offense_in_paint", "n_defense_in_paint", "n_defense_in_ra", "players_tracked",
    "ballhandler_dist_to_nearest_defender", "nearest_defender_to_rim_dist", "handler_undefined",
    "prev_possession_end_reason",
    "ball_speed", "ball_oob",
]
CATEGORICAL_COLUMNS = ["court_region", "prev_possession_end_reason"]
LABEL_COLUMN = "points"
GROUP_COLUMN = "game_id"
WEIGHT_COLUMN = "weight"
META_COLUMNS = [GROUP_COLUMN, "possession_id", "wall_clock_ms", LABEL_COLUMN, WEIGHT_COLUMN]


def _score_diff_by_possession(possessions: pl.DataFrame) -> pl.DataFrame:
    """Pre-possession score margin (offense - defense), SCORE-column-independent.

    Prefix-sum of the segmenter's per-possession ``points`` by team over possessions that fully
    ended before the current one started; signed for the current offense. Never includes the
    current possession's own points (that is the label).
    """
    rows = possessions.sort("period", "start_event_id").select(
        "possession_id", "offense_team_id", "defense_team_id", "points"
    ).iter_rows(named=True)
    cum: dict[int, int] = {}
    recs = []
    for p in rows:
        o, d = p["offense_team_id"], p["defense_team_id"]
        recs.append({"possession_id": p["possession_id"], "score_diff": cum.get(o, 0) - cum.get(d, 0)})
        cum[o] = cum.get(o, 0) + (p["points"] or 0)
    return pl.DataFrame(recs, schema={"possession_id": pl.Int64, "score_diff": pl.Int64})


def canonicalize(deduped: pl.DataFrame, rim_map: dict[tuple[int, int], str]) -> pl.DataFrame:
    """Add ``x_canon``/``y_canon`` (offense attacks the right rim) and an ``oob`` flag; drop far-OOB ball.

    Ball coordinates are clamped into the court with an ``oob`` flag; a ball observed out of bounds
    by more than :data:`OOB_DROP_FT` is dropped (whole frame) as untrustworthy.
    """
    ar = attacked_rim_expr(pl.col("offense_team_id"), pl.col("quarter"), rim_map)
    is_ball = pl.col("entity") == "ball"
    far_oob = is_ball & (
        (pl.col("x") < -OOB_DROP_FT) | (pl.col("x") > COURT_LENGTH + OOB_DROP_FT)
        | (pl.col("y") < -OOB_DROP_FT) | (pl.col("y") > COURT_WIDTH + OOB_DROP_FT)
    )
    bad_frames = deduped.filter(far_oob).select("possession_id", "wall_clock_ms").unique()
    df = deduped.join(bad_frames, on=["possession_id", "wall_clock_ms"], how="anti")

    oob = is_ball & (
        (pl.col("x") < 0) | (pl.col("x") > COURT_LENGTH) | (pl.col("y") < 0) | (pl.col("y") > COURT_WIDTH)
    )
    x_clamped = pl.col("x").clip(0.0, COURT_LENGTH)
    y_clamped = pl.col("y").clip(0.0, COURT_WIDTH)
    x_c, y_c = canonicalize_xy(x_clamped, y_clamped, ar)
    return df.with_columns(
        ar.alias("attacked_rim"),
        x_c.alias("x_canon"),
        y_c.alias("y_canon"),
        pl.when(is_ball).then(oob).otherwise(False).alias("oob"),
    )


def build_features(
    deduped: pl.DataFrame,
    possessions: pl.DataFrame,
    rim_map: dict[tuple[int, int], str],
    game_id: str,
) -> pl.DataFrame:
    """Assemble the wide per-frame feature matrix (one row per kept ~10 fps moment)."""
    df = canonicalize(deduped, rim_map)
    frame_keys = ["possession_id", "wall_clock_ms"]

    ball = df.filter(pl.col("entity") == "ball").select(
        *frame_keys, "quarter", "game_clock", "shot_clock", "offense_team_id", "defense_team_id",
        "points", "end_reason",
        pl.col("x_canon").alias("ball_x_canon"),
        pl.col("y_canon").alias("ball_y_canon"),
        pl.col("z").alias("ball_z"),
        pl.col("oob").alias("ball_oob"),
    )

    players = (
        df.filter(pl.col("entity") == "player")
        .join(ball.select(*frame_keys, "ball_x_canon", "ball_y_canon"), on=frame_keys, how="left")
        .with_columns(
            (pl.col("team_id") == pl.col("offense_team_id")).alias("is_off"),
            (pl.col("team_id") == pl.col("defense_team_id")).alias("is_def"),
            ((pl.col("x_canon") - pl.col("ball_x_canon")) ** 2
             + (pl.col("y_canon") - pl.col("ball_y_canon")) ** 2).sqrt().alias("dist_to_ball"),
        )
    )

    in_paint = in_paint_expr(pl.col("x_canon"), pl.col("y_canon"))
    in_ra = in_restricted_area_expr(pl.col("x_canon"), pl.col("y_canon"))
    counts = players.group_by(frame_keys).agg(
        pl.len().alias("players_tracked"),
        pl.col("is_def").sum().alias("n_def_tracked"),
        (pl.col("is_off") & in_paint).sum().alias("n_offense_in_paint"),
        (pl.col("is_def") & in_paint).sum().alias("n_defense_in_paint"),
        (pl.col("is_def") & in_ra).sum().alias("n_defense_in_ra"),
    )

    # ball-handler = offensive player nearest the ball (must be within HANDLER_BALL_MAX_FT)
    handler = (
        players.filter(pl.col("is_off"))
        .sort("dist_to_ball")
        .group_by(frame_keys)
        .first()
        .select(*frame_keys, pl.col("x_canon").alias("h_x"), pl.col("y_canon").alias("h_y"),
                pl.col("dist_to_ball").alias("h_ball_dist"))
    )
    defenders = (
        players.filter(pl.col("is_def"))
        .join(handler, on=frame_keys, how="inner")
        .with_columns(
            (((pl.col("x_canon") - pl.col("h_x")) ** 2 + (pl.col("y_canon") - pl.col("h_y")) ** 2).sqrt())
            .alias("d_to_h"),
            (((pl.col("x_canon") - RIM_X) ** 2 + (pl.col("y_canon") - RIM_Y) ** 2).sqrt()).alias("d_to_rim"),
        )
        .sort("d_to_h")
        .group_by(frame_keys)
        .first()
        .select(*frame_keys, pl.col("d_to_h").alias("ballhandler_dist_to_nearest_defender"),
                pl.col("d_to_rim").alias("nearest_defender_to_rim_dist"))
    )

    score_diff = _score_diff_by_possession(possessions)
    prev_reason = possessions.select("possession_id", "previous_possession_end_reason")
    poss_weight = (
        ball.group_by("possession_id").agg(pl.len().alias("_n"))
        .with_columns((1.0 / pl.col("_n")).alias("weight")).select("possession_id", "weight")
    )

    feats = (
        ball.join(counts, on=frame_keys, how="left")
        .join(handler.select(*frame_keys, "h_ball_dist"), on=frame_keys, how="left")
        .join(defenders, on=frame_keys, how="left")
        .join(score_diff, on="possession_id", how="left")
        .join(prev_reason, on="possession_id", how="left")
        .join(poss_weight, on="possession_id", how="left")
    )

    angle = pl.arctan2(pl.col("ball_y_canon") - RIM_Y, RIM_X - pl.col("ball_x_canon"))
    half_idx = pl.when(pl.col("quarter") <= 2).then(2 - pl.col("quarter")).otherwise(
        pl.when(pl.col("quarter") <= 4).then(4 - pl.col("quarter")).otherwise(0)
    )
    dist_rim = ((pl.col("ball_x_canon") - RIM_X) ** 2 + (pl.col("ball_y_canon") - RIM_Y) ** 2).sqrt()
    feats = feats.with_columns(
        dist_rim.alias("ball_dist_to_rim"),
        angle.sin().alias("ball_angle_sin"),
        angle.cos().alias("ball_angle_cos"),
        region_expr(pl.col("ball_x_canon"), pl.col("ball_y_canon")).alias("court_region"),
        in_paint_expr(pl.col("ball_x_canon"), pl.col("ball_y_canon")).alias("in_paint"),
        in_restricted_area_expr(pl.col("ball_x_canon"), pl.col("ball_y_canon")).alias("in_restricted_area"),
        pl.col("game_clock").alias("game_clock_sec"),
        pl.col("shot_clock").alias("shot_clock_sec"),
        pl.col("shot_clock").is_null().alias("shot_clock_missing"),
        (half_idx * PERIOD_SECONDS + pl.col("game_clock")).alias("seconds_remaining_in_half"),
        pl.col("quarter").alias("period"),
        (pl.col("quarter") > 4).alias("is_overtime"),
        pl.col("previous_possession_end_reason").fill_null("none").alias("prev_possession_end_reason"),
        (pl.col("h_ball_dist") > HANDLER_BALL_MAX_FT).fill_null(True).alias("handler_undefined"),
        pl.lit(game_id).alias("game_id"),
    ).with_columns(
        (pl.col("ball_dist_to_rim") >= THREE_POINT_RADIUS).alias("beyond_arc"),
        (pl.col("court_region") == "corner_3").alias("in_corner"),
        (
            (pl.col("period") >= 4) & (pl.col("game_clock_sec") <= 300) & (pl.col("score_diff").abs() <= 5)
        ).alias("is_clutch"),
    )

    # ball_speed: BACKWARD difference on the per-possession time-sorted series (past frames only)
    feats = feats.sort("possession_id", "wall_clock_ms").with_columns(
        (((pl.col("ball_x_canon").diff() ** 2 + pl.col("ball_y_canon").diff() ** 2).sqrt())
         / ((pl.col("wall_clock_ms").diff() / 1000.0).clip(0.001, None)))
        .over("possession_id").alias("ball_speed")
    )

    # pressure features are undefined when the handler is undefined or defenders are missing
    feats = feats.with_columns(
        pl.when(pl.col("handler_undefined") | (pl.col("n_def_tracked") < 2))
        .then(None).otherwise(pl.col("ballhandler_dist_to_nearest_defender"))
        .alias("ballhandler_dist_to_nearest_defender"),
        pl.when(pl.col("handler_undefined") | (pl.col("n_def_tracked") < 2))
        .then(None).otherwise(pl.col("nearest_defender_to_rim_dist"))
        .alias("nearest_defender_to_rim_dist"),
    )

    return feats.select(*FEATURE_COLUMNS, *META_COLUMNS)
