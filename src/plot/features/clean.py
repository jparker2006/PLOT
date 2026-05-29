"""Dedup the boundary bleed, attach the live possession, and downsample to ~10 fps.

The hard data reality (verified across the 14 local games; see project memory): each event's
moment buffer bleeds heavily into its neighbours, so a physical instant ``(quarter,
wall_clock_ms)`` appears ~2.4-2.7x across adjacent events — carrying the SAME position each time
(it is the same instant). Worse, the tracking ``game_clock`` and the PBP ``PCTIMESTRING`` are
offset by a large, variable amount (~25 s here), so clock-matching frames to PBP-derived
possession windows mislabels ~⅓ of them.

The robust fix uses two reliable facts: (1) the segmenter's ``event_id -> possession_id`` map,
and (2) the bleed lives at the buffer EDGES (an event's own action sits deep inside its array,
the bled neighbours at the ends). So we assign each physical instant to the event in which it is
most CENTRAL (farthest from either array edge) and inherit that event's possession — no clock
alignment needed. Then we downsample 25->~10 fps by keeping one instant per 100 ms wall-clock
bucket within a possession (time-bucketing, robust to dropped frames).

This is correct for ~10/14 local games (rim-frame canonicalization error 1-15%); a few heavy-bleed
games stay noisy and are caught downstream by the backcourt-fraction QC quarantine
(:func:`backcourt_fraction`). Operates per game in isolation, so splits stay structurally safe.
"""

from __future__ import annotations

import polars as pl

_POSS_LABEL_COLS = [
    "offense_team_id", "defense_team_id", "counts_as_possession",
    "points", "end_reason", "previous_possession_end_reason",
]


def central_event_per_frame(moments: pl.DataFrame) -> pl.DataFrame:
    """Map each physical ``(quarter, wall_clock_ms)`` to the event where it is most central.

    Centrality = distance from either end of that event's moment array; ties broken by smallest
    event_id (deterministic). Returns one row per physical instant: (quarter, wall_clock_ms, event_id).
    """
    ball = moments.filter((pl.col("entity") == "ball") & pl.col("game_clock").is_not_null())
    n_mom = ball.group_by("event_id").agg(pl.col("moment_idx").max().alias("_mmax"))
    cand = (
        ball.select("quarter", "wall_clock_ms", "event_id", "moment_idx")
        .join(n_mom, on="event_id", how="left")
        .with_columns(
            pl.min_horizontal(pl.col("moment_idx"), pl.col("_mmax") - pl.col("moment_idx")).alias("_centrality")
        )
    )
    return (
        cand.sort(["_centrality", "event_id"], descending=[True, False])
        .group_by("quarter", "wall_clock_ms")
        .first()
        .select("quarter", "wall_clock_ms", "event_id")
    )


def label_moments(
    moments: pl.DataFrame, events_with_pid: pl.DataFrame, possessions: pl.DataFrame
) -> pl.DataFrame:
    """Dedup to one event-copy per physical instant and attach its possession + attributes."""
    if possessions.height == 0:
        nulls = [pl.lit(None).alias(c) for c in ("possession_id", *_POSS_LABEL_COLS)]
        return moments.with_columns(nulls)
    chosen = central_event_per_frame(moments)
    keyed = moments.join(chosen, on=["quarter", "wall_clock_ms", "event_id"], how="inner")
    return (
        keyed.join(events_with_pid.select("event_id", "possession_id"), on="event_id", how="left")
        .join(possessions.select("possession_id", *_POSS_LABEL_COLS), on="possession_id", how="left")
    )


def dedup_downsample(labeled: pl.DataFrame, *, bucket_ms: int = 100, counts_only: bool = True) -> pl.DataFrame:
    """Keep one physical instant per (possession, 100 ms bucket): the earliest ``wall_clock_ms``.

    ``labeled`` is already one event-copy per instant, so this is purely the 25->10 fps downsample.
    All ~11 entity rows of each kept instant are retained.
    """
    df = labeled.filter(pl.col("possession_id").is_not_null())
    if counts_only:
        df = df.filter(pl.col("counts_as_possession"))
    if df.height == 0:
        return df
    df = df.with_columns((pl.col("wall_clock_ms") // bucket_ms).alias("_bucket"))
    rep = (
        df.select("possession_id", "_bucket", "wall_clock_ms")
        .unique()
        .sort("possession_id", "_bucket", "wall_clock_ms")
        .group_by("possession_id", "_bucket")
        .first()
    )
    return df.join(rep, on=["possession_id", "_bucket", "wall_clock_ms"], how="inner").drop("_bucket")


def backcourt_fraction(features: pl.DataFrame) -> float:
    """Fraction of feature rows whose canonical ball position is in the backcourt (x < 47).

    A QC signal: ~0.15-0.30 on well-labeled games, ~0.50 when the bleed flips frames the wrong
    way. The dataset builder quarantines games above a threshold rather than train on them.
    """
    if features.height == 0 or "ball_x_canon" not in features.columns:
        return 0.0
    return float((features["ball_x_canon"] < 47.0).mean())


def clean_stats(labeled: pl.DataFrame, deduped: pl.DataFrame) -> dict:
    """Per-game cleanup counters for the build report."""
    ball_raw = labeled.filter(pl.col("entity") == "ball").height
    ball_kept = deduped.filter(pl.col("entity") == "ball").height
    return {
        "central_ball_frames": ball_raw,
        "kept_frames": ball_kept,
        "downsample_factor": round(ball_raw / ball_kept, 2) if ball_kept else None,
        "labeled_null_possession": labeled.filter(pl.col("possession_id").is_null()).height,
        "n_possessions_kept": deduped["possession_id"].n_unique(),
    }
