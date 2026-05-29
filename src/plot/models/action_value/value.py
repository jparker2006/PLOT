"""Value the on-ball actions by ΔEPV and verify the per-possession decomposition.

Given the action table (``plot.possessions.actions``) and a per-frame EPV trace, each action's
value is the change in expected possession value across it:

    value(action_k) = EPV(start of action_{k+1}) - EPV(start of action_k)        (non-terminal)
    value(terminal) = realized_points            - EPV(start of terminal action) (absorbing end)

Pinning the terminal action's end-state to the **realized** outcome makes the chain telescope
EXACTLY: Σ_k value_k = realized_points - EPV(first frame) = realized - initial. The decomposition
check below confirms this holds per possession (a regression guard against frame-misalignment
bugs). Realized points are clamped to the model's {0..K-1} target space so the telescoping lives in
the same units as the EPV (rare 4-point plays are clipped to 3).

Pure polars/numpy — the EPV trace is just a (game_id, possession_id, wall_clock_ms, epv) table, so
this works with the sequence model's trace OR a stub (tests use a stub: telescoping is a structural
property, independent of the EPV values).
"""

from __future__ import annotations

import polars as pl

EPV_MAX_POINTS = 3  # realized points clamped to the multiclass head's top class {0,1,2,3}
_KEYS = ["game_id", "possession_id"]


def attach_epv(actions: pl.DataFrame, epv_trace: pl.DataFrame) -> pl.DataFrame:
    """Join EPV at each action's START frame (epv_start). ``epv_trace``: game_id, possession_id, wall_clock_ms, epv."""
    trace = epv_trace.select(
        "game_id", "possession_id",
        pl.col("wall_clock_ms").alias("start_wall_ms"), pl.col("epv").alias("epv_start"),
    )
    return actions.join(trace, on=["game_id", "possession_id", "start_wall_ms"], how="left")


def compute_values(actions_with_epv: pl.DataFrame) -> pl.DataFrame:
    """Add epv_end and value. Terminal end = realized (clamped) points; else next action's epv_start."""
    df = actions_with_epv.sort(*_KEYS, "action_idx").with_columns(
        pl.col("points").clip(0, EPV_MAX_POINTS).cast(pl.Float64).alias("_realized")
    )
    next_start = pl.col("epv_start").shift(-1).over(_KEYS)
    df = df.with_columns(
        pl.when(pl.col("is_terminal")).then(pl.col("_realized")).otherwise(next_start).alias("epv_end")
    )
    return df.with_columns((pl.col("epv_end") - pl.col("epv_start")).alias("value")).drop("_realized")


def decomposition_check(valued: pl.DataFrame, *, tol: float = 1e-6) -> dict:
    """Per possession: Σ value should equal realized - initial EPV. Report the residuals.

    Possessions with any missing epv_start (a trace gap at a segment boundary) cannot telescope and
    are excluded from the residual stats; their count is reported.
    """
    bad = valued.filter(pl.col("epv_start").is_null())["possession_id"].n_unique()
    ok = valued.join(
        valued.filter(pl.col("epv_start").is_null()).select(_KEYS).unique(),
        on=_KEYS, how="anti",
    )
    per_poss = ok.group_by(_KEYS).agg(
        pl.col("value").sum().alias("sum_value"),
        pl.col("points").first().clip(0, EPV_MAX_POINTS).alias("realized"),
        pl.col("epv_start").sort_by("action_idx").first().alias("initial_epv"),
    ).with_columns(
        (pl.col("sum_value") - (pl.col("realized") - pl.col("initial_epv"))).abs().alias("residual")
    )
    res = per_poss["residual"]
    return {
        "n_possessions_checked": per_poss.height,
        "n_possessions_excluded_trace_gap": int(bad),
        "max_abs_residual": float(res.max()) if per_poss.height else 0.0,
        "mean_abs_residual": float(res.mean()) if per_poss.height else 0.0,
        "n_residual_over_tol": int((res > tol).sum()) if per_poss.height else 0,
        "telescopes": bool(per_poss.height and float(res.max()) <= tol),
    }


def value_summary(valued: pl.DataFrame) -> dict:
    """Sensibility breakdown: count + mean/median/std value by action_type (and overall)."""
    df = valued.filter(pl.col("value").is_not_null())
    by_type = (
        df.group_by("action_type").agg(
            pl.len().alias("n"),
            pl.col("value").mean().round(4).alias("mean_value"),
            pl.col("value").median().round(4).alias("median_value"),
            pl.col("value").std().round(4).alias("std_value"),
        ).sort("n", descending=True)
    )
    return {
        "n_actions": df.height,
        "n_possessions": df.select(_KEYS).unique().height,
        "actions_per_possession": round(df.height / max(df.select(_KEYS).unique().height, 1), 2),
        "by_action_type": by_type.to_dicts(),
        "overall_mean_value": round(float(df["value"].mean()), 4),
    }
