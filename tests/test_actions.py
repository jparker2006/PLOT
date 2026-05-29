"""Tests for the Stage-3 per-action value layer (action extraction + ΔEPV decomposition).

All pure-polars — the EPV trace is a stub, since telescoping is a structural property independent
of the EPV values. The real-game checks are gated on a local tracking JSON being present.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from plot.models.action_value.value import (
    attach_epv,
    compute_values,
    decomposition_check,
    value_summary,
)
from plot.possessions.actions import extract_actions

_GAME = "0021500053"
_HAVE_DATA = Path("data/raw/json", f"{_GAME}.json").exists()

OFF, DEF = 100, 200


def _frame(poss, wall, ball_xy, players):
    """Build the long rows for one frame: a ball entity + player entities (all carry offense_team_id)."""
    rows = [{"entity": "ball", "team_id": -1, "player_id": -1, "x_canon": ball_xy[0], "y_canon": ball_xy[1],
             "possession_id": poss, "wall_clock_ms": wall, "offense_team_id": OFF, "defense_team_id": DEF}]
    for pid, team, (x, y) in players:
        rows.append({"entity": "player", "team_id": team, "player_id": pid, "x_canon": x, "y_canon": y,
                     "possession_id": poss, "wall_clock_ms": wall, "offense_team_id": OFF, "defense_team_id": DEF})
    return rows


def _toy_cdf_and_poss():
    rows = []
    # possession 0: A holds (f0), passes to B (f1), B shoots & makes (f2) -> 2 actions
    rows += _frame(0, 0, (10, 25), [(1, OFF, (10, 25)), (2, OFF, (40, 25)), (9, DEF, (12, 25))])
    rows += _frame(0, 100, (40, 25), [(1, OFF, (10, 25)), (2, OFF, (40, 25)), (9, DEF, (38, 25))])
    rows += _frame(0, 200, (88, 25), [(1, OFF, (60, 25)), (2, OFF, (88, 25)), (9, DEF, (80, 25))])
    # possession 1: A holds throughout, turns it over -> 1 (terminal) action
    rows += _frame(1, 0, (30, 25), [(1, OFF, (30, 25)), (2, OFF, (50, 25)), (9, DEF, (33, 25))])
    rows += _frame(1, 100, (30, 25), [(1, OFF, (30, 25)), (2, OFF, (50, 25)), (9, DEF, (31, 25))])
    cdf = pl.DataFrame(rows)
    poss = pl.DataFrame({
        "possession_id": [0, 1],
        "offense_team_id": [OFF, OFF],
        "end_reason": ["made_fg", "turnover"],
        "points": [2, 0],
    })
    return cdf, poss


def _stub_trace(cdf: pl.DataFrame, game_id: str, epv_by_wall: dict | None = None) -> pl.DataFrame:
    ball = cdf.filter(pl.col("entity") == "ball").select("possession_id", "wall_clock_ms")
    if epv_by_wall is None:  # default: any deterministic function of position works
        return ball.join(
            cdf.filter(pl.col("entity") == "ball").select(
                "possession_id", "wall_clock_ms", (pl.col("x_canon") / 47.0).alias("epv")),
            on=["possession_id", "wall_clock_ms"]).with_columns(pl.lit(game_id).alias("game_id"))
    return ball.with_columns(
        pl.lit(game_id).alias("game_id"),
        pl.struct("possession_id", "wall_clock_ms").map_elements(
            lambda s: epv_by_wall[(s["possession_id"], s["wall_clock_ms"])], return_dtype=pl.Float64
        ).alias("epv"),
    )


def test_action_segmentation_and_types():
    cdf, poss = _toy_cdf_and_poss()
    actions = extract_actions(cdf, poss, "g")
    p0 = actions.filter(pl.col("possession_id") == 0).sort("action_idx")
    assert p0.height == 2  # A's carry, then B's terminal shot
    assert p0["actor_player_id"].to_list() == [1, 2]
    assert p0["action_type"].to_list() == ["pass", "shot_make"]
    assert p0["is_terminal"].to_list() == [False, True]
    assert p0["start_wall_ms"].to_list() == [0, 100]

    p1 = actions.filter(pl.col("possession_id") == 1)
    assert p1.height == 1 and p1["action_type"][0] == "turnover" and bool(p1["is_terminal"][0])


def test_decomposition_telescopes_exactly():
    cdf, poss = _toy_cdf_and_poss()
    epv = {(0, 0): 0.5, (0, 100): 1.0, (0, 200): 1.8, (1, 0): 0.6, (1, 100): 0.4}
    actions = extract_actions(cdf, poss, "g")
    valued = compute_values(attach_epv(actions, _stub_trace(cdf, "g", epv)))

    p0 = valued.filter(pl.col("possession_id") == 0).sort("action_idx")
    # action0 (pass): epv_start(B=1.0) - epv_start(A=0.5) = 0.5 ; action1 (terminal make): 2 - 1.0 = 1.0
    assert p0["value"].to_list() == pytest.approx([0.5, 1.0])
    p1 = valued.filter(pl.col("possession_id") == 1)
    assert p1["value"][0] == pytest.approx(0.0 - 0.6)  # terminal turnover: realized 0 - initial 0.6

    chk = decomposition_check(valued)
    assert chk["telescopes"] and chk["max_abs_residual"] < 1e-9
    assert chk["n_possessions_checked"] == 2 and chk["n_possessions_excluded_trace_gap"] == 0


def test_value_clamps_realized_points():
    # a 4-point play (and-1 three) must clamp to the head's top class so EPV units match
    cdf, poss = _toy_cdf_and_poss()
    poss = poss.with_columns(pl.when(pl.col("possession_id") == 0).then(4).otherwise(pl.col("points")).alias("points"))
    epv = {(0, 0): 0.5, (0, 100): 1.0, (0, 200): 1.8, (1, 0): 0.6, (1, 100): 0.4}
    valued = compute_values(attach_epv(extract_actions(cdf, poss, "g"), _stub_trace(cdf, "g", epv)))
    term = valued.filter((pl.col("possession_id") == 0) & pl.col("is_terminal"))
    assert term["value"][0] == pytest.approx(3.0 - 1.0)  # clamped to 3, not 4


@pytest.mark.skipif(not _HAVE_DATA, reason="local tracking JSON not present")
def test_real_game_actions_telescope():
    from plot.models.eval_bar.seq_dataset import build_game_canonical

    cdf, poss, rep = build_game_canonical(_GAME)
    assert rep["status"] == "ok"
    actions = extract_actions(cdf, poss, _GAME)
    assert actions.height > 0

    # exactly one terminal action per possession, and types are from the known vocabulary
    term = actions.filter(pl.col("is_terminal"))
    assert term.height == actions["possession_id"].n_unique()
    assert term.group_by("possession_id").len()["len"].max() == 1
    known = {"shot_make", "shot_miss", "turnover", "foul_drawn", "period_end", "jump_ball",
             "control_change", "other", "pass"}
    assert set(actions["action_type"].unique()).issubset(known)

    valued = compute_values(attach_epv(actions, _stub_trace(cdf, _GAME)))
    chk = decomposition_check(valued)
    assert chk["telescopes"], chk
    assert chk["n_possessions_excluded_trace_gap"] == 0
    summ = value_summary(valued)
    assert summ["n_actions"] == actions.height and summ["actions_per_possession"] >= 1.0
