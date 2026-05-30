"""Tests for Stage 4 — shot extraction, xPoints, and the narrow-counterfactual regret.

Pure polars/sklearn (no torch); the EPV trace is stubbed. Real-game checks are data-gated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from plot.models.counterfactual.regret import decision_states, regret_from_states
from plot.models.counterfactual.xpoints import (
    logo_xpoints,
    predict_xpoints,
    predict_xpoints_open,
    train_xpoints,
)
from plot.possessions.shots import extract_shots

_GAME = "0021500053"
_HAVE_DATA = Path("data/raw/json", f"{_GAME}.json").exists()
OFF, DEF = 100, 200


# --------------------------------------------------------------------------------------------
# shot extraction (synthetic)
# --------------------------------------------------------------------------------------------
def _ball(ev, poss, wall, xy):
    return {"entity": "ball", "event_id": ev, "possession_id": poss, "wall_clock_ms": wall,
            "player_id": -1, "team_id": -1, "x_canon": xy[0], "y_canon": xy[1], "z": 6.0,
            "offense_team_id": OFF, "defense_team_id": DEF}


def _player(ev, poss, wall, pid, team, xy):
    return {"entity": "player", "event_id": ev, "possession_id": poss, "wall_clock_ms": wall,
            "player_id": pid, "team_id": team, "x_canon": xy[0], "y_canon": xy[1], "z": 0.0,
            "offense_team_id": OFF, "defense_team_id": DEF}


def test_extract_shots_locates_a_three():
    rows = []
    for wall in (0, 100):  # shooter holds the ball at the arc in the frontcourt
        rows += [_ball(10, 0, wall, (65, 25)),
                 _player(10, 0, wall, 1, OFF, (65, 25)),    # shooter, on the ball
                 _player(10, 0, wall, 2, OFF, (80, 20)),
                 _player(10, 0, wall, 9, DEF, (70, 25))]    # defender 5 ft away -> open
    cdf = pl.DataFrame(rows)
    events = pl.DataFrame({"event_id": [10], "msg_type": [1], "PLAYER1_ID": [1],
                           "PLAYER1_TEAM_ID": [OFF], "description": ["Player 26' 3PT Jump Shot (3 PTS)"]})
    shots = extract_shots(cdf, events, "g")
    assert shots.height == 1
    s = shots.row(0, named=True)
    assert s["shooter_id"] == 1 and s["made"] and s["three_pt"]
    assert s["pts_if_made"] == 3 and s["fg_points"] == 3
    assert 4.5 < s["nearest_def_dist"] < 5.5 and s["is_open"]
    assert abs(s["dist_to_rim"] - 23.75) < 0.5


# --------------------------------------------------------------------------------------------
# xPoints model
# --------------------------------------------------------------------------------------------
def _synth_shots(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(0, 28, n)
    three = (dist >= 23.75).astype(int)
    nd = rng.uniform(0, 10, n)
    # true make prob: closer + more open -> higher; 3s a touch lower
    p = 1 / (1 + np.exp(-(2.2 - 0.11 * dist + 0.06 * nd - 0.2 * three)))
    made = (rng.uniform(size=n) < p).astype(int)
    return pl.DataFrame({
        "game_id": [f"g{i % 5}" for i in range(n)], "dist_to_rim": dist, "nearest_def_dist": nd,
        "three_pt": three.astype(bool), "made": made.astype(bool),
        "pts_if_made": np.where(three == 1, 3, 2), "fg_points": made * np.where(three == 1, 3, 2),
    })


def test_xpoints_mechanics_and_monotonicity():
    shots = _synth_shots()
    model = train_xpoints(shots)
    p, xp = predict_xpoints(model, shots)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.allclose(xp, p * shots["pts_if_made"].to_numpy())
    # a rim 2 should have higher make prob than a deep 3 (distance dominates)
    probe = pl.DataFrame({"dist_to_rim": [2.0, 25.0], "nearest_def_dist": [3.0, 3.0],
                          "three_pt": [False, True], "pts_if_made": [2, 3]})
    pr, _ = predict_xpoints(model, probe)
    assert pr[0] > pr[1]


def test_xpoints_open_substitution_raises_value():
    shots = _synth_shots()
    model = train_xpoints(shots)
    contested = shots.with_columns(pl.lit(0.5).alias("nearest_def_dist"))
    _, xp_real = predict_xpoints(model, contested)
    xp_open = predict_xpoints_open(model, contested, openness_ft=8.0)
    # the model learned openness helps, so the wide-open counterfactual is >= the contested value
    assert xp_open.mean() > xp_real.mean()


def test_xpoints_logo_calibrated_in_the_large():
    from plot.eval import calibration as cal
    oof = logo_xpoints(_synth_shots(n=600))
    s = cal.calibration_slope(oof["xpoints"].to_numpy(), oof["fg_points"].to_numpy().astype(float))
    assert abs(s["calibration_in_the_large"]) < 0.06


def test_epv_recalibration_fixes_over_spread():
    from plot.eval import calibration as cal
    from plot.models.counterfactual.calibrate import recalibrate_epv_oof

    rng = np.random.default_rng(0)
    rows, real = [], []
    for g in ("a", "b", "c"):
        for pid in range(200):
            epv = float(rng.uniform(0, 2.5))
            realized = 1.0 + 0.4 * (epv - 1.0) + float(rng.normal(0, 0.05))  # compressed -> EPV over-spread
            rows.append({"game_id": g, "possession_id": pid, "wall_clock_ms": 0, "epv": epv})
            real.append({"game_id": g, "possession_id": pid, "realized": realized})
    trace = pl.DataFrame(rows)
    out = recalibrate_epv_oof(trace, pl.DataFrame(real), min_fit=50)
    assert "epv_cal" in out.columns and out.height == trace.height
    joined = out.join(pl.DataFrame(real), on=["game_id", "possession_id"])
    raw = cal.calibration_slope(joined["epv"].to_numpy(), joined["realized"].to_numpy())["slope"]
    rec = cal.calibration_slope(joined["epv_cal"].to_numpy(), joined["realized"].to_numpy())["slope"]
    assert abs(rec - 1.0) < abs(raw - 1.0)  # recalibrated EPV is closer to the diagonal
    # monotone within a game: ranking preserved
    a = out.filter(pl.col("game_id") == "a").sort("epv")
    assert (a["epv_cal"].to_numpy()[1:] - a["epv_cal"].to_numpy()[:-1]).min() >= -1e-9


# --------------------------------------------------------------------------------------------
# regret (synthetic states + stub trace)
# --------------------------------------------------------------------------------------------
def test_regret_open_pass_decision():
    model = train_xpoints(_synth_shots())
    # one possession: action 0 = open pass at the arc, action 1 = terminal
    actions = pl.DataFrame({
        "game_id": ["g", "g"], "possession_id": [0, 0], "action_idx": [0, 1],
        "actor_player_id": [1, 2], "action_type": ["pass", "shot_make"],
        "is_terminal": [False, True], "start_wall_ms": [0, 500],
    })
    states = pl.DataFrame({
        "game_id": ["g"], "possession_id": [0], "action_idx": [0], "action_type": ["pass"],
        "is_terminal": [False], "actor": [1], "sx": [65.0], "sy": [25.0],
        "dist_to_rim": [23.75], "three_pt": [True], "nearest_def_dist": [8.0], "pts_if_made": [3],
    })
    # post-pass EPV at action 1's start (wall 500) = 0.7
    trace = pl.DataFrame({"game_id": ["g"], "possession_id": [0], "wall_clock_ms": [500], "epv": [0.7]})
    out = regret_from_states(states, actions, model, trace, "g")
    assert out.height == 1
    r = out.row(0, named=True)
    best = predict_xpoints_open(model, states, openness_ft=6.0)[0]
    assert r["best_available"] == pytest.approx(best, abs=1e-6)
    assert r["post_epv"] == pytest.approx(0.7)
    assert r["regret_signed"] == pytest.approx(best - 0.7, abs=1e-6)
    assert r["regret_clipped"] == pytest.approx(max(0.0, best - 0.7), abs=1e-6)


def test_regret_skips_contested_handler():
    model = train_xpoints(_synth_shots())
    actions = pl.DataFrame({
        "game_id": ["g", "g"], "possession_id": [0, 0], "action_idx": [0, 1],
        "actor_player_id": [1, 2], "action_type": ["pass", "shot_make"],
        "is_terminal": [False, True], "start_wall_ms": [0, 500],
    })
    states = pl.DataFrame({  # handler tightly guarded (nearest_def 2 ft) -> no open shot available
        "game_id": ["g"], "possession_id": [0], "action_idx": [0], "action_type": ["pass"],
        "is_terminal": [False], "actor": [1], "sx": [65.0], "sy": [25.0],
        "dist_to_rim": [23.75], "three_pt": [True], "nearest_def_dist": [2.0], "pts_if_made": [3],
    })
    trace = pl.DataFrame({"game_id": ["g"], "possession_id": [0], "wall_clock_ms": [500], "epv": [0.7]})
    assert regret_from_states(states, actions, model, trace, "g").height == 0


# --------------------------------------------------------------------------------------------
# data-gated sanity on a real game
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_DATA, reason="local tracking JSON not present")
def test_real_game_shots_are_realistic():
    from plot.io.loaders import events_table
    from plot.models.eval_bar.seq_dataset import build_game_canonical

    cdf, poss, rep = build_game_canonical(_GAME)
    shots = extract_shots(cdf, events_table("data/raw/2015-16_pbp.csv", _GAME), _GAME)
    assert shots.height > 50
    assert 0.35 < float(shots["made"].mean()) < 0.60          # realistic FG%
    assert 0.10 < float(shots["three_pt"].mean()) < 0.45      # realistic 3PA share
    # make rate falls with distance: rim shots beat threes
    rim = shots.filter(pl.col("dist_to_rim") < 4)
    three = shots.filter(pl.col("three_pt"))
    if rim.height and three.height:
        assert float(rim["made"].mean()) > float(three["made"].mean())


@pytest.mark.skipif(not _HAVE_DATA, reason="local tracking JSON not present")
def test_real_game_decision_states():
    from plot.models.eval_bar.seq_dataset import build_game_canonical
    from plot.possessions.actions import extract_actions

    cdf, poss, rep = build_game_canonical(_GAME)
    actions = extract_actions(cdf, poss, _GAME)
    ds = decision_states(cdf, actions)
    assert ds.height > 0
    assert {"sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist", "pts_if_made"}.issubset(ds.columns)
    assert ds["nearest_def_dist"].max() <= 15.0 + 1e-6
