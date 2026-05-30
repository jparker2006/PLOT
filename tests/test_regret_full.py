"""Stage 8 (v2 regret) unit tests — the full available set scored at shot decisions.

The key behaviour v1 lacked: a contested shot taken over a wide-open teammate must score positive
regret (the "should've passed to the open man" read), while taking the best available look scores
zero. Uses a transparent stub xPoints (closer + more open ⇒ higher make prob) so the arithmetic is
checkable by hand.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from plot.models.counterfactual.regret_full import (
    offensive_options,
    regret_full,
    shot_selection_regret,
)


class StubXPoints:
    """predict_proba over the xPoints design [dist_to_rim, dist_to_rim_sq, nearest_def_dist, 3pt]."""

    def predict_proba(self, X):
        dist, openness = X[:, 0], X[:, 2]
        p = np.clip(0.55 - 0.015 * dist + 0.02 * openness, 0.05, 0.95)
        return np.column_stack([1 - p, p])


def _cdf(players):
    # players: list of (entity, pid, team, x, y); offense=1, defense=2
    rows = []
    for entity, pid, team, x, y in players:
        rows.append({
            "entity": entity, "player_id": pid, "team_id": team, "x_canon": float(x), "y_canon": float(y),
            "possession_id": 0, "wall_clock_ms": 0, "offense_team_id": 1, "defense_team_id": 2,
        })
    return pl.DataFrame(rows)


def _actions(action_type):
    return pl.DataFrame({
        "possession_id": [0], "action_idx": [0], "action_type": [action_type], "is_terminal": [True],
        "actor_player_id": [10], "start_wall_ms": [0],
    })


_TRACE = pl.DataFrame({"game_id": ["g"], "possession_id": [0], "wall_clock_ms": [999], "epv": [1.0]})


def test_contested_shot_over_open_teammate_has_regret():
    # handler 10 shoots a contested mid-range; teammate 11 is wide open at the rim
    cdf = _cdf([
        ("ball", -1, 0, 70, 25),
        ("player", 10, 1, 70, 25), ("player", 11, 1, 85, 25),
        ("player", 12, 1, 60, 20), ("player", 13, 1, 60, 30), ("player", 14, 1, 50, 25),
        ("player", 20, 2, 71, 25),   # tight on the handler (1 ft)
        ("player", 21, 2, 85, 40),   # 15 ft off teammate 11 -> open
        ("player", 22, 2, 61, 20), ("player", 23, 2, 61, 30), ("player", 24, 2, 51, 25),
    ])
    opts = offensive_options(cdf, _actions("shot_make"))
    out = regret_full(opts, _actions("shot_make"), StubXPoints(), _TRACE, "g").to_dicts()[0]
    assert out["decision_kind"] == "shot"
    assert out["best_teammate_xp"] > out["own_shot_xp"]
    assert out["regret_clipped"] > 0.4


def test_best_available_shot_has_zero_regret():
    # handler 10 is the wide-open one at the rim; everyone else is guarded
    cdf = _cdf([
        ("ball", -1, 0, 86, 25),
        ("player", 10, 1, 86, 25), ("player", 11, 1, 70, 25),
        ("player", 12, 1, 60, 20), ("player", 13, 1, 60, 30), ("player", 14, 1, 50, 25),
        ("player", 20, 2, 80, 40),   # far from handler -> handler open
        ("player", 21, 2, 71, 25),   # tight on 11
        ("player", 22, 2, 61, 20), ("player", 23, 2, 61, 30), ("player", 24, 2, 51, 25),
    ])
    opts = offensive_options(cdf, _actions("shot_make"))
    out = regret_full(opts, _actions("shot_make"), StubXPoints(), _TRACE, "g").to_dicts()[0]
    assert out["regret_clipped"] == 0.0  # taking the best look leaves nothing on the table


def test_offensive_options_counts_five_offense():
    cdf = _cdf([
        ("ball", -1, 0, 70, 25),
        ("player", 10, 1, 70, 25), ("player", 11, 1, 85, 25), ("player", 12, 1, 60, 20),
        ("player", 13, 1, 60, 30), ("player", 14, 1, 50, 25),
        ("player", 20, 2, 71, 25), ("player", 21, 2, 85, 40),
    ])
    opts = offensive_options(cdf, _actions("pass"))
    assert opts.height == 5 and opts.filter(pl.col("is_handler")).height == 1


def test_shot_selection_regret_flags_contested_over_wide_open():
    cdf = _cdf([
        ("ball", -1, 0, 70, 25),
        ("player", 10, 1, 70, 25), ("player", 11, 1, 85, 25),
        ("player", 12, 1, 60, 20), ("player", 13, 1, 60, 30), ("player", 14, 1, 50, 25),
        ("player", 20, 2, 71, 25),   # tight on handler
        ("player", 21, 2, 85, 40),   # 15 ft off teammate 11 -> wide open
        ("player", 22, 2, 61, 20), ("player", 23, 2, 61, 30), ("player", 24, 2, 51, 25),
    ])
    out = shot_selection_regret(offensive_options(cdf, _actions("shot_make")), StubXPoints(), "g").to_dicts()[0]
    assert out["regret_clipped"] > 0.3


def test_shot_selection_ignores_merely_open_teammate():
    # teammate 11 is only OPEN (5 ft), not WIDE-open (>=6 ft) -> not a credible kick-out -> no regret
    cdf = _cdf([
        ("ball", -1, 0, 86, 25),
        ("player", 10, 1, 86, 25), ("player", 11, 1, 70, 25),
        ("player", 12, 1, 60, 20), ("player", 13, 1, 60, 30), ("player", 14, 1, 50, 25),
        ("player", 20, 2, 80, 40),   # handler open at rim (own shot strong)
        ("player", 21, 2, 70, 30),   # 5 ft off teammate 11 -> open but not wide-open
        ("player", 22, 2, 61, 20), ("player", 23, 2, 61, 30), ("player", 24, 2, 51, 25),
    ])
    out = shot_selection_regret(offensive_options(cdf, _actions("shot_make")), StubXPoints(), "g").to_dicts()[0]
    assert out["regret_clipped"] == 0.0
