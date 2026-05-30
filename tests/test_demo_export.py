"""Stage 7 demo-export unit tests — badge tiers, featured-possession ranking, and the per-possession
payload assembly (frame mapping + decision badges)."""

from __future__ import annotations

import polars as pl

from plot.viz.demo_export import badge_for, build_possession, select_featured_possessions


def test_badge_for_tiers():
    assert badge_for(-0.5, 0.0) == "great"        # pass strongly beat the open shot
    assert badge_for(-0.05, 0.0) == "good"        # roughly even
    assert badge_for(0.18, 0.18) == "inaccuracy"
    assert badge_for(0.30, 0.30) == "mistake"
    assert badge_for(0.70, 0.70) == "blunder"


def test_select_featured_ranks_by_max_regret():
    dec = pl.DataFrame({
        "possession_id": [1, 1, 2, 3],
        "regret_clipped": [0.1, 0.6, 0.2, 0.4],
    })
    order = select_featured_possessions(dec, top_n=3)
    assert order == [1, 3, 2]   # poss 1 has the single biggest regret (0.6)


def _frames():
    rows = []
    for w, ex in [(0, 80.0), (100, 79.9), (200, 79.8)]:
        rows.append({"wall_clock_ms": w, "game_clock": ex, "entity": "ball",
                     "player_id": -1, "team_id": 0, "x": 80.0, "y": 25.0, "z": 5.0, "epv": 1.0 + w / 1000})
        for pid, tid, x in [(10, 100, 70.0), (20, 200, 60.0)]:
            rows.append({"wall_clock_ms": w, "game_clock": ex, "entity": "player",
                         "player_id": pid, "team_id": tid, "x": x, "y": 25.0, "z": 0.0, "epv": 1.0 + w / 1000})
    return pl.DataFrame(rows)


def test_build_possession_payload():
    decisions = pl.DataFrame({
        "wall_clock_ms": [100], "player_id": [10], "regret_signed": [0.3],
        "regret_clipped": [0.3], "best_available": [1.2], "post_epv": [0.9],
    })
    poss_row = {"period": 2, "offense_team_id": 100, "end_reason": "made_fg", "points": 2}
    payload = build_possession(7, _frames(), decisions, poss_row, stride=1)
    assert payload["possession_id"] == 7 and payload["n_frames"] == 3 and payload["points"] == 2
    f = payload["frames"][1]
    assert len(f["pl"]) == 2 and f["ball"] is not None and f["epv"] == 1.1
    assert len(payload["decisions"]) == 1
    d = payload["decisions"][0]
    assert d["frame"] == 1 and d["player"] == 10 and d["badge"] == "mistake"
