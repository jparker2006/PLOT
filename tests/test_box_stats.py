"""Gate G4 unit tests — box-stat computation from PBP + the incremental-info / reliability gate.

Synthetic PBP with hand-countable events checks the SCORE-column-independent box math; small
constructed frames check the correlation / variance-explained statistics and the gate's logic,
including the falsy-0.0 trap (a zero correlation must count as *below* the threshold, not be
silently swapped for the missing-value default).
"""

from __future__ import annotations

import polars as pl

from plot.eval.box_stats import (
    correlate_plot_with_box,
    g4_gate,
    normalize_pbp,
    per_player_box_stats,
    variance_explained,
)


def _pbp_row(game_id, msg, pid, name, desc):
    return {
        "GAME_ID": game_id, "EVENTMSGTYPE": msg, "PLAYER1_ID": pid, "PLAYER1_NAME": name,
        "HOMEDESCRIPTION": desc, "VISITORDESCRIPTION": None, "NEUTRALDESCRIPTION": None,
    }


def _synthetic_pbp() -> pl.DataFrame:
    rows = [
        # player 100 across two games: 3 made FG (1 is a 3PT), 2 missed FG, 3 made + 1 missed FT, 1 TOV
        _pbp_row(1, 1, 100, "Test Guy", "Test Guy 18' Jump Shot"),
        _pbp_row(1, 1, 100, "Test Guy", "Test Guy 26' 3PT Jump Shot"),
        _pbp_row(2, 1, 100, "Test Guy", "Test Guy Layup"),
        _pbp_row(1, 2, 100, "Test Guy", "MISS Test Guy 20' Jump Shot"),
        _pbp_row(2, 2, 100, "Test Guy", "MISS Test Guy 27' 3PT Jump Shot"),
        _pbp_row(1, 3, 100, "Test Guy", "Test Guy Free Throw 1 of 2"),
        _pbp_row(1, 3, 100, "Test Guy", "Test Guy Free Throw 2 of 2"),
        _pbp_row(2, 3, 100, "Test Guy", "Test Guy Free Throw 1 of 1"),
        _pbp_row(2, 3, 100, "Test Guy", "MISS Test Guy Free Throw 1 of 1"),
        _pbp_row(2, 5, 100, "Test Guy", "Test Guy Bad Pass Turnover"),
        # a team turnover — PLAYER1_ID carries the ~1.61e9 team id, must be dropped
        _pbp_row(1, 5, 1610612737, None, "Team Turnover"),
        # a low-volume player to exercise the min-attempts filter
        _pbp_row(1, 1, 200, "Bench Guy", "Bench Guy Dunk"),
        _pbp_row(1, 4, 100, "Test Guy", "Test Guy REBOUND (Off:1 Def:2)"),  # non-shooting, ignored
    ]
    return pl.DataFrame(rows)


def test_per_player_box_stats_counts_and_ts():
    box = per_player_box_stats(normalize_pbp(_synthetic_pbp()), min_fga_fta=0)
    p = box.filter(pl.col("player_id") == 100).to_dicts()[0]
    assert p["fgm"] == 3 and p["fga"] == 5 and p["fg3m"] == 1
    assert p["fta"] == 4 and p["ftm"] == 3 and p["tov"] == 1
    assert p["gp"] == 2
    # PTS = 2*FGM + 3PM + FTM = 6 + 1 + 3 = 10
    assert p["pts"] == 10
    # TS% = PTS / (2*(FGA + 0.44*FTA)) = 10 / (2 * (5 + 1.76)) = 10 / 13.52
    assert abs(p["ts_pct"] - 10 / 13.52) < 1e-9
    # usage proxy = (FGA + 0.44*FTA + TOV) / GP = 7.76 / 2
    assert abs(p["usg_proxy_pg"] - 7.76 / 2) < 1e-9
    assert abs(p["pts_pg"] - 5.0) < 1e-9


def test_box_stats_excludes_team_entities():
    box = per_player_box_stats(normalize_pbp(_synthetic_pbp()), min_fga_fta=0)
    assert 1610612737 not in box["player_id"].to_list()


def test_box_stats_min_attempts_filter():
    box = per_player_box_stats(normalize_pbp(_synthetic_pbp()), min_fga_fta=5)
    pids = set(box["player_id"].to_list())
    assert 100 in pids          # 5 FGA + 4 FTA = 9 ≥ 5
    assert 200 not in pids      # 1 FGA < 5


def test_correlate_and_variance_explained():
    # y = 2x + 1 exactly ⇒ Pearson 1.0 and a single-predictor OLS R² of 1.0
    x = [1.0, 2, 3, 4, 5, 6, 7, 8]
    joined = pl.DataFrame({
        "plot_per100_clipped": [2 * v + 1 for v in x],
        "ts_pct": x,
        "usg_proxy_pg": [9, 1, 8, 2, 7, 3, 6, 4],  # ~uncorrelated with y
    })
    corr = correlate_plot_with_box(joined, target="plot_per100_clipped", stat_cols=["ts_pct", "usg_proxy_pg"])
    assert abs(corr["ts_pct"]["pearson_r"] - 1.0) < 1e-6
    assert abs(corr["usg_proxy_pg"]["pearson_r"]) < 0.5
    ve = variance_explained(joined, target="plot_per100_clipped", predictors=["ts_pct"])
    assert abs(ve["r2"] - 1.0) < 1e-6


def test_g4_gate_pass():
    corrs = {"ts_pct": {"pearson_r": 0.21}, "usg_proxy_pg": {"pearson_r": -0.08}}
    g = g4_gate(corrs, reliability_sb=0.68, corr_max=0.5, reliability_min=0.6,
                gate_stats=["ts_pct", "usg_proxy_pg"])
    assert g["gate"]["incremental_info"] is True
    assert g["gate"]["reliable"] is True
    assert g["pass"] is True


def test_g4_gate_fails_on_high_correlation():
    corrs = {"ts_pct": {"pearson_r": 0.72}, "usg_proxy_pg": {"pearson_r": 0.1}}
    g = g4_gate(corrs, reliability_sb=0.68, gate_stats=["ts_pct", "usg_proxy_pg"])
    assert g["gate"]["incremental_info"] is False
    assert g["pass"] is False


def test_g4_gate_fails_on_low_reliability():
    corrs = {"ts_pct": {"pearson_r": 0.1}, "usg_proxy_pg": {"pearson_r": 0.1}}
    g = g4_gate(corrs, reliability_sb=0.40, reliability_min=0.6, gate_stats=["ts_pct", "usg_proxy_pg"])
    assert g["gate"]["reliable"] is False
    assert g["pass"] is False


def test_g4_gate_zero_correlation_counts_as_incremental():
    # a *legitimately* zero correlation is the best possible incremental-info case; the None-safe
    # read must keep it (not swap in the 1.0 missing-default, which would flip the gate to FAIL).
    corrs = {"ts_pct": {"pearson_r": 0.0}, "usg_proxy_pg": {"pearson_r": 0.0}}
    g = g4_gate(corrs, reliability_sb=0.68, corr_max=0.5, gate_stats=["ts_pct", "usg_proxy_pg"])
    assert g["gate"]["incremental_info"] is True
    assert g["pass"] is True


def test_g4_gate_missing_stat_fails_closed():
    # a stat with no correlation entry must fail closed (treated as |1.0| ≥ corr_max), not pass.
    g = g4_gate({"ts_pct": {"pearson_r": 0.1}}, reliability_sb=0.68, gate_stats=["ts_pct", "usg_proxy_pg"])
    assert g["gate"]["incremental_info"] is False
