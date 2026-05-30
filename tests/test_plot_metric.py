"""Tests for Stage 5 — the PLOT metric aggregation, G3 statistics, and the group k-fold partition."""

import numpy as np
import polars as pl

from plot.models.eval_bar.seq_crossval import kfold_groups
from plot.models.regret.plot_metric import (
    aggregate_per_player,
    finishing_independence,
    finishing_residual_by_player,
    g3_gate,
    player_fixed_effect_test,
    split_half_stability,
)

# ------------------------------- group k-fold partition -------------------------------

def test_kfold_groups_balanced_and_total_cover():
    games = [f"g{i:02d}" for i in range(10)]
    folds = kfold_groups(games, 3)
    assert len(folds) == 3
    sizes = sorted(len(f) for f in folds)
    assert sizes == [3, 3, 4]                       # within one of each other
    flat = sorted(g for f in folds for g in f)
    assert flat == sorted(games)                    # every game covered exactly once (leakage-safe)


def test_kfold_groups_deterministic():
    games = ["b", "a", "c", "d", "e"]
    assert kfold_groups(games, 2) == kfold_groups(games, 2)
    # round-robin over SORTED games -> fold0 = a,c,e ; fold1 = b,d
    assert kfold_groups(games, 2) == [["a", "c", "e"], ["b", "d"]]


def test_kfold_drops_empty_folds():
    assert kfold_groups(["a", "b"], 5) == [["a"], ["b"]]   # no empty trailing folds


# ------------------------------- per-player aggregation -------------------------------

def _regret_rows(rows):
    return pl.DataFrame(rows, schema_overrides={"three_pt": pl.Boolean})


def test_aggregate_per_player_scales_and_filters():
    rows = []
    # player 1: 40 decisions, mean signed 0.1 ; player 2: only 5 decisions -> dropped
    for _ in range(40):
        rows.append({"game_id": "g0", "player_id": 1, "regret_signed": 0.1, "regret_clipped": 0.1,
                     "dist_to_rim": 20.0, "three_pt": False, "nearest_def_dist": 6.0})
    for _ in range(5):
        rows.append({"game_id": "g0", "player_id": 2, "regret_signed": 0.5, "regret_clipped": 0.5,
                     "dist_to_rim": 20.0, "three_pt": False, "nearest_def_dist": 6.0})
    out = aggregate_per_player(_regret_rows(rows), per=100, min_decisions=30)
    assert out["player_id"].to_list() == [1]                  # player 2 filtered out
    assert abs(out["plot_per100_signed"][0] - 10.0) < 1e-9    # 0.1 * 100


def test_split_half_stability_perfectly_stable():
    # each player has a CONSTANT regret across all games -> halves agree perfectly (r=1)
    rng = np.random.default_rng(0)
    levels = {p: float(rng.uniform(0, 0.3)) for p in range(8)}
    rows = []
    for gi in range(12):                                       # 12 games -> 6 per half
        for p, lvl in levels.items():
            for _ in range(4):                                 # 4 decisions/player/game -> 24/half
                rows.append({"game_id": f"g{gi:02d}", "player_id": p,
                             "regret_signed": lvl, "regret_clipped": lvl,
                             "dist_to_rim": 18.0, "three_pt": False, "nearest_def_dist": 6.0})
    stab = split_half_stability(_regret_rows(rows), value_col="regret_clipped", min_decisions_per_half=15)
    assert stab["n_players"] == 8
    assert stab["pearson_r"] > 0.99                            # constant per player -> perfectly stable


# ------------------------------- G3(b): decision vs location / finishing -------------------------------

def test_player_fixed_effect_test_detects_player_signal():
    # regret driven mostly by a per-player offset (a real decision tendency) beyond location
    rng = np.random.default_rng(1)
    offsets = {p: (p - 2) * 0.4 for p in range(5)}             # distinct player means
    rows = []
    for p, off in offsets.items():
        for _ in range(60):
            d = float(rng.uniform(3, 25))
            rows.append({"game_id": "g0", "player_id": p,
                         "regret_signed": off + 0.001 * d + float(rng.normal(0, 0.05)),
                         "regret_clipped": 0.0, "dist_to_rim": d, "three_pt": d > 22,
                         "nearest_def_dist": float(rng.uniform(4, 10))})
    fe = player_fixed_effect_test(_regret_rows(rows), value_col="regret_signed", min_decisions=30)
    assert fe["p_value"] < 0.01                                # player FE highly significant
    assert fe["incremental_r2_player"] > 0.5                   # players explain most of the variance


def test_player_fixed_effect_test_null_when_no_player_signal():
    # regret is pure location + noise, identical generating process per player -> player FE not significant
    rng = np.random.default_rng(2)
    rows = []
    for p in range(5):
        for _ in range(60):
            d = float(rng.uniform(3, 25))
            rows.append({"game_id": "g0", "player_id": p,
                         "regret_signed": 0.01 * d + float(rng.normal(0, 0.3)),
                         "regret_clipped": 0.0, "dist_to_rim": d, "three_pt": d > 22,
                         "nearest_def_dist": float(rng.uniform(4, 10))})
    fe = player_fixed_effect_test(_regret_rows(rows), value_col="regret_signed", min_decisions=30)
    assert fe["p_value"] > 0.05                                # no real between-player signal


def test_finishing_independence_orthogonal_vs_correlated():
    per_player = pl.DataFrame({"player_id": list(range(10)),
                               "mean_regret_signed": [0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15, 0.0]})
    # finishing constructed orthogonal to regret (mean 0, dot(regret, finishing) = 0) -> corr 0
    ortho = pl.DataFrame({"player_id": list(range(10)), "n_shots": [20] * 10,
                          "finishing_resid": [0.2, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, -0.2, -0.2, 0.0]})
    r_ortho = finishing_independence(per_player, ortho)["pearson_r"]
    # finishing == regret -> correlation 1
    same = per_player.rename({"mean_regret_signed": "finishing_resid"}).with_columns(pl.lit(20).alias("n_shots"))
    r_same = finishing_independence(per_player, same)["pearson_r"]
    assert abs(r_ortho) < 0.3 and r_same > 0.99


def test_g3_gate_treats_zero_pvalue_as_significant():
    # regression guard: an F-test p-value of exactly 0.0 (below machine epsilon) is FALSY — a
    # `d.get(k) or default` read would swap in 1.0 and wrongly fail the gate. It must pass here.
    stab = {"pearson_r": 0.38, "pearson_p": 2e-6}
    fe = {"p_value": 0.0}                          # decisively significant, but falsy
    indep = {"pearson_r": -0.04}
    out = g3_gate(stab, fe, indep, finishing_corr_max=0.40)
    assert out["gate"]["decision_not_location"] is True
    assert out["pass"] is True
    # and a genuinely non-significant FE must fail that leg
    assert g3_gate(stab, {"p_value": 0.2}, indep)["gate"]["decision_not_location"] is False
    # missing keys fall back to the conservative default (fail), never crash
    assert g3_gate({}, {}, {})["pass"] is False


def test_finishing_residual_by_player_filters_and_signs():
    shots = pl.DataFrame({
        "shooter_id": [1] * 12 + [2] * 5,
        "made": [1] * 12 + [0] * 5,
        "p_make": [0.5] * 12 + [0.5] * 5,
    })
    out = finishing_residual_by_player(shots, min_shots=10)
    assert out["player_id"].to_list() == [1]                   # player 2 (<10 shots) dropped
    assert abs(out["finishing_resid"][0] - 0.5) < 1e-9         # made everything vs p=0.5 -> +0.5
