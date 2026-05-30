"""Gate G5 unit tests — calibration helper, selection diagnostic, shooter-aware re-valuation,
ordering robustness, and the gate (including the falsy-0.0 guard: a perfectly-calibrated leg has
cal-in-large 0.0, which must count as *passing*, not be swapped for the missing-value default)."""

from __future__ import annotations

import numpy as np
import polars as pl

from plot.eval.validity import (
    calibration,
    calibration_gate,
    g5_gate,
    ordering_robustness,
    per_player_shot_rates,
    selection_diagnostic,
    shooter_adjusted_regret,
    shooter_skill_ratios,
)


def test_calibration_perfect():
    x = np.linspace(0.0, 3.0, 200)
    c = calibration(x, x.copy())
    assert abs(c["cal_in_large"]) < 1e-9
    assert abs(c["slope"] - 1.0) < 1e-6
    assert c["ece"] < 1e-9


def test_calibration_biased():
    x = np.linspace(0.0, 1.0, 200)
    c = calibration(x + 0.2, x)  # predictions run 0.2 high
    assert abs(c["cal_in_large"] - 0.2) < 1e-6


def test_calibration_gate_pass_and_zero_is_safe():
    legs = {"a": {"cal_in_large": 0.0, "ece": 0.0}, "b": {"cal_in_large": 0.01, "ece": 0.05}}
    g = calibration_gate(legs, cal_max=0.05, ece_max=0.10)
    assert g["per_leg"]["a"] is True   # a perfect leg (0.0) must PASS, not be swapped to the default
    assert g["pass"] is True


def test_calibration_gate_fails_on_bias():
    g = calibration_gate({"a": {"cal_in_large": 0.2, "ece": 0.03}}, cal_max=0.05, ece_max=0.10)
    assert g["pass"] is False


def test_selection_diagnostic_runs():
    rng = np.random.default_rng(0)
    taken = pl.DataFrame({
        "dist_to_rim": rng.normal(8, 2, 60), "nearest_def_dist": rng.normal(7, 1, 60),
        "three_pt": rng.integers(0, 2, 60).astype(float),
        "open_xpoints": rng.uniform(0.6, 1.2, 60), "made": rng.integers(0, 2, 60),
    })
    passed = pl.DataFrame({
        "dist_to_rim": rng.normal(12, 2, 40), "nearest_def_dist": rng.normal(5, 1, 40),
        "three_pt": rng.integers(0, 2, 40).astype(float),
        "open_xpoints": rng.uniform(0.6, 1.2, 40),
    })
    out = selection_diagnostic(taken, passed, feature_cols=["dist_to_rim", "nearest_def_dist", "three_pt"])
    assert out["n_taken"] == 60 and out["n_passed"] == 40
    # passed looks are farther from rim ⇒ taken−passed SMD on dist_to_rim is clearly negative
    assert out["feature_smd_taken_minus_passed"]["dist_to_rim"] < -0.5
    assert 0.0 <= out["propensity_auc"] <= 1.0


def _shot_pbp(rows):
    return pl.DataFrame([
        {"GAME_ID": 1, "EVENTMSGTYPE": m, "PLAYER1_ID": p, "PLAYER1_NAME": n,
         "HOMEDESCRIPTION": d, "VISITORDESCRIPTION": None, "NEUTRALDESCRIPTION": None}
        for (m, p, n, d) in rows
    ])


def test_per_player_shot_rates_and_skill_ratios():
    from plot.eval.box_stats import normalize_pbp
    rows = (
        [(1, 1, "A", "A Jump Shot")] * 15 + [(2, 1, "A", "MISS A Jump Shot")] * 5   # A: 15/20 twos
        + [(1, 2, "B", "B Jump Shot")] * 5 + [(2, 2, "B", "MISS B Jump Shot")] * 15  # B: 5/20 twos
    )
    rates = per_player_shot_rates(normalize_pbp(_shot_pbp(rows)))
    a = rates.filter(pl.col("player_id") == 1).to_dicts()[0]
    assert a["fg2m"] == 15 and a["fg2a"] == 20
    ratios, lg2, lg3 = shooter_skill_ratios(rates, k2=5, k3=5)
    assert abs(lg2 - 0.5) < 1e-9
    ra = ratios.filter(pl.col("player_id") == 1)["ratio_2p"][0]
    rb = ratios.filter(pl.col("player_id") == 2)["ratio_2p"][0]
    assert ra > 1.0 > rb   # the better shooter's open shot is worth more, the worse one's less


def test_shooter_adjusted_regret():
    ratios = pl.DataFrame({"player_id": [1], "ratio_2p": [1.4], "ratio_3p": [1.0]})
    dec = pl.DataFrame({
        "player_id": [1], "three_pt": [False], "best_available": [1.0], "post_epv": [0.6],
        "regret_signed": [0.4], "regret_clipped": [0.4],
    })
    out = shooter_adjusted_regret(dec, ratios).to_dicts()[0]
    # p_make_pop = 1.0/2 = 0.5; ×1.4 = 0.7; ×2 pts = 1.4
    assert abs(out["best_available_sh"] - 1.4) < 1e-9
    assert abs(out["regret_signed_sh"] - 0.8) < 1e-9


def test_ordering_robustness_identical():
    pop = pl.DataFrame({"player_id": list(range(10)), "mean_regret_clipped": np.linspace(0, 1, 10)})
    adj = pop.clone()
    r = ordering_robustness(pop, adj, value_col="mean_regret_clipped", top_k=5)
    assert abs(r["spearman_r"] - 1.0) < 1e-9
    assert r["top_k_overlap"] == 1.0


def test_g5_gate():
    cal_gate = {"pass": True}
    g = g5_gate(cal_gate, {"spearman_r": 0.9}, spearman_min=0.8)
    assert g["pass"] is True
    g2 = g5_gate({"pass": True}, {"spearman_r": 0.5}, spearman_min=0.8)
    assert g2["gate"]["ordering_robust_to_finishing"] is False and g2["pass"] is False
    g3 = g5_gate({"pass": False}, {"spearman_r": 0.99}, spearman_min=0.8)
    assert g3["pass"] is False
