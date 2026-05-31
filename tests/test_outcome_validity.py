"""Gate G6 step 2 (outcome validity) unit tests.

Two synthetic regimes over the open-look decision frame:

* **real cost** — taking a look of value S realizes ≈ S; declining it loses ``cost_slope·S`` (so the
  cost grows with the look's value). The adjusted estimate should find a positive, CI-excludes-zero
  points-lost at high value, a negative declined×S interaction, and the gate should PASS.
* **null** — realized points don't depend on the decision (R ≈ S either way). The cost is ~0, its CI
  straddles zero, and the gate must NOT pass.

Also checks taker calibration recovers the S→realized diagonal and that the raw cost rises with S.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from plot.eval.outcome_validity import (
    adjusted_decline_cost,
    cost_of_declining_by_value,
    outcome_validity_gate,
    player_cross_fit,
    taker_calibration,
)

CONTROLS = ["epv_at_decision", "dist_to_rim", "three_pt", "nearest_def_dist"]


def _looks(seed: int, *, cost_slope: float, n: int = 1600) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    S = rng.uniform(0.5, 1.6, n)
    declined = rng.integers(0, 2, n).astype(float)
    # takers realize ≈ S (calibrated); decliners lose cost_slope·S (cost grows with look value)
    R = S - declined * cost_slope * S + rng.normal(0, 0.3, n)
    return pl.DataFrame({
        "game_id": [f"g{i % 12:02d}" for i in range(n)],
        "offense_team_id": [1610612700 + (i % 5) for i in range(n)],
        "declined": declined.astype(np.int64), "S": S, "R": R,
        "epv_at_decision": rng.normal(1.0, 0.2, n),
        "dist_to_rim": rng.uniform(3, 25, n),
        "three_pt": rng.integers(0, 2, n),
        "nearest_def_dist": rng.uniform(4, 10, n),
    })


def test_real_cost_is_detected_and_gate_passes():
    df = _looks(0, cost_slope=0.6)
    calib = taker_calibration(df, n_bins=6)
    cbv = cost_of_declining_by_value(df, n_bins=5)
    adj = adjusted_decline_cost(df, controls=CONTROLS, n_boot=200, seed=0)
    gate = outcome_validity_gate(adj, cbv)

    # takers calibrate to the diagonal
    mv = np.array([c["mean_value"] for c in calib])
    mr = np.array([c["mean_realized"] for c in calib])
    assert np.corrcoef(mv, mr)[0, 1] > 0.9

    # the model: cost at high value positive with CI excluding 0, interaction negative
    assert adj["points_lost_by_declining_at_highS"] > 0
    assert adj["ci_highS"][0] > 0
    assert adj["interaction_declinedxS"] < 0
    # raw cost rises with look value
    assert cbv[-1]["raw_cost_of_declining"] > cbv[0]["raw_cost_of_declining"]
    assert gate["gate"]["high_value_decline_costs_points"] is True
    assert gate["pass"] is True


def test_null_regime_does_not_pass():
    df = _looks(1, cost_slope=0.0)
    cbv = cost_of_declining_by_value(df, n_bins=5)
    adj = adjusted_decline_cost(df, controls=CONTROLS, n_boot=200, seed=1)
    gate = outcome_validity_gate(adj, cbv)

    # cost at mean S is ~0 — its bootstrap CI straddles zero
    assert adj["ci_meanS"][0] <= 0 <= adj["ci_meanS"][1]
    assert gate["gate"]["high_value_decline_costs_points"] is False
    assert gate["pass"] is False


def test_calibration_and_cost_shapes():
    df = _looks(2, cost_slope=0.4)
    assert len(taker_calibration(df, n_bins=6)) >= 4
    cbv = cost_of_declining_by_value(df, n_bins=5)
    assert all({"n_took", "n_declined", "raw_cost_of_declining"}.issubset(r) for r in cbv)


# ----------------------------- step 2c: player-level cross-fit -----------------------------
# Two synthetic worlds over (model regret frame, open-look frame), keyed by game + player:
#  * SIGNAL — each player has a latent "points left" skill θ. Their model regret tracks θ; on declined
#    looks they realize θ fewer points than matched takers (takers realize ≈ S). The cross-fit (model
#    on one half, realized cost on the other, benchmarked on real takers) must recover a positive,
#    significant correlation and validate.
#  * NULL — model regret still tracks θ, but declined looks realize ≈ takers (no points actually left).
#    The realized-cost axis is noise, so the cross-fit correlation is small and must NOT validate.

_XF_KW = dict(n_bins=4, min_takers_per_bin=10, min_declined_half=10, min_decisions_half=10, min_players=12)


def _crossfit_world(seed: int, *, real_cost: bool, n_players: int = 48, n_games: int = 24):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 0.5, n_players)  # per-player points-left skill
    games = [f"g{gi:02d}" for gi in range(n_games)]

    reg = {"game_id": [], "player_id": [], "regret_clipped": [], "regret_clipped_resid": []}
    looks = {"game_id": [], "player_id": [], "declined": [], "S": [], "R": []}

    def add_look(g, p, declined, s, r):
        looks["game_id"].append(g)
        looks["player_id"].append(p)
        looks["declined"].append(declined)
        looks["S"].append(s)
        looks["R"].append(r)

    for g in games:
        for _ in range(110):  # global taker pool, calibrated R ≈ S
            s = rng.uniform(0.4, 1.6)
            add_look(g, -1, 0, s, s + rng.normal(0, 0.25))
        for p in range(n_players):
            for _ in range(5):  # model regret tracks θ (both raw + residual)
                val = theta[p] + rng.normal(0, 0.15)
                reg["game_id"].append(g)
                reg["player_id"].append(p)
                reg["regret_clipped"].append(max(0.0, val))
                reg["regret_clipped_resid"].append(val)
            for _ in range(4):  # this player's declined looks
                s = rng.uniform(0.4, 1.6)
                drop = theta[p] if real_cost else 0.0
                add_look(g, p, 1, s, s - drop + rng.normal(0, 0.25))
    return pl.DataFrame(looks), pl.DataFrame(reg)


def test_cross_fit_recovers_player_signal():
    looks, reg = _crossfit_world(0, real_cost=True)
    res = player_cross_fit(looks, reg, **_XF_KW)
    pooled = res["pooled_crossfit"]
    assert pooled["within_role_per100"]["n_players"] >= 12
    # both axes track θ → strong positive cross-fit correlation that validates
    assert pooled["within_role_per100"]["pearson_r"] > 0.3
    assert pooled["within_role_per100"]["pearson_p"] < 0.05
    assert res["direction_modelA_costB"]["within_role_per100"]["pearson_r"] > 0
    assert res["direction_modelB_costA"]["within_role_per100"]["pearson_r"] > 0
    assert res["gate"]["within_role_validated"] is True


def test_cross_fit_null_does_not_validate():
    looks, reg = _crossfit_world(1, real_cost=False)
    res = player_cross_fit(looks, reg, **_XF_KW)
    pooled = res["pooled_crossfit"]
    # realized points-left is noise around 0 → weak correlation, gate must not validate
    assert abs(pooled["within_role_per100"]["pearson_r"]) < 0.3
    assert res["gate"]["within_role_validated"] is False
