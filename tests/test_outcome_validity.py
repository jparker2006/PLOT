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
