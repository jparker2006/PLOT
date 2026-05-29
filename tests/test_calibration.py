"""Closed-form and synthetic checks for the G1 calibration metrics."""

import numpy as np

from plot.eval import calibration as cal


def test_epv_from_proba_is_expected_points():
    probs = np.array([[0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(cal.epv_from_proba(probs), [1.0, 3.0])


def test_perfectly_calibrated_has_unit_slope_zero_ece():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 3, 40000)
    # E[y|pred]=pred: draw y as Poisson-ish around pred, clipped — mean tracks pred
    y = rng.normal(pred, 0.5)
    s = cal.calibration_slope(pred, y)
    assert abs(s["slope"] - 1.0) < 0.05 and abs(s["intercept"]) < 0.05
    assert cal.ece(pred, y, n_bins=15)["ece"] < 0.05


def test_known_affine_miscalibration_recovers_slope_intercept():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 3, 50000)
    y = 0.5 * pred + 0.2 + rng.normal(0, 0.3, pred.size)
    s = cal.calibration_slope(pred, y)
    assert abs(s["slope"] - 0.5) < 0.03 and abs(s["intercept"] - 0.2) < 0.03


def test_rps_respects_ordinality():
    # truth is class 0; predicting class 3 must be penalized more than predicting class 1
    y = np.array([0])
    far = np.array([[0.0, 0.0, 0.0, 1.0]])
    near = np.array([[0.0, 1.0, 0.0, 0.0]])
    assert cal.rps(far, y) > cal.rps(near, y)


def test_brier_and_logloss_closed_form():
    probs = np.array([[0.7, 0.1, 0.1, 0.1]])
    y = np.array([0])
    # brier = (0.7-1)^2 + 0.1^2*3 = 0.09 + 0.03 = 0.12
    assert abs(cal.brier(probs, y) - 0.12) < 1e-9
    assert abs(cal.multiclass_logloss(probs, y) - (-np.log(0.7))) < 1e-9


def test_constant_baseline_matches_marginals():
    y = np.array([0, 0, 2, 2, 2, 3])  # marginals 2/6,0,3/6,1/6
    base = cal.constant_baseline(y, K=4)
    # EPV of marginals = 0*2/6 + 2*3/6 + 3*1/6 = 1.5
    assert abs(base["epv"] - 1.5) < 1e-9


def test_weights_change_the_mean():
    pred = np.array([0.0, 2.0])
    y = np.array([0.0, 2.0])
    w = np.array([3.0, 1.0])
    s = cal.calibration_slope(pred, y, w)
    assert abs(s["mean_obs"] - 0.5) < 1e-9  # weighted toward the 0


def test_collapse_to_possessions_one_row_per_possession():
    pred = np.array([0.5, 1.5, 2.0, 2.0])
    y = np.array([2.0, 2.0, 0.0, 0.0])
    g = np.array(["A", "A", "A", "A"])
    p = np.array([0, 0, 1, 1])
    out = cal.collapse_to_possessions(pred, y, g, p)
    assert out["pred"].size == 2
    assert set(np.round(out["pred"], 3)) == {1.0, 2.0}  # poss0 mean 1.0, poss1 mean 2.0


def test_game_block_bootstrap_is_finite_and_seeded():
    rng = np.random.default_rng(2)
    g = np.repeat([f"g{i}" for i in range(8)], 100)
    vals = rng.uniform(0, 3, g.size)
    ci1 = cal.game_block_bootstrap(lambda idx: float(vals[idx].mean()), g, B=200, seed=7)
    ci2 = cal.game_block_bootstrap(lambda idx: float(vals[idx].mean()), g, B=200, seed=7)
    assert np.isfinite(ci1["lo"]) and np.isfinite(ci1["hi"]) and ci1["lo"] <= ci1["hi"]
    assert ci1 == ci2  # reproducible under fixed seed
