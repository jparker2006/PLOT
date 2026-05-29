"""Tests for the baseline-EPV feature pipeline: court geometry, orientation, cleaning, leakage."""

from pathlib import Path

import polars as pl
import pytest

from plot.features import court
from plot.features.clean import backcourt_fraction, central_event_per_frame, dedup_downsample
from plot.features.eval_bar import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, _score_diff_by_possession
from plot.features.orientation import canonicalize_xy, period_half, rim_map_from_means

# ------------------------------- court geometry -------------------------------

def test_court_regions():
    assert court.court_region(88.5, 25) == "restricted_area"
    assert court.court_region(80, 25) == "paint"
    assert court.court_region(60, 25) == "above_break_3"   # >23.75 ft, not corner
    assert court.court_region(70, 25) == "mid_range"
    assert court.court_region(30, 25) == "backcourt"
    assert court.court_region(90, 1) == "corner_3"


def test_dist_and_canonical_rotation_is_180_not_mirror():
    assert abs(float(court.dist_to_rim(88.75, 25))) < 1e-9
    # left rim maps to right rim
    x, y = court.rotate_to_canonical(5.25, 25.0, court.ATTACK_LEFT)
    assert abs(x - 88.75) < 1e-9 and abs(y - 25.0) < 1e-9
    # a LEFT-corner (y small) rotates to the RIGHT corner with y large (rotation, not x-mirror)
    x2, y2 = court.rotate_to_canonical(5.0, 2.0, court.ATTACK_LEFT)
    assert abs(x2 - 89.0) < 1e-9 and abs(y2 - 48.0) < 1e-9
    # right-attacking is identity
    assert court.rotate_to_canonical(70.0, 10.0, court.ATTACK_RIGHT) == (70.0, 10.0)


# ------------------------------- orientation -------------------------------

def test_period_half():
    assert period_half(1) == 1 and period_half(2) == 1
    assert period_half(3) == 2 and period_half(4) == 2 and period_half(5) == 2


def _cells(rows):
    return pl.DataFrame(rows, schema={"team_id": pl.Int64, "half": pl.Int64, "mean_x": pl.Float64, "n": pl.Int64})


def test_orientation_decisive_game_propagates_full_map():
    cells = _cells([
        {"team_id": 1, "half": 1, "mean_x": 70.0, "n": 5000},
        {"team_id": 1, "half": 2, "mean_x": 24.0, "n": 5000},
        {"team_id": 2, "half": 1, "mean_x": 26.0, "n": 5000},
        {"team_id": 2, "half": 2, "mean_x": 68.0, "n": 5000},
    ])
    rim_map, report = rim_map_from_means(cells, (1, 2))
    assert not report["quarantined"]
    assert rim_map == {(1, 1): "R", (1, 2): "L", (2, 1): "L", (2, 2): "R"}


def test_orientation_quarantines_impossible_same_side():
    # both teams "attack" the right rim in half 1 (physically impossible) -> conflict -> quarantine
    cells = _cells([
        {"team_id": 1, "half": 1, "mean_x": 70.0, "n": 5000},
        {"team_id": 2, "half": 1, "mean_x": 69.0, "n": 5000},
    ])
    rim_map, report = rim_map_from_means(cells, (1, 2))
    assert report["quarantined"] and report["reason"] == "confident_cell_conflict"
    assert rim_map == {}


def test_orientation_quarantines_when_no_confident_cell():
    cells = _cells([{"team_id": 1, "half": 1, "mean_x": 48.0, "n": 5000}])  # 1 ft off midcourt
    rim_map, report = rim_map_from_means(cells, (1, 2))
    assert report["quarantined"] and report["reason"] == "no_confident_orientation_cell"


def test_orientation_quarantines_single_confident_cell_no_corroboration():
    # one strong cell (could be on the wrong side of midcourt); the others near midcourt -> no
    # corroboration, so a lone cell must NOT silently determine the whole map.
    cells = _cells([
        {"team_id": 1, "half": 1, "mean_x": 60.0, "n": 8000},   # confident
        {"team_id": 1, "half": 2, "mean_x": 48.0, "n": 8000},   # not confident
        {"team_id": 2, "half": 1, "mean_x": 47.5, "n": 8000},   # not confident
        {"team_id": 2, "half": 2, "mean_x": 46.0, "n": 8000},   # not confident
    ])
    rim_map, report = rim_map_from_means(cells, (1, 2))
    assert report["quarantined"] and report["reason"] == "single_confident_cell_no_corroboration"
    assert rim_map == {}


def test_canonicalize_xy_expr():
    df = pl.DataFrame({"x": [88.75, 5.25], "y": [25.0, 25.0], "ar": ["R", "L"]})
    xc, yc = canonicalize_xy(pl.col("x"), pl.col("y"), pl.col("ar"))
    out = df.with_columns(xc.alias("xc"), yc.alias("yc"))
    assert out["xc"].to_list() == [88.75, 88.75]  # both end up at the right rim
    assert out["yc"].to_list() == [25.0, 25.0]


# ------------------------------- cleaning / dedup -------------------------------

def _ball(event_id, moment_idx, wall, gc, q=1):
    return {"entity": "ball", "event_id": event_id, "moment_idx": moment_idx,
            "wall_clock_ms": wall, "game_clock": gc, "quarter": q}


def test_central_event_prefers_central_frame_over_bled_edge():
    # instant (wall=1000) appears in event 1 at the END (edge) and event 2 in the MIDDLE (central)
    rows = [
        _ball(1, 0, 100, 700), _ball(1, 1, 1000, 695),                       # ev1: wall=1000 at the END
        _ball(2, 0, 980, 696), _ball(2, 1, 1000, 695), _ball(2, 2, 1020, 694),  # ev2: wall=1000 in the MIDDLE
    ]
    chosen = central_event_per_frame(pl.DataFrame(rows))
    pick = chosen.filter(pl.col("wall_clock_ms") == 1000)["event_id"].to_list()
    assert pick == [2]


def test_dedup_downsample_one_frame_per_bucket():
    rows = []
    for w in (1000, 1040, 1080, 1120):  # 4 instants, two per 100ms bucket
        rows.append({"entity": "ball", "event_id": 1, "wall_clock_ms": w, "possession_id": 0,
                     "counts_as_possession": True})
    df = pl.DataFrame(rows)
    kept = dedup_downsample(df, bucket_ms=100)
    assert kept.height == 2  # one per 100ms bucket (1000-1099, 1100-1199)


def test_backcourt_fraction():
    df = pl.DataFrame({"ball_x_canon": [10.0, 20.0, 60.0, 80.0]})
    assert abs(backcourt_fraction(df) - 0.5) < 1e-9


# ------------------------------- features: leakage + contract -------------------------------

def test_score_diff_is_strictly_pre_possession():
    # offense team 1 scores 2 then defends; score_diff for each possession excludes its own points
    poss = pl.DataFrame({
        "possession_id": [0, 1, 2], "period": [1, 1, 1], "start_event_id": [1, 5, 9],
        "offense_team_id": [1, 2, 1], "defense_team_id": [2, 1, 2], "points": [2, 3, 0],
    })
    sd = _score_diff_by_possession(poss).sort("possession_id")
    # poss0: nobody scored yet -> 0 ; poss1 (off=2): team2 0 - team1 2 = -2 ; poss2 (off=1): team1 2 - team2 3 = -1
    assert sd["score_diff"].to_list() == [0, -2, -1]


def test_score_diff_invariant_to_current_possession_points():
    base = pl.DataFrame({
        "possession_id": [0, 1], "period": [1, 1], "start_event_id": [1, 5],
        "offense_team_id": [1, 2], "defense_team_id": [2, 1], "points": [2, 3],
    })
    zeroed = base.with_columns(pl.lit(0).alias("points"))
    # the FIRST possession's score_diff (0) must not depend on its own points
    assert _score_diff_by_possession(base).sort("possession_id")["score_diff"][0] == \
           _score_diff_by_possession(zeroed).sort("possession_id")["score_diff"][0]


def test_feature_contract_has_no_identity_or_outcome_columns():
    # no player/team identity, and no CURRENT-possession outcome columns.
    # prev_possession_end_reason is allowed: it describes the PREVIOUS, fully-resolved possession.
    identity = ("player_id", "team_id", "_name", "shot_type")
    current_outcome = ("shot_made", "action_type", "outcome")
    for col in FEATURE_COLUMNS:
        low = col.lower()
        assert not any(b in low for b in identity), f"identity leak: {col}"
        assert not any(b in low for b in current_outcome), f"outcome leak: {col}"
        if "end_reason" in low:
            assert low.startswith("prev_"), f"current-possession outcome leak: {col}"
    assert set(CATEGORICAL_COLUMNS) <= set(FEATURE_COLUMNS)
    assert "points" not in FEATURE_COLUMNS  # the label must not be a feature


# ------------------------------- end-to-end smoke (data-gated) -------------------------------

_LOCAL = sorted(Path("data/raw/json").glob("*.json")) if Path("data/raw/json").exists() else []
_PBP = Path("data/raw/2015-16_pbp.csv")


@pytest.mark.skipif(not _LOCAL or not _PBP.exists(), reason="local tracking data not present")
def test_epv_smoke_end_to_end():
    import numpy as np

    from plot.models.eval_bar.dataset import build_game_features
    from plot.models.eval_bar.model import N_CLASSES, predict_epv, train_booster

    # find a clean (non-quarantined) game
    X = None
    for p in _LOCAL:
        cand, rep = build_game_features(p.stem)
        if cand is not None:
            X = cand
            break
    assert X is not None, "no clean local game for smoke test"
    assert X.height > 1000 and set(FEATURE_COLUMNS) <= set(X.columns)

    booster = train_booster(X, num_boost_round=40)
    epv, probs = predict_epv(booster, X)
    assert np.all((epv >= 0) & (epv <= N_CLASSES - 1))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert 0.7 < float(np.average(epv, weights=X["weight"].to_numpy())) < 1.4  # ~league PPP band

    # location-prior sanity: restricted-area EPV > backcourt EPV
    Xe = X.with_columns(pl.Series("epv", epv))
    ra = Xe.filter(pl.col("court_region") == "restricted_area")["epv"].mean()
    bc = Xe.filter(pl.col("court_region") == "backcourt")["epv"].mean()
    assert ra > bc
