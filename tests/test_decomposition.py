"""Gate G6 step 1 (role vs within-role) unit tests.

Two synthetic regimes pin down that the decomposition measures what it claims:

* **pure role** — regret is a clean function of touch-context (rim distance), and each player lives in
  a fixed distance band. The context model should absorb it: role_share high, and the within-role
  residual should carry NO reliable player signal.
* **within-role** — touch-context is decoupled from the player, and each player carries a fixed
  skill offset on top. The context model can't see the offset: it lands in the residual, so the
  within-role residual is reliable and within_role_share is high.

Plus a closed-form check that the variance shares obey Var(raw)=Var(role)+Var(resid)+2Cov, and that
residualize_on_context returns ``resid == value − role`` exactly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from plot.eval.decomposition import (
    g6_step1_gate,
    residualize_on_context,
    variance_decomposition,
)
from plot.models.regret.plot_metric import split_half_stability

N_GAMES, N_PLAYERS, PER_CELL = 16, 12, 6


def _frame(regret_fn, *, seed: int, dist_from_player: bool) -> pl.DataFrame:
    """Build a synthetic per-decision frame. ``dist_from_player`` ties rim distance to the player
    (a fixed band) when True, else draws it independently of the player. ``regret_fn(dist, pid)``
    returns the signed regret; clipped is its positive part."""
    rng = np.random.default_rng(seed)
    rows = []
    home = {p: 4.0 + p * 1.6 for p in range(N_PLAYERS)}  # per-player rim-distance band
    for g in range(N_GAMES):
        for p in range(N_PLAYERS):
            for k in range(PER_CELL):
                dist = (home[p] + rng.normal(0, 0.3)) if dist_from_player else rng.uniform(3.0, 25.0)
                dist = float(np.clip(dist, 1.0, 30.0))
                signed = float(regret_fn(dist, p) + rng.normal(0, 0.004))
                rows.append({
                    "game_id": f"g{g:02d}", "possession_id": g * 100 + p, "action_idx": k,
                    "player_id": p, "decision_kind": "pass_up" if k % 2 else "shot_selection",
                    "sx": 88.0 - dist, "sy": 25.0, "dist_to_rim": dist,
                    "three_pt": dist > 22.0, "nearest_def_dist": float(rng.uniform(2.0, 8.0)),
                    "regret_signed": signed, "regret_clipped": max(0.0, signed),
                })
    return pl.DataFrame(rows)


def test_pure_role_is_absorbed_and_residual_is_unreliable():
    # regret depends ONLY on rim distance, and distance is fixed per player -> all role.
    df = _frame(lambda dist, p: 0.02 * dist, seed=1, dist_from_player=True)
    res = residualize_on_context(df, value_col="regret_clipped")
    decomp = variance_decomposition(res, min_decisions=20)
    rel_raw = split_half_stability(res, value_col="regret_clipped", min_decisions_per_half=10)
    rel_resid = split_half_stability(res, value_col="regret_clipped_resid", min_decisions_per_half=10)

    assert decomp["role_share"] > 0.6
    assert decomp["role_share"] > decomp["within_role_share"]
    # raw is reliable (player bands are stable) but the residual should lose most of that reliability
    assert rel_raw["spearman_brown_full"] > 0.7
    assert rel_resid["spearman_brown_full"] < rel_raw["spearman_brown_full"] - 0.3

    gate = g6_step1_gate(decomp, rel_raw, rel_resid)
    assert gate["gate"]["role_dominated"] is True
    assert gate["gate"]["within_role_reliable"] is False


def test_within_role_offset_survives_as_reliable_residual():
    # distance is decoupled from the player; each player carries a fixed skill offset on top.
    offset = {p: (p - N_PLAYERS / 2) * 0.012 for p in range(N_PLAYERS)}
    df = _frame(lambda dist, p: 0.02 * dist + offset[p], seed=2, dist_from_player=False)
    res = residualize_on_context(df, value_col="regret_clipped")
    decomp = variance_decomposition(res, min_decisions=20)
    rel_raw = split_half_stability(res, value_col="regret_clipped", min_decisions_per_half=10)
    rel_resid = split_half_stability(res, value_col="regret_clipped_resid", min_decisions_per_half=10)

    assert decomp["within_role_share"] > 0.6
    assert decomp["within_role_share"] > decomp["role_share"]
    assert rel_resid["spearman_brown_full"] > 0.5  # the offset is a stable within-role signal

    gate = g6_step1_gate(decomp, rel_raw, rel_resid)
    assert gate["gate"]["within_role_reliable"] is True


def test_residualize_identity_and_columns():
    df = _frame(lambda dist, p: 0.02 * dist, seed=3, dist_from_player=True)
    res = residualize_on_context(df, value_col="regret_clipped")
    assert {"regret_clipped_role", "regret_clipped_resid"}.issubset(res.columns)
    recon = (res["regret_clipped_role"] + res["regret_clipped_resid"]).to_numpy()
    np.testing.assert_allclose(recon, res["regret_clipped"].to_numpy(), atol=1e-9)


def test_variance_shares_sum_to_one():
    # hand-built residualized frame (bypass the model) -> shares must obey the exact identity.
    rng = np.random.default_rng(0)
    rows = []
    for p in range(6):
        for _k in range(5):
            role = float(rng.normal(0.1 * p, 0.02))
            resid = float(rng.normal(0.05 * ((-1) ** p), 0.02))
            rows.append({"player_id": p, "regret_clipped": role + resid,
                         "regret_clipped_role": role, "regret_clipped_resid": resid})
    decomp = variance_decomposition(pl.DataFrame(rows), min_decisions=2)
    total = decomp["role_share"] + decomp["within_role_share"] + decomp["cross_share"]
    assert abs(total - 1.0) < 1e-3  # exact identity, up to the 4-dp rounding of each share
    assert 0.0 <= decomp["r2_role_reconstructs_leaderboard"] <= 1.0
