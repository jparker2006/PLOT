#!/usr/bin/env python
"""Stage 6.5 — Gate G5: is the PLOT regret metric *valid* (high regret = value genuinely left
behind), or an artifact of how its inputs are valued?

    uv run --extra seq python scripts/build_g5.py     # uses the local 42-game OOF trace

Rebuilds the per-decision regret over the local clean games and runs three validity checks:

  G5(a) inputs calibrated in-subspace — restricted to the open pass-up decisions (and the near-rim /
        high-value slice), post-pass EPV matches realized possession points and xPoints matches
        realized makes. Calibrated inputs ⇒ positive regret means points were really not realized.
  G5(b) selection-on-observables bounded — taken vs passed open looks compared on observable
        difficulty (standardized mean differences + a propensity AUC), anchored on the empirical
        make rate of taken open shots by value bin. Reported as a BOUND, not a pass.
  G5(c) ordering robust to finishing — rebuild regret with shooter-aware xPoints (each shot valued
        at the shooter's own season make rate, EB-shrunk) and check per-player PLOT barely reorders.

G5 PASS = (a) AND (c). NOTE: runs on the LOCAL 42-game corpus (the 208-game per-decision data is not
local); these are properties of the *method* and so transfer to the 208-game headline — documented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from plot.eval.box_stats import normalize_pbp  # noqa: E402
from plot.eval.validity import (  # noqa: E402
    calibration,
    calibration_gate,
    g5_gate,
    ordering_robustness,
    per_player_shot_rates,
    selection_diagnostic,
    shooter_adjusted_regret,
    shooter_skill_ratios,
)
from plot.io.loaders import events_table  # noqa: E402
from plot.models.counterfactual.regret import decision_states, regret_from_states  # noqa: E402
from plot.models.counterfactual.xpoints import logo_xpoints, predict_xpoints_open, train_xpoints  # noqa: E402
from plot.models.eval_bar.seq_dataset import build_game_canonical  # noqa: E402
from plot.possessions.actions import extract_actions  # noqa: E402
from plot.possessions.shots import OPEN_FT, WIDE_OPEN_FT, extract_shots  # noqa: E402

MIN_DECISIONS = 15        # smaller corpus than G3 (42 games) -> a lower entry bar for the ordering test
NEAR_RIM_FT = 10.0        # the "high-value slice" where the leaderboard's bigs sit
CAL_MAX, ECE_MAX = 0.05, 0.10
SPEARMAN_MIN = 0.80


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/G5")
    ap.add_argument("--oof-trace", default="data/processed/oof_epv_trace.parquet")
    ap.add_argument("--pbp", default="data/raw/2015-16_pbp.csv")
    args = ap.parse_args()

    trace = pl.read_parquet(args.oof_trace)
    if "epv_cal" in trace.columns:  # the metric consumes the recalibrated EPV
        trace = trace.with_columns(pl.col("epv_cal").alias("epv"))

    # only the games with an OOF EPV trace are usable (post-pass EPV comes from it)
    trace_games = set(trace["game_id"].unique().to_list())
    games = sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json") if p.stem in trace_games)
    print(f"games with OOF trace coverage: {len(games)}")
    shots_parts, actions_parts, states_parts, poss_parts, quarantined = [], [], [], [], []
    for g in games:
        cdf, poss, _ = build_game_canonical(g, raw_dir=args.raw_dir)
        if cdf is None:
            quarantined.append(g)
            continue
        ev = events_table(Path(args.raw_dir, "2015-16_pbp.csv"), g)
        actions = extract_actions(cdf, poss, g)
        shots_parts.append(extract_shots(cdf, ev, g))
        actions_parts.append(actions)
        states_parts.append(decision_states(cdf, actions).with_columns(pl.lit(g).alias("game_id")))
        poss_parts.append(poss.select("possession_id", "points").with_columns(pl.lit(g).alias("game_id")))
        del cdf
    clean = [g for g in games if g not in quarantined]
    shots = pl.concat(shots_parts)
    actions = pl.concat(actions_parts)
    states = pl.concat(states_parts)
    poss_all = pl.concat(poss_parts)
    print(f"clean games: {len(clean)} | shots: {shots.height}")

    # ---- per-decision regret (population xPoints + leakage-free trace), same as the metric ----
    full_model = train_xpoints(shots)
    regret = pl.concat([
        r for r in (
            regret_from_states(states.filter(pl.col("game_id") == g), actions.filter(pl.col("game_id") == g),
                               full_model, trace, g)
            for g in clean
        ) if r.height
    ])
    print(f"open pass-up decisions: {regret.height} over {regret['player_id'].n_unique()} players")

    # =================== G5(a) — inputs calibrated in the decision subspace ===================
    # post-pass EPV leg: chosen-pass value vs realized possession points (clipped to the {0..3} head)
    pp = regret.join(poss_all, on=["game_id", "possession_id"], how="inner").with_columns(
        pl.col("points").clip(0, 3).alias("realized")
    )
    cal_epv = calibration(pp["post_epv"].to_numpy(), pp["realized"].to_numpy())
    near = pp.filter(pl.col("dist_to_rim") <= NEAR_RIM_FT)
    cal_epv_nearrim = calibration(near["post_epv"].to_numpy(), near["realized"].to_numpy())

    # xPoints leg: P(make) vs realized makes on TAKEN OPEN shots (leakage-free OOF), + near-rim slice
    shots_oof = logo_xpoints(shots)
    open_taken = shots_oof.filter(pl.col("is_open"))
    cal_make = calibration(open_taken["p_make"].to_numpy(), open_taken["made"].cast(pl.Float64).to_numpy())
    open_taken_near = open_taken.filter(pl.col("dist_to_rim") <= NEAR_RIM_FT)
    cal_make_nearrim = calibration(open_taken_near["p_make"].to_numpy(),
                                   open_taken_near["made"].cast(pl.Float64).to_numpy())

    cal_legs = {"post_pass_epv_vs_realized": cal_epv, "xpoints_make_vs_made": cal_make}
    cal_gate = calibration_gate(cal_legs, cal_max=CAL_MAX, ece_max=ECE_MAX)

    # =================== G5(b) — selection-on-observables bound ===================
    # taken open shots, valued by the SAME open-shot counterfactual as the passed ones
    open_taken_v = open_taken.with_columns(
        pl.Series("open_xpoints", predict_xpoints_open(full_model, open_taken, openness_ft=WIDE_OPEN_FT)),
        pl.col("three_pt").cast(pl.Float64),
    )
    passed_v = regret.with_columns(
        pl.col("best_available").alias("open_xpoints"), pl.col("three_pt").cast(pl.Float64)
    )
    feat = ["dist_to_rim", "nearest_def_dist", "three_pt"]
    selection = selection_diagnostic(open_taken_v, passed_v, feature_cols=feat, value_col="open_xpoints")

    # =================== G5(c) — ordering robust to finishing skill ===================
    pbp_norm = normalize_pbp(pl.read_csv(
        args.pbp, infer_schema_length=20000,
        columns=["GAME_ID", "EVENTMSGTYPE", "PLAYER1_ID", "PLAYER1_NAME",
                 "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"],
    ))
    ratios, lg2, lg3 = shooter_skill_ratios(per_player_shot_rates(pbp_norm))
    regret_sh = shooter_adjusted_regret(regret, ratios)
    pop_pp = (regret_sh.group_by("player_id").agg(
        pl.len().alias("n"), pl.col("regret_clipped").mean().alias("mean_regret_clipped"))
        .filter(pl.col("n") >= MIN_DECISIONS))
    adj_pp = (regret_sh.group_by("player_id").agg(
        pl.len().alias("n"), pl.col("regret_clipped_sh").mean().alias("mean_regret_clipped"))
        .filter(pl.col("n") >= MIN_DECISIONS))
    robustness = ordering_robustness(pop_pp, adj_pp, value_col="mean_regret_clipped")

    gate = g5_gate(cal_gate, robustness, spearman_min=SPEARMAN_MIN)

    report = {
        "corpus": {"clean_games": len(clean), "decisions": regret.height,
                   "note": "local 42-game corpus; method properties transfer to the 208-game headline"},
        "thresholds": {"cal_max": CAL_MAX, "ece_max": ECE_MAX, "spearman_min": SPEARMAN_MIN,
                       "min_decisions": MIN_DECISIONS, "near_rim_ft": NEAR_RIM_FT, "open_ft": OPEN_FT},
        "G5a_calibration": {
            "post_pass_epv_vs_realized": cal_epv, "post_pass_epv_near_rim": cal_epv_nearrim,
            "xpoints_make_vs_made": cal_make, "xpoints_make_near_rim": cal_make_nearrim,
            "gate": cal_gate,
        },
        "G5b_selection_bound": {"league_2p": round(lg2, 4), "league_3p": round(lg3, 4), **selection},
        "G5c_ordering_robustness": {"shrink_k2": 200.0, "shrink_k3": 100.0, **robustness},
        "G5_gate": gate["gate"], "G5_pass": bool(gate["pass"]), "G5_gate_detail": gate,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g5.json").write_text(json.dumps(report, indent=2))
    _plots(pp, open_taken, pop_pp, adj_pp, out)

    print("\n=== Gate G5 (PLOT validity) ===")
    print(f"  G5(a) post-pass EPV vs realized: cal-in-large {cal_epv['cal_in_large']}, ECE {cal_epv['ece']}, "
          f"slope {cal_epv['slope']} (n={cal_epv['n']}); near-rim ECE {cal_epv_nearrim.get('ece')}")
    print(f"  G5(a) xPoints make vs made: cal-in-large {cal_make['cal_in_large']}, ECE {cal_make['ece']}, "
          f"slope {cal_make['slope']} (n={cal_make['n']}); near-rim ECE {cal_make_nearrim.get('ece')}")
    print(f"  G5(b) selection: propensity AUC {selection['propensity_auc']}, "
          f"SMD {selection['feature_smd_taken_minus_passed']}")
    print(f"  G5(c) ordering robustness: Spearman {robustness.get('spearman_r')}, "
          f"Pearson {robustness.get('pearson_r')}, top-{robustness.get('top_k')} overlap "
          f"{robustness.get('top_k_overlap')} (n={robustness.get('n_players')})")
    print(f"\n  G5 {'PASS' if report['G5_pass'] else 'FAIL'}  (gate: {gate['gate']})")
    print(f"  wrote {out}/g5.json + figures")


def _binned(pred: np.ndarray, realized: np.ndarray, n_bins: int = 10):
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2)
    xs, ys = [], []
    for bb in np.unique(idx):
        m = idx == bb
        xs.append(pred[m].mean())
        ys.append(realized[m].mean())
    return np.array(xs), np.array(ys)


def _plots(pp, open_taken, pop_pp, adj_pp, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a1) post-pass EPV reliability
    x, y = _binned(pp["post_epv"].to_numpy(), pp["realized"].to_numpy())
    axes[0].plot([0, 3], [0, 3], "k--", lw=0.8, alpha=0.5)
    axes[0].plot(x, y, "o-", color="#2c7fb8")
    axes[0].set(xlabel="predicted post-pass EPV", ylabel="realized possession points",
                title="G5(a) post-pass EPV (pass-up subspace)")
    # (a2) xPoints make reliability on taken open shots
    x, y = _binned(open_taken["p_make"].to_numpy(), open_taken["made"].cast(pl.Float64).to_numpy())
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[1].plot(x, y, "o-", color="#c0392b")
    axes[1].set(xlabel="predicted P(make)", ylabel="empirical make rate",
                title="G5(a) xPoints (taken open shots)")
    # (c) shooter-aware vs population per-player PLOT
    j = pop_pp.select("player_id", pl.col("mean_regret_clipped").alias("pop")).join(
        adj_pp.select("player_id", pl.col("mean_regret_clipped").alias("adj")), on="player_id", how="inner")
    axes[2].scatter(j["pop"].to_numpy(), j["adj"].to_numpy(), s=16, alpha=0.6, color="#27ae60")
    lo = float(min(j["pop"].min(), j["adj"].min()))
    hi = float(max(j["pop"].max(), j["adj"].max()))
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    axes[2].set(xlabel="population xPoints PLOT", ylabel="shooter-aware xPoints PLOT",
                title="G5(c) ordering robustness")
    fig.tight_layout()
    fig.savefig(out / "g5_validity.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
