#!/usr/bin/env python
"""Stage 4 — xPoints + the narrow counterfactual, and the Gate G2 report (the crux).

    uv run --extra seq python scripts/build_oof_epv.py        # once: cache the leakage-free EPV trace
    uv run --extra seq python scripts/train_xpoints_g2.py     # -> reports/G2/

G2 (the crux) asks whether the one-step-ahead counterfactual is CALIBRATED on held-out continuations:
  (1) when an open shot WAS taken, does xPoints match the realized field-goal points? (LOGO xPoints)
  (2) when a pass WAS made, does the post-pass EPV match the realized possession points? (OOF trace)
If both calibrate, regret = best-available-open-shot − chosen-value is built on honest inputs, not an
artifact. The per-decision regret is then computed (open-shot vs post-pass EPV) as a demonstration;
per-player aggregation + the stability gate (G3) are Stage 5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import polars as pl  # noqa: E402

from plot.eval import calibration as cal  # noqa: E402
from plot.eval.plots import reliability_diagram  # noqa: E402
from plot.io.loaders import events_table  # noqa: E402
from plot.models.counterfactual.regret import decision_states, regret_from_states, regret_summary  # noqa: E402
from plot.models.counterfactual.xpoints import logo_xpoints, train_xpoints  # noqa: E402
from plot.models.eval_bar.seq_dataset import build_game_canonical  # noqa: E402
from plot.possessions.actions import extract_actions  # noqa: E402
from plot.possessions.shots import extract_shots  # noqa: E402

EPV_MAX = 3


def _proper_binary(p, y):
    import numpy as np
    p = np.clip(p, 1e-9, 1 - 1e-9)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    br = float(((p - y) ** 2).mean())
    return ll, br


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/G2")
    ap.add_argument("--oof-trace", default="data/processed/oof_epv_trace.parquet")
    args = ap.parse_args()

    trace_path = Path(args.oof_trace)
    if not trace_path.exists():
        raise SystemExit(f"missing {trace_path} — run: uv run --extra seq python scripts/build_oof_epv.py")
    trace = pl.read_parquet(trace_path)
    # the counterfactual consumes the isotonic-recalibrated EPV (see counterfactual.calibrate)
    epv_recalibrated = "epv_cal" in trace.columns
    if epv_recalibrated:
        trace = trace.with_columns(pl.col("epv_cal").alias("epv"))

    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    shots_parts, actions_parts, states_parts, poss_parts, quarantined = [], [], [], [], []
    for g in games:
        cdf, poss, rep = build_game_canonical(g, raw_dir=args.raw_dir)
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
    shots = pl.concat(shots_parts)
    clean = [g for g in games if g not in quarantined]
    print(f"clean games: {len(clean)} | shots: {shots.height} | trace frames: {trace.height}")

    # ---- G2 part 1: xPoints calibration (LOGO, held-out games) ----
    oof = logo_xpoints(shots)
    xp = oof["xpoints"].to_numpy()
    fg = oof["fg_points"].to_numpy().astype(float)
    pm = oof["p_make"].to_numpy()
    made = oof["made"].to_numpy().astype(int)
    xp_slope = cal.calibration_slope(xp, fg)
    xp_ece = cal.ece(xp, fg, n_bins=10)
    ll, br = _proper_binary(pm, made)
    base_ll, base_br = _proper_binary(pm * 0 + made.mean(), made)
    xpoints_block = {
        "n_shots": int(len(xp)), "make_rate": round(float(made.mean()), 4),
        "calibration_in_the_large": xp_slope["calibration_in_the_large"],
        "mean_xpoints": xp_slope["mean_pred"], "mean_fg_points": xp_slope["mean_obs"],
        "slope": xp_slope["slope"], "ece_points": xp_ece["ece"], "mce_points": xp_ece["mce"],
        "p_make_logloss": round(ll, 4), "p_make_logloss_baseline": round(base_ll, 4),
        "p_make_brier": round(br, 4), "p_make_brier_baseline": round(base_br, 4),
        "reliability_bins": cal.reliability_bins(xp, fg, n_bins=8),
    }

    # ---- G2 part 2: post-pass EPV calibration (OOF trace at receiver catch frames) ----
    actions = pl.concat(actions_parts)
    poss_pts = pl.concat(poss_parts)
    post_pass = (
        actions.filter(pl.col("action_idx") >= 1)  # a catch after a pass
        .select("game_id", "possession_id", pl.col("start_wall_ms").alias("wall_clock_ms"))
        .join(trace.select("game_id", "possession_id", "wall_clock_ms", "epv"),
              on=["game_id", "possession_id", "wall_clock_ms"], how="inner")
        .join(poss_pts, on=["game_id", "possession_id"], how="left")
        .with_columns(pl.col("points").clip(0, EPV_MAX).cast(pl.Float64).alias("realized"))
    )
    pe = post_pass["epv"].to_numpy()
    pr = post_pass["realized"].to_numpy()
    pp_slope = cal.calibration_slope(pe, pr)
    pp_ece = cal.ece(pe, pr, n_bins=10)
    postpass_block = {
        "n_post_pass_states": int(len(pe)),
        "calibration_in_the_large": pp_slope["calibration_in_the_large"],
        "mean_post_pass_epv": pp_slope["mean_pred"], "mean_realized": pp_slope["mean_obs"],
        "slope": pp_slope["slope"], "ece_points": pp_ece["ece"], "mce_points": pp_ece["mce"],
        "reliability_bins": cal.reliability_bins(pe, pr, n_bins=8),
    }

    # ---- regret demo (full-data xPoints model + the OOF trace) ----
    full_model = train_xpoints(shots)
    states = pl.concat(states_parts)
    regret_parts = [
        regret_from_states(states.filter(pl.col("game_id") == g), actions.filter(pl.col("game_id") == g),
                           full_model, trace, g)
        for g in clean
    ]
    regret = pl.concat([r for r in regret_parts if r.height]) if any(r.height for r in regret_parts) else pl.DataFrame()

    # ---- gate ----
    def _calibrated(block):
        return abs(block["calibration_in_the_large"]) < 0.05 and block["ece_points"] < 0.10

    g2 = {
        "xpoints_calibrated": _calibrated(xpoints_block),
        "xpoints_beats_baseline": xpoints_block["p_make_logloss"] < xpoints_block["p_make_logloss_baseline"],
        "post_pass_epv_calibrated": _calibrated(postpass_block),
    }
    report = {
        "n_games_clean": len(clean), "n_games_quarantined": len(quarantined),
        "epv_recalibrated": epv_recalibrated,
        "xpoints": xpoints_block, "post_pass_epv": postpass_block,
        "regret_demo": regret_summary(regret),
        "G2_gate": g2, "G2_pass": bool(all(g2.values())),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g2_calibration.json").write_text(json.dumps(report, indent=2))
    reliability_diagram(xp, fg, None, out / "g2_xpoints_reliability.png", slope=xp_slope, n_bins=10,
                        title="xPoints calibration (G2, held-out games)")
    reliability_diagram(pe, pr, None, out / "g2_postpass_epv_reliability.png", slope=pp_slope, n_bins=10,
                        title="post-pass EPV calibration (G2)")

    print("\n=== Gate G2 (one-step-ahead counterfactual calibration) ===")
    print(f"  xPoints (LOGO, n={xpoints_block['n_shots']}): cal-in-large "
          f"{xpoints_block['calibration_in_the_large']:+.4f} (xP {xpoints_block['mean_xpoints']:.3f} vs fg "
          f"{xpoints_block['mean_fg_points']:.3f}) | ECE {xpoints_block['ece_points']:.4f} | slope "
          f"{xpoints_block['slope']:.3f} | logloss {xpoints_block['p_make_logloss']:.4f} "
          f"(base {xpoints_block['p_make_logloss_baseline']:.4f})")
    pb = postpass_block
    print(f"  post-pass EPV (n={pb['n_post_pass_states']}): cal-in-large "
          f"{pb['calibration_in_the_large']:+.4f} (EPV {pb['mean_post_pass_epv']:.3f} vs "
          f"{pb['mean_realized']:.3f}) | ECE {pb['ece_points']:.4f} | slope {pb['slope']:.3f}")
    print(f"  regret demo: {report['regret_demo']}")
    print(f"\n  G2 {'PASS' if report['G2_pass'] else 'FAIL'}  (gate: {g2})")
    print(f"  wrote {out}/g2_calibration.json + reliability figures")


if __name__ == "__main__":
    main()
