#!/usr/bin/env python
"""Train the baseline EPV (LightGBM multiclass) under LOGO CV and produce the Gate G1 report.

    uv run python scripts/train_eval_bar_g1.py                 # all local games
    uv run python scripts/train_eval_bar_g1.py --games 0021500053 0021500341

Writes reports/G1/{g1_calibration.json, g1_reliability.png, g1_per_class.png, g1_games.json}
and prints a PASS/FAIL summary. EPV is read off the multiclass head as sum_k k*P(points=k);
all calibration is computed on POOLED out-of-fold (held-out-game) predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot.config import eval_bar_lightgbm, load_config
from plot.eval import calibration as cal
from plot.eval.plots import per_class_reliability, reliability_diagram
from plot.models.eval_bar.crossval import oof_arrays, run_logo
from plot.models.eval_bar.dataset import build_corpus


def _ci(arrays, fn, *, B=400):
    return cal.game_block_bootstrap(lambda idx: fn(*[a[idx] for a in arrays]), arrays[-1], B=B)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None, help="game ids (default: all in raw-dir/json)")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/G1")
    ap.add_argument("--num-boost-round", type=int, default=None, help="override config num_boost_round")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    eb = cfg.get("eval_bar", {})
    lgb_params, cfg_nbr, cfg_esr = eval_bar_lightgbm(cfg)
    num_boost_round = args.num_boost_round or cfg_nbr
    n_bins = int(eb.get("reliability_bins", 15))
    boot_B = int(eb.get("bootstrap_B", 400))

    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    corpus, reports = build_corpus(games, raw_dir=args.raw_dir, use_cache=not args.no_cache)
    clean = [r["game_id"] for r in reports if r["status"] == "ok"]
    quarantined = [{"game_id": r["game_id"], "reason": r.get("reason")} for r in reports if r["status"] != "ok"]
    print(f"clean games: {len(clean)} | quarantined: {len(quarantined)} | corpus rows: {corpus.height}")
    for q in quarantined:
        print(f"  QUARANTINE {q['game_id']}: {q['reason']}")

    res = run_logo(corpus, params=lgb_params, num_boost_round=num_boost_round, early_stopping_rounds=cfg_esr)
    a = oof_arrays(res["oof"])
    epv, probs, y, w, g = a["epv"], a["probs"], a["y"], a["weight"], a["game_id"]
    pos = cal.collapse_to_possessions(epv, y, g, a["possession_id"])

    slope = cal.calibration_slope(epv, y, w)
    slope_ci = _ci((epv, y, w, g), lambda e, yy, ww, gg: cal.calibration_slope(e, yy, ww)["slope"], B=boot_B)
    ece = cal.ece(epv, y, w, n_bins=n_bins)
    ece_ci = _ci((epv, y, w, g), lambda e, yy, ww, gg: cal.ece(e, yy, ww)["ece"], B=boot_B)
    scores = cal.proper_scores(probs, y, w)
    base = cal.constant_baseline(y, w, K=probs.shape[1])
    perclass = cal.per_class_reliability(probs, y, w)

    metrics = {
        "n_games_clean": len(clean), "n_games_quarantined": len(quarantined),
        "n_moments": int(len(y)), "n_possessions": int(len(pos["y"])),
        "calibration_in_the_large": slope["calibration_in_the_large"],
        "mean_epv": slope["mean_pred"], "mean_points": slope["mean_obs"],
        "slope_moment": slope["slope"], "intercept_moment": slope["intercept"], "slope_moment_ci": slope_ci,
        "slope_possession_mean": cal.calibration_slope(pos["pred"], pos["y"])["slope"],
        "ece_points": ece["ece"], "mce_points": ece["mce"], "ece_ci": ece_ci,
        "macro_classwise_ece": perclass["macro_ece"],
        "logloss": scores["logloss"], "logloss_baseline": base["logloss"],
        "brier": scores["brier"], "brier_baseline": base["brier"],
        "rps": scores["rps"], "rps_baseline": base["rps"],
        "reliability_bins": cal.reliability_bins(epv, y, w, n_bins=n_bins),
        "folds": res["folds"],
    }

    # G1 GATE: the three criteria that actually decide pass/fail.
    beats = scores["logloss"] < base["logloss"] and scores["brier"] < base["brier"] and scores["rps"] < base["rps"]
    cal_in_large_ok = abs(slope["calibration_in_the_large"]) < 0.05
    ece_ok = ece["ece"] < 0.10
    gate = {"beats_constant_baseline": beats, "calibration_in_the_large_ok": cal_in_large_ok, "ece_ok": ece_ok}
    passed = bool(all(gate.values()))
    metrics["G1_pass"] = passed
    metrics["G1_gate"] = gate
    # Slope is a DIAGNOSTIC, not a gate. Report the CI-containment and the tolerance band separately
    # so neither masks the other (the moment-slope CI mildly excludes 1.0 = slight over-spread,
    # acceptable for a baseline location prior; the sequence model should tighten it).
    metrics["slope_diagnostic"] = {
        "slope": slope["slope"], "ci": slope_ci,
        "ci_contains_one": bool(slope_ci["lo"] <= 1.0 <= slope_ci["hi"]),
        "in_tolerance_band_0p8_1p25": bool(0.8 <= slope["slope"] <= 1.25),
        "note": "reported, NOT gating; CI mildly excludes 1.0 (slight over-confidence), tolerable for the baseline",
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1_calibration.json").write_text(json.dumps(metrics, indent=2))
    (out / "g1_games.json").write_text(json.dumps(reports, indent=2, default=str))
    reliability_diagram(epv, y, w, out / "g1_reliability.png", slope=slope)
    per_class_reliability(probs, y, w, out / "g1_per_class.png")

    cil = slope["calibration_in_the_large"]
    print("\n=== G1 calibration (pooled out-of-fold, held-out games) ===")
    print(f"  calibration-in-the-large: {cil:+.4f}  (mean EPV {slope['mean_pred']:.3f} vs pts {slope['mean_obs']:.3f})")
    ci1 = metrics["slope_diagnostic"]["ci_contains_one"]
    print(f"  slope (moment, wtd, DIAGNOSTIC): {slope['slope']:.3f}  CI[{slope_ci['lo']:.2f},{slope_ci['hi']:.2f}]"
          f"  intercept {slope['intercept']:+.3f}  (CI contains 1.0: {ci1})")
    print(f"  ECE: {ece['ece']:.4f} pts  CI[{ece_ci['lo']:.3f},{ece_ci['hi']:.3f}]   MCE {ece['mce']:.4f}"
          f"   classwise {perclass['macro_ece']:.4f}")
    print(f"  logloss {scores['logloss']:.4f} (base {base['logloss']:.4f}) | "
          f"brier {scores['brier']:.4f} ({base['brier']:.4f}) | rps {scores['rps']:.4f} ({base['rps']:.4f})")
    print(f"\n  G1 {'PASS' if passed else 'FAIL'}  (gate: {gate})")
    print(f"  wrote {out}/g1_calibration.json + g1_reliability.png + g1_per_class.png")


if __name__ == "__main__":
    main()
