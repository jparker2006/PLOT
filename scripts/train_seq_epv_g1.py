#!/usr/bin/env python
"""Train the sequence EPV model under LOGO CV and produce its Gate G1 report (head-to-head vs baseline).

    uv run --extra seq python scripts/train_seq_epv_g1.py                 # all local games
    uv run --extra seq python scripts/train_seq_epv_g1.py --games 0021500053 0021500341 --max-epochs 8

Writes reports/G1_seq/{g1_seq_calibration.json, g1_seq_reliability.png, g1_seq_per_class.png,
g1_seq_games.json}. EPV = sum_k k*P(points=k) from the per-frame outcome head; all calibration is
computed on POOLED out-of-fold (held-out-game) predictions with the SAME calibration code, clean
games, and quarantine as the baseline — so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot.config import load_config, seq_epv_config
from plot.eval import calibration as cal
from plot.eval.plots import per_class_reliability, reliability_diagram
from plot.models.eval_bar.folds import oof_arrays
from plot.models.eval_bar.seq_crossval import run_logo_seq
from plot.models.eval_bar.seq_dataset import build_sequence_corpus


def _ci(arrays, fn, *, B=400):
    return cal.game_block_bootstrap(lambda idx: fn(*[a[idx] for a in arrays]), arrays[-1], B=B)


def _compute_metrics(a, clean, quarantined, n_bins, boot_B, folds) -> dict:
    """Identical metric set + gate as scripts/train_eval_bar_g1.py (shared calibration.py)."""
    epv, probs, y, w, g = a["epv"], a["probs"], a["y"], a["weight"], a["game_id"]
    pos = cal.collapse_to_possessions(epv, y, g, a["possession_id"])
    slope = cal.calibration_slope(epv, y, w)
    slope_ci = _ci((epv, y, w, g), lambda e, yy, ww, gg: cal.calibration_slope(e, yy, ww)["slope"], B=boot_B)
    ece = cal.ece(epv, y, w, n_bins=n_bins)
    ece_ci = _ci((epv, y, w, g), lambda e, yy, ww, gg: cal.ece(e, yy, ww)["ece"], B=boot_B)
    scores = cal.proper_scores(probs, y, w)
    base = cal.constant_baseline(y, w, K=probs.shape[1])
    perclass = cal.per_class_reliability(probs, y, w)

    beats = scores["logloss"] < base["logloss"] and scores["brier"] < base["brier"] and scores["rps"] < base["rps"]
    cal_in_large_ok = abs(slope["calibration_in_the_large"]) < 0.05
    ece_ok = ece["ece"] < 0.10
    gate = {"beats_constant_baseline": beats, "calibration_in_the_large_ok": cal_in_large_ok, "ece_ok": ece_ok}
    return {
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
        "folds": folds,
        "G1_pass": bool(all(gate.values())), "G1_gate": gate,
        "slope_diagnostic": {
            "slope": slope["slope"], "ci": slope_ci,
            "ci_contains_one": bool(slope_ci["lo"] <= 1.0 <= slope_ci["hi"]),
            "in_tolerance_band_0p8_1p25": bool(0.8 <= slope["slope"] <= 1.25),
        },
    }


def _delta(seq: dict, base_path: Path) -> str:
    if not base_path.exists():
        return "  (baseline report not found; run scripts/train_eval_bar_g1.py for a head-to-head)"
    b = json.loads(base_path.read_text())

    def row(name, s, bb, lower_better=True):
        d = s - bb
        better = (d < 0) if lower_better else (abs(s - 1) < abs(bb - 1))
        return f"    {name:<26} seq {s:.4f}  base {bb:.4f}  d {d:+.4f}  {'BETTER' if better else 'worse'}"
    lines = ["  head-to-head vs baseline (lower is better unless noted):",
             row("logloss", seq["logloss"], b["logloss"]),
             row("brier", seq["brier"], b["brier"]),
             row("rps", seq["rps"], b["rps"]),
             row("ECE (points)", seq["ece_points"], b["ece_points"]),
             row("|cal-in-large|", abs(seq["calibration_in_the_large"]), abs(b["calibration_in_the_large"])),
             row("slope dist from 1", abs(seq["slope_moment"] - 1), abs(b["slope_moment"] - 1))]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None, help="game ids (default: all in raw-dir/json)")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/G1_seq")
    ap.add_argument("--max-epochs", type=int, default=None, help="override config seq_epv.max_epochs")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    eb = cfg.get("eval_bar", {})
    seq_cfg = seq_epv_config(cfg)
    if args.max_epochs is not None:
        seq_cfg["max_epochs"] = args.max_epochs
    n_bins = int(eb.get("reliability_bins", 15))
    boot_B = int(eb.get("bootstrap_B", 400))

    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    corpus, reports = build_sequence_corpus(games, raw_dir=args.raw_dir, use_cache=not args.no_cache)
    clean = [r["game_id"] for r in reports if r["status"] == "ok"]
    quarantined = [{"game_id": r["game_id"], "reason": r.get("reason")} for r in reports if r["status"] != "ok"]
    n_frames = sum(r.get("n_frames", 0) for r in reports if r["status"] == "ok")
    print(f"clean games: {len(clean)} | quarantined: {len(quarantined)} | frames: {n_frames}")
    for q in quarantined:
        print(f"  QUARANTINE {q['game_id']}: {q['reason']}")

    res = run_logo_seq(corpus, clean, cfg=seq_cfg, device=args.device, verbose=args.verbose)
    a = oof_arrays(res["oof"])
    metrics = _compute_metrics(a, clean, quarantined, n_bins, boot_B, res["folds"])
    metrics["model"] = "seq_epv (DeepSets set encoder -> causal GRU -> multiclass head)"
    metrics["seq_epv_config"] = seq_cfg

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1_seq_calibration.json").write_text(json.dumps(metrics, indent=2))
    (out / "g1_seq_games.json").write_text(json.dumps(reports, indent=2, default=str))
    epv, probs, y, w = a["epv"], a["probs"], a["y"], a["weight"]
    reliability_diagram(epv, y, w, out / "g1_seq_reliability.png", slope=cal.calibration_slope(epv, y, w))
    per_class_reliability(probs, y, w, out / "g1_seq_per_class.png")

    s = metrics
    print("\n=== G1 (sequence EPV) calibration (pooled out-of-fold, held-out games) ===")
    print(f"  calibration-in-the-large: {s['calibration_in_the_large']:+.4f}  "
          f"(mean EPV {s['mean_epv']:.3f} vs pts {s['mean_points']:.3f})")
    sd = s["slope_diagnostic"]
    print(f"  slope (moment, DIAGNOSTIC): {s['slope_moment']:.3f}  "
          f"CI[{sd['ci']['lo']:.2f},{sd['ci']['hi']:.2f}]  (CI contains 1.0: {sd['ci_contains_one']})")
    print(f"  ECE: {s['ece_points']:.4f} pts  CI[{s['ece_ci']['lo']:.3f},{s['ece_ci']['hi']:.3f}]  "
          f"MCE {s['mce_points']:.4f}  classwise {s['macro_classwise_ece']:.4f}")
    print(f"  logloss {s['logloss']:.4f} (base {s['logloss_baseline']:.4f}) | "
          f"brier {s['brier']:.4f} ({s['brier_baseline']:.4f}) | rps {s['rps']:.4f} ({s['rps_baseline']:.4f})")
    print(f"\n  G1 {'PASS' if s['G1_pass'] else 'FAIL'}  (gate: {s['G1_gate']})")
    print(_delta(s, Path("reports/G1/g1_calibration.json")))
    print(f"  wrote {out}/g1_seq_calibration.json + figures")


if __name__ == "__main__":
    main()
