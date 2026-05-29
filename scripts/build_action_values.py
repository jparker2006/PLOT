#!/usr/bin/env python
"""Stage 3 — build the per-action value layer (ΔEPV) and run the decomposition check.

    uv run --extra seq python scripts/build_action_values.py            # all clean local games
    uv run --extra seq python scripts/build_action_values.py --games 0021500053 0021500101

Pipeline: one game-load per game yields BOTH the canonical frames (-> on-ball actions) and the
sequence tensors (-> a per-frame EPV trace from a SeqEPV trained on all clean games). Each action
is valued by EPV(end) - EPV(start), with the terminal action pinned to the realized outcome, then
the per-possession decomposition is verified to telescope to (realized - initial EPV).

Outputs reports/stage3/{action_values_summary.json, value_by_type.png}, caches the valued action
table to data/processed/action_values.parquet and the model to artifacts/seq_epv_full.pt (both
gitignored). NOTE: the EPV trace here is in-sample (one model trained on all clean games) — this is
a descriptive decomposition; Stage 4 (regret) will switch to leakage-free out-of-fold traces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from plot.config import load_config, seq_epv_config  # noqa: E402
from plot.models.action_value.value import (  # noqa: E402
    attach_epv,
    compute_values,
    decomposition_check,
    value_summary,
)
from plot.models.eval_bar.folds import inner_val_game  # noqa: E402
from plot.models.eval_bar.seq_dataset import build_game_canonical, sequences_from_canonical  # noqa: E402
from plot.models.eval_bar.seq_model import epv_frame_table, save_model, train_seq  # noqa: E402
from plot.possessions.actions import extract_actions  # noqa: E402


def _value_by_type_plot(summary: dict, path: Path) -> None:
    rows = [r for r in summary["by_action_type"] if r["n"] >= 5]
    rows.sort(key=lambda r: r["mean_value"])
    labels = [f'{r["action_type"]}\n(n={r["n"]})' for r in rows]
    means = [r["mean_value"] for r in rows]
    colors = ["#c0392b" if m < 0 else "#27ae60" for m in means]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, means, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("mean action value  (ΔEPV, points)")
    ax.set_title("Stage 3 — per-action value by type (sequence EPV)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/stage3")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seq_cfg = seq_epv_config(cfg)
    if args.max_epochs is not None:
        seq_cfg["max_epochs"] = args.max_epochs

    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))

    # Pass A: one load per game -> sequences (for the trace) + actions (model-free). Discard cdf.
    per_game: dict[str, dict] = {}
    quarantined = []
    for gid in games:
        cdf, poss, rep = build_game_canonical(gid, raw_dir=args.raw_dir)
        if cdf is None:
            quarantined.append({"game_id": gid, "reason": rep.get("reason")})
            print(f"  QUARANTINE {gid}: {rep.get('reason')}")
            continue
        per_game[gid] = {
            "seqs": sequences_from_canonical(gid, cdf, poss),
            "actions": extract_actions(cdf, poss, gid),
        }
        del cdf
    clean = sorted(per_game)
    print(f"clean games: {len(clean)} | quarantined: {len(quarantined)}")

    # train ONE SeqEPV on all clean games (carve one inner-val game for early stopping)
    val_id = inner_val_game(clean, clean[0]) if len(clean) > 1 else None
    fit_seqs = [s for g in clean if g != val_id for s in per_game[g]["seqs"]]
    val_seqs = per_game[val_id]["seqs"] if val_id else None
    print(f"training seq EPV on {len(clean)} games (inner-val={val_id}, fit_poss={len(fit_seqs)})...")
    model, info = train_seq(fit_seqs, val_seqs, cfg=seq_cfg, device=args.device, verbose=args.verbose)
    print(f"  trained: best_epoch={info['best_epoch']} val_logloss={info['best_val_logloss']:.4f} "
          f"device={info['device']}")
    Path("artifacts").mkdir(exist_ok=True)
    save_model(model, "artifacts/seq_epv_full.pt")

    # Pass B: per-game EPV trace -> value the actions
    valued_parts = []
    for gid in clean:
        trace = epv_frame_table(model, per_game[gid]["seqs"], device=args.device)
        valued_parts.append(compute_values(attach_epv(per_game[gid]["actions"], trace)))
    valued = pl.concat(valued_parts)

    chk = decomposition_check(valued)
    summary = value_summary(valued)
    summary["decomposition_check"] = chk
    summary["n_games_clean"] = len(clean)
    summary["n_games_quarantined"] = len(quarantined)
    summary["seq_epv_inner_val"] = val_id
    summary["epv_trace"] = "in-sample (one model on all clean games); Stage 4 will use OOF traces"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "action_values_summary.json").write_text(json.dumps(summary, indent=2))
    _value_by_type_plot(summary, out / "value_by_type.png")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    valued.write_parquet("data/processed/action_values.parquet")

    print("\n=== Stage 3 — per-action value decomposition ===")
    print(f"  actions: {summary['n_actions']}  | possessions: {summary['n_possessions']}  "
          f"| actions/poss: {summary['actions_per_possession']}")
    print(f"  decomposition telescopes: {chk['telescopes']}  "
          f"(max|residual| {chk['max_abs_residual']:.2e}, excluded {chk['n_possessions_excluded_trace_gap']})")
    print("  mean value by action type:")
    for r in summary["by_action_type"]:
        print(f"    {r['action_type']:<14} n={r['n']:<6} mean {r['mean_value']:+.3f}  median {r['median_value']:+.3f}")
    print(f"  wrote {out}/action_values_summary.json + value_by_type.png "
          f"| data/processed/action_values.parquet | artifacts/seq_epv_full.pt")


if __name__ == "__main__":
    main()
