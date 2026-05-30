#!/usr/bin/env python
"""Build and cache the leakage-free out-of-fold per-frame EPV trace (for Stage 4+).

    uv run --extra seq python scripts/build_oof_epv.py

Runs the sequence-EPV LOGO protocol but emits, for every frame of every clean game, the EPV from a
model that never trained on that game — keyed game_id/possession_id/wall_clock_ms/frame_idx/epv.
Cached to data/processed/oof_epv_trace.parquet (gitignored). Stage 4 (post-pass EPV calibration and
regret) and Stage 5 consume this instead of the in-sample Stage-3 trace, so no possession is ever
valued by a model that saw it.
"""

from __future__ import annotations

import argparse
import resource
from pathlib import Path

import polars as pl

from plot.config import load_config, seq_epv_config
from plot.models.counterfactual.calibrate import recalibrate_epv_oof
from plot.models.eval_bar.seq_crossval import run_kfold_epv_traces, run_logo_epv_traces
from plot.models.eval_bar.seq_dataset import build_sequence_corpus


def _peak_rss_gb() -> float:
    """Peak resident memory this process has used, in GB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_048_576


def _realized_from_corpus(corpus: dict) -> pl.DataFrame:
    rows = [{"game_id": g, "possession_id": s.possession_id, "realized": float(s.label)}
            for g, seqs in corpus.items() for s in seqs]
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/oof_epv_trace.parquet")
    ap.add_argument("--device", default=None)
    ap.add_argument("--kfold", type=int, default=None,
                    help="use group k-fold with this many folds (leakage-free, scalable) instead of LOGO; "
                         "recommended once the corpus exceeds ~15 games (LOGO becomes one training per game)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seq_cfg = seq_epv_config(load_config())
    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    corpus, reports = build_sequence_corpus(games, raw_dir=args.raw_dir)
    clean = sorted(corpus)
    n_frames = sum(s.length for seqs in corpus.values() for s in seqs)
    print(f"clean games: {len(clean)} | corpus frames: {n_frames} | peak RSS {_peak_rss_gb():.1f}GB")
    if args.kfold:
        print(f"building OOF EPV traces (group {args.kfold}-fold)...")
        trace = run_kfold_epv_traces(corpus, clean, n_folds=args.kfold, cfg=seq_cfg,
                                     device=args.device, verbose=args.verbose)
    else:
        print("building OOF EPV traces (LOGO)...")
        trace = run_logo_epv_traces(corpus, clean, cfg=seq_cfg, device=args.device, verbose=args.verbose)
    print(f"trace built: {trace.height} frames | peak RSS {_peak_rss_gb():.1f}GB")
    # leakage-free isotonic recalibration -> epv_cal (the calibrated EPV the counterfactual consumes)
    trace = recalibrate_epv_oof(trace, _realized_from_corpus(corpus))
    print(f"recalibrated | peak RSS {_peak_rss_gb():.1f}GB")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trace.write_parquet(args.out)
    print(f"wrote {args.out}: {trace.height} frames over {trace['possession_id'].n_unique()} "
          f"possessions x {trace['game_id'].n_unique()} games (epv + epv_cal)")


if __name__ == "__main__":
    main()
