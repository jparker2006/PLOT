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
from pathlib import Path

import polars as pl

from plot.config import load_config, seq_epv_config
from plot.models.counterfactual.calibrate import recalibrate_epv_oof
from plot.models.eval_bar.seq_crossval import run_logo_epv_traces
from plot.models.eval_bar.seq_dataset import build_sequence_corpus


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
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seq_cfg = seq_epv_config(load_config())
    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    corpus, reports = build_sequence_corpus(games, raw_dir=args.raw_dir)
    clean = sorted(corpus)
    print(f"clean games: {len(clean)} | building OOF EPV traces (LOGO)...")
    trace = run_logo_epv_traces(corpus, clean, cfg=seq_cfg, device=args.device, verbose=args.verbose)
    # fold-safe isotonic recalibration -> epv_cal (the calibrated EPV the counterfactual consumes)
    trace = recalibrate_epv_oof(trace, _realized_from_corpus(corpus))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    trace.write_parquet(args.out)
    print(f"wrote {args.out}: {trace.height} frames over {trace['possession_id'].n_unique()} "
          f"possessions x {trace['game_id'].n_unique()} games (epv + epv_cal)")


if __name__ == "__main__":
    main()
