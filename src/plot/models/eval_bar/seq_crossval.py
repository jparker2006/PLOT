"""Leave-one-game-out CV for the sequence EPV model — same protocol as the LightGBM baseline.

Mirrors :func:`plot.models.eval_bar.crossval.run_logo`: hold out each whole game once, carve one
further whole game (via the baseline's deterministic ``_inner_val_game``) as a grouped watch set
for early stopping, train on the rest, and pool the held-out per-frame predictions. The pooled OOF
frame has the identical column shape the baseline produces, so ``oof_arrays`` + ``calibration``
score both models on the same footing.
"""

from __future__ import annotations

import polars as pl

from plot.models.eval_bar.folds import inner_val_game
from plot.models.eval_bar.seq_dataset import PossessionSeq
from plot.models.eval_bar.seq_model import predict_seq, train_seq


def run_logo_seq(
    corpus: dict[str, list[PossessionSeq]],
    game_ids: list[str] | None = None,
    *,
    cfg: dict | None = None,
    device: str | None = None,
    seed: int = 1729,
    verbose: bool = False,
) -> dict:
    """Run LOGO CV training one SeqEPV per fold; return {oof, folds, games}."""
    games = game_ids or sorted(corpus.keys())
    max_frames = int((cfg or {}).get("max_frames", 320))
    batch = int((cfg or {}).get("batch_possessions", 64))
    oof_parts, folds = [], []
    for held in games:
        train_ids = [g for g in games if g != held]
        val_id = inner_val_game(train_ids, held) if len(train_ids) > 1 else None
        fit_seqs = [s for g in train_ids if g != val_id for s in corpus[g]]
        val_seqs = corpus[val_id] if val_id else None
        held_seqs = corpus[held]
        if verbose:
            print(f"  fold held={held} val={val_id} fit_poss={len(fit_seqs)} test_poss={len(held_seqs)}")
        model, info = train_seq(fit_seqs, val_seqs, cfg=cfg, device=device, seed=seed, verbose=verbose)
        part = predict_seq(model, held_seqs, device=device, batch_possessions=batch, max_frames=max_frames)
        oof_parts.append(part)
        folds.append({
            "held_out": held, "inner_val": val_id,
            "n_fit_poss": len(fit_seqs), "n_test_poss": len(held_seqs),
            "n_test_frames": part.height,
            "best_epoch": info["best_epoch"], "best_val_logloss": round(info["best_val_logloss"], 5),
            "n_truncated_train": info["n_truncated_train"],
        })
    oof = pl.concat(oof_parts) if oof_parts else pl.DataFrame()
    return {"oof": oof, "folds": folds, "games": games}
