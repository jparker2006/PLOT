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
from plot.models.eval_bar.seq_model import epv_frame_table, predict_seq, train_seq


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


def run_logo_epv_traces(
    corpus: dict[str, list[PossessionSeq]],
    game_ids: list[str] | None = None,
    *,
    cfg: dict | None = None,
    device: str | None = None,
    seed: int = 1729,
    verbose: bool = False,
) -> pl.DataFrame:
    """Leakage-free per-frame OOF EPV trace, keyed game_id/possession_id/wall_clock_ms/frame_idx/epv.

    Same LOGO protocol as ``run_logo_seq`` but each held-out game is scored with the UNCAPPED
    frame-keyed ``epv_frame_table`` (every frame gets an EPV), by a model that never saw it. This is
    the leakage-free EPV trace Stage 4 (post-pass calibration + regret) consumes — distinct from the
    capped, weighted G1 OOF.
    """
    games = game_ids or sorted(corpus.keys())
    batch = int((cfg or {}).get("batch_possessions", 64))
    parts = []
    for held in games:
        train_ids = [g for g in games if g != held]
        val_id = inner_val_game(train_ids, held) if len(train_ids) > 1 else None
        fit_seqs = [s for g in train_ids if g != val_id for s in corpus[g]]
        val_seqs = corpus[val_id] if val_id else None
        if verbose:
            print(f"  trace fold held={held} val={val_id} fit_poss={len(fit_seqs)}")
        model, _ = train_seq(fit_seqs, val_seqs, cfg=cfg, device=device, seed=seed, verbose=verbose)
        parts.append(epv_frame_table(model, corpus[held], device=device, batch_possessions=batch))
    return pl.concat(parts) if parts else pl.DataFrame()


def kfold_groups(games: list[str], n_folds: int) -> list[list[str]]:
    """Deterministic round-robin partition of sorted games into ``n_folds`` balanced groups.

    Round-robin (game i -> fold i % n_folds over sorted games) keeps fold sizes within one of each
    other and avoids any temporal block bias from contiguous-date chunks.
    """
    ordered = sorted(games)
    folds: list[list[str]] = [[] for _ in range(max(1, n_folds))]
    for i, g in enumerate(ordered):
        folds[i % len(folds)].append(g)
    return [f for f in folds if f]


def run_kfold_epv_traces(
    corpus: dict[str, list[PossessionSeq]],
    game_ids: list[str] | None = None,
    *,
    n_folds: int = 7,
    cfg: dict | None = None,
    device: str | None = None,
    seed: int = 1729,
    verbose: bool = False,
) -> pl.DataFrame:
    """Leakage-free per-frame OOF EPV trace via GROUP k-fold — the scalable LOGO substitute.

    Identical contract to :func:`run_logo_epv_traces` (every frame of every game gets an EPV from a
    model that never trained on that game), but holds out a *group* of games per fold so the corpus
    is covered in ``n_folds`` trainings rather than one-per-game. At 42 games LOGO is 42 trainings;
    7 folds of ~6 games is 7. Still strictly leakage-free: a held game is never in its scorer's
    training set. One further game (outside the held group) is carved as the early-stop watch set.
    """
    games = game_ids or sorted(corpus.keys())
    batch = int((cfg or {}).get("batch_possessions", 64))
    groups = kfold_groups(games, n_folds)
    parts = []
    for fi, held_group in enumerate(groups):
        held_set = set(held_group)
        train_ids = [g for g in games if g not in held_set]
        val_id = inner_val_game(train_ids, held_group[0]) if len(train_ids) > 1 else None
        fit_seqs = [s for g in train_ids if g != val_id for s in corpus[g]]
        val_seqs = corpus[val_id] if val_id else None
        if verbose:
            print(f"  kfold {fi + 1}/{len(groups)} held={held_group} val={val_id} fit_poss={len(fit_seqs)}")
        model, _ = train_seq(fit_seqs, val_seqs, cfg=cfg, device=device, seed=seed, verbose=verbose)
        for g in held_group:
            parts.append(epv_frame_table(model, corpus[g], device=device, batch_possessions=batch))
    return pl.concat(parts) if parts else pl.DataFrame()
