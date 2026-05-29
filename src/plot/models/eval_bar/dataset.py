"""Assemble the multi-game EPV feature corpus, with caching, quarantine, and LOGO folds.

Per game: load -> segment -> central-event label -> infer orientation (skip if quarantined) ->
dedup/downsample -> build features -> backcourt-fraction QC (quarantine heavy-bleed games). Clean
games are cached to ``data/processed/{game_id}_eval_bar_features.parquet`` and concatenated. The
held-out unit for G1 is always the whole GAME (group = game_id).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from plot.features.clean import backcourt_fraction, clean_stats, dedup_downsample, label_moments
from plot.features.eval_bar import build_features
from plot.features.orientation import infer_orientation
from plot.io.loaders import events_table, load_game_json, moments_dataframe, team_info
from plot.possessions.segment import segment_possessions

BACKCOURT_QUARANTINE = 0.40


def build_game_features(
    game_id: str, raw_dir: str | Path = "data/raw", pbp_path: str | Path | None = None
) -> tuple[pl.DataFrame | None, dict]:
    """Build the feature matrix for one game, or (None, report) if it is quarantined."""
    raw = Path(raw_dir)
    pbp = pbp_path or (raw / "2015-16_pbp.csv")
    game = load_game_json(raw / "json" / f"{game_id}.json")
    ti = team_info(game)
    events = events_table(pbp, game_id)
    epid, poss = segment_possessions(events, ti)
    labeled = label_moments(moments_dataframe(game), epid, poss)

    orient = infer_orientation(labeled, ti)
    report = {"game_id": game_id, "matchup": f"{ti['visitor_abbr']}@{ti['home_abbr']}",
              "orientation": orient.report}
    if orient.quarantined:
        report["status"], report["reason"] = "quarantined", f"orientation:{orient.report.get('reason')}"
        return None, report

    deduped = dedup_downsample(labeled)
    X = build_features(deduped, poss, orient.rim_map, game_id)
    bc = backcourt_fraction(X)
    report["clean_stats"] = clean_stats(labeled, deduped)
    report["backcourt_fraction"] = round(bc, 4)
    report["n_rows"] = X.height
    if bc > BACKCOURT_QUARANTINE:
        report["status"], report["reason"] = "quarantined", f"backcourt_fraction={bc:.3f}>{BACKCOURT_QUARANTINE}"
        return None, report
    report["status"] = "ok"
    return X, report


def build_corpus(
    game_ids: list[str],
    raw_dir: str | Path = "data/raw",
    pbp_path: str | Path | None = None,
    cache_dir: str | Path | None = "data/processed",
    use_cache: bool = True,
) -> tuple[pl.DataFrame, list[dict]]:
    """Build (or load cached) features for many games; return (concatenated clean X, per-game reports)."""
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    frames, reports = [], []
    for gid in game_ids:
        cpath = cache / f"{gid}_eval_bar_features.parquet" if cache else None
        if use_cache and cpath and cpath.exists():
            X = pl.read_parquet(cpath)
            # re-validate the QC gate on the cached frame: never trust a stale/hand-placed parquet
            # to be clean (the quarantine, not the filesystem, decides what enters the corpus).
            bc = backcourt_fraction(X)
            if bc > BACKCOURT_QUARANTINE:
                reports.append({"game_id": gid, "status": "quarantined", "cached": True,
                                "reason": f"backcourt_fraction={bc:.3f}>{BACKCOURT_QUARANTINE}"})
                continue
            frames.append(X)
            reports.append({"game_id": gid, "status": "ok", "n_rows": X.height,
                            "backcourt_fraction": round(bc, 4), "cached": True})
            continue
        X, rep = build_game_features(gid, raw_dir, pbp_path)
        reports.append(rep)
        if X is not None:
            if cpath:
                X.write_parquet(cpath)
            frames.append(X)
    corpus = pl.concat(frames) if frames else pl.DataFrame()
    return corpus, reports


def logo_folds(game_ids: list[str]):
    """Yield (train_game_ids, held_out_game_id) for leave-one-game-out cross-validation."""
    for held in game_ids:
        yield [g for g in game_ids if g != held], held
