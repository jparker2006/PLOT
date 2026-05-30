"""Per-possession, per-player, per-frame sequence tensors for the Stage-2 sequence EPV upgrade.

The baseline eval bar aggregated the ten players into COUNTS (n_offense_in_paint, ...); the
sequence model instead consumes the raw configuration: at every ~10 fps frame, up to ten player
tokens (canonical x/y + offense/defense/ball-handler flags + distance to the ball) plus a ball
token and a small context vector. This module produces that representation as pure numpy arrays
(no torch — fully unit-testable), one ragged sequence per possession.

It deliberately reuses the baseline's per-game pipeline end to end — ``segment_possessions`` ->
``label_moments`` (central-event de-bleed) -> ``infer_orientation`` (quarantine) ->
``dedup_downsample`` -> ``canonicalize`` -> the same ``backcourt_fraction`` QC and the same
``BACKCOURT_QUARANTINE`` threshold. Because every step is the identical deterministic function on
the identical input, the clean-game / quarantine SET is guaranteed to match the baseline corpus,
which is what makes the sequence model's Gate G1 a fair head-to-head against it.

Label = the points the possession ultimately yields, clamped to {0,1,2,3} (the shared target).
Per-frame weight = 1 / (frames in the possession), so each possession contributes equal total
weight, exactly as the baseline does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from plot.features.clean import backcourt_fraction, dedup_downsample, label_moments
from plot.features.court import RIM_X, RIM_Y
from plot.features.eval_bar import (
    HANDLER_BALL_MAX_FT,
    PERIOD_SECONDS,
    _score_diff_by_possession,
    canonicalize,
)
from plot.features.orientation import infer_orientation
from plot.io.loaders import events_table, load_game_json, moments_dataframe, team_info
from plot.models.eval_bar.dataset import BACKCOURT_QUARANTINE
from plot.possessions.segment import segment_possessions

# ----- frozen feature contract (single source of truth; mirrored by tests) -----------------
PLAYER_FEATURES = ["px", "py", "is_off", "is_def", "is_handler", "dist_to_ball"]
BALL_FEATURES = ["bx", "by", "bz", "dist_to_rim", "angle_sin", "angle_cos", "oob"]
CONTEXT_FEATURES = [
    "game_clock", "shot_clock", "shot_clock_missing", "sec_remaining_half",
    "period", "is_overtime", "score_diff", "is_clutch",
]
N_PLAYER_SLOTS = 10
N_CLASSES = 4

# Bound each possession's STORED length when assembling the corpus. Boundary-merge segmentation
# artifacts can run to thousands of near-identical frames; they compress ~40x on disk (tiny npz) but
# explode in RAM when decompressed, so an uncapped corpus OOMs at scale (208 games -> >125GB). Capping
# to the last CORPUS_MAX_FRAMES is lossless for every consumer: training uses the last 320, the OOF
# trace the last 640, and real possessions (<= ~40s) are far shorter. We COPY the slice so the giant
# base array is released (a numpy view would keep it alive).
CORPUS_MAX_FRAMES = 640

# fixed physical normalization scales (leakage-free, reproducible — never fit to data)
_X_C, _X_S = 47.0, 47.0          # center at half court; +/-1 spans the length
_Y_C, _Y_S = 25.0, 25.0
_Z_S = 12.0
_DIST_BALL_S = 25.0
_DIST_RIM_S = 30.0
_GAME_CLOCK_S = 720.0
_SHOT_CLOCK_S = 24.0
_HALF_S = 1440.0
_SCORE_S = 10.0


@dataclass
class PossessionSeq:
    """One possession's ragged sequence (numpy; the torch Dataset tensorizes + pads it)."""

    game_id: str
    possession_id: int
    players: np.ndarray   # [T, 10, len(PLAYER_FEATURES)]  float32
    pmask: np.ndarray     # [T, 10]                          bool (real player vs pad)
    ball: np.ndarray      # [T, len(BALL_FEATURES)]          float32
    ctx: np.ndarray       # [T, len(CONTEXT_FEATURES)]       float32
    label: int            # possession points in {0..3}
    wall_ms: np.ndarray   # [T]                              int64 (debug / demo ordering)

    @property
    def length(self) -> int:
        return int(self.players.shape[0])

    @property
    def weight_per_frame(self) -> float:
        return 1.0 / self.length


def cap_seq(s: PossessionSeq, max_len: int = CORPUS_MAX_FRAMES) -> PossessionSeq:
    """Truncate a possession to its most-recent ``max_len`` frames, COPYING the slices so the
    original (possibly giant, artifact) base arrays can be freed. Shorter possessions pass through."""
    if s.length <= max_len:
        return s
    sl = slice(s.length - max_len, s.length)
    return PossessionSeq(
        game_id=s.game_id, possession_id=s.possession_id,
        players=s.players[sl].copy(), pmask=s.pmask[sl].copy(),
        ball=s.ball[sl].copy(), ctx=s.ctx[sl].copy(),
        label=s.label, wall_ms=s.wall_ms[sl].copy(),
    )


def _frames_to_arrays(cdf: pl.DataFrame, possessions: pl.DataFrame) -> tuple[np.ndarray, ...]:
    """Vectorized tensorization of one game's canonicalized long moments.

    Returns dense per-frame arrays (players, pmask, ball, ctx, wall_ms) plus the per-possession
    index (ids, contiguous [start, end) slices, labels). Each possession occupies a contiguous
    block of frame rows, time-sorted.
    """
    ball = cdf.filter(pl.col("entity") == "ball").select(
        "possession_id", "wall_clock_ms", "quarter", "game_clock", "shot_clock",
        "offense_team_id", "defense_team_id", "points",
        pl.col("x_canon").alias("ball_x_canon"),
        pl.col("y_canon").alias("ball_y_canon"),
        pl.col("z").alias("ball_z"),
        pl.col("oob").cast(pl.Float32).alias("ball_oob"),
    )
    # one ball row per frame, ordered (possession, time) -> a contiguous global frame index
    ordered = ball.sort("possession_id", "wall_clock_ms").with_row_index("frame_idx")

    # ---- players: flags, distance to ball, ball-handler, deterministic slot 0..9 ----
    players = (
        cdf.filter(pl.col("entity") == "player")
        .join(
            ordered.select("possession_id", "wall_clock_ms", "frame_idx",
                           "ball_x_canon", "ball_y_canon", "offense_team_id", "defense_team_id"),
            on=["possession_id", "wall_clock_ms"], how="inner",
        )
        .with_columns(
            (pl.col("team_id") == pl.col("offense_team_id")).cast(pl.Float32).alias("is_off"),
            (pl.col("team_id") == pl.col("defense_team_id")).cast(pl.Float32).alias("is_def"),
            (((pl.col("x_canon") - pl.col("ball_x_canon")) ** 2
              + (pl.col("y_canon") - pl.col("ball_y_canon")) ** 2).sqrt()).alias("dist_to_ball_ft"),
        )
    )
    min_off = (
        players.filter(pl.col("is_off") == 1.0)
        .group_by("frame_idx").agg(pl.col("dist_to_ball_ft").min().alias("_min_off"))
    )
    players = players.join(min_off, on="frame_idx", how="left").with_columns(
        ((pl.col("is_off") == 1.0) & (pl.col("dist_to_ball_ft") <= pl.col("_min_off"))
         & (pl.col("dist_to_ball_ft") <= HANDLER_BALL_MAX_FT)).cast(pl.Float32).alias("is_handler"),
        ((pl.col("x_canon") - _X_C) / _X_S).alias("px"),
        ((pl.col("y_canon") - _Y_C) / _Y_S).alias("py"),
        (pl.col("dist_to_ball_ft") / _DIST_BALL_S).alias("dist_to_ball"),
    )
    players = (
        players.sort("frame_idx", "player_id")
        .with_columns(pl.int_range(pl.len()).over("frame_idx").alias("slot"))
        .filter(pl.col("slot") < N_PLAYER_SLOTS)
    )

    n = ordered.height
    players_arr = np.zeros((n, N_PLAYER_SLOTS, len(PLAYER_FEATURES)), dtype=np.float32)
    pmask = np.zeros((n, N_PLAYER_SLOTS), dtype=bool)
    if players.height:
        fidx = players["frame_idx"].to_numpy()
        slot = players["slot"].to_numpy()
        feat = np.column_stack([players[c].to_numpy() for c in PLAYER_FEATURES]).astype(np.float32)
        players_arr[fidx, slot] = feat
        pmask[fidx, slot] = True

    # ---- ball + context features on the ordered frame table ----
    score_diff = _score_diff_by_possession(possessions)
    angle = pl.arctan2(pl.col("ball_y_canon") - RIM_Y, RIM_X - pl.col("ball_x_canon"))
    half_idx = pl.when(pl.col("quarter") <= 2).then(2 - pl.col("quarter")).otherwise(
        pl.when(pl.col("quarter") <= 4).then(4 - pl.col("quarter")).otherwise(0)
    )
    dist_rim = ((pl.col("ball_x_canon") - RIM_X) ** 2 + (pl.col("ball_y_canon") - RIM_Y) ** 2).sqrt()
    ordered = ordered.join(score_diff, on="possession_id", how="left").with_columns(
        ((pl.col("ball_x_canon") - _X_C) / _X_S).alias("bx"),
        ((pl.col("ball_y_canon") - _Y_C) / _Y_S).alias("by"),
        (pl.col("ball_z") / _Z_S).alias("bz"),
        (dist_rim / _DIST_RIM_S).alias("dist_to_rim"),
        angle.sin().alias("angle_sin"),
        angle.cos().alias("angle_cos"),
        pl.col("ball_oob").alias("oob"),
        (pl.col("game_clock") / _GAME_CLOCK_S).alias("game_clock"),
        (pl.col("shot_clock").fill_null(0.0) / _SHOT_CLOCK_S).alias("shot_clock"),
        pl.col("shot_clock").is_null().cast(pl.Float32).alias("shot_clock_missing"),
        ((half_idx * PERIOD_SECONDS + pl.col("game_clock")) / _HALF_S).alias("sec_remaining_half"),
        ((pl.min_horizontal(pl.col("quarter"), 5) - 1) / 4.0).alias("period"),
        (pl.col("quarter") > 4).cast(pl.Float32).alias("is_overtime"),
        (pl.col("score_diff").fill_null(0).clip(-30, 30) / _SCORE_S).alias("score_diff"),
        (
            (pl.col("quarter") >= 4) & (pl.col("game_clock") <= 300)
            & (pl.col("score_diff").fill_null(0).abs() <= 5)
        ).cast(pl.Float32).alias("is_clutch"),
    )
    ball_arr = np.column_stack([ordered[c].to_numpy() for c in BALL_FEATURES]).astype(np.float32)
    ctx_arr = np.column_stack([ordered[c].to_numpy() for c in CONTEXT_FEATURES]).astype(np.float32)
    wall_ms = ordered["wall_clock_ms"].to_numpy().astype(np.int64)

    poss = (
        ordered.group_by("possession_id", maintain_order=True)
        .agg(pl.col("frame_idx").min().alias("start"), pl.col("frame_idx").max().alias("end"),
             pl.first("points").alias("points"))
        .sort("start")
    )
    poss_id = poss["possession_id"].to_numpy().astype(np.int64)
    starts = poss["start"].to_numpy().astype(np.int64)
    ends = (poss["end"].to_numpy() + 1).astype(np.int64)  # exclusive
    labels = np.clip(np.nan_to_num(poss["points"].to_numpy(), nan=0.0), 0, N_CLASSES - 1).astype(np.int64)
    return players_arr, pmask, ball_arr, ctx_arr, wall_ms, poss_id, starts, ends, labels


def _slice_to_seqs(game_id: str, arrays: tuple[np.ndarray, ...]) -> list[PossessionSeq]:
    players_arr, pmask, ball_arr, ctx_arr, wall_ms, poss_id, starts, ends, labels = arrays
    seqs = []
    for pid, s, e, lab in zip(poss_id, starts, ends, labels, strict=True):
        seqs.append(PossessionSeq(
            game_id=game_id, possession_id=int(pid),
            players=players_arr[s:e], pmask=pmask[s:e], ball=ball_arr[s:e], ctx=ctx_arr[s:e],
            label=int(lab), wall_ms=wall_ms[s:e],
        ))
    return seqs


def build_game_sequences(
    game_id: str, raw_dir: str | Path = "data/raw", pbp_path: str | Path | None = None
) -> tuple[list[PossessionSeq] | None, dict]:
    """Build per-possession sequences for one game, or (None, report) if quarantined.

    Mirrors :func:`plot.models.eval_bar.dataset.build_game_features` exactly through orientation
    + the backcourt-fraction QC, so a game is clean here iff it is clean in the baseline corpus.
    """
    cdf, poss, report = build_game_canonical(game_id, raw_dir, pbp_path)
    if cdf is None:
        return None, report
    seqs = sequences_from_canonical(game_id, cdf, poss)
    report["n_possessions"] = len(seqs)
    report["n_frames"] = int(sum(s.length for s in seqs))
    return seqs, report


def build_game_canonical(
    game_id: str, raw_dir: str | Path = "data/raw", pbp_path: str | Path | None = None
) -> tuple[pl.DataFrame | None, pl.DataFrame, dict]:
    """Shared per-game intermediate: the canonicalized, de-bleeded ~10 fps long moments + possessions.

    Returns ``(cdf | None, possessions, report)`` — ``cdf`` is None (with status=quarantined in the
    report) when orientation or the backcourt QC rejects the game. One game-load feeds BOTH the
    sequence tensors (:func:`sequences_from_canonical`) and the action layer (``plot.possessions.actions``).
    """
    raw = Path(raw_dir)
    pbp = pbp_path or (raw / "2015-16_pbp.csv")
    game = load_game_json(raw / "json" / f"{game_id}.json")
    ti = team_info(game)
    events = events_table(pbp, game_id)
    epid, poss = segment_possessions(events, ti)
    labeled = label_moments(moments_dataframe(game), epid, poss)

    orient = infer_orientation(labeled, ti)
    report = {"game_id": game_id, "matchup": f"{ti['visitor_abbr']}@{ti['home_abbr']}"}
    if orient.quarantined:
        report["status"] = "quarantined"
        report["reason"] = f"orientation:{orient.report.get('reason')}"
        return None, poss, report

    deduped = dedup_downsample(labeled)
    cdf = canonicalize(deduped, orient.rim_map)
    bc = backcourt_fraction(
        cdf.filter(pl.col("entity") == "ball").select(pl.col("x_canon").alias("ball_x_canon"))
    )
    report["backcourt_fraction"] = round(bc, 4)
    if bc > BACKCOURT_QUARANTINE:
        report["status"] = "quarantined"
        report["reason"] = f"backcourt_fraction={bc:.3f}>{BACKCOURT_QUARANTINE}"
        return None, poss, report
    report["status"] = "ok"
    return cdf, poss, report


def sequences_from_canonical(game_id: str, cdf: pl.DataFrame, poss: pl.DataFrame) -> list[PossessionSeq]:
    """Tensorize already-canonicalized long moments into per-possession sequences."""
    return _slice_to_seqs(game_id, _frames_to_arrays(cdf, poss))


# ----- caching (npz under data/processed; gitignored) ---------------------------------------
def _cache_path(cache_dir: Path, game_id: str) -> Path:
    return cache_dir / f"{game_id}_seq.npz"


def _save_cache(path: Path, seqs: list[PossessionSeq], bc: float) -> None:
    players = np.concatenate([s.players for s in seqs], axis=0)
    pmask = np.concatenate([s.pmask for s in seqs], axis=0)
    ball = np.concatenate([s.ball for s in seqs], axis=0)
    ctx = np.concatenate([s.ctx for s in seqs], axis=0)
    wall = np.concatenate([s.wall_ms for s in seqs], axis=0)
    lengths = np.array([s.length for s in seqs], dtype=np.int64)
    ends = np.cumsum(lengths)
    starts = ends - lengths
    np.savez_compressed(
        path, players=players, pmask=pmask, ball=ball, ctx=ctx, wall_ms=wall,
        poss_id=np.array([s.possession_id for s in seqs], dtype=np.int64),
        labels=np.array([s.label for s in seqs], dtype=np.int64),
        starts=starts, ends=ends,
        game_id=np.array(seqs[0].game_id), backcourt_fraction=np.array(bc, dtype=np.float64),
    )


def _load_cache(path: Path) -> tuple[list[PossessionSeq], float]:
    z = np.load(path, allow_pickle=False)
    gid = str(z["game_id"])
    # Materialize each flat array ONCE, then store COPIES of capped per-possession slices. Slicing
    # without copying (the old code) left every PossessionSeq as a numpy VIEW into the full per-game
    # array, so a single short possession's view pinned the whole array (monster boundary-merge
    # possessions included) in RAM — the corpus then retained every game's full uncapped array and
    # OOM'd at scale regardless of cap_seq. Copying the capped slice frees the full arrays on return.
    players, pmask, ball = z["players"], z["pmask"], z["ball"]
    ctx, wall = z["ctx"], z["wall_ms"]
    seqs = []
    for pid, s, e, lab in zip(z["poss_id"], z["starts"], z["ends"], z["labels"], strict=True):
        lo = int(e) - min(int(e) - int(s), CORPUS_MAX_FRAMES)   # keep only the last CORPUS_MAX_FRAMES
        e = int(e)
        seqs.append(PossessionSeq(
            game_id=gid, possession_id=int(pid),
            players=players[lo:e].copy(), pmask=pmask[lo:e].copy(), ball=ball[lo:e].copy(),
            ctx=ctx[lo:e].copy(), label=int(lab), wall_ms=wall[lo:e].copy(),
        ))
    bc = float(z["backcourt_fraction"])
    z.close()
    return seqs, bc


def build_sequence_corpus(
    game_ids: list[str],
    raw_dir: str | Path = "data/raw",
    pbp_path: str | Path | None = None,
    cache_dir: str | Path | None = "data/processed",
    use_cache: bool = True,
) -> tuple[dict[str, list[PossessionSeq]], list[dict]]:
    """Build (or load cached) per-game sequences; return ({game_id: [PossessionSeq]}, reports).

    On a cache hit the backcourt QC is re-validated (never trust a stale npz to be clean), exactly
    as the baseline corpus builder does for its parquet cache.
    """
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    corpus: dict[str, list[PossessionSeq]] = {}
    reports: list[dict] = []
    for gid in game_ids:
        cpath = _cache_path(cache, gid) if cache else None
        if use_cache and cpath and cpath.exists():
            seqs, bc = _load_cache(cpath)
            if bc > BACKCOURT_QUARANTINE:
                reports.append({"game_id": gid, "status": "quarantined", "cached": True,
                                "reason": f"backcourt_fraction={bc:.3f}>{BACKCOURT_QUARANTINE}"})
                continue
            seqs = [cap_seq(s) for s in seqs]  # bound RAM (artifact possessions); frees giant bases
            corpus[gid] = seqs
            reports.append({"game_id": gid, "status": "ok", "cached": True, "n_possessions": len(seqs),
                            "n_frames": int(sum(s.length for s in seqs)), "backcourt_fraction": round(bc, 4)})
            continue
        seqs, rep = build_game_sequences(gid, raw_dir, pbp_path)
        reports.append(rep)
        if seqs is not None:
            seqs = [cap_seq(s) for s in seqs]
            corpus[gid] = seqs
            if cpath:
                _save_cache(cpath, seqs, rep["backcourt_fraction"])
    return corpus, reports


def corpus_summary(reports: list[dict]) -> str:
    clean = [r["game_id"] for r in reports if r["status"] == "ok"]
    quar = [(r["game_id"], r.get("reason")) for r in reports if r["status"] != "ok"]
    return json.dumps({"clean": clean, "quarantined": quar}, indent=2)
