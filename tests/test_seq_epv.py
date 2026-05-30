"""Tests for the Stage-2 sequence EPV model and its tensor dataset.

Pure-function and torch-model invariants run with no data; the dataset-build and quarantine-match
checks are gated on a local game being present (skipped in CI without the raw tracking JSON).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plot.models.eval_bar.seq_dataset import (
    BALL_FEATURES,
    CONTEXT_FEATURES,
    N_CLASSES,
    N_PLAYER_SLOTS,
    PLAYER_FEATURES,
    PossessionSeq,
)

_GAME = "0021500053"
_HAVE_DATA = Path("data/raw/json", f"{_GAME}.json").exists()


# --------------------------------------------------------------------------------------------
# feature contract
# --------------------------------------------------------------------------------------------
def test_feature_contract_dims():
    assert PLAYER_FEATURES == ["px", "py", "is_off", "is_def", "is_handler", "dist_to_ball"]
    assert len(BALL_FEATURES) == 7
    assert len(CONTEXT_FEATURES) == 8
    assert N_PLAYER_SLOTS == 10
    assert N_CLASSES == 4


def _toy_seq(t: int, label: int, *, off=5, deff=5, seed=0) -> PossessionSeq:
    rng = np.random.default_rng(seed)
    players = rng.standard_normal((t, N_PLAYER_SLOTS, len(PLAYER_FEATURES))).astype(np.float32)
    players[..., 2] = 0.0
    players[..., 3] = 0.0
    players[:, :off, 2] = 1.0
    players[:, off:off + deff, 3] = 1.0
    pmask = np.zeros((t, N_PLAYER_SLOTS), dtype=bool)
    pmask[:, : off + deff] = True
    return PossessionSeq(
        game_id="g", possession_id=1,
        players=players,
        pmask=pmask,
        ball=rng.standard_normal((t, len(BALL_FEATURES))).astype(np.float32),
        ctx=rng.standard_normal((t, len(CONTEXT_FEATURES))).astype(np.float32),
        label=label, wall_ms=np.arange(t, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------------
# torch model invariants
# --------------------------------------------------------------------------------------------
def test_model_forward_shape_and_invariants():
    torch = pytest.importorskip("torch")
    from plot.models.eval_bar.seq_model import SeqEPV

    torch.manual_seed(0)
    b, t = 2, 6
    fp, fb, fc = len(PLAYER_FEATURES), len(BALL_FEATURES), len(CONTEXT_FEATURES)
    players = torch.randn(b, t, N_PLAYER_SLOTS, fp)
    players[..., 2] = 0.0
    players[..., 3] = 0.0
    players[:, :, :5, 2] = 1.0
    players[:, :, 5:, 3] = 1.0
    pmask = torch.ones(b, t, N_PLAYER_SLOTS, dtype=torch.bool)
    ball = torch.randn(b, t, fb)
    ctx = torch.randn(b, t, fc)

    model = SeqEPV().eval()
    with torch.no_grad():
        out = model(players, pmask, ball, ctx)
    assert out.shape == (b, t, N_CLASSES)

    # permutation invariance: shuffling player slots (+mask) cannot change the output
    perm = torch.randperm(N_PLAYER_SLOTS)
    with torch.no_grad():
        out_perm = model(players[:, :, perm, :], pmask[:, :, perm], ball, ctx)
    assert torch.allclose(out, out_perm, atol=1e-5)

    # causality: perturbing the LAST frame leaves all earlier-frame predictions exactly unchanged
    players_p = players.clone()
    ball_p, ctx_p = ball.clone(), ctx.clone()
    players_p[:, t - 1] = torch.randn(b, N_PLAYER_SLOTS, fp)
    ball_p[:, t - 1] = torch.randn(b, fb)
    ctx_p[:, t - 1] = torch.randn(b, fc)
    with torch.no_grad():
        out_p = model(players_p, pmask, ball_p, ctx_p)
    assert torch.allclose(out[:, : t - 1], out_p[:, : t - 1], atol=1e-6)
    assert (out[:, t - 1] - out_p[:, t - 1]).abs().max() > 1e-4


def test_empty_player_set_is_finite():
    torch = pytest.importorskip("torch")
    from plot.models.eval_bar.seq_model import SeqEPV

    b, t, fp = 1, 3, len(PLAYER_FEATURES)
    players = torch.randn(b, t, N_PLAYER_SLOTS, fp)
    pmask = torch.zeros(b, t, N_PLAYER_SLOTS, dtype=torch.bool)  # no players tracked at all
    ball = torch.randn(b, t, len(BALL_FEATURES))
    ctx = torch.randn(b, t, len(CONTEXT_FEATURES))
    with torch.no_grad():
        out = SeqEPV().eval()(players, pmask, ball, ctx)
    assert torch.isfinite(out).all()


def test_collate_pads_and_masks():
    torch = pytest.importorskip("torch")
    from plot.models.eval_bar.seq_model import SeqDataset, collate

    seqs = [_toy_seq(4, 2, seed=1), _toy_seq(7, 0, seed=2)]
    batch = collate([SeqDataset(seqs)[i] for i in range(2)])
    assert batch["players"].shape == (2, 7, N_PLAYER_SLOTS, len(PLAYER_FEATURES))
    # the shorter possession (len 4) is padded; its tail timesteps are invalid + carry weight 0
    assert batch["tmask"][0, :4].all() and not batch["tmask"][0, 4:].any()
    assert torch.allclose(batch["wframe"][0, :4], torch.full((4,), 0.25))
    assert float(batch["wframe"][0, 4:].sum()) == 0.0
    # each possession's frame weights sum to 1 (equal total weight per possession)
    assert pytest.approx(float(batch["wframe"][0].sum()), abs=1e-6) == 1.0
    assert pytest.approx(float(batch["wframe"][1].sum()), abs=1e-6) == 1.0


def test_predict_seq_columns_feed_oof_arrays():
    pytest.importorskip("torch")
    from plot.models.eval_bar.crossval import oof_arrays
    from plot.models.eval_bar.seq_model import SeqEPV, predict_seq

    seqs = [_toy_seq(5, 1, seed=3), _toy_seq(8, 3, seed=4)]
    oof = predict_seq(SeqEPV().eval(), seqs, device="cpu", max_frames=320)
    assert set(["game_id", "possession_id", "weight", "y", "epv", *[f"p{k}" for k in range(N_CLASSES)]]).issubset(
        set(oof.columns)
    )
    assert oof.height == 5 + 8
    # probabilities sum to 1 and EPV = sum_k k*p_k
    probs = np.column_stack([oof[f"p{k}"].to_numpy() for k in range(N_CLASSES)])
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)
    assert np.allclose(oof["epv"].to_numpy(), probs @ np.arange(N_CLASSES), atol=1e-5)
    a = oof_arrays(oof)  # the shared calibration entrypoint must accept it unchanged
    assert a["probs"].shape == (13, N_CLASSES) and a["epv"].shape == (13,)


def test_max_frames_cap_keeps_most_recent():
    pytest.importorskip("torch")
    from plot.models.eval_bar.seq_model import SeqDataset

    s = _toy_seq(20, 2, seed=5)
    ds = SeqDataset([s], max_frames=8)
    item = ds[0]
    assert item["length"] == 8
    assert ds.n_truncated == 1
    # kept frames are the LAST 8 (most recent), in order
    assert np.array_equal(item["wall_ms"].numpy(), np.arange(12, 20))


def test_epv_frame_table_caps_pathological_possession():
    # Regression: a boundary-merge artifact (thousands of frames) must NOT pad a whole batch and
    # OOM — epv_frame_table caps each possession to its last max_frames. Short possessions untouched.
    pytest.importorskip("torch")
    import polars as pl

    from plot.models.eval_bar.seq_model import SeqEPV, epv_frame_table

    short = _toy_seq(50, 1, seed=1)
    short.possession_id = 1
    monster = _toy_seq(5000, 2, seed=2)   # artifact spanning thousands of frames
    monster.possession_id = 2
    out = epv_frame_table(SeqEPV().eval(), [short, monster], device="cpu",
                          batch_possessions=8, max_frames=640)
    n_short = out.filter(pl.col("possession_id") == 1).height
    n_monster = out.filter(pl.col("possession_id") == 2).height
    assert n_short == 50          # real possession fully scored
    assert n_monster == 640       # artifact truncated to the cap, not 5000 -> bounded memory
    wm = out.filter(pl.col("possession_id") == 2)["wall_clock_ms"].to_list()
    assert wm[0] == 5000 - 640 and wm[-1] == 4999   # kept frames are the most recent


def test_cap_seq_bounds_corpus_memory():
    # Regression: build_sequence_corpus holds the whole corpus in RAM; an artifact possession of
    # thousands of frames (compresses ~40x on disk, explodes decompressed) must be capped on load,
    # and the slice COPIED so the giant base array is freed (a view would keep it alive -> OOM).
    from plot.models.eval_bar.seq_dataset import cap_seq

    long = _toy_seq(5000, 2, seed=3)
    capped = cap_seq(long, 640)
    assert capped.length == 640
    assert capped.wall_ms[0] == 5000 - 640 and capped.wall_ms[-1] == 4999   # most-recent frames
    assert capped.players.base is None and capped.ball.base is None         # copies, not views
    assert cap_seq(_toy_seq(50, 1, seed=4), 640).length == 50               # short passes through


def test_load_cache_copies_and_caps(tmp_path):
    # Regression for the corpus-load OOM: _load_cache must store COPIES (not numpy views) of capped
    # slices, so a short possession's slice can't pin the whole game's array (monster possessions
    # included) and balloon the resident corpus at scale.
    from plot.models.eval_bar.seq_dataset import _load_cache, _save_cache

    short = _toy_seq(50, 1, seed=1)
    short.possession_id = 1
    monster = _toy_seq(5000, 2, seed=2)   # artifact possession
    monster.possession_id = 2
    p = tmp_path / "g_seq.npz"
    _save_cache(p, [short, monster], 0.1)
    seqs, bc = _load_cache(p)
    assert abs(bc - 0.1) < 1e-9
    by = {s.possession_id: s for s in seqs}
    assert by[1].length == 50 and by[2].length == 640          # monster capped to last 640 on load
    for s in seqs:                                             # every array is an owned copy, no view
        assert s.players.base is None and s.ball.base is None and s.wall_ms.base is None


def test_tiny_training_runs_and_improves():
    pytest.importorskip("torch")
    from plot.eval.calibration import multiclass_logloss
    from plot.models.eval_bar.seq_model import predict_seq, train_seq

    # a learnable signal: label is encoded in the context so the model can drive logloss down
    seqs = []
    for i in range(40):
        lab = i % N_CLASSES
        s = _toy_seq(6, lab, seed=100 + i)
        s.ctx[:] = 0.0
        s.ctx[:, 0] = lab  # plant the label in a context channel
        seqs.append(s)
    cfg = {"max_epochs": 12, "patience": 12, "batch_possessions": 16, "gru_hidden": 32,
           "frame_embed_dim": 32, "player_embed_dim": 16}
    model, info = train_seq(seqs, seqs, cfg=cfg, device="cpu", verbose=False)
    oof = predict_seq(model, seqs, device="cpu")
    probs = np.column_stack([oof[f"p{k}"].to_numpy() for k in range(N_CLASSES)])
    ll = multiclass_logloss(probs, oof["y"].to_numpy(), oof["weight"].to_numpy())
    assert ll < np.log(N_CLASSES)  # beats the uniform-prior logloss (ln 4 ~ 1.386)
    assert info["best_epoch"] >= 0


# --------------------------------------------------------------------------------------------
# data-gated: real-game build + quarantine parity with the baseline corpus
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_DATA, reason="local tracking JSON not present")
def test_build_game_sequences_shapes():
    from plot.models.eval_bar.seq_dataset import build_game_sequences

    seqs, rep = build_game_sequences(_GAME)
    assert rep["status"] == "ok" and seqs
    s = seqs[0]
    assert s.players.shape[1:] == (N_PLAYER_SLOTS, len(PLAYER_FEATURES))
    assert s.ball.shape[1] == len(BALL_FEATURES) and s.ctx.shape[1] == len(CONTEXT_FEATURES)
    assert s.players.shape[0] == s.pmask.shape[0] == s.ball.shape[0] == s.ctx.shape[0]
    assert 0 <= s.label <= N_CLASSES - 1
    # offense/defense flags are mutually exclusive on real players
    m = s.pmask
    both = (s.players[..., 2] > 0.5) & (s.players[..., 3] > 0.5)
    assert not both[m].any()


@pytest.mark.skipif(not _HAVE_DATA, reason="local tracking JSON not present")
def test_quarantine_and_frame_count_match_baseline():
    from plot.models.eval_bar.dataset import build_game_features
    from plot.models.eval_bar.seq_dataset import build_game_sequences

    X, base_rep = build_game_features(_GAME)
    seqs, seq_rep = build_game_sequences(_GAME)
    # identical clean/quarantine verdict (same pipeline, same threshold)
    assert (X is None) == (seqs is None)
    assert base_rep["status"] == seq_rep["status"]
    if X is not None:
        # same frames as the baseline feature matrix (one row per kept ~10 fps moment)
        total = sum(s.length for s in seqs)
        assert total == X.height
        assert pytest.approx(base_rep["backcourt_fraction"], abs=1e-6) == seq_rep["backcourt_fraction"]


@pytest.mark.skipif(not _HAVE_DATA, reason="local tracking JSON not present")
def test_seq_label_matches_possession_points():
    from plot.io.loaders import events_table, load_game_json, team_info
    from plot.models.eval_bar.seq_dataset import build_game_sequences
    from plot.possessions.segment import segment_possessions

    seqs, _ = build_game_sequences(_GAME)
    game = load_game_json(Path("data/raw/json", f"{_GAME}.json"))
    _, poss = segment_possessions(events_table(Path("data/raw/2015-16_pbp.csv"), _GAME), team_info(game))
    by_id = dict(zip(poss["possession_id"].to_list(), poss["points"].to_list(), strict=True))
    for s in seqs[:50]:
        expected = min(int(by_id[s.possession_id] or 0), N_CLASSES - 1)
        assert s.label == expected
