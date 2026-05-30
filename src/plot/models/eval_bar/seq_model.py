"""Sequence EPV model: permutation-invariant per-frame set encoder -> causal GRU -> outcome head.

Architecture (Stage-2 upgrade over the LightGBM baseline):

1. **Set encoder (DeepSets, permutation-invariant).** Each of the up-to-ten player tokens passes
   through a shared MLP ``phi``; the embeddings are pooled SEPARATELY over offense and defense
   (masked mean AND max), so the frame embedding sees both squads' configurations while staying
   invariant to player ordering. The ball-handler flag rides along as a per-player input feature.
2. **Frame embedding.** The pooled set vector is concatenated with the ball token and the context
   scalars and projected down.
3. **Temporal model.** A UNIDIRECTIONAL (causal) GRU consumes the frame-embedding sequence; the
   hidden state at frame t depends only on frames <= t, so the per-frame EPV read-off never leaks
   the future (a bidirectional encoder would, and is deliberately avoided).
4. **Outcome head.** A linear layer at every timestep yields logits over points {0,1,2,3}; EPV =
   sum_k k * softmax(logits)_k. Trained with per-frame multiclass cross-entropy, each possession
   weighted to contribute equal total weight (1 / frames), matching the baseline exactly.

This module owns the torch dependency (the ``seq`` extra). The data builder in ``seq_dataset`` is
torch-free; everything here tensorizes ``PossessionSeq`` objects, pads per batch, and trains.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from plot.models.eval_bar.seq_dataset import (
    BALL_FEATURES,
    CONTEXT_FEATURES,
    N_CLASSES,
    N_PLAYER_SLOTS,
    PLAYER_FEATURES,
    PossessionSeq,
)

_IS_OFF_IDX = PLAYER_FEATURES.index("is_off")
_IS_DEF_IDX = PLAYER_FEATURES.index("is_def")


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------
def _masked_mean(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean of h over the player axis (dim=2) using float mask m [B,T,10,1]; empty set -> 0."""
    return (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)


def _masked_max(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Max of h over the player axis; positions with m==0 are excluded; empty set -> 0."""
    neg = torch.finfo(h.dtype).min
    mx = h.masked_fill(m == 0, neg).max(dim=2).values
    return torch.where(m.sum(dim=2) > 0, mx, torch.zeros_like(mx))


class SetEncoder(nn.Module):
    """Permutation-invariant DeepSets encoder with separate offense/defense (mean+max) pooling."""

    def __init__(self, in_dim: int, embed_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(4 * embed_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, players: torch.Tensor, pmask: torch.Tensor) -> torch.Tensor:
        # players [B,T,10,Fp]; pmask [B,T,10] bool
        h = self.phi(players)                                        # [B,T,10,E]
        present = pmask.unsqueeze(-1).float()
        off = present * (players[..., _IS_OFF_IDX] > 0.5).unsqueeze(-1).float()
        deff = present * (players[..., _IS_DEF_IDX] > 0.5).unsqueeze(-1).float()
        pooled = torch.cat(
            [_masked_mean(h, off), _masked_max(h, off), _masked_mean(h, deff), _masked_max(h, deff)],
            dim=-1,
        )
        return self.proj(pooled)                                     # [B,T,out_dim]


class SeqEPV(nn.Module):
    """Set encoder -> frame embedding (with ball + context) -> causal GRU -> per-frame head."""

    def __init__(
        self,
        *,
        player_embed_dim: int = 64,
        frame_embed_dim: int = 96,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        self.set_enc = SetEncoder(len(PLAYER_FEATURES), player_embed_dim, frame_embed_dim, dropout)
        frame_in = frame_embed_dim + len(BALL_FEATURES) + len(CONTEXT_FEATURES)
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_in, frame_embed_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            frame_embed_dim, gru_hidden, num_layers=gru_layers, batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(gru_hidden, n_classes)

    def forward(
        self, players: torch.Tensor, pmask: torch.Tensor, ball: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        set_emb = self.set_enc(players, pmask)                       # [B,T,F]
        frame = self.frame_mlp(torch.cat([set_emb, ball, ctx], dim=-1))
        seq, _ = self.gru(frame)                                     # [B,T,H] (causal)
        return self.head(seq)                                        # [B,T,K] logits


# --------------------------------------------------------------------------------------------
# data plumbing
# --------------------------------------------------------------------------------------------
class SeqDataset(Dataset):
    """Wraps a list of ``PossessionSeq``; caps over-long possessions to the most recent frames."""

    def __init__(self, seqs: list[PossessionSeq], max_frames: int = 320):
        self.seqs = seqs
        self.max_frames = max_frames
        self.n_truncated = sum(s.length > max_frames for s in seqs)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> dict:
        s = self.seqs[i]
        t = s.length
        sl = slice(t - self.max_frames, t) if t > self.max_frames else slice(0, t)
        kept = min(t, self.max_frames)
        return {
            "players": torch.from_numpy(np.ascontiguousarray(s.players[sl])),
            "pmask": torch.from_numpy(np.ascontiguousarray(s.pmask[sl])),
            "ball": torch.from_numpy(np.ascontiguousarray(s.ball[sl])),
            "ctx": torch.from_numpy(np.ascontiguousarray(s.ctx[sl])),
            "wall_ms": torch.from_numpy(np.ascontiguousarray(s.wall_ms[sl])),
            "label": int(s.label),
            "length": kept,
            "wframe": 1.0 / kept,
            "game_id": s.game_id,
            "possession_id": s.possession_id,
        }


def collate(batch: list[dict]) -> dict:
    """Pad a batch of variable-length possessions to the batch-max time; build a validity mask."""
    b = len(batch)
    t = max(x["length"] for x in batch)
    fp, fb, fc = len(PLAYER_FEATURES), len(BALL_FEATURES), len(CONTEXT_FEATURES)
    players = torch.zeros(b, t, N_PLAYER_SLOTS, fp)
    pmask = torch.zeros(b, t, N_PLAYER_SLOTS, dtype=torch.bool)
    ball = torch.zeros(b, t, fb)
    ctx = torch.zeros(b, t, fc)
    tmask = torch.zeros(b, t, dtype=torch.bool)
    wframe = torch.zeros(b, t)
    labels = torch.zeros(b, dtype=torch.long)
    meta = []
    for i, x in enumerate(batch):
        n = x["length"]
        players[i, :n] = x["players"]
        pmask[i, :n] = x["pmask"]
        ball[i, :n] = x["ball"]
        ctx[i, :n] = x["ctx"]
        tmask[i, :n] = True
        wframe[i, :n] = x["wframe"]
        labels[i] = x["label"]
        meta.append((x["game_id"], x["possession_id"], n, x["wall_ms"]))
    return {"players": players, "pmask": pmask, "ball": ball, "ctx": ctx,
            "tmask": tmask, "wframe": wframe, "labels": labels, "meta": meta}


def _move(batch: dict, device: str) -> dict:
    keys = ("players", "pmask", "ball", "ctx", "tmask", "wframe", "labels")
    return {**{k: batch[k].to(device) for k in keys}, "meta": batch["meta"]}


# --------------------------------------------------------------------------------------------
# train / eval / predict
# --------------------------------------------------------------------------------------------
def _frame_logloss(logits: torch.Tensor, labels: torch.Tensor, wframe: torch.Tensor) -> torch.Tensor:
    """Possession-weighted per-frame multiclass cross-entropy (pads carry weight 0)."""
    b, t, k = logits.shape
    ce = nn.functional.cross_entropy(
        logits.reshape(b * t, k),
        labels.unsqueeze(1).expand(b, t).reshape(b * t),
        reduction="none",
    ).reshape(b, t)
    return (ce * wframe).sum() / wframe.sum().clamp(min=1e-8)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    num = den = 0.0
    for batch in loader:
        bt = _move(batch, device)
        logits = model(bt["players"], bt["pmask"], bt["ball"], bt["ctx"])
        loss = _frame_logloss(logits, bt["labels"], bt["wframe"])
        w = float(bt["wframe"].sum())
        num += float(loss) * w
        den += w
    return num / den if den else float("nan")


def train_seq(
    train_seqs: list[PossessionSeq],
    val_seqs: list[PossessionSeq] | None,
    *,
    cfg: dict | None = None,
    device: str | None = None,
    seed: int = 1729,
    verbose: bool = False,
) -> tuple[SeqEPV, dict]:
    """Train a SeqEPV with early stopping on the validation per-frame logloss; return (best model, info)."""
    c = {**_DEFAULTS, **(cfg or {})}
    device = device or default_device()
    torch.manual_seed(seed)

    model = SeqEPV(
        player_embed_dim=c["player_embed_dim"], frame_embed_dim=c["frame_embed_dim"],
        gru_hidden=c["gru_hidden"], gru_layers=c["gru_layers"], dropout=c["dropout"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

    g = torch.Generator().manual_seed(seed)
    train_ds = SeqDataset(train_seqs, c["max_frames"])
    train_dl = DataLoader(train_ds, batch_size=c["batch_possessions"], shuffle=True,
                          collate_fn=collate, generator=g)
    val_dl = (DataLoader(SeqDataset(val_seqs, c["max_frames"]), batch_size=c["batch_possessions"],
                         shuffle=False, collate_fn=collate) if val_seqs else None)

    def _snapshot() -> dict:
        return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    # Best-selection is restricted to epochs >= min_epochs: a single inner-val game can bottom out
    # after one epoch by chance, and freezing a 1-epoch model there cripples that held-out fold.
    min_epochs = int(c["min_epochs"])
    best = {"val": float("inf"), "epoch": -1, "state": None}
    last = {"val": float("inf"), "epoch": -1, "state": None}
    no_improve = 0
    for epoch in range(c["max_epochs"]):
        model.train()
        for batch in train_dl:
            bt = _move(batch, device)
            opt.zero_grad()
            logits = model(bt["players"], bt["pmask"], bt["ball"], bt["ctx"])
            loss = _frame_logloss(logits, bt["labels"], bt["wframe"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            opt.step()
        v = evaluate(model, val_dl, device) if val_dl else float(loss.detach())
        last = {"val": v, "epoch": epoch, "state": _snapshot()}
        if verbose:
            print(f"    epoch {epoch:2d}  val_logloss {v:.4f}")
        if epoch + 1 < min_epochs:
            continue
        if v < best["val"] - 1e-4:
            best, no_improve = last, 0
        else:
            no_improve += 1
            if no_improve >= c["patience"]:
                break
    chosen = best if best["state"] is not None else last
    if chosen["state"] is not None:
        model.load_state_dict(chosen["state"])
    info = {"best_epoch": chosen["epoch"], "best_val_logloss": chosen["val"],
            "n_truncated_train": train_ds.n_truncated, "device": device}
    return model, info


@torch.no_grad()
def predict_seq(model: nn.Module, seqs: list[PossessionSeq], *, device: str | None = None,
                batch_possessions: int = 64, max_frames: int = 320) -> pl.DataFrame:
    """Per-frame OOF predictions as a polars frame: game_id, possession_id, weight, y, p0..pK, epv.

    Shape matches the baseline crossval ``part`` so ``oof_arrays`` + ``calibration`` reuse 1:1.
    """
    device = device or default_device()
    model.eval()
    dl = DataLoader(SeqDataset(seqs, max_frames), batch_size=batch_possessions, shuffle=False,
                    collate_fn=collate)
    points = torch.arange(N_CLASSES, dtype=torch.float32)
    rows = {k: [] for k in ("game_id", "possession_id", "weight", "y")}
    prob_cols: list[list[float]] = [[] for _ in range(N_CLASSES)]
    epv_col: list[float] = []
    for batch in dl:
        bt = _move(batch, device)
        probs = torch.softmax(model(bt["players"], bt["pmask"], bt["ball"], bt["ctx"]), dim=-1).cpu()
        epv = (probs * points).sum(-1)                              # [B,T]
        for i, (gid, pid, n, _wall) in enumerate(batch["meta"]):
            w = 1.0 / n
            rows["game_id"].extend([gid] * n)
            rows["possession_id"].extend([pid] * n)
            rows["weight"].extend([w] * n)
            rows["y"].extend([int(batch["labels"][i])] * n)
            epv_col.extend(epv[i, :n].tolist())
            for k in range(N_CLASSES):
                prob_cols[k].extend(probs[i, :n, k].tolist())
    out = pl.DataFrame({
        **rows, "epv": epv_col,
        **{f"p{k}": prob_cols[k] for k in range(N_CLASSES)},
    })
    return out


@torch.no_grad()
def epv_frame_table(model: nn.Module, seqs: list[PossessionSeq], *, device: str | None = None,
                    batch_possessions: int = 64, max_frames: int = 640) -> pl.DataFrame:
    """Per-frame EPV trace keyed for the action-value layer: game_id, possession_id, wall_clock_ms, frame_idx, epv.

    Caps each possession to its most-recent ``max_frames`` frames (default 640 = 64s, ~2x the 320
    training cap) so every action boundary in a *real* possession (≤ ~40s) still gets an EPV, while
    bounding memory. This matters at scale: boundary-merge segmentation artifacts can span many
    thousands of frames, and uncapped those padded a whole batch to that length and OOM'd the box
    (a single monster possession × batch 64 → tens of GB). The model was itself trained on ≤320-frame
    contexts, so scoring far beyond that is extrapolation regardless. Possessions shorter than the cap
    are unaffected; only artifact mega-possessions are truncated to their last ``max_frames`` (their
    earliest action boundaries then lack an EPV and are simply skipped downstream — they are joined on
    wall_clock_ms, not frame_idx). Used by Stage 3 (``plot.models.action_value``) and the OOF trace.
    """
    device = device or default_device()
    model.eval()
    dl = DataLoader(SeqDataset(seqs, max_frames), batch_size=batch_possessions, shuffle=False,
                    collate_fn=collate)
    points = torch.arange(N_CLASSES, dtype=torch.float32)
    cols: dict[str, list] = {k: [] for k in ("game_id", "possession_id", "wall_clock_ms", "frame_idx", "epv")}
    for batch in dl:
        bt = _move(batch, device)
        probs = torch.softmax(model(bt["players"], bt["pmask"], bt["ball"], bt["ctx"]), dim=-1).cpu()
        epv = (probs * points).sum(-1)                              # [B,T]
        for i, (gid, pid, n, wall) in enumerate(batch["meta"]):
            cols["game_id"].extend([gid] * n)
            cols["possession_id"].extend([pid] * n)
            cols["wall_clock_ms"].extend(wall[:n].tolist())
            cols["frame_idx"].extend(range(n))
            cols["epv"].extend(epv[i, :n].tolist())
    return pl.DataFrame(cols)


def save_model(model: SeqEPV, path) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(p))


def load_model(path, *, cfg: dict | None = None, device: str | None = None) -> SeqEPV:
    c = {**_DEFAULTS, **(cfg or {})}
    device = device or default_device()
    model = SeqEPV(
        player_embed_dim=c["player_embed_dim"], frame_embed_dim=c["frame_embed_dim"],
        gru_hidden=c["gru_hidden"], gru_layers=c["gru_layers"], dropout=c["dropout"],
    )
    model.load_state_dict(torch.load(str(path), map_location="cpu"))
    return model.to(device)


_DEFAULTS = {
    "player_embed_dim": 64, "frame_embed_dim": 96, "gru_hidden": 128, "gru_layers": 1,
    "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5, "batch_possessions": 64,
    "min_epochs": 5, "max_epochs": 40, "patience": 6, "grad_clip": 5.0, "max_frames": 320,
}
