"""Matplotlib prototype animation of an event/possession (Stage-1 proof of life).

Driven by our internal ``moments`` table (see ``plot.io.loaders``). Renders the 10 players
(colored by team, labelled by jersey) and the ball on a drawn court, with the game/shot clock.
Saves a GIF via PillowWriter, so it needs neither a display nor ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

COURT_W, COURT_H = 94.0, 50.0
HOOPS = (5.25, 88.75)
_TEAM_COLORS = ["#1d428a", "#c8102e", "#007a33", "#fdb927"]


def draw_court(ax, color: str = "#444444", lw: float = 1.3) -> None:
    """Draw a simplified full court (feet): boundary, half line, center + rims + paint."""
    ax.add_patch(Rectangle((0, 0), COURT_W, COURT_H, fill=False, lw=lw, ec=color))
    ax.plot([47, 47], [0, 50], color=color, lw=lw)
    ax.add_patch(Circle((47, 25), 6, fill=False, lw=lw, ec=color))
    for hx in HOOPS:
        ax.add_patch(Circle((hx, 25), 0.75, fill=False, lw=lw, ec=color))
        x0 = 0 if hx < 47 else COURT_W
        depth = 19 if hx < 47 else -19
        ax.add_patch(Rectangle((x0, 17), depth, 16, fill=False, lw=lw, ec=color))
    ax.set_xlim(-3, 97)
    ax.set_ylim(-3, 53)
    ax.set_aspect("equal")
    ax.axis("off")


def _clock(sec: float | None) -> str:
    if sec is None:
        return "--:--"
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def _event_frames(moments: pl.DataFrame, event_id: int, stride: int) -> list[dict]:
    df = moments.filter(pl.col("event_id") == event_id).sort("moment_idx")
    frames = []
    for mi in df["moment_idx"].unique().sort().to_list()[::stride]:
        sub = df.filter(pl.col("moment_idx") == mi)
        ball = sub.filter(pl.col("entity") == "ball")
        pl_ = sub.filter(pl.col("entity") == "player")
        frames.append(
            {
                "bx": ball["x"][0] if ball.height else None,
                "by": ball["y"][0] if ball.height else None,
                "px": pl_["x"].to_numpy(),
                "py": pl_["y"].to_numpy(),
                "pteam": pl_["team_id"].to_list(),
                "pid": pl_["player_id"].to_list(),
                "gc": sub["game_clock"][0],
                "sc": sub["shot_clock"][0],
            }
        )
    return frames


def _team_color_map(team_ids) -> dict[int, str]:
    teams = sorted(set(team_ids))
    return {t: _TEAM_COLORS[i % len(_TEAM_COLORS)] for i, t in enumerate(teams)}


def animate_event(
    moments: pl.DataFrame,
    players: pl.DataFrame,
    event_id: int,
    out_path: str | Path,
    *,
    fps: int = 12,
    stride: int = 2,
    title: str = "",
) -> int:
    """Save a GIF of one event. Returns the number of rendered frames."""
    frames = _event_frames(moments, event_id, stride)
    if not frames:
        raise ValueError(f"no moments for event_id={event_id}")
    tcolor = _team_color_map(t for f in frames for t in f["pteam"])
    jersey = {r["player_id"]: r["jersey"] for r in players.iter_rows(named=True)}

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    draw_court(ax)
    pscat = ax.scatter([], [], s=420, zorder=3, edgecolors="white", linewidths=1.0)
    bscat = ax.scatter([], [], s=90, c="#ee6730", zorder=4, edgecolors="black", linewidths=0.6)
    labels = [
        ax.text(0, 0, "", ha="center", va="center", color="white", fontsize=7,
                fontweight="bold", zorder=5)
        for _ in range(10)
    ]
    clock = ax.text(47, 51.5, "", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=9)

    def update(i: int):
        f = frames[i]
        pscat.set_offsets(np.c_[f["px"], f["py"]])
        pscat.set_color([tcolor[t] for t in f["pteam"]])
        if f["bx"] is not None:
            bscat.set_offsets([[f["bx"], f["by"]]])
        for j, txt in enumerate(labels):
            if j < len(f["pid"]):
                txt.set_position((f["px"][j], f["py"][j]))
                txt.set_text(str(jersey.get(f["pid"][j], "")))
            else:
                txt.set_text("")
        clock.set_text(f"game {_clock(f['gc'])}    shot {_clock(f['sc'])}")
        return [pscat, bscat, clock, *labels]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(frames)


def save_frame(
    moments: pl.DataFrame,
    players: pl.DataFrame,
    event_id: int,
    moment_idx: int,
    out_png: str | Path,
    title: str = "",
) -> None:
    """Render one moment to a PNG (handy for quick visual sanity checks)."""
    sub = moments.filter((pl.col("event_id") == event_id) & (pl.col("moment_idx") == moment_idx))
    ball = sub.filter(pl.col("entity") == "ball")
    pl_ = sub.filter(pl.col("entity") == "player")
    tcolor = _team_color_map(pl_["team_id"].to_list())
    jersey = {r["player_id"]: r["jersey"] for r in players.iter_rows(named=True)}

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    draw_court(ax)
    ax.scatter(
        pl_["x"].to_numpy(), pl_["y"].to_numpy(), s=420, zorder=3, edgecolors="white",
        c=[tcolor[t] for t in pl_["team_id"].to_list()],
    )
    if ball.height:
        ax.scatter([ball["x"][0]], [ball["y"][0]], s=90, c="#ee6730", zorder=4, edgecolors="black")
    for r in pl_.iter_rows(named=True):
        ax.text(r["x"], r["y"], str(jersey.get(r["player_id"], "")), ha="center", va="center",
                color="white", fontsize=7, fontweight="bold", zorder=5)
    ax.set_title(title, fontsize=9)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
