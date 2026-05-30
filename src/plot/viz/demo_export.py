"""Build the precomputed per-possession JSON the Next.js demo plays — a chess.com-style review.

Each possession becomes: the canonical 10 fps tracking (ball + 10 players, offense attacking the
right rim), the per-frame **eval bar** (EPV from the leakage-free recalibrated trace), and the
**decision markers** — at each open ball-handler pass-up, the regret and a chess.com-style badge
(great / good / inaccuracy / mistake / blunder), thresholded on *points left on the table*.

No model runs in the browser: this module turns already-computed frames (canonical positions + the
OOF EPV trace + the per-decision regret) into static JSON. The badge logic + payload assembly are
pure (testable); the script ``scripts/export_demo_json.py`` does the I/O.
"""

from __future__ import annotations

import polars as pl

# badge tiers on the regret of a pass-up decision (points left on the table).
# A clearly *good* pass (signed regret strongly negative — the pass beat the open shot) is "great";
# otherwise tier on the clipped points left behind. Thresholds documented, not tuned.
GREAT_SIGNED = -0.30
BADGE_EDGES = [(0.10, "good"), (0.25, "inaccuracy"), (0.50, "mistake")]  # upper-exclusive; else "blunder"


def badge_for(regret_signed: float, regret_clipped: float) -> str:
    """chess.com-style label for one pass-up decision."""
    if regret_signed <= GREAT_SIGNED:
        return "great"
    for hi, name in BADGE_EDGES:
        if regret_clipped < hi:
            return name
    return "blunder"


def _round_cols(df: pl.DataFrame, mapping: dict[str, int]) -> pl.DataFrame:
    return df.with_columns([pl.col(c).round(n) for c, n in mapping.items()])


def build_possession(
    pid: int,
    frames: pl.DataFrame,
    decisions: pl.DataFrame,
    poss_row: dict,
    *,
    stride: int = 1,
) -> dict:
    """Assemble one possession payload.

    ``frames``: long canonical frames for THIS possession, already joined to EPV, with columns
    ``wall_clock_ms, game_clock, entity, player_id, team_id, x, y, z, epv`` (x/y canonical).
    ``decisions``: this possession's pass-up regret rows with ``wall_clock_ms, player_id,
    regret_signed, regret_clipped, best_available, post_epv``.
    """
    wall = sorted(frames["wall_clock_ms"].unique().to_list())[::stride]
    wall_to_idx = {w: i for i, w in enumerate(wall)}
    fset = frames.filter(pl.col("wall_clock_ms").is_in(wall))

    ball = {r["wall_clock_ms"]: r for r in fset.filter(pl.col("entity") == "ball").to_dicts()}
    players_by_w: dict[int, list] = {w: [] for w in wall}
    for r in fset.filter(pl.col("entity") == "player").to_dicts():
        players_by_w[r["wall_clock_ms"]].append(
            [int(r["player_id"]), int(r["team_id"]), round(r["x"], 2), round(r["y"], 2)]
        )
    epv_by_w = {r["wall_clock_ms"]: r["epv"] for r in
                fset.filter(pl.col("entity") == "ball").select("wall_clock_ms", "epv").to_dicts()}

    out_frames = []
    for w in wall:
        b = ball.get(w)
        out_frames.append({
            "gc": round(b["game_clock"], 1) if b and b["game_clock"] is not None else None,
            "ball": [round(b["x"], 2), round(b["y"], 2), round(b["z"], 2)] if b else None,
            "pl": players_by_w[w],
            "epv": round(epv_by_w.get(w), 3) if epv_by_w.get(w) is not None else None,
        })

    decs = []
    for d in decisions.sort("wall_clock_ms").to_dicts():
        # snap the decision to the nearest exported frame
        w = min(wall, key=lambda x: abs(x - d["wall_clock_ms"]))
        decs.append({
            "frame": wall_to_idx[w],
            "player": int(d["player_id"]),
            "regret_signed": round(float(d["regret_signed"]), 3),
            "regret_clipped": round(float(d["regret_clipped"]), 3),
            "best_available": round(float(d["best_available"]), 3),
            "post_epv": round(float(d["post_epv"]), 3),
            "badge": badge_for(float(d["regret_signed"]), float(d["regret_clipped"])),
        })

    return {
        "possession_id": int(pid),
        "period": int(poss_row["period"]),
        "offense_team_id": int(poss_row["offense_team_id"]),
        "end_reason": poss_row["end_reason"],
        "points": int(poss_row["points"]) if poss_row.get("points") is not None else 0,
        "n_frames": len(out_frames),
        "frames": out_frames,
        "decisions": decs,
    }


def select_featured_possessions(decisions: pl.DataFrame, *, top_n: int = 10, min_regret: float = 0.0) -> list[int]:
    """Possessions worth featuring: those containing a pass-up decision, ranked by their single
    most-regretful decision (so the demo opens on the clearest 'points left on the table')."""
    if decisions.height == 0:
        return []
    ranked = (
        decisions.group_by("possession_id")
        .agg(pl.col("regret_clipped").max().alias("max_regret"))
        .filter(pl.col("max_regret") >= min_regret)
        .sort("max_regret", descending=True)
    )
    return ranked.head(top_n)["possession_id"].to_list()
