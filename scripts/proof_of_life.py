#!/usr/bin/env python
"""Stage-1 proof of life: animate one possession/event and dump its tracking table.

    uv run python scripts/proof_of_life.py                 # LeBron floater (default)
    uv run python scripts/proof_of_life.py --game 0021500499 --event 13
"""

from __future__ import annotations

import argparse

import polars as pl

from plot.io.loaders import load_game
from plot.viz.animate import animate_event, save_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="0021500499")
    ap.add_argument("--event", type=int, default=2)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth 25fps moment")
    args = ap.parse_args()

    moments, players, events = load_game(args.game, event_ids=[args.event])
    row = events.filter(pl.col("event_id") == args.event)
    desc = row["description"][0] if row.height and row["description"][0] else "(no PBP description)"
    title = f"{args.game}  event {args.event}: {desc}"
    print(title)

    mids = moments["moment_idx"].unique().sort().to_list()
    gif = f"{args.out_dir}/proof_of_life.gif"
    png = f"{args.out_dir}/proof_of_life_frame.png"
    n = animate_event(moments, players, args.event, gif, stride=args.stride, title=title)
    save_frame(moments, players, args.event, mids[len(mids) // 2], png, title=title)

    # Proof-of-life table: clock + ball xy + the 10 player xy at a sampled moment.
    mid = mids[len(mids) // 2]
    sub = moments.filter(pl.col("moment_idx") == mid)
    print(f"\nTable dump at moment_idx={mid} (game_clock {sub['game_clock'][0]:.1f}s):")
    view = sub.join(players.select("player_id", "name", "team_abbr"), on="player_id", how="left")
    print(view.select("entity", "team_abbr", "name", "x", "y", "z").sort("entity"))
    print(f"\nWrote {gif} ({n} frames) and {png}")


if __name__ == "__main__":
    main()
