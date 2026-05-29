#!/usr/bin/env python
"""CLI to download public 2015-16 SportVU tracking + play-by-play.

Examples:
    # List what's available
    uv run python scripts/download_data.py --list

    # Proof-of-life: one LeBron Cavs game (no PBP)
    uv run python scripts/download_data.py --game 01.02.2016.ORL.at.CLE --no-pbp

    # Reproducible subsets
    uv run python scripts/download_data.py --tier T1     # ~5 games
"""

from __future__ import annotations

import argparse

from plot.io.download import download_game, download_pbp, list_games, select_games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", help="substring matching specific game(s), e.g. 'CLE'")
    ap.add_argument("--tier", default="T0", choices=["T0", "T1", "T2", "T3"])
    ap.add_argument("--n", type=int, default=None, help="override number of games selected")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--no-extract", action="store_true", help="keep .7z, do not extract JSON")
    ap.add_argument("--no-pbp", action="store_true", help="skip the play-by-play CSV download")
    ap.add_argument("--list", action="store_true", help="list available games and exit")
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    games = list_games()
    print(f"{len(games)} games available: {games[0].date} .. {games[-1].date}")

    if args.list:
        show = (
            [g for g in games if args.game.lower() in g.name.lower()] if args.game else games
        )
        for g in show:
            print(f"  {g.date}  {g.away:>3} @ {g.home:<3}  {g.size / 1e6:5.1f} MB  {g.name}")
        return

    sel = select_games(games, game=args.game, tier=args.tier, n=args.n, seed=args.seed)
    if not sel:
        raise SystemExit(f"No games matched (game={args.game!r}, tier={args.tier}).")
    print(f"Selected {len(sel)} game(s):")
    for g in sel:
        print(f"  {g.date}  {g.away} @ {g.home}  {g.name}")

    if not args.no_pbp:
        print(f"PBP      -> {download_pbp(args.raw_dir)}")
    for g in sel:
        print(f"tracking -> {download_game(g, args.raw_dir, extract=not args.no_extract)}")


if __name__ == "__main__":
    main()
