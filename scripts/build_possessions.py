#!/usr/bin/env python
"""Segment possessions for a game and report validation checks.

    uv run python scripts/build_possessions.py                       # game 0021500499
    uv run python scripts/build_possessions.py --game 0021500499 --write
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot.io.loaders import events_table, load_game_json, team_info
from plot.possessions.segment import segment_possessions, validate_segmentation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="0021500499")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--write", action="store_true", help="write events+possessions parquet to out-dir")
    args = ap.parse_args()

    ti = team_info(load_game_json(Path(args.raw_dir) / "json" / f"{args.game}.json"))
    events = events_table(Path(args.raw_dir) / "2015-16_pbp.csv", args.game)
    events_pid, poss = segment_possessions(events, ti)
    checks = validate_segmentation(events_pid, poss, ti)

    print(f"game {args.game}  ({ti['visitor_abbr']} @ {ti['home_abbr']})")
    ok = True
    for k, v in checks.items():
        if k == "summary":
            continue
        ok = ok and v
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  summary:", checks["summary"])
    print("  ALL CHECKS PASS" if ok else "  *** SOME CHECKS FAILED ***")

    if args.write:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        events_pid.write_parquet(out / f"{args.game}_events.parquet")
        poss.write_parquet(out / f"{args.game}_possessions.parquet")
        print(f"  wrote {out}/{args.game}_events.parquet and _possessions.parquet")


if __name__ == "__main__":
    main()
