#!/usr/bin/env python
"""Build (and cache) per-game eval-bar feature parquets, reporting orientation + cleanup + QC.

    uv run python scripts/build_eval_bar_features.py                       # all local games
    uv run python scripts/build_eval_bar_features.py --games 0021500053    # one game

Writes data/processed/{game_id}_eval_bar_features.parquet for clean games and prints the
orientation map, cleanup stats, backcourt-fraction QC, and quarantine status per game.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot.models.eval_bar.dataset import build_game_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for gid in games:
        X, rep = build_game_features(gid, raw_dir=args.raw_dir)
        if X is not None:
            X.write_parquet(out / f"{gid}_eval_bar_features.parquet")
            n_ok += 1
            cs = rep.get("clean_stats", {})
            print(f"[ok]   {gid} {rep['matchup']:>8}  rows={rep['n_rows']:>6} "
                  f"backcourt={rep['backcourt_fraction']}  kept={cs.get('kept_frames')} "
                  f"ds={cs.get('downsample_factor')}x")
        else:
            print(f"[QUAR] {gid}  {rep.get('reason')}")
    print(f"\n{n_ok}/{len(games)} games written to {out}")


if __name__ == "__main__":
    main()
