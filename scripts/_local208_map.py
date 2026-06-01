#!/usr/bin/env python
"""One-off: map the 208-trace game-ids to linouk23 archive names so we can fetch exactly the
games missing locally, then (optionally) download + build_g6 over the full 208. Dry-run by default.

    uv run python scripts/_local208_map.py            # dry run: build + verify mapping only
    uv run python scripts/_local208_map.py --download  # also download the missing archives

Mapping: each game-id's (away,home) abbreviations come from the PBP (home team = the team whose
events populate HOMEDESCRIPTION); archives are MM.DD.YYYY.AWAY.at.HOME.7z. Where a team pair meets
more than once in the window, ids and archives are both chronological, so we zip them in order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from plot.io.download import download_game, list_games

REPO = Path("/Users/jakeparker/Desktop/PLOT")
PBP = REPO / "data/raw/2015-16_pbp.csv"
JSON_DIR = REPO / "data/raw/json"
TRACE = REPO / "data/processed/oof_epv_trace.parquet"


def home_away_by_game() -> dict[str, tuple[str, str]]:
    pbp = pl.read_csv(PBP, infer_schema_length=50000,
                      columns=["GAME_ID", "PLAYER1_TEAM_ABBREVIATION", "HOMEDESCRIPTION",
                               "VISITORDESCRIPTION"])
    pbp = pbp.with_columns(pl.col("GAME_ID").cast(pl.Utf8).str.zfill(10).alias("gid"))

    def side_team(df, col):
        d = df.filter(pl.col(col).is_not_null() & pl.col("PLAYER1_TEAM_ABBREVIATION").is_not_null())
        if d.height == 0:
            return None
        return d["PLAYER1_TEAM_ABBREVIATION"].value_counts(sort=True)["PLAYER1_TEAM_ABBREVIATION"][0]

    out = {}
    for gid, g in pbp.group_by("gid"):
        gid = gid[0] if isinstance(gid, tuple) else gid
        home = side_team(g, "HOMEDESCRIPTION")
        away = side_team(g, "VISITORDESCRIPTION")
        if home and away:
            out[gid] = (away, home)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    trace_ids = set(pl.read_parquet(TRACE)["game_id"].unique().to_list())
    local = {p.stem for p in JSON_DIR.glob("*.json")}
    missing = sorted(trace_ids - local)
    print(f"trace={len(trace_ids)} local={len(local)} overlap={len(trace_ids & local)} missing={len(missing)}")

    ha = home_away_by_game()
    games = list_games()  # one GitHub API call
    arch_by_pair: dict[tuple[str, str], list] = {}
    for g in games:
        arch_by_pair.setdefault((g.away, g.home), []).append(g)
    for v in arch_by_pair.values():
        v.sort(key=lambda g: g.date)

    # assign archives to ALL trace ids (chronological zip per pair) so the mapping is injective
    trace_sorted = sorted(trace_ids)
    pair_ids: dict[tuple[str, str], list[str]] = {}
    no_pair = []
    for gid in trace_sorted:
        if gid not in ha:
            no_pair.append(gid)
            continue
        pair_ids.setdefault(ha[gid], []).append(gid)

    id_to_arch: dict[str, object] = {}
    unmapped = []
    for pair, ids in pair_ids.items():
        archs = arch_by_pair.get(pair, [])
        for i, gid in enumerate(sorted(ids)):
            if i < len(archs):
                id_to_arch[gid] = archs[i]
            else:
                unmapped.append(gid)

    mapped_missing = [g for g in missing if g in id_to_arch]
    unmapped_missing = [g for g in missing if g not in id_to_arch]
    print(f"no (away,home) from PBP: {len(no_pair)} | unmapped (no archive): {len(unmapped)}")
    print(f"MISSING mapped to an archive: {len(mapped_missing)} / {len(missing)}")
    if unmapped_missing:
        print(f"MISSING still unmapped ({len(unmapped_missing)}): {unmapped_missing[:10]}")
    # injectivity check
    archs_used = [id_to_arch[g].name for g in id_to_arch]
    print(f"injective: {len(archs_used) == len(set(archs_used))} ({len(archs_used)} ids -> {len(set(archs_used))} archives)")

    Path("/tmp/to_download.txt").write_text("\n".join(id_to_arch[g].name for g in mapped_missing))
    print(f"wrote /tmp/to_download.txt ({len(mapped_missing)} archives)")

    if args.download:
        # Robust: fetch EVERY archive whose (away,home) matches a missing id's pair. Extraction names
        # files by the true game-id inside the .7z, so this guarantees each missing id is covered even
        # if a team pair meets twice; a few extra (non-trace) jsons are harmless (build_g6 intersects).
        want = {}
        for gid in mapped_missing:
            for g in arch_by_pair.get(ha[gid], []):
                want[g.name] = g
        archives = sorted(want.values(), key=lambda g: g.date)
        print(f"downloading {len(archives)} archives (covers {len(mapped_missing)} missing ids + pair-extras)")
        ok = 0
        for i, g in enumerate(archives, 1):
            try:
                download_game(g, str(REPO / "data/raw"), extract=True)
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {g.name}: {e}")
            if i % 20 == 0:
                now = {p.stem for p in JSON_DIR.glob("*.json")}
                print(f"  {i}/{len(archives)} (ok={ok}); trace∩local {len(trace_ids & now)}/{len(trace_ids)}")
        now_local = {p.stem for p in JSON_DIR.glob("*.json")}
        cov = len(trace_ids & now_local)
        print(f"DONE downloads ok={ok}; trace∩local now {cov}/{len(trace_ids)}")
        still = sorted(trace_ids - now_local)
        if still:
            print(f"STILL MISSING {len(still)}: {still[:15]}")


if __name__ == "__main__":
    main()
