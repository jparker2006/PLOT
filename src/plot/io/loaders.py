"""Parse a raw SportVU game + play-by-play into tidy polars tables.

Three tables come out of a game:

- ``moments``  : long format, one row per (event, moment, entity). The ball is an entity
                 with ``player_id == -1`` / ``team_id == -1``; the other 10 rows per moment
                 are players. Coordinates are in feet (court is 94 x 50).
- ``players``  : id -> name / team / jersey / position (union over the game's rosters).
- ``events``   : the play-by-play rows for this game (joined later by ``event_id``).

Note the documented data quirks (see DATA.md): a moment may be missing coordinates, and an
event's moments often bleed past the event boundary, so the same wall-clock moment can appear
in two adjacent events. We keep moments event-scoped here; de-duplication onto a single game
timeline is handled downstream when needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

BALL_ID = -1

# Play-by-play columns we keep (the source CSV has 34).
_PBP_COLS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
    "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION",
    "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ID",
    "PERSON2TYPE", "PLAYER2_ID", "PLAYER2_TEAM_ID",
    "PERSON3TYPE", "PLAYER3_ID", "PLAYER3_TEAM_ID",
    "SCORE", "SCOREMARGIN",
]


def load_game_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def players_table(game: dict) -> pl.DataFrame:
    """Player metadata, unioned over every event's home/visitor rosters."""
    rows: dict[int, tuple] = {}
    for ev in game["events"]:
        for side in ("home", "visitor"):
            team = ev[side]
            for p in team["players"]:
                rows[p["playerid"]] = (
                    p["playerid"],
                    f'{p["firstname"]} {p["lastname"]}',
                    team["teamid"],
                    team["abbreviation"],
                    str(p.get("jersey", "")),
                    p.get("position", ""),
                )
    return pl.DataFrame(
        list(rows.values()),
        schema=["player_id", "name", "team_id", "team_abbr", "jersey", "position"],
        orient="row",
    )


def moments_dataframe(game: dict, event_ids: list[int] | None = None) -> pl.DataFrame:
    """Long per-(event, moment, entity) table. Optionally restrict to ``event_ids``."""
    keep = set(event_ids) if event_ids is not None else None
    cols: dict[str, list] = {k: [] for k in (
        "event_id", "moment_idx", "quarter", "wall_clock_ms", "game_clock", "shot_clock",
        "entity", "team_id", "player_id", "x", "y", "z",
    )}
    for ev in game["events"]:
        eid = int(ev["eventId"])
        if keep is not None and eid not in keep:
            continue
        for mi, m in enumerate(ev["moments"]):
            quarter, wall_ms, gclock, sclock = m[0], m[1], m[2], m[3]
            for pos in m[5]:
                team, pid, x, y, z = pos[0], pos[1], pos[2], pos[3], pos[4]
                cols["event_id"].append(eid)
                cols["moment_idx"].append(mi)
                cols["quarter"].append(quarter)
                cols["wall_clock_ms"].append(wall_ms)
                cols["game_clock"].append(gclock)
                cols["shot_clock"].append(sclock)
                cols["entity"].append("ball" if pid == BALL_ID else "player")
                cols["team_id"].append(team)
                cols["player_id"].append(pid)
                cols["x"].append(x)
                cols["y"].append(y)
                cols["z"].append(z)
    return pl.DataFrame(cols)


def events_table(pbp_path: str | Path, game_id: int | str) -> pl.DataFrame:
    """Play-by-play rows for one game, with a coalesced ``description`` and tidy names."""
    gid = int(game_id)
    df = (
        pl.scan_csv(pbp_path, infer_schema_length=20000)
        .filter(pl.col("GAME_ID") == gid)
        .select(_PBP_COLS)
        .collect()
    )
    return (
        df.rename({"EVENTNUM": "event_id", "EVENTMSGTYPE": "msg_type",
                   "EVENTMSGACTIONTYPE": "action_type", "PERIOD": "period"})
        .with_columns(
            pl.coalesce(
                pl.col("HOMEDESCRIPTION"),
                pl.col("VISITORDESCRIPTION"),
                pl.col("NEUTRALDESCRIPTION"),
            ).alias("description")
        )
        .sort("event_id")
    )


def load_game(
    game_id: str,
    raw_dir: str | Path = "data/raw",
    pbp_path: str | Path | None = None,
    event_ids: list[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load (moments, players, events) for a game id like ``"0021500499"``."""
    raw = Path(raw_dir)
    game = load_game_json(raw / "json" / f"{game_id}.json")
    moments = moments_dataframe(game, event_ids=event_ids)
    players = players_table(game)
    pbp_path = pbp_path or (raw / "2015-16_pbp.csv")
    events = events_table(pbp_path, game_id)
    return moments, players, events
