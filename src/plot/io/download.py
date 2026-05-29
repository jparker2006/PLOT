"""Fetch public 2015-16 SportVU tracking + play-by-play from their original sources.

We deliberately do NOT redistribute the raw data (see DATA.md); this module downloads
from the public community archives on demand.

- Tracking: per-game ``.7z`` in ``linouk23/NBA-Player-Movements``. NOTE: the public archive
  only covers **2015-10-27 .. 2016-01-23** (~half the season); there are no post-January games
  (so e.g. Kobe's April farewell is not available anywhere public).
- Play-by-play: full-season CSV in ``sumitrodatta/nba-alt-awards``, joined to tracking events
  by ``EVENTNUM == eventId``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import py7zr
import requests
from tqdm import tqdm

GH_API = (
    "https://api.github.com/repos/linouk23/NBA-Player-Movements/"
    "contents/data/2016.NBA.Raw.SportVU.Game.Logs"
)
PBP_URL = (
    "https://github.com/sumitrodatta/nba-alt-awards/raw/main/"
    "Historical/PBP%20Data/2015-16_pbp.csv"
)

_NAME_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\.([A-Z]{3})\.at\.([A-Z]{3})\.7z$")
_TIER_N = {"T0": 1, "T1": 5, "T2": 25, "T3": None}  # None => all


@dataclass(frozen=True)
class Game:
    name: str  # canonical "MM.DD.YYYY.AWAY.at.HOME.7z"
    date: str  # ISO "YYYY-MM-DD"
    away: str
    home: str
    download_url: str
    size: int  # bytes of the .7z

    @property
    def stem(self) -> str:
        return self.name[:-3]  # drop ".7z"


def parse_game_filename(name: str) -> tuple[str, str, str] | None:
    """Return ``(iso_date, away, home)`` from a (possibly dir-prefixed) archive name, or None.

    A handful of archive names in the source repo are concatenated with the directory
    prefix (e.g. ``2016.NBA.Raw.SportVU.Game.Logs12.05.2015.BOS.at.SAS.7z``); the regex
    search recovers the canonical tail.
    """
    m = _NAME_RE.search(name)
    if not m:
        return None
    mm, dd, yyyy, away, home = m.groups()
    return f"{yyyy}-{mm}-{dd}", away, home


def list_games(timeout: int = 60) -> list[Game]:
    """List every available tracking game via the GitHub contents API (one request)."""
    r = requests.get(GH_API, params={"per_page": 1000}, timeout=timeout)
    r.raise_for_status()
    games: list[Game] = []
    for x in r.json():
        nm = x.get("name", "")
        if not nm.endswith(".7z"):
            continue
        m = _NAME_RE.search(nm)
        if m is None:
            continue
        mm, dd, yyyy, away, home = m.groups()
        games.append(
            Game(
                name=m.group(0),
                date=f"{yyyy}-{mm}-{dd}",
                away=away,
                home=home,
                download_url=x["download_url"],
                size=int(x.get("size", 0)),
            )
        )
    games.sort(key=lambda g: (g.date, g.away, g.home))
    return games


def select_games(
    games: list[Game],
    *,
    game: str | None = None,
    tier: str = "T0",
    n: int | None = None,
    seed: int = 9,
) -> list[Game]:
    """Filter by substring (``game``) and/or sample a reproducible subset by ``tier``/``n``."""
    sel = games
    if game:
        sub = game.lower()
        sel = [g for g in games if sub in g.name.lower()]
    count = n if n is not None else _TIER_N[tier]
    if count is None or count >= len(sel):
        return sel
    rng = random.Random(seed)
    return sorted(rng.sample(sel, count), key=lambda g: (g.date, g.away, g.home))


def _stream_download(url: str, dest: Path, desc: str, timeout: int = 180) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))


def download_game(
    g: Game, raw_dir: str | Path = "data/raw", *, extract: bool = True, overwrite: bool = False
) -> Path:
    """Download (and optionally extract) one game. Returns the archive or extracted-json path."""
    raw = Path(raw_dir)
    arc = raw / g.name
    if overwrite or not arc.exists():
        _stream_download(g.download_url, arc, desc=g.name)
    if not extract:
        return arc
    out = raw / "json"
    out.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(arc, "r") as z:
        members = [m for m in z.getnames() if m.endswith(".json")]
        target = out / members[0]
        if overwrite or not target.exists():
            z.extract(path=out, targets=members)
    return out / members[0]


def download_pbp(raw_dir: str | Path = "data/raw", *, overwrite: bool = False) -> Path:
    """Download the full-season play-by-play CSV once."""
    dest = Path(raw_dir) / "2015-16_pbp.csv"
    if overwrite or not dest.exists():
        _stream_download(PBP_URL, dest, desc="pbp.csv")
    return dest
