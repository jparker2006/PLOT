"""Possession segmentation from play-by-play (stats.nba.com 2015-16 format).

A single forward pass over PBP sorted by (period, event_id), implementing PLOT's locked
conventions: a possession ends on a made FG (+ and-1 / drawn FTs), a defensive rebound, a
turnover, or end of period; an OFFENSIVE REBOUND CONTINUES the possession; free-throw points
belong to the drawing possession.

Designed from the segmentation design workflow and hardened by an adversarial review that ran
the code across the locally-available games. Key robustness properties (verified across games,
incl. OT and games with missing/duplicate period markers):

- A possession never spans a period: PERIOD_END anchors its marker to the just-closed possession,
  PERIOD_BEGIN force-closes any still-open possession, and backfill never crosses a period.
- Points are SCORE-COLUMN-INDEPENDENT: a made FG is 2 (or 3 if the description says "3PT"), a made
  FT is 1, attributed to the scorer's team via the authoritative PLAYER1_TEAM_ID. This is immune to
  the non-monotonic / orientation-inconsistent SCORE column. Technical FTs go to a separate bucket.
- Missed and-1 free throws leave the ball live (the rebound decides), not a forced close.
- Offense is self-corrected when a shot's shooter != the current offense (an unlogged change of
  control from a PBP gap), so a stray made basket cannot be credited to the wrong team.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

# EVENTMSGTYPE codes
MADE_FG = 1
MISSED_FG = 2
FREE_THROW = 3
REBOUND = 4
TURNOVER = 5
FOUL = 6
VIOLATION = 7
SUB = 8
TIMEOUT = 9
JUMP_BALL = 10
EJECTION = 11
PERIOD_BEGIN = 12
PERIOD_END = 13
REPLAY = 18

_ADMIN = {SUB, TIMEOUT, REPLAY}          # possession-neutral; skipped in forward/backward scans
_FT_FINAL_ACTIONS = {10, 12, 15}         # last FT of a 1 / 2 / 3 trip
_FT_TECH_ACTION = 16                     # technical free throw (point -> tech bucket, no boundary)
_FOUL_OFFENSIVE_ACTION = 4
_FOUL_SHOOTING_ACTION = 2

_INT_COLS = [
    "event_id", "msg_type", "action_type", "period",
    "PLAYER1_ID", "PLAYER1_TEAM_ID", "PERSON1TYPE",
    "PLAYER2_ID", "PLAYER2_TEAM_ID", "PERSON2TYPE",
    "PLAYER3_ID", "PLAYER3_TEAM_ID", "PERSON3TYPE",
]
_VALID_END_REASONS = {
    "made_fg", "made_ft", "defensive_rebound", "turnover", "end_period", "jump_ball", "control_change",
}
_REB_RE = re.compile(r"Off:(\d+)\s+Def:(\d+)")  # parse running "(Off:O Def:D)" rebound totals

_POSS_SCHEMA = {
    "possession_id": pl.Int64, "game_id": pl.Utf8, "period": pl.Int64,
    "offense_team_id": pl.Int64, "defense_team_id": pl.Int64,
    "start_event_id": pl.Int64, "end_event_id": pl.Int64,
    "end_reason": pl.Utf8, "end_reason_detail": pl.Utf8, "points": pl.Int64, "n_events": pl.Int64,
    "start_game_clock": pl.Float64, "end_game_clock": pl.Float64,
    "counts_as_possession": pl.Boolean, "previous_possession_end_reason": pl.Utf8,
}


@dataclass
class _Poss:
    possession_id: int
    period: int
    offense_team_id: int
    start_event_id: int
    end_event_id: int | None = None
    end_reason: str | None = None
    end_reason_detail: str | None = None
    points: int = 0
    start_game_clock: float | None = None
    end_game_clock: float | None = None


def _parse_clock(s) -> float | None:
    """Seconds-remaining from a 'M:SS' PCTIMESTRING; None on anything malformed."""
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + float(sec)
    except (ValueError, AttributeError):
        return None


def _fg_points(desc_upper: str) -> int:
    return 3 if "3PT" in desc_upper else 2


def _scored_points(msg_type: int, desc_upper: str) -> int:
    """Points scored on an event from its own semantics (SCORE-column independent)."""
    if msg_type == MADE_FG:
        return _fg_points(desc_upper)
    if msg_type == FREE_THROW and "MISS" not in desc_upper:
        return 1
    return 0


def segment_possessions(events: pl.DataFrame, team_info: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (events_with_possession_id, possessions_table)."""
    home, vis = int(team_info["home_id"]), int(team_info["visitor_id"])
    teams = {home, vis}
    home_u = str(team_info.get("home_name", "")).upper()
    vis_u = str(team_info.get("visitor_name", "")).upper()

    def opp(t: int) -> int:
        return vis if t == home else home

    if events.height == 0:  # empty-input guard
        ev = events.with_columns(pl.lit(None, dtype=pl.Int64).alias("possession_id"))
        return ev, pl.DataFrame(schema=_POSS_SCHEMA)

    df = events.sort(["period", "event_id"]).with_columns(
        [pl.col(c).cast(pl.Int64, strict=False) for c in _INT_COLS if c in events.columns]
    )
    rows = df.to_dicts()
    n = len(rows)

    def resolve_team(r: dict, which: int) -> int | None:
        tid = r.get(f"PLAYER{which}_TEAM_ID")
        if tid in teams:
            return tid
        pid = r.get(f"PLAYER{which}_ID")  # team events stash the team id in PLAYER*_ID
        if pid in teams:
            return pid
        side = {2: home, 4: home, 3: vis, 5: vis}.get(r.get(f"PERSON{which}TYPE"))
        if side is not None:
            return side
        d = (r.get("description") or "").upper()
        if home_u and home_u in d:
            return home
        if vis_u and vis_u in d:
            return vis
        return None

    def offense(r: dict) -> int | None:
        mt = r["msg_type"]
        d = (r.get("description") or "").upper()
        # Steal turnovers: PLAYER1 is the committer (the offense losing the ball), PLAYER2 the
        # stealer (verified across games). So PLAYER1 is correct for the offense here.
        if mt in (MADE_FG, MISSED_FG, FREE_THROW, REBOUND, TURNOVER, VIOLATION):
            return resolve_team(r, 1)
        if mt == FOUL:
            if r.get("action_type") == _FOUL_OFFENSIVE_ACTION or "OFF.FOUL" in d:
                return resolve_team(r, 1)
            if "T.FOUL" in d or "FLAGRANT" in d or "DEF. 3 SEC" in d or "TECHNICAL" in d:
                return None
            return resolve_team(r, 2)  # personal/shooting foul -> the fouled (offensive) team
        if mt == JUMP_BALL:
            return resolve_team(r, 3)  # "Tip to <player>"
        return None

    def next_live(i: int) -> int | None:
        j = i + 1
        while j < n and rows[j]["msg_type"] in _ADMIN:
            j += 1
        return j if j < n else None

    def prev_live(i: int) -> int | None:
        j = i - 1
        while j >= 0 and rows[j]["msg_type"] in _ADMIN:
            j -= 1
        return j if j >= 0 else None

    def is_andone(i: int) -> int | None:
        """If made FG i is an and-1, return the event_id of its bonus '1 of 1' FT (made or missed)."""
        scorer = rows[i].get("PLAYER1_ID")
        j = next_live(i)
        if j is None:
            return None
        f = rows[j]
        if f["msg_type"] != FOUL:
            return None
        is_shooting = f.get("action_type") == _FOUL_SHOOTING_ACTION or "S.FOUL" in (f.get("description") or "").upper()
        if not is_shooting or f.get("PLAYER2_ID") != scorer:
            return None
        k = next_live(j)
        if k is None:
            return None
        ft = rows[k]
        if ft["msg_type"] == FREE_THROW and ft.get("action_type") == 10 and ft.get("PLAYER1_ID") == scorer:
            return ft["event_id"]
        return None

    reb_counts: dict[int, tuple[int, int]] = {}

    def classify_rebound(r: dict) -> str | None:
        """'off' / 'def' from the running '(Off:O Def:D)' counts; None for team rebounds."""
        m = _REB_RE.search(r.get("description") or "")
        pid = r.get("PLAYER1_ID")
        if not m or pid is None:
            return None
        o, d = int(m.group(1)), int(m.group(2))
        po, pd = reb_counts.get(pid, (0, 0))
        reb_counts[pid] = (o, d)
        if o > po:
            return "off"
        if d > pd:
            return "def"
        return None

    event_pid: list[int | None] = [None] * n
    poss: list[_Poss] = []
    cur: _Poss | None = None
    pending_offense: int | None = None
    pending_andone_ft: int | None = None
    home_default_opens = 0

    def resolve_open_offense(i: int, primary: int | None) -> int:
        """Best-effort offense for a possession opening at event i."""
        nonlocal home_default_opens
        o = offense(rows[i])  # the event's own offense is authoritative
        if o in teams:
            return o
        if primary in teams:
            return primary
        j = next_live(i)
        if j is not None:
            o = offense(rows[j])
            if o in teams:
                return o
        home_default_opens += 1
        return home

    def open_poss(off: int, idx: int) -> None:
        nonlocal cur, pending_offense
        cur = _Poss(
            possession_id=len(poss), period=rows[idx]["period"], offense_team_id=off,
            start_event_id=rows[idx]["event_id"], start_game_clock=_parse_clock(rows[idx].get("PCTIMESTRING")),
        )
        poss.append(cur)
        pending_offense = None

    def close_poss(end_idx: int, reason: str, detail: str | None = None, next_off: int | None = None) -> None:
        nonlocal cur, pending_offense
        if cur is None:
            return
        cur.end_event_id = rows[end_idx]["event_id"]
        cur.end_reason = reason
        cur.end_reason_detail = detail
        cur.end_game_clock = _parse_clock(rows[end_idx].get("PCTIMESTRING"))
        pending_offense = next_off
        cur = None

    for i, r in enumerate(rows):
        mt = r["msg_type"]
        dU = (r.get("description") or "").upper()

        if mt == PERIOD_END:
            if cur is not None:
                event_pid[i] = cur.possession_id
                close_poss(i, "end_period")
            elif poss:  # anchor a trailing marker to the just-closed possession (same period)
                event_pid[i] = poss[-1].possession_id
            pending_offense = None
            continue
        if mt == PERIOD_BEGIN:
            if cur is not None:  # force-close anything still open (missing/late PERIOD_END)
                p = prev_live(i)
                close_poss(p if p is not None else i, "end_period")
            pending_offense = None
            continue
        if mt in _ADMIN:
            if cur is not None:
                event_pid[i] = cur.possession_id
            continue

        # Technical FT: possession-neutral; never opens a possession.
        if mt == FREE_THROW and r.get("action_type") == _FT_TECH_ACTION:
            if cur is not None:
                event_pid[i] = cur.possession_id
            continue

        if mt == JUMP_BALL:
            tip = resolve_team(r, 3)
            if cur is not None and tip == cur.offense_team_id:
                event_pid[i] = cur.possession_id  # same team controls the tip -> continue
                continue
            if cur is not None:
                close_poss(i, "jump_ball")
            open_poss(resolve_open_offense(i, tip), i)
            event_pid[i] = cur.possession_id
            continue

        if mt == REBOUND:
            reb = resolve_team(r, 1)
            rtype = classify_rebound(r)
            if cur is not None:
                if rtype == "off" or (rtype is None and reb == cur.offense_team_id):
                    event_pid[i] = cur.possession_id  # offensive rebound -> continue
                    continue
                p = prev_live(i)
                close_poss(p if p is not None else i, "defensive_rebound")
                open_poss(resolve_open_offense(i, reb), i)
                event_pid[i] = cur.possession_id
                continue
            if rtype == "off":
                continue  # phantom offensive rebound off a made shot -> not a real possession
            open_poss(resolve_open_offense(i, reb), i)
            event_pid[i] = cur.possession_id
            continue

        # An offensive foul is double-encoded as msg-6 (action 4) + a paired msg-5 'Foul Turnover'.
        # Skip the paired turnover so we don't reopen/close a phantom possession.
        if mt == TURNOVER and "FOUL TURNOVER" in dU:
            p = prev_live(i)
            if p is not None and rows[p]["msg_type"] == FOUL and (
                rows[p].get("action_type") == _FOUL_OFFENSIVE_ACTION
                or "OFF.FOUL" in (rows[p].get("description") or "").upper()
            ):
                if cur is not None:
                    event_pid[i] = cur.possession_id
                continue

        if cur is None:
            open_poss(resolve_open_offense(i, pending_offense), i)

        # Offense self-correction: a shooter (FG or FT) is always the offense; a mismatch means an
        # unlogged change of control (a PBP gap / phantom rebound), so flip to the shooter's team.
        if mt in (MADE_FG, MISSED_FG, FREE_THROW):
            sh = resolve_team(r, 1)
            if sh in teams and sh != cur.offense_team_id:
                p = prev_live(i)
                close_poss(p if p is not None else i, "control_change")
                open_poss(sh, i)

        event_pid[i] = cur.possession_id

        if mt == MADE_FG:
            ft_id = is_andone(i)
            if ft_id is not None:
                pending_andone_ft = ft_id  # keep possession open through the bonus FT
            else:
                close_poss(i, "made_fg", next_off=opp(cur.offense_team_id))
        elif mt == FREE_THROW:
            if pending_andone_ft is not None and r["event_id"] == pending_andone_ft:
                pending_andone_ft = None
                if "MISS" not in dU:  # missed and-1 leaves the ball live for the rebound
                    close_poss(i, "made_fg", "and1", next_off=opp(cur.offense_team_id))
            elif r.get("action_type") in _FT_FINAL_ACTIONS and "MISS" not in dU:
                close_poss(i, "made_ft", next_off=opp(cur.offense_team_id))
        elif mt == TURNOVER:
            if "STEAL" in dU:
                stealer = resolve_team(r, 2)
                close_poss(i, "turnover", "steal", next_off=stealer if stealer in teams else opp(cur.offense_team_id))
            else:
                detail = "offensive_foul" if "FOUL" in dU else None
                close_poss(i, "turnover", detail, next_off=opp(cur.offense_team_id))
        elif mt == FOUL and (r.get("action_type") == _FOUL_OFFENSIVE_ACTION or "OFF.FOUL" in dU):
            close_poss(i, "turnover", "offensive_foul", next_off=opp(cur.offense_team_id))
        # other fouls / violations: non-closing in v1

    if cur is not None:  # safety: game didn't end on a period-end marker
        close_poss(n - 1, "end_period")

    # backfill unlabeled events to a possession IN THE SAME PERIOD (following preferred, else previous)
    for i in range(n):
        if event_pid[i] is not None:
            continue
        per = rows[i]["period"]
        chosen = None
        for j in range(i + 1, n):
            pj = event_pid[j]
            if pj is not None and poss[pj].period == per:
                chosen = pj
                break
        if chosen is None:
            for j in range(i - 1, -1, -1):
                pj = event_pid[j]
                if pj is not None and poss[pj].period == per:
                    chosen = pj
                    break
        event_pid[i] = chosen

    # points: SCORE-column independent (from each scoring event's own semantics), by scorer team
    tech_points = {home: 0, vis: 0}
    for i, r in enumerate(rows):
        pts = _scored_points(r["msg_type"], (r.get("description") or "").upper())
        if pts == 0:
            continue
        scorer = resolve_team(r, 1)
        if scorer not in teams:
            continue
        if r["msg_type"] == FREE_THROW and r.get("action_type") == _FT_TECH_ACTION:
            tech_points[scorer] += pts
        else:
            pid = event_pid[i]
            if pid is not None:
                poss[pid].points += pts

    n_events: dict[int, int] = {}
    for v in event_pid:
        if v is not None:
            n_events[v] = n_events.get(v, 0) + 1

    recs = []
    prev_reason = None
    for p in poss:
        counts = not (p.start_game_clock is not None and p.start_game_clock <= 2 and p.points == 0)
        recs.append({
            "possession_id": p.possession_id,
            "game_id": str(rows[0].get("GAME_ID", "")),
            "period": p.period,
            "offense_team_id": p.offense_team_id,
            "defense_team_id": opp(p.offense_team_id),
            "start_event_id": p.start_event_id,
            "end_event_id": p.end_event_id,
            "end_reason": p.end_reason,
            "end_reason_detail": p.end_reason_detail,
            "points": p.points,
            "n_events": n_events.get(p.possession_id, 0),
            "start_game_clock": p.start_game_clock,
            "end_game_clock": p.end_game_clock,
            "counts_as_possession": counts,
            "previous_possession_end_reason": prev_reason,
        })
        prev_reason = p.end_reason

    events_out = df.with_columns(pl.Series("possession_id", event_pid, dtype=pl.Int64))
    poss_df = pl.DataFrame(recs, schema=_POSS_SCHEMA) if recs else pl.DataFrame(schema=_POSS_SCHEMA)
    poss_df = poss_df.with_columns(
        pl.lit(tech_points[home]).alias("_tech_home"),
        pl.lit(tech_points[vis]).alias("_tech_visitor"),
        pl.lit(home_default_opens).alias("_home_default_opens"),
    )
    return events_out, poss_df


def map_moments_to_possessions(moments: pl.DataFrame, events_with_pid: pl.DataFrame) -> pl.DataFrame:
    """v1 (event-key) mapping: label each moment by its event's possession_id.

    A clock-window clamp to de-duplicate moments that bleed across adjacent events is deferred
    (see segmentation spec); driving the label off PBP means missing coordinates never orphan a
    possession.
    """
    return moments.join(
        events_with_pid.select("event_id", "possession_id"), on="event_id", how="left"
    )


def _final_score(events_with_pid: pl.DataFrame) -> tuple[int, int] | None:
    """Return (a, b) from the last non-null SCORE 'a - b' (orientation not assumed)."""
    for sc in reversed(events_with_pid["SCORE"].drop_nulls().to_list()):
        if sc and "-" in str(sc):
            try:
                a, b = (int(x) for x in str(sc).split("-"))
                return a, b
            except ValueError:
                continue
    return None


def validate_segmentation(events_with_pid: pl.DataFrame, poss: pl.DataFrame, team_info: dict) -> dict:
    """Validation checks (bools + a summary). Points are reconciled with a SCORE-independent,
    description-based recount compared to the final SCORE as an orientation-free multiset."""
    home, vis = int(team_info["home_id"]), int(team_info["visitor_id"])

    # independent, description-based team totals (the robust gate)
    team_pts = {home: 0, vis: 0}
    for r in events_with_pid.iter_rows(named=True):
        pts = _scored_points(r["msg_type"], (r.get("description") or "").upper())
        if pts and r.get("PLAYER1_TEAM_ID") in (home, vis):
            team_pts[r["PLAYER1_TEAM_ID"]] += pts
    final = _final_score(events_with_pid)

    th, tv = poss["_tech_home"][0], poss["_tech_visitor"][0]
    seg_home = (poss.filter(pl.col("offense_team_id") == home)["points"].sum() or 0) + th
    seg_vis = (poss.filter(pl.col("offense_team_id") == vis)["points"].sum() or 0) + tv

    ids = poss["possession_id"].to_list()
    span = (
        events_with_pid.group_by("possession_id").agg(pl.col("period").n_unique().alias("nper"))
        .filter(pl.col("nper") > 1).height
    )
    btb = (
        poss.sort("possession_id")
        .with_columns(po=pl.col("offense_team_id").shift(1), pp=pl.col("period").shift(1))
        .filter((pl.col("offense_team_id") == pl.col("po")) & (pl.col("period") == pl.col("pp"))).height
    )
    checks = {
        "coverage_all_labeled": events_with_pid["possession_id"].null_count() == 0,
        "ids_contiguous": ids == list(range(len(ids))),
        "offense_in_two_teams": poss.filter(~pl.col("offense_team_id").is_in([home, vis])).height == 0,
        "end_reasons_valid": poss.filter(~pl.col("end_reason").is_in(list(_VALID_END_REASONS))).height == 0,
        "no_possession_spans_period": span == 0,
        "points_reconcile": final is not None and sorted(team_pts.values()) == sorted(final),
        "segmenter_points_consistent": seg_home == team_pts[home] and seg_vis == team_pts[vis],
    }
    checks["summary"] = {
        "n_possessions": poss.height,
        "home_possessions": poss.filter(pl.col("offense_team_id") == home).height,
        "visitor_possessions": poss.filter(pl.col("offense_team_id") == vis).height,
        "team_points": {"home": team_pts[home], "visitor": team_pts[vis]},
        "segmenter_points": {"home": int(seg_home), "visitor": int(seg_vis)},
        "final_score": final,
        "back_to_back_same_offense": btb,
        "home_default_opens": int(poss["_home_default_opens"][0]),
    }
    return checks
