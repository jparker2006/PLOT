"""Offline tests for possession segmentation on hand-built synthetic games."""

import polars as pl

from plot.possessions.segment import segment_possessions, validate_segmentation

TI = {
    "home_id": 1, "home_name": "Aaa", "home_abbr": "AAA",
    "visitor_id": 2, "visitor_name": "Bbb", "visitor_abbr": "BBB",
}


def _ev(event_id, msg_type, action_type=0, p1=0, p1t=0, pt1=0, p2=0, p2t=0, pt2=0,
        p3=0, p3t=0, pt3=0, score=None, home=None, vis=None):
    return {
        "GAME_ID": 1, "event_id": event_id, "msg_type": msg_type, "action_type": action_type,
        "period": 1, "PCTIMESTRING": "6:00",
        "PLAYER1_ID": p1, "PLAYER1_TEAM_ID": p1t, "PERSON1TYPE": pt1,
        "PLAYER2_ID": p2, "PLAYER2_TEAM_ID": p2t, "PERSON2TYPE": pt2,
        "PLAYER3_ID": p3, "PLAYER3_TEAM_ID": p3t, "PERSON3TYPE": pt3,
        "SCORE": score, "HOMEDESCRIPTION": home, "VISITORDESCRIPTION": vis,
        "description": home or vis,
    }


def _game() -> pl.DataFrame:
    rows = [
        _ev(1, 10, p3=10, p3t=1, home="Jump Ball: Tip to P10"),              # -> home
        _ev(2, 1, p1=10, p1t=1, score="0 - 2", home="P10 Layup (2 PTS)"),    # home FG (close)
        _ev(3, 2, p1=20, p1t=2, vis="MISS P20 Jump Shot"),                   # visitor miss
        _ev(4, 4, p1=10, p1t=1, home="P10 REBOUND (Off:0 Def:1)"),           # home DEF reb (new poss)
        _ev(5, 2, p1=11, p1t=1, home="MISS P11 Jump Shot"),                  # home miss
        _ev(6, 4, p1=11, p1t=1, home="P11 REBOUND (Off:1 Def:0)"),           # home OFF reb (continue)
        _ev(7, 1, p1=11, p1t=1, score="0 - 4", home="P11 Layup (2 PTS)"),    # home FG (close)
        _ev(8, 5, p1=20, p1t=2, p2=10, p2t=1, vis="P20 Bad Pass", home="P10 STEAL"),  # TO + steal
        _ev(9, 1, p1=12, p1t=1, score="0 - 6", home="P12 Layup (2 PTS)"),    # home FG (and-1 start)
        _ev(10, 6, action_type=2, p1=21, p1t=2, p2=12, p2t=1, vis="P21 S.FOUL"),      # shooting foul
        _ev(11, 3, action_type=10, p1=12, p1t=1, score="0 - 7",
            home="P12 Free Throw 1 of 1 (3 PTS)"),                           # and-1 FT (close as made_fg)
        _ev(12, 1, p1=20, p1t=2, score="2 - 7", vis="P20 Layup (2 PTS)"),    # visitor FG (close)
        _ev(13, 13, home="End of Period"),
    ]
    return pl.DataFrame(rows)


def test_segmentation_core():
    events_pid, poss = segment_possessions(_game(), TI)
    assert poss.height == 6
    assert poss["offense_team_id"].to_list() == [1, 2, 1, 2, 1, 2]
    p2 = poss.filter(pl.col("possession_id") == 2).row(0, named=True)
    assert p2["end_reason"] == "made_fg" and p2["n_events"] == 4 and p2["end_event_id"] == 7
    p3 = poss.filter(pl.col("possession_id") == 3).row(0, named=True)
    assert p3["end_reason"] == "turnover" and p3["end_reason_detail"] == "steal"
    p4 = poss.filter(pl.col("possession_id") == 4).row(0, named=True)
    assert p4["points"] == 3 and p4["end_reason"] == "made_fg" and p4["end_event_id"] == 11


def test_offensive_rebound_does_not_split_possession():
    events_pid, _ = segment_possessions(_game(), TI)
    pids = events_pid.filter(pl.col("event_id").is_in([4, 5, 6, 7]))["possession_id"].unique().to_list()
    assert len(pids) == 1


def test_validation_all_checks_pass_and_points_reconcile():
    events_pid, poss = segment_possessions(_game(), TI)
    checks = validate_segmentation(events_pid, poss, TI)
    for k, v in checks.items():
        if k != "summary":
            assert v, f"check failed: {k}"
    assert checks["summary"]["points_home"] == 7
    assert checks["summary"]["points_visitor"] == 2


def test_phantom_offensive_rebound_after_made_fg_is_skipped():
    rows = [
        _ev(1, 10, p3=10, p3t=1, home="Tip to P10"),
        _ev(2, 1, p1=10, p1t=1, score="0 - 2", home="P10 Layup (2 PTS)"),     # made FG closes poss0
        _ev(3, 4, p1=10, p1t=1, home="P10 REBOUND (Off:1 Def:0)"),            # phantom OFF reb -> skip
        _ev(4, 1, p1=20, p1t=2, score="2 - 2", vis="P20 Layup (2 PTS)"),
        _ev(5, 13, home="End"),
    ]
    _, poss = segment_possessions(pl.DataFrame(rows), TI)
    assert poss.height == 2 and poss["offense_team_id"].to_list() == [1, 2]


def test_jump_ball_retained_by_same_team_continues():
    rows = [
        _ev(1, 10, p3=10, p3t=1, home="Tip to P10"),
        _ev(2, 2, p1=10, p1t=1, home="MISS P10 Shot"),
        _ev(3, 10, p3=10, p3t=1, home="Jump Ball: Tip to P10"),  # same team retains -> continue
        _ev(4, 1, p1=10, p1t=1, score="0 - 2", home="P10 Layup (2 PTS)"),
        _ev(5, 13, home="End"),
    ]
    events_pid, poss = segment_possessions(pl.DataFrame(rows), TI)
    assert poss.height == 1
    assert events_pid.filter(pl.col("event_id").is_in([1, 2, 3, 4]))["possession_id"].n_unique() == 1
