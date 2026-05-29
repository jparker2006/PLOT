"""Offline tests for the game loader using a tiny synthetic game + PBP."""

import polars as pl

from plot.io.loaders import events_table, moments_dataframe, players_table

# A minimal two-event game in the raw SportVU shape.
# moment = [quarter, wall_ms, game_clock, shot_clock, None, positions]
# positions[0] = ball [-1,-1,x,y,z]; positions[1:] = players [team,pid,x,y,z]
_GAME = {
    "gameid": "0021500499",
    "gamedate": "2016-01-02",
    "events": [
        {
            "eventId": "2",
            "visitor": {"name": "Magic", "teamid": 200, "abbreviation": "ORL",
                        "players": [{"firstname": "A", "lastname": "B", "playerid": 11,
                                     "jersey": "1", "position": "G"}]},
            "home": {"name": "Cavaliers", "teamid": 100, "abbreviation": "CLE",
                     "players": [{"firstname": "Le", "lastname": "James", "playerid": 2544,
                                  "jersey": "23", "position": "F"}]},
            "moments": [
                [1, 1000, 700.0, 18.0, None,
                 [[-1, -1, 47.0, 25.0, 6.0], [100, 2544, 40.0, 20.0, 0.0]]],
                [1, 1040, 699.96, 17.96, None,
                 [[-1, -1, 46.0, 24.0, 5.0], [100, 2544, 41.0, 21.0, 0.0]]],
            ],
        },
        {
            "eventId": "3",
            "visitor": {"name": "Magic", "teamid": 200, "abbreviation": "ORL",
                        "players": [{"firstname": "C", "lastname": "D", "playerid": 12,
                                     "jersey": "2", "position": "F"}]},
            "home": {"name": "Cavaliers", "teamid": 100, "abbreviation": "CLE",
                     "players": [{"firstname": "Le", "lastname": "James", "playerid": 2544,
                                  "jersey": "23", "position": "F"}]},
            "moments": [
                [1, 1080, 699.0, 17.0, None,
                 [[-1, -1, 30.0, 25.0, 4.0], [200, 12, 31.0, 26.0, 0.0]]],
            ],
        },
    ],
}


def test_players_table_unions_rosters():
    pt = players_table(_GAME)
    assert set(pt["player_id"]) == {11, 2544, 12}
    lebron = pt.filter(pl.col("player_id") == 2544)
    assert lebron["name"][0] == "Le James" and lebron["team_abbr"][0] == "CLE"


def test_moments_long_shape_and_ball():
    m = moments_dataframe(_GAME)
    # 2 moments * 2 entities + 1 moment * 2 entities = 6 rows
    assert m.height == 6
    ball = m.filter(pl.col("entity") == "ball")
    assert ball.height == 3 and set(ball["player_id"]) == {-1}
    assert m["event_id"].unique().sort().to_list() == [2, 3]


def test_moments_event_filter():
    m = moments_dataframe(_GAME, event_ids=[3])
    assert m["event_id"].unique().to_list() == [3] and m.height == 2


def test_events_table_filters_and_coalesces(tmp_path):
    csv = tmp_path / "pbp.csv"
    csv.write_text(
        "GAME_ID,EVENTNUM,EVENTMSGTYPE,EVENTMSGACTIONTYPE,PERIOD,PCTIMESTRING,"
        "HOMEDESCRIPTION,VISITORDESCRIPTION,NEUTRALDESCRIPTION,PERSON1TYPE,PLAYER1_ID,"
        "PLAYER1_NAME,PLAYER1_TEAM_ID,PERSON2TYPE,PLAYER2_ID,PLAYER2_TEAM_ID,PERSON3TYPE,"
        "PLAYER3_ID,PLAYER3_TEAM_ID,SCORE,SCOREMARGIN\n"
        "21500499,2,1,101,1,11:42,James Jump Shot,,,4,2544,LeBron James,100,5,0,0,0,0,0,2 - 0,2\n"
        "21500498,5,2,1,1,10:00,Other game,,,4,1,X,100,5,0,0,0,0,0,,\n"
    )
    ev = events_table(csv, "0021500499")
    assert ev.height == 1
    row = ev.row(0, named=True)
    assert row["event_id"] == 2 and row["msg_type"] == 1
    assert row["description"] == "James Jump Shot"
