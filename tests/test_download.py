"""Offline unit tests for the download helpers (no network)."""

from plot.io.download import Game, parse_game_filename, select_games


def test_parse_canonical():
    assert parse_game_filename("01.02.2016.ORL.at.CLE.7z") == ("2016-01-02", "ORL", "CLE")


def test_parse_dir_prefixed():
    # A handful of source archives are concatenated with the directory prefix.
    nm = "2016.NBA.Raw.SportVU.Game.Logs12.05.2015.BOS.at.SAS.7z"
    assert parse_game_filename(nm) == ("2015-12-05", "BOS", "SAS")


def test_parse_rejects_nongame():
    assert parse_game_filename("README.md") is None
    assert parse_game_filename("2016.NBA.Raw.SportVU.Game.Logs") is None


def _g(name, date, away, home):
    return Game(name=name, date=date, away=away, home=home, download_url="", size=0)


def test_select_substring_and_tier():
    games = [
        _g("01.02.2016.ORL.at.CLE.7z", "2016-01-02", "ORL", "CLE"),
        _g("01.04.2016.TOR.at.CLE.7z", "2016-01-04", "TOR", "CLE"),
        _g("01.01.2016.CHA.at.TOR.7z", "2016-01-01", "CHA", "TOR"),
    ]
    cle = select_games(games, game="CLE", tier="T3")
    assert {g.home for g in cle} == {"CLE"} and len(cle) == 2
    one = select_games(games, tier="T0")
    assert len(one) == 1


def test_select_sample_is_reproducible():
    games = [_g(f"01.0{i}.2016.AAA.at.BBB.7z", f"2016-01-0{i}", "AAA", "BBB") for i in range(1, 6)]
    assert select_games(games, n=3, seed=9) == select_games(games, n=3, seed=9)
