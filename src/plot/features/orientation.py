"""Infer which rim each team attacks per half, and canonicalize coordinates.

Coordinates in the raw tracking data are NOT pre-oriented: a team attacks one rim in the first
half and the other in the second, and the JSON never says which. Every distance/region feature
is meaningless until this is resolved, and a silent wrong flip yields a confidently
ANTI-calibrated eval bar — so this module ABORTS (quarantines the game) rather than guess.

Signal (the only one empirically robust across all 14 local games — verified 14/14): the mean
x of the offense's BALL position restricted to the DEEP FRONTCOURT (``|x - 47| > deep_margin``).
A team only ventures deep down a court when attacking that rim (never deep into its own
defensive end on offense), so deep-frontcourt frames separate the two ends cleanly (~52-62 vs
~33-44 across the local games), and the tens of thousands of frames per cell make the mean
rock-stable. Per-event made-FG votes and shallow ball/player means were tried and are NOT
reliable (boundary-bleed reattributes makes to the wrong rim; transition frames pull means to
midcourt). half = 1 for periods {1,2}, else 2 (3, 4, and all OT keep the second-half baskets).

Because the two teams attack opposite rims each half and each team flips at halftime, the full
per-(team, half) map is determined by a single cell: we ANCHOR on the most decisive cell and
PROPAGATE, then quarantine if any other CONFIDENT cell contradicts the propagation. Requires
asof-labeled moments (offense_team_id + counts_as_possession per row; see ``plot.features.clean``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from plot.features.court import ATTACK_LEFT, ATTACK_RIGHT, COURT_LENGTH, COURT_WIDTH, HALF_COURT_X


@dataclass
class Orientation:
    """Result of orientation inference for one game."""

    rim_map: dict[tuple[int, int], str]  # (team_id, half) -> "L"/"R"
    quarantined: bool
    report: dict = field(default_factory=dict)

    def attacked_rim(self, team_id: int, period: int) -> str | None:
        return self.rim_map.get((int(team_id), period_half(period)))


def period_half(period: int) -> int:
    """1 for periods 1-2, else 2 (3, 4, and all OT keep the second-half baskets)."""
    return 1 if int(period) <= 2 else 2


def deep_frontcourt_means(
    labeled_moments: pl.DataFrame, *, deep_margin: float = 20.0
) -> pl.DataFrame:
    """Per-(team_id, half) mean ball x over deep-frontcourt offensive frames + frame counts.

    ``labeled_moments`` must have columns: entity, x, quarter, offense_team_id, counts_as_possession.
    """
    ball = labeled_moments.filter(
        (pl.col("entity") == "ball")
        & pl.col("x").is_not_null()
        & pl.col("offense_team_id").is_not_null()
        & pl.col("counts_as_possession")
        & ((pl.col("x") - HALF_COURT_X).abs() > deep_margin)
    ).with_columns(pl.when(pl.col("quarter") <= 2).then(1).otherwise(2).alias("half"))
    return (
        ball.group_by(pl.col("offense_team_id").cast(pl.Int64).alias("team_id"), "half")
        .agg(pl.col("x").mean().alias("mean_x"), pl.len().alias("n"))
        .filter(pl.col("half").is_in([1, 2]))
    )


def _flip(rim: str) -> str:
    return ATTACK_LEFT if rim == ATTACK_RIGHT else ATTACK_RIGHT


def rim_map_from_means(
    cells: pl.DataFrame, teams: tuple[int, int], *, min_frames: int = 200, strong_ft: float = 5.0
) -> tuple[dict[tuple[int, int], str], dict]:
    """Anchor on the most decisive (team, half) cell and propagate the 2x2 rim map.

    A cell's rim is L if mean_x < 47 else R; its strength is |mean_x - 47| ft. Quarantines if no
    cell clears (min_frames, strong_ft), or if any other cell that IS strong+populous contradicts
    the propagated map. Pure over ``cells`` (columns: team_id, half, mean_x, n) for unit testing.
    """
    a, b = int(teams[0]), int(teams[1])
    parsed: dict[tuple[int, int], dict] = {}
    for r in cells.iter_rows(named=True):
        rim = ATTACK_LEFT if r["mean_x"] < HALF_COURT_X else ATTACK_RIGHT
        parsed[(int(r["team_id"]), int(r["half"]))] = {
            "mean_x": round(float(r["mean_x"]), 2), "n": int(r["n"]),
            "rim": rim, "strength": round(abs(float(r["mean_x"]) - HALF_COURT_X), 2),
        }
    report = {"cells": {f"{t}|{h}": v for (t, h), v in sorted(parsed.items())}}

    confident = {k: v for k, v in parsed.items() if v["n"] >= min_frames and v["strength"] >= strong_ft}
    if not confident:
        report["quarantined"], report["reason"] = True, "no_confident_orientation_cell"
        return {}, report
    # A lone confident cell fully determines the 2x2 map with NO corroboration: a single
    # wrong-but-confident cell would silently invert the whole game. Require >=2 confident cells
    # (the conflict check below then cross-checks them). EMPIRICALLY VALIDATED on the 74-game corpus:
    # of 12 single-confident-cell games, 11 were wrong flips (caught by the backcourt QC when this
    # guard was relaxed) — so the guard is correct, not over-conservative; keep it.
    if len(confident) < 2:
        report["quarantined"], report["reason"] = True, "single_confident_cell_no_corroboration"
        return {}, report

    (at, ah), av = max(confident.items(), key=lambda kv: kv[1]["strength"])
    anchor_rim = av["rim"]
    report["anchor"] = {"team": at, "half": ah, "rim": anchor_rim, "mean_x": av["mean_x"], "n": av["n"]}

    rim_map: dict[tuple[int, int], str] = {}
    for team in (a, b):
        for half in (1, 2):
            rim = anchor_rim
            if team != at:
                rim = _flip(rim)
            if half != ah:
                rim = _flip(rim)
            rim_map[(team, half)] = rim

    conflicts = [f"{t}|{h}:obs={v['rim']}!=prop={rim_map[(t, h)]}(mx={v['mean_x']})"
                 for (t, h), v in confident.items() if rim_map.get((t, h)) != v["rim"]]
    if conflicts:
        report["quarantined"], report["reason"], report["conflicts"] = True, "confident_cell_conflict", conflicts
        return {}, report
    report["quarantined"] = False
    return rim_map, report


def infer_orientation(
    labeled_moments: pl.DataFrame,
    team_info: dict,
    *,
    deep_margin: float = 20.0,
    min_frames: int = 200,
    strong_ft: float = 5.0,
) -> Orientation:
    """Infer the per-(team, half) attacked-rim map for one game, with quarantine self-checks."""
    teams = (int(team_info["home_id"]), int(team_info["visitor_id"]))
    cells = deep_frontcourt_means(labeled_moments, deep_margin=deep_margin)
    rim_map, report = rim_map_from_means(cells, teams, min_frames=min_frames, strong_ft=strong_ft)
    return Orientation(rim_map=rim_map, quarantined=report.get("quarantined", False), report=report)


def attacked_rim_expr(team_col: pl.Expr, period_col: pl.Expr, rim_map: dict[tuple[int, int], str]) -> pl.Expr:
    """Polars expression mapping (team_id, period) -> attacked rim 'L'/'R' via ``rim_map``."""
    half = pl.when(period_col <= 2).then(1).otherwise(2)
    expr = pl.lit(None, dtype=pl.Utf8)
    for (team, h), rim in rim_map.items():
        expr = pl.when((team_col == team) & (half == h)).then(pl.lit(rim)).otherwise(expr)
    return expr


def canonicalize_xy(x: pl.Expr, y: pl.Expr, attacked_rim: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """180-degree rotation to the canonical frame where the offense attacks the right rim.

    Left-attacking rows rotate (x->94-x, y->50-y); right-attacking rows pass through. A true
    rotation (not an x-mirror), so a left-corner shot maps to the right corner.
    """
    x_c = pl.when(attacked_rim == ATTACK_LEFT).then(COURT_LENGTH - x).otherwise(x)
    y_c = pl.when(attacked_rim == ATTACK_LEFT).then(COURT_WIDTH - y).otherwise(y)
    return x_c, y_c
