"""Court geometry: canonicalization, distance/angle, and region classification.

All modeling features live in a CANONICAL frame in which the offense always attacks the
RIGHT rim at ``(RIM_X, RIM_Y) = (88.75, 25)``. A possession whose offense attacks the left
rim is mapped in by a 180-degree ROTATION about center court (``x -> 94 - x`` AND
``y -> 50 - y``) — never an x-mirror, so left/right court chirality is preserved (a left-corner
shot maps to the right corner, not its mirror).

Pure functions only (numpy/scalar + polars-expression builders); no IO, no model. Shared by
the eval-bar features and every later spatial-feature stage. Geometry matches ``configs`` and
the standard NBA court verified against the tracking data in Stage 1.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Court (feet). Full court is 94 x 50; rims on the y midline.
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
RIM_X = 88.75            # canonical attacked rim (right)
RIM_Y = 25.0
RIM_LEFT_X = 5.25
HALF_COURT_X = 47.0
RIM_RADIUS = 0.75

# Shot geometry.
THREE_POINT_RADIUS = 23.75     # above-the-break arc
CORNER_THREE_DIST = 22.0       # straight corner line distance from rim
CORNER_Y_MARGIN = 3.0          # corner strip: within 3 ft of a sideline (y<=3 or y>=47)
RESTRICTED_AREA_RADIUS = 4.0

# The painted lane (right end): 16 ft wide (y in [17, 33]), 19 ft deep from the baseline.
LANE_HALF_WIDTH = 8.0          # y in [25-8, 25+8] = [17, 33]
LANE_DEPTH = 19.0              # x in [94-19, 94] = [75, 94]
LANE_X_MIN = COURT_LENGTH - LANE_DEPTH   # 75.0
LANE_Y_MIN = RIM_Y - LANE_HALF_WIDTH     # 17.0
LANE_Y_MAX = RIM_Y + LANE_HALF_WIDTH     # 33.0

# Canonical region labels (coarse, interpretable; raw xy carry the fine signal).
REGIONS = (
    "backcourt",
    "restricted_area",
    "paint",            # in the lane, outside the restricted area
    "mid_range",
    "corner_3",
    "above_break_3",
)
ATTACK_RIGHT = "R"
ATTACK_LEFT = "L"


def rotate_to_canonical(x, y, attacked_rim: str):
    """Map (x, y) into the canonical frame (offense attacks the right rim).

    ``attacked_rim`` is ``"R"`` (already canonical, identity) or ``"L"`` (rotate 180 deg).
    Works on python scalars and on numpy arrays.
    """
    if attacked_rim == ATTACK_RIGHT:
        return x, y
    if attacked_rim == ATTACK_LEFT:
        return COURT_LENGTH - x, COURT_WIDTH - y
    raise ValueError(f"attacked_rim must be 'L' or 'R', got {attacked_rim!r}")


def dist_to_rim(x, y):
    """Euclidean distance from a canonical (x, y) to the attacked rim (88.75, 25)."""
    return np.hypot(np.asarray(x, dtype=float) - RIM_X, np.asarray(y, dtype=float) - RIM_Y)


def angle_to_rim(x, y):
    """Bearing (radians) from the rim to a canonical (x, y).

    ``atan2(y - RIM_Y, RIM_X - x)``: ~0 straight out in front of the rim (toward half court),
    positive toward the +y sideline. Encode as sin/cos in features to avoid wraparound.
    """
    return np.arctan2(np.asarray(y, dtype=float) - RIM_Y, RIM_X - np.asarray(x, dtype=float))


def court_region(x: float, y: float) -> str:
    """Coarse canonical-frame zone of a single (x, y) point."""
    if x < HALF_COURT_X:
        return "backcourt"
    d = float(np.hypot(x - RIM_X, y - RIM_Y))
    if d <= RESTRICTED_AREA_RADIUS:
        return "restricted_area"
    if LANE_X_MIN <= x <= COURT_LENGTH and LANE_Y_MIN <= y <= LANE_Y_MAX:
        return "paint"
    in_corner = y <= CORNER_Y_MARGIN or y >= (COURT_WIDTH - CORNER_Y_MARGIN)
    if in_corner and d >= CORNER_THREE_DIST:
        return "corner_3"
    if d >= THREE_POINT_RADIUS:
        return "above_break_3"
    return "mid_range"


def region_expr(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """Polars expression mirroring :func:`court_region` for vectorized featurization."""
    d = ((x - RIM_X) ** 2 + (y - RIM_Y) ** 2).sqrt()
    in_corner = (y <= CORNER_Y_MARGIN) | (y >= COURT_WIDTH - CORNER_Y_MARGIN)
    in_lane = (x >= LANE_X_MIN) & (x <= COURT_LENGTH) & (y >= LANE_Y_MIN) & (y <= LANE_Y_MAX)
    return (
        pl.when(x < HALF_COURT_X).then(pl.lit("backcourt"))
        .when(d <= RESTRICTED_AREA_RADIUS).then(pl.lit("restricted_area"))
        .when(in_lane).then(pl.lit("paint"))
        .when(in_corner & (d >= CORNER_THREE_DIST)).then(pl.lit("corner_3"))
        .when(d >= THREE_POINT_RADIUS).then(pl.lit("above_break_3"))
        .otherwise(pl.lit("mid_range"))
    )


def in_paint_expr(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """Boolean: canonical point inside the painted lane (right end)."""
    return (x >= LANE_X_MIN) & (x <= COURT_LENGTH) & (y >= LANE_Y_MIN) & (y <= LANE_Y_MAX)


def in_restricted_area_expr(x: pl.Expr, y: pl.Expr) -> pl.Expr:
    """Boolean: canonical point inside the 4 ft restricted-area arc."""
    return ((x - RIM_X) ** 2 + (y - RIM_Y) ** 2).sqrt() <= RESTRICTED_AREA_RADIUS
