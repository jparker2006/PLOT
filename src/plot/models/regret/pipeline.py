"""Build the per-decision regret frame (the v1.5 *credible set*) from raw games + the EPV trace.

Factored out of the v1.5 build loop so the gates that consume per-decision regret (G6) build it the
*same* way the metric does, rather than drifting a second copy of the loading loop.
Returns one row per scored decision with ``decision_kind`` (``pass_up`` | ``shot_selection``) and the
shared regret schema, plus the trained xPoints model and the shot table (for finishing-skill checks).

The credible set is the two decisions v2 taught us to trust: an open handler who *passed up* their
own open shot (v1), and a handler who *shot over* a wide-open, reachable, completion-discounted
teammate (shot-selection). Local 42-game corpus; method transfers to scale.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from plot.models.counterfactual.regret import decision_states, regret_from_states
from plot.models.counterfactual.regret_full import offensive_options, shot_selection_regret
from plot.models.counterfactual.xpoints import predict_xpoints, train_xpoints
from plot.models.eval_bar.seq_dataset import build_game_canonical
from plot.possessions.actions import extract_actions
from plot.possessions.shots import OPEN_FT, extract_shots

# the shared per-decision schema both halves emit (post_epv holds the chosen value in either slot)
V15_COLS = [
    "game_id", "possession_id", "action_idx", "player_id", "decision_kind",
    "sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist",
    "best_available", "post_epv", "regret_signed", "regret_clipped",
]
_SHOT_ACTIONS = ["shot_make", "shot_miss"]
_OPEN_LOOK_ACTIONS = ["pass", "shot_make", "shot_miss"]
MAX_SHOT_FT = 30.0  # beyond this a "declined shot" is a backcourt/bring-up, not a real open look


def _events(raw_dir: str, g: str) -> pl.DataFrame:
    return (
        pl.read_csv(Path(raw_dir, "2015-16_pbp.csv"), infer_schema_length=20000)
        .filter(pl.col("GAME_ID") == int(g))
        .rename({"EVENTNUM": "event_id", "EVENTMSGTYPE": "msg_type",
                 "EVENTMSGACTIONTYPE": "action_type", "PERIOD": "period"})
        .with_columns(pl.coalesce("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION").alias("description"))
    )


def load_game_intermediates(games, *, raw_dir: str) -> dict:
    """Load each game **once** (the heavy step: ``build_game_canonical`` per game) and return the
    per-game intermediates every G6 analysis is a pure transform over: ``opts`` (offensive options),
    ``actions``, ``states`` (decision states), ``poss`` (possessions w/ realized points), plus the
    pooled ``shots`` and the trained population ``xmodel``, and ``clean`` (games that loaded)."""
    shots_parts = []
    opts_by_game, actions_by_game, states_by_game, poss_by_game = {}, {}, {}, {}
    for g in games:
        cdf, poss, _ = build_game_canonical(g, raw_dir=raw_dir)
        if cdf is None:
            continue
        actions = extract_actions(cdf, poss, g)
        shots_parts.append(extract_shots(cdf, _events(raw_dir, g), g))
        opts_by_game[g] = offensive_options(cdf, actions)
        actions_by_game[g] = actions
        states_by_game[g] = decision_states(cdf, actions).with_columns(pl.lit(g).alias("game_id"))
        poss_by_game[g] = poss.with_columns(pl.lit(g).alias("game_id"))
        del cdf
    shots = pl.concat(shots_parts)
    return {
        "opts": opts_by_game, "actions": actions_by_game, "states": states_by_game,
        "poss": poss_by_game, "shots": shots, "xmodel": train_xpoints(shots),
        "clean": list(opts_by_game.keys()),
    }


def regret_from_intermediates(inter: dict, *, trace: pl.DataFrame, completion: float = 0.80) -> pl.DataFrame:
    """v1.5 per-decision regret (pass-up + shot-selection) from loaded intermediates (schema ``V15_COLS``)."""
    passup_parts, shotsel_parts = [], []
    for g in inter["clean"]:
        pu = regret_from_states(inter["states"][g], inter["actions"][g], inter["xmodel"], trace, g)
        if pu.height:
            passup_parts.append(pu.with_columns(pl.lit("pass_up").alias("decision_kind")))
        ss = shot_selection_regret(inter["opts"][g], inter["xmodel"], g, completion=completion)
        if ss.height:
            shotsel_parts.append(ss.with_columns(pl.lit("shot_selection").alias("decision_kind")))
    passup = pl.concat(passup_parts)
    shotsel = pl.concat(shotsel_parts)
    return pl.concat([passup.select(V15_COLS), shotsel.select(V15_COLS)])


def open_looks_from_intermediates(inter: dict, *, trace: pl.DataFrame) -> pl.DataFrame:
    """The open-look decision frame for G6 step 2 (outcome validity).

    One row per OPEN ball-handler (nearest defender ≥ ``OPEN_FT``, frontcourt ``≤ MAX_SHOT_FT``) who
    either TOOK the look (terminal shot ⇒ ``declined=0``) or DECLINED it (pass ⇒ ``declined=1``).
    Carries the own open-shot model value ``S`` (xPoints at actual contest), the REALIZED possession
    points ``R``, the model's possession value at the decision ``epv_at_decision``, plus location /
    openness / team for the matched outcome test. Realized points come at the possession's terminal
    action, so ``R`` (possession total) is the realized value from the decision onward."""
    parts = []
    for g in inter["clean"]:
        opts = inter["opts"][g]
        if opts.height == 0:
            continue
        _, xp = predict_xpoints(inter["xmodel"], opts)
        opts = opts.with_columns(pl.Series("xpoints", xp))
        handler = opts.filter(
            pl.col("is_handler") & (pl.col("nearest_def_dist") >= OPEN_FT)
            & (pl.col("dist_to_rim") <= MAX_SHOT_FT)
            & pl.col("action_type").is_in(_OPEN_LOOK_ACTIONS)
        )
        if handler.height == 0:
            continue
        h = handler.select(
            pl.lit(g).alias("game_id"), "possession_id", "action_idx", "wall_clock_ms",
            "player_id", "offense_team_id",
            pl.col("action_type").is_in(_SHOT_ACTIONS).not_().cast(pl.Int64).alias("declined"),
            pl.col("xpoints").alias("S"),
            "dist_to_rim", pl.col("three_pt").cast(pl.Int64).alias("three_pt"), "nearest_def_dist",
        )
        poss = inter["poss"][g].select(
            "possession_id", pl.col("points").cast(pl.Float64).alias("R"), "period", "end_reason")
        h = h.join(poss, on="possession_id", how="left")
        tr = trace.filter(pl.col("game_id") == g).select(
            "possession_id", "wall_clock_ms", pl.col("epv").alias("epv_at_decision"))
        h = h.join(tr, on=["possession_id", "wall_clock_ms"], how="left")
        parts.append(h)
    return pl.concat(parts)


def build_v15_regret(games, *, raw_dir: str, trace: pl.DataFrame, completion: float = 0.80):
    """Build the concatenated v1.5 per-decision regret over ``games`` present locally.

    Returns ``(regret, shots, xmodel, clean_games)``: the per-decision frame (schema ``V15_COLS``,
    with ``decision_kind``), the trained shot table, the population xPoints model, and the list of
    games that actually loaded. Thin wrapper over ``load_game_intermediates`` +
    ``regret_from_intermediates`` (back-compat for step 1 callers)."""
    inter = load_game_intermediates(games, raw_dir=raw_dir)
    regret = regret_from_intermediates(inter, trace=trace, completion=completion)
    return regret, inter["shots"], inter["xmodel"], inter["clean"]
