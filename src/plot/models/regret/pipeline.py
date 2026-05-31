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

from plot.features.court import HALF_COURT_X
from plot.models.counterfactual.regret import decision_states, regret_from_states
from plot.models.counterfactual.regret_full import offensive_options, shot_selection_regret
from plot.models.counterfactual.xpoints import predict_xpoints, train_xpoints
from plot.models.eval_bar.seq_dataset import build_game_canonical
from plot.possessions.actions import extract_actions
from plot.possessions.shots import OPEN_FT, WIDE_OPEN_FT, extract_shots

# the shared per-decision schema both halves emit (post_epv holds the chosen value in either slot)
V15_COLS = [
    "game_id", "possession_id", "action_idx", "player_id", "decision_kind",
    "sx", "sy", "dist_to_rim", "three_pt", "nearest_def_dist",
    "best_available", "post_epv", "regret_signed", "regret_clipped",
]
_SHOT_ACTIONS = ["shot_make", "shot_miss"]
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


_OPEN_LOOK_COLS = [
    "game_id", "possession_id", "decision_wall_ms", "player_id", "declined", "S",
    "dist_to_rim", "three_pt", "nearest_def_dist", "fg_points",
]


def open_looks_from_intermediates(inter: dict, *, trace: pl.DataFrame) -> pl.DataFrame:
    """The open-look decision frame for G6 step 2 (outcome validity), shooter-attributed.

    One row per OPEN offensive player (nearest defender ≥ ``OPEN_FT``, frontcourt ``≤ MAX_SHOT_FT``)
    who either TOOK the look or DECLINED it:

    * **takers** (``declined=0``) are the REAL PBP shots (``extract_shots`` → the ``PLAYER1`` shooter
      at release), so the decision is correctly attributed to the player who shot — not the coarse
      tracking ball-handler at possession end. ``fg_points`` is the shot's OWN realized points (no
      offensive-rebound inflation), used to calibrate the benchmark ``S`` cleanly.
    * **decliners** (``declined=1``) are open ball-handlers who passed (a pass has no PBP shot event
      to anchor, so tracking is the only source).

    Carries: own open-shot model value ``S`` (xPoints at actual contest, same model both sides), the
    REALIZED possession points ``R`` (the decision's realized value, O-rebs accruing to both groups),
    the model EPV at the decision, location/openness, period, and ``poss_elapsed_s`` — seconds into
    the possession, a shot-clock-pressure proxy that also controls for takers acting later than
    decliners (a forced late pass is not a free decision)."""
    shots, xmodel = inter["shots"], inter["xmodel"]
    parts = []
    for g in inter["clean"]:
        rows = []
        opts = inter["opts"][g]
        if opts.height:  # ---- decliners: open handler who passed ----
            _, xp = predict_xpoints(xmodel, opts)
            h = opts.with_columns(pl.Series("xpoints", xp)).filter(
                pl.col("is_handler") & (pl.col("nearest_def_dist") >= OPEN_FT)
                & (pl.col("dist_to_rim") <= MAX_SHOT_FT) & (pl.col("action_type") == "pass")
            )
            if h.height:
                rows.append(h.select(
                    pl.lit(g).alias("game_id"), "possession_id",
                    pl.col("wall_clock_ms").alias("decision_wall_ms"), "player_id",
                    pl.lit(1, dtype=pl.Int64).alias("declined"), pl.col("xpoints").alias("S"),
                    "dist_to_rim", pl.col("three_pt").cast(pl.Int64).alias("three_pt"), "nearest_def_dist",
                    pl.lit(None, dtype=pl.Float64).alias("fg_points"),
                ))
        gshots = shots.filter((pl.col("game_id") == g) & pl.col("is_open")
                              & (pl.col("dist_to_rim") <= MAX_SHOT_FT))
        if gshots.height:  # ---- takers: real open PBP shots ----
            _, sxp = predict_xpoints(xmodel, gshots)
            rows.append(gshots.with_columns(pl.Series("S", sxp)).select(
                pl.lit(g).alias("game_id"), "possession_id",
                pl.col("shot_wall_ms").alias("decision_wall_ms"), pl.col("shooter_id").alias("player_id"),
                pl.lit(0, dtype=pl.Int64).alias("declined"), "S",
                "dist_to_rim", pl.col("three_pt").cast(pl.Int64).alias("three_pt"), "nearest_def_dist",
                pl.col("fg_points").cast(pl.Float64).alias("fg_points"),
            ))
        if not rows:
            continue
        base = pl.concat([r.select(_OPEN_LOOK_COLS) for r in rows])
        poss = inter["poss"][g].select(
            "possession_id", pl.col("points").cast(pl.Float64).alias("R"), "period", "end_reason",
            "offense_team_id")
        poss_start = inter["actions"][g].group_by("possession_id").agg(
            pl.col("start_wall_ms").min().alias("poss_start_wall"))
        tr = trace.filter(pl.col("game_id") == g).select(
            "possession_id", pl.col("wall_clock_ms").alias("decision_wall_ms"),
            pl.col("epv").alias("epv_at_decision"))
        parts.append(
            base.join(poss, on="possession_id", how="left")
            .join(poss_start, on="possession_id", how="left")
            .join(tr, on=["possession_id", "decision_wall_ms"], how="left")
            .with_columns(((pl.col("decision_wall_ms") - pl.col("poss_start_wall")) / 1000.0)
                          .clip(0.0, 24.0).alias("poss_elapsed_s"))
        )
    return pl.concat(parts)


_KICK_COLS = [
    "game_id", "possession_id", "decision_wall_ms", "player_id", "declined", "S", "own_shot_xp",
    "dist_to_rim", "three_pt", "nearest_def_dist", "fg_points",
]


def open_kick_decisions_from_intermediates(
    inter: dict, *, trace: pl.DataFrame, completion: float = 0.80,
    wide_open_ft: float = WIDE_OPEN_FT, max_pass_ft: float = 28.0,
) -> pl.DataFrame:
    """The shot-over-open-man decision frame for G6 step 2b (outcome validity, decision type #2).

    The mirror of the open-look test, with roles swapped. At a ball-handler decision where a
    WIDE-OPEN (≥ ``wide_open_ft``), frontcourt, reachable (≤ ``max_pass_ft``) teammate exists, the
    best available option is the **kick** to that teammate (their shot, ``completion``-discounted):

    * **took it** (``declined=0``) — the handler PASSED and the recipient (the next ball-handler) is
      one of those wide-open teammates;
    * **declined it** (``declined=1``) — the handler SHOT over the open man.

    ``S`` is the kick's value (``completion`` × best open-teammate xPoints); ``own_shot_xp`` is the
    handler's own look (a control, since the choice is shoot-own vs kick); ``R`` is realized possession
    points. Feeds the same ``outcome_validity`` functions as the open-look test (``declined`` ≡ shot
    over the open man), so the cost of shooting over a *good* kick is estimated the same way."""
    xmodel = inter["xmodel"]
    parts = []
    for g in inter["clean"]:
        opts = inter["opts"][g]
        if opts.height == 0:
            continue
        _, xp = predict_xpoints(xmodel, opts)
        opts = opts.with_columns(pl.Series("xpoints", xp))
        handler = opts.filter(pl.col("is_handler")).select(
            "possession_id", "action_idx", "action_type", "player_id",
            pl.col("wall_clock_ms").alias("decision_wall_ms"), pl.col("xpoints").alias("own_shot_xp"),
            "dist_to_rim", pl.col("three_pt").cast(pl.Int64).alias("three_pt"), "nearest_def_dist",
            pl.col("x_canon").alias("hx"), pl.col("y_canon").alias("hy"))
        if handler.height == 0:
            continue
        tm = (
            opts.filter((~pl.col("is_handler")) & (pl.col("nearest_def_dist") >= wide_open_ft)
                        & (pl.col("x_canon") >= HALF_COURT_X))
            .join(handler.select("possession_id", "action_idx", "hx", "hy"),
                  on=["possession_id", "action_idx"], how="inner")
            .with_columns((((pl.col("x_canon") - pl.col("hx")) ** 2
                            + (pl.col("y_canon") - pl.col("hy")) ** 2).sqrt()).alias("_pd"))
            .filter(pl.col("_pd") <= max_pass_ft)
        )
        if tm.height == 0:
            continue
        best_tm = tm.group_by("possession_id", "action_idx").agg(
            pl.col("xpoints").max().alias("best_tm_xp"), pl.col("player_id").alias("open_ids"))
        # recipient of a pass = the actor of the NEXT action in the possession
        acts = inter["actions"][g].select("possession_id", "action_idx", "actor_player_id")
        recip = acts.with_columns((pl.col("action_idx") - 1).alias("_k")).select(
            "possession_id", pl.col("_k").alias("action_idx"), pl.col("actor_player_id").alias("recipient"))
        h = (
            handler.join(best_tm, on=["possession_id", "action_idx"], how="inner")
            .join(recip, on=["possession_id", "action_idx"], how="left")
            .with_columns(
                pl.col("action_type").is_in(_SHOT_ACTIONS).alias("_is_shot"),
                ((pl.col("action_type") == "pass")
                 & pl.col("recipient").is_in(pl.col("open_ids"))).alias("_kicked"),
                (completion * pl.col("best_tm_xp")).alias("S"),
            )
            .filter(pl.col("_is_shot") | pl.col("_kicked"))  # shot-over-open-man, or kicked to him
            .with_columns(pl.when(pl.col("_is_shot")).then(1).otherwise(0).cast(pl.Int64).alias("declined"),
                          pl.lit(g).alias("game_id"),
                          pl.lit(None, dtype=pl.Float64).alias("fg_points"))
        )
        if h.height == 0:
            continue
        base = h.select(_KICK_COLS)
        poss = inter["poss"][g].select(
            "possession_id", pl.col("points").cast(pl.Float64).alias("R"), "period", "end_reason",
            "offense_team_id")
        poss_start = inter["actions"][g].group_by("possession_id").agg(
            pl.col("start_wall_ms").min().alias("poss_start_wall"))
        tr = trace.filter(pl.col("game_id") == g).select(
            "possession_id", pl.col("wall_clock_ms").alias("decision_wall_ms"),
            pl.col("epv").alias("epv_at_decision"))
        parts.append(
            base.join(poss, on="possession_id", how="left")
            .join(poss_start, on="possession_id", how="left")
            .join(tr, on=["possession_id", "decision_wall_ms"], how="left")
            .with_columns(((pl.col("decision_wall_ms") - pl.col("poss_start_wall")) / 1000.0)
                          .clip(0.0, 24.0).alias("poss_elapsed_s"))
        )
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
