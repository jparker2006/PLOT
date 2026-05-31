"""Gate G6, step 2 — outcome validity: does "points left on the table" cost *real* points?

Everything through G6 step 1 is internal: regret = best_available(model) − chosen(model), and every
gate compares the model to itself or its own construction. This module runs the one test that is
*not* model-vs-model — does the signal predict what actually happened on the floor.

The cleanest decision the metric makes is the **open look**: a ball-handler with nobody within
``OPEN_FT`` either takes the shot or declines it (passes). We observe REALIZED possession points for
*both* choices — takers from the shot they took, decliners from the continuation — so the
counterfactual "what taking would have yielded" comes from real shooters at matched situations, not
from the model. The test:

* **taker calibration** — do open looks of model-value ``S`` actually realize ≈ ``S`` when taken?
  (validates ``S`` as a real benchmark, on these specific looks, not just the G2 average.)
* **cost of declining** — at matched ``S``, ``E[R | took] − E[R | declined]`` is the realized points
  lost by passing the look up. The validity signature is that this cost is **positive and grows with
  ``S``** (declining a *valuable* look costs more; declining a bad one is correct and cheap).
* **adjusted estimate** — OLS of realized points on ``declined``, ``S``, their interaction, and
  observable controls (possession EPV at the decision, location, openness) + team fixed effects, with
  a **game-cluster bootstrap** so inference respects that decisions within a game are not independent.
  The headline is the realized points lost by declining a *high-value* look, net of observables.

What this can and cannot do: conditioning on observables (crucially the model's own EPV at the
decision, which captures a lot of possession context) bounds selection on *observables*; it cannot
rule out that decliners see something the tracking doesn't (selection on unobservables — the G5b
caveat). Pure over its assembled input frame; the per-game assembly lives in ``scripts/build_g6.py``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from plot.models.regret.plot_metric import _half_assignment


def _qbin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin (0..n_bins-1); ties collapse so empty bins just don't appear downstream."""
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(x, qs)


def taker_calibration(df: pl.DataFrame, *, value_col: str = "S", outcome_col: str = "R",
                      declined_col: str = "declined", n_bins: int = 8) -> list[dict]:
    """Per S-bin (takers only): mean model value vs mean REALIZED points. If xPoints is a fair
    benchmark on these open looks, ``mean_realized ≈ mean_value`` along the bins."""
    takers = df.filter(pl.col(declined_col) == 0)
    if takers.height < n_bins * 5:
        return []
    s = takers[value_col].to_numpy()
    r = takers[outcome_col].to_numpy().astype(float)
    b = _qbin(s, n_bins)
    out = []
    for bi in sorted(set(b.tolist())):
        m = b == bi
        out.append({"s_bin": int(bi), "n": int(m.sum()),
                    "mean_value": round(float(s[m].mean()), 4),
                    "mean_realized": round(float(r[m].mean()), 4)})
    return out


def cost_of_declining_by_value(df: pl.DataFrame, *, value_col: str = "S", outcome_col: str = "R",
                               declined_col: str = "declined", n_bins: int = 6) -> list[dict]:
    """Per S-bin: realized points for takers vs decliners and the raw (unadjusted) cost of declining
    = ``mean_R_took − mean_R_declined``. Validity signature: cost > 0 and rising across S-bins."""
    s_all = df[value_col].to_numpy()
    b_all = _qbin(s_all, n_bins)
    df = df.with_columns(pl.Series("_sb", b_all))
    rows = []
    for bi in sorted(set(b_all.tolist())):
        sub = df.filter(pl.col("_sb") == bi)
        took = sub.filter(pl.col(declined_col) == 0)
        dec = sub.filter(pl.col(declined_col) == 1)
        if took.height < 5 or dec.height < 5:
            continue
        rt = float(took[outcome_col].mean())
        rd = float(dec[outcome_col].mean())
        rows.append({
            "s_bin": int(bi), "mean_value": round(float(sub[value_col].mean()), 4),
            "n_took": took.height, "n_declined": dec.height,
            "mean_realized_took": round(rt, 4), "mean_realized_declined": round(rd, 4),
            "raw_cost_of_declining": round(rt - rd, 4),
        })
    return rows


def _design(df: pl.DataFrame, controls: list[str], declined_col: str, value_col: str,
            team_col: str | None) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """Build [intercept, declined, S_centered, declined*S_centered, controls, team dummies], y=R."""
    n = df.height
    d = df[declined_col].to_numpy().astype(float)
    s = df[value_col].to_numpy().astype(float)
    s_mean = float(s.mean())
    sc = s - s_mean
    cols = [np.ones(n), d, sc, d * sc]
    names = ["intercept", "declined", "S_c", "declined:S_c"]
    for c in controls:
        cols.append(df[c].to_numpy().astype(float))
        names.append(c)
    if team_col is not None:
        levels = sorted(set(df[team_col].to_list()))[1:]  # drop one as reference
        for lv in levels:
            cols.append((df[team_col].to_numpy() == lv).astype(float))
            names.append(f"team_{lv}")
    X = np.column_stack(cols)
    y = df["R"].to_numpy().astype(float)
    return X, y, s_mean, names


def adjusted_decline_cost(
    df: pl.DataFrame, *, controls: list[str], declined_col: str = "declined", value_col: str = "S",
    team_col: str | None = "offense_team_id", game_col: str = "game_id",
    s_high_quantile: float = 0.90, n_boot: int = 500, seed: int = 0,
) -> dict:
    """Adjusted realized cost of declining an open look, with a game-cluster bootstrap.

    Fits ``R ~ declined + S_c + declined:S_c + controls + team FE`` and reports the realized points
    LOST by declining (= −βdeclined at mean S; and at a high-value look ``s_high_quantile``, adding the
    interaction). Inference by resampling whole GAMES with replacement so within-game/possession
    dependence is respected. ``ci_low_high > 0`` ⇒ declining valuable open looks costs real points,
    net of observables + team.
    """
    X, y, s_mean, names = _design(df, controls, declined_col, value_col, team_col)
    s = df[value_col].to_numpy().astype(float)
    s_high = float(np.quantile(s, s_high_quantile))
    delta = s_high - s_mean
    i_dec, i_int = names.index("declined"), names.index("declined:S_c")

    def fit(Xs, ys):
        beta, _, _, _ = np.linalg.lstsq(Xs, ys, rcond=None)
        return beta

    beta = fit(X, y)
    cost_mean = -float(beta[i_dec])                       # points lost by declining at mean S
    cost_high = -float(beta[i_dec] + beta[i_int] * delta)  # ...at a high-value look

    games = df[game_col].to_list()
    uniq = sorted(set(games))
    idx_by_game = {g: np.where(np.array(games) == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    boot_mean, boot_high, boot_slope = [], [], []
    ng = len(uniq)
    for _ in range(n_boot):
        pick = rng.integers(0, ng, ng)
        rows = np.concatenate([idx_by_game[uniq[j]] for j in pick])
        try:
            b = fit(X[rows], y[rows])
        except np.linalg.LinAlgError:
            continue
        boot_mean.append(-b[i_dec])
        boot_high.append(-(b[i_dec] + b[i_int] * delta))
        boot_slope.append(b[i_int])  # d(R)/d(S) extra for decliners; negative ⇒ cost grows with S

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]

    def p_two_sided(a):  # share on the far side of 0, doubled
        a = np.asarray(a)
        return round(float(2 * min((a <= 0).mean(), (a >= 0).mean())), 4)

    return {
        "n_decisions": df.height, "n_games": ng,
        "n_took": int((df[declined_col] == 0).sum()), "n_declined": int((df[declined_col] == 1).sum()),
        "s_mean": round(s_mean, 4), "s_high_quantile": s_high_quantile, "s_high": round(s_high, 4),
        "points_lost_by_declining_at_meanS": round(cost_mean, 4),
        "ci_meanS": ci(boot_mean), "p_meanS": p_two_sided(boot_mean),
        "points_lost_by_declining_at_highS": round(cost_high, 4),
        "ci_highS": ci(boot_high), "p_highS": p_two_sided(boot_high),
        "interaction_declinedxS": round(float(beta[i_int]), 4),
        "ci_interaction": ci(boot_slope), "p_interaction": p_two_sided(boot_slope),
        "controls": controls, "team_fe": team_col is not None, "n_boot": n_boot,
    }


def outcome_validity_gate(adjusted: dict, cost_by_value: list[dict]) -> dict:
    """Decision-level G6 step 2 verdict.

    PASS = declining a HIGH-VALUE open look costs realized points with a bootstrap CI excluding 0
    (the signal predicts a real cost, net of observables + team), AND the raw cost rises with shot
    value (the dose-response the model predicts). ``cost_grows_with_value`` compares the top vs bottom
    populated S-bin so it is robust to a single noisy middle bin.
    """
    ci_low = adjusted.get("ci_highS", [None, None])[0]
    cost_high = adjusted.get("points_lost_by_declining_at_highS")
    rising = None
    valued = [r for r in cost_by_value if r["n_took"] >= 5 and r["n_declined"] >= 5]
    if len(valued) >= 2:
        rising = valued[-1]["raw_cost_of_declining"] > valued[0]["raw_cost_of_declining"]
    gate = {
        "high_value_decline_costs_points": bool(cost_high is not None and cost_high > 0
                                                 and ci_low is not None and ci_low > 0),
        "cost_grows_with_value": bool(rising) if rising is not None else False,
    }
    return {
        "gate": gate,
        "pass": all(gate.values()),
        "headline_points_lost_high_value": cost_high,
        "headline_ci": adjusted.get("ci_highS"),
        "verdict": (
            "declining valuable open looks costs real points — the regret signal is outcome-valid at the decision level"
            if all(gate.values())
            else "no decision-level outcome cost detected (null / underpowered / selection) — see report"
        ),
    }


# ============================== step 2c — player-level cross-fit ==============================
# The decision-level keystone (step 2) is non-circular by construction: the counterfactual comes from
# real shooters. The *player-level* read ("does PLOT identify who leaves points?") is the prize, but the
# in-sample version (correlate a player's model metric with their own S−R shortfall) leaks twice: it
# reuses the model's S on both sides (mechanical), and measures the metric and the shortfall on the SAME
# games. This module removes both leaks with a cross-fit: model metric on one half of games, realized
# points-left on the OTHER half, benchmarked against real takers (empirical R, not the model's S).

_CROSSFIT_METRICS = {"plot_per100": "raw PLOT", "within_role_per100": "within-role residual"}


def _matched_taker_cost(
    looks_half: pl.DataFrame, *, value_col: str, outcome_col: str, declined_col: str,
    player_col: str, n_bins: int, min_takers_per_bin: int, min_declined: int,
) -> pl.DataFrame:
    """Per-player realized points-left on declined open looks, for ONE half of games.

    The counterfactual yardstick is **empirical, not the model**: bin looks by model value ``S`` (a
    pure matching device), and within each bin take the mean REALIZED possession points of real TAKERS
    as the benchmark for what a look of that quality yields when taken. A decliner's realized cost is
    ``benchmark(its S-bin) − R``; averaged per player it is the realized points they left by passing,
    measured against real shooters — sharing no construction with the model regret metric (only the
    coarse bin edges come from ``S``). Bins with < ``min_takers_per_bin`` takers form no benchmark and
    their decliners are dropped; players need ≥ ``min_declined`` scorable declines to enter.
    """
    takers = looks_half.filter(pl.col(declined_col) == 0)
    decliners = looks_half.filter(pl.col(declined_col) == 1)
    if takers.height < n_bins * min_takers_per_bin or decliners.height == 0:
        return pl.DataFrame()
    s_t = takers[value_col].to_numpy().astype(float)
    r_t = takers[outcome_col].to_numpy().astype(float)
    edges = np.quantile(s_t, np.linspace(0, 1, n_bins + 1)[1:-1])
    tb = np.digitize(s_t, edges)
    bench = {int(bi): float(r_t[tb == bi].mean())
             for bi in set(tb.tolist()) if int((tb == bi).sum()) >= min_takers_per_bin}
    if not bench:
        return pl.DataFrame()
    db = np.digitize(decliners[value_col].to_numpy().astype(float), edges)
    benchvals = np.array([bench.get(int(b), np.nan) for b in db])
    cost = benchvals - decliners[outcome_col].to_numpy().astype(float)
    return (
        decliners.with_columns(pl.Series("_cost", cost))
        .filter(pl.col("_cost").is_not_nan())
        .group_by(player_col)
        .agg(pl.len().alias("n_declined_cost"), pl.col("_cost").mean().alias("realized_points_left"))
        .filter(pl.col("n_declined_cost") >= min_declined)
    )


def _model_metric_half(
    reg_half: pl.DataFrame, *, player_col: str, min_decisions: int, per: int = 100,
) -> pl.DataFrame:
    """Per-player model metric on ONE half of games: raw PLOT and within-role residual per ``per``."""
    return (
        reg_half.group_by(player_col)
        .agg(pl.len().alias("n_decisions_model"),
             (pl.col("regret_clipped").mean() * per).alias("plot_per100"),
             (pl.col("regret_clipped_resid").mean() * per).alias("within_role_per100"))
        .filter(pl.col("n_decisions_model") >= min_decisions)
    )


def _crossfit_corr(j: pl.DataFrame, metric_col: str) -> dict:
    from scipy import stats  # noqa: PLC0415
    a = j[metric_col].to_numpy().astype(float)
    b = j["realized_points_left"].to_numpy().astype(float)
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {"n_players": int(j.height),
            "pearson_r": round(float(pr), 4), "pearson_p": round(float(pp), 6),
            "spearman_r": round(float(sr), 4), "spearman_p": round(float(sp), 6)}


def _crossfit_verdict(res: dict) -> dict:
    """Cross-fit verdict on the WITHIN-ROLE residual (the role-adjusted decision signal): validated if
    its pooled cross-fit correlation is positive and significant AND both single directions agree in
    sign. The raw-PLOT pooled result is carried for context."""
    pooled, da, db = (res.get(k, {}) for k in
                      ("pooled_crossfit", "direction_modelA_costB", "direction_modelB_costA"))

    def r_of(block, metric):
        b = block.get(metric) if isinstance(block, dict) else None
        return b["pearson_r"] if isinstance(b, dict) and "pearson_r" in b else None

    by_metric = {}
    for metric in _CROSSFIT_METRICS:
        p = pooled.get(metric) if isinstance(pooled, dict) else None
        ra, rb = r_of(da, metric), r_of(db, metric)
        agree = ra is not None and rb is not None and (ra > 0) == (rb > 0)
        validated = bool(isinstance(p, dict) and p.get("pearson_r", 0.0) > 0
                         and p.get("pearson_p", 1.0) < 0.05 and agree)
        by_metric[metric] = {
            "pooled_pearson_r": p.get("pearson_r") if isinstance(p, dict) else None,
            "pooled_pearson_p": p.get("pearson_p") if isinstance(p, dict) else None,
            "directions_agree_sign": agree, "validated": validated,
        }
    primary = by_metric.get("within_role_per100", {})
    return {
        "by_metric": by_metric,
        "within_role_validated": primary.get("validated", False),
        "verdict": (
            "player-level attribution is non-circular: the within-role metric predicts realized "
            "points-left out of sample"
            if primary.get("validated")
            else "no clean player-level attribution out of sample — the decision-level effect does not "
                 "cross-fit to individual players (honest null)"
        ),
    }


def player_cross_fit(
    looks: pl.DataFrame, residualized_regret: pl.DataFrame, *,
    value_col: str = "S", outcome_col: str = "R", declined_col: str = "declined",
    game_col: str = "game_id", player_col: str = "player_id",
    n_bins: int = 8, min_takers_per_bin: int = 25,
    min_declined_half: int = 15, min_decisions_half: int = 20, min_players: int = 10,
) -> dict:
    """Player-level cross-fit de-circularization of the outcome-validity signal (G6 step 2c).

    The in-sample player-level check (``_player_level`` in the build script) correlates a player's
    model metric with their realized shortfall ``S − R`` on the SAME games, reusing ``S`` on both
    sides — exploratory only. This is the rigorous version, breaking both leaks:

    * **out-of-sample across games** — the season's games split odd/even (``_half_assignment``). The
      MODEL metric (raw PLOT and within-role residual, per 100 decisions) is computed on one half; the
      REALIZED points-left on the *other* half. No game contributes to both sides of a pair.
    * **empirical, not model, benchmark** — realized points-left uses the mean realized outcome of
      real TAKERS in the same look-value bin (``_matched_taker_cost``), not the model's ``S``. The two
      axes share no construction.

    Both directions are run (model-half-0/cost-half-1 and the swap) and pooled — every pair is
    cross-fit, each player contributing up to twice (mild within-player dependence; the single-direction
    blocks have independent player sets and are the conservative read). A POSITIVE, significant
    correlation ⇒ the players the model flags as leaving points genuinely realize fewer points than
    matched shooters, out of sample — non-circular player-level outcome validity. A null ⇒ the effect
    is real at the decision level but does not cleanly attribute to individual players.

    Pure over its two input frames (``looks`` carries ``S``/``R``/``declined``; ``residualized_regret``
    carries ``regret_clipped`` + ``regret_clipped_resid``); the heavy assembly lives in the build script.
    """
    games = sorted(set(looks[game_col].to_list()) | set(residualized_regret[game_col].to_list()))
    halves = _half_assignment(games)
    h0 = [g for g, h in halves.items() if h == 0]
    h1 = [g for g, h in halves.items() if h == 1]

    def direction(model_games, cost_games):
        model = _model_metric_half(residualized_regret.filter(pl.col(game_col).is_in(model_games)),
                                   player_col=player_col, min_decisions=min_decisions_half)
        cost = _matched_taker_cost(
            looks.filter(pl.col(game_col).is_in(cost_games)),
            value_col=value_col, outcome_col=outcome_col, declined_col=declined_col,
            player_col=player_col, n_bins=n_bins, min_takers_per_bin=min_takers_per_bin,
            min_declined=min_declined_half)
        if model.height == 0 or cost.height == 0:
            return pl.DataFrame()
        return model.join(cost, on=player_col, how="inner")

    j_ab, j_ba = direction(h0, h1), direction(h1, h0)
    if j_ab.height and j_ba.height:
        pooled = pl.concat([j_ab, j_ba])
    else:
        pooled = j_ab if j_ab.height else j_ba

    def block(j):
        if j is None or j.height < min_players:
            return {"n_players": int(0 if j is None else j.height), "note": "too few players"}
        return {m: _crossfit_corr(j, m) for m in _CROSSFIT_METRICS}

    res = {
        "n_games": len(games), "n_games_half0": len(h0), "n_games_half1": len(h1),
        "design": {
            "model_metric_half": "per-player regret_clipped (raw PLOT) and regret_clipped_resid "
                                 "(within-role residual), per 100 decisions, on one half of games",
            "cost_half": "per-player mean (matched-S-bin TAKER realized R − decliner R) on declined "
                         "open looks on the OTHER half — empirical benchmark, not model S",
            "n_bins": n_bins, "min_takers_per_bin": min_takers_per_bin,
            "min_declined_half": min_declined_half, "min_decisions_half": min_decisions_half,
        },
        "direction_modelA_costB": block(j_ab),
        "direction_modelB_costA": block(j_ba),
        "pooled_crossfit": block(pooled),
    }
    res["gate"] = _crossfit_verdict(res)
    return res
