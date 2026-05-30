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
