#!/usr/bin/env python
"""Gate G6, step 1 — role-of-touch vs within-role decision-quality decomposition.

    uv run --extra seq python scripts/build_g6.py     # local 42-game OOF trace

Builds the v1.5 per-decision regret (pass-up + shot-selection), residualizes it on touch-context
out-of-fold by game, and answers the founding question with a number:

  * variance split — of the between-player variance in mean regret, how much the role component
    (touch-context expectation) reconstructs vs the within-role residual;
  * reliability — split-half (Spearman-Brown) of RAW regret vs the WITHIN-ROLE residual: does any
    reliable signal survive removing role?
  * the step-1 gate + a side-by-side of the top players by raw PLOT vs by within-role residual (does
    the leaderboard reshuffle once role is removed?).

Writes reports/G6/g6_step1.json (+ figure) and data/processed/plot_within_role.parquet (gitignored).
42-game local corpus; method/conclusions transfer to scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from plot.eval.box_stats import normalize_pbp, per_player_box_stats  # noqa: E402
from plot.eval.decomposition import (  # noqa: E402
    CONTEXT_FEATURES,
    context_capacity_sensitivity,
    g6_step1_gate,
    player_components,
    residualize_on_context,
    variance_decomposition,
)
from plot.eval.outcome_validity import (  # noqa: E402
    adjusted_decline_cost,
    cost_of_declining_by_value,
    outcome_validity_gate,
    taker_calibration,
)
from plot.models.regret.pipeline import (  # noqa: E402
    load_game_intermediates,
    open_looks_from_intermediates,
    regret_from_intermediates,
)
from plot.models.regret.plot_metric import split_half_stability  # noqa: E402

OPEN_LOOK_CONTROLS = ["epv_at_decision", "dist_to_rim", "three_pt", "nearest_def_dist",
                      "poss_elapsed_s", "period"]

MIN_DECISIONS = 30
MIN_PER_HALF = 15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/G6")
    ap.add_argument("--oof-trace", default="data/processed/oof_epv_trace.parquet")
    ap.add_argument("--pbp", default="data/raw/2015-16_pbp.csv")
    ap.add_argument("--completion", type=float, default=0.80)
    ap.add_argument("--per-player-out", default="data/processed/plot_within_role.parquet")
    args = ap.parse_args()

    trace = pl.read_parquet(args.oof_trace)
    if "epv_cal" in trace.columns:
        trace = trace.with_columns(pl.col("epv_cal").alias("epv"))
    trace_games = set(trace["game_id"].unique().to_list())
    games = sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json") if p.stem in trace_games)

    inter = load_game_intermediates(games, raw_dir=args.raw_dir)
    clean = inter["clean"]
    regret = regret_from_intermediates(inter, trace=trace, completion=args.completion)
    print(f"clean games: {len(clean)} | decisions: {regret.height} "
          f"({regret['player_id'].n_unique()} players)")

    # ---- residualize on touch-context (out-of-fold by game), for clipped + signed ----
    res = residualize_on_context(regret, value_col="regret_clipped")
    res = residualize_on_context(res, value_col="regret_signed")

    decomp_c = variance_decomposition(res, value_col="regret_clipped", min_decisions=MIN_DECISIONS)
    decomp_s = variance_decomposition(res, value_col="regret_signed", min_decisions=MIN_DECISIONS)

    # ---- reliability: raw vs within-role residual (clipped headline) ----
    rel_raw = split_half_stability(res, value_col="regret_clipped", min_decisions_per_half=MIN_PER_HALF)
    rel_resid = split_half_stability(res, value_col="regret_clipped_resid", min_decisions_per_half=MIN_PER_HALF)
    gate = g6_step1_gate(decomp_c, rel_raw, rel_resid)

    # ---- leakage robustness: is the within-role residual stable as the role model gets stronger? ----
    sensitivity = context_capacity_sensitivity(regret, value_col="regret_clipped",
                                               min_decisions=MIN_DECISIONS, min_decisions_per_half=MIN_PER_HALF)

    # ---- per-player components + names; does removing role reshuffle the leaderboard? ----
    comp = player_components(res, value_col="regret_clipped", per=100, min_decisions=MIN_DECISIONS)
    box = per_player_box_stats(normalize_pbp(pl.read_csv(
        args.pbp, infer_schema_length=20000,
        columns=["GAME_ID", "EVENTMSGTYPE", "PLAYER1_ID", "PLAYER1_NAME",
                 "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"])), min_fga_fta=50)
    named = comp.join(box.select("player_id", "player_name"), on="player_id", how="left").with_columns(
        pl.col("player_name").fill_null(pl.col("player_id").cast(pl.Utf8))
    )
    Path(args.per_player_out).parent.mkdir(parents=True, exist_ok=True)
    named.write_parquet(args.per_player_out)

    top_raw = named.sort("plot_per100", descending=True).select(
        "player_name", "n_decisions", "plot_per100", "role_per100", "within_role_per100").head(15)
    top_resid = named.sort("within_role_per100", descending=True).select(
        "player_name", "n_decisions", "plot_per100", "role_per100", "within_role_per100").head(15)

    report = {
        "corpus": {"clean_games": len(clean), "decisions": regret.height,
                   "n_players_in_metric": int(comp.height)},
        "context_features": CONTEXT_FEATURES + ["decision_kind"],
        "variance_decomposition_clipped": decomp_c,
        "variance_decomposition_signed": decomp_s,
        "reliability_raw_clipped": rel_raw,
        "reliability_within_role_clipped": rel_resid,
        "context_capacity_sensitivity": sensitivity,
        "gate": gate,
        "leaderboard_top15_by_raw_plot": top_raw.to_dicts(),
        "leaderboard_top15_by_within_role": top_resid.to_dicts(),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g6_step1.json").write_text(json.dumps(report, indent=2))
    _plot(named, decomp_c, rel_raw, rel_resid, out)

    print("\n=== G6 step 1 — role vs within-role ===")
    print(f"  players in metric: {comp.height}")
    print(f"  role_share (clipped): {decomp_c.get('role_share')}  "
          f"within_role_share: {decomp_c.get('within_role_share')}  "
          f"cross: {decomp_c.get('cross_share')}")
    print(f"  R² role reconstructs leaderboard: {decomp_c.get('r2_role_reconstructs_leaderboard')}")
    print(f"  reliability  raw SB: {gate['reliability_raw_sb']}  "
          f"within-role SB: {gate['reliability_within_role_sb']} "
          f"(p={rel_resid.get('pearson_p')})")
    print("  leakage sensitivity (role model weak→strong): "
          + "  ".join(f"{k}[role={v['role_share']},residSB={v['resid_reliability_sb']}]"
                      for k, v in sensitivity.items()))
    print(f"  GATE: role_dominated={gate['gate']['role_dominated']}  "
          f"within_role_reliable={gate['gate']['within_role_reliable']}")
    print(f"  VERDICT: {gate['verdict']}")
    print(f"  wrote {out}/g6_step1.json + {args.per_player_out}")

    # ===================== STEP 2 — outcome validity (the keystone) =====================
    looks = open_looks_from_intermediates(inter, trace=trace).drop_nulls(
        ["R", "epv_at_decision", "S", "poss_elapsed_s"])
    n_took = int((looks["declined"] == 0).sum())
    n_dec = int((looks["declined"] == 1).sum())
    print(f"\nopen-look decisions: {looks.height} (took {n_took}, declined {n_dec})")

    # calibrate the benchmark S two ways: vs the shot's OWN points (clean) and vs possession points
    calib_fg = taker_calibration(looks, outcome_col="fg_points", n_bins=8)
    calib = taker_calibration(looks, n_bins=8)
    cbv = cost_of_declining_by_value(looks, n_bins=6)
    adj = adjusted_decline_cost(looks, controls=OPEN_LOOK_CONTROLS, n_boot=500, seed=0)
    ov_gate = outcome_validity_gate(adj, cbv)
    player_lv = _player_level(looks, named)

    # robustness: restrict to possessions that ended in a FIELD-GOAL ATTEMPT (no turnovers) so both
    # groups sit in the same structural position — isolates "declining ⇒ worse downstream shot" from
    # "declining ⇒ turnover exposure". If the cost survives here it is not just turnover-risk.
    shotend = looks.filter(pl.col("end_reason").is_in(["made_fg", "defensive_rebound"]))
    adj_shotend = adjusted_decline_cost(shotend, controls=OPEN_LOOK_CONTROLS, n_boot=500, seed=0)
    survives_shotend = bool(adj_shotend["points_lost_by_declining_at_highS"] > 0
                            and adj_shotend["ci_highS"][0] > 0)

    report2 = {
        "corpus": {"clean_games": len(clean), "open_look_decisions": looks.height,
                   "n_took": n_took, "n_declined": n_dec},
        "design": {"universe": "open ball-handler (nearest def >= 4ft, <=30ft from rim) who took (terminal shot) "
                               "or declined (pass) the look", "outcome": "realized possession points R",
                   "benchmark": "S = own open-shot xPoints at actual contest", "controls": OPEN_LOOK_CONTROLS,
                   "team_fe": True, "inference": "game-cluster bootstrap"},
        "taker_calibration_fg_points": calib_fg,
        "taker_calibration_possession": calib,
        "cost_of_declining_by_value": cbv,
        "adjusted_decline_cost": adj,
        "robustness_shot_ending_possessions": {
            "note": "possessions that ended in a FGA (no turnovers); same structural position for both groups",
            "n_decisions": adj_shotend["n_decisions"], "n_took": adj_shotend["n_took"],
            "n_declined": adj_shotend["n_declined"],
            "points_lost_at_highS": adj_shotend["points_lost_by_declining_at_highS"],
            "ci_highS": adj_shotend["ci_highS"], "p_highS": adj_shotend["p_highS"],
            "interaction": adj_shotend["interaction_declinedxS"], "survives": survives_shotend,
        },
        "gate": ov_gate,
        "player_level_exploratory": player_lv,
        "caveats": [
            "taker calibration shows realized > model value S (offensive-rebound putbacks counted in "
            "possession R + coarse terminal-attribution) ⇒ the ABSOLUTE points-lost is an upper bound; "
            "the SIGN and the dose-response (cost grows with look value) are the robust claims",
            "92% of open-handler decisions are declines; the taker pool (1.2k) is structurally drawn "
            "from shot-ending possessions — the shot-ending robustness cut addresses this",
            "selection on UNOBSERVABLES (a developing better play the tracking can't see) is bounded by "
            "conditioning on observables incl. model EPV at the decision, not eliminated (G5b caveat)",
        ],
    }
    (out / "g6_step2.json").write_text(json.dumps(report2, indent=2))
    _plot_step2(calib_fg, cbv, adj, out)

    print("=== G6 step 2 — outcome validity (decision-level keystone) ===")
    print("  taker calib S→own-shot pts (clean): "
          + ", ".join(f"{c['mean_value']:.2f}->{c['mean_realized']:.2f}" for c in calib_fg))
    print("  taker calib S→possession pts: "
          + ", ".join(f"{c['mean_value']:.2f}->{c['mean_realized']:.2f}" for c in calib))
    print("  raw cost of declining by S-bin: "
          + ", ".join(f"{r['mean_value']:.2f}:{r['raw_cost_of_declining']:+.3f}" for r in cbv))
    print(f"  ADJUSTED points lost by declining — meanS {adj['points_lost_by_declining_at_meanS']} "
          f"CI{adj['ci_meanS']} p={adj['p_meanS']} | highS({adj['s_high']}) "
          f"{adj['points_lost_by_declining_at_highS']} CI{adj['ci_highS']} p={adj['p_highS']}")
    print(f"  interaction declined×S {adj['interaction_declinedxS']} CI{adj['ci_interaction']} "
          f"(negative ⇒ cost grows with look value)")
    print(f"  ROBUSTNESS (shot-ending possessions only, no TOs): highS lost "
          f"{adj_shotend['points_lost_by_declining_at_highS']} CI{adj_shotend['ci_highS']} "
          f"survives={survives_shotend} (n_took={adj_shotend['n_took']}, n_dec={adj_shotend['n_declined']})")
    print(f"  player-level (exploratory): {player_lv}")
    print(f"  GATE: {ov_gate['gate']} PASS={ov_gate['pass']}")
    print(f"  VERDICT: {ov_gate['verdict']}")
    print(f"  wrote {out}/g6_step2.json")


def _player_level(looks: pl.DataFrame, named: pl.DataFrame) -> dict:
    """Exploratory player-level: does model PLOT / within-role residual track the realized shortfall
    (S−R) on a player's DECLINED open looks? Shares S with regret (partly mechanical) and is thin at
    42 games — reported with that caveat, not as a gate."""
    from scipy import stats  # noqa: PLC0415
    dec = looks.filter(pl.col("declined") == 1)
    per = (
        dec.group_by("player_id")
        .agg(pl.len().alias("n_declined"), (pl.col("S") - pl.col("R")).mean().alias("realized_decline_shortfall"))
        .filter(pl.col("n_declined") >= 20)
    )
    j = per.join(named.select("player_id", "plot_per100", "within_role_per100"), on="player_id", how="inner")
    if j.height < 5:
        return {"n_players": int(j.height), "note": "too few players"}
    out = {"n_players": int(j.height)}
    y = j["realized_decline_shortfall"].to_numpy()
    for col in ("plot_per100", "within_role_per100"):
        r, p = stats.pearsonr(j[col].to_numpy(), y)
        out[col] = {"pearson_r": round(float(r), 4), "pearson_p": round(float(p), 6)}
    out["caveat"] = "S-R shares S with model regret (partly mechanical); exploratory + underpowered at 42 games"
    return out


def _plot(named: pl.DataFrame, decomp: dict, rel_raw: dict, rel_resid: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    raw = named["plot_per100"].to_numpy()
    role = named["role_per100"].to_numpy()
    ax[0].scatter(role, raw, s=18, alpha=0.65, color="#2c3e50")
    lo, hi = float(min(role.min(), raw.min())), float(max(role.max(), raw.max()))
    ax[0].plot([lo, hi], [lo, hi], "--", color="#aaaaaa", lw=1)
    ax[0].set(xlabel="role component (touch-context) / 100", ylabel="raw PLOT / 100",
              title=f"role reconstructs the leaderboard\nR²={decomp.get('r2_role_reconstructs_leaderboard')}, "
                    f"role_share={decomp.get('role_share')}")
    labels = ["raw regret", "within-role residual"]
    sb = [rel_raw.get("spearman_brown_full") or 0.0, rel_resid.get("spearman_brown_full") or 0.0]
    ax[1].bar(labels, sb, color=["#27ae60", "#c0392b"], alpha=0.85)
    ax[1].axhline(0.0, color="#333", lw=0.8)
    ax[1].set(ylabel="split-half reliability (Spearman-Brown)",
              title="does reliable signal survive removing role?")
    for i, v in enumerate(sb):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout()
    fig.savefig(out / "g6_step1_decomposition.png", dpi=120)
    plt.close(fig)


def _plot_step2(calib: list[dict], cbv: list[dict], adj: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    if calib:
        mv = [c["mean_value"] for c in calib]
        mr = [c["mean_realized"] for c in calib]
        ax[0].scatter(mv, mr, s=28, color="#2c3e50", zorder=3)
        lo, hi = min(mv + mr), max(mv + mr)
        ax[0].plot([lo, hi], [lo, hi], "--", color="#aaaaaa", lw=1)
        ax[0].set(xlabel="model open-shot value S (xPoints)", ylabel="realized own-shot points when TAKEN",
                  title="taker calibration: is S a fair benchmark?")
    if cbv:
        mv = [r["mean_value"] for r in cbv]
        cost = [r["raw_cost_of_declining"] for r in cbv]
        ax[1].plot(mv, cost, "-o", color="#c0392b")
        ax[1].axhline(0.0, color="#333", lw=0.8)
        ax[1].set(xlabel="model open-shot value S (xPoints)",
                  ylabel="realized points lost by declining",
                  title=f"cost of declining vs look value\nadj high-S: "
                        f"{adj.get('points_lost_by_declining_at_highS')} CI{adj.get('ci_highS')}")
    fig.tight_layout()
    fig.savefig(out / "g6_step2_outcome_validity.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
