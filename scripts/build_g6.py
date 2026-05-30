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
from plot.models.regret.pipeline import build_v15_regret  # noqa: E402
from plot.models.regret.plot_metric import split_half_stability  # noqa: E402

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

    regret, _shots, _xmodel, clean = build_v15_regret(games, raw_dir=args.raw_dir, trace=trace,
                                                       completion=args.completion)
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


if __name__ == "__main__":
    main()
