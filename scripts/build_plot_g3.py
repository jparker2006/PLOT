#!/usr/bin/env python
"""Stage 5 — the PLOT metric (points left on the table / 100 decisions) and Gate G3.

    uv run --extra seq python scripts/build_oof_epv.py --kfold 7   # once: leakage-free EPV trace (42 games)
    uv run --extra seq python scripts/build_plot_g3.py             # -> reports/G3/

Builds the per-decision narrow regret over every clean game (open pass-up: xPoints(open shot) −
post-pass EPV, on the recalibrated leakage-free trace), aggregates it per player per 100 decisions,
and runs the two G3 checks:

  G3(a) stable     — split the season's games odd/even, correlate per-player regret across halves.
  G3(b) decision   — (i) player fixed effects explain regret beyond shot location (nested-OLS F-test)
                     and (ii) per-player regret is ~uncorrelated with finishing skill (so it isn't a
                     relabeled finishing metric).

G3 PASS = stable AND decision-not-location AND decision-not-finishing. Whatever the verdict, the raw
statistics are written to reports/G3/g3_stability.json so the result stands on its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from plot.io.loaders import events_table  # noqa: E402
from plot.models.counterfactual.regret import decision_states, regret_from_states, regret_summary  # noqa: E402
from plot.models.counterfactual.xpoints import logo_xpoints, train_xpoints  # noqa: E402
from plot.models.eval_bar.seq_dataset import build_game_canonical  # noqa: E402
from plot.models.regret.plot_metric import (  # noqa: E402
    aggregate_per_player,
    finishing_independence,
    finishing_residual_by_player,
    g3_gate,
    player_fixed_effect_test,
    split_half_stability,
)
from plot.possessions.actions import extract_actions  # noqa: E402
from plot.possessions.shots import extract_shots  # noqa: E402

# G3 gate thresholds (documented, not tuned to pass)
MIN_DECISIONS = 30            # a player needs this many evaluated open pass-ups to enter the metric
MIN_PER_HALF = 15             # ... and this many in EACH season-half to enter the stability test
FINISHING_CORR_MAX = 0.40     # |corr(regret, finishing skill)| must stay below this to be "not finishing"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="reports/G3")
    ap.add_argument("--oof-trace", default="data/processed/oof_epv_trace.parquet")
    ap.add_argument("--per-player-out", default="data/processed/plot_per_player.parquet")
    args = ap.parse_args()

    trace_path = Path(args.oof_trace)
    if not trace_path.exists():
        raise SystemExit(f"missing {trace_path} — run: uv run --extra seq python scripts/build_oof_epv.py --kfold 7")
    trace = pl.read_parquet(trace_path)
    if "epv_cal" in trace.columns:  # the counterfactual consumes the isotonic-recalibrated EPV
        trace = trace.with_columns(pl.col("epv_cal").alias("epv"))

    games = args.games or sorted(p.stem for p in Path(args.raw_dir, "json").glob("*.json"))
    shots_parts, actions_parts, states_parts, quarantined = [], [], [], []
    for g in games:
        cdf, poss, _ = build_game_canonical(g, raw_dir=args.raw_dir)
        if cdf is None:
            quarantined.append(g)
            continue
        ev = events_table(Path(args.raw_dir, "2015-16_pbp.csv"), g)
        actions = extract_actions(cdf, poss, g)
        shots_parts.append(extract_shots(cdf, ev, g))
        actions_parts.append(actions)
        states_parts.append(decision_states(cdf, actions).with_columns(pl.lit(g).alias("game_id")))
        del cdf
    clean = [g for g in games if g not in quarantined]
    shots = pl.concat(shots_parts)
    actions = pl.concat(actions_parts)
    states = pl.concat(states_parts)
    print(f"clean games: {len(clean)} | shots: {shots.height} | trace frames: {trace.height}")

    # ---- per-decision regret over every clean game (full-data xPoints model + leakage-free trace) ----
    full_model = train_xpoints(shots)
    regret_parts = [
        regret_from_states(states.filter(pl.col("game_id") == g), actions.filter(pl.col("game_id") == g),
                           full_model, trace, g)
        for g in clean
    ]
    regret = pl.concat([r for r in regret_parts if r.height]) if any(r.height for r in regret_parts) else pl.DataFrame()
    print(f"per-decision regret: {regret.height} decisions over {regret['player_id'].n_unique()} players")

    # ---- the PLOT metric: per player per 100 decisions ----
    per_player = aggregate_per_player(regret, per=100, min_decisions=MIN_DECISIONS)
    Path(args.per_player_out).parent.mkdir(parents=True, exist_ok=True)
    per_player.write_parquet(args.per_player_out)

    # ---- G3(a): split-half stability (headline = clipped; also report signed) ----
    stab_clipped = split_half_stability(regret, value_col="regret_clipped", min_decisions_per_half=MIN_PER_HALF)
    stab_signed = split_half_stability(regret, value_col="regret_signed", min_decisions_per_half=MIN_PER_HALF)

    # ---- G3(b-i): player fixed effects beyond shot location ----
    fe = player_fixed_effect_test(regret, value_col="regret_signed", min_decisions=MIN_DECISIONS)

    # ---- G3(b-ii): regret vs finishing skill (leakage-free p_make on REAL shots) ----
    shots_oof = logo_xpoints(shots)
    finishing = finishing_residual_by_player(shots_oof, min_shots=10)
    indep = finishing_independence(per_player, finishing, regret_col="mean_regret_signed")

    # ---- gate ----
    gate = g3_gate(stab_clipped, fe, indep, finishing_corr_max=FINISHING_CORR_MAX)
    g3 = gate["gate"]
    report = {
        "n_games_clean": len(clean), "n_games_quarantined": len(quarantined),
        "thresholds": {"min_decisions": MIN_DECISIONS, "min_decisions_per_half": MIN_PER_HALF,
                       "finishing_corr_max": FINISHING_CORR_MAX},
        "regret_distribution": regret_summary(regret),
        "n_players_in_metric": int(per_player.height),
        "plot_leaderboard_top10": per_player.head(10).to_dicts(),
        "plot_leaderboard_bottom10": per_player.tail(10).to_dicts(),
        "G3a_split_half_clipped": stab_clipped,
        "G3a_split_half_signed": stab_signed,
        "G3b_player_fixed_effects": fe,
        "G3b_finishing_independence": indep,
        "G3_gate": g3, "G3_pass": bool(gate["pass"]),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g3_stability.json").write_text(json.dumps(report, indent=2))
    _plots(regret, per_player, out)

    print("\n=== Gate G3 (PLOT metric: stable & not just finishing) ===")
    print(f"  regret demo: {report['regret_distribution']}")
    print(f"  players in metric (≥{MIN_DECISIONS} decisions): {per_player.height}")
    sc = stab_clipped
    print(f"  G3(a) split-half [clipped] n={sc.get('n_players')}: pearson {sc.get('pearson_r')} "
          f"(p={sc.get('pearson_p')}), spearman {sc.get('spearman_r')}, SB-full {sc.get('spearman_brown_full')}")
    print(f"  G3(b-i) player FE beyond location: incremental R² {fe.get('incremental_r2_player')}, "
          f"F{tuple(fe.get('f_df', []))}={fe.get('f_stat')} (p={fe.get('p_value')})")
    print(f"  G3(b-ii) regret vs finishing: pearson {indep.get('pearson_r')} (p={indep.get('pearson_p')}), "
          f"n={indep.get('n_players')}")
    print(f"\n  G3 {'PASS' if report['G3_pass'] else 'FAIL'}  (gate: {g3})")
    print(f"  wrote {out}/g3_stability.json + figures, {args.per_player_out}")


def _plots(regret: pl.DataFrame, per_player: pl.DataFrame, out: Path) -> None:
    if regret.height == 0:
        return
    # split-half scatter (clipped)
    halves = {g: (i % 2) for i, g in enumerate(sorted(regret["game_id"].unique().to_list()))}
    tagged = regret.with_columns(pl.col("game_id").replace_strict(halves, default=0).alias("_half"))
    by = (tagged.group_by("player_id", "_half")
          .agg(pl.len().alias("n"), pl.col("regret_clipped").mean().alias("m"))
          .filter(pl.col("n") >= MIN_PER_HALF))
    wide = by.pivot(values=["n", "m"], index="player_id", on="_half").drop_nulls()
    m_cols = sorted(c for c in wide.columns if c.startswith("m_"))
    if len(m_cols) >= 2 and wide.height >= 3:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(wide[m_cols[0]].to_numpy(), wide[m_cols[1]].to_numpy(), s=16, alpha=0.6)
        lo = min(wide[m_cols[0]].min(), wide[m_cols[1]].min())
        hi = max(wide[m_cols[0]].max(), wide[m_cols[1]].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("regret/decision — season half A")
        ax.set_ylabel("regret/decision — season half B")
        ax.set_title("G3(a) split-half stability (clipped regret)")
        fig.tight_layout()
        fig.savefig(out / "g3_split_half_scatter.png", dpi=120)
        plt.close(fig)
    # per-player PLOT distribution
    if per_player.height:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(per_player["plot_per100_clipped"].to_numpy(), bins=30, color="#c0392b", alpha=0.8)
        ax.set_xlabel("PLOT — points left on the table / 100 decisions (clipped)")
        ax.set_ylabel("players")
        ax.set_title("Per-player PLOT distribution")
        fig.tight_layout()
        fig.savefig(out / "g3_plot_distribution.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
