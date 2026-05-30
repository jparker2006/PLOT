#!/usr/bin/env python
"""Stage 6 — player aggregation + headline, and Gate G4: is PLOT new information?

    uv run python scripts/build_g4.py     # needs data/processed/plot_per_player.parquet (from G3)

Reads the per-player PLOT metric (built by ``build_plot_g3.py``), computes season-long per-player
box stats from play-by-play (TS%, a usage proxy = scoring possessions / game, points / game,
FGA / game), and runs the two G4 checks:

  G4 incremental info — per-player PLOT is only weakly correlated with each box stat (so PLOT is not
                        efficiency / usage / scoring relabeled), and a joint OLS of PLOT on the box
                        stats explains little of its variance.
  G4 reliable         — the metric repeats across season halves. REUSED from G3(a) (split-half /
                        Spearman–Brown), read from reports/G3/g3_stability.json — not recomputed,
                        since the per-decision regret for the 208-game run is not local.

G4 PASS = incremental info (every |Pearson| < CORR_MAX) AND reliable (Spearman–Brown ≥ RELIABILITY_MIN).
Whatever the verdict, the raw statistics are written to reports/G4/g4.json so the result stands alone.

NOTE on window: the box stats are season-long while PLOT is from the 208 clean tracking games
(2015-10-27 .. 2016-01-23). The 208-game id list is not local, and a player's efficiency/usage
reputation is conventionally season-long, so this is the reproducible comparator — documented as a
limitation in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from plot.eval.box_stats import (  # noqa: E402
    correlate_plot_with_box,
    g4_gate,
    normalize_pbp,
    per_player_box_stats,
    variance_explained,
)

# G4 thresholds (pre-registered, documented — not tuned to pass)
CORR_MAX = 0.50          # |Pearson(PLOT, box stat)| below this ⇒ PLOT carries info beyond that stat
RELIABILITY_MIN = 0.60   # Spearman–Brown full-sample reliability (G3a) must clear this
MIN_FGA_FTA = 50         # a player needs this many true-shooting attempts for stable box rates

# the box stats PLOT could be "secretly measuring"; TS% + usage are gated, the rest are reported
GATE_STATS = ["ts_pct", "usg_proxy_pg"]
ALL_STATS = ["ts_pct", "usg_proxy_pg", "pts_pg", "fga_pg"]
_STAT_LABELS = {
    "ts_pct": "true-shooting %",
    "usg_proxy_pg": "usage proxy (scoring poss / game)",
    "pts_pg": "points / game",
    "fga_pg": "FGA / game",
}
PLOT_COL = "plot_per100_clipped"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbp", default="data/raw/2015-16_pbp.csv")
    ap.add_argument("--per-player", default="data/processed/plot_per_player.parquet")
    ap.add_argument("--g3-json", default="reports/G3/g3_stability.json")
    ap.add_argument("--out-dir", default="reports/G4")
    ap.add_argument("--box-out", default="data/processed/plot_box_stats.parquet")
    args = ap.parse_args()

    pp_path = Path(args.per_player)
    if not pp_path.exists():
        raise SystemExit(f"missing {pp_path} — run scripts/build_plot_g3.py first")
    per_player = pl.read_parquet(pp_path)

    # ---- season-long per-player box stats from PBP ----
    pbp = pl.read_csv(
        args.pbp,
        infer_schema_length=20000,
        columns=["GAME_ID", "EVENTMSGTYPE", "PLAYER1_ID", "PLAYER1_NAME",
                 "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"],
    )
    box = per_player_box_stats(normalize_pbp(pbp), min_fga_fta=MIN_FGA_FTA)
    Path(args.box_out).parent.mkdir(parents=True, exist_ok=True)
    box.write_parquet(args.box_out)

    joined = per_player.join(box, on="player_id", how="inner")
    n_matched, n_plot = joined.height, per_player.height
    print(f"box stats: {box.height} players | PLOT players: {n_plot} | matched: {n_matched}")

    # ---- G4 incremental information: correlations + joint variance explained ----
    corr_clipped = correlate_plot_with_box(joined, target=PLOT_COL, stat_cols=ALL_STATS)
    corr_signed = correlate_plot_with_box(joined, target="plot_per100_signed", stat_cols=ALL_STATS)
    var_exp = variance_explained(joined, target=PLOT_COL, predictors=GATE_STATS)
    var_exp_all = variance_explained(joined, target=PLOT_COL, predictors=ALL_STATS)

    # ---- G4 reliability: reused from G3(a) ----
    g3 = json.loads(Path(args.g3_json).read_text()) if Path(args.g3_json).exists() else {}
    g3a = g3.get("G3a_split_half_clipped", {})
    reliability_sb = g3a.get("spearman_brown_full")
    reliability_split_half = g3a.get("pearson_r")

    # ---- gate ----
    gate = g4_gate(corr_clipped, reliability_sb, corr_max=CORR_MAX,
                   reliability_min=RELIABILITY_MIN, gate_stats=GATE_STATS)

    report = {
        "thresholds": {"corr_max": CORR_MAX, "reliability_min": RELIABILITY_MIN,
                       "min_fga_fta": MIN_FGA_FTA, "gate_stats": GATE_STATS},
        "n_players_plot": int(n_plot),
        "n_players_box": int(box.height),
        "n_players_matched": int(n_matched),
        "plot_col": PLOT_COL,
        "reliability": {
            "source": "G3(a), reused (not recomputed)",
            "split_half_pearson": reliability_split_half,
            "spearman_brown_full": reliability_sb,
        },
        "incremental_info_clipped": corr_clipped,
        "incremental_info_signed": corr_signed,
        "variance_explained_gated": var_exp,
        "variance_explained_all": var_exp_all,
        "headline_leaderboard_top10": joined.sort(PLOT_COL, descending=True)
            .select("player_id", "player_name", "n_decisions", PLOT_COL, "ts_pct", "usg_proxy_pg")
            .head(10).to_dicts(),
        "headline_leaderboard_bottom10": joined.sort(PLOT_COL, descending=True)
            .select("player_id", "player_name", "n_decisions", PLOT_COL, "ts_pct", "usg_proxy_pg")
            .tail(10).to_dicts(),
        "G4_gate": gate["gate"], "G4_pass": bool(gate["pass"]),
        "G4_gate_detail": gate,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g4.json").write_text(json.dumps(report, indent=2))
    _plots(joined, corr_clipped, out)

    print("\n=== Gate G4 (PLOT is new information, not box-score relabeled) ===")
    print(f"  matched players: {n_matched}/{n_plot}")
    for s in ALL_STATS:
        c = corr_clipped[s]
        flag = "  [GATED]" if s in GATE_STATS else ""
        print(f"  corr(PLOT, {_STAT_LABELS[s]}): Pearson {c.get('pearson_r')} "
              f"(p={c.get('pearson_p')}), Spearman {c.get('spearman_r')}, n={c.get('n')}{flag}")
    print(f"  box stats jointly explain PLOT: R² {var_exp.get('r2')} (gated 2 stats), "
          f"{var_exp_all.get('r2')} (all 4)")
    print(f"  reliability (G3a, reused): split-half {reliability_split_half}, "
          f"Spearman–Brown {reliability_sb}")
    print(f"\n  G4 {'PASS' if report['G4_pass'] else 'FAIL'}  (gate: {gate['gate']})")
    print(f"  wrote {out}/g4.json + figures, {args.box_out}")


def _plots(joined: pl.DataFrame, corr: dict, out: Path) -> None:
    if joined.height == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, stat in zip(axes.ravel(), ALL_STATS, strict=True):
        sub = joined.select([PLOT_COL, stat]).drop_nulls()
        x = sub[stat].to_numpy().astype(float)
        y = sub[PLOT_COL].to_numpy().astype(float)
        ax.scatter(x, y, s=14, alpha=0.5, color="#2c7fb8")
        if len(x) >= 2:
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, b0 + b1 * xs, "k--", lw=0.9, alpha=0.7)
        r = corr.get(stat, {}).get("pearson_r")
        ax.set_title(f"{_STAT_LABELS[stat]}  (r={r})", fontsize=10)
        ax.set_xlabel(_STAT_LABELS[stat])
        ax.set_ylabel("PLOT / 100 (clipped)")
    fig.suptitle("G4 — per-player PLOT vs box-score stats (low corr ⇒ new information)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "g4_plot_vs_box.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
