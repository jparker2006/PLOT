#!/usr/bin/env python
"""Stage 7 — export the precomputed per-possession JSON the Next.js demo plays.

    uv run --extra seq python scripts/export_demo_json.py --games 0021500308 0021500203

For each game: canonical 10 fps tracking + the leakage-free EPV trace (the eval bar) + per-decision
regret with chess.com-style badges, assembled by ``plot.viz.demo_export`` into static JSON under
``web/public/demo/``. No model runs in the browser.

xPoints (the open-shot value behind regret) is fit on the *featured games'* shots here — enough for
an illustrative demo; the validated metric (G3/G4/G5) uses the full corpus. The EPV eval bar is the
already-computed leakage-free recalibrated trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from plot.io.loaders import load_game_json, players_table, team_info
from plot.models.counterfactual.regret import decision_states, regret_from_states
from plot.models.counterfactual.xpoints import train_xpoints
from plot.models.eval_bar.seq_dataset import build_game_canonical
from plot.possessions.actions import extract_actions
from plot.possessions.shots import extract_shots
from plot.viz.demo_export import build_possession, select_featured_possessions

_BADGES = ["great", "good", "inaccuracy", "mistake", "blunder"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--oof-trace", default="data/processed/oof_epv_trace.parquet")
    ap.add_argument("--out-dir", default="web/public/demo")
    ap.add_argument("--top-n", type=int, default=12, help="featured possessions per game")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth 10fps frame")
    args = ap.parse_args()

    trace = pl.read_parquet(args.oof_trace)
    if "epv_cal" in trace.columns:
        trace = trace.with_columns(pl.col("epv_cal").alias("epv"))

    # --- build canonical + shots for every requested game, then a shared xPoints ---
    built: dict[str, tuple] = {}
    shots_parts = []
    for g in args.games:
        cdf, poss, _ = build_game_canonical(g, raw_dir=args.raw_dir)
        if cdf is None:
            print(f"  {g}: quarantined (no canonical frames) — skipping")
            continue
        game = load_game_json(Path(args.raw_dir, "json", f"{g}.json"))
        ev = pl.read_csv(Path(args.raw_dir, "2015-16_pbp.csv"), infer_schema_length=20000).filter(
            pl.col("GAME_ID") == int(g)
        ).rename({"EVENTNUM": "event_id", "EVENTMSGTYPE": "msg_type",
                  "EVENTMSGACTIONTYPE": "action_type", "PERIOD": "period"}).with_columns(
            pl.coalesce("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION").alias("description")
        )
        actions = extract_actions(cdf, poss, g)
        shots = extract_shots(cdf, ev, g)
        shots_parts.append(shots)
        built[g] = (cdf, poss, actions, shots, game)
    if not built:
        raise SystemExit("no games built")
    xmodel = train_xpoints(pl.concat(shots_parts))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for g, (cdf, poss, actions, _shots, game) in built.items():
        ti = team_info(game)
        ptab = players_table(game)
        team_abbr = {int(r["team_id"]): r["team_abbr"] for r in ptab.to_dicts()}
        player_meta = {int(r["player_id"]): {"name": r["name"], "team": int(r["team_id"])}
                       for r in ptab.to_dicts()}

        # per-frame eval bar: attach EPV to the canonical frames
        tg = trace.filter(pl.col("game_id") == g).select("possession_id", "wall_clock_ms", "epv")
        frames_all = (
            cdf.filter(pl.col("entity").is_in(["ball", "player"]))
            .select("possession_id", "wall_clock_ms", "game_clock", "entity", "player_id", "team_id",
                    pl.col("x_canon").alias("x"), pl.col("y_canon").alias("y"), "z")
            .join(tg, on=["possession_id", "wall_clock_ms"], how="inner")
        )

        # per-decision regret (+ recover wall_clock_ms from the decision states)
        states = decision_states(cdf, actions).with_columns(pl.lit(g).alias("game_id"))
        regret = regret_from_states(states, actions, xmodel, trace, g)
        if regret.height:
            regret = regret.join(
                states.select("possession_id", "action_idx", "wall_clock_ms"),
                on=["possession_id", "action_idx"], how="left",
            )

        featured = select_featured_possessions(regret, top_n=args.top_n)
        poss_rows = {int(r["possession_id"]): r for r in poss.to_dicts()}
        payloads, badge_counts = [], dict.fromkeys(_BADGES, 0)
        for pid in featured:
            pf = frames_all.filter(pl.col("possession_id") == pid)
            if pf.height == 0:
                continue
            pd = regret.filter(pl.col("possession_id") == pid) if regret.height else regret
            payload = build_possession(pid, pf, pd, poss_rows[pid], stride=args.stride)
            for d in payload["decisions"]:
                badge_counts[d["badge"]] += 1
            payloads.append(payload)

        matchup = f'{ti.get("visitor_abbr", "AWAY")} @ {ti.get("home_abbr", "HOME")}'
        doc = {
            "game_id": g, "matchup": matchup,
            "teams": {"home": {"id": int(ti["home_id"]), "abbr": ti.get("home_abbr", "HOME")},
                      "visitor": {"id": int(ti["visitor_id"]), "abbr": ti.get("visitor_abbr", "AWAY")}},
            "team_abbr": team_abbr,
            "players": player_meta,
            "court": {"length": 94.0, "width": 50.0, "rim_x": 88.75, "rim_y": 25.0},
            "badge_counts": badge_counts,
            "possessions": payloads,
        }
        (out / f"{g}.json").write_text(json.dumps(doc, separators=(",", ":")))
        size_kb = (out / f"{g}.json").stat().st_size // 1024
        index.append({"game_id": g, "matchup": matchup, "n_possessions": len(payloads),
                      "badge_counts": badge_counts})
        print(f"  {g} [{matchup}]: {len(payloads)} possessions, badges {badge_counts}, {size_kb} KB")

    (out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {out}/index.json ({len(index)} games)")


if __name__ == "__main__":
    main()
