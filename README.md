# PLOT — Points Left On the Table

> chess.com's post-game review, but for NBA possessions.

At every instant of a possession there is an **eval bar** — the expected points right now.
For each ball-handler decision (pass / drive / shoot), **PLOT** measures **regret**: the gap
between the best genuinely-available option and the one the player chose. Regret = points left
on the table.

This repository contains three deliverables built on one spine:

1. **Value layer** — an open, reproducible per-action expected-points model for NBA possessions
   (a "VAEP-for-basketball"), from public player + ball tracking.
2. **PLOT metric** — decision regret, valued against the distribution of genuinely-available
   alternatives, aggregated to a per-player *points left on the table*.
3. **Live demo** — an animated, chess.com-style possession review (Next.js).

An accompanying arXiv preprint documents the value layer + PLOT with calibration, validation,
and honest limitations.

> **Status:** early. Built on the only frozen public tracking season (2015-16 SportVU).

## Proof of life (Stage 1)

A possession rebuilt from raw 2015-16 SportVU tracking — 10 players + ball on the court with the
game/shot clock. The eval bar and per-decision regret overlay come in later stages.

![PLOT Stage-1 proof of life: a LeBron Cavs possession animated from tracking data](docs/proof_of_life.gif)

## Eval bar (Stage 2) — Gate G1: PASS

The baseline **eval bar** predicts per-moment EPV (expected points the possession will yield)
from coarse location/context features via a LightGBM multiclass outcome head. On leave-one-game-out
held-out games it is well calibrated — calibration-in-the-large **+0.006**, ECE **0.054** points,
and it beats a constant-rate baseline on log-loss, Brier, and the ordinal RPS. It is a deliberately
decision-insensitive *location prior*; the per-action value and regret metric build on it in later
stages. Full report + figures: [`reports/G1/`](reports/G1/README.md).

```bash
uv run python scripts/build_eval_bar_features.py   # cache per-game features (+orientation/QC)
uv run python scripts/train_eval_bar_g1.py         # LOGO calibration -> reports/G1/
```

## Quickstart

```bash
# Python env (pinned to 3.12 via uv)
uv sync          # on macOS, LightGBM needs the OpenMP runtime: `brew install libomp`

# Download one game and verify the data schema
uv run python scripts/download_data.py --tier T0

# Tests
uv run pytest
```

See [`DATA.md`](DATA.md) for data sources, provenance, and the data-availability/ethics stance.

## Layout

```
src/plot/
  io/             # loaders + internal-schema adapter
  possessions/    # possession segmentation + action extraction
  features/       # feature engineering
  models/
    eval_bar/        # EPV (expected possession value): baseline -> sequence model
    action_value/    # per-action value = ΔEPV (VAEP-analog)
    counterfactual/  # xPoints + counterfactual valuation
    regret/          # the PLOT metric
  viz/            # matplotlib prototype animation
  eval/           # calibration, reliability, gate tests
scripts/          # data download, possession build, demo export
notebooks/        # one validation report per gate (G1–G4)
web/              # Next.js demo app
paper/            # arXiv write-up + figures
tests/            # pytest
```

## Acknowledgements / prior art

- **EPV** — Cervone, D'Amour, Bornn, Goldsberry (arXiv:1408.0777; JASA 2016).
- **VAEP** — Decroos, Bransen, Van Haaren, Davis (KDD 2019); `ML-KULeuven/socceraction`.
- Tracking data and animation reference: `dcayton/nba_tracking_data_15_16`,
  `linouk23/NBA-Player-Movements`.
