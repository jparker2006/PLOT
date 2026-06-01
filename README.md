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
3. **Live demo** — an animated, chess.com-style possession review (Next.js), **live at
   <https://plot-nba.vercel.app>**.

A drafted arXiv preprint ([`paper/`](paper/)) documents the value layer + PLOT with calibration, the
outcome-validity tests, and honest limitations.

> **Status:** the gated program **G1–G6 is complete** — including outcome validity against *real*
> possession outcomes (G6). Built on the only frozen public tracking season (2015-16 SportVU); the
> outcome-validated claim is deliberately narrow and modest in magnitude (see G6).

## Proof of life (Stage 1)

A possession rebuilt from raw 2015-16 SportVU tracking — 10 players + ball on the court with the
game/shot clock. The eval bar and per-decision regret overlay come in later stages.

![PLOT Stage-1 proof of life: a LeBron Cavs possession animated from tracking data](docs/proof_of_life.gif)

## Eval bar (Stage 2) — Gate G1: PASS

The **eval bar** predicts per-moment EPV (expected points the possession will yield) and reads it
off a multiclass outcome head over points `{0,1,2,3}` as `EPV = Σ_k k·P(points=k)`. Both models are
validated under leave-one-game-out cross-validation on held-out **games**, with the identical
calibration code and quarantine — so the upgrade below is an apples-to-apples comparison.

- **Baseline (LightGBM, location/context prior).** Coarse per-frame features (canonical ball
  geometry, region, clocks, score margin, paint counts, ball-handler pressure). Well calibrated:
  calibration-in-the-large **+0.006**, ECE **0.054** pts, beats a constant-rate baseline on log-loss,
  Brier, and the ordinal RPS. Deliberately decision-insensitive. Report: [`reports/G1/`](reports/G1/README.md).
- **Sequence upgrade (PyTorch, full ten-player tracking).** A permutation-invariant DeepSets set
  encoder over the ten players → a **causal** GRU over the 10 fps sequence → the same outcome head.
  It beats the baseline on every proper score — log-loss **0.819 vs 1.014**, Brier **0.461 vs 0.570**,
  RPS **0.428 vs 0.545** — and tightens the calibration slope from 1.15 to **0.96**, while staying
  calibrated (ECE 0.063, cal-in-large −0.012). Report: [`reports/G1_seq/`](reports/G1_seq/README.md).

```bash
uv run python scripts/build_eval_bar_features.py     # cache per-game features (+orientation/QC)
uv run python scripts/train_eval_bar_g1.py           # baseline LOGO calibration -> reports/G1/

uv sync --extra seq                                  # installs torch (MPS/CPU on macOS)
uv run --extra seq python scripts/train_seq_epv_g1.py  # sequence LOGO calibration -> reports/G1_seq/
```

## Per-action value (Stage 3)

The eval bar gives expected points at every instant; the **per-action value layer** turns that into
credit per decision: `value(action) = EPV(end) − EPV(start)` over each on-ball action (a
"VAEP-for-basketball"). Pinning each possession's terminal action to its realized outcome makes the
values **telescope exactly** to `realized − initial EPV` (verified on all 1,774 possessions), and the
credit is sensible — made shots **+0.84**, drawn fouls **+0.15**, passes **≈0**, missed shots
**−0.47**, turnovers **−0.54** (mean ΔEPV). This is the first decision-sensitive signal; the
counterfactual + PLOT regret metric build on it next. Report: [`reports/stage3/`](reports/stage3/README.md).

```bash
uv run --extra seq python scripts/build_action_values.py   # value the actions -> reports/stage3/
```

## Counterfactual + regret (Stage 4) — Gate G2: PASS

The crux. **Decision regret** = `xPoints(best available open shot) − value(chosen action)` is only
meaningful if its inputs are calibrated one step ahead. **xPoints** (expected points of shooting now,
a logistic fit on real shots) is well-calibrated on held-out games — cal-in-large **+0.002**, ECE
**0.077**, and it beats the base make-rate. The **post-pass EPV** that scores a chosen pass (from the
leakage-free out-of-fold trace, isotonic-recalibrated) is calibrated too — cal-in-large **−0.014**,
ECE **0.066**. So regret rests on honest inputs, not an artifact. A v1 narrow regret (open-shot vs
post-pass EPV) is demonstrated over 5,525 decisions; the per-player metric + stability gate are next.
Report: [`reports/G2/`](reports/G2/README.md).

```bash
uv run --extra seq python scripts/build_oof_epv.py       # leakage-free OOF EPV trace
uv run --extra seq python scripts/train_xpoints_g2.py    # xPoints + G2 -> reports/G2/
```

## PLOT metric (Stage 5) — Gate G3: PASS

The flagship. Per-decision regret is aggregated to a per-player **points left on the table / 100
decisions**, and G3 asks the two questions that decide whether that number is real: does it **repeat**
(a skill, not noise) and is it **distinct from finishing** (a decision property, not shot-making)?
On **124,953 open pass-up decisions across 208 games** (380 players in the metric), both hold:

- **Stable across the season.** Split the games odd/even, aggregate per player, correlate: split-half
  Pearson **0.514**, Spearman–Brown full-sample reliability **0.68** — a solid, decisively repeatable
  signal (up from 0.38 / SB 0.55 on the first 42-game pass — scaling the corpus tightened it).
- **A decision, not just location.** Player fixed effects are jointly significant *beyond* a full
  shot-location model — F(380, 124568) = **4.96, p≈0**.
- **A decision, not finishing.** Per-player regret is only weakly related to finishing skill (xPoints
  uses a population make model): correlation **−0.17** — well short of being a finishing proxy.

So PLOT measures a genuine decision attribute — the project's thesis. v1 scores one decision type
(an open handler passing up the shot). The 208-game run is memory-bounded at any scale (peak ~5GB).
Report: [`reports/G3/`](reports/G3/README.md).

```bash
uv run --extra seq python scripts/build_oof_epv.py --kfold 5   # leakage-free OOF EPV trace (group k-fold)
uv run --extra seq python scripts/build_plot_g3.py             # PLOT metric + G3 -> reports/G3/
```

## Player headline (Stage 6) — Gate G4: PASS

The skeptic's question: *isn't PLOT just shooting efficiency (or volume) relabeled?* G4 correlates
the per-player headline against the box-score quantities it could be confused for, and pairs that
with the reliability already shown in G3a. Over **374 of the 380** metric players (matched to
season-long box stats):

- **Unrelated to efficiency.** Correlation with true-shooting % is **+0.086** (not significant) —
  PLOT is not a good-shooters metric.
- **Only weakly tied to usage/volume.** Usage proxy **−0.24**, points/game −0.21, FGA/game −0.24 —
  a legible *role* pattern (rim-running bigs leave the most on the table; high-usage lead guards the
  least), but all far below the 0.50 bar. A joint OLS on all four box stats explains just **7.6%**
  of PLOT's variance — ~92% is information the box score cannot reconstruct.
- **Reliable.** Split-half **0.51**, Spearman–Brown **0.68** (reused from G3a).

So PLOT is a repeatable signal the box score is structurally blind to — the thesis. It is **new
information**; whether that information is *decision quality* (vs. role-driven deference) is the
honest open question, and v1 still scores one decision type. Report: [`reports/G4/`](reports/G4/README.md).

```bash
uv run python scripts/build_g4.py                              # box stats + G4 -> reports/G4/
```

## Validity (Stage 6.5) — Gate G5: PASS (with a stated caveat)

G4 showed PLOT is *new* information; G5 asks whether it is *valid* — are the "points left on the
table" genuinely available, or an artifact of how the inputs are valued? Three checks on the 42-game
decision set:

- **Inputs calibrated where the metric lives.** On the open pass-up decisions, post-pass EPV matches
  realized possession points (cal-in-large −0.002, ECE 0.055) and xPoints matches realized makes
  (ECE 0.040). Positive regret really is value not realized.
- **Not a finishing-skill artifact.** Rebuilding regret with shooter-aware xPoints (each shot valued
  at the shooter's own make rate) barely reorders the per-player metric — **Spearman 0.93**.
- **Selection is bounded, not eliminated.** Players select which open looks to take (propensity AUC
  0.82) — but the selectors are *distance* and *openness*, exactly xPoints' own features, so the
  metric conditions on them. The residual threat is selection on **unobservables** (shot difficulty
  beyond location/openness), which no observational metric can rule out — reported as PLOT's honest
  ceiling.

So PLOT measures genuinely-left value **conditional on observables**; it does not prove causal
decision quality (that needs the missing counterfactual). Report: [`reports/G5/`](reports/G5/README.md).

```bash
uv run --extra seq python scripts/build_g5.py                  # validity checks + G5 -> reports/G5/
```

## Outcome validity + what it measures — Gate G6 (the keystone)

G1–G5 prove PLOT is calibrated, stable, distinct, and valid *conditional on observables* — but all of
that is the model judged against itself. **G6 is the first test against reality**, and it answers the two
questions that decide whether the metric is *actually useful*. On the 208-game corpus:

- **Within-role decision quality, not just role-of-touch.** Residualizing per-decision regret on
  touch-context (location, openness, rim distance, decision kind), out-of-fold by game, splits the
  between-player signal: within-role decision quality is the **majority (52%)**, role-of-touch the
  minority (**29%**), and the within-role residual is itself reliable (Spearman–Brown **0.54**) and robust
  to the strength of the role control (not leakage). The leaderboard is mostly real, role-adjusted
  decision-making — not just position.
- **The keystone — declining your own open look costs real points.** The cleanest decision (an open
  ball-handler who *takes* vs *declines* the shot) has a *directly observable* counterfactual: real
  shooters at matched looks. Net of observables, shot-clock, and team fixed effects, **declining an open
  look costs ≈0.11 points on an average look, ≈0.19 on a great one**; declining a *bad* look is free; the
  cost grows with look value; and it survives removing turnover exposure and action-layer mis-typing. The
  natural selection objection cuts the *wrong* way — decliners realize *fewer* points, not more. This is
  the one result that touches reality, and it **passes**.
- **The open-man myth (a defining negative).** The mirror test — *shooting over a wide-open teammate* —
  does **not** validate: the teammate-kick counterfactual is flat in its modeled value, and shooting over
  realizes *more* points, not fewer. An "open man" is often open *because* he is not a threat. So PLOT is
  outcome-valid for the decision whose counterfactual is observable (your own open look), **not** for the
  conventional "pass to the open man" read.
- **Player-level attribution is weak.** A de-circularized cross-fit (model metric on one half of games,
  realized points-left on the other, benchmarked against real takers — not the model) finds only **weak,
  fragile** support that the metric ranks *which* players leave points: pooled within-role r ≈ **0.11**,
  one of two folds individually null, heavily attenuated from an in-sample 0.33. The robust claim stays at
  the *decision* level.

**Verdict.** Players systematically leave real points by declining good open looks — measurable,
reliable, box-invisible, and outcome-validated to cost points — while the intuitive "pass to the open
man" does not survive the same test. Modest magnitude, narrow scope, honest about
selection-on-unobservables. Report: [`reports/G6/`](reports/G6/README.md).

```bash
uv run --extra seq python scripts/build_g6.py                 # decomposition + outcome validity -> reports/G6/
```

## Live demo (Stage 7)

**Live: <https://plot-nba.vercel.app>** — a **chess.com-style possession review** in Next.js: an animated
court (team colors, player initials, ball trail), the **eval bar** (per-frame EPV) rising and falling,
clickable **decision badges** on each open ball-handler pass-up — great / good / inaccuracy / mistake /
blunder, tiered on *points left on the table* — and a "biggest misses" highlight reel. Six marquee games;
everything is precomputed static JSON, **no model runs in the browser**. See [`web/`](web/README.md).

```bash
uv run --extra seq python scripts/export_demo_json.py \
  --games 0021500336 0021500214 0021500013 0021500367 0021500308 0021500203  # -> web/public/demo/
cd web && npm install && npm run dev                           # http://localhost:3000
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
scripts/          # data download, possession build, gate builds (build_g3..g6), demo export
reports/          # one machine-readable report (JSON + figures) per gate (G1–G6)
web/              # Next.js demo app (deployed to Vercel)
paper/            # arXiv write-up (LaTeX) + figures
tests/            # pytest
```

## Acknowledgements / prior art

- **EPV** — Cervone, D'Amour, Bornn, Goldsberry (arXiv:1408.0777; JASA 2016).
- **VAEP** — Decroos, Bransen, Van Haaren, Davis (KDD 2019); `ML-KULeuven/socceraction`.
- Tracking data and animation reference: `dcayton/nba_tracking_data_15_16`,
  `linouk23/NBA-Player-Movements`.
