# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PLOT ("Points Left On the Table") is a solo research project shipping three things off one spine, built
on the only frozen public season of NBA SportVU tracking (2015-16): an open per-action **expected-points
value layer** (an "eval bar"), the flagship **PLOT regret metric** (best genuinely-available option −
chosen action, scored by *model expected value* not realized outcome), and a **chess.com-style Next.js
demo**. An arXiv paper draft lives in `paper/`. The thesis is that decision quality is a distinct,
measurable, box-score-invisible skill — and the project is deliberately **honesty-first**: every stage is
a go/no-go gate (G1–G6), and negative results (e.g. the "open-man myth") are reported as prominently as
positive ones. Match that skeptical, hedge-don't-inflate tone in any prose you write here.

## Environment & commands

- **Python is pinned to 3.12** (the system Python is newer and breaks the ML stack). **Always use `uv run …`** — never bare `python`.
- `uv sync` installs the base env; **`uv sync --extra seq`** adds PyTorch. The `seq` extra is required for anything that imports the sequence EPV model or the regret pipeline — i.e. `build_oof_epv.py`, `train_seq_epv_g1.py`, **and `build_g6.py`** (it pulls torch transitively via `models/regret/pipeline.py → models/eval_bar/seq_dataset.py`).
- macOS only: `brew install libomp` is needed for LightGBM.
- Tests: `uv run pytest` (all), `uv run pytest tests/test_outcome_validity.py` (one file), `uv run pytest tests/test_outcome_validity.py::test_real_cost_is_detected_and_gate_passes` (one test), `-k <pattern>` to filter. Config in `pyproject.toml` puts `src/` on the path and defaults to `-q`.
- Lint: `uv run ruff check .` / `uv run ruff check --fix .`. **line-length 120**; lint set `E,F,I,UP,B`. Keep it clean before committing.
- Data: `uv run python scripts/download_data.py --list` / `--tier {T0,T1,T2,T3}` / `--game <substr>`. See "Data" below.
- Paper: `cd paper && tectonic main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`). `main.pdf` is gitignored.

## The build pipeline (artifacts feed each other — this is the big picture)

Scripts in `scripts/` are run roughly in this order; each later stage consumes an earlier artifact, so a
change upstream means rebuilding downstream:

1. `download_data.py` → `data/raw/json/<game_id>.json` (extracted, **named by NBA game-id**) + `data/raw/2015-16_pbp.csv`.
2. `train_seq_epv_g1.py` → the sequence EPV model + `reports/G1_seq/` (baseline is `train_eval_bar_g1.py` → `reports/G1/`).
3. **`build_oof_epv.py --kfold K` → `data/processed/oof_epv_trace.parquet`** — the leakage-free, group-k-fold **out-of-fold EPV trace**. This is the **linchpin artifact**: the regret metric and every G3–G6 gate read it (they use the isotonic-recalibrated `epv_cal` column, aliased to `epv`). It is gitignored (large). Most "rebuild at scale" work is regenerating this trace.
4. `build_action_values.py` (Stage-3 ΔEPV decomposition), `train_xpoints_g2.py` (G2 counterfactual), `build_plot_g3.py` (G3), `build_g4.py`, `build_g5.py`, `build_g6.py` (G6) → each writes `reports/G<n>/` with a machine-readable JSON (carrying a `*_pass` / `gate` boolean) + figures.
5. `export_demo_json.py` → `web/public/demo/<game>.json` for the Next.js app in `web/` (precomputed JSON, **zero in-browser inference**).

`build_g6.py` is the current frontier and the most important entry point: it builds the **credible set**,
runs the decomposition, the outcome-validity keystone, the open-man asymmetry, and the player-level
cross-fit in one pass.

## Source layout (the parts that need cross-file context)

- `src/plot/io/` — `download.py` (fetch from public archives; `Game` refs + `select_games`), `loaders.py`.
- `src/plot/possessions/` — `segment.py` (possession segmentation; **offensive rebounds *continue* a possession**), `actions.py` (SPADL-analog action table from the ball-handler trace — coarse: an intermediate oreb'd shot reads as a `pass`), `shots.py` (`extract_shots`: one row per *real PBP* FG attempt, anchored on the shot event, attributed to the `PLAYER1` shooter).
- `src/plot/features/` — `court.py`, `orientation.py` (infer attacking basket + **quarantine** uncorroborated games), `clean.py`, `eval_bar.py`.
- `src/plot/models/eval_bar/` — baseline LightGBM (`model.py`, `crossval.py`) and the sequence EPV (`seq_model.py` = DeepSets set-encoder → **causal** GRU → multiclass head over points {0,1,2,3}; `seq_dataset.py`, `seq_crossval.py`). **`folds.py` exists specifically to keep the torch path from importing lightgbm** (see gotchas).
- `src/plot/models/counterfactual/` — `xpoints.py` (the "shoot now" shot model; population make model, **no shooter identity**), `regret.py` (v1 open pass-up regret), `regret_full.py` (v1.5/v2: + shot-selection / teammate options), `calibrate.py` (fold-safe isotonic EPV recalibration → `epv_cal`).
- `src/plot/models/regret/` — `plot_metric.py` (per-player aggregation + G3 stats; reusable helpers `_half_assignment`, `split_half_stability`, `_num`), `pipeline.py` (**the decision-frame builders** — see below).
- `src/plot/eval/` — one module per gate concept: `calibration.py` (G1), `box_stats.py` (G4), `validity.py` (G5), `decomposition.py` (G6 role/within-role), `outcome_validity.py` (G6 keystone + asymmetry + `player_cross_fit`).

### Decision frames (defined in `models/regret/pipeline.py`, consumed across G6)

The same loaded game intermediates produce three different decision tables — don't conflate them:
- **credible set** (`regret_from_intermediates`): open pass-ups + shot-selection. The PLOT metric and the role/within-role decomposition operate here (~153k decisions / 208 games).
- **open-look** (`open_looks_from_intermediates`): an open ball-handler who *took* (real PBP shot) vs *declined* (passed). The outcome-validity **keystone** — the one test against realized outcomes.
- **kick** (`open_kick_decisions_from_intermediates`): shot-over-a-wide-open-teammate vs kicked. The **asymmetry** test (does not validate — a defining negative).

## Non-obvious gotchas (each cost real time to find)

- **Never re-host raw tracking data.** `data/` is gitignored on purpose; the project ships a download script + `DATA.md`, not the data. Verify `git check-ignore data/raw/json` before any `git add`. (We download games locally to build, but they stay out of git.)
- **Never commit `.claude/`** — it holds machine memory and is gitignored.
- **LightGBM + PyTorch in one process segfaults** (EXIT 139) from dual OpenMP runtimes. The seq path is kept lightgbm-free via `models/eval_bar/folds.py`, and `tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` for the combined test process. Don't add a `import lightgbm` to any module the torch path imports.
- **Regret is scored by model expected value, NOT realized outcome** — this is the core design choice (measures decision quality, not luck). Don't "fix" it to use realized points.
- **Possession points are computed description-based** from the authoritative `PLAYER1_TEAM_ID`, never from the PBP `SCORE` column (which is orientation-inconsistent and occasionally non-monotonic). For steal turnovers, `PLAYER1` is the committer, `PLAYER2` the stealer.
- **Orientation quarantine trades coverage for correctness**: ~208 of ~636 available games pass QC and form the headline corpus. This is intentional; don't relax the single-confident-cell guard (it was validated to catch real wrong-flips).
- **Gate verdict helper `_num` (in `plot_metric.py`) is None-safe on purpose**: `(d.get(k) or default)` would swap the default in for a legitimately-falsy `0.0` (e.g. a p-value below machine epsilon), silently flipping a PASS to a FAIL. Use `_num`, not `or`.
- **Commit/push only on explicit per-turn user authorization** (pushing to `main` especially). Commit messages end with the `Co-Authored-By: Claude …` trailer.

## Corpora (which N is which)

Numbers in the paper/reports come from nested samples: ~636 available half-season games → a **10-game**
dev subset (G1/G2 calibration plumbing) → a **42-game** dev corpus (G5 method checks) → the **208 clean**
headline corpus (G3, G4, G6) → ~374–387 players with ≥30 decisions. When citing a statistic, check which
corpus its gate JSON used. The cross-fit (`reports/G6_local208/`) ran on 207/208 (one corrupt source
archive) and reproduces the canonical pod numbers; `reports/G6/` is the canonical 208.
