# Gate G3 — the PLOT metric is a stable skill, distinct from finishing

**Status: PASS (on 208 games).** Stage 5 turns per-decision regret into the flagship per-player
metric — *points left on the table* per 100 decisions — and G3 asks the two questions that decide
whether that metric is real: does it **repeat** (a skill, not noise), and is it **distinct from
finishing** (a decision property, not just shot-making or court location)? Both hold, and on the
expanded corpus the stability is markedly stronger than the first pass.

## Corpus
**208 clean games** (≈5× the original 42-game pass), the clean subset of the 391 games downloaded
for the scale-up run (the full 636-game public window was capped by a cloud disk quota; orientation
quarantine removed the rest). **124,953 open pass-up decisions**, 424 players, **380** clearing the
≥30-decision bar to enter the metric.

## What G3 checks
Per-decision regret (`xPoints(open shot here) − post-pass EPV`, on the leakage-free recalibrated
trace) is aggregated per player, then:
1. **G3(a) — stable.** Split the season's games odd/even, aggregate per player within each half,
   correlate. A skill repeats across independent samples; noise does not.
2. **G3(b-i) — decision, not location.** Nested OLS: `regret ~ shot-location controls`, then add
   **player fixed effects**; an F-test asks whether players differ beyond distance/openness/3-pt.
3. **G3(b-ii) — decision, not finishing.** xPoints uses a *population* make model (no shooter
   identity), so per-player regret should be ~uncorrelated with finishing skill.

## Results

| check | statistic | verdict |
|---|---|---|
| **G3(a) split-half** (clipped, n=347 in both halves) | Pearson **0.514**, Spearman 0.550, **Spearman–Brown full 0.679** | ✅ stable |
| &nbsp;&nbsp;— signed regret | Pearson 0.450, Spearman 0.462, SB-full 0.621 | ✅ |
| **G3(b-i) player FE beyond location** | F(380, 124568) = **4.96, p≈0**; R² 0.084 → 0.098 (incremental **0.014**) | ✅ not just location |
| **G3(b-ii) regret vs finishing** (n=351) | Pearson **−0.173** (p=0.001), Spearman −0.189 | ✅ not finishing (\|r\|<0.40) |

![split-half stability](g3_split_half_scatter.png)
![per-player PLOT distribution](g3_plot_distribution.png)

**Scaling tightened the headline.** Split-half reliability rose from the 42-game pass (Pearson
**0.38**, SB 0.55) to **0.51 / SB 0.68** here — the soft early number became a solid one, which was
the whole reason to scale. Player identity still explains regret beyond a full shot-location model
(F=4.96), and regret remains well short of being a finishing proxy.

## The metric (v1 narrow)
`PLOT = points left on the table / 100 decisions`, at open ball-handler **pass** decisions: mean
clipped **0.143 pts/decision (14.3/100)**, mean signed −0.068, 38% of decisions positive, p90 clipped
0.71. Per-player table at `data/processed/plot_per_player.parquet`.

## Honest limitations
- **v1 evaluates ONE decision type** — an *open* handler (nearest defender ≥ 4 ft) passing up the
  shot. Drives and open-teammate passes are not scored yet, so the headline is "value left by passing
  up open looks," not all decisions.
- **Regret *magnitudes* are not directly comparable to the 42-game report.** The recalibration was
  changed from leave-one-game-out to **group k-fold + subsampled** isotonic (an O(games²)→O(folds)
  fix required to run at scale), which shifts the `epv_cal` values that feed regret. The G3
  *conclusions* (stability, distinctness) are robust; the raw means moved.
- **Finishing is now weakly *negatively* correlated** (−0.17, p=0.001; ~3% shared variance): better
  finishers leave slightly less on the table. It clears the "not finishing" bar (\|r\|<0.40) by a wide
  margin — regret is not a relabeled finishing metric — but at this sample size it is no longer
  perfectly orthogonal, and the paper should report it as such rather than claim independence.
- **Decision-level incremental R² is small (1.4%)** — single decisions are noisy; the signal lives at
  the player-average level, which is exactly what the split-half correlation (0.51) measures.

## Reproduce
```bash
uv sync --extra seq
uv run --extra seq python scripts/build_oof_epv.py --kfold 5   # leakage-free OOF EPV trace
uv run --extra seq python scripts/build_plot_g3.py             # -> this report
uv run python -m pytest tests/test_plot_metric.py -q           # aggregation / stability / FE / gate
```
Artifacts: `g3_stability.json` (distribution, leaderboard, the three G3 blocks, the gate),
`g3_split_half_scatter.png`, `g3_plot_distribution.png`, and `data/processed/plot_per_player.parquet`
(gitignored). The 208-game run was executed on a rented GPU; the OOF trace + corpus load are now
**memory-bounded at any scale** (peak ~5GB for 208 games) after fixing a numpy-view retention leak in
the sequence cache loader.
