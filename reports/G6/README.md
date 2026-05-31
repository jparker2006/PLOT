# G6 — "actually-useful": outcome validity + decision-quality decomposition

G1–G5 proved PLOT is *internally* sound (calibrated, stable, box-orthogonal, valid-conditional-on-observables).
G6 asks the two questions that decide whether the paper says something **actually true and useful**:

1. **Role-of-touch vs within-role decision quality** — is PLOT measuring *who decides well*, or just
   *what kind of touches a player gets*? (step 1)
2. **Outcome validity** — does the "points left" signal predict *real* outcomes, not just model-vs-model?
   (steps 2 / 2b)

**All numbers below are the canonical 208-game results** (the full local corpus; 153,098 decisions).
A 42-game pilot was run first; where the scaled estimate revised it materially (the keystone
magnitude), that is called out — the 42-game figure was small-sample-optimistic.

---

## Step 1 — role vs within-role decomposition (`g6_step1.json`)

**Question.** Every PLOT variant (v1/v1.5/v2) put bigs at the top: rim-catch players "leave points"
because their *role* sits them where the open-shot counterfactual is high. Is the reliable signal
just role-of-touch, or is there real within-role decision skill underneath?

**Method.** Build the v1.5 credible decision set (open pass-ups + shooting over a wide-open teammate;
**153,098 decisions, 387 players ≥30 decisions, 208 games**). Fit a **context model** `regret ~
touch-context` (court location, openness, rim distance, 3-pt, decision kind) with **no player
identity**, out-of-fold by game. The OOF prediction is the **role component** (regret anyone faces in
that spot); the residual is the **within-role component** (how much *this* player beats/trails the
situation's expectation). Then decompose between-player variance and re-measure split-half
reliability. `src/plot/eval/decomposition.py` + `scripts/build_g6.py`; tested in `tests/test_decomposition.py`.

**Result — within-role decision quality is the LARGER, reliable component.** At scale the "it's just
role-of-touch" worry is decisively dead.

| quantity | clipped | reading |
|---|---|---|
| `role_share` (Var of role component / Var raw) | **0.29** | role is now the *smaller* slice |
| `within_role_share` | **0.52** | within-role decision quality is the majority of between-player variance |
| `cross_share` (2·Cov) | 0.19 | mild positive overlap |
| R² role reconstructs leaderboard | **0.51** | touch profile alone recovers ~half the ordering |
| split-half reliability, **raw** (Spearman-Brown) | **0.73** | the PLOT reliability at scale |
| split-half reliability, **within-role residual** | **0.54** (p≈0) | a reliable signal *survives* removing role — and it strengthened with data (was 0.44 at 42g) |

**Leakage robustness (the key check).** If the residual's reliability were just *unabsorbed* role
leaking back in, a stronger role model would eat it and the residual reliability would collapse. It
doesn't — as the context model goes weak→default→strong (depth 2→5, 100→400 trees), `role_share`
rises then **plateaus** (0.15→0.29→0.32) while residual reliability stays **stable** (0.60→0.54→0.52).
Genuine within-role signal, not an artifact.

**Leaderboard (the gut check).** Top by **raw** PLOT and by **within-role residual** are both
big-heavy (Brandan Wright, Willie Reed, Montrezl Harrell), but the within-role view surfaces wings
the raw metric under-weights (Caron Butler, Luc Mbah a Moute) — players who leave points *relative to
their own context*. With scale the two orderings overlap more, consistent with within-role now being
the dominant component.

**Verdict / what it means for the paper.** The early "role-dominated" read is overturned by the data:
role explains ~29% of between-player variance, within-role decision quality ~52%, and the within-role
residual is reliable (SB 0.54) and capacity-robust. The honest framing is *"PLOT's between-player
spread is mostly a real, role-adjusted decision-quality signal — here it is, isolated"* (per player in
`data/processed/plot_within_role.parquet`, gitignored). That residual is the target for step 2.

---

## Step 2 — outcome validity: does "points left" cost REAL points? (`g6_step2.json`)

**The keystone.** Everything through step 1 is internal (model-vs-model). This is the one test that
touches reality. The cleanest decision the metric makes is the **open look**: a ball-handler with
nobody within 4 ft either takes the shot or declines it (passes). We observe REALIZED possession
points for *both* choices, so the counterfactual "what taking would have yielded" comes from real
shooters at matched situations, not the model.

**Method (hardened).** **79,877 open-offensive-player decisions (208 games)**: `declined` (open
handler passed, 68,512) vs `took` (11,365). **Takers are the real PBP shots** (`extract_shots` → the
`PLAYER1` shooter at release), so the decision is attributed to the player who actually shot — not the
coarse tracking ball-handler at possession end — and the shot's own points `fg_points` calibrate the
benchmark cleanly. Benchmark `S` = own open-shot xPoints (same model both sides). Outcome `R` =
realized possession points. Controls: model EPV at the decision, location, openness, **seconds into
the possession** (shot-clock-pressure proxy — also corrects for takers acting later than decliners),
period, + team fixed effects. Inference by **game-cluster bootstrap**. `src/plot/eval/outcome_validity.py`
+ `src/plot/models/regret/pipeline.py` + `scripts/build_g6.py`; tested in `tests/test_outcome_validity.py`.

**Result — the regret signal is outcome-valid at the decision level.**

| readout | value | reading |
|---|---|---|
| taker calibration, `S` → own-shot points | 0.75→0.77, 0.91→0.96, 1.17→1.37 … | `S` is a fair benchmark (≈ diagonal once shooter-attributed) |
| raw cost of declining, by look value `S` | **−0.005** (bad look) → **+0.42** (great look) | declining a *bad* look is free; a *good* look is costly — exactly regret's logic |
| adjusted points lost, average look | **0.11** [0.086, 0.137], p≈0 | net of observables + clock + team |
| adjusted points lost, high-value look (S≈1.45) | **0.19** [0.14, 0.25], p≈0 | passing up a great open look costs ~⅕ pt |
| `declined × S` interaction | **−0.23** [−0.35, −0.12] | cost grows with look value (CI excludes 0) |
| **robustness: shot-ending possessions only** (no TOs) | **0.21** [0.15, 0.28], survives ✓ | not turnover-exposure — a *worse downstream shot* |

> **Magnitude revision (42g → 208g).** The pilot put the high-value cost at ~0.47; the well-powered
> 208-game estimate is **~0.19** (tight CI). The 42-game figure was small-sample-inflated. The *sign,
> dose-response, and every robustness cut hold* — the effect is real and modest, not large.

**Why it's not an artifact.**
* **Dose-response.** The cost scales with `S`: declining a *low-value* look costs ≈0 (−0.005); the cost
  only appears for good looks. A pool-composition confound would be ~flat in `S`; this isn't.
* **Benchmark validated.** With proper shooter attribution, takers realize ≈ `S` in their own shot
  points — so `S` is a fair yardstick and the magnitude is not just an upper bound.
* **Turnover-exposure ruled out.** Shot-ending-only (both groups same structural position): 0.21, CI
  excludes 0. Declining a good look yields a genuinely *worse* shot, not just turnover risk.
* **Against selection-on-unobservables.** If decliners passed because they saw a better play, their
  realized outcome should be *better*; it's *worse*. So on average these declines weren't justified by
  unobserved options (individual ones may be) — bounded, not eliminated (the standing G5b limitation).
* **Not a segmentation artifact.** Dropping every decliner whose handler took a PBP shot that
  possession (19,466 rows, ~28% — the mis-typed-missed-shot guard) barely moves the estimate — 0.13
  [0.11, 0.16] mean / 0.21 [0.17, 0.27] high-value, survives — so the result is not driven by
  action-layer mis-typing, and the full action-layer refactor is unnecessary for this claim.

**Player level — now genuinely powered.** Per-player within-role residual correlates **r=0.33** (p≈0,
**n=373**) with a player's realized decline-shortfall (model PLOT r=0.16); both up from the 42-game
pilot. `S−R` shares `S` with regret (partly mechanical), so this is exploratory — the cross-fit
de-circularization is the rigorous version — but it is no longer underpowered.

**Verdict.** **G6 PASSES at the decision level**: *players systematically leave points by declining
good open looks — ≈0.11 on an average look, ≈0.19 on a great one; declining a bad look is free; the
effect is dose-responsive, survives removing turnover-risk, clock pressure, team quality, and
action-layer mis-typing; and it is invisible to the box score.* Modest magnitude, well powered, honest.

---

## Step 2b — the SECOND decision type does NOT validate (a defining negative; `g6_step2b.json`)

Symmetric test for the other half of the credible set — **shooting over a wide-open teammate**. When a
handler had a wide-open (≥6 ft), frontcourt, reachable teammate, does *shooting over him* cost points
vs *kicking* to him? Benchmark `S` = the kick's value (0.80 × best open-teammate xPoints); `declined`
≡ shot over the open man; same controls (+ the handler's own look) + team FE + game-cluster bootstrap.
**56,785 decisions (14,761 shot over, 42,024 kicked).**

**Result — it does not validate, and the opposite is true** (and the 208-game CIs are tight).

| readout | value | reading |
|---|---|---|
| kick calibration `S` → realized | 0.77→1.02, 1.24→0.98, 1.35→0.99 (**flat ~1.0**) | the teammate-kick value `S` does NOT predict what the kick actually yields |
| raw cost of shooting over, by `S` | **all negative** (−0.29 … −0.33) | shooting over the open man realizes *more*, not fewer, points |
| adjusted points lost by shooting over | **−0.22** [−0.26, −0.19] | shooting over is associated with ~0.22 *more* realized points |
| `declined × S` interaction | +0.03 [−0.05, +0.12] | no dose-response |
| shot-ending robustness | −0.17, **survives = False** | — |
| **gate** | **FAIL** | correctly — no positive cost |

**Why this is one of the most valuable results in the project.**
* **It explains the v2 failure rigorously.** v2 collapsed into a role-of-touch sort because the "pass
  to the open teammate" counterfactual was unreliable. We diagnosed that by *face validity* before;
  now **outcome validity confirms it** — the teammate-kick benchmark is flat in `S`. An "open" man is
  often open *because* he is not a threat, so the kick doesn't deliver its modeled value.
* **It sharpens the paper's claim to exactly what is true.** PLOT is outcome-valid *only* for the
  decision whose counterfactual is directly observable — **declining your own open look** (its value
  is the shooter's own calibrated xPoints). It is *not* valid for the "should've passed to the open
  man" read. That asymmetry **contradicts conventional basketball wisdom**.
* **It refutes "score more decisions" with data, not opinion.** Breadth doesn't validate; realness
  comes from **scale + a clean foundation + the one validated decision**, not from adding decision types.

**Caveat.** "Kicked to the open man" is identified via the next ball-handler being a wide-open teammate
(pass recipient from action order). Whether the negative means "the read genuinely isn't a mistake" or
"the teammate counterfactual is too crude to score," both lead to the same scope: **score and claim
only the own-open-look decision.**
