# G6 — "actually-useful": outcome validity + decision-quality decomposition

G1–G5 proved PLOT is *internally* sound (calibrated, stable, box-orthogonal, valid-conditional-on-observables).
G6 asks the two questions that decide whether the paper says something **actually true and useful**:

1. **Role-of-touch vs within-role decision quality** — is PLOT measuring *who decides well*, or just
   *what kind of touches a player gets*? (step 1, below — **done**)
2. **Outcome validity** — does the "points left" signal predict *real* outcomes, not just model-vs-model?
   (step 2 — next)

---

## Step 1 — role vs within-role decomposition (`g6_step1.json`)

**Question.** Every PLOT variant (v1/v1.5/v2) put bigs at the top: rim-catch players "leave points"
because their *role* sits them where the open-shot counterfactual is high. Is the reliable signal
just role-of-touch, or is there real within-role decision skill underneath?

**Method.** Build the v1.5 credible decision set (open pass-ups + shooting over a wide-open teammate;
30,195 decisions, 271 players ≥30 decisions, 42 games). Fit a **context model** `regret ~ touch-context`
(court location, openness, rim distance, 3-pt, decision kind) with **no player identity**,
out-of-fold by game. The OOF prediction is the **role component** (regret anyone faces in that spot);
the residual is the **within-role component** (how much *this* player beats/trails the situation's
expectation). Then decompose between-player variance and re-measure split-half reliability.
`src/plot/eval/decomposition.py` + `scripts/build_g6.py`; unit-tested in `tests/test_decomposition.py`.

**Result.** PLOT is substantially — but **not exclusively** — role-of-touch, and a reliable
within-role decision signal survives.

| quantity | clipped | reading |
|---|---|---|
| `role_share` (Var of role component / Var raw) | **0.41** | role is the single largest factor… |
| `within_role_share` | **0.38** | …but within-role is comparably large |
| `cross_share` (2·Cov) | 0.21 | players in high-regret roles *also* trail within-role |
| R² role reconstructs leaderboard | **0.64** | touch profile alone recovers ⅔ of the ordering |
| split-half reliability, **raw** (Spearman-Brown) | **0.79** | the known PLOT reliability |
| split-half reliability, **within-role residual** | **0.44** (p<0.001) | a reliable signal *survives* removing role |

**Leakage robustness (the key check).** If the residual's reliability were just *unabsorbed* role
leaking back in, a stronger role model would eat it and the residual reliability would collapse. It
doesn't — as the context model goes weak→default→strong (depth 2→5, 100→400 trees), `role_share`
rises then **plateaus** (0.35→0.41→0.41) and residual reliability stays **stable** (0.48→0.44→0.44).
The within-role signal is robust to context-model capacity → genuine, not an artifact.

**Leaderboard reshuffle (the gut check).** By **raw** PLOT the top 15 are all bigs (Splitter, Henson,
Hickson, Gortat, Adams…). By **within-role residual**, perimeter players the raw metric buried surface
— **Seth Curry** (role 5.4, within-role 10.2), Robert Covington, Doug McDermott, Jabari Parker, Ramon
Sessions — players who leave points *relative to their own context*, which raw PLOT can't see.

**Verdict / what it means for the paper.** The earlier "role-dominated" read was directionally right
but quantitatively an overstatement: role is the biggest single factor (~40% of variance, ⅔ of the
ordering), yet a comparably-sized, reliable, capacity-robust within-role decision signal exists. That
within-role residual (per player in `data/processed/plot_within_role.parquet`, gitignored) is the
**right target for step 2's outcome validation** — and the honest framing for the paper is *"PLOT
blends role-of-touch with a real, role-adjusted decision-quality signal; here is that signal,
isolated."*

**Caveat.** 42-game corpus; the residual reliability (0.44) is solid but lower than raw (0.79), so the
within-role number is the less-powered one — a reason step 3 (scale to 208) exists if step 2 needs it.

---

## Step 2 — outcome validity: does "points left" cost REAL points? (`g6_step2.json`)

**The keystone.** Everything through step 1 is internal (model-vs-model). This is the one test that
touches reality. The cleanest decision the metric makes is the **open look**: a ball-handler with
nobody within 4 ft either takes the shot or declines it (passes). We observe REALIZED possession
points for *both* choices, so the counterfactual "what taking would have yielded" comes from real
shooters at matched situations, not the model.

**Method (hardened).** 16,448 open-offensive-player decisions (42 games): `declined` (open handler
passed, 14,087) vs `took` (2,361). **Takers are the real PBP shots** (`extract_shots` → the
`PLAYER1` shooter at release), so the decision is attributed to the player who actually shot — not
the coarse tracking ball-handler at possession end — and the shot's own points `fg_points` calibrate
the benchmark cleanly. Benchmark `S` = own open-shot xPoints (same model both sides). Outcome `R` =
realized possession points. Controls: model EPV at the decision, location, openness, **seconds into
the possession** (shot-clock-pressure proxy — also corrects for takers acting later than decliners),
period, + team fixed effects. Inference by **game-cluster bootstrap**. `src/plot/eval/outcome_validity.py`
+ `src/plot/models/regret/pipeline.py` + `scripts/build_g6.py`; unit-tested in `tests/test_outcome_validity.py`.

**Result — the regret signal is outcome-valid at the decision level.**

| readout | value | reading |
|---|---|---|
| taker calibration, `S` → own-shot points | 0.76→0.72, 0.94→0.98, 1.21→1.55 … | `S` is a fair benchmark (≈ diagonal once shooter-attributed) |
| raw cost of declining, by look value `S` | **−0.06** (bad look) → **+0.53** (great look) | declining a *bad* look is free; a *good* look is costly — exactly regret's logic |
| adjusted points lost, average look | **0.22** [0.17, 0.28], p≈0 | net of observables + clock + team |
| adjusted points lost, high-value look (S≈1.55) | **0.47** [0.34, 0.62], p≈0 | passing up a great open look costs ~½ pt |
| `declined × S` interaction | **−0.62** [−0.89, −0.37] | cost grows sharply with look value (CI excludes 0) |
| **robustness: shot-ending possessions only** (no TOs) | **0.47** [0.34, 0.62], survives ✓ | not turnover-exposure — a *worse downstream shot* |

**Why it's not an artifact.**
* **Dose-response.** The cost scales with `S` — exactly what regret predicts. The hardened cut makes
  it cleaner still: declining a *low-value* look costs ≈0 (−0.06), the cost only appears for good
  looks. A pool-composition confound would be ~flat in `S`; this isn't.
* **Benchmark validated.** With proper shooter attribution, takers realize ≈ `S` in their own shot
  points (the earlier "realized > S" anomaly was offensive-rebound putbacks in possession `R` + coarse
  attribution — both now addressed). So `S` is a fair yardstick, and the magnitude is no longer just
  an upper bound.
* **Clock & timing controlled.** Adding seconds-into-possession (a forced late pass is not a free
  choice; takers shoot later than decliners) *lowered* the average cost 0.36→0.22 — the honest number
  — while the high-value cost held (~0.47) and the dose-response sharpened.
* **Turnover-exposure ruled out.** Shot-ending-only (both groups same structural position): 0.47, CI
  excludes 0. Declining a good look yields a genuinely *worse* shot, not just turnover risk.
* **Against selection-on-unobservables.** If decliners passed because they saw a better play, their
  realized outcome should be *better*; it's *worse*. So on average these declines weren't justified by
  unobserved options (individual ones may be) — bounded, not eliminated (the standing G5b limitation).

**Still soft — the player level.** Per-player model PLOT / within-role residual correlate with a
player's realized decline-shortfall (r≈0.22 / **0.27**, p<0.001, n=239), but `S−R` shares `S` with
regret (partly mechanical) and 42 games is thin. Suggestive, not a gate. The cleaner player-level
test (PLOT → team offense beyond box) is what the **208-game scale-up** would power.

**Verdict.** **G6 PASSES at the decision level**, and the hardened analysis strengthens it: *players
systematically leave points by declining good open looks — ≈0.22 on an average look, ≈0.47 on a great
one; declining a bad look is free; the effect survives removing turnover-risk, clock pressure, and
team quality; and it is invisible to the box score.* The exact player-level attribution is where the
208-game run comes in; the decision-level finding is real and defensible now.

---

## Step 2b — the SECOND decision type does NOT validate (a defining negative; `g6_step2b.json`)

Symmetric test for the other half of the credible set — **shooting over a wide-open teammate**. When a
handler had a wide-open (≥6 ft), frontcourt, reachable teammate, does *shooting over him* cost points
vs *kicking* to him? Benchmark `S` = the kick's value (0.80 × best open-teammate xPoints); `declined`
≡ shot over the open man; same controls (+ the handler's own look) + team FE + game-cluster bootstrap.
11,572 decisions (2,982 shot over, 8,590 kicked).

**Result — it does not validate, and the opposite is true.**

| readout | value | reading |
|---|---|---|
| kick calibration `S` → realized | 0.77→0.98, 1.15→0.99, 1.46→1.00 (**flat ~1.0**) | the teammate-kick value `S` does NOT predict what the kick actually yields |
| raw cost of shooting over, by `S` | **all negative** (−0.40 … −0.42) | shooting over the open man realizes *more*, not fewer, points |
| adjusted points lost by shooting over | **−0.35** [−0.44, −0.27] | shooting over is associated with ~0.35 *more* realized points |
| `declined × S` interaction | +0.09 [−0.04, +0.25] | no dose-response | 
| shot-ending robustness | −0.31, **survives = False** | — |
| **gate** | **FAIL** | correctly — no positive cost |

**Why this is one of the most valuable results in the project.**
* **It explains the v2 failure rigorously.** v2 (score every decision against the full available set)
  collapsed into a role-of-touch sort because the "pass to the open teammate" counterfactual was
  unreliable. We diagnosed that by *face validity* before; now **outcome validity confirms it** — the
  teammate-kick benchmark is flat in `S` (doesn't predict reality). An "open" man is often open
  *because* he is not a threat, so the kick doesn't deliver its modeled value.
* **It sharpens the paper's claim to exactly what is true.** PLOT is outcome-valid *only* for the
  decision whose counterfactual is directly observable — **declining your own open look** (its value
  is the shooter's own calibrated xPoints). It is *not* valid for the "should've passed to the open
  man" read. That asymmetry is itself a finding that **contradicts conventional basketball wisdom**.
* **It refutes "V2 = score more decisions" with data, not opinion.** Breadth doesn't validate; realness
  comes from **scale + clean foundation + the one validated decision**, not from adding decision types.

**Caveat.** "Kicked to the open man" is identified via the next ball-handler being a wide-open teammate
(pass recipient from action order); the planned action-layer oreb/shot fix will sharpen this, but is
unlikely to flip a result this clear. Whether the negative means "the read genuinely isn't a mistake"
or "the teammate counterfactual is too crude to score" — both lead to the same scope: **score and
claim only the own-open-look decision.**

