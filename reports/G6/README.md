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

**Method.** 15,287 open-handler decisions (42 games), each labelled `declined` (passed, 14,087) vs
`took` (terminal shot, 1,200), with the own open-shot model value `S` (xPoints at actual contest),
the REALIZED possession points `R`, and observable controls (model EPV at the decision, location,
openness) + team fixed effects. Inference by **game-cluster bootstrap**.
`src/plot/eval/outcome_validity.py` + `scripts/build_g6.py`; unit-tested in `tests/test_outcome_validity.py`.

**Result — the regret signal is outcome-valid at the decision level.**

| readout | value | reading |
|---|---|---|
| raw cost of declining, by look value `S` | +0.19 → **+0.57** as S: 0.82→1.64 | declining a *better* look costs *more* — the dose-response regret predicts |
| adjusted points lost by declining, mean S | **0.36** [0.28, 0.45], p≈0 | net of observables + team |
| adjusted points lost, high-value look (S≈1.58) | **0.50** [0.35, 0.67], p≈0 | passing up a great open look costs ~½ pt |
| `declined × S` interaction | **−0.35** [−0.67, −0.07] | cost grows with look value (CI excludes 0) |
| **robustness: shot-ending possessions only** (no TOs) | **0.49** [0.30, 0.69], survives ✓ | not turnover-exposure — a *worse downstream shot* |

**Why it's not an artifact.**
* **Dose-response.** The cost scales with the model's shot value `S` — exactly what regret predicts,
  and hard to produce with simple selection (a pure pool-composition confound would be ~flat in `S`).
* **Turnover-exposure ruled out.** Restricting to possessions that ended in a field-goal attempt
  (both groups same structural position) the cost barely moves (0.50→0.49, CI still excludes 0). So
  it isn't "passing risks turnovers" — declining a good look yields a genuinely *worse* shot later.
* **Against selection-on-unobservables.** If decliners passed because they saw a better play
  developing, their realized outcome should be *better*; it's *worse*. So on average these declines
  weren't justified by unobserved options (individual ones may be).

**Honest caveats (in the report).**
* **Magnitude is an upper bound.** Taker calibration shows realized > model value `S` (0.84→1.14, …)
  because possession-level `R` counts offensive-rebound putbacks and the terminal action is
  attributed coarsely to the tracking handler. The robust claims are the **sign** and the
  **dose-response**, not the exact 0.5-point figure.
* **Selection on unobservables** is *bounded* (we condition on the model's EPV at the decision +
  openness/location) but never fully eliminated — the standing G5b limitation.
* **Player-level (exploratory, stretch).** Per-player model PLOT and the within-role residual
  correlate with a player's realized decline-shortfall (r≈0.22 / **0.27**, p<0.001, n=239) — but
  `S−R` shares `S` with regret (partly mechanical) and 42 games is thin. Suggestive, not a gate. The
  cleaner player-level test (PLOT → team offense beyond box) is what step 3's scale-up would power.

**Verdict.** **G6 PASSES at the decision level** — the minimum bar for the paper's central claim.
*Players systematically leave points by declining open looks; the cost is measurable, scales with how
good the look was, survives removing turnover-risk, and is invisible to the box score.* The
player-level attribution and the exact magnitude are where more data (step 3) would help; the
decision-level finding is real and defensible now.

