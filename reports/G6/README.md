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
