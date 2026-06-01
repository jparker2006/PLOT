# Paper — *Points Left on the Table*

arXiv preprint draft for the PLOT project: an open per-action expected-points value layer and an
outcome-validated decision-regret metric for NBA tracking data.

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# or, if you have it:  latexmk -pdf main.tex
# or, hermetic:        tectonic main.tex
```

Outputs `main.pdf`.

## Layout

- `main.tex` — the full draft (single file).
- `refs.bib` — bibliography.
- `figures/` — figures copied from the gate reports (`reports/G*/*.png`). Regenerate the sources by
  re-running the gate scripts (`scripts/build_g6.py`, etc.) and re-copy.

## Status / what's deferred

Every number in the draft is taken from the gated reproduction pipeline:

- **G1** value-layer calibration — 10-game LOGO development subset.
- **G2** counterfactual-input calibration.
- **G3 / G4** stability, distinctness, incrementality — **208 games**.
- **G5** validity (method properties) — 42-game development corpus.
- **G6.1 / G6.2 / G6.2b** decomposition, the open-look keystone, and the open-man asymmetry — **208 games**.

The **one** deferred item is **§8.4, the player-level cross-fit at 208 games** — marked with a red
`\TODO{}` in `main.tex`. The method, design, and gate are final and the code is committed; only the
final magnitude/CI and the de-circularized scatter figure await a full-corpus run (blocked on GPU
availability). A preliminary 87-game local run validates it (pooled within-role $r=0.189$, $p=6.3\times
10^{-5}$); see the LaTeX comment under that `\TODO` for the provisional numbers. Swap them in and drop
the `\TODO` once the 208 run lands.
