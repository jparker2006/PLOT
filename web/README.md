# PLOT demo — chess.com-style NBA possession review

A Next.js (App Router) app that plays precomputed per-possession JSON: an animated court, the
**eval bar** (per-frame EPV), and **decision badges** (regret at each open ball-handler pass-up).
No model runs in the browser — everything is static JSON in `public/demo/`.

## Run

```bash
# 1) export the demo data (from the repo root, needs the OOF EPV trace)
uv run --extra seq python scripts/export_demo_json.py --games 0021500308 0021500203

# 2) run the app
cd web
npm install
npm run dev          # http://localhost:3000
npm run build && npm run start   # production
```

`export_demo_json.py` writes `web/public/demo/<game_id>.json` + `index.json`. Each game JSON holds
the featured possessions (ranked by their single most-regretful decision), each with 10 fps canonical
tracking (ball + 10 players), the per-frame EPV eval bar, and the pass-up decisions tiered into
chess.com badges (great / good / inaccuracy / mistake / blunder) on *points left on the table*.

## Layout
- `app/` — App Router page + global styles.
- `components/Demo.jsx` — state, the rAF animation loop (interpolates between 10 fps keyframes),
  controls. `Court.jsx` (canvas court + players + ball), `EvalBar.jsx` (vertical EPV bar),
  `DecisionList.jsx` (scrubbable, clickable decisions).
- `lib/badges.js` — badge labels + the EPV→bar mapping (mirrors `src/plot/viz/demo_export.py`).
- `public/demo/` — the exported JSON (generated; committed so the static build/deploy is self-contained).

## Notes
- The eval trace is leakage-free + recalibrated; xPoints (behind regret) is fit on the featured
  games for the demo (the validated metric uses the full corpus — see the repo's G1–G5 reports).
- Deploys to any static host / Vercel free tier (`next build`).
