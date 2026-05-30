// chess.com-style decision badges, mirroring src/plot/viz/demo_export.py.
export const BADGE_LABEL = {
  great: "Great",
  good: "Good",
  inaccuracy: "Inaccuracy",
  mistake: "Mistake",
  blunder: "Blunder",
};

export const BADGE_BLURB = {
  great: "passed up the shot for a clearly better option",
  good: "passing was about as good as the open shot",
  inaccuracy: "a fine open look, slightly under-used",
  mistake: "left real points on the table",
  blunder: "passed up a high-value open shot",
};

// EPV -> eval-bar fill fraction. Possession value runs ~0..~2.2; clamp for the bar.
export function epvToFraction(epv) {
  if (epv == null) return 0;
  return Math.max(0, Math.min(1, epv / 2.0));
}
