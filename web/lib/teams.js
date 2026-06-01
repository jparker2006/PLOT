// Franchise-ish primaries, brightened for legibility on the dark court. Distinct within every
// curated matchup; unknown abbreviations fall back to a neutral slate.
export const NBA_COLORS = {
  ATL: "#E8565B", BOS: "#1AA06D", BKN: "#C9CED6", CHA: "#6E63D6", CHI: "#E1394A",
  CLE: "#C24A77", DAL: "#3D8FD6", DEN: "#4FB3D0", DET: "#E1556A", GSW: "#2E7BE0",
  HOU: "#E14655", IND: "#F2C94C", LAC: "#E1556A", LAL: "#9B6BD6", MEM: "#7C93D6",
  MIA: "#D14A6A", MIL: "#2FA060", MIN: "#5AA9E0", NOP: "#C9A24A", NYK: "#5AA9E0",
  OKC: "#E56020", ORL: "#3D9FD6", PHI: "#3D8FD6", PHX: "#E56020", POR: "#E1394A",
  SAC: "#9B5BD6", SAS: "#C9CED6", TOR: "#E1556A", UTA: "#5AA9E0", WAS: "#C9A24A",
};

export function teamColor(abbr) {
  return NBA_COLORS[abbr] || "#7f93a8";
}

// Black ink on light fills, white on dark — keeps player initials readable on any team color.
export function inkOn(hex) {
  if (!hex || hex[0] !== "#" || hex.length < 7) return "#ffffff";
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#0c1118" : "#ffffff";
}

export function initials(name) {
  if (!name) return "";
  const parts = name.replace(/[^A-Za-z .'-]/g, "").split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
