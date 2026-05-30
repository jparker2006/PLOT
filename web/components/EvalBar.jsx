"use client";
import { epvToFraction } from "../lib/badges";

// cool (low) -> warm/green (high) for the offense's expected points
function color(epv) {
  const f = epvToFraction(epv);
  if (f < 0.5) return `rgb(${90 + f * 60}, ${110 + f * 80}, ${150 - f * 40})`;
  const g = (f - 0.5) * 2;
  return `rgb(${120 - g * 90}, ${150 + g * 50}, ${110 - g * 40})`;
}

export default function EvalBar({ epv }) {
  const pct = epvToFraction(epv) * 100;
  return (
    <div className="evalbar">
      <div className="val">{epv == null ? "—" : epv.toFixed(2)}</div>
      <div className="track">
        <div className="fill" style={{ height: `${pct}%`, background: color(epv) }} />
        <div className="mid" />
      </div>
      <div className="cap">EPV</div>
    </div>
  );
}
