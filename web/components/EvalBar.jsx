"use client";
import { epvToFraction } from "../lib/badges";

export default function EvalBar({ epv }) {
  const pct = epvToFraction(epv) * 100;
  return (
    <div className="evalbar">
      <div className="val">{epv == null ? "—" : epv.toFixed(2)}</div>
      <div className="track" title="expected points (EPV); dashed line = 1.0">
        <div className="fill" style={{ height: `${pct}%` }} />
        {/* scale: bar runs EPV 0..2.0; faint ticks at 0.5 / 1.5, dashed reference at 1.0 */}
        <div className="tick" style={{ bottom: "25%" }} />
        <div className="tick ref" style={{ bottom: "50%" }} />
        <div className="tick" style={{ bottom: "75%" }} />
      </div>
      <div className="cap">EPV</div>
    </div>
  );
}
