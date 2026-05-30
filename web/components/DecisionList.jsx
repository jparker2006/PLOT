"use client";
import { BADGE_LABEL } from "../lib/badges";

export default function DecisionList({ decisions, players, activeIdx, onPick }) {
  if (!decisions || decisions.length === 0) {
    return <div className="callout empty">No open pass-up decisions in this possession.</div>;
  }
  return (
    <div className="declist">
      {decisions.map((d, i) => {
        const nm = players?.[String(d.player)]?.name || `#${d.player}`;
        const left = d.regret_clipped;
        return (
          <div
            key={i}
            className={`decrow${i === activeIdx ? " active" : ""}`}
            onClick={() => onPick(d.frame, i)}
          >
            <span className={`badge ${d.badge}`}>{BADGE_LABEL[d.badge]}</span>
            <span className="who">
              {nm}
              <div className="sub">
                open shot {d.best_available.toFixed(2)} → pass {d.post_epv.toFixed(2)}
              </div>
            </span>
            <span className="pts">{left > 0 ? `−${left.toFixed(2)}` : "0.00"}</span>
          </div>
        );
      })}
    </div>
  );
}
