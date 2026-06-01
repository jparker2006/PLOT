"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Court from "./Court";
import DecisionList from "./DecisionList";
import EvalBar from "./EvalBar";
import { BADGE_BLURB, BADGE_LABEL } from "../lib/badges";

const FPS = 10;

function interpFrame(frames, t) {
  if (!frames || frames.length === 0) return null;
  const x = Math.max(0, Math.min(frames.length - 1, t * FPS));
  const i0 = Math.floor(x);
  const i1 = Math.min(frames.length - 1, i0 + 1);
  const f = x - i0;
  const a = frames[i0];
  const b = frames[i1];
  const L = (p, q) => p + (q - p) * f;
  const ball = a.ball && b.ball ? [L(a.ball[0], b.ball[0]), L(a.ball[1], b.ball[1]), L(a.ball[2] || 0, b.ball[2] || 0)] : a.ball;
  const bmap = new Map((b.pl || []).map((p) => [p[0], p]));
  const pl = (a.pl || []).map((p) => {
    const bp = bmap.get(p[0]) || p;
    return [p[0], p[1], L(p[2], bp[2]), L(p[3], bp[3])];
  });
  const epv = a.epv != null && b.epv != null ? L(a.epv, b.epv) : a.epv;
  return { ball, pl, epv, gc: a.gc };
}

export default function Demo() {
  const [index, setIndex] = useState([]);
  const [gameId, setGameId] = useState(null);
  const [data, setData] = useState(null);
  const [possIdx, setPossIdx] = useState(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const raf = useRef(0);
  const last = useRef(0);

  useEffect(() => {
    fetch("/demo/index.json").then((r) => r.json()).then((idx) => {
      setIndex(idx);
      if (idx.length) setGameId(idx[0].game_id);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!gameId) return;
    setData(null);
    fetch(`/demo/${gameId}.json`).then((r) => r.json()).then((d) => {
      setData(d);
      setPossIdx(0);
      setT(0);
      setPlaying(false);
    }).catch(() => {});
  }, [gameId]);

  const poss = data?.possessions?.[possIdx] || null;
  const duration = poss ? (poss.n_frames - 1) / FPS : 0;

  // animation loop
  useEffect(() => {
    if (!playing || !poss) return;
    last.current = 0;
    const step = (now) => {
      if (!last.current) last.current = now;
      const dt = (now - last.current) / 1000;
      last.current = now;
      setT((prev) => {
        const next = prev + dt;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, poss, duration]);

  const frame = useMemo(() => interpFrame(poss?.frames, t), [poss, t]);
  const curIdx = Math.round(t * FPS);

  // active decision: the nearest pass-up within a short window of the playhead
  const activeIdx = useMemo(() => {
    if (!poss?.decisions?.length) return -1;
    let best = -1, bestd = 6;
    poss.decisions.forEach((d, i) => {
      const dd = Math.abs(d.frame - curIdx);
      if (dd <= bestd) { bestd = dd; best = i; }
    });
    return best;
  }, [poss, curIdx]);
  const activeDec = activeIdx >= 0 ? poss.decisions[activeIdx] : null;
  const highlight = activeDec ? activeDec.player : null;

  const possPointsLeft = useMemo(
    () => (poss?.decisions || []).reduce((s, d) => s + Math.max(0, d.regret_clipped), 0),
    [poss]
  );
  // game-level highlight reel: the most-regretful decisions across all featured possessions
  const topMoments = useMemo(() => {
    if (!data?.possessions) return [];
    const all = [];
    data.possessions.forEach((p, pi) =>
      (p.decisions || []).forEach((d) => { if (d.regret_clipped > 0) all.push({ pi, ...d }); })
    );
    return all.sort((a, b) => b.regret_clipped - a.regret_clipped).slice(0, 6);
  }, [data]);

  const pick = useCallback((frameIdx) => {
    setPlaying(false);
    setT(frameIdx / FPS);
  }, []);
  const jumpTo = useCallback((pi, frameIdx) => {
    setPlaying(false);
    setPossIdx(pi);
    setT(frameIdx / FPS);
  }, []);

  if (!data) {
    return <div className="panel" style={{ marginTop: 16 }}>Loading possessions…</div>;
  }

  const offAbbr = data.team_abbr?.[String(poss?.offense_team_id)] || "OFF";
  const outcomePts = poss ? `${poss.points} pt${poss.points === 1 ? "" : "s"}` : "";

  return (
    <>
      <div className="controls">
        <select value={gameId || ""} onChange={(e) => setGameId(e.target.value)}>
          {index.map((g) => (
            <option key={g.game_id} value={g.game_id}>{g.matchup}</option>
          ))}
        </select>
        <select value={possIdx} onChange={(e) => { setPossIdx(+e.target.value); setT(0); setPlaying(false); }}>
          {data.possessions.map((p, i) => (
            <option key={i} value={i}>
              Possession {i + 1} — {data.team_abbr?.[String(p.offense_team_id)] || "?"} ({p.points} pt)
            </option>
          ))}
        </select>
        <button onClick={() => { if (t >= duration) setT(0); setPlaying((p) => !p); }} className="play">
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        <button onClick={() => { setPossIdx((i) => Math.max(0, i - 1)); setT(0); setPlaying(false); }}>‹ Prev</button>
        <button onClick={() => { setPossIdx((i) => Math.min(data.possessions.length - 1, i + 1)); setT(0); setPlaying(false); }}>Next ›</button>
      </div>

      <div className="stage">
        <div className="courtcard">
          <div className="courtrow">
            <EvalBar epv={frame?.epv} />
            <Court frame={frame} teams={data.teams} highlight={highlight}
                   players={data.players} teamAbbr={data.team_abbr} />
          </div>
          <div className="scrub">
            <div className="scrubwrap">
              <input
                type="range" min={0} max={duration || 0} step={0.05} value={t}
                onChange={(e) => { setPlaying(false); setT(+e.target.value); }}
              />
              {duration > 0 && (poss?.decisions || []).map((d, i) => (
                <button
                  key={i} className={`tickmark ${d.badge}`}
                  style={{ left: `${Math.max(0, Math.min(100, (d.frame / FPS / duration) * 100))}%` }}
                  title={`${BADGE_LABEL[d.badge]} — ${data.players?.[String(d.player)]?.name || "?"}`}
                  onClick={() => pick(d.frame)}
                />
              ))}
            </div>
            <div className="clock">
              <span>game clock {frame?.gc != null ? frame.gc.toFixed(1) : "—"}s</span>
              <span>{t.toFixed(1)} / {duration.toFixed(1)}s</span>
            </div>
          </div>
          <div className={`callout${activeDec ? "" : " empty"}`}>
            {activeDec ? (
              <>
                <div className="cline">
                  <span className={`badge ${activeDec.badge}`}>{BADGE_LABEL[activeDec.badge]}</span>{" "}
                  <b>{data.players?.[String(activeDec.player)]?.name || "?"}</b> {BADGE_BLURB[activeDec.badge]}.
                </div>
                {(() => {
                  const lo = Math.max(activeDec.best_available, activeDec.post_epv, 0.1);
                  const w = (v) => `${Math.max(3, (v / lo) * 100)}%`;
                  return (
                    <div className="cmp">
                      <div className="cmprow">
                        <span className="lbl">open shot</span>
                        <div className="track2"><div className="bar best" style={{ width: w(activeDec.best_available) }} /></div>
                        <span className="num">{activeDec.best_available.toFixed(2)}</span>
                      </div>
                      <div className="cmprow">
                        <span className="lbl">the pass</span>
                        <div className="track2"><div className="bar chosen" style={{ width: w(activeDec.post_epv) }} /></div>
                        <span className="num">{activeDec.post_epv.toFixed(2)}</span>
                      </div>
                    </div>
                  );
                })()}
                <div className={`cnote${activeDec.regret_clipped > 0 ? "" : " good"}`}>
                  {activeDec.regret_clipped > 0
                    ? <><b>{activeDec.regret_clipped.toFixed(2)}</b> left on the table</>
                    : <>good read — the pass beat the open shot</>}
                </div>
              </>
            ) : (
              "Scrub or play; badges appear at each open ball-handler pass-up decision. Click a tick on the timeline to jump to one."
            )}
          </div>
        </div>

        <div className="panel">
          {topMoments.length > 0 && (
            <>
              <h3>Biggest misses this game</h3>
              <div className="reel">
                {topMoments.map((m, i) => (
                  <div className="reelrow" key={i} onClick={() => jumpTo(m.pi, m.frame)} title={`Jump to possession ${m.pi + 1}`}>
                    <span className={`badge ${m.badge}`}>{BADGE_LABEL[m.badge]}</span>
                    <span className="who">{data.players?.[String(m.player)]?.name || "?"}</span>
                    <span className="pts">{m.regret_clipped.toFixed(2)} left</span>
                  </div>
                ))}
              </div>
            </>
          )}
          <h3>Possession review</h3>
          <div className="poss-meta">
            <b>{offAbbr}</b> on offense · Q{poss?.period} · ended <b>{poss?.end_reason?.replace(/_/g, " ")}</b> ({outcomePts})
            {possPointsLeft > 0 && <> · <b>{possPointsLeft.toFixed(2)}</b> pts left on the table</>}
          </div>
          <h3>Decisions</h3>
          <DecisionList decisions={poss?.decisions} players={data.players} activeIdx={activeIdx} onPick={pick} />
          <div className="legend">
            <span className="badge great">Great</span><span className="badge good">Good</span>
            <span className="badge inaccuracy">Inaccuracy</span><span className="badge mistake">Mistake</span>
            <span className="badge blunder">Blunder</span>
            <div style={{ marginTop: 8 }}>tiered on points left on the table when an open look is passed up.</div>
          </div>
        </div>
      </div>
    </>
  );
}
