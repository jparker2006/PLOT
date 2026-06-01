"use client";
import { useEffect, useRef } from "react";
import { inkOn, initials, teamColor } from "../lib/teams";

const PXFT = 10;            // pixels per foot
const W = 94 * PXFT;        // 940
const H = 50 * PXFT;        // 500
const BALL = "#f5a623";

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1 * PXFT, y1 * PXFT);
  ctx.lineTo(x2 * PXFT, y2 * PXFT);
  ctx.stroke();
}

function drawCourt(ctx) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#17222e";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#33445a";
  ctx.lineWidth = 2;
  ctx.strokeRect(2, 2, W - 4, H - 4);
  line(ctx, 47, 0, 47, 50);
  ctx.beginPath();
  ctx.arc(47 * PXFT, 25 * PXFT, 6 * PXFT, 0, Math.PI * 2);
  ctx.stroke();
  for (const rim of [5.25, 88.75]) {
    const dir = rim < 47 ? 1 : -1;
    const base = rim < 47 ? 0 : 94;
    ctx.strokeRect(base * PXFT - (rim < 47 ? 0 : 19 * PXFT), 17 * PXFT, 19 * PXFT, 16 * PXFT);
    ctx.beginPath();
    ctx.arc((base + dir * 19) * PXFT, 25 * PXFT, 6 * PXFT, 0, Math.PI * 2);
    ctx.stroke();
    line(ctx, base + dir * 4, 22, base + dir * 4, 28);
    ctx.beginPath();
    ctx.arc(rim * PXFT, 25 * PXFT, 0.75 * PXFT, 0, Math.PI * 2);
    ctx.stroke();
    line(ctx, base, 3, base + dir * 14, 3);
    line(ctx, base, 47, base + dir * 14, 47);
    ctx.beginPath();
    ctx.arc(rim * PXFT, 25 * PXFT, 23.75 * PXFT,
      dir > 0 ? -1.05 : Math.PI - 1.05, dir > 0 ? 1.05 : Math.PI + 1.05);
    ctx.stroke();
  }
}

export default function Court({ frame, teams, highlight, players, teamAbbr }) {
  const ref = useRef(null);
  const trail = useRef([]);

  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    drawCourt(ctx);
    if (!frame) { trail.current = []; return; }

    // ball trail: keep recent positions; reset on a big jump (scrub / new possession)
    if (frame.ball) {
      const last = trail.current[trail.current.length - 1];
      if (last && Math.hypot(frame.ball[0] - last[0], frame.ball[1] - last[1]) > 15) trail.current = [];
      trail.current.push([frame.ball[0], frame.ball[1]]);
      if (trail.current.length > 12) trail.current.shift();
      ctx.lineCap = "round";
      for (let i = 1; i < trail.current.length; i++) {
        const a = trail.current[i - 1], b = trail.current[i];
        ctx.strokeStyle = `rgba(245,166,35,${(i / trail.current.length) * 0.5})`;
        ctx.lineWidth = (i / trail.current.length) * 4 + 0.5;
        line(ctx, a[0], a[1], b[0], b[1]);
      }
    }

    const homeId = teams?.home?.id;
    for (const p of frame.pl || []) {
      const [pid, team, x, y] = p;
      const abbr = teamAbbr?.[String(team)];
      const fill = abbr ? teamColor(abbr) : team === homeId ? "#4aa3df" : "#d6453d";
      const isHi = pid === highlight;
      ctx.beginPath();
      ctx.arc(x * PXFT, y * PXFT, 1.6 * PXFT, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = isHi ? 3 : 1.2;
      ctx.strokeStyle = isHi ? "#ffffff" : "rgba(0,0,0,0.45)";
      ctx.stroke();
      const name = players?.[String(pid)]?.name;
      const tag = initials(name);
      if (tag) {
        ctx.fillStyle = inkOn(fill);
        ctx.font = "700 11px -apple-system, Segoe UI, Roboto, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(tag, x * PXFT, y * PXFT + 0.5);
      }
    }

    if (frame.ball) {
      ctx.beginPath();
      ctx.arc(frame.ball[0] * PXFT, frame.ball[1] * PXFT, 0.95 * PXFT, 0, Math.PI * 2);
      ctx.fillStyle = BALL;
      ctx.fill();
      ctx.strokeStyle = "#7a4a12";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [frame, teams, highlight, players, teamAbbr]);

  return <canvas className="court" ref={ref} width={W} height={H} />;
}
