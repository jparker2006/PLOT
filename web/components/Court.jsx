"use client";
import { useEffect, useRef } from "react";

const PXFT = 10;            // pixels per foot
const W = 94 * PXFT;        // 940
const H = 50 * PXFT;        // 500
const HOME = "#4aa3df";
const AWAY = "#d6453d";
const BALL = "#f0a23a";

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1 * PXFT, y1 * PXFT);
  ctx.lineTo(x2 * PXFT, y2 * PXFT);
  ctx.stroke();
}

function drawCourt(ctx) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#16202c";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#33445a";
  ctx.lineWidth = 2;
  ctx.strokeRect(2, 2, W - 4, H - 4);
  // half court + center circle
  line(ctx, 47, 0, 47, 50);
  ctx.beginPath();
  ctx.arc(47 * PXFT, 25 * PXFT, 6 * PXFT, 0, Math.PI * 2);
  ctx.stroke();
  for (const rim of [5.25, 88.75]) {
    const dir = rim < 47 ? 1 : -1;
    const base = rim < 47 ? 0 : 94;
    // paint (lane): 19 ft deep, 16 ft wide
    ctx.strokeRect((base) * PXFT - (rim < 47 ? 0 : 19 * PXFT), 17 * PXFT, 19 * PXFT, 16 * PXFT);
    // free-throw circle
    ctx.beginPath();
    ctx.arc((base + dir * 19) * PXFT, 25 * PXFT, 6 * PXFT, 0, Math.PI * 2);
    ctx.stroke();
    // backboard + rim
    line(ctx, base + dir * 4, 22, base + dir * 4, 28);
    ctx.beginPath();
    ctx.arc(rim * PXFT, 25 * PXFT, 0.75 * PXFT, 0, Math.PI * 2);
    ctx.stroke();
    // 3pt: corners (14 ft straight) then arc r=23.75
    line(ctx, base, 3, base + dir * 14, 3);
    line(ctx, base, 47, base + dir * 14, 47);
    ctx.beginPath();
    const a = Math.acos((dir * (14)) / 23.75); // not exact; visual approximation
    ctx.arc(rim * PXFT, 25 * PXFT, 23.75 * PXFT,
      dir > 0 ? -1.05 : Math.PI - 1.05, dir > 0 ? 1.05 : Math.PI + 1.05);
    ctx.stroke();
  }
}

export default function Court({ frame, teams, highlight }) {
  const ref = useRef(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    drawCourt(ctx);
    if (!frame) return;
    const homeId = teams?.home?.id;
    // players
    for (const p of frame.pl || []) {
      const [pid, team, x, y] = p;
      ctx.beginPath();
      ctx.arc(x * PXFT, y * PXFT, 1.5 * PXFT, 0, Math.PI * 2);
      ctx.fillStyle = team === homeId ? HOME : AWAY;
      ctx.fill();
      if (pid === highlight) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#fff";
        ctx.stroke();
      }
      ctx.lineWidth = 1;
    }
    // ball
    if (frame.ball) {
      ctx.beginPath();
      ctx.arc(frame.ball[0] * PXFT, frame.ball[1] * PXFT, 0.9 * PXFT, 0, Math.PI * 2);
      ctx.fillStyle = BALL;
      ctx.fill();
      ctx.strokeStyle = "#7a4a12";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [frame, teams, highlight]);

  return <canvas className="court" ref={ref} width={W} height={H} />;
}
