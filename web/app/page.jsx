import Demo from "../components/Demo";

export default function Page() {
  return (
    <main className="wrap">
      <div className="title">
        <h1>PLOT</h1>
        <span className="tag">Points Left On the Table — chess.com&apos;s post-game review, for NBA possessions</span>
      </div>
      <p className="lede">
        Watch a possession with an <b>eval bar</b> — the expected points right now (EPV), read off a
        ten-player tracking model. Each time a wide-open ball-handler <b>passes up</b> the shot, a
        chess.com-style <b>badge</b> grades the decision by the points left on the table.
      </p>
      <Demo />
      <div className="foot">
        <p>
          <b>This isn&apos;t just a pretty meter — we checked it against reality.</b> Across 208 games,
          declining a good open look really does cost points (about <b>0.11</b> on an average look,
          <b> ~0.19</b> on a great one); declining a <i>bad</i> look is free; and the cost grows the better
          the look was. The signal is invisible to the box score.
        </p>
        <p className="limits">
          The honest limits (the same ones in the paper): the mirror move — <i>shooting over a wide-open
          teammate</i> — does <b>not</b> hold up, because an &ldquo;open man&rdquo; is often open precisely
          because the defense doesn&apos;t respect him; the effect is modest; this scores open pass-ups
          only; it measures left value <i>conditional on what the cameras can see</i>, not proven causal
          decision quality; and pinning down <i>which</i> players are best or worst at it is weak.
          Built on 2015-16 SportVU tracking; the eval trace is leakage-free and recalibrated.{" "}
          <a href="https://github.com/jparker2006/PLOT" target="_blank" rel="noreferrer">
            Code, paper &amp; gate reports →
          </a>
        </p>
      </div>
    </main>
  );
}
