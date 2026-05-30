import Demo from "../components/Demo";

export default function Page() {
  return (
    <main className="wrap">
      <div className="title">
        <h1>PLOT</h1>
        <span className="tag">Points Left On the Table — chess.com&apos;s post-game review, for NBA possessions</span>
      </div>
      <Demo />
      <div className="foot">
        The <b>eval bar</b> is the expected points the possession will yield (EPV), read off a
        ten-player tracking model. At each open ball-handler <b>pass-up</b>, <b>regret</b> = the open
        shot&apos;s value minus the pass&apos;s value — points left on the table — tiered into a badge.
        Built on 2015-16 SportVU tracking; the eval trace is leakage-free and recalibrated. This is a
        narrow v1 (it scores open pass-ups only) and measures left value <i>conditional on observables</i>,
        not proven causal decision quality. See the repo&apos;s G1–G5 reports.
      </div>
    </main>
  );
}
