// Sweep the widget's controls in node and print every rate the prose quotes.
//   node posts/cluster-interpretation/src/check_model.mjs            # summary grid
//   node posts/cluster-interpretation/src/check_model.mjs --seeds 5  # seed spread
// Asserts the t-test machinery against known quantiles before sweeping.
import assert from "node:assert/strict";
import { simulate, tTwoSided } from "../model.js";

// Student t sanity: two-sided p at the 97.5% quantile is 0.05.
for (const [t, df] of [[1.959964, 1e7], [2.228139, 10], [12.7062, 1], [2.570582, 5]]) {
  assert.ok(Math.abs(tTwoSided(t, df) - 0.05) < 1e-4, `t=${t} df=${df}`);
}

const args = process.argv.slice(2);
const nSeeds = args.includes("--seeds") ? Number(args[args.indexOf("--seeds") + 1]) : 1;
const PS = [10, 20, 50, 100, 200, 500, 1000, 2000];
const SNRS = [0, 1, 2, 3, 4];
const pct = (x) => (x === null ? "  —  " : (100 * x).toFixed(1).padStart(5));

for (const snr of SNRS) {
  console.log(`\nsnr=${snr}   rows: p   | false-positive % (dd split fission) | driver power % (dd split fission)  [min..max over ${nSeeds} seed(s)]`);
  for (const p of PS) {
    const runs = [];
    for (let s = 1; s <= nSeeds; s++) runs.push(simulate({ p, snr, seed: s }));
    const cell = (k, f) => {
      const v = runs.map((r) => r.rates[k][f]).filter((x) => x !== null);
      if (!v.length) return "  —  ";
      const lo = Math.min(...v), hi = Math.max(...v);
      return nSeeds === 1 ? pct(lo) : `${pct(lo)}..${pct(hi)}`;
    };
    const t0 = Date.now();
    simulate({ p, snr, seed: 99 });
    const ms = Date.now() - t0;
    console.log(
      `${String(p).padStart(5)} | ${[0, 1, 2].map((k) => cell(k, "falsePositive")).join(" ")} | ${[0, 1, 2]
        .map((k) => cell(k, "driverPower"))
        .join(" ")}   (${runs[0].reps} reps, ${ms} ms, one-cluster ${runs.map((r) => r.rates[1].oneCluster).join("/")})`,
    );
  }
}
