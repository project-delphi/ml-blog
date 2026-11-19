// Sweep the confounding widget's controls and assert what the prose claims.
// Run: node posts/two-cultures-pushback-and-causality/src/check_model.js
"use strict";
const assert = require("node:assert/strict");
const M = require("../model.js");

let worstNaive = 0, worstAdj = 0, n = 0;
for (let a = 0; a <= 0.951; a += 0.05)
  for (let b = -2; b <= 0.001; b += 0.05)
    for (let tau = -0.5; tau <= 1.001; tau += 0.05)
      for (const seed of [1, 2, 3, 4, 5]) {
        const p = { a: +a.toFixed(2), b: +b.toFixed(2), tau: +tau.toFixed(2) };
        const s = M.stats(p, seed);
        worstNaive = Math.max(worstNaive, Math.abs(s.naiveSlope - s.theorySlope));
        worstAdj = Math.max(worstAdj, Math.abs(s.adjSlope - p.tau));
        n++;
      }
console.log(`settings x seeds swept: ${n}`);
console.log(`max |naive - (tau + a b)|: ${worstNaive.toFixed(3)}`);
console.log(`max |adjusted - tau|:      ${worstAdj.toFixed(3)}`);
assert.ok(worstAdj < 0.35, "adjusted slope strays from tau");

function show(p, label) {
  const r = [1, 2, 3, 4, 5].map((s) => M.stats(p, s));
  const rng = (k) => {
    const v = r.map((x) => x[k]);
    return `${Math.min(...v).toFixed(2)} to ${Math.max(...v).toFixed(2)}`;
  };
  console.log(`${label} a=${p.a} b=${p.b} tau=${p.tau}: naive ${rng("naiveSlope")}, ` +
    `R2 ${rng("naiveR2")}, adjusted ${rng("adjSlope")}, theory ${(p.tau + p.a * p.b).toFixed(2)}`);
  return r;
}
const d = show({ a: 0.9, b: -1.6, tau: 0.5 }, "default");
d.forEach((s) => assert.ok(s.naiveR2 > 0.4, "default predictive R2 should be above 0.4"));
d.forEach((s) => assert.ok(s.naiveSlope < 0, "default naive slope should be negative"));
d.forEach((s) => assert.ok(s.adjSlope > 0.3, "default adjusted slope should be near 0.5"));
show({ a: 0, b: -1.6, tau: 0.5 }, "no confounding");
show({ a: 0.95, b: -2, tau: 0.5 }, "extreme");
console.log("ok");
