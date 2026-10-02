/* Sweeps the widget controls and prints every number the post quotes from a
 * widget, so the prose can be checked against the model rather than against
 * one screenshot. Run from the repo root:
 *
 *   node posts/collinearity/src/check_model.js
 */
"use strict";
const M = require("../model.js");

const f = (x, d = 2) => Number(x).toFixed(d);
const angles = [90, 45, 25.84, 8.11, 2.56, 1, 0.5];

console.log("Two columns: correlation, condition number, VIF, singular values (unit columns)");
for (const t of angles) {
  const s = M.singular(t);
  console.log(`  theta ${f(t).padStart(6)}  r ${f(M.corr(t), 5)}  kappa ${f(M.kappa(t)).padStart(7)}  ` +
    `VIF ${f(M.vif(t), 1).padStart(8)}  sigma ${f(s[0], 3)}, ${f(s[1], 4)}`);
}

console.log("\nPlane on a fence: spread of 24 fits (sd), and of predictions at the two query points");
for (const t of angles) {
  const s = M.spread(t, 24);
  console.log(`  theta ${f(t).padStart(6)}  sd b1 ${f(s.b1).padStart(6)}  sd b2 ${f(s.b2).padStart(6)}  ` +
    `sd sum ${f(s.sum)}  sd diff ${f(s.diff).padStart(6)}  on ${f(s.on)} [${f(s.onRange[0])}, ${f(s.onRange[1])}]  ` +
    `off ${f(s.off).padStart(6)} [${f(s.offRange[0])}, ${f(s.offRange[1])}]  negative coefs ${s.flips}/48`);
}
console.log("  first six fits at 8.11 deg:", [0, 1, 2, 3, 4, 5].map((k) => M.fit(8.11, k).map((x) => f(x)).join(",")).join("  "));
console.log("  first six fits at 0.5 deg: ", [0, 1, 2, 3, 4, 5].map((k) => M.fit(0.5, k).map((x) => f(x)).join(",")).join("  "));
const mses = [...Array(24).keys()].map((k) => M.sample(8.11, k).mse);
console.log(`  mean squared residual over the 24 draws: ${f(Math.min(...mses), 3)} to ${f(Math.max(...mses), 3)} (same at every angle)`);
console.log(`  draw 0 mse at 90 / 8.11 / 0.5: ${[90, 8.11, 0.5].map((t) => f(M.sample(t, 0).mse, 4)).join(" / ")}`);

console.log("\nValley: spread of all 200 fits");
for (const t of angles) {
  const s = M.spread(t, M.DRAWS);
  console.log(`  theta ${f(t).padStart(6)}  sd b1 ${f(s.b1).padStart(6)}  sd sum ${f(s.sum)}  sd diff ${f(s.diff).padStart(6)}  ` +
    `curvature across ${f(2 * M.singular(t)[0] ** 2, 3)}  along ${f(2 * M.singular(t)[1] ** 2, 5)}`);
}

console.log("\nWhich 2%: the same-size nudge of y in three directions");
for (const t of angles) {
  const row = ["strong", "weak", "out"].map((w) => {
    const r = M.nudge(t, w);
    return `${w}: beta (${f(r.beta[0])}, ${f(r.beta[1])}) change ${f(100 * r.betaChange, 1)}% fitted ${f(100 * r.fittedChange, 2)}% rss ${f(100 * r.rssChange, 1)}%`;
  });
  console.log(`  theta ${f(t).padStart(6)}  ${row.join("  |  ")}`);
}

console.log("\nThree columns: lift alpha of x3 out of the plane of x1, x2");
for (const a of [90, 45, 20, 8.11, 2.56, 1, 0.5]) {
  const r = M.three(a);
  console.log(`  alpha ${f(a).padStart(6)}  largest pairwise r ${f(r.pairwise, 3)}  kappa ${f(r.kappa).padStart(7)}  ` +
    `volume ${f(r.volume, 4)}  VIF ${r.vif.map((v) => f(v, 1)).join(", ")}`);
}
