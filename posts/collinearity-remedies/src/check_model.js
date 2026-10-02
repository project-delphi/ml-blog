/* Sweeps the widget controls and prints every number the post quotes from a
 * widget, and checks the exact two-variable solver against a brute-force
 * search. Run from the repo root:
 *
 *   node posts/collinearity-remedies/src/check_model.js
 */
"use strict";
const M = require("../model.js");

const f = (x, d = 2) => Number(x).toFixed(d);
const deg = (r) => Math.acos(r) * 180 / Math.PI;
const PRESETS = [["r = 0", 90], ["0.9", deg(0.9)], ["0.99", deg(0.99)], ["0.999", deg(0.999)]];
const LAMBDAS = [4, 8, 12, 16, 20, 24, 28].map(M.lambdaOf);  // 0.0032, 0.01, 0.032, 0.1, 0.32, 1, 3.2

// 1. The closed-form solver against coordinate descent from a cold start.
let worst = 0;
for (const theta of [90, deg(0.9), deg(0.99)]) {
  const c = Math.cos(theta * Math.PI / 180);
  for (const method of ["ridge", "lasso", "enet"]) {
    const m = M.METHODS[method];
    for (const lambda of LAMBDAS) {
      for (let k = 0; k < 40; k++) {
        const bhat = M.ols(theta, k), exact = M.estimate(theta, k, method, lambda);
        const g = [bhat[0] + c * bhat[1], c * bhat[0] + bhat[1]];
        const q = 1 + lambda * m.l2, a = lambda * m.l1 / 2;
        const soft = (x) => (x > a ? x - a : x < -a ? x + a : 0);
        let b = [0, 0];
        for (let it = 0; it < 200000; it++) {
          const n0 = soft(g[0] - c * b[1]) / q, n1 = soft(g[1] - c * n0) / q;
          const step = Math.abs(n0 - b[0]) + Math.abs(n1 - b[1]);
          b = [n0, n1];
          if (step < 1e-14) break;
        }
        worst = Math.max(worst, Math.abs(b[0] - exact[0]), Math.abs(b[1] - exact[1]));
      }
    }
  }
}
console.log(`solver check: largest gap between the exact solver and coordinate descent = ${worst.toExponential(2)}`);

// 2. What each method does over the 200 draws.
for (const [name, theta] of PRESETS) {
  console.log(`\n=== ${name}  (theta ${f(theta)}, kappa ${f(M.kappa(theta), 1)})`);
  const row = (label, s) => console.log(
    `  ${label.padEnd(22)} centre (${f(s.mean[0])}, ${f(s.mean[1])})  sd b1 ${f(s.sd[0])}  dist ${f(s.rmse)}  ` +
    `zero x1 ${String(s.zero1).padStart(3)} x2 ${String(s.zero2).padStart(3)}  ` +
    `agree ${f(s.on.mean)}±${f(s.on.sd)} rmse ${f(s.on.rmse)}  disagree ${f(s.off.mean)}±${f(s.off.sd)} rmse ${f(s.off.rmse)}`);
  row("least squares", M.summary(theta, "ols", 0));
  row("one component", M.summary(theta, "pcr", 0));
  for (const method of ["ridge", "lasso", "enet"]) {
    for (const lambda of LAMBDAS) row(`${method} ${lambda.toPrecision(2)}`, M.summary(theta, method, lambda));
  }
}

// 3. The first draws under each method at r = 0.99, for the "New noise" steps.
const t99 = deg(0.99);
for (const [method, lambda] of [["ols", 0], ["ridge", 0.1], ["lasso", 0.1], ["enet", 0.1], ["pcr", 0]]) {
  console.log(`\n${method} ${lambda} at r = 0.99, draws 0-5: ` +
    [0, 1, 2, 3, 4, 5].map((k) => "(" + M.estimate(t99, k, method, lambda).map((x) => f(x)).join(", ") + ")").join(" "));
}
console.log("slider stops: lambda 0.1 ->", M.lambdaStop(0.1), " 0.01 ->", M.lambdaStop(0.01), " 1 ->", M.lambdaStop(1));
