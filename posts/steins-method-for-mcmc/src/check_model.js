/* Checks the Part 5 widget models and prints every number the post quotes.
 * Run: node posts/steins-method-for-mcmc/src/check_model.js */
"use strict";

const assert = require("node:assert/strict");
const M = require("../model.js");
const f = (x, d = 3) => x.toFixed(d);
const range = (xs, d = 3) => `[${f(Math.min(...xs), d)}, ${f(Math.max(...xs), d)}]`;
const KEYS = ["gauss", "heavy", "mixture"];

// 1. Scores are derivatives of the log-densities, and ds of the scores.
for (const k of KEYS) {
  const T = M.TARGETS[k];
  for (let x = -6; x <= 6; x += 0.37) {
    const h = 1e-5;
    assert.ok(Math.abs((T.logp(x + h) - T.logp(x - h)) / (2 * h) - T.s(x)) < 1e-6, k + " score");
    assert.ok(Math.abs((T.s(x + h) - T.s(x - h)) / (2 * h) - T.ds(x)) < 1e-5, k + " ds");
  }
}

// 2. Gaussian, h = x^2: ULA's stationary variance is exactly 1 / (1 - eps/2), so the bias is eps / (2 - eps).
for (const e of [0.01, 0.05, 0.1, 0.2, 0.4]) {
  const a = M.analyse("gauss", "ula", e);
  assert.ok(Math.abs(a.bias[1] - e / (2 - e)) < 2e-6, "gaussian variance bias at " + e + ": " + a.bias[1]);
}
console.log("gauss x^2: grid bias matches eps/(2 - eps) to 2e-6 at eps in {0.01, ..., 0.4}");

// 3. MALA's stationary law is the target on the grid; ULA's is not.
for (const k of KEYS) for (const e of M.EPS_GRID) {
  const b = M.analyse(k, "mala", e);
  b.bias.forEach((v) => assert.ok(Math.abs(v) < 1e-9, `MALA bias ${k} ${e}: ${v}`));
}
console.log("MALA: |bias| < 1e-9 for every target, test function and step on the curve grid");

// 4. Stein's first-order prediction: ratio exact / predicted -> 1 as eps -> 0.
console.log("\nStein coefficients c (bias ~ c eps) and ratio exact/predicted at eps = 0.01, 0.03, 0.1, 0.4:");
for (const k of KEYS) {
  const G = M.setup(k);
  M.HS.forEach((h, i) => {
    if (Math.abs(G.coef[i]) < 1e-12) {
      for (const e of [0.01, 0.1, 0.4]) assert.ok(Math.abs(M.analyse(k, "ula", e).bias[i]) < 1e-12);
      console.log(`  ${k.padEnd(7)} ${h.key.padEnd(6)}: E_p h = ${f(G.Eh[i], 5)}, c = 0 and ULA bias = 0 (symmetry)`);
      return;
    }
    const rs = [0.01, 0.03, 0.1, 0.4].map((e) => M.analyse(k, "ula", e).bias[i] / (G.coef[i] * e));
    const b01 = M.analyse(k, "ula", 0.1).bias[i];
    console.log(`  ${k.padEnd(7)} ${h.key.padEnd(6)}: E_p h = ${f(G.Eh[i], 5)}, c = ${f(G.coef[i], 5)}, ratios ${rs.map((r) => f(r, 3)).join(", ")}; at eps 0.1 bias ${f(b01, 5)} vs ${f(G.coef[i] * 0.1, 5)}`);
    assert.ok(Math.abs(rs[0] - 1) < 0.04, "first-order ratio at 0.01");
    assert.ok(Math.abs(rs[0] - 1) < Math.abs(rs[2] - 1), "ratio improves as eps falls");
  });
}

// 5. Asymptotic variance: grid value against the diffusion's 2 E f_h^2 / eps - Var h.
console.log("\nasymptotic variance at eps = 0.01: grid ULA / leading order 2E[f_h^2]/eps / grid MALA");
for (const k of KEYS) {
  const G = M.setup(k), a = M.analyse(k, "ula", 0.01), b = M.analyse(k, "mala", 0.01);
  M.HS.forEach((h, i) => {
    const pr = G.avarCont[i] / 0.01;
    console.log(`  ${k.padEnd(7)} ${h.key.padEnd(6)}: ${f(a.avar[i], 2)} / ${f(pr, 2)} / ${f(b.avar[i], 2)}   (2E f^2 = ${f(G.avarCont[i], 4)})`);
    console.log(`      ratio grid/leading ULA ${f(a.avar[i] / pr, 4)}, MALA ${f(b.avar[i] / pr, 4)}`);
    assert.ok(Math.abs(a.avar[i] / pr - 1) < 0.03 && Math.abs(b.avar[i] / pr - 1) < 0.03, "avar leading order");
  });
}

// 6. Grid convergence: 241 points against 481 for the quoted biases.
{
  const coarse = {};
  for (const k of KEYS) coarse[k] = [0.01, 0.1].map((e) => M.analyse(k, "ula", e).bias);
  M.setGridSize(481);
  let worst = 0;
  for (const k of KEYS) [0.01, 0.1].forEach((e, j) => {
    const fine = M.analyse(k, "ula", e).bias;
    fine.forEach((v, i) => { if (Math.abs(v) > 1e-9) worst = Math.max(worst, Math.abs(coarse[k][j][i] / v - 1)); });
  });
  M.setGridSize(241);
  console.log(`\ngrid 241 vs 481: largest relative change in a nonzero bias = ${worst.toExponential(2)}`);
  assert.ok(worst < 2e-3);
}

// 7. Best step and RMSE against budget (curve interpolation, checked against golden-section search).
console.log("\nbest step and RMSE by budget N (ULA eps, ULA rmse | MALA eps, MALA rmse):");
for (const k of KEYS) M.HS.forEach((h, i) => {
  const rows = [3, 4, 5, 6].map((lN) => {
    const N = 10 ** lN, u = M.rmseCurve(k, "ula", i, N).best, m = M.rmseCurve(k, "mala", i, N).best;
    return `N=1e${lN}: ${f(u.eps, 3)}, ${u.rmse.toPrecision(3)} | ${f(m.eps, 3)}, ${m.rmse.toPrecision(3)}`;
  });
  console.log(`  ${k.padEnd(7)} ${h.key.padEnd(6)} ${rows.join("  ;  ")}`);
});
for (const [k, i, N] of [["gauss", 1, 1e4], ["mixture", 0, 1e5]]) {
  const a = M.rmseCurve(k, "ula", i, N).best, g = M.bestEps(k, "ula", i, N);
  assert.ok(Math.abs(a.eps / g.eps - 1) < 0.03 && Math.abs(a.rmse / g.rmse - 1) < 0.01, "interpolated optimum");
}
{
  // ULA: best eps and RMSE fall by about 10^(1/3) per decade of N (gauss, x^2)
  const r = [3, 4, 5, 6].map((lN) => M.rmseCurve("gauss", "ula", 1, 10 ** lN).best);
  for (let j = 1; j < 4; j++) {
    const re = r[j - 1].eps / r[j].eps, rr = r[j - 1].rmse / r[j].rmse;
    assert.ok(Math.abs(re - 10 ** (1 / 3)) < 0.1 && Math.abs(rr - 10 ** (1 / 3)) < 0.1, "N^-1/3 scaling");
  }
  console.log(`gauss x^2: per decade of N, ULA's best eps shrinks by ${r.slice(1).map((x, j) => f(r[j].eps / x.eps, 2)).join(", ")} and its RMSE by ${r.slice(1).map((x, j) => f(r[j].rmse / x.rmse, 2)).join(", ")} (10^(1/3) = ${f(10 ** (1 / 3), 2)})`);
}
{
  const N = 1e4, eps = 0.1, k = "gauss", i = 1;
  const a = M.analyse(k, "ula", eps), b = M.analyse(k, "mala", eps);
  console.log(`defaults (gauss, x^2, eps 0.1, N 1e4): ULA bias ${f(a.bias[i], 4)}, Stein ${f(0.05, 4)}, RMSE ULA ${M.rmse(a.bias[i], a.avar[i], N).toPrecision(3)}, MALA ${M.rmse(b.bias[i], b.avar[i], N).toPrecision(3)}, acc ${f(b.acc, 3)}`);
}
console.log("MALA acceptance by step (gauss / heavy / mixture):",
  [0.01, 0.1, 0.2, 0.4].map((e) => `eps ${e}: ` + KEYS.map((k) => f(M.analyse(k, "mala", e).acc, 3)).join(" / ")).join("; "));

// 8. Zero-variance control variates: reference values and MSE reductions over seed sets.
{
  const ref = M.lrReference();
  console.log(`\nlogistic posterior: n = ${M.LR.n}, sum y = ${M.LR.y.reduce((a, b) => a + b, 0)}, reference E[theta1] = ${f(ref[0], 4)}, E[theta1^2] = ${f(ref[1], 4)}, P(theta1 > 1) = ${f(ref[2], 4)}`);
  // score is the gradient of logp
  for (const [a, b] of [[0.3, -0.2], [1.1, 0.4], [-0.5, -1.2]]) {
    const s = M.lrScore(a, b, [0, 0]), h = 1e-5;
    assert.ok(Math.abs((M.lrLogp(a + h, b) - M.lrLogp(a - h, b)) / (2 * h) - s[0]) < 1e-5);
    assert.ok(Math.abs((M.lrLogp(a, b + h) - M.lrLogp(a, b - h)) / (2 * h) - s[1]) < 1e-5);
  }
  for (const n of [1000, 5000, 20000]) {
    M.LR_H.forEach((h, i) => {
      const v1 = [], v2 = [], mse = [], acc = [];
      for (let s = 1; s <= 8; s++) { const r = M.replicate(i, n, 0.15, 20, 100 * s); v1.push(r.vr1); v2.push(r.vr2); mse.push(r.mse.plain); acc.push(r.acc); }
      console.log(`  n=${String(n).padEnd(5)} ${h.key.padEnd(6)}: plain MSE ${range(mse.map((x) => x * 1e4), 2)}e-4, reduction deg1 ${range(v1, 1)}, deg2 ${range(v2, 1)}, acceptance ${range(acc, 2)}`);
      if (h.key === "mean") assert.ok(Math.min(...v2) > 100);
      if (h.key === "tail") assert.ok(Math.max(...v2) < 20);
    });
  }
}
console.log("\nall assertions passed");
