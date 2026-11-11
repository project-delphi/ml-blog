/* Checks the normal-approximation widget model and prints the numbers the post quotes.
 * Run: node posts/steins-method-normal-approximation/src/check_model.js */
"use strict";

const assert = require("node:assert/strict");
global.window = {};
const data = require("../widget-data/data.js");
const M = require("../model.js");
M.setData(data);

// 1. The theorem holds: bound >= exact d_W at every n, every law, every grid p.
let worst = Infinity, where = "";
for (const e of data.laws.bernoulli) {
  const r = M.worstRatio(M.law("bernoulli", e.p));
  if (r < worst) { worst = r; where = `bernoulli p=${e.p}`; }
}
for (const k of ["exponential", "uniform", "mixture"]) {
  const r = M.worstRatio(M.law(k));
  if (r < worst) { worst = r; where = k; }
}
assert.ok(worst > 1);
console.log(`bound / d_W >= ${worst.toFixed(2)} everywhere (tightest: ${where})`);

// 2. d_W decreases in n for the three continuous laws (the chart's curve has no bumps). The Bernoulli is
//    left out: its lattice makes d_W oscillate in n, and the post makes no monotonicity claim about it.
for (const k of ["exponential", "uniform", "mixture"]) {
  const dw = M.law(k).dw;
  for (let n = 1; n < dw.length; n++) assert.ok(dw[n] < dw[n - 1], `${k} not decreasing at n=${n + 1}`);
}

// 3. Draws of W are standardised: mean ~ 0, variance ~ 1, for several n and seeds.
for (const k of ["exponential", "uniform", "mixture"]) {
  for (const n of [1, 5, 50, 200]) {
    for (const seed of [1, 2, 3]) {
      const xs = M.drawW(k, n, 40000, seed);
      let m = 0, v = 0;
      for (const x of xs) m += x;
      m /= xs.length;
      for (const x of xs) v += (x - m) ** 2;
      v /= xs.length - 1;
      assert.ok(Math.abs(m) < 0.03 && Math.abs(v - 1) < 0.04, `${k} n=${n} seed=${seed}: mean ${m}, var ${v}`);
    }
  }
}
for (const p of [0.05, 0.3, 0.5]) for (const n of [1, 7, 200]) {
  const { atoms } = M.bernoulliAtoms(n, p);
  const tot = atoms.reduce((s, a) => s + a.mass, 0), mean = atoms.reduce((s, a) => s + a.mass * a.w, 0);
  const v = atoms.reduce((s, a) => s + a.mass * a.w * a.w, 0);
  assert.ok(Math.abs(tot - 1) < 1e-9 && Math.abs(mean) < 1e-9 && Math.abs(v - 1) < 1e-9);
}

// 4. Numbers for the prose.
const show = (label, L) => {
  const at = (n) => L.dw[n - 1];
  const C = M.bound(L, 1);
  console.log(`${label.padEnd(18)} C = ${C.toFixed(3)}  d_W(1) ${at(1).toFixed(4)}  d_W(30) ${at(30).toFixed(4)}  ` +
    `d_W(200) ${at(200).toFixed(5)}  bound(30) ${M.bound(L, 30).toFixed(4)}  bound(200) ${M.bound(L, 200).toFixed(4)}  ` +
    `ratio(200) ${(M.bound(L, 200) / at(200)).toFixed(1)}  sqrt(n)*d_W(200) ${(Math.sqrt(200) * at(200)).toFixed(3)}  n*d_W(200) ${(200 * at(200)).toFixed(3)}`);
};
for (const p of [0.05, 0.1, 0.3, 0.5]) show(`bernoulli ${p}`, M.law("bernoulli", p));
for (const k of ["exponential", "uniform", "mixture"]) show(k, M.law(k));
// n at which d_W first drops below 0.05 and 0.01
for (const k of ["exponential", "uniform", "mixture"]) {
  const dw = M.law(k).dw;
  const first = (t) => dw.findIndex((x) => x < t) + 1 || "> 200";
  console.log(`${k}: d_W < 0.05 from n = ${first(0.05)}, < 0.01 from n = ${first(0.01)}`);
}
for (const p of [0.05, 0.5]) {
  const dw = M.law("bernoulli", p).dw;
  const first = (t) => dw.findIndex((x) => x < t) + 1 || "> 200";
  console.log(`bernoulli ${p}: d_W < 0.05 from n = ${first(0.05)}, < 0.1 from n = ${first(0.1)}`);
}
console.log("checks pass");

// 7. The prose's slopes: least-squares log-log slope of the exact d_W over n = 50..200 (as make_data.py fits it)
//    is -0.50 to -0.51 for the exponential, the mixture and every Bernoulli p on the slider, and -1.00 for the uniform.
{
  const fit = (dw) => {
    const xs = [], ys = [];
    for (let n = 50; n <= 200; n++) { xs.push(Math.log(n)); ys.push(Math.log(dw[n - 1])); }
    const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
    return sxy / sxx;
  };
  const slopes = [fit(M.law("exponential").dw), fit(M.law("mixture").dw), ...data.laws.bernoulli.map((e) => fit(e.dw))];
  for (const sl of slopes) assert.ok(sl.toFixed(2) === "-0.50" || sl.toFixed(2) === "-0.51", `slope ${sl}`);
  const su = fit(M.law("uniform").dw);
  assert.equal(su.toFixed(2), "-1.00");
  console.log(`log-log slopes over n = 50..200: ${Math.min(...slopes).toFixed(3)} to ${Math.max(...slopes).toFixed(3)} ` +
    `(exponential, mixture, ${data.laws.bernoulli.length} Bernoulli p); uniform ${su.toFixed(3)}`);
}
