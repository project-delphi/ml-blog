/* Checks the Part 6 widget models and prints every number the post quotes.
 * Run: node posts/steins-method-in-neural-networks/src/check_model.js */
"use strict";

const assert = require("node:assert/strict");
global.window = {};
const D = require("../widget-data/data.js");
const M = require("../model.js");
const f = (x, d = 3) => x.toFixed(d);
const range = (xs, d = 2) => `[${f(Math.min(...xs), d)}, ${f(Math.max(...xs), d)}]`;

// 1. Activation constants against closed forms.
{
  const relu = M.constants("relu"), lin = M.constants("linear"), sg = M.constants("sign");
  assert.ok(Math.abs(relu.c2 - 0.5) < 1e-9 && Math.abs(relu.kappa - 5) < 1e-6);
  assert.ok(Math.abs(relu.EX3 - 8 * Math.SQRT2 / Math.PI) < 1e-6 && Math.abs(relu.EX4 - 18) < 1e-6);
  assert.ok(Math.abs(lin.kappa - 2) < 1e-6 && Math.abs(sg.kappa) < 1e-9);
  console.log("activation  c2      kappa   E|X|^3  EX^4    Thm4 C  Cor2 2nd  Cor2 1st  limit n*dW");
  for (const a of Object.keys(M.ACTS)) {
    const C = M.constants(a);
    console.log(`${a.padEnd(10)}  ${f(C.c2, 4)}  ${f(C.kappa, 4)}  ${f(C.EX3, 3)}   ${f(C.EX4, 3).padEnd(6)}  ${f(C.thm4, 3)}   ${f(C.mix2, 3)}     ${f(C.mix1, 3)}     ${f(C.limit, 4)}`);
  }
  // E|He_3(Z)| / 8 = (e^{-3/2} + 1/4) / sqrt(2 pi)
  const eHe3 = M.gauss((z) => Math.abs(z * z * z - 3 * z));
  assert.ok(Math.abs(eHe3 / 8 - (Math.exp(-1.5) + 0.25) / Math.sqrt(2 * Math.PI)) < 1e-7);
  console.log(`E|He3(Z)|/8 = ${f(eHe3 / 8, 5)} = (e^-1.5 + 1/4)/sqrt(2 pi) = ${f((Math.exp(-1.5) + 0.25) / Math.sqrt(2 * Math.PI), 5)}`);
}

// 2. Exact distances: both bounds hold at every width, and n d_W approaches the limit.
console.log("\nexact d_W and bounds (Theorem 4 / Corollary 2), ratio bound/exact:");
for (const a of Object.keys(D.acts)) {
  const A = D.acts[a], C = M.constants(a);
  D.n.forEach((n, j) => {
    const B = M.bounds(a, n);
    assert.ok(B.thm4 >= A.dW[j] - 1e-9, `Thm4 ${a} ${n}`);
    assert.ok(B.mix >= A.dW[j] - 1e-5, `Cor2 ${a} ${n}: ${B.mix} vs ${A.dW[j]}`);
  });
  if (a !== "sign") {
    const j1000 = D.n.indexOf(1000);
    assert.ok(Math.abs(1000 * A.dW[j1000] / C.limit - 1) < 0.01, `limit ${a}`);
  } else A.dW.forEach((v) => assert.equal(v, 0));
  const rows = [1, 10, 100, 1000].map((n) => {
    const j = D.n.indexOf(n), B = M.bounds(a, n), e = A.dW[j];
    return `n=${n}: ${e.toPrecision(3)} | ${B.thm4.toPrecision(3)} (${e > 0 ? f(B.thm4 / e, 0) : "inf"}x) | ${B.mix.toPrecision(3)} (${e > 0 ? f(B.mix / e, 1) : "-"}x) | n dW ${f(n * e, 4)}`;
  });
  console.log(`  ${a.padEnd(7)} ${rows.join("  ;  ")}`);
}
console.log("ReLU atom at 0 by width:", [1, 2, 3, 4, 5].map((n) => `n=${n}: ${D.acts.relu.atom[D.n.indexOf(n)]}`).join(", "));

// 3. Real sampled networks agree with the exact law: variance 1 and E|W| from the exact density.
for (const a of ["relu", "tanh", "linear"]) for (const n of [1, 10, 100]) {
  const s = M.sampleOutputs(a, n, 40000, 5), j = D.n.indexOf(n);
  const v = s.reduce((x, y) => x + y * y, 0) / s.length, mabs = s.reduce((x, y) => x + Math.abs(y), 0) / s.length;
  // E|W| from the exact density on the display grid (plus nothing from the atom) — the grid stops at |w| = 4
  let ea = 0, mass = 0; const dw = D.w[1] - D.w[0];
  D.w.forEach((w, i) => { ea += Math.abs(w) * D.acts[a].dens[j][i] * dw; mass += D.acts[a].dens[j][i] * dw; });
  const mcIn = s.filter((x) => Math.abs(x) < 4).reduce((x, y) => x + Math.abs(y), 0) / s.length;
  assert.ok(Math.abs(v - 1) < 0.05, `variance ${a} ${n}: ${v}`);
  assert.ok(Math.abs(mcIn - ea) < 0.02, `E|W| on [-4,4] ${a} ${n}: mc ${mcIn} exact ${ea}`);
  if (n === 10) console.log(`  sampled ${a} n=10: var ${f(v, 3)}, E|W| ${f(mabs, 3)}; exact E|W|1{|W|<4} ${f(ea, 3)} vs sampled ${f(mcIn, 3)} (mass on grid ${f(mass, 3)})`);
}

// 4. Price's theorem for ReLU: K' is the derivative of K and equals E[1{U>0}1{U'>0}] / c2.
for (const rho of [-0.8, -0.3, 0, 0.4, 0.9]) {
  const h = 1e-6, num = (M.reluK(rho + h) - M.reluK(rho - h)) / (2 * h);
  assert.ok(Math.abs(num - M.reluKprime(rho)) < 1e-6);
  // P(U > 0, U' > 0) = 1/4 + asin(rho) / (2 pi); divided by c2 = 1/2
  assert.ok(Math.abs(M.reluKprime(rho) - 2 * (0.25 + Math.asin(rho) / (2 * Math.PI))) < 1e-12);
}
console.log(`\nReLU kernels: K(0) = ${f(M.reluK(0), 4)} (1/pi = ${f(1 / Math.PI, 4)}), K'(0) = ${M.reluKprime(0)}, K(1) = ${M.reluK(1)}, K'(1) = ${M.reluKprime(1)}; NTK K + rho K' at rho = 0.5: ${f(M.reluK(0.5) + 0.5 * M.reluKprime(0.5), 4)}`);

// 5. Hermite coefficients in Stein form: c_k = E[g^(k)(Z)] for smooth links.
{
  const L = M.linkInfo("tanh"), sd = L.sd;
  const d1 = M.gauss((t) => 1 / Math.cosh(t) ** 2) / sd;
  const d3 = M.gauss((t) => { const s = 1 / Math.cosh(t) ** 2; return (6 * Math.tanh(t) ** 2 - 2) * s; }) / sd;
  assert.ok(Math.abs(L.c[0] - d1) < 1e-6 && Math.abs(L.c[2] - d3) < 1e-6 && Math.abs(L.c[1]) < 1e-9);
  const he2 = M.linkInfo("he2");
  assert.ok(Math.abs(he2.c[1] - Math.SQRT2) < 1e-6); // g = (t^2 - 1)/sqrt 2, g'' = sqrt 2
  console.log("\nlink     c1       c2       c3       c4      k*");
  for (const k of Object.keys(M.LINKS)) {
    const I = M.linkInfo(k);
    console.log(`${k.padEnd(7)} ${I.c.map((c) => (Math.abs(c) < 5e-5 ? 0 : c).toFixed(4).padStart(7)).join("  ")}   ${I.kstar}`);
  }
}

// 6. Direction recovery: overlaps over 5 seed sets of 6 data sets (30 data sets) per setting.
console.log("\noverlap |cos| (mean over 30 data sets) at n = 100, 400, 1600, 12800; first order / second order:");
const links = ["linear", "tanh", "relu", "abs", "he2"];
const summary = {};
for (const d of [10, 20, 40]) for (const lk of links) {
  const runs = [];
  for (let s = 0; s < 5; s++) runs.push(M.sweep(lk, d, 0.5, 6, 1000 * (s + 1) + 7));
  const idx = [100, 400, 1600, 12800].map((n) => M.N_LIST.indexOf(n));
  const mean = (field, i) => runs.reduce((a, r) => a + r[field][i].mean, 0) / runs.length;
  const lo = (field, i) => Math.min(...runs.map((r) => r[field][i].min));
  const hi = (field, i) => Math.max(...runs.map((r) => r[field][i].max));
  summary[`${lk}-${d}`] = { first: idx.map((i) => mean("first", i)), second: idx.map((i) => mean("second", i)) };
  console.log(`  d=${String(d).padEnd(2)} ${lk.padEnd(6)} ` + idx.map((i, q) => `n=${[100, 400, 1600, 12800][q]}: ${f(mean("first", i), 2)} [${f(lo("first", i), 2)},${f(hi("first", i), 2)}] / ${f(mean("second", i), 2)} [${f(lo("second", i), 2)},${f(hi("second", i), 2)}]`).join("  "));
}
// estimators whose population target is nonzero converge; the others stay low
for (const d of [10, 20, 40]) {
  for (const lk of ["linear", "tanh", "relu"]) assert.ok(summary[`${lk}-${d}`].first[3] > 0.98, `first ${lk} ${d}`);
  for (const lk of ["relu", "abs", "he2"]) assert.ok(summary[`${lk}-${d}`].second[3] > 0.97, `second ${lk} ${d}`);
  for (const lk of ["abs", "he2"]) assert.ok(summary[`${lk}-${d}`].first[3] < 0.5, `first floor ${lk} ${d}`);
  for (const lk of ["linear", "tanh"]) assert.ok(summary[`${lk}-${d}`].second[3] < 0.65, `second floor ${lk} ${d}`);
}
// the floors do not climb with n
for (const lk of ["abs", "he2"]) for (const d of [10, 20, 40]) {
  const s = summary[`${lk}-${d}`].first; assert.ok(s[3] < s[0] + 0.1, `first-order floor flat ${lk} ${d}`);
}
for (const lk of ["linear", "tanh"]) for (const d of [10, 20, 40]) {
  // the second-order floor is set by anisotropic noise (the row and column of w are sqrt 3 noisier); it never nears 1
  summary[`${lk}-${d}`].second.forEach((v) => assert.ok(v < 0.65, `second-order floor ${lk} ${d}`));
}
// the n at which ReLU's mean overlap first reaches 0.9, by d
for (const field of ["first", "second"]) {
  const row = [10, 20, 40].map((d) => {
    const runs = []; for (let s = 0; s < 5; s++) runs.push(M.sweep("relu", d, 0.5, 6, 1000 * (s + 1) + 7));
    const i = M.N_LIST.findIndex((_, k) => runs.reduce((a, r) => a + r[field][k].mean, 0) / runs.length >= 0.9);
    return `d=${d}: n=${M.N_LIST[i]}`;
  });
  console.log(`ReLU ${field}-order mean overlap first reaches 0.9 at: ${row.join(", ")}`);
}
// default view of the widget (seed 1 -> s0 = 1007)
{
  const r = M.sweep("relu", 20, 0.5, 6, 1007);
  console.log(`widget default (ReLU, d 20, sigma 0.5, n 400): first ${f(r.first[4].mean, 2)}, second ${f(r.second[4].mean, 2)}; at n 12800: ${f(r.first[9].mean, 2)}, ${f(r.second[9].mean, 2)}`);
}
console.log("\nall assertions passed");
