/* Checks the SVGD / KSD widget model and prints every number the post quotes.
 * Run: node posts/kernel-stein-discrepancy-and-svgd/src/check_model.js */
"use strict";

const assert = require("node:assert/strict");
const M = require("../model.js");
const f3 = (x) => (Number.isFinite(x) ? x.toFixed(3) : "NaN");
const range = (xs) => `[${f3(Math.min(...xs))}, ${f3(Math.max(...xs))}]`;
const SEEDS = [1, 2, 3, 4, 5];

// 1. Scores are the gradients of log p~ (finite differences).
for (const key of ["banana", "mixture", "donut", "gauss"]) {
  const T = M.target(key, { sep: 5 }), r = M.rng(9), out = [0, 0], h = 1e-5;
  for (let k = 0; k < 200; k++) {
    const x = 8 * r() - 4, y = 8 * r() - 4;
    T.score(x, y, out);
    const gx = (T.logp(x + h, y) - T.logp(x - h, y)) / (2 * h), gy = (T.logp(x, y + h) - T.logp(x, y - h)) / (2 * h);
    assert.ok(Math.abs(out[0] - gx) < 1e-4 * (1 + Math.abs(gx)) && Math.abs(out[1] - gy) < 1e-4 * (1 + Math.abs(gy)), key);
  }
}

// 2. The IMQ Stein kernel matches its definition with numerically differentiated k.
{
  const c = 1, beta = -0.5, h = 1e-4;
  const k = (x, y) => Math.pow(c * c + (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2, beta);
  const r = M.rng(3);
  for (let t = 0; t < 100; t++) {
    const x = [4 * r() - 2, 4 * r() - 2], y = [4 * r() - 2, 4 * r() - 2], sx = [r() - 0.5, r() - 0.5], sy = [r() - 0.5, r() - 0.5];
    const gx = [0, 1].map((i) => { const a = x.slice(), b = x.slice(); a[i] += h; b[i] -= h; return (k(a, y) - k(b, y)) / (2 * h); });
    const gy = [0, 1].map((i) => { const a = y.slice(), b = y.slice(); a[i] += h; b[i] -= h; return (k(x, a) - k(x, b)) / (2 * h); });
    let tr = 0;
    for (const i of [0, 1]) {
      const pp = (dx, dy) => { const a = x.slice(), b = y.slice(); a[i] += dx; b[i] += dy; return k(a, b); };
      tr += (pp(h, h) - pp(h, -h) - pp(-h, h) + pp(-h, -h)) / (4 * h * h);
    }
    const direct = (sx[0] * sy[0] + sx[1] * sy[1]) * k(x, y) + sx[0] * gy[0] + sx[1] * gy[1] + sy[0] * gx[0] + sy[1] * gx[1] + tr;
    assert.ok(Math.abs(direct - M.steinKernelIMQ(x, y, sx, sy, c, beta)) < 1e-5);
  }
}

// 3. KSD of exact N(0, I) draws falls with n, at about the 1/sqrt(n) rate of a V-statistic.
{
  const T = M.target("gauss"), means = {};
  for (const n of [50, 200, 800]) {
    const ks = SEEDS.map((s) => M.ksd(M.init(n, 1, s), T));
    means[n] = ks.reduce((a, b) => a + b) / ks.length;
  }
  assert.ok(means[200] < means[50] && means[800] < means[200]);
  console.log(`KSD of exact N(0,I) draws: n=50 ${f3(means[50])}, n=200 ${f3(means[200])}, n=800 ${f3(means[800])}`);
}

// 4. Defaults (banana, n = 100, eps = 0.05, median heuristic, spread 1, 500 iterations).
const base = { target: "banana", n: 100, spread: 1, eps: 0.05, steps: 500, every: 500 };
{
  const a = [], b = [], a0 = [];
  for (const seed of SEEDS) {
    const r = M.race({ ...base, seed });
    a0.push(r.trace[0].svgd); a.push(r.trace[1].svgd); b.push(r.trace[1].ula);
    assert.ok(r.trace[1].svgd < r.trace[1].ula);
  }
  console.log(`banana default: start ${range(a0)}, SVGD after 500 its ${range(a)}, ULA ${range(b)}`);
  // i.i.d. floor: KSD of 100 exact banana draws
  const T = M.target("banana"), fl = SEEDS.map((s) => {
    const nrm = M.normalGen(M.rng(s)), P = new Float64Array(200);
    for (let i = 0; i < 100; i++) { const x = 1.5 * nrm(); P[2 * i] = x; P[2 * i + 1] = 0.6 * x * x - 1.5 + 0.6 * nrm(); }
    return M.ksd(P, T);
  });
  console.log(`  KSD of 100 exact banana draws: ${range(fl)}`);
}

// 5. Step size: ULA diverges on the banana at eps = 0.1; SVGD stays finite up to 0.2.
{
  for (const seed of SEEDS) {
    const r = M.race({ ...base, eps: 0.1, seed });
    assert.ok(!Number.isFinite(r.trace[1].ula) || r.trace[1].ula > 1e3, `ULA did not diverge, seed ${seed}`);
    const s = M.race({ ...base, eps: 0.2, seed });
    assert.ok(Number.isFinite(s.trace[1].svgd), `SVGD diverged at 0.2, seed ${seed}`);
  }
  const ok = SEEDS.map((seed) => M.race({ ...base, eps: 0.08, seed }).trace[1].ula);
  console.log(`ULA on banana diverges at eps = 0.1 for all 5 seeds; at eps = 0.08 its KSD is ${range(ok)}`);
}

// 6. Other targets at the defaults.
for (const tg of ["donut", "mixture"]) {
  const a = [], b = [];
  for (const seed of SEEDS) { const r = M.race({ ...base, target: tg, sep: 4, seed }); a.push(r.trace[1].svgd); b.push(r.trace[1].ula); assert.ok(r.trace[1].svgd < r.trace[1].ula); }
  console.log(`${tg} (sep 4): SVGD ${range(a)}, ULA ${range(b)}`);
}

// 7. Bandwidth: median heuristic vs fixed; a tiny bandwidth stalls.
{
  for (const ell of [0.05, 0.5, 2]) {
    const ks = SEEDS.map((seed) => M.race({ ...base, seed, median: false, ell }).trace[1].svgd);
    console.log(`fixed ell = ${ell}: KSD after 500 its ${range(ks)}`);
    if (ell === 0.05) assert.ok(Math.min(...ks) > 0.6);
  }
  const ells = SEEDS.map((seed) => M.medianEll(M.race({ ...base, seed }).svgd));
  console.log(`median-heuristic ell after 500 its: ${range(ells)}`);
}

// 8. Mixture blindness: sep 8, particles started on the light mode.
{
  const right = [], ks = [], spread = [];
  for (const seed of SEEDS) {
    const r = M.race({ ...base, target: "mixture", sep: 8, centre: [4, 0], steps: 1000, every: 1000, seed });
    right.push(M.rightFraction(r.svgd)); ks.push(r.trace[1].svgd);
    const r2 = M.race({ ...base, target: "mixture", sep: 8, steps: 1000, every: 1000, seed });
    spread.push(M.rightFraction(r2.svgd));
  }
  assert.ok(Math.min(...right) === 1);
  const T = M.target("mixture", { sep: 8 }), exact = SEEDS.map((s) => {
    const r = M.rng(s), nrm = M.normalGen(r), P = new Float64Array(200);
    for (let i = 0; i < 100; i++) { P[2 * i] = (r() < M.MIX_W ? -4 : 4) + nrm(); P[2 * i + 1] = nrm(); }
    return M.ksd(P, T);
  });
  assert.ok(Math.max(...ks) < Math.min(...exact));
  console.log(`mixture sep 8, start x0 = 4: share right ${range(right)} (target ${M.mixtureRightMass(8).toFixed(2)}), KSD ${range(ks)}; ` +
    `KSD of 100 exact draws ${range(exact)}; start x0 = 0: share right ${range(spread)}`);
}
console.log("checks pass");

// 9. Early on, ULA is ahead: find the iteration after which SVGD's KSD stays below ULA's.
{
  const cross = [], early = [], firstBelow = [];
  for (const seed of SEEDS) {
    const r = M.race({ ...base, seed, every: 1 });
    let last = 0, first = null;
    for (const p of r.trace) { if (!(p.svgd < p.ula)) last = p.t; else if (first === null) first = p.t; }
    cross.push(last + 1); firstBelow.push(first);
    const at50 = r.trace.find((p) => p.t === 50);
    early.push(`${f3(at50.svgd)}/${f3(at50.ula)}`);
  }
  console.log(`banana default: at iteration 50 SVGD/ULA KSD = ${early.join(", ")}; SVGD first below ULA at iterations ${firstBelow.join(", ")}; stays below from iterations ${cross.join(", ")}`);
}

// 10. Item 3's bandwidth ranges over 20 seeds (a reader pressing Reset repeatedly walks through seeds).
{
  const seeds20 = Array.from({ length: 20 }, (_, i) => i + 1);
  for (const ell of [0.05, 0.5, 2]) {
    const ks = seeds20.map((seed) => M.race({ ...base, seed, median: false, ell }).trace[1].svgd);
    if (ell === 0.05) assert.ok(Math.min(...ks) > 0.6);
    console.log(`fixed ell = ${ell}, 20 seeds: KSD after 500 its ${range(ks)}`);
  }
  const med = seeds20.map((seed) => M.race({ ...base, seed }).trace[1].svgd);
  console.log(`median heuristic, 20 seeds: KSD after 500 its ${range(med)}`);
}

// 11. Items 4 and 5 over 20 seeds: by then a reader has pressed Reset several times, and each Reset
//     moves to the next seed, so five seeds would not cover what they see.
{
  const seeds20 = Array.from({ length: 20 }, (_, i) => i + 1);
  const right = [], ks = [], spread = [];
  for (const seed of seeds20) {
    const r = M.race({ ...base, target: "mixture", sep: 8, centre: [4, 0], steps: 1000, every: 1000, seed });
    right.push(M.rightFraction(r.svgd)); ks.push(r.trace[1].svgd);
    const r2 = M.race({ ...base, target: "mixture", sep: 8, steps: 1000, every: 1000, seed });
    spread.push(M.rightFraction(r2.svgd));
  }
  assert.ok(Math.min(...right) === 1);
  const T = M.target("mixture", { sep: 8 }), exact = seeds20.map((s) => {
    const r = M.rng(s), nrm = M.normalGen(r), P = new Float64Array(200);
    for (let i = 0; i < 100; i++) { P[2 * i] = (r() < M.MIX_W ? -4 : 4) + nrm(); P[2 * i + 1] = nrm(); }
    return M.ksd(P, T);
  });
  assert.ok(Math.max(...ks) < Math.min(...exact));
  const sorted = [...spread].sort((a, b) => a - b), median = (sorted[9] + sorted[10]) / 2;
  assert.equal(median.toFixed(2), "0.43"); // item 5's "median of 0.43"
  console.log(`20 seeds, mixture sep 8, start x0 = 4: share right ${range(right)}, KSD ${range(ks)}; ` +
    `KSD of 100 exact draws ${range(exact)}; start x0 = 0: share right (the lighter mode) ${range(spread)}, median ${median.toFixed(3)}`);
}

// 12. Item 2's SVGD claim: finite at eps = 0.2 from the default start on 20 seeds; started at x0 = 5
//     it can diverge, which the widget reports as "diverged".
{
  const finite = Array.from({ length: 20 }, (_, i) => M.race({ ...base, eps: 0.2, seed: i + 1 }).trace[1].svgd).filter(Number.isFinite).length;
  assert.equal(finite, 20);
  const far = Array.from({ length: 10 }, (_, i) => M.race({ ...base, eps: 0.2, centre: [5, 0], seed: i + 1 }).svgd)
    .filter((A) => A.some((v) => !Number.isFinite(v) || Math.abs(v) > 1e6)).length;
  assert.ok(far > 0);
  console.log(`SVGD at eps = 0.2: finite on ${finite}/20 seeds from the default start; diverges on ${far}/10 from x0 = 5`);
}
