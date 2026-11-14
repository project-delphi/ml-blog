// Checks model.js and prints every number the post quotes from the interactive chart,
// over the seeds a reader reaches with "New draw" (1-5) and three batch sizes.
//
//   node posts/kernel-stein-discrepancy-as-a-regulariser/src/check_model.js
//
// Exits non-zero if a gradient check or a stated claim fails.
"use strict";
const path = require("path");
const M = require(path.join(__dirname, "..", "model.js"));

const OPT = { ustat: false }; // the chart uses V-statistics (see widgets.js)
let failed = 0;
function check(ok, msg) { if (!ok) { failed++; console.log("FAIL", msg); } }
function range(xs) { return "[" + Math.min(...xs).toFixed(2) + ", " + Math.max(...xs).toFixed(2) + "]"; }

// 1. Gradients against central differences, U and V.
for (const reg of M.REGS) for (const ustat of [true, false]) {
  const X = M.sample("clusters", 30, 5);
  const o = { seed: 3, ustat };
  const f = (Z) => M.evaluate(reg, Z, o).v, g = M.evaluate(reg, X, o).g;
  let worst = 0;
  for (let i = 0; i < X.length; i++) {
    const h = 1e-5, A = Float64Array.from(X), B = Float64Array.from(X);
    A[i] += h; B[i] -= h;
    worst = Math.max(worst, Math.abs((f(A) - f(B)) / (2 * h) - g[i]) / (1e-8 + M.maxNorm(g)));
  }
  check(worst < 1e-6, `${reg} ${ustat ? "U" : "V"} gradient relative error ${worst}`);
}

// 2. The V-statistic of the linear-kernel KSD is |mean|^2 + |M2 - I|_F^2, by brute force.
{
  const X = M.sample("square", 40, 2), n = 40;
  let s = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const xy = X[2 * i] * X[2 * j] + X[2 * i + 1] * X[2 * j + 1];
    const xi = X[2 * i] ** 2 + X[2 * i + 1] ** 2, xj = X[2 * j] ** 2 + X[2 * j + 1] ** 2;
    s += xy * (1 + xy) - xi - xj + 2;
  }
  check(Math.abs(s / (n * n) - M.moments(X, OPT).v) < 1e-10, "linear-kernel V-statistic != moment penalty");
}

// 3. Bars: z-scores against Gaussian batches (80 of them, seed 1, as the chart), seeds 1-5.
const SEEDS = [1, 2, 3, 4, 5], NS = [100, 150, 200];
const nulls = {};
function zscore(reg, X, n, seed) {
  const key = reg + "|" + n;
  nulls[key] = nulls[key] || M.nullStats(reg, n, 80, 1, OPT);
  const ns = nulls[key], v = M.evaluate(reg, X, Object.assign({ seed: seed * 17 + 3 }, OPT)).v;
  return { z: (v - ns.mean) / ns.sd, z95: (ns.q95 - ns.mean) / ns.sd };
}
console.log("\nz-scores (null SDs above a Gaussian batch), range over seeds 1-5:");
for (const n of NS) for (const law of ["ring", "clusters", "square", "student", "gaussian", "shrunk"]) {
  const row = [];
  for (const reg of M.REGS) {
    const zs = SEEDS.map((s) => zscore(reg, M.sample(law, n, s), n, s));
    const hits = zs.filter((r) => r.z > r.z95).length;
    row.push(`${reg} ${range(zs.map((r) => r.z))} (${hits}/5 past tick)`);
    if (law === "ring") {
      if (reg === "moments") check(hits === 0, `moments flags ${law} at n=${n}`);
      else check(hits === 5, `${reg} misses ${law} at n=${n}`);
    }
    if (law === "shrunk") check(hits === 5, `${reg} misses shrunk at n=${n}`);
  }
  console.log(`n=${n} ${law.padEnd(8)} ` + row.join(" | "));
}

// 4. Descent, 3000 steps at the chart's step size: mean |z|^2 and |z|^4 (Gaussian: 2, 8).
console.log("\nafter 3000 descent steps (n = 150), range over seeds 1-5:");
for (const law of ["ring", "clusters"]) for (const reg of M.REGS) {
  const r2 = [], r4 = [], start = [];
  for (const s of SEEDS) {
    const n = 150, X = M.sample(law, n, s);
    start.push(M.radialMoments(X).r4);
    const gref = M.maxNorm(M.evaluate(reg, M.sample("shrunk", n, 424242), Object.assign({ seed: 5 }, OPT)).g);
    M.descend(X, reg, 3000, 0.02, gref, Object.assign({ nrm: M.normalGen(M.rng(s * 31 + 7)) }, OPT));
    const rm = M.radialMoments(X);
    r2.push(rm.r2); r4.push(rm.r4);
  }
  console.log(`${law.padEnd(8)} ${reg.padEnd(8)} |z|^4 ${range(start)} -> ${range(r4)}   |z|^2 -> ${range(r2)}`);
  if (reg === "moments") check(Math.max(...r4) < 6, `moments reshaped the ${law}`);
  if (law === "ring" && reg !== "moments") check(Math.min(...r4) > 5.5, `${reg} left the ring a ring`);
}

console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
process.exit(failed ? 1 : 0);
