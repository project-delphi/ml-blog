/* Checks the Stein's-lemma widget model and prints every number the post quotes.
 * Run: node posts/steins-lemma/src/check_model.js */
"use strict";

const assert = require("node:assert/strict");
const M = require("../model.js");

const DISTS = ["gauss", "laplace", "t", "unif"];
const FNS = ["sin", "tanh", "cube"];
const OPS = ["gauss", "langevin"];
const NUS = [2.5, 3, 4.5, 5, 9, 10, 30];
const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8];

// 1. Quadrature agrees with every closed form.
for (const d of DISTS) {
  for (const nu of d === "t" ? NUS : [5]) {
    const D = M.dist(d, nu);
    // density integrates to 1 and the variance is 1 (f(x) = x under the Gaussian operator gives E[1 - X^2] = 0)
    const lin = M.quadrature("gauss", D, { f: (x) => x, df: () => 1 });
    if (!(d === "t" && nu <= 2.5)) assert.ok(Math.abs(lin) < 2e-4, `variance of ${d} nu=${nu}: ${lin}`);
    for (const fn of FNS) {
      for (const op of OPS) {
        const c = M.conditions(op, d, nu, fn);
        if (!c.mean || !c.variance) continue;
        const q = M.quadrature(op, D, M.testFn(fn));
        const e = M.exact(op, d, nu, fn);
        assert.ok(Math.abs(q - e) < 2e-4, `quadrature ${op} ${d} nu=${nu} ${fn}: ${q} vs ${e}`);
      }
    }
  }
}

// 2. The Langevin operator gives 0 for every law on the whole line, and the boundary term for the uniform.
for (const fn of FNS) {
  for (const d of ["gauss", "laplace", "t"]) assert.equal(M.exact("langevin", d, 5, fn), 0);
  const b = M.conditions("langevin", "unif", 5, fn).boundary;
  assert.ok(Math.abs(b) > 0.5, `uniform boundary term for ${fn} is ${b}`);
}

// 3. Monte Carlo lands within 4 SE of the exact value wherever the variance is finite.
let runs = 0, inside2 = 0;
const finals = {};
for (const op of OPS) {
  for (const d of DISTS) {
    for (const fn of FNS) {
      const nu = 10;
      const c = M.conditions(op, d, nu, fn);
      if (!c.variance) continue;
      const e = M.exact(op, d, nu, fn);
      const est = [];
      for (const seed of SEEDS) {
        const r = M.run({ dist: d, nu, fn, op, seed, N: 100000 });
        const z = Math.abs(r.final.mean - e) / r.final.se;
        assert.ok(z < 4.5, `MC ${op} ${d} ${fn} seed ${seed}: z = ${z.toFixed(2)}`);
        runs++; if (z < 2) inside2++;
        est.push(r.final.mean);
      }
      finals[`${op} ${d} ${fn}`] = { exact: e, lo: Math.min(...est), hi: Math.max(...est) };
    }
  }
}

// 4. The custom parser: matches the built-ins, rejects anything outside the grammar.
const cube = M.parse("x^3"), mix = M.parse("sin(x)*exp(-x^2/4) + 2*x");
for (const x of [-2, -0.3, 0, 1.7]) {
  assert.ok(Math.abs(cube(x) - x ** 3) < 1e-12);
  assert.ok(Math.abs(mix(x) - (Math.sin(x) * Math.exp(-x * x / 4) + 2 * x)) < 1e-12);
}
for (const bad of ["alert(1)", "x;1", "window", "2x", "", "x^"]) assert.throws(() => M.parse(bad), `parser accepted ${bad}`);
const custom = M.testFn("custom", "x^3");
assert.ok(Math.abs(custom.df(1.3) - 3 * 1.3 * 1.3) < 1e-6);
assert.ok(Math.abs(M.exact("gauss", "laplace", 5, "custom", "x^3") - -3) < 1e-4);

// 5. Under x^3 the Gaussian operator's expectation exists only for nu > 4.
assert.equal(M.exact("gauss", "t", 4, "cube"), null);
assert.ok(Math.abs(M.exact("gauss", "t", 5, "cube") - -6) < 1e-12);

console.log(`checks pass: ${runs} Monte Carlo runs, ${inside2} inside 2 SE (${(100 * inside2 / runs).toFixed(0)}%)`);
console.log("\nExact expectations (nu = 10 for Student-t):");
for (const op of OPS) for (const d of DISTS) {
  const row = FNS.map((fn) => { const e = M.exact(op, d, 10, fn); return `${fn} ${e === null ? "undef" : e.toFixed(4)}`; });
  console.log(`  ${op.padEnd(8)} ${d.padEnd(7)} ${row.join("   ")}`);
}
console.log("\nStudent-t, x^3, Gaussian operator: -6/(nu-4)");
for (const nu of [4.5, 5, 6, 10, 30]) console.log(`  nu=${nu}: ${M.exact("gauss", "t", nu, "cube").toFixed(4)}`);
console.log("\nEstimate ranges over 8 seeds at n = 100,000 (nu = 10):");
for (const [k, v] of Object.entries(finals)) console.log(`  ${k.padEnd(24)} exact ${v.exact.toFixed(4)}  seeds [${v.lo.toFixed(4)}, ${v.hi.toFixed(4)}]`);

// 6. Claims made in the post's "Things to try" list.
for (const d of DISTS) {
  // f(x) = x: E[1 - X^2] = 0 for every standardised law, under the Gaussian operator
  const e = M.exact("gauss", d, 10, "custom", "x");
  assert.ok(Math.abs(e) < 2e-4, `custom x under ${d}: ${e}`);
}
// (3 - x^2) x^3 vanishes at ±√3, so the uniform's boundary term is 0
assert.ok(Math.abs(M.exact("langevin", "unif", 10, "custom", "(3-x^2)*x^3")) < 1e-6);
// below nu = 4 the Gaussian-operator mean of x^3 does not exist: the running mean drifts down
const drift = SEEDS.map((seed) => M.run({ dist: "t", nu: 3, fn: "cube", op: "gauss", seed, N: 100000 }));
const at = (r, n) => r.path.find((p) => p.n >= n).mean;
console.log("\nnu = 3, x^3, Gaussian operator, running mean at n = 1k / 10k / 100k by seed:");
for (const r of drift) console.log(`  ${at(r, 1000).toFixed(2)}  ${at(r, 10000).toFixed(2)}  ${r.final.mean.toFixed(2)}`);
const lang3 = SEEDS.map((seed) => M.run({ dist: "t", nu: 3, fn: "cube", op: "langevin", seed, N: 100000 }).final.mean);
console.log(`nu = 3, x^3, Langevin operator, final means over seeds: [${Math.min(...lang3).toFixed(3)}, ${Math.max(...lang3).toFixed(3)}]`);
console.log("post claims pass");

// 7. Student-t quadrature near nu = 2.5 (heavy tails): custom f must still give the exact zeros.
for (const nu of [2.5, 2.75, 3]) {
  const a = M.exact("gauss", "t", nu, "custom", "x"), b = M.exact("langevin", "t", nu, "custom", "x^3");
  assert.ok(Math.abs(a) < 1e-5 && Math.abs(b) < 1e-5, `t quadrature at nu=${nu}: ${a}, ${b}`);
}
console.log("heavy-tail quadrature checks pass");
