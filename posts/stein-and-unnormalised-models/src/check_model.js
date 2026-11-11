/* Checks the Part 4 widget models and prints every number the post quotes.
 * Run: node posts/stein-and-unnormalised-models/src/check_model.js */
"use strict";

const assert = require("node:assert/strict");
const M = require("../model.js");
const f = (x, d = 3) => x.toFixed(d);
const range = (xs, d = 3) => `[${f(Math.min(...xs), d)}, ${f(Math.max(...xs), d)}]`;

// 1. The score is d/dx log p~.
for (let x = -5; x <= 5; x += 0.37) {
  const h = 1e-5, fd = (Math.log(M.ptilde(x + h)) - Math.log(M.ptilde(x - h))) / (2 * h);
  assert.ok(Math.abs(fd - M.score(x)) < 1e-6);
}

// 2. Multiplying p~ by c moves the mass and the log-likelihood, and nothing score-based.
{
  const xs = M.sample(60, 4), base = M.constantStats(1, xs);
  for (const lc of [-3, -1.5, 0.7, 3]) {
    const c = 10 ** lc, s = M.constantStats(c, xs);
    assert.equal(s.ksd, base.ksd);
    assert.equal(s.scoreMatching, base.scoreMatching);
    assert.ok(Math.abs(s.loglik - base.loglik - 60 * Math.log(c)) < 1e-9);
    assert.ok(Math.abs(s.mass - c * M.Z) < 1e-9 * c * M.Z);
  }
  const lo = M.constantStats(1e-3, xs), hi = M.constantStats(1e3, xs);
  console.log(`constant: Z = ${f(M.Z, 4)}; c from 1e-3 to 1e3 moves the mass ${lo.mass.toExponential(3)} -> ${hi.mass.toExponential(3)}, ` +
    `log-likelihood ${f(lo.loglik, 2)} -> ${f(hi.loglik, 2)} (60 ln 10^6 = ${f(60 * Math.log(1e6), 2)}); KSD ${f(base.ksd, 6)} and J ${f(base.scoreMatching, 6)} at every c`);
}

// 3. SURE is unbiased for the risk: average SURE - loss over noise draws is 0 within 4 standard errors.
{
  const mu = M.signal(256, 25, 1), D = 3000;
  for (const kind of ["soft", "ridge"]) for (const sigma of [0.5, 1, 2]) {
    for (const lamF of [0.1, 0.3, 0.6]) {
      const lam = lamF * (kind === "soft" ? 4 * sigma : 3);
      let m = 0, m2 = 0, risk = 0;
      for (let s = 0; s < D; s++) {
        const y = M.observe(mu, sigma, 5000 + s), d = M.sure(kind, y, lam, sigma) - M.loss(kind, y, mu, lam);
        m += d; m2 += d * d; risk += M.loss(kind, y, mu, lam);
      }
      m /= D; const se = Math.sqrt((m2 / D - m * m) / D);
      assert.ok(Math.abs(m) < 4 * se, `${kind} sigma=${sigma} lam=${lam}: mean diff ${m}, se ${se}`);
    }
  }
  console.log(`SURE - loss averages to 0 within 4 SE over ${D} noise draws, for both estimators, three sigmas, three lambdas`);
}

// 4. Numbers for the prose: SURE's lambda against the oracle's, over 8 noise draws.
{
  const mu = M.signal(256, 25, 1);
  for (const kind of ["soft", "ridge"]) for (const sigma of [0.5, 1, 2]) {
    const ratio = [], ls = [], lo = [], raw = [], best = [];
    for (let s = 1; s <= 8; s++) {
      const y = M.observe(mu, sigma, 100 + s), c = M.curves(kind, mu, y, sigma);
      ratio.push(c.lossAtSure / c.lossAtOracle); ls.push(c.lamSure); lo.push(c.lamOracle); raw.push(c.lossAtZero); best.push(c.lossAtOracle);
      assert.ok(c.lossAtSure < c.lossAtZero);
    }
    console.log(`${kind.padEnd(5)} sigma ${sigma}: lam_SURE ${range(ls, 2)}, lam_oracle ${range(lo, 2)}, loss ratio ${range(ratio)}, ` +
      `raw loss ${range(raw, 0)}, oracle loss ${range(best, 0)}`);
  }
  // the default view: seed 1, soft, sigma 1
  const y = M.observe(mu, 1, 101), c = M.curves("soft", mu, y, 1);
  console.log(`default view (soft, sigma 1, first draw): lam_SURE ${f(c.lamSure, 2)} (loss ${f(c.lossAtSure, 1)}), ` +
    `lam_oracle ${f(c.lamOracle, 2)} (loss ${f(c.lossAtOracle, 1)}), raw loss ${f(c.lossAtZero, 1)}`);
  const r = M.curves("ridge", mu, y, 1);
  console.log(`  same draw, ridge: lam_SURE ${f(r.lamSure, 2)} (loss ${f(r.lossAtSure, 1)}), oracle loss ${f(r.lossAtOracle, 1)}`);
}
console.log("checks pass");

// 5. The whole sigma slider (0.25 to 2 in steps of 0.05), eight draws each: worst loss ratio at SURE's lambda,
//    and the lambda gaps at sigma = 1 for both estimators.
{
  const mu = M.signal(256, 25, 1);
  let worst = 0, at = 0, lowWorst = 0;
  for (let k = 0; k <= 35; k++) {
    const sigma = Math.round((0.25 + 0.05 * k) * 100) / 100;
    for (let s = 1; s <= 8; s++) {
      const c = M.curves("soft", mu, M.observe(mu, sigma, 100 + s), sigma), r = c.lossAtSure / c.lossAtOracle;
      if (r > worst) { worst = r; at = sigma; }
      if (sigma <= 0.55) lowWorst = Math.max(lowWorst, r);
    }
  }
  const gap = (kind) => Math.max(...[1, 2, 3, 4, 5, 6, 7, 8].map((s) => { const c = M.curves(kind, mu, M.observe(mu, 1, 100 + s), 1); return Math.abs(c.lamSure - c.lamOracle); }));
  console.log(`soft threshold over the whole sigma slider: worst loss ratio ${f(worst)} at sigma ${at} (${f(lowWorst)} for sigma <= 0.55); ` +
    `largest |lam_SURE - lam_oracle| at sigma 1: soft ${f(gap("soft"), 2)}, ridge ${f(gap("ridge"), 2)}`);
}

// 6. The first draw (the page's default, seed 1) across the sigma slider: lambda gap and loss ratio.
{
  const mu = M.signal(256, 25, 1), rows = [];
  let worstGapUpTo175 = 0;
  for (let k = 0; k <= 35; k++) {
    const sigma = Math.round((0.25 + 0.05 * k) * 100) / 100;
    const c = M.curves("soft", mu, M.observe(mu, sigma, 101), sigma);
    const gap = Math.abs(c.lamSure - c.lamOracle) / sigma, ratio = c.lossAtSure / c.lossAtOracle;
    if (sigma <= 1.75) worstGapUpTo175 = Math.max(worstGapUpTo175, ratio);
    if ([0.5, 1, 1.5, 1.75, 1.8, 1.9, 2].includes(sigma)) rows.push(`s=${sigma}: lam ${f(c.lamSure, 2)}/${f(c.lamOracle, 2)} ratio ${f(ratio)}`);
  }
  console.log(`first draw across sigma: ${rows.join("; ")}; worst loss ratio for sigma <= 1.75: ${f(worstGapUpTo175)}`);
}

// 7. First draw, sigma <= 1.75: the largest gap between the two lambda values (item 4 says "up to 0.32").
{
  const mu = M.signal(256, 25, 1);
  let gap = 0, at = 0;
  for (let k = 0; k <= 30; k++) {
    const sigma = Math.round((0.25 + 0.05 * k) * 100) / 100;
    const c = M.curves("soft", mu, M.observe(mu, sigma, 101), sigma);
    if (Math.abs(c.lamSure - c.lamOracle) > gap) { gap = Math.abs(c.lamSure - c.lamOracle); at = sigma; }
  }
  console.log(`first draw, sigma <= 1.75: largest |lam_SURE - lam_oracle| = ${f(gap, 2)} at sigma ${at}`);
}
