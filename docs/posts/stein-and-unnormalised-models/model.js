/* Two models for Part 4. Numbers only; widgets.js draws them and
 * src/check_model.js checks them under node.
 *  1. An unnormalised 1-D density multiplied by c: what changes with c and what does not.
 *  2. SURE against the true loss for soft-threshold and ridge denoising of a sparse signal. */
"use strict";

var SteinModelsModel = (function () {
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normalGen(r) {
    var spare = null;
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var u = 0, v = r();
      while (u === 0) u = r();
      var rad = Math.sqrt(-2 * Math.log(u)), ang = 2 * Math.PI * v;
      spare = rad * Math.sin(ang);
      return rad * Math.cos(ang);
    };
  }

  // ------------------------------------------------- 1. the constant c
  // p~(x) = 0.6 exp(-(x + 1.5)^2 / (2 * 0.7^2)) + 0.4 exp(-(x - 1.5)^2 / 2): no 1/sqrt(2 pi) factors.
  var A1 = 0.6, M1 = -1.5, S1 = 0.7, A2 = 0.4, M2 = 1.5, S2 = 1.0;
  function ptilde(x) {
    return A1 * Math.exp(-0.5 * (x - M1) * (x - M1) / (S1 * S1)) + A2 * Math.exp(-0.5 * (x - M2) * (x - M2) / (S2 * S2));
  }
  /** The score of c p~, computed from c p~ itself as (c p~)'/(c p~). The factor c
   *  multiplies numerator and denominator and cancels, so every c > 0 gives the
   *  same function up to rounding in the last binary digit. */
  function scoreOf(cMul) {
    return function (x) {
      var e1 = cMul * A1 * Math.exp(-0.5 * (x - M1) * (x - M1) / (S1 * S1)),
        e2 = cMul * A2 * Math.exp(-0.5 * (x - M2) * (x - M2) / (S2 * S2));
      return (e1 * (-(x - M1) / (S1 * S1)) + e2 * (-(x - M2) / (S2 * S2))) / (e1 + e2);
    };
  }
  /** d/dx of a score, by central difference, for the score-matching objective. */
  function derivative(sc) { return function (x) { var h = 1e-5; return (sc(x + h) - sc(x - h)) / (2 * h); }; }
  var score = scoreOf(1), dscore = derivative(score);
  /** The normalising constant of p~, for display only: the methods never use it. */
  var Z = Math.sqrt(2 * Math.PI) * (A1 * S1 + A2 * S2);

  /** A fixed sample of size n from the normalised density. */
  function sample(n, seed) {
    var r = rng(seed), nrm = normalGen(r), out = [];
    var w1 = A1 * S1 / (A1 * S1 + A2 * S2);
    for (var i = 0; i < n; i++) out.push(r() < w1 ? M1 + S1 * nrm() : M2 + S2 * nrm());
    return out;
  }

  /** 1-D Stein kernel of the IMQ base kernel (c^2 + r^2)^beta; V-statistic KSD. */
  function ksd1(xs, sc, c, beta) {
    c = c === undefined ? 1 : c; beta = beta === undefined ? -0.5 : beta;
    var n = xs.length, s = xs.map(sc), sum = 0;
    for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
      var r = xs[i] - xs[j], q = c * c + r * r, qb = Math.pow(q, beta), qb1 = qb / q, qb2 = qb1 / q;
      sum += s[i] * s[j] * qb - 2 * beta * qb1 * s[i] * r + 2 * beta * qb1 * s[j] * r
        - 4 * beta * (beta - 1) * qb2 * r * r - 2 * beta * qb1;
    }
    return Math.sqrt(Math.max(0, sum) / (n * n));
  }

  /** Everything the first chart reports for a multiplier c. */
  function constantStats(cMul, xs) {
    var ll = 0, sm = 0, sc = scoreOf(cMul), dsc = derivative(sc);
    xs.forEach(function (x) {
      ll += Math.log(cMul * ptilde(x));
      var s = sc(x);
      sm += 0.5 * s * s + dsc(x);
    });
    return { mass: cMul * Z, loglik: ll, ksd: ksd1(xs, sc), scoreMatching: sm / xs.length };
  }

  // ------------------------------------------------- 2. SURE
  /** A synthetic sparse signal: d coordinates, k of them nonzero with |mu| in [2, 6]. */
  function signal(d, k, seed) {
    var r = rng(seed), mu = new Float64Array(d), idx = [];
    for (var i = 0; i < d; i++) idx.push(i);
    for (i = d - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    for (i = 0; i < k; i++) mu[idx[i]] = (r() < 0.5 ? -1 : 1) * (2 + 4 * r());
    return mu;
  }
  function observe(mu, sigma, seed) {
    var nrm = normalGen(rng(seed)), y = new Float64Array(mu.length);
    for (var i = 0; i < mu.length; i++) y[i] = mu[i] + sigma * nrm();
    return y;
  }
  function soft(v, lam) { var a = Math.abs(v) - lam; return a > 0 ? (v > 0 ? a : -a) : 0; }
  function estimate(kind, y, lam) {
    var out = new Float64Array(y.length);
    for (var i = 0; i < y.length; i++) out[i] = kind === "soft" ? soft(y[i], lam) : y[i] / (1 + lam);
    return out;
  }
  /** SURE(lambda): soft threshold d s^2 - 2 s^2 #{|y_i| <= lam} + sum min(y_i^2, lam^2);
   *  ridge y / (1 + lam): d s^2 + lam^2 |y|^2 / (1 + lam)^2 - 2 s^2 d lam / (1 + lam). */
  function sure(kind, y, lam, sigma) {
    var d = y.length, s2 = sigma * sigma, i;
    if (kind === "soft") {
      var cnt = 0, sm = 0;
      for (i = 0; i < d; i++) { var a = Math.abs(y[i]); if (a <= lam) cnt++; sm += Math.min(y[i] * y[i], lam * lam); }
      return d * s2 - 2 * s2 * cnt + sm;
    }
    var yy = 0;
    for (i = 0; i < d; i++) yy += y[i] * y[i];
    return d * s2 + lam * lam * yy / ((1 + lam) * (1 + lam)) - 2 * s2 * d * lam / (1 + lam);
  }
  /** The loss the oracle sees: |mu_hat(y) - mu|^2 for this realisation. */
  function loss(kind, y, mu, lam) {
    var e = estimate(kind, y, lam), s = 0;
    for (var i = 0; i < y.length; i++) s += (e[i] - mu[i]) * (e[i] - mu[i]);
    return s;
  }
  function lamGrid(kind, sigma) {
    var g = [], hi = kind === "soft" ? 4 * sigma : 3, N = 241;
    for (var i = 0; i < N; i++) g.push(hi * i / (N - 1));
    return g;
  }
  /** SURE and loss curves on a lambda grid, and the two minimisers. */
  function curves(kind, mu, y, sigma) {
    var g = lamGrid(kind, sigma), S = [], L = [], bs = 0, bl = 0;
    for (var i = 0; i < g.length; i++) {
      S.push(sure(kind, y, g[i], sigma)); L.push(loss(kind, y, mu, g[i]));
      if (S[i] < S[bs]) bs = i;
      if (L[i] < L[bl]) bl = i;
    }
    return { lam: g, sure: S, loss: L, lamSure: g[bs], lamOracle: g[bl], lossAtSure: L[bs], lossAtOracle: L[bl], lossAtZero: L[0] };
  }

  return { rng: rng, normalGen: normalGen, ptilde: ptilde, score: score, scoreOf: scoreOf, dscore: dscore, Z: Z, sample: sample,
    ksd1: ksd1, constantStats: constantStats, signal: signal, observe: observe, soft: soft, estimate: estimate,
    sure: sure, loss: loss, lamGrid: lamGrid, curves: curves };
})();

if (typeof module !== "undefined") module.exports = SteinModelsModel;
