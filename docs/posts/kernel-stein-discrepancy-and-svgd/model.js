/* SVGD, unadjusted Langevin and the IMQ kernel Stein discrepancy on three 2-D
 * targets known only up to their normalising constants. Numbers only; widgets.js
 * draws them and src/check_model.js checks them under node. */
"use strict";

var SteinSvgdModel = (function () {
  // ---------------------------------------------------------------- targets
  // Each target gives log p~ (unnormalised) and its score. The normalising
  // constant is never computed anywhere in this file.
  var MIX_W = 0.7; // weight of the left component

  function target(key, opts) {
    opts = opts || {};
    if (key === "mixture") {
      var a = (opts.sep === undefined ? 4 : opts.sep) / 2;
      return {
        key: key, view: [-6, 6, -6, 6], leftWeight: MIX_W,
        logp: function (x, y) {
          var l1 = Math.log(MIX_W) - 0.5 * ((x + a) * (x + a) + y * y);
          var l2 = Math.log(1 - MIX_W) - 0.5 * ((x - a) * (x - a) + y * y);
          var m = Math.max(l1, l2);
          return m + Math.log(Math.exp(l1 - m) + Math.exp(l2 - m));
        },
        score: function (x, y, out) {
          var l1 = Math.log(MIX_W) - 0.5 * ((x + a) * (x + a) + y * y);
          var l2 = Math.log(1 - MIX_W) - 0.5 * ((x - a) * (x - a) + y * y);
          var m = Math.max(l1, l2), e1 = Math.exp(l1 - m), e2 = Math.exp(l2 - m), r1 = e1 / (e1 + e2);
          out[0] = r1 * (-a - x) + (1 - r1) * (a - x);
          out[1] = -y;
        }
      };
    }
    if (key === "banana") {
      // x ~ N(0, 1.5^2); y | x ~ N(0.6 x^2 - 1.5, 0.6^2)
      var s1 = 1.5, s2 = 0.6, b = 0.6, c = 1.5;
      return {
        key: key, view: [-6, 6, -4, 8],
        logp: function (x, y) {
          var u = y - (b * x * x - c);
          return -0.5 * x * x / (s1 * s1) - 0.5 * u * u / (s2 * s2);
        },
        score: function (x, y, out) {
          var u = y - (b * x * x - c);
          out[0] = -x / (s1 * s1) + (u / (s2 * s2)) * 2 * b * x;
          out[1] = -u / (s2 * s2);
        }
      };
    }
    if (key === "donut") {
      var R = 3, sd = 0.5;
      return {
        key: key, view: [-6, 6, -6, 6],
        logp: function (x, y) { var r = Math.hypot(x, y); return -0.5 * (r - R) * (r - R) / (sd * sd); },
        score: function (x, y, out) {
          var r = Math.hypot(x, y);
          if (r < 1e-12) { out[0] = 0; out[1] = 0; return; }
          var g = -(r - R) / (sd * sd) / r;
          out[0] = g * x; out[1] = g * y;
        }
      };
    }
    if (key === "gauss") { // standard normal, used only by the checks
      return {
        key: key, view: [-4, 4, -4, 4],
        logp: function (x, y) { return -0.5 * (x * x + y * y); },
        score: function (x, y, out) { out[0] = -x; out[1] = -y; }
      };
    }
    throw new Error("unknown target " + key);
  }

  // ------------------------------------------------------------ random draws
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

  /** n particles drawn from N(centre, spread^2 I), as a flat [x0, y0, x1, y1, ...] array. */
  function init(n, spread, seed, centre) {
    var nrm = normalGen(rng(seed)), P = new Float64Array(2 * n), c = centre || [0, 0];
    for (var i = 0; i < n; i++) { P[2 * i] = c[0] + spread * nrm(); P[2 * i + 1] = c[1] + spread * nrm(); }
    return P;
  }

  // ----------------------------------------------------------------- kernels
  /** Median heuristic: ell^2 = med^2 / (2 log n), so that k = exp(-r^2 / (2 ell^2)) is
   *  Liu & Wang's exp(-r^2 / h) with h = med^2 / log n. */
  function medianEll(P) {
    var n = P.length / 2, d = [];
    for (var i = 0; i < n; i++) for (var j = i + 1; j < n; j++) d.push(Math.hypot(P[2 * i] - P[2 * j], P[2 * i + 1] - P[2 * j + 1]));
    if (!d.length) return 1;
    d.sort(function (a, b) { return a - b; });
    var med = d[Math.floor(d.length / 2)];
    return Math.max(1e-3, med / Math.sqrt(2 * Math.log(Math.max(n, 2))));
  }

  /** One SVGD step with the RBF kernel k(x, x') = exp(-|x - x'|^2 / (2 ell^2)).
   *  phi(x_i) = (1/n) sum_j [ k(x_j, x_i) s(x_j) + (x_i - x_j) / ell^2 k(x_j, x_i) ]. */
  function svgdStep(P, T, eps, ell, scratch) {
    var n = P.length / 2, S = scratch || new Float64Array(2 * n), sc = [0, 0], i, j;
    for (i = 0; i < n; i++) { T.score(P[2 * i], P[2 * i + 1], sc); S[2 * i] = sc[0]; S[2 * i + 1] = sc[1]; }
    var inv = 1 / (ell * ell), phi = new Float64Array(2 * n);
    for (i = 0; i < n; i++) {
      var xi = P[2 * i], yi = P[2 * i + 1], fx = 0, fy = 0;
      for (j = 0; j < n; j++) {
        var dx = xi - P[2 * j], dy = yi - P[2 * j + 1];
        var k = Math.exp(-0.5 * (dx * dx + dy * dy) * inv);
        fx += k * (S[2 * j] + dx * inv);
        fy += k * (S[2 * j + 1] + dy * inv);
      }
      phi[2 * i] = fx / n; phi[2 * i + 1] = fy / n;
    }
    var maxMove = 0;
    for (i = 0; i < 2 * n; i++) { P[i] += eps * phi[i]; maxMove = Math.max(maxMove, Math.abs(eps * phi[i])); }
    return maxMove;
  }

  /** One unadjusted Langevin step for each of n independent chains. */
  function ulaStep(P, T, eps, nrm) {
    var n = P.length / 2, sc = [0, 0], noise = Math.sqrt(2 * eps);
    for (var i = 0; i < n; i++) {
      T.score(P[2 * i], P[2 * i + 1], sc);
      P[2 * i] += eps * sc[0] + noise * nrm();
      P[2 * i + 1] += eps * sc[1] + noise * nrm();
    }
  }

  /** Stein kernel of the IMQ base kernel k = (c^2 + |r|^2)^beta in d = 2:
   *  k_p(x, y) = s(x).s(y) k + s(x).grad_y k + s(y).grad_x k + tr(grad_x grad_y k). */
  function steinKernelIMQ(x, y, sx, sy, c, beta) {
    var rx = x[0] - y[0], ry = x[1] - y[1], r2 = rx * rx + ry * ry, q = c * c + r2;
    var qb = Math.pow(q, beta), qb1 = qb / q, qb2 = qb1 / q, d = 2;
    var dot = sx[0] * sy[0] + sx[1] * sy[1];
    // grad_x k = 2 beta q^(beta-1) r ; grad_y k = -2 beta q^(beta-1) r
    var t2 = -2 * beta * qb1 * (sx[0] * rx + sx[1] * ry);
    var t3 = 2 * beta * qb1 * (sy[0] * rx + sy[1] * ry);
    var t4 = -4 * beta * (beta - 1) * qb2 * r2 - 2 * beta * d * qb1;
    return dot * qb + t2 + t3 + t4;
  }

  /** V-statistic KSD (the KSD of the empirical measure) with the IMQ kernel. */
  function ksd(P, T, c, beta) {
    c = c === undefined ? 1 : c; beta = beta === undefined ? -0.5 : beta;
    var n = P.length / 2, S = new Float64Array(2 * n), sc = [0, 0], i, j, sum = 0;
    for (i = 0; i < n; i++) { T.score(P[2 * i], P[2 * i + 1], sc); S[2 * i] = sc[0]; S[2 * i + 1] = sc[1]; }
    var xi = [0, 0], xj = [0, 0], si = [0, 0], sj = [0, 0];
    for (i = 0; i < n; i++) {
      xi[0] = P[2 * i]; xi[1] = P[2 * i + 1]; si[0] = S[2 * i]; si[1] = S[2 * i + 1];
      sum += steinKernelIMQ(xi, xi, si, si, c, beta);
      for (j = i + 1; j < n; j++) {
        xj[0] = P[2 * j]; xj[1] = P[2 * j + 1]; sj[0] = S[2 * j]; sj[1] = S[2 * j + 1];
        sum += 2 * steinKernelIMQ(xi, xj, si, sj, c, beta);
      }
    }
    return Math.sqrt(Math.max(0, sum) / (n * n));
  }

  /** Fraction of particles with x > 0 (the mixture's lighter, right-hand component). */
  function rightFraction(P) {
    var n = P.length / 2, k = 0;
    for (var i = 0; i < n; i++) if (P[2 * i] > 0) k++;
    return k / n;
  }

  /** Run SVGD and n parallel ULA chains side by side for `steps` iterations, so the
   *  two use the same number of score evaluations (n per iteration). */
  function race(o) {
    var T = target(o.target, o), n = o.n, seed = o.seed || 1;
    var A = init(n, o.spread, seed, o.centre), B = Float64Array.from(A);
    var nrm = normalGen(rng(seed + 7919)), trace = [], every = o.every || 10;
    for (var t = 0; t <= o.steps; t++) {
      if (t % every === 0 || t === o.steps) trace.push({ t: t, evals: t * n, svgd: ksd(A, T), ula: ksd(B, T) });
      if (t === o.steps) break;
      var ell = o.median === false ? o.ell : medianEll(A);
      svgdStep(A, T, o.eps, ell);
      ulaStep(B, T, o.eps, nrm);
    }
    return { svgd: A, ula: B, trace: trace };
  }

  /** Standard normal CDF (Abramowitz & Stegun 7.1.26 via erf; error below 1.5e-7). */
  function normCdf(z) {
    var t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }
  /** Target probability of x > 0 under the mixture with separation sep. */
  function mixtureRightMass(sep) { var a = sep / 2; return MIX_W * normCdf(-a) + (1 - MIX_W) * normCdf(a); }

  return { target: target, rng: rng, normalGen: normalGen, init: init, medianEll: medianEll, normCdf: normCdf, mixtureRightMass: mixtureRightMass,
    svgdStep: svgdStep, ulaStep: ulaStep, steinKernelIMQ: steinKernelIMQ, ksd: ksd,
    rightFraction: rightFraction, race: race, MIX_W: MIX_W };
})();

if (typeof module !== "undefined") module.exports = SteinSvgdModel;
