/* The numbers behind the widgets in "Collinearity, Part 2".
 *
 * Kept apart from the drawing (widgets.js) so the prose can be checked against
 * it: `node posts/collinearity-remedies/src/check_model.js` sweeps the controls
 * and prints every figure the post quotes from a widget. Plain JavaScript with
 * no dependencies; exposes one global, `RemedyModel`, and `module.exports`
 * under node.
 *
 * The synthetic data are Part 1's with one change. n = 40 rows, two centred
 * predictor columns with standard deviation 1 at an angle theta (correlation
 * cos theta), the same x values and the same 200 noise vectors (sd 0.6) from
 * the same seeds. The true coefficients are (1.5, 0.5) instead of (1, 1), so
 * the direction the data cannot see, beta2 - beta1, has a true value that is
 * not zero and a remedy that shrinks it pays for doing so.
 *
 * Every estimator minimises  MSE(beta) + lambda * penalty(beta), where
 * MSE(beta) = min + (beta - ols)' G (beta - ols) and G = [[1, c], [c, 1]]:
 *
 *   ols      no penalty
 *   ridge    beta1^2 + beta2^2
 *   lasso    |beta1| + |beta2|
 *   enet     half of each
 *   pcr      no penalty; beta restricted to the line beta1 = beta2
 */
"use strict";

var RemedyModel = (function () {
  var RAD = Math.PI / 180;
  var N = 40, DRAWS = 200, SIGMA = 0.6, BETA = [1.5, 0.5];
  var METHODS = {
    ols: { label: "least squares", l1: 0, l2: 0, lambda: false },
    ridge: { label: "ridge", l1: 0, l2: 1, lambda: true },
    lasso: { label: "lasso", l1: 1, l2: 0, lambda: true },
    enet: { label: "elastic net", l1: 0.5, l2: 0.5, lambda: true },
    pcr: { label: "one component", l1: 0, l2: 0, lambda: false }
  };

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    var u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function mean(a) { return a.reduce(function (s, x) { return s + x; }, 0) / a.length; }
  function sd(a) {
    var m = mean(a);
    return Math.sqrt(a.reduce(function (s, x) { return s + (x - m) * (x - m); }, 0) / (a.length - 1));
  }
  function centred(rng) {
    var v = [], i;
    for (i = 0; i < N; i++) v.push(gaussian(rng));
    var m = mean(v);
    return v.map(function (x) { return x - m; });
  }

  // The same z1, z2 and noise vectors as Part 1 (same seeds, same order).
  var Z1, Z2;
  (function () {
    var rng = mulberry32(20261001), a = centred(rng), b = centred(rng);
    var na = Math.sqrt(dot(a, a) / N);
    Z1 = a.map(function (x) { return x / na; });
    var p = dot(b, Z1) / N;
    b = b.map(function (x, i) { return x - p * Z1[i]; });
    var nb = Math.sqrt(dot(b, b) / N);
    Z2 = b.map(function (x) { return x / nb; });
  })();
  var NOISE = [];
  (function () {
    var rng = mulberry32(77);
    for (var k = 0; k < DRAWS; k++) {
      var e = centred(rng).map(function (x) { return SIGMA * x; });
      NOISE.push({ e: e, e1: dot(e, Z1) / N, e2: dot(e, Z2) / N });
    }
  })();

  function thetaOf(v) { return 90 * Math.pow(0.5 / 90, v / 100); }
  function sliderOf(theta) { return 100 * Math.log(theta / 90) / Math.log(0.5 / 90); }
  /** Slider stop 0..28 to a penalty weight: eight stops per decade from 0.001
   *  to 3.16, so 0.01, 0.1 and 1 are stops the reader can land on. */
  var LAMBDA_STOPS = 28;
  function lambdaOf(i) { return Math.pow(10, -3 + i / 8); }
  function lambdaStop(lambda) { return Math.round(8 * (Math.log10(lambda) + 3)); }
  function corr(theta) { return Math.cos(theta * RAD); }
  function kappa(theta) { return 1 / Math.tan(theta * RAD / 2); }

  function columns(theta) {
    var c = Math.cos(theta * RAD), s = Math.sin(theta * RAD);
    return { x1: Z1.slice(), x2: Z1.map(function (z, i) { return c * z + s * Z2[i]; }) };
  }

  /** Least-squares coefficients for noise draw k at angle theta. */
  function ols(theta, k) {
    var c = Math.cos(theta * RAD), s = Math.sin(theta * RAD), d = NOISE[k];
    var b2 = BETA[1] + d.e2 / s;
    return [BETA[0] + d.e1 - c * (b2 - BETA[1]), b2];
  }

  function soft(x, a) { return x > a ? x - a : x < -a ? x + a : 0; }

  /** The minimiser of (b - bhat)' G (b - bhat) + lambda (l1 |b|_1 + l2 |b|^2)
   *  for G = [[1, c], [c, 1]], found exactly by checking each sign pattern
   *  against its optimality conditions. */
  function penalised(bhat, c, lambda, l1, l2) {
    var g1 = bhat[0] + c * bhat[1], g2 = c * bhat[0] + bhat[1];
    var q = 1 + lambda * l2, a = lambda * l1 / 2, det = q * q - c * c, tol = 1e-12;
    if (a === 0) return [(q * g1 - c * g2) / det, (q * g2 - c * g1) / det];
    var signs = [[1, 1], [1, -1], [-1, 1], [-1, -1]], i;
    for (i = 0; i < 4; i++) {
      var r1 = g1 - a * signs[i][0], r2 = g2 - a * signs[i][1];
      var b1 = (q * r1 - c * r2) / det, b2 = (q * r2 - c * r1) / det;
      if (b1 * signs[i][0] > tol && b2 * signs[i][1] > tol) return [b1, b2];
    }
    var only1 = soft(g1, a) / q;
    if (only1 !== 0 && Math.abs(g2 - c * only1) <= a + tol) return [only1, 0];
    var only2 = soft(g2, a) / q;
    if (only2 !== 0 && Math.abs(g1 - c * only2) <= a + tol) return [0, only2];
    return [0, 0];
  }

  /** The estimate a method returns for a sample whose least-squares
   *  coefficients are `b`. */
  function estimateFrom(b, theta, method, lambda) {
    var m = METHODS[method];
    if (method === "ols") return b;
    if (method === "pcr") { var h = (b[0] + b[1]) / 2; return [h, h]; }
    return penalised(b, Math.cos(theta * RAD), lambda, m.l1, m.l2);
  }
  /** The estimate a method returns for noise draw k. */
  function estimate(theta, k, method, lambda) { return estimateFrom(ols(theta, k), theta, method, lambda); }

  /** The objective a method minimises on a sample whose least-squares
   *  coefficients are `b`, measured from the least-squares minimum: excess
   *  MSE plus the penalty. */
  function objectiveFrom(b, theta, method, lambda) {
    var c = Math.cos(theta * RAD), m = METHODS[method];
    return function (p) {
      var d1 = p[0] - b[0], d2 = p[1] - b[1];
      var v = d1 * d1 + 2 * c * d1 * d2 + d2 * d2;
      if (m.lambda) v += lambda * (m.l1 * (Math.abs(p[0]) + Math.abs(p[1])) + m.l2 * (p[0] * p[0] + p[1] * p[1]));
      return v;
    };
  }
  function objective(theta, k, method, lambda) { return objectiveFrom(ols(theta, k), theta, method, lambda); }

  var QUERY = { on: [1.5, 1.5], off: [1.5, -1.5] };
  function predict(b, q) { return b[0] * q[0] + b[1] * q[1]; }

  /** What a method does over all 200 noise draws. */
  function summary(theta, method, lambda) {
    var b1 = [], b2 = [], dist2 = 0, zero1 = 0, zero2 = 0, k;
    var q = { on: [], off: [] };
    for (k = 0; k < DRAWS; k++) {
      var b = estimate(theta, k, method, lambda);
      b1.push(b[0]); b2.push(b[1]);
      dist2 += Math.pow(b[0] - BETA[0], 2) + Math.pow(b[1] - BETA[1], 2);
      if (b[0] === 0) zero1++;
      if (b[1] === 0) zero2++;
      q.on.push(predict(b, QUERY.on)); q.off.push(predict(b, QUERY.off));
    }
    function at(name) {
      var truth = predict(BETA, QUERY[name]), m = mean(q[name]), s = sd(q[name]);
      var mse = q[name].reduce(function (t, v) { return t + (v - truth) * (v - truth); }, 0) / DRAWS;
      return { truth: truth, mean: m, sd: s, rmse: Math.sqrt(mse) };
    }
    return { mean: [mean(b1), mean(b2)], sd: [sd(b1), sd(b2)], rmse: Math.sqrt(dist2 / DRAWS),
      zero1: zero1, zero2: zero2, on: at("on"), off: at("off") };
  }

  var api = {
    N: N, DRAWS: DRAWS, SIGMA: SIGMA, BETA: BETA, QUERY: QUERY, NOISE: NOISE, METHODS: METHODS,
    thetaOf: thetaOf, sliderOf: sliderOf, LAMBDA_STOPS: LAMBDA_STOPS, lambdaOf: lambdaOf, lambdaStop: lambdaStop,
    corr: corr, kappa: kappa, columns: columns, ols: ols, penalised: penalised,
    estimate: estimate, estimateFrom: estimateFrom, objective: objective, objectiveFrom: objectiveFrom,
    predict: predict, summary: summary
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
