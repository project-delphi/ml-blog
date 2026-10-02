/* The numbers behind the widgets in "Collinearity, Part 1".
 *
 * Kept apart from the drawing (widgets.js) so the prose can be checked against
 * it: every figure the post quotes from a widget is a call into this file, and
 * `node posts/collinearity/src/check_model.js` sweeps the controls and prints
 * them. Plain JavaScript with no dependencies; exposes one global,
 * `CollinearityModel`, and `module.exports` under node.
 *
 * The synthetic data: n = 40 rows, two centred predictor columns with standard
 * deviation 1 at an angle theta (so their correlation is cos theta), and
 * y = x1 + x2 + noise with noise sd 0.6. The two columns are built from a fixed
 * orthonormal pair (z1, z2) as x1 = z1, x2 = cos(theta) z1 + sin(theta) z2, so
 * changing theta turns x2 towards x1 inside one fixed plane and nothing else
 * about the data moves. 200 noise vectors are drawn once from a fixed seed.
 */
"use strict";

var CollinearityModel = (function () {
  var RAD = Math.PI / 180;
  var N = 40, DRAWS = 200, SIGMA = 0.6, BETA = [1, 1];

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

  // z1, z2: centred, orthogonal, each with norm sqrt(N) (so sd close to 1).
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

  // Noise vectors, and what least squares needs from each: its components
  // along z1 and z2 (e1, e2), and the part orthogonal to both (the residual).
  var NOISE = [];
  (function () {
    var rng = mulberry32(77);
    for (var k = 0; k < DRAWS; k++) {
      var e = centred(rng).map(function (x) { return SIGMA * x; });
      var e1 = dot(e, Z1) / N, e2 = dot(e, Z2) / N;
      var res = e.map(function (x, i) { return x - e1 * Z1[i] - e2 * Z2[i]; });
      NOISE.push({ e: e, e1: e1, e2: e2, res: res, rss: dot(res, res) });
    }
  })();

  /** Slider position 0..100 to an angle in degrees, 90 down to 0.5 on a log scale. */
  function thetaOf(v) { return 90 * Math.pow(0.5 / 90, v / 100); }
  function sliderOf(theta) { return 100 * Math.log(theta / 90) / Math.log(0.5 / 90); }

  /** Correlation, condition number and variance inflation factor of two
   *  equal-length columns at angle theta (degrees). */
  function corr(theta) { return Math.cos(theta * RAD); }
  function kappa(theta) { return 1 / Math.tan(theta * RAD / 2); }
  function vif(theta) { var s = Math.sin(theta * RAD); return 1 / (s * s); }
  /** Singular values of [x1 x2] for unit-length columns. */
  function singular(theta) {
    return [Math.SQRT2 * Math.cos(theta * RAD / 2), Math.SQRT2 * Math.sin(theta * RAD / 2)];
  }

  /** The two columns at angle theta: arrays of length N. */
  function columns(theta) {
    var c = Math.cos(theta * RAD), s = Math.sin(theta * RAD);
    return { x1: Z1.slice(), x2: Z1.map(function (z, i) { return c * z + s * Z2[i]; }) };
  }

  /** Least-squares coefficients for noise draw k at angle theta. */
  function fit(theta, k) {
    var c = Math.cos(theta * RAD), s = Math.sin(theta * RAD), d = NOISE[k];
    var b2 = BETA[1] + d.e2 / s;
    return [BETA[0] + d.e1 - c * (b2 - BETA[1]), b2];
  }

  /** The sample for draw k: its rows, its fit, and its mean squared residual. */
  function sample(theta, k) {
    var X = columns(theta), d = NOISE[k], b = fit(theta, k);
    var y = X.x1.map(function (x, i) { return BETA[0] * x + BETA[1] * X.x2[i] + d.e[i]; });
    return { x1: X.x1, x2: X.x2, y: y, beta: b, mse: d.rss / N };
  }

  /** Spread of the fits over the first `count` noise draws. */
  function spread(theta, count) {
    var b1 = [], b2 = [], sum = [], diff = [], on = [], off = [], k;
    for (k = 0; k < count; k++) {
      var b = fit(theta, k);
      b1.push(b[0]); b2.push(b[1]);
      sum.push(b[0] + b[1]); diff.push(b[0] - b[1]);
      on.push(predict(b, QUERY.on)); off.push(predict(b, QUERY.off));
    }
    return { b1: sd(b1), b2: sd(b2), sum: sd(sum), diff: sd(diff), on: sd(on), off: sd(off),
      onRange: [Math.min.apply(null, on), Math.max.apply(null, on)],
      offRange: [Math.min.apply(null, off), Math.max.apply(null, off)],
      flips: b1.filter(function (x) { return x < 0; }).length + b2.filter(function (x) { return x < 0; }).length };
  }

  // Two places to ask for a prediction: one where both predictors are high
  // together, as they are in the data, and one where they disagree.
  var QUERY = { on: [1.5, 1.5], off: [1.5, -1.5] };
  function predict(b, q) { return b[0] * q[0] + b[1] * q[1]; }

  // ----------------------------------------------------- three observations
  // The column-space picture: x1 and x2 are unit vectors in the floor of R^3,
  // y0 = 1.4 x1 + 0.7 e3. A nudge is 2% of |y0| along one of three directions.
  var Y0 = [1.4, 0, 0.7], NUDGE = 0.02;
  function nudgeDirections(theta) {
    var h = theta * RAD / 2;
    return { none: [0, 0, 0], strong: [Math.cos(h), Math.sin(h), 0],
      weak: [-Math.sin(h), Math.cos(h), 0], out: [0, 0, 1] };
  }
  function project(theta, y) {
    var c = Math.cos(theta * RAD), s = Math.sin(theta * RAD);
    var b2 = y[1] / s, b1 = y[0] - c * b2;
    return { beta: [b1, b2], fitted: [y[0], y[1], 0], rss: y[2] * y[2] };
  }
  /** What a 2% nudge of y in the named direction does at angle theta. */
  function nudge(theta, which) {
    var len = Math.sqrt(dot(Y0, Y0)), u = nudgeDirections(theta)[which];
    var size = which === "none" ? 0 : NUDGE * len;
    var y = Y0.map(function (v, i) { return v + size * u[i]; });
    var base = project(theta, Y0), now = project(theta, y);
    var db = [now.beta[0] - base.beta[0], now.beta[1] - base.beta[1]];
    var df = [now.fitted[0] - base.fitted[0], now.fitted[1] - base.fitted[1]];
    return {
      y: y, beta: now.beta, fitted: now.fitted, rss: now.rss, base: base,
      x1: [1, 0, 0], x2: [Math.cos(theta * RAD), Math.sin(theta * RAD), 0],
      betaChange: Math.sqrt(dot(db, db)) / Math.sqrt(dot(base.beta, base.beta)),
      fittedChange: Math.sqrt(dot(df, df)) / Math.sqrt(dot(base.fitted, base.fitted)),
      rssChange: now.rss / base.rss - 1
    };
  }

  // ------------------------------------------------------------ three columns
  // x1 and x2 at right angles in the floor; x3 lifted by alpha out of the
  // floor above their diagonal. Gram matrix [[1,0,k],[0,1,k],[k,k,1]] with
  // k = cos(alpha)/sqrt(2), whose eigenvalues are 1 + cos(alpha), 1, 1 - cos(alpha).
  function three(alpha) {
    var c = Math.cos(alpha * RAD), s = Math.sin(alpha * RAD), k = c / Math.SQRT2;
    return {
      x1: [1, 0, 0], x2: [0, 1, 0], x3: [k, k, s],
      pairwise: k,
      kappa: Math.sqrt((1 + c) / (1 - c)),
      volume: s,
      vif: [(1 - k * k) / (s * s), (1 - k * k) / (s * s), 1 / (s * s)]
    };
  }

  var api = {
    N: N, DRAWS: DRAWS, SIGMA: SIGMA, BETA: BETA, QUERY: QUERY, Y0: Y0, NUDGE: NUDGE, NOISE: NOISE,
    thetaOf: thetaOf, sliderOf: sliderOf, corr: corr, kappa: kappa, vif: vif, singular: singular,
    columns: columns, fit: fit, sample: sample, spread: spread, predict: predict,
    project: project, nudge: nudge, three: three
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
