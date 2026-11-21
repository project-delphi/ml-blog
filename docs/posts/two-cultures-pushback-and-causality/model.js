/* Confounding model for the Part 5 widget. Pure numbers; widgets.js draws them.
 *
 * Structural model (all synthetic, standardised):
 *   S ~ N(0, 1)                          severity
 *   T = a S + sqrt(1 - a^2) U_T          treatment dose, Var(T) = 1, corr(S, T) = a
 *   Y = tau T + b S + sigma U_Y          outcome
 * Intervening on T (do(T = t)) changes E[Y] by tau per unit, whatever a and b are.
 * The observational slope of Y on T alone is tau + a b.
 */
"use strict";

var ConfoundModel = (function () {
  var N = 400, SIGMA = 0.6;

  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Standard normals for one seed, fixed across slider settings so that only
  // the structural coefficients move the points.
  var cache = {};
  function noise(seed) {
    if (cache[seed]) return cache[seed];
    var u = rng(seed), out = { s: [], ut: [], uy: [] };
    function z() {
      var a = Math.max(u(), 1e-12), b = u();
      return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
    }
    for (var i = 0; i < N; i++) { out.s.push(z()); out.ut.push(z()); out.uy.push(z()); }
    cache[seed] = out;
    return out;
  }

  function sample(p, seed) {
    var e = noise(seed === undefined ? 1 : seed), rows = [];
    var c = Math.sqrt(Math.max(0, 1 - p.a * p.a));
    for (var i = 0; i < N; i++) {
      var s = e.s[i], t = p.a * s + c * e.ut[i];
      rows.push({ s: s, t: t, y: p.tau * t + p.b * s + SIGMA * e.uy[i] });
    }
    return rows;
  }

  // Least squares of y on an intercept and the listed columns, via the normal
  // equations (at most three unknowns), plus R^2.
  function ols(rows, cols) {
    var k = cols.length + 1, n = rows.length, i, j, r;
    var A = [], v = [];
    for (i = 0; i < k; i++) { A.push(new Array(k).fill(0)); v.push(0); }
    for (r = 0; r < n; r++) {
      var x = [1].concat(cols.map(function (c) { return rows[r][c]; }));
      for (i = 0; i < k; i++) {
        v[i] += x[i] * rows[r].y;
        for (j = 0; j < k; j++) A[i][j] += x[i] * x[j];
      }
    }
    for (i = 0; i < k; i++) {               // Gauss-Jordan elimination
      var piv = A[i][i];
      for (j = 0; j < k; j++) A[i][j] /= piv;
      v[i] /= piv;
      for (r = 0; r < k; r++) {
        if (r === i) continue;
        var f = A[r][i];
        for (j = 0; j < k; j++) A[r][j] -= f * A[i][j];
        v[r] -= f * v[i];
      }
    }
    var ybar = rows.reduce(function (s, q) { return s + q.y; }, 0) / n, sse = 0, sst = 0;
    rows.forEach(function (q) {
      var fit = v[0];
      cols.forEach(function (c, m) { fit += v[m + 1] * q[c]; });
      sse += (q.y - fit) * (q.y - fit);
      sst += (q.y - ybar) * (q.y - ybar);
    });
    return { coef: v, r2: 1 - sse / sst };
  }

  function stats(p, seed) {
    var rows = sample(p, seed);
    var naive = ols(rows, ["t"]), adj = ols(rows, ["t", "s"]);
    return {
      rows: rows,
      naiveSlope: naive.coef[1], naiveIntercept: naive.coef[0], naiveR2: naive.r2,
      adjSlope: adj.coef[1], adjR2: adj.r2,
      theorySlope: p.tau + p.a * p.b, tau: p.tau
    };
  }

  var api = { N: N, SIGMA: SIGMA, sample: sample, ols: ols, stats: stats };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
