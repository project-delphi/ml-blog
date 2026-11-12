/* Two models for Stein's Method, Part 5. Numbers only; widgets.js draws them and
 * src/check_model.js checks them under node.
 *  1. Unadjusted Langevin (ULA) and Metropolis-adjusted Langevin (MALA) on three 1-D
 *     targets, with the transition kernel discretised on a grid. One LU factorisation per
 *     sampler and step size gives the chain's exact stationary law, its bias and its
 *     asymptotic variance, with no Monte Carlo noise. Stein's first-order prediction of
 *     ULA's bias is computed beside it.
 *  2. Zero-variance control variates on MALA output for a 2-D Bayesian
 *     logistic-regression posterior with synthetic data. */
"use strict";

var SteinMCMCModel = (function () {
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

  // ===================================================================== 1. 1-D targets
  // Mixture 0.6 N(-1.2, 0.6^2) + 0.4 N(1.3, 0.9^2).
  var MW1 = 0.6, MM1 = -1.2, MS1 = 0.6, MW2 = 0.4, MM2 = 1.3, MS2 = 0.9;
  function mixParts(x) {
    var a = MW1 / MS1 * Math.exp(-0.5 * (x - MM1) * (x - MM1) / (MS1 * MS1));
    var b = MW2 / MS2 * Math.exp(-0.5 * (x - MM2) * (x - MM2) / (MS2 * MS2));
    return [a, b];
  }
  var TARGETS = {
    gauss: {
      label: "Gaussian", lo: -7, hi: 7,
      logp: function (x) { return -0.5 * x * x; },
      s: function (x) { return -x; },
      ds: function () { return -1; }
    },
    heavy: {
      label: "Heavier tails", lo: -18, hi: 18,
      // log p = -sqrt(1 + x^2): Gaussian in the middle, exponential tails, Lipschitz score.
      logp: function (x) { return -Math.sqrt(1 + x * x); },
      s: function (x) { return -x / Math.sqrt(1 + x * x); },
      ds: function (x) { return -Math.pow(1 + x * x, -1.5); }
    },
    mixture: {
      label: "Two modes", lo: -7, hi: 7,
      logp: function (x) { var q = mixParts(x); return Math.log(q[0] + q[1]); },
      s: function (x) {
        var q = mixParts(x);
        return (q[0] * (-(x - MM1) / (MS1 * MS1)) + q[1] * (-(x - MM2) / (MS2 * MS2))) / (q[0] + q[1]);
      },
      ds: function (x) {
        var q = mixParts(x), sa = -(x - MM1) / (MS1 * MS1), sb = -(x - MM2) / (MS2 * MS2);
        var pp = q[0] * (sa * sa - 1 / (MS1 * MS1)) + q[1] * (sb * sb - 1 / (MS2 * MS2));
        var s = (q[0] * sa + q[1] * sb) / (q[0] + q[1]);
        return pp / (q[0] + q[1]) - s * s;
      }
    }
  };
  /** Test functions h whose expectation the chain estimates. */
  var HS = [
    { key: "mean", label: "Mean E[X]", f: function (x) { return x; } },
    { key: "second", label: "Second moment E[X²]", f: function (x) { return x * x; } },
    { key: "tail", label: "Tail P(X > 2)", f: function (x) { return x > 2 ? 1 : 0; },
      // on the grid, an indicator is averaged over each point's cell, so its expectation is accurate to O(dx^2)
      cell: function (x, dx) { return Math.min(1, Math.max(0, (x + dx / 2 - 2) / dx)); } }
  ];
  var NGRID = 241;
  /** The step-size range of the chart's slider and curves. */
  var EPS_MIN = 0.01, EPS_MAX = 0.4;
  /** Step sizes for the error-against-step-size curves: 17 log-spaced points spanning [EPS_MIN, EPS_MAX]. */
  var EPS_GRID = (function () { var o = []; for (var i = 0; i < 17; i++) o.push(EPS_MIN * Math.pow(EPS_MAX / EPS_MIN, i / 16)); return o; })();

  var gridCache = {}, curveCache = {};
  /** Change the grid size (for checking convergence under node); clears every cache. */
  function setGridSize(n) { NGRID = n; gridCache = {}; curveCache = {}; analyseCache = {}; analyseKeys = []; }
  /** The current grid size (a function, so it follows setGridSize). */
  function gridSize() { return NGRID; }
  /** The grid, the target's normalised weights on it, and Stein's first-order coefficients. */
  function setup(key) {
    if (gridCache[key]) return gridCache[key];
    var T = TARGETS[key], N = NGRID, x = new Float64Array(N), lp = new Float64Array(N), i, k;
    var dx = (T.hi - T.lo) / (N - 1), mx = -Infinity;
    for (i = 0; i < N; i++) { x[i] = T.lo + i * dx; lp[i] = T.logp(x[i]); if (lp[i] > mx) mx = lp[i]; }
    var w = new Float64Array(N), tot = 0;
    for (i = 0; i < N; i++) { w[i] = Math.exp(lp[i] - mx); tot += w[i]; }
    for (i = 0; i < N; i++) w[i] /= tot;
    var s = new Float64Array(N), psi = new Float64Array(N);
    for (i = 0; i < N; i++) { s[i] = T.s(x[i]); psi[i] = T.ds(x[i]) + 0.5 * s[i] * s[i]; }
    var hv = HS.map(function (h) { var v = new Float64Array(N); for (var j = 0; j < N; j++) v[j] = h.cell ? h.cell(x[j], dx) : h.f(x[j]); return v; });
    var Epsi = 0; for (i = 0; i < N; i++) Epsi += w[i] * psi[i];
    var Eh = [], coef = [], varh = [], avarCont = [];
    for (k = 0; k < HS.length; k++) {
      var e = 0, ehp = 0, e2 = 0;
      for (i = 0; i < N; i++) { e += w[i] * hv[k][i]; ehp += w[i] * hv[k][i] * psi[i]; e2 += w[i] * hv[k][i] * hv[k][i]; }
      Eh.push(e); varh.push(e2 - e * e);
      // Stein's first-order prediction: bias of E h under ULA ~ eps * coef, coef = Cov_p(h, s' + s^2/2) / 2
      coef.push(0.5 * (ehp - e * Epsi));
      // the Stein solution f_h = (1/p) int_{-inf}^x (h - E h) p; the diffusion's asymptotic variance is 2 E_p f_h^2
      var cum = 0, a = 0;
      for (i = 0; i < N; i++) {
        var inc = (hv[k][i] - e) * w[i];
        // midpoint cumulative sum of hbar p dx, divided by the density p(x_i) = w_i / dx
        var f = (cum + 0.5 * inc) * dx / w[i];
        cum += inc;
        a += w[i] * f * f;
      }
      avarCont.push(2 * a);
    }
    var out = { key: key, x: x, dx: dx, w: w, s: s, psi: psi, Epsi: Epsi, hv: hv, Eh: Eh, coef: coef, varh: varh, avarCont: avarCont, N: N };
    gridCache[key] = out;
    return out;
  }

  /** Row-stochastic transition matrix of ULA or MALA on the grid, row-major. */
  function kernel(key, sampler, eps) {
    var G = setup(key), T = TARGETS[key], N = G.N, x = G.x, P = new Float64Array(N * N), i, j;
    var v2 = 4 * eps; // 2 * variance of the proposal N(x + eps s(x), 2 eps)
    var lp = new Float64Array(N);
    for (i = 0; i < N; i++) lp[i] = T.logp(x[i]);
    var accRow = new Float64Array(N);
    for (i = 0; i < N; i++) {
      var m = x[i] + eps * G.s[i], row = 0, off = i * N;
      for (j = 0; j < N; j++) {
        var d = x[j] - m, q = Math.exp(-d * d / v2) * G.dx / Math.sqrt(Math.PI * v2);
        if (sampler === "mala" && j !== i) {
          var mj = x[j] + eps * G.s[j], db = x[i] - mj;
          var la = lp[j] - lp[i] - db * db / v2 + d * d / v2;
          if (la < 0) q *= Math.exp(la);
        }
        P[off + j] = q; row += q;
      }
      // accepted proposals, including one that lands in its own cell
      accRow[i] = row;
      if (sampler === "mala") {
        // proposals off the grid or rejected stay put: the diagonal takes the remainder
        var rest = 0; for (j = 0; j < N; j++) if (j !== i) rest += P[off + j];
        P[off + i] = 1 - rest;
      } else {
        // ULA has no rejection; mass proposed beyond the grid is renormalised over it
        for (j = 0; j < N; j++) P[off + j] /= row;
        accRow[i] = 1;
      }
    }
    P.accRow = accRow;
    return P;
  }

  // dense LU with partial pivoting, in place
  function lu(A, N) {
    var piv = new Int32Array(N), i, j, k;
    for (i = 0; i < N; i++) piv[i] = i;
    for (k = 0; k < N; k++) {
      var p = k, best = Math.abs(A[k * N + k]);
      for (i = k + 1; i < N; i++) { var v = Math.abs(A[i * N + k]); if (v > best) { best = v; p = i; } }
      if (p !== k) {
        for (j = 0; j < N; j++) { var t = A[k * N + j]; A[k * N + j] = A[p * N + j]; A[p * N + j] = t; }
        var tp = piv[k]; piv[k] = piv[p]; piv[p] = tp;
      }
      var d = A[k * N + k];
      for (i = k + 1; i < N; i++) {
        var f = A[i * N + k] / d;
        if (f === 0) continue;
        A[i * N + k] = f;
        var ri = i * N, rk = k * N;
        for (j = k + 1; j < N; j++) A[ri + j] -= f * A[rk + j];
      }
    }
    return piv;
  }
  function luSolve(A, piv, N, b) {
    var y = new Float64Array(N), i, j;
    for (i = 0; i < N; i++) { var s = b[piv[i]]; for (j = 0; j < i; j++) s -= A[i * N + j] * y[j]; y[i] = s; }
    for (i = N - 1; i >= 0; i--) { var t = y[i]; for (j = i + 1; j < N; j++) t -= A[i * N + j] * y[j]; y[i] = t / A[i * N + i]; }
    return y;
  }
  function luSolveT(A, piv, N, b) {
    // A[piv] = L U, so A^T = U^T L^T Pi: solve U^T z = b, L^T w = z, then x[piv[i]] = w[i]
    var z = new Float64Array(N), i, j;
    for (i = 0; i < N; i++) { var s = b[i]; for (j = 0; j < i; j++) s -= A[j * N + i] * z[j]; z[i] = s / A[i * N + i]; }
    for (i = N - 1; i >= 0; i--) { var t = z[i]; for (j = i + 1; j < N; j++) t -= A[j * N + i] * z[j]; z[i] = t; }
    var x = new Float64Array(N);
    for (i = 0; i < N; i++) x[piv[i]] = z[i];
    return x;
  }

  /** Stationary law, bias and asymptotic variance of one sampler at one step size.
   *  With M = I - P + 1 r^T (r uniform), pi M = r^T gives the stationary law, and
   *  M u = hbar gives a solution of the discrete Poisson equation (I - P) u = hbar. */
  var analyseCache = {}, analyseKeys = [];
  /** analyse(), memoised on (target, sampler, step) for the most recent 64 calls, so a redraw that
   *  leaves the step unchanged (a budget-slider move) does not refactorise. */
  function analyse(key, sampler, eps) {
    var ck = key + "|" + sampler + "|" + eps;
    if (analyseCache[ck]) return analyseCache[ck];
    var out = analyseUncached(key, sampler, eps);
    analyseCache[ck] = out; analyseKeys.push(ck);
    if (analyseKeys.length > 64) delete analyseCache[analyseKeys.shift()];
    return out;
  }
  function analyseUncached(key, sampler, eps) {
    var G = setup(key), N = G.N, P = kernel(key, sampler, eps), i, j, k;
    var M = new Float64Array(N * N);
    for (i = 0; i < N; i++) for (j = 0; j < N; j++) M[i * N + j] = (i === j ? 1 : 0) - P[i * N + j] + 1 / N;
    var acc = 0;
    var piv = lu(M, N);
    var r = new Float64Array(N); for (i = 0; i < N; i++) r[i] = 1 / N;
    var pi = luSolveT(M, piv, N, r), tot = 0;
    for (i = 0; i < N; i++) { if (pi[i] < 0) pi[i] = 0; tot += pi[i]; }
    for (i = 0; i < N; i++) pi[i] /= tot;
    if (sampler === "mala") for (i = 0; i < N; i++) acc += pi[i] * Math.min(1, P.accRow[i]);
    var bias = [], avar = [], mean = [];
    for (k = 0; k < HS.length; k++) {
      var hv = G.hv[k], e = 0;
      for (i = 0; i < N; i++) e += pi[i] * hv[i];
      var hbar = new Float64Array(N); for (i = 0; i < N; i++) hbar[i] = hv[i] - e;
      var u = luSolve(M, piv, N, hbar), a = 0, b = 0;
      for (i = 0; i < N; i++) { a += pi[i] * hbar[i] * u[i]; b += pi[i] * hbar[i] * hbar[i]; }
      mean.push(e); bias.push(e - G.Eh[k]); avar.push(2 * a - b);
    }
    return { pi: pi, mean: mean, bias: bias, avar: avar, acc: sampler === "mala" ? acc : 1 };
  }

  /** analyse() for both samplers at every step size in EPS_GRID. */
  function curves(key) {
    if (curveCache[key]) return curveCache[key];
    var out = { eps: EPS_GRID, ula: [], mala: [] };
    EPS_GRID.forEach(function (e) { out.ula.push(analyse(key, "ula", e)); out.mala.push(analyse(key, "mala", e)); });
    curveCache[key] = out;
    return out;
  }
  /** Root mean squared error of an N-step average: sqrt(bias^2 + asymptotic variance / N). */
  function rmse(bias, avar, n) { return Math.sqrt(bias * bias + Math.max(avar, 0) / n); }
  /** Best step size at budget n, by golden-section search on log eps over [EPS_MIN, EPS_MAX]. */
  function bestEps(key, sampler, hIdx, n) {
    var a = Math.log(EPS_MIN), b = Math.log(EPS_MAX), g = (Math.sqrt(5) - 1) / 2;
    function f(le) { var r = analyse(key, sampler, Math.exp(le)); return rmse(r.bias[hIdx], r.avar[hIdx], n); }
    var c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
    for (var it = 0; it < 22; it++) {
      if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
      else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
    }
    var le = (a + b) / 2;
    return { eps: Math.exp(le), rmse: f(le) };
  }
  /** RMSE against step size at budget n, from the cached curves: bias/eps and
   *  log(asymptotic variance) are interpolated linearly in log eps, both smooth there. */
  function rmseCurve(key, sampler, hIdx, n, points) {
    var C = curves(key), E = C.eps, R = C[sampler], K = E.length, out = { eps: [], rmse: [], bias: [] };
    points = points || 160;
    for (var q = 0; q < points; q++) {
      var le = Math.log(EPS_MIN) + (Math.log(EPS_MAX) - Math.log(EPS_MIN)) * q / (points - 1);
      var t = (le - Math.log(E[0])) / (Math.log(E[K - 1]) - Math.log(E[0])) * (K - 1);
      var k = Math.max(0, Math.min(K - 2, Math.floor(t))), f = t - k, e = Math.exp(le);
      var b = e * ((1 - f) * R[k].bias[hIdx] / E[k] + f * R[k + 1].bias[hIdx] / E[k + 1]);
      var lv = (1 - f) * Math.log(Math.max(R[k].avar[hIdx], 1e-300)) + f * Math.log(Math.max(R[k + 1].avar[hIdx], 1e-300));
      out.eps.push(e); out.bias.push(b); out.rmse.push(rmse(b, Math.exp(lv), n));
    }
    var j = 0; for (var i = 1; i < points; i++) if (out.rmse[i] < out.rmse[j]) j = i;
    out.best = { eps: out.eps[j], rmse: out.rmse[j] };
    return out;
  }
  /** Stein's first-order density of ULA's stationary law: p (1 + (eps/2)(psi - E psi)). */
  function predictedDensity(key, eps) {
    var G = setup(key), out = new Float64Array(G.N);
    for (var i = 0; i < G.N; i++) out[i] = G.w[i] * (1 + 0.5 * eps * (G.psi[i] - G.Epsi));
    return out;
  }

  // ============================================== 2. logistic-regression posterior + ZV
  var LR = (function () {
    var r = rng(20261111), nrm = normalGen(r), n = 30, X = [], y = [], th = [1, -0.5];
    for (var i = 0; i < n; i++) {
      var a = nrm(), b = nrm();
      X.push([a, b]);
      var z = a * th[0] + b * th[1];
      y.push(r() < 1 / (1 + Math.exp(-z)) ? 1 : 0);
    }
    return { n: n, X: X, y: y, tau: 3, thetaTrue: th };
  })();
  function lrLogp(t0, t1) {
    var s = -0.5 * (t0 * t0 + t1 * t1) / (LR.tau * LR.tau);
    for (var i = 0; i < LR.n; i++) {
      var z = LR.X[i][0] * t0 + LR.X[i][1] * t1;
      s += LR.y[i] * z - (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
    }
    return s;
  }
  function lrScore(t0, t1, out) {
    var g0 = -t0 / (LR.tau * LR.tau), g1 = -t1 / (LR.tau * LR.tau);
    for (var i = 0; i < LR.n; i++) {
      var z = LR.X[i][0] * t0 + LR.X[i][1] * t1, e = LR.y[i] - 1 / (1 + Math.exp(-z));
      g0 += e * LR.X[i][0]; g1 += e * LR.X[i][1];
    }
    out[0] = g0; out[1] = g1;
    return out;
  }
  var LR_H = [
    { key: "mean", label: "Posterior mean of θ₁", f: function (t0) { return t0; } },
    { key: "second", label: "Posterior E[θ₁²]", f: function (t0) { return t0 * t0; } },
    { key: "tail", label: "Posterior P(θ₁ > 1)", f: function (t0) { return t0 > 1 ? 1 : 0; } }
  ];
  var LR_VIEW = [-1.1, 2.6, -2.5, 1.2];
  var refCache = null;
  /** Reference posterior expectations by quadrature on a fine grid (the evidence is a sum here, for checking only). */
  function lrReference() {
    if (refCache) return refCache;
    var G = 401, lo0 = -4, hi0 = 7, lo1 = -6, hi1 = 4, lps = new Float64Array(G * G), mx = -Infinity, i, j;
    for (i = 0; i < G; i++) for (j = 0; j < G; j++) {
      var t0 = lo0 + (hi0 - lo0) * i / (G - 1), t1 = lo1 + (hi1 - lo1) * j / (G - 1), l = lrLogp(t0, t1);
      lps[i * G + j] = l; if (l > mx) mx = l;
    }
    var tot = 0, acc = [0, 0, 0];
    for (i = 0; i < G; i++) for (j = 0; j < G; j++) {
      var w = Math.exp(lps[i * G + j] - mx), a0 = lo0 + (hi0 - lo0) * i / (G - 1);
      tot += w;
      // the indicator is averaged over each point's cell, so the threshold is not moved to a grid line
      var d0 = (hi0 - lo0) / (G - 1), cell = Math.min(1, Math.max(0, (a0 + d0 / 2 - 1) / d0));
      for (var k = 0; k < 3; k++) acc[k] += w * (LR_H[k].key === "tail" ? cell : LR_H[k].f(a0));
    }
    refCache = acc.map(function (v) { return v / tot; });
    return refCache;
  }
  /** Relative posterior density on a G x G display grid over LR_VIEW, row-major from the top. */
  function lrDensityGrid(G) {
    var v = LR_VIEW, out = new Float64Array(G * G), mx = -Infinity, i, j;
    for (j = 0; j < G; j++) for (i = 0; i < G; i++) {
      var t0 = v[0] + (i + 0.5) / G * (v[1] - v[0]), t1 = v[3] - (j + 0.5) / G * (v[3] - v[2]);
      var l = lrLogp(t0, t1); out[j * G + i] = l; if (l > mx) mx = l;
    }
    for (i = 0; i < G * G; i++) out[i] = Math.exp(out[i] - mx);
    return out;
  }
  /** A MALA chain of n kept states after `burn` discarded ones, started at the origin. */
  function mala(n, eps, seed, burn) {
    burn = burn === undefined ? 200 : burn;
    var r = rng(seed), nrm = normalGen(r), t = [0, 0], st = lrScore(0, 0, [0, 0]), lt = lrLogp(0, 0);
    var th = new Float64Array(2 * n), sc = new Float64Array(2 * n), acc = 0, sp = [0, 0];
    var c = Math.sqrt(2 * eps);
    for (var it = -burn; it < n; it++) {
      var p0 = t[0] + eps * st[0] + c * nrm(), p1 = t[1] + eps * st[1] + c * nrm();
      lrScore(p0, p1, sp);
      var lp = lrLogp(p0, p1);
      var f0 = p0 - t[0] - eps * st[0], f1 = p1 - t[1] - eps * st[1];
      var b0 = t[0] - p0 - eps * sp[0], b1 = t[1] - p1 - eps * sp[1];
      var la = lp - lt - (b0 * b0 + b1 * b1) / (4 * eps) + (f0 * f0 + f1 * f1) / (4 * eps);
      if (Math.log(r()) < la) { t = [p0, p1]; st = [sp[0], sp[1]]; lt = lp; if (it >= 0) acc++; }
      if (it >= 0) { th[2 * it] = t[0]; th[2 * it + 1] = t[1]; sc[2 * it] = st[0]; sc[2 * it + 1] = st[1]; }
    }
    return { theta: th, score: sc, n: n, acc: acc / n };
  }
  /** Control variates L u for polynomial u: degree 1 gives the score, degree 2 adds
   *  L(t_i^2) = 2 + 2 t_i s_i and L(t_0 t_1) = t_1 s_0 + t_0 s_1. */
  function cvRow(t0, t1, s0, s1, deg, out) {
    out[0] = s0; out[1] = s1;
    if (deg >= 2) { out[2] = 2 + 2 * t0 * s0; out[3] = 2 + 2 * t1 * s1; out[4] = t1 * s0 + t0 * s1; }
    return out;
  }
  function solveSmall(A, b, k) {
    // Gaussian elimination with partial pivoting on a k x k system (A row-major)
    var M = A.slice(), x = b.slice(), i, j, c;
    for (c = 0; c < k; c++) {
      var p = c; for (i = c + 1; i < k; i++) if (Math.abs(M[i * k + c]) > Math.abs(M[p * k + c])) p = i;
      if (p !== c) { for (j = 0; j < k; j++) { var t = M[c * k + j]; M[c * k + j] = M[p * k + j]; M[p * k + j] = t; } var tb = x[c]; x[c] = x[p]; x[p] = tb; }
      for (i = c + 1; i < k; i++) { var f = M[i * k + c] / M[c * k + c]; for (j = c; j < k; j++) M[i * k + j] -= f * M[c * k + j]; x[i] -= f * x[c]; }
    }
    for (i = k - 1; i >= 0; i--) { var s = x[i]; for (j = i + 1; j < k; j++) s -= M[i * k + j] * x[j]; x[i] = s / M[i * k + i]; }
    return x;
  }
  /** Plain and zero-variance estimates from the first m states of a chain. The ZV
   *  coefficients are fitted by least squares on the same states (Mira et al. 2013). */
  function zv(chain, hIdx, m) {
    m = m || chain.n;
    var h = LR_H[hIdx].f, K = 5, out = { plain: 0, deg1: 0, deg2: 0 };
    var row = new Float64Array(5), i, a, b;
    var mh = 0, mpsi = new Float64Array(K), Spp = new Float64Array(K * K), Sph = new Float64Array(K);
    for (i = 0; i < m; i++) {
      var t0 = chain.theta[2 * i], t1 = chain.theta[2 * i + 1];
      cvRow(t0, t1, chain.score[2 * i], chain.score[2 * i + 1], 2, row);
      var hv = h(t0);
      mh += hv;
      for (a = 0; a < K; a++) { mpsi[a] += row[a]; Sph[a] += row[a] * hv; for (b = 0; b < K; b++) Spp[a * K + b] += row[a] * row[b]; }
    }
    mh /= m;
    for (a = 0; a < K; a++) mpsi[a] /= m;
    out.plain = mh;
    [1, 2].forEach(function (deg) {
      var k = deg === 1 ? 2 : 5, C = [], c = [];
      for (a = 0; a < k; a++) {
        c.push(Sph[a] / m - mpsi[a] * mh);
        for (b = 0; b < k; b++) C.push(Spp[a * K + b] / m - mpsi[a] * mpsi[b]);
      }
      var beta = solveSmall(C, c, k), est = mh;
      for (a = 0; a < k; a++) est -= beta[a] * mpsi[a];
      out[deg === 1 ? "deg1" : "deg2"] = est;
    });
    return out;
  }
  /** R independent chains of length n; they do not depend on the test function. */
  function replicateChains(n, eps, R, seed0) {
    var out = [];
    for (var rr = 0; rr < R; rr++) out.push(mala(n, eps, seed0 + 1000 * rr + 17));
    return out;
  }
  /** Mean squared error of each estimator of test function hIdx over a set of chains. */
  function mseOver(chains, hIdx) {
    var ref = lrReference()[hIdx], se = { plain: 0, deg1: 0, deg2: 0 }, acc = 0, R = chains.length;
    chains.forEach(function (ch) {
      var e = zv(ch, hIdx);
      acc += ch.acc;
      for (var key in se) se[key] += (e[key] - ref) * (e[key] - ref);
    });
    for (var k2 in se) se[k2] /= R;
    return { mse: se, acc: acc / R, vr1: se.plain / se.deg1, vr2: se.plain / se.deg2 };
  }
  /** Mean squared error over R independent chains, for each estimator. */
  function replicate(hIdx, n, eps, R, seed0) { return mseOver(replicateChains(n, eps, R, seed0), hIdx); }

  return {
    rng: rng, normalGen: normalGen,
    TARGETS: TARGETS, HS: HS, EPS_GRID: EPS_GRID, EPS_MIN: EPS_MIN, EPS_MAX: EPS_MAX, gridSize: gridSize,
    setGridSize: setGridSize, setup: setup, kernel: kernel, analyse: analyse, curves: curves, rmse: rmse, bestEps: bestEps, rmseCurve: rmseCurve,
    predictedDensity: predictedDensity,
    LR: LR, LR_H: LR_H, LR_VIEW: LR_VIEW, lrLogp: lrLogp, lrScore: lrScore, lrReference: lrReference,
    lrDensityGrid: lrDensityGrid, mala: mala, zv: zv, replicate: replicate, replicateChains: replicateChains, mseOver: mseOver
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SteinMCMCModel;
