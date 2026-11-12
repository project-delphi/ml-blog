/* Two models for Stein's Method, Part 6. Numbers only; widgets.js draws them and
 * src/check_model.js checks them under node.
 *  1. A random one-hidden-layer network at one input: constants of each activation,
 *     the two Stein bounds on its distance from N(0, 1), and samples of real networks.
 *     The exact distances come precomputed from src/make_data.py (widget-data/data.js).
 *  2. A single-index model y = g(w.x) + noise with Gaussian inputs: the link's Hermite
 *     coefficients, and the first- and second-order Stein estimators of w. */
"use strict";

var SteinNetsModel = (function () {
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
  /** Standard normal CDF by the Abramowitz & Stegun 26.2.17 (Zelen–Severo) polynomial,
   *  absolute error below 7.5e-8. */
  function normCdf(z) {
    var t = 1 / (1 + 0.2316419 * Math.abs(z));
    var d = 0.3989422804014327 * Math.exp(-z * z / 2);
    var p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return z > 0 ? 1 - p : p;
  }
  var SQ2PI = Math.sqrt(2 / Math.PI);

  // ======================================================= 1. wide random networks
  var ACTS = {
    linear: { label: "Linear", f: function (g) { return g; } },
    relu: { label: "ReLU", f: function (g) { return g > 0 ? g : 0; } },
    gelu: { label: "GELU", f: function (g) { return g * normCdf(g); } },
    tanh: { label: "tanh", f: function (g) { return Math.tanh(g); } },
    sign: { label: "sign", f: function (g) { return g > 0 ? 1 : g < 0 ? -1 : 0; } }
  };
  /** E[F(Z)] for Z ~ N(0, 1), by the midpoint rule on [-12, 12]. The kinks and jumps of
   *  ReLU, abs and sign sit on cell edges, never on a sample point. */
  function gauss(F) {
    var K = 48000, h = 24 / K, s = 0;
    for (var i = 0; i < K; i++) {
      var z = -12 + (i + 0.5) * h;
      s += F(z) * Math.exp(-z * z / 2);
    }
    return s * h / Math.sqrt(2 * Math.PI);
  }
  var constCache = {};
  /** Moments of sigma(Z) and the constants of the two bounds. */
  function constants(act) {
    if (constCache[act]) return constCache[act];
    var f = ACTS[act].f;
    var c2 = gauss(function (z) { var v = f(z); return v * v; });
    var c3 = gauss(function (z) { return Math.pow(Math.abs(f(z)), 3); });
    var c4 = gauss(function (z) { var v = f(z); return v * v * v * v; });
    var kappa = c4 / (c2 * c2) - 1;
    // sign has kappa = 0 exactly; quadrature leaves ~1e-14, which would plot as a curve on the axis
    if (Math.abs(kappa) < 1e-12) kappa = 0;
    // the summand X = v sigma(g) / sqrt(c2) with v ~ N(0, 1): E|v|^3 = 2 sqrt(2/pi), E v^4 = 3
    var EX3 = 2 * SQ2PI * c3 / Math.pow(c2, 1.5), EX4 = 3 * c4 / (c2 * c2);
    var out = {
      c2: c2, kappa: kappa, EX3: EX3, EX4: EX4,
      thm4: EX3 + SQ2PI * Math.sqrt(EX4),       // Part 2's Theorem 4: d_W <= thm4 / sqrt(n)
      mix2: 2 * SQ2PI * kappa,                  // Corollary 2: d_W <= mix2 / n
      mix1: SQ2PI * Math.sqrt(kappa),           // first-order form: d_W <= mix1 / sqrt(n)
      limit: (Math.exp(-1.5) + 0.25) / Math.sqrt(2 * Math.PI) * kappa  // n d_W -> limit
    };
    constCache[act] = out;
    return out;
  }
  function bounds(act, n) {
    var C = constants(act);
    return { thm4: C.thm4 / Math.sqrt(n), mix: Math.min(C.mix2 / n, C.mix1 / Math.sqrt(n)), mix2: C.mix2 / n, mix1: C.mix1 / Math.sqrt(n) };
  }
  /** Outputs of M random width-n networks at one fixed input x with |x|^2 = d:
   *  W = (1 / sqrt(n c2)) sum_i v_i sigma(w_i . x), with w_i ~ N(0, I/d), v_i ~ N(0, 1).
   *  w_i . x ~ N(0, 1), so only those inner products are drawn. */
  function sampleOutputs(act, n, M, seed) {
    var r = rng(seed), nrm = normalGen(r), f = ACTS[act].f, c2 = constants(act).c2, out = new Float64Array(M);
    var sc = 1 / Math.sqrt(n * c2);
    for (var m = 0; m < M; m++) {
      var s = 0;
      for (var i = 0; i < n; i++) s += nrm() * f(nrm());
      out[m] = s * sc;
    }
    return out;
  }
  /** ReLU's limit kernels (Cho & Saul's arc-cosine kernels, normalised so K(1) = 1). */
  function reluK(rho) { return (Math.sqrt(1 - rho * rho) + (Math.PI - Math.acos(rho)) * rho) / Math.PI; }
  function reluKprime(rho) { return (Math.PI - Math.acos(rho)) / Math.PI; }

  // ===================================================== 2. hidden directions
  var LINKS = {
    linear: { label: "t", f: function (t) { return t; } },
    tanh: { label: "tanh t", f: function (t) { return Math.tanh(t); } },
    relu: { label: "ReLU", f: function (t) { return t > 0 ? t : 0; } },
    abs: { label: "|t|", f: function (t) { return Math.abs(t); } },
    he2: { label: "t² − 1", f: function (t) { return t * t - 1; } },
    he3: { label: "t³ − 3t", f: function (t) { return t * t * t - 3 * t; } }
  };
  var HE = [
    function () { return 1; },
    function (t) { return t; },
    function (t) { return t * t - 1; },
    function (t) { return t * t * t - 3 * t; },
    function (t) { return t * t * t * t - 6 * t * t + 3; }
  ];
  var linkCache = {};
  /** Mean, standard deviation and Stein-form Hermite coefficients c_k = E[g^(k)(Z)] = E[g(Z) He_k(Z)]
   *  of the standardised link (g - mean) / sd. */
  function linkInfo(key) {
    if (linkCache[key]) return linkCache[key];
    var g = LINKS[key].f;
    var mu = gauss(g), sd = Math.sqrt(gauss(function (t) { var v = g(t) - mu; return v * v; }));
    var c = [1, 2, 3, 4].map(function (k) { return gauss(function (t) { return (g(t) - mu) * HE[k](t); }) / sd; });
    var kstar = 0; for (var k = 0; k < 4; k++) if (Math.abs(c[k]) > 1e-6) { kstar = k + 1; break; }
    var out = { mean: mu, sd: sd, c: c, kstar: kstar };
    linkCache[key] = out;
    return out;
  }
  /** Cyclic Jacobi eigen-decomposition of a symmetric d x d matrix (row-major). */
  function eigSym(A, d) {
    var a = Float64Array.from(A), V = new Float64Array(d * d), i, j, k;
    for (i = 0; i < d; i++) V[i * d + i] = 1;
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0;
      for (i = 0; i < d; i++) for (j = i + 1; j < d; j++) off += a[i * d + j] * a[i * d + j];
      if (off < 1e-22) break;
      for (var p = 0; p < d; p++) for (var q = p + 1; q < d; q++) {
        var apq = a[p * d + q];
        if (Math.abs(apq) < 1e-300) continue;
        var th = (a[q * d + q] - a[p * d + p]) / (2 * apq);
        var t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        var c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (k = 0; k < d; k++) {
          var akp = a[k * d + p], akq = a[k * d + q];
          a[k * d + p] = c * akp - s * akq; a[k * d + q] = s * akp + c * akq;
        }
        for (k = 0; k < d; k++) {
          var apk = a[p * d + k], aqk = a[q * d + k];
          a[p * d + k] = c * apk - s * aqk; a[q * d + k] = s * apk + c * aqk;
        }
        for (k = 0; k < d; k++) {
          var vkp = V[k * d + p], vkq = V[k * d + q];
          V[k * d + p] = c * vkp - s * vkq; V[k * d + q] = s * vkp + c * vkq;
        }
      }
    }
    var vals = []; for (i = 0; i < d; i++) vals.push(a[i * d + i]);
    return { values: vals, vectors: V };
  }
  var N_LIST = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800];
  /** One data set of size max(nList) with hidden direction w = e_1, x ~ N(0, I_d),
   *  y = standardised g(x_1) + sigma * noise. At each n in nList (nested prefixes),
   *  the overlaps |cos| of the first-order estimate mean(y x) and of the top eigenvector
   *  (largest |eigenvalue|) of mean(y (x x^T - I)) with w. */
  function experiment(key, d, sigma, seed, nList) {
    nList = nList || N_LIST;
    var info = linkInfo(key), g = LINKS[key].f, r = rng(seed), nrm = normalGen(r);
    var Sx = new Float64Array(d), Sxx = new Float64Array(d * d), Sy = 0, x = new Float64Array(d);
    var out = { n: nList.slice(), first: [], second: [] }, next = 0, nMax = nList[nList.length - 1], i, j;
    for (var t = 1; t <= nMax; t++) {
      for (i = 0; i < d; i++) x[i] = nrm();
      var y = (g(x[0]) - info.mean) / info.sd + sigma * nrm();
      Sy += y;
      for (i = 0; i < d; i++) {
        var yi = y * x[i];
        Sx[i] += yi;
        for (j = i; j < d; j++) Sxx[i * d + j] += yi * x[j];
      }
      if (t === nList[next]) {
        var nrm2 = 0; for (i = 0; i < d; i++) nrm2 += Sx[i] * Sx[i];
        out.first.push(Math.abs(Sx[0]) / Math.sqrt(nrm2));
        var A = new Float64Array(d * d);
        for (i = 0; i < d; i++) for (j = i; j < d; j++) {
          var v = Sxx[i * d + j] / t - (i === j ? Sy / t : 0);
          A[i * d + j] = v; A[j * d + i] = v;
        }
        var E = eigSym(A, d), best = 0;
        for (i = 1; i < d; i++) if (Math.abs(E.values[i]) > Math.abs(E.values[best])) best = i;
        var vn = 0; for (i = 0; i < d; i++) vn += E.vectors[i * d + best] * E.vectors[i * d + best];
        out.second.push(Math.abs(E.vectors[best]) / Math.sqrt(vn));
        next++;
      }
    }
    return out;
  }
  /** experiment() over seeds s0 .. s0 + K - 1: mean, min and max overlap at each n. */
  function sweep(key, d, sigma, K, s0, nList) {
    var runs = [];
    for (var k = 0; k < K; k++) runs.push(experiment(key, d, sigma, s0 + k, nList));
    function agg(field) {
      return runs[0].n.map(function (_, i) {
        var v = runs.map(function (r) { return r[field][i]; });
        return { mean: v.reduce(function (a, b) { return a + b; }, 0) / v.length, min: Math.min.apply(null, v), max: Math.max.apply(null, v) };
      });
    }
    return { n: runs[0].n, first: agg("first"), second: agg("second") };
  }

  return {
    rng: rng, normalGen: normalGen, normCdf: normCdf, gauss: gauss,
    ACTS: ACTS, constants: constants, bounds: bounds, sampleOutputs: sampleOutputs, reluK: reluK, reluKprime: reluKprime,
    LINKS: LINKS, linkInfo: linkInfo, eigSym: eigSym, N_LIST: N_LIST, experiment: experiment, sweep: sweep
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = SteinNetsModel;
