/* Normal approximation for standardised sums. Exact d_W values come from
 * widget-data/data.js (src/make_data.py); this file adds the Ross (2011,
 * Thm 3.2) bound and seeded draws of W for the histogram. Numbers only. */
"use strict";

var SteinCltModel = (function () {
  var SQ2PI = Math.sqrt(2 / Math.PI);
  var data = null;

  function setData(d) { data = d; }
  function getData() {
    if (!data && typeof window !== "undefined" && window.WK_DATA) data = window.WK_DATA.steinClt;
    return data;
  }

  /** The law's moments and exact d_W for n = 1..200. Bernoulli snaps p to the grid. */
  function law(key, p) {
    var d = getData();
    if (key === "bernoulli") {
      var best = d.laws.bernoulli[0];
      d.laws.bernoulli.forEach(function (e) { if (Math.abs(e.p - p) < Math.abs(best.p - p)) best = e; });
      return { key: key, p: best.p, m3: best.m3, m4: best.m4, dw: best.dw };
    }
    var e = d.laws[key];
    return { key: key, m3: e.m3, m4: e.m4, dw: e.dw };
  }

  /** Ross (2011, Thm 3.2) for i.i.d. summands: (E|X|^3 + sqrt(2/pi) sqrt(E X^4)) / sqrt(n). */
  function bound(L, n) { return (L.m3 + SQ2PI * Math.sqrt(L.m4)) / Math.sqrt(n); }

  /** Smallest bound / d_W over n = 1..200 for a law. */
  function worstRatio(L) {
    var r = Infinity;
    for (var n = 1; n <= L.dw.length; n++) r = Math.min(r, bound(L, n) / L.dw[n - 1]);
    return r;
  }

  // -------------------------------------------------------------- sampling
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
  function gammaDraw(k, r, nrm) { // Marsaglia & Tsang, k >= 1
    var d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      var x, v;
      do { x = nrm(); v = 1 + c * x; } while (v <= 0);
      v = v * v * v;
      var u = r();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  /** M draws of W = (sum of n standardised summands) / sqrt(n). */
  function drawW(key, n, M, seed) {
    var r = rng(seed), nrm = normalGen(r), out = new Float64Array(M), i, j, s;
    var sn = Math.sqrt(n);
    if (key === "exponential") {
      for (i = 0; i < M; i++) out[i] = (gammaDraw(n, r, nrm) - n) / sn;
    } else if (key === "uniform") {
      var a = Math.sqrt(3);
      for (i = 0; i < M; i++) {
        s = 0;
        for (j = 0; j < n; j++) s += (2 * r() - 1) * a;
        out[i] = s / sn;
      }
    } else if (key === "mixture") {
      var mx = getData().mixture;
      for (i = 0; i < M; i++) {
        var k = 0;
        for (j = 0; j < n; j++) if (r() < mx.w) k++;
        var mean = k * mx.m1 + (n - k) * mx.m2, sd = Math.sqrt(k * mx.s1 * mx.s1 + (n - k) * mx.s2 * mx.s2);
        out[i] = (mean + sd * nrm() - n * mx.mean) / (mx.sd * sn);
      }
    } else throw new Error("drawW: " + key);
    return out;
  }

  /** Histogram on [lo, hi) with the given bin width, as densities. */
  function histogram(xs, lo, hi, width) {
    var nb = Math.round((hi - lo) / width), counts = new Array(nb).fill(0), outside = 0;
    for (var i = 0; i < xs.length; i++) {
      var b = Math.floor((xs[i] - lo) / width);
      if (b >= 0 && b < nb) counts[b]++; else outside++;
    }
    return { lo: lo, width: width, density: counts.map(function (c) { return c / (xs.length * width); }),
      outside: outside / xs.length };
  }

  /** Exact pmf of the standardised binomial: atoms at (k - np)/sqrt(npq). */
  function bernoulliAtoms(n, p) {
    var q = 1 - p, sd = Math.sqrt(n * p * q), out = [], logp = Math.log(p), logq = Math.log(q);
    var lfact = [0];
    for (var i = 1; i <= n; i++) lfact.push(lfact[i - 1] + Math.log(i));
    for (var k = 0; k <= n; k++) {
      var lp = lfact[n] - lfact[k] - lfact[n - k] + k * logp + (n - k) * logq;
      out.push({ w: (k - n * p) / sd, mass: Math.exp(lp) });
    }
    return { atoms: out, spacing: 1 / sd };
  }

  function phi(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }

  return { setData: setData, law: law, bound: bound, worstRatio: worstRatio,
    drawW: drawW, histogram: histogram, bernoulliAtoms: bernoulliAtoms, phi: phi };
})();

if (typeof module !== "undefined") module.exports = SteinCltModel;
