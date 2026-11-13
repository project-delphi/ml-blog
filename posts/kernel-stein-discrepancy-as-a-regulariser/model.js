/* Batch regularisers toward N(0, I_2) on a free 2-D batch: their values, their
 * gradients with respect to every point, how far each value sits from its spread on
 * Gaussian batches, and gradient descent of the points. Numbers only; widgets.js
 * draws them and src/check_model.js checks them under node. */
"use strict";

var KsdRegModel = (function () {
  var D = 2;
  var MIX_W = 10; // weight of the Gaussian part in the mixed kernel IMQ + 10 Gaussian

  // ------------------------------------------------------------------ random numbers
  function rng(seed) {
    var a = seed >>> 0 || 1;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function normalGen(u) {
    var spare = null;
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var a = u() || 1e-12, b = u(), r = Math.sqrt(-2 * Math.log(a));
      spare = r * Math.sin(2 * Math.PI * b);
      return r * Math.cos(2 * Math.PI * b);
    };
  }
  function gamma(shape, u, nrm) { // Marsaglia-Tsang, shape >= 1
    var d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
    for (;;) {
      var x = nrm(), v = 1 + c * x;
      if (v <= 0) continue;
      v = v * v * v;
      var w = u();
      if (w < 1 - 0.0331 * x * x * x * x || Math.log(w) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  // ------------------------------------------------------------------ batch laws
  // Every law but "shrunk" has mean 0 and covariance I, so it agrees with N(0, I)
  // on the first two moments.
  var LAWS = ["gaussian", "ring", "square", "clusters", "student", "shrunk"];
  var NU = 5;
  function sample(law, n, seed) {
    var u = rng(seed), nrm = normalGen(u), X = new Float64Array(2 * n), i;
    for (i = 0; i < n; i++) {
      var x, y;
      if (law === "gaussian") { x = nrm(); y = nrm(); }
      else if (law === "shrunk") { x = 0.3 * nrm(); y = 0.3 * nrm(); }
      else if (law === "ring") {
        // radius^2 has mean 2, so E x^2 = E y^2 = 1
        var th = 2 * Math.PI * u(), r = Math.sqrt(2 - 0.0025) + 0.05 * nrm();
        x = r * Math.cos(th); y = r * Math.sin(th);
      } else if (law === "square") { x = Math.sqrt(3) * (2 * u() - 1); y = Math.sqrt(3) * (2 * u() - 1); }
      else if (law === "clusters") {
        // four clusters at (+-0.9, +-0.9), within-cluster variance 1 - 0.81
        var s = Math.sqrt(1 - 0.81);
        x = (u() < 0.5 ? -0.9 : 0.9) + s * nrm(); y = (u() < 0.5 ? -0.9 : 0.9) + s * nrm();
      } else if (law === "student") {
        // multivariate t with 5 degrees of freedom, scaled to covariance I
        var g = gamma(NU / 2, u, nrm) * 2, sc = Math.sqrt((NU - 2) / g);
        x = nrm() * sc; y = nrm() * sc;
      } else throw new Error("law " + law);
      X[2 * i] = x; X[2 * i + 1] = y;
    }
    return X;
  }

  // ------------------------------------------------------------------ regularisers
  // Each returns { v: value, g: gradient with respect to X (length 2n) }.
  // `opts.ustat` (default true) picks the U-statistic; false gives the V-statistic.

  // KSD under the linear kernel 1 + x^T y. V-statistic: |mean|^2 + |M2 - I|_F^2, whose
  // gradient on z_i is affine, (2/n) mean + (4/n)(M2 - I) z_i. U-statistic:
  // grad U = n/(n-1) grad V + (2 - 4|z_i|^2) z_i / (n(n-1)).
  function moments(X, opts) {
    var n = X.length / 2, m0 = 0, m1 = 0, a = 0, b = 0, c = 0, s4 = 0, i;
    var ustat = !opts || opts.ustat !== false;
    for (i = 0; i < n; i++) {
      var x = X[2 * i], y = X[2 * i + 1], r2 = x * x + y * y;
      m0 += x; m1 += y; a += x * x; b += x * y; c += y * y; s4 += r2 * r2;
    }
    var s2 = a + c; // sum |z_i|^2
    m0 /= n; m1 /= n; a = a / n - 1; b /= n; c = c / n - 1; // M2 - I = [[a, b], [b, c]]
    var v = m0 * m0 + m1 * m1 + a * a + 2 * b * b + c * c, g = new Float64Array(2 * n);
    if (ustat) {
      // sum_{i != j} k_p = n^2 V - sum_i k_p(z_i, z_i),  k_p(z, z) = |z|^4 - |z|^2 + d
      v = (n * n * v - (s4 - s2 + D * n)) / (n * (n - 1));
    }
    for (i = 0; i < n; i++) {
      var xi = X[2 * i], yi = X[2 * i + 1];
      var gx = (2 * m0 + 4 * (a * xi + b * yi)) / n, gy = (2 * m1 + 4 * (b * xi + c * yi)) / n;
      if (ustat) {
        var f = (2 - 4 * (xi * xi + yi * yi)) / (n * (n - 1));
        gx = n / (n - 1) * gx + f * xi; gy = n / (n - 1) * gy + f * yi;
      }
      g[2 * i] = gx; g[2 * i + 1] = gy;
    }
    return { v: v, g: g };
  }

  // Stein kernel of N(0, I_2) and its gradient in the first argument.
  // IMQ base kernel (c2 + u)^beta, u = |x - y|^2.
  function steinImq(x0, x1, y0, y1, c2, beta, out) {
    var r0 = x0 - y0, r1 = x1 - y1, u = r0 * r0 + r1 * r1, b = c2 + u, xy = x0 * y0 + x1 * y1;
    var kb = Math.pow(b, beta), k1 = kb / b, k2 = k1 / b, k3 = k2 / b, B = beta;
    var kp = xy * kb + 2 * B * u * k1 - 2 * B * D * k1 - 4 * B * (B - 1) * u * k2;
    if (out) {
      var G = 2 * B * xy * k1 + 4 * B * k1 + 4 * B * (B - 1) * u * k2 - 4 * B * D * (B - 1) * k2 -
        8 * B * (B - 1) * k2 - 8 * B * (B - 1) * (B - 2) * u * k3;
      out[0] = y0 * kb + r0 * G;
      out[1] = y1 * kb + r1 * G;
    }
    return kp;
  }
  // Gaussian base kernel exp(-u / (2 l2)).
  function steinRbf(x0, x1, y0, y1, l2, out) {
    var r0 = x0 - y0, r1 = x1 - y1, u = r0 * r0 + r1 * r1, xy = x0 * y0 + x1 * y1;
    var k = Math.exp(-u / (2 * l2)), gg = xy + D / l2 - u * (1 / l2 + 1 / (l2 * l2));
    if (out) {
      var G = 2 / l2 + 2 / (l2 * l2) + gg / l2;
      out[0] = k * (y0 - r0 * G);
      out[1] = k * (y1 - r1 * G);
    }
    return k * gg;
  }

  function ksd(X, kernel, opts) {
    var n = X.length / 2, ustat = opts.ustat !== false, g = new Float64Array(2 * n), tmp = [0, 0];
    var c2 = 1, beta = -0.5, l2 = opts.l2 || rbfL2(n), sum = 0, diag = 0, i, j;
    for (i = 0; i < n; i++) {
      var xi = X[2 * i], yi = X[2 * i + 1];
      for (j = 0; j < n; j++) {
        if (j === i) continue;
        var kp = kernel === "imq"
          ? steinImq(xi, yi, X[2 * j], X[2 * j + 1], c2, beta, tmp)
          : steinRbf(xi, yi, X[2 * j], X[2 * j + 1], l2, tmp);
        sum += kp;
        g[2 * i] += 2 * tmp[0]; g[2 * i + 1] += 2 * tmp[1];
      }
      if (!ustat) {
        // k_p(x, x) = |x|^2 c2^beta - 2 beta d c2^(beta-1)   (IMQ)
        //           = |x|^2 + d / l2                          (Gaussian)
        var s = kernel === "imq" ? Math.pow(c2, beta) : 1;
        diag += (xi * xi + yi * yi) * s + (kernel === "imq" ? -2 * beta * D * Math.pow(c2, beta - 1) : D / l2);
        g[2 * i] += 2 * xi * s; g[2 * i + 1] += 2 * yi * s;
      }
    }
    var norm = ustat ? n * (n - 1) : n * n;
    for (i = 0; i < 2 * n; i++) g[i] /= norm;
    return { v: (sum + diag) / norm, g: g };
  }

  // The rule the post uses: l^2 = d / log n.
  function rbfL2(n) { return D / Math.log(n); }

  // MMD^2 (unbiased) between the batch X and a target sample Y under the Wasserstein
  // autoencoder's kernel k = sum_s C / (C + u), C = 2 d s, the kernel the CIFAR-10 runs use.
  // kwae returns k and c with grad_x k(x, y) = c (x - y).
  var WAE_SCALES = [0.1, 0.2, 0.5, 1, 2, 5, 10];
  function kwae(u, out) {
    var k = 0, c = 0;
    for (var s = 0; s < WAE_SCALES.length; s++) {
      var C = 2 * D * WAE_SCALES[s], q = C + u;
      k += C / q; c += -2 * C / (q * q);
    }
    out[1] = c;
    return k;
  }
  function mmd(X, Y) {
    var n = X.length / 2, m = Y.length / 2, g = new Float64Array(2 * n), sxx = 0, syy = 0, sxy = 0, i, j;
    var t = [0, 0];
    for (i = 0; i < n; i++) {
      var xi = X[2 * i], yi = X[2 * i + 1];
      for (j = 0; j < n; j++) {
        if (j === i) continue;
        var r0 = xi - X[2 * j], r1 = yi - X[2 * j + 1];
        sxx += kwae(r0 * r0 + r1 * r1, t);
        g[2 * i] += 2 * t[1] * r0 / (n * (n - 1)); g[2 * i + 1] += 2 * t[1] * r1 / (n * (n - 1));
      }
      for (j = 0; j < m; j++) {
        var q0 = xi - Y[2 * j], q1 = yi - Y[2 * j + 1];
        sxy += kwae(q0 * q0 + q1 * q1, t);
        g[2 * i] -= 2 * t[1] * q0 / (n * m); g[2 * i + 1] -= 2 * t[1] * q1 / (n * m);
      }
    }
    for (i = 0; i < m; i++) for (j = 0; j < m; j++) {
      if (j === i) continue;
      var p0 = Y[2 * i] - Y[2 * j], p1 = Y[2 * i + 1] - Y[2 * j + 1];
      syy += kwae(p0 * p0 + p1 * p1, t);
    }
    return { v: sxx / (n * (n - 1)) + syy / (m * (m - 1)) - 2 * sxy / (n * m), g: g };
  }

  // SIGReg: Epps-Pulley on M evenly spaced directions, 17 knots on [-5, 5].
  var KN = 17, TK = [], WK_ = [];
  (function () {
    for (var k = 0; k < KN; k++) {
      var t = -5 + 10 * k / (KN - 1), trap = (k === 0 || k === KN - 1) ? 5 / 16 : 10 / 16;
      TK.push(t); WK_.push(Math.exp(-t * t / 2) * trap);
    }
  })();
  function sigreg(X, M) {
    M = M || 32;
    var n = X.length / 2, g = new Float64Array(2 * n), total = 0, C = new Float64Array(KN), S = new Float64Array(KN);
    var p = new Float64Array(n), m, i, k;
    for (m = 0; m < M; m++) {
      var th = Math.PI * m / M, a0 = Math.cos(th), a1 = Math.sin(th);
      for (i = 0; i < n; i++) p[i] = a0 * X[2 * i] + a1 * X[2 * i + 1];
      for (k = 0; k < KN; k++) {
        var c = 0, s = 0;
        for (i = 0; i < n; i++) { c += Math.cos(TK[k] * p[i]); s += Math.sin(TK[k] * p[i]); }
        C[k] = c / n - Math.exp(-TK[k] * TK[k] / 2); S[k] = s / n;
        total += n * WK_[k] * (C[k] * C[k] + S[k] * S[k]);
      }
      for (i = 0; i < n; i++) {
        var dp = 0;
        for (k = 0; k < KN; k++) dp += 2 * WK_[k] * TK[k] * (S[k] * Math.cos(TK[k] * p[i]) - C[k] * Math.sin(TK[k] * p[i]));
        g[2 * i] += a0 * dp / M; g[2 * i + 1] += a1 * dp / M;
      }
    }
    return { v: total / M, g: g };
  }

  var REGS = ["moments", "ksd_imq", "ksd_rbf", "mmd", "sigreg"];
  var LABELS = { moments: "Moments", ksd_imq: "KSD\u00a0IMQ", ksd_rbf: "KSD\u00a0Gaussian",
                 ksd_mix: "KSD\u00a0mixed", mmd: "MMD", sigreg: "SIGReg" };

  // One evaluation. MMD draws a fresh N(0, I) sample of the batch's size from `nrm`.
  function evaluate(reg, X, opts) {
    opts = opts || {};
    if (reg === "moments") return moments(X, opts);
    if (reg === "ksd_imq") return ksd(X, "imq", opts);
    if (reg === "ksd_rbf") return ksd(X, "rbf", opts);
    if (reg === "ksd_mix") {
      var a = ksd(X, "imq", opts), b = ksd(X, "rbf", opts), gm = new Float64Array(a.g.length);
      for (var q = 0; q < gm.length; q++) gm[q] = a.g[q] + MIX_W * b.g[q];
      return { v: a.v + MIX_W * b.v, g: gm };
    }
    if (reg === "sigreg") return sigreg(X, opts.M);
    if (reg === "mmd") {
      var n = X.length / 2, nrm = opts.nrm || normalGen(rng(opts.seed || 99)), Y = new Float64Array(2 * n);
      for (var i = 0; i < 2 * n; i++) Y[i] = nrm();
      return mmd(X, Y);
    }
    throw new Error("reg " + reg);
  }

  // Spread of each regulariser on Gaussian batches of size n: mean, sd and 95th percentile.
  function nullStats(reg, n, reps, seed, opts) {
    reps = reps || 100;
    var vals = [], nrm = normalGen(rng((seed || 1) * 7919 + 13));
    for (var r = 0; r < reps; r++) {
      var X = sample("gaussian", n, (seed || 1) * 100003 + r);
      vals.push(evaluate(reg, X, Object.assign({}, opts, { nrm: nrm })).v);
    }
    vals.sort(function (a, b) { return a - b; });
    var mu = vals.reduce(function (s, v) { return s + v; }, 0) / reps;
    var sd = Math.sqrt(vals.reduce(function (s, v) { return s + (v - mu) * (v - mu); }, 0) / (reps - 1));
    return { mean: mu, sd: sd, q95: vals[Math.floor(0.95 * reps)] };
  }

  function maxNorm(g) {
    var m = 0;
    for (var i = 0; i < g.length; i += 2) m = Math.max(m, Math.hypot(g[i], g[i + 1]));
    return m;
  }

  // Gradient-descent step on the points. The step is scaled per regulariser by
  // `gref`, the largest gradient it puts on the shrunk batch, so every regulariser
  // moves that batch's fastest point by `eta` per step.
  function descend(X, reg, steps, eta, gref, opts) {
    var last = null;
    for (var s = 0; s < steps; s++) {
      last = evaluate(reg, X, opts);
      for (var i = 0; i < X.length; i++) X[i] -= eta / gref * last.g[i];
    }
    return last;
  }

  // Moments of the radius: mean |x|^2 (2 for N(0, I_2)) and mean |x|^4 (8 for N(0, I_2)).
  function radialMoments(X) {
    var n = X.length / 2, m2 = 0, m4 = 0;
    for (var i = 0; i < n; i++) { var r2 = X[2 * i] * X[2 * i] + X[2 * i + 1] * X[2 * i + 1]; m2 += r2; m4 += r2 * r2; }
    return { r2: m2 / n, r4: m4 / n };
  }

  var api = { D: D, LAWS: LAWS, REGS: REGS, LABELS: LABELS, rng: rng, normalGen: normalGen, sample: sample,
              moments: moments, ksd: ksd, mmd: mmd, sigreg: sigreg, steinImq: steinImq, steinRbf: steinRbf,
              rbfL2: rbfL2, evaluate: evaluate, nullStats: nullStats, maxNorm: maxNorm, descend: descend,
              radialMoments: radialMoments };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
