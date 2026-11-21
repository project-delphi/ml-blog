/* Stein's identity tester: four standardised laws, three test functions plus a
 * parsed custom one, the Gaussian and Langevin Stein operators, seeded Monte
 * Carlo, and the exact expectation each run should converge to. Numbers only;
 * widgets.js draws them and src/check_model.js checks them under node. */
"use strict";

var SteinLemmaModel = (function () {
  var SQ3 = Math.sqrt(3), LAPLACE_B = 1 / Math.SQRT2;

  // ------------------------------------------------------------ random numbers
  /** mulberry32: a small seeded uniform generator on [0, 1). */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /** Standard normal draws by Box-Muller, caching the second of each pair. */
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
  /** Gamma(k, 1) for k >= 1 (Marsaglia & Tsang 2000). */
  function gammaDraw(k, r, nrm) {
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
  /** log Gamma by the Lanczos approximation (g = 7). */
  function lgamma(x) {
    var c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    var a = c[0], t = x + 7.5;
    for (var i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // ------------------------------------------------------------ distributions
  /** Every law is standardised to mean 0 and variance 1, so only shape differs. */
  function dist(key, nu) {
    if (key === "gauss") return {
      key: key, label: "Gaussian", lo: -Infinity, hi: Infinity,
      pdf: function (x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); },
      score: function (x) { return -x; },
      sample: function (r, nrm) { return nrm(); }
    };
    if (key === "laplace") return {
      key: key, label: "Laplace", lo: -Infinity, hi: Infinity,
      pdf: function (x) { return Math.exp(-Math.abs(x) / LAPLACE_B) / (2 * LAPLACE_B); },
      score: function (x) { return x > 0 ? -1 / LAPLACE_B : (x < 0 ? 1 / LAPLACE_B : 0); },
      sample: function (r) {
        var u = r() - 0.5;
        while (u === -0.5) u = r() - 0.5; // r() can return exactly 0, and log(0) is -Infinity
        return -LAPLACE_B * (u < 0 ? -1 : 1) * Math.log(1 - 2 * Math.abs(u));
      }
    };
    if (key === "t") {
      var s = Math.sqrt((nu - 2) / nu);
      var logc = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI);
      return {
        key: key, label: "Student-t", nu: nu, scale: s, lo: -Infinity, hi: Infinity,
        pdf: function (x) {
          var t = x / s;
          return Math.exp(logc - 0.5 * (nu + 1) * Math.log1p(t * t / nu)) / s;
        },
        // d/dx log p for X = s T: -(nu + 1) x / (nu - 2 + x^2)
        score: function (x) { return -(nu + 1) * x / (nu - 2 + x * x); },
        sample: function (r, nrm) {
          var z = nrm(), v = 2 * gammaDraw(nu / 2, r, nrm);
          return s * z / Math.sqrt(v / nu);
        },
        logc: logc
      };
    }
    if (key === "unif") return {
      key: key, label: "Uniform", lo: -SQ3, hi: SQ3,
      pdf: function (x) { return Math.abs(x) <= SQ3 ? 1 / (2 * SQ3) : 0; },
      score: function () { return 0; },
      sample: function (r) { return (2 * r() - 1) * SQ3; }
    };
    throw new Error("unknown distribution " + key);
  }

  // ------------------------------------------------------------ test functions
  var FUNCS = ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh", "sinh", "cosh", "atan"];

  /** Parse an expression in x into a function. Whitelisted grammar, no eval:
   *  numbers, x, pi, e, + - * / ^, parentheses, and the functions in FUNCS. */
  function parse(src) {
    var toks = [], re = /\s*(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+|[a-z]+|[-+*/^()])/gy, m;
    src = String(src).toLowerCase();
    re.lastIndex = 0;
    while (re.lastIndex < src.length) {
      var at = re.lastIndex;
      m = re.exec(src);
      if (!m) {
        if (/^\s*$/.test(src.slice(at))) break;
        throw new Error("cannot read '" + src.slice(at).trim().slice(0, 8) + "'");
      }
      toks.push(m[1]);
    }
    var i = 0;
    function peek() { return toks[i]; }
    function eat(t) { if (toks[i] !== t) throw new Error("expected '" + t + "'"); i++; }
    function expr() {
      var a = term();
      while (peek() === "+" || peek() === "-") {
        var op = toks[i++], b = term(), l = a;
        a = op === "+" ? (function (l, b) { return function (x) { return l(x) + b(x); }; })(l, b)
                       : (function (l, b) { return function (x) { return l(x) - b(x); }; })(l, b);
      }
      return a;
    }
    function term() {
      var a = unary();
      while (peek() === "*" || peek() === "/") {
        var op = toks[i++], b = unary(), l = a;
        a = op === "*" ? (function (l, b) { return function (x) { return l(x) * b(x); }; })(l, b)
                       : (function (l, b) { return function (x) { return l(x) / b(x); }; })(l, b);
      }
      return a;
    }
    function unary() {
      if (peek() === "-") { i++; var u = unary(); return function (x) { return -u(x); }; }
      if (peek() === "+") { i++; return unary(); }
      return power();
    }
    function power() {
      var base = atom();
      if (peek() === "^") { i++; var ex = unary(); return function (x) { return Math.pow(base(x), ex(x)); }; }
      return base;
    }
    function atom() {
      var t = toks[i++];
      if (t === undefined) throw new Error("unexpected end");
      if (/^[\d.]/.test(t)) { var c = Number(t); return function () { return c; }; }
      if (t === "x") return function (x) { return x; };
      if (t === "pi") return function () { return Math.PI; };
      if (t === "e") return function () { return Math.E; };
      if (t === "(") { var inner = expr(); eat(")"); return inner; }
      if (FUNCS.indexOf(t) >= 0) {
        eat("("); var arg = expr(); eat(")");
        var fn = Math[t];
        return function (x) { return fn(arg(x)); };
      }
      throw new Error("unknown name '" + t + "'");
    }
    if (!toks.length) throw new Error("empty expression");
    var out = expr();
    if (i !== toks.length) throw new Error("unexpected '" + toks[i] + "'");
    return out;
  }

  function testFn(key, custom) {
    if (key === "sin") return { key: key, label: "sin x", f: Math.sin, df: Math.cos };
    if (key === "tanh") return { key: key, label: "tanh x", f: Math.tanh,
      df: function (x) { var t = Math.tanh(x); return 1 - t * t; } };
    if (key === "cube") return { key: key, label: "x³", f: function (x) { return x * x * x; },
      df: function (x) { return 3 * x * x; } };
    var f = parse(custom);
    return { key: "custom", label: custom, f: f, df: function (x) {
      var h = 1e-5 * (1 + Math.abs(x));
      return (f(x + h) - f(x - h)) / (2 * h);
    } };
  }

  /** The two Stein operators applied to f at x. */
  function opValue(op, D, F, x) {
    return op === "gauss" ? F.df(x) - x * F.f(x) : F.df(x) + D.score(x) * F.f(x);
  }

  // --------------------------------------------------------------- quadrature
  function simpson(g, a, b, n) {
    var hh = (b - a) / n, s = g(a) + g(b);
    for (var k = 1; k < n; k++) s += (k % 2 ? 4 : 2) * g(a + k * hh);
    return s * hh / 3;
  }
  /** E_p[(op f)(X)] by quadrature. Student-t substitutes x = s sqrt(nu) tan(theta),
   *  which maps the heavy tails onto a finite interval, then grades the mesh toward
   *  theta = ±pi/2 with theta = sign(v) (pi/2) (1 - (1 - |v|)^4), Jacobian 2 pi (1 - |v|)^3,
   *  because for nu < 3 an integrand growing like x^2 is singular there (midpoint rule in v). */
  /*  L (Gaussian and Laplace only) is the half-width of the window, default 12 and 50, with the
   *  grid spacing kept fixed as it widens; refine > 1 uses a proportionally finer, differently
   *  placed grid, which exactInfo compares against the default to detect poles. */
  function quadrature(op, D, F, L, refine) {
    var g = function (x) { return opValue(op, D, F, x) * D.pdf(x); }, r = refine || 1;
    var even = function (n) { n = Math.round(n); return n % 2 ? n + 1 : n; };
    if (D.key === "gauss") { L = L || 12; return simpson(g, -L, L, even(8000 * r * L / 12)); }
    if (D.key === "laplace") { L = L || 50; var nl = even(40000 * r * L / 50); return simpson(g, -L, 0, nl) + simpson(g, 0, L, nl); }
    if (D.key === "unif") return simpson(function (x) { return opValue(op, D, F, x) / (2 * SQ3); }, -SQ3, SQ3, even(8000 * r));
    var nu = D.nu, s = D.scale, N = Math.round(200000 * r), hv = 2 / N, sum = 0, w0 = Math.exp(D.logc) * Math.sqrt(nu);
    for (var k = 0; k < N; k++) {
      var v = -1 + (k + 0.5) * hv, a = 1 - Math.abs(v);
      var th = (v < 0 ? -1 : 1) * (Math.PI / 2) * (1 - a * a * a * a);
      var x = s * Math.sqrt(nu) * Math.tan(th);
      sum += opValue(op, D, F, x) * w0 * Math.pow(Math.cos(th), nu - 1) * 2 * Math.PI * a * a * a;
    }
    return sum * hv;
  }

  /** Largest value of logh over [x, 1.5x] (65 points): an oscillating f such as
   *  sin x is judged by its peaks, not by one sample that may sit near a zero.
   *  Far out in a tail a NaN is an overflow (exp x − x exp x is ∞ − ∞), so it
   *  counts as unbounded growth. */
  function logEnvelope(logh, x) {
    var m = -Infinity;
    for (var k = 0; k <= 64; k++) {
      var v = logh(x * (1 + 0.5 * k / 64));
      if (v !== v) return Infinity;
      if (v > m) m = v;
    }
    return m;
  }

  /** For a custom f, decide numerically whether (op f)(X) has a mean and a
   *  variance. Student-t: the density falls like |x|^-(ν+1), so E|g| is finite
   *  iff |g| grows slower than |x|^ν; the growth exponent is read off the
   *  envelope of |g| at 10³ and 10⁴ scale units. Gaussian and Laplace: |g|·p and
   *  g²·p must fall by more than a factor e between L/2 and the quadrature limit L.
   *  A NaN on the central stretch of the support (sqrt or log of a negative x)
   *  is reported as "domain": the expectation is not defined at all.
   *  For the Gaussian and Laplace it also picks the quadrature window L: the first
   *  candidate at whose edge |g|·p is below e^-30 on both sides, or null if none is. */
  function tailCheck(op, D, F) {
    var out = { mean: true, variance: true, domain: true, alpha: null, L: undefined };
    var logG = function (x) { return Math.log(Math.abs(opValue(op, D, F, x))); };
    var lo = D.key === "unif" ? -SQ3 : -6 * (D.scale || 1), hi = -lo;
    for (var k = 0; k <= 2000; k++) {
      var v = opValue(op, D, F, lo + (hi - lo) * k / 2000);
      if (v !== v) { out.domain = false; break; }
    }
    if (D.key === "unif" || !out.domain) {
      if (!out.domain) { out.mean = false; out.variance = false; }
      return out;
    }
    [-1, 1].forEach(function (side) {
      if (D.key === "t") {
        var e1 = logEnvelope(logG, side * 1e3 * D.scale), e2 = logEnvelope(logG, side * 1e4 * D.scale);
        var a = (e2 === Infinity) ? Infinity : (e2 === -Infinity ? -Infinity : (e2 - Math.max(e1, -700)) / Math.LN10);
        if (out.alpha === null || a > out.alpha) out.alpha = a;
      } else {
        var L = D.key === "gauss" ? 12 : 50, lp = function (x) { return Math.log(D.pdf(x)); };
        var m1 = logEnvelope(function (x) { return logG(x) + lp(x); }, side * L / 2);
        var m2 = logEnvelope(function (x) { return logG(x) + lp(x); }, side * L);
        var v1 = logEnvelope(function (x) { return 2 * logG(x) + lp(x); }, side * L / 2);
        var v2 = logEnvelope(function (x) { return 2 * logG(x) + lp(x); }, side * L);
        if (!(m2 === -Infinity || m2 < m1 - 1)) out.mean = false;
        if (!(v2 === -Infinity || v2 < v1 - 1)) out.variance = false;
      }
    });
    if (D.key === "t") {
      out.mean = out.alpha < D.nu - 0.01;
      out.variance = 2 * out.alpha < D.nu - 0.01;
    } else {
      var lpw = function (x) { return logG(x) + Math.log(D.pdf(x)); };
      var Ls = D.key === "gauss" ? [12, 24, 40, 60] : [50, 100, 200];
      out.L = null;
      for (var i = 0; i < Ls.length && out.L === null; i++) {
        if (logEnvelope(lpw, -Ls[i]) < -30 && logEnvelope(lpw, Ls[i]) < -30) out.L = Ls[i];
      }
    }
    return out;
  }

  /** What the theory says about (op, law, f): whether the mean and the variance
   *  of (op f)(X) exist, and the boundary term for the uniform. Built-in f use
   *  the closed-form conditions; a custom f goes through tailCheck. */
  function conditions(op, distKey, nu, fnKey, custom) {
    var c = { mean: true, variance: true, domain: true, boundary: 0, note: "", L: undefined };
    var F = testFn(fnKey, custom), tc = null;
    if (fnKey === "custom") {
      tc = tailCheck(op, dist(distKey, nu), F);
      c.mean = tc.mean; c.variance = tc.variance; c.domain = tc.domain; c.L = tc.L;
    } else if (distKey === "t" && fnKey === "cube") {
      if (op === "gauss") { c.mean = nu > 4; c.variance = nu > 8; }
      else c.variance = nu > 4;
    }
    if (distKey === "unif" && op === "langevin" && c.domain) c.boundary = (F.f(SQ3) - F.f(-SQ3)) / (2 * SQ3);
    var grow = tc && tc.alpha !== null && isFinite(tc.alpha) ? " grows like |x|^" + tc.alpha.toFixed(1) + " in the tails" : "";
    if (!c.domain) c.note = "f or its derivative is not finite somewhere in the support of this law (sqrt or log of a negative x, say), so the expectation is not defined.";
    else if (!c.mean && fnKey !== "custom") c.note = "E[X⁴] is infinite for ν ≤ 4, so this expectation does not exist and the running mean drifts.";
    else if (!c.mean && distKey === "t") c.note = "The operator applied to f" + (grow || " grows too fast in the tails") + ", which a Student-t with ν = " + nu + " cannot integrate, so this expectation does not exist and the running mean drifts.";
    else if (!c.mean) c.note = "The operator applied to f, times the density, does not decay in the tails, so this expectation does not exist and the running mean drifts.";
    else if (!c.variance && fnKey !== "custom") c.note = "The variance is infinite here (ν ≤ " + (op === "gauss" ? 8 : 4) + "), so the ±2 SE band is not a valid interval.";
    else if (!c.variance) c.note = "The operator applied to f" + (grow ? grow + ", so its variance is infinite for ν = " + nu : " has infinite variance under this law") + ", and the ±2 SE band is not a valid interval.";
    else if (distKey === "unif" && op === "langevin" && Math.abs(c.boundary) >= 5e-4) c.note = "Boundary term f(√3)p(√3) − f(−√3)p(−√3) = " + c.boundary.toFixed(4) + ", not 0.";
    return c;
  }

  /** The exact expectation the run converges to, with the conditions and, when there is
   *  no value, the reason. Closed form where one exists, quadrature otherwise. A custom f
   *  gets a value only if its tails fit a quadrature window and two different grids agree
   *  (a pole inside the support, as in tan x, makes them disagree wildly). */
  function exactInfo(op, distKey, nu, fnKey, custom) {
    var c = conditions(op, distKey, nu, fnKey, custom), out = { value: null, cond: c, reason: c.note };
    if (!c.mean) return out;
    if (fnKey === "custom") {
      var D = dist(distKey, nu), F = testFn(fnKey, custom);
      if (c.L === null) {
        out.reason = "The operator applied to f, times the density, decays too slowly for the quadrature to reach the end of its tails, so no exact value is shown.";
        return out;
      }
      var q1 = quadrature(op, D, F, c.L, 1), q2 = quadrature(op, D, F, c.L, 1.137);
      if (!isFinite(q1) || !isFinite(q2) || Math.abs(q1 - q2) > 1e-4 * (1 + Math.abs(q1))) {
        out.reason = "f or its derivative blows up inside the support (as tan x or 1/(x − 1) do), so no exact value can be computed, and the expectation may not exist.";
        return out;
      }
      out.value = q1;
      return out;
    }
    out.value = exactBuiltIn(op, distKey, nu, fnKey, c);
    if (out.value === null && !out.reason) out.reason = "No exact value could be computed for this f.";
    return out;
  }
  function exact(op, distKey, nu, fnKey, custom) { return exactInfo(op, distKey, nu, fnKey, custom).value; }
  function exactBuiltIn(op, distKey, nu, fnKey, c) {
    if (fnKey === "cube") {
      if (op === "gauss") {
        // E[3X^2 - X^4] = 3 - E X^4, with E X^4 the fourth moment of the standardised law
        var m4 = { gauss: 3, laplace: 6, unif: 9 / 5, t: 3 + 6 / (nu - 4) }[distKey];
        return 3 - m4;
      }
      return distKey === "unif" ? c.boundary : 0;
    }
    if (op === "langevin" && distKey !== "unif" && fnKey !== "custom") return 0;
    if (op === "langevin" && distKey === "unif" && fnKey !== "custom") return c.boundary;
    if (op === "gauss" && distKey === "gauss" && fnKey !== "custom") return 0;
    var q = quadrature(op, dist(distKey, nu), testFn(fnKey));
    return isFinite(q) ? q : null;
  }

  /** Log-spaced sample sizes from 10 to N at which the running mean is recorded. */
  function checkpoints(N) {
    var out = [], K = 120;
    for (var k = 0; k <= K; k++) {
      var n = Math.round(Math.pow(10, 1 + (Math.log10(N) - 1) * k / K));
      if (!out.length || n > out[out.length - 1]) out.push(n);
    }
    return out;
  }

  /** Seeded Monte Carlo: the running mean of (op f)(X) and its standard error. */
  function run(o) {
    var r = rng(o.seed), nrm = normalGen(r), D = dist(o.dist, o.nu), F = testFn(o.fn, o.custom);
    var N = o.N || 100000, cps = checkpoints(N), next = 0, mean = 0, m2 = 0, bad = 0, out = [];
    for (var i = 1; i <= N; i++) {
      var v = opValue(o.op, D, F, D.sample(r, nrm));
      if (!isFinite(v)) { bad++; v = 0; }
      var d = v - mean;
      mean += d / i;
      m2 += d * (v - mean);
      if (i === cps[next]) { out.push({ n: i, mean: mean, se: Math.sqrt(m2 / (i - 1) / i) }); next++; }
    }
    return { path: out, final: out[out.length - 1], nonFinite: bad };
  }

  return { rng: rng, normalGen: normalGen, lgamma: lgamma, dist: dist, parse: parse,
    testFn: testFn, opValue: opValue, quadrature: quadrature, tailCheck: tailCheck, conditions: conditions,
    exact: exact, exactInfo: exactInfo, checkpoints: checkpoints, run: run, SQ3: SQ3 };
})();

if (typeof module !== "undefined") module.exports = SteinLemmaModel;
