/* Widgets for "Calibrated Decisions, Part 2: Calibration and Proper Scoring Rules".
 *
 * 1. widget-scoring-rules: a forecaster believes P(Y = 1) = q and reports p.
 *    Expected loss under four rules, as a function of p, in closed form:
 *    Brier, log and spherical (strictly proper: minimised at p = q) and
 *    absolute error (improper: minimised at 0 or 1).
 * 2. widget-calibration-lab: the post's generating process, X ~ N(0, 1),
 *    p*(x) = sigmoid(2x), Y ~ Bernoulli(p*), and a forecaster
 *    p_hat = sigmoid(alpha * 2x). Reliability bins, a LOESS curve (local
 *    linear, tricube weights, no robustness iterations) and five metrics.
 *    Draws come from a seeded PRNG, so a given seed gives the same sample on
 *    every visit; "Draw a new sample" moves to the next seed.
 *
 * The models live in `CAL2`, separate from the drawing, and are exported under
 * Node so the prose's claims can be checked over the whole control range.
 * Built on widget-kit (WK); colours are theme tokens.
 */
"use strict";

var CAL2 = (function () {
  var EPS = 1e-3;

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
  function clip(p) { return Math.min(1 - EPS, Math.max(EPS, p)); }

  // ------------------------------------------------------- scoring rules
  var RULES = [
    { key: "brier", name: "Brier", proper: true,
      loss: function (p, q) { return q * (1 - p) * (1 - p) + (1 - q) * p * p; } },
    { key: "log", name: "Log", proper: true,
      loss: function (p, q) { return -q * Math.log(p) - (1 - q) * Math.log(1 - p); } },
    { key: "spherical", name: "Spherical", proper: true,
      loss: function (p, q) { return -(q * p + (1 - q) * (1 - p)) / Math.sqrt(p * p + (1 - p) * (1 - p)); } },
    { key: "absolute", name: "Absolute", proper: false,
      loss: function (p, q) { return q * (1 - p) + (1 - q) * p; } }
  ];

  // Best report on a grid of step 0.001 inside [0.001, 0.999].
  function bestReport(rule, q) {
    var best = 0.001, bestLoss = Infinity;
    for (var i = 1; i <= 999; i++) {
      var p = i / 1000, l = rule.loss(p, q);
      if (l < bestLoss - 1e-12) { bestLoss = l; best = p; }
    }
    return best;
  }

  // ------------------------------------------------------ calibration lab
  var N_MAX = 20000;

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // x ~ N(0, 1) by Box-Muller, y ~ Bernoulli(sigmoid(2x)).
  function sample(seed) {
    var rng = mulberry32(seed), x = new Float64Array(N_MAX), y = new Uint8Array(N_MAX);
    for (var i = 0; i < N_MAX; i++) {
      var u = 1 - rng(), v = rng();
      x[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      y[i] = rng() < sigmoid(2 * x[i]) ? 1 : 0;
    }
    return { x: x, y: y };
  }

  // Indices of the first n cases sorted by x. Forecasts are increasing in x
  // for every alpha > 0, so this order serves every alpha.
  function orderByX(data, n) {
    var idx = [];
    for (var i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return data.x[a] - data.x[b]; });
    return idx;
  }

  function loess(ps, ys, span, grid) {
    // ps sorted ascending; k nearest neighbours of each grid point.
    var n = ps.length, k = Math.min(n, Math.max(10, Math.ceil(span * n)));
    return grid.map(function (g) {
      var lo = 0, hi = n;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (ps[mid] < g) lo = mid + 1; else hi = mid; }
      var L = lo - 1, Rr = lo, taken = 0;
      while (taken < k) {
        if (L < 0) Rr++;
        else if (Rr >= n) L--;
        else if (g - ps[L] <= ps[Rr] - g) L--;
        else Rr++;
        taken++;
      }
      var a = L + 1, b = Rr - 1;
      var h = Math.max(g - ps[a], ps[b] - g) * 1.000001 || 1e-9;
      var sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (var i = a; i <= b; i++) {
        var u = Math.abs(ps[i] - g) / h, w = Math.pow(1 - u * u * u, 3);
        sw += w; sx += w * ps[i]; sy += w * ys[i];
        sxx += w * ps[i] * ps[i]; sxy += w * ps[i] * ys[i];
      }
      var mx = sx / sw, my = sy / sw, vx = sxx / sw - mx * mx;
      if (vx < 1e-12) return my;
      return my + (sxy / sw - mx * my) / vx * (g - mx);
    });
  }

  function evaluate(data, order, alpha, bins, span) {
    var n = order.length, ps = new Float64Array(n), ys = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      ps[i] = sigmoid(alpha * 2 * data.x[order[i]]);
      ys[i] = data.y[order[i]];
    }
    var correct = 0, br = 0, ll = 0, pos = 0, rankSum = 0;
    var bSum = [], bY = [], bN = [];
    for (var b = 0; b < bins; b++) { bSum.push(0); bY.push(0); bN.push(0); }
    for (i = 0; i < n; i++) {
      var p = ps[i], yy = ys[i], c = clip(p);
      correct += (p > 0.5) === (yy === 1) ? 1 : 0;
      br += (p - yy) * (p - yy);
      ll += -(yy * Math.log(c) + (1 - yy) * Math.log(1 - c));
      if (yy === 1) { pos++; rankSum += i + 1; }
      var j = Math.min(bins - 1, Math.floor(p * bins));
      bSum[j] += p; bY[j] += yy; bN[j]++;
    }
    var ece = 0, rel = [];
    for (b = 0; b < bins; b++) {
      if (!bN[b]) continue;
      ece += Math.abs(bY[b] / bN[b] - bSum[b] / bN[b]) * bN[b];
      rel.push({ p: bSum[b] / bN[b], f: bY[b] / bN[b], n: bN[b] });
    }
    var neg = n - pos;
    var grid = [], lo = ps[0], hi = ps[n - 1];
    for (i = 0; i < 60; i++) grid.push(lo + (hi - lo) * i / 59);
    return {
      n: n,
      accuracy: correct / n,
      auc: pos && neg ? (rankSum - pos * (pos + 1) / 2) / (pos * neg) : NaN,
      brier: br / n,
      logloss: ll / n,
      ece: ece / n,
      bins: rel,
      ps: ps,
      loess: { x: grid, y: loess(ps, ys, span, grid) }
    };
  }

  return { RULES: RULES, bestReport: bestReport, sample: sample, orderByX: orderByX,
           evaluate: evaluate, loess: loess, sigmoid: sigmoid };
})();

if (typeof module === "object" && module.exports) module.exports = CAL2;

if (typeof WK !== "undefined") {
  // ------------------------------------------------ widget 1: scoring rules
  WK.mount("widget-scoring-rules", function (root, WK) {
    var state = { q: 0.3, p: 0.6 };
    var f = WK.frame({
      title: "Which report does each rule reward?",
      note: "Expected loss when the outcome is Yes with probability q and the forecaster reports p. " +
        "Each panel has its own vertical scale. The dashed line is the honest report p = q; " +
        "the diamond marks the report with the lowest expected loss."
    });
    var qs = WK.slider({ label: "True probability q", min: 0.02, max: 0.98, step: 0.01,
      value: state.q, fmt: WK.fmt.num, oninput: function (v) { state.q = v; draw(); } });
    var pr = WK.slider({ label: "Reported probability p", min: 0.02, max: 0.98, step: 0.01,
      value: state.p, fmt: WK.fmt.num, oninput: function (v) { state.p = v; draw(); } });
    f.controls.appendChild(qs.root);
    f.controls.appendChild(pr.root);

    var W = 520, H = 400, svg = WK.svg(W, H, { "aria-label": "Expected loss against reported probability for four scoring rules" });
    f.body.appendChild(svg);
    var stats = CAL2.RULES.map(function (r) {
      var s = WK.stat({ label: r.name + ": extra loss from reporting p", value: "" });
      f.stats.appendChild(s.root);
      return s;
    });
    root.appendChild(f.root);

    var PW = 205, PH = 125, pads = [[30, 40], [290, 40], [30, 240], [290, 240]];

    function draw() {
      WK.clear(svg);
      CAL2.RULES.forEach(function (rule, k) {
        var ox = pads[k][0], oy = pads[k][1], lo = Infinity, hi = -Infinity, pts = [];
        for (var i = 2; i <= 98; i++) {
          var pp = i / 100, l = rule.loss(pp, state.q);
          pts.push([pp, l]); lo = Math.min(lo, l); hi = Math.max(hi, l);
        }
        var top = hi - lo > 1e-9 ? hi + (hi - lo) * 0.05 : lo + 1;
        var sx = WK.lin([0, 1], [ox, ox + PW]), sy = WK.lin([hi - lo > 1e-9 ? lo : lo - 1, top], [oy + PH, oy]);
        var best = CAL2.bestReport(rule, state.q), flat = hi - lo < 1e-9;
        svg.appendChild(WK.h("text", { x: ox, y: oy - 12, "font-size": 14, "font-weight": 600,
          fill: rule.proper ? "ink" : "c3" }, rule.name + (rule.proper ? " (proper)" : " (improper)")));
        svg.appendChild(WK.h("text", { x: ox + PW, y: oy - 12, "font-size": 12, "text-anchor": "end",
          fill: "muted" }, flat ? "all tie" : "best " + best.toFixed(2)));
        svg.appendChild(WK.h("line", { x1: ox, x2: ox + PW, y1: oy + PH, y2: oy + PH, stroke: "rule" }));
        [0, 0.5, 1].forEach(function (t) {
          svg.appendChild(WK.h("text", { x: sx(t), y: oy + PH + 15, "font-size": 12,
            "text-anchor": "middle", fill: "muted" }, String(t)));
        });
        svg.appendChild(WK.h("line", { x1: sx(state.q), x2: sx(state.q), y1: oy, y2: oy + PH,
          stroke: "muted", "stroke-dasharray": "4 3" }));
        svg.appendChild(WK.h("path", { d: "M" + pts.map(function (d) { return sx(d[0]).toFixed(1) + "," + sy(d[1]).toFixed(1); }).join("L"),
          fill: "none", stroke: rule.proper ? "c1" : "c3", "stroke-width": 2 }));
        var bx = sx(Math.min(0.98, Math.max(0.02, best))), by = sy(rule.loss(Math.min(0.98, Math.max(0.02, best)), state.q));
        if (!flat) {
          svg.appendChild(WK.h("path", { d: "M" + bx + "," + (by - 6) + "L" + (bx + 6) + "," + by + "L" + bx + "," + (by + 6) + "L" + (bx - 6) + "," + by + "Z",
            fill: "c2" }));
        }
        svg.appendChild(WK.h("circle", { cx: sx(state.p), cy: sy(rule.loss(state.p, state.q)), r: 5,
          fill: "accent", stroke: "surface", "stroke-width": 1.5 }));
        var extra = rule.loss(state.p, state.q) - rule.loss(state.q, state.q);
        stats[k].set((extra >= 0 ? "+" : "−") + Math.abs(extra).toFixed(3));
      });
      svg.appendChild(WK.h("text", { x: W / 2, y: H - 4, "font-size": 12, "text-anchor": "middle",
        fill: "muted" }, "reported probability p (dot = current report)"));
    }
    draw();
  });

  // ---------------------------------------------- widget 2: calibration lab
  WK.mount("widget-calibration-lab", function (root, WK) {
    var state = { alpha: 2.5, n: 1000, bins: 15, span: 0.3, seed: 1 };
    var data = CAL2.sample(state.seed), order = CAL2.orderByX(data, state.n);

    var f = WK.frame({
      title: "Calibration lab",
      note: "X ~ N(0, 1), true probability sigmoid(2x), forecast sigmoid(α·2x). Dots are equal-width " +
        "bins (larger dots hold more cases; bars are 95% Wilson intervals); the curve is LOESS with no robustness " +
        "iterations, drawn up to 0.05 beyond [0, 1] so overshoot stays visible. Log loss clips forecasts to [0.001, 0.999]."
    });
    var aS = WK.slider({ label: "Distortion α (1 = calibrated)", min: -2, max: 2, step: 0.01,
      value: Math.log2(state.alpha), fmt: function (v) { return Math.pow(2, v).toFixed(2); },
      oninput: function (v) { state.alpha = Math.pow(2, v); draw(); } });
    var nT = WK.toggle({ label: "Sample size n", value: state.n,
      options: [200, 1000, 5000, 20000].map(function (v) { return { value: v, label: String(v) }; }),
      onchange: function (v) { state.n = v; order = CAL2.orderByX(data, v); draw(); } });
    var bS = WK.slider({ label: "Bins", min: 5, max: 30, step: 1, value: state.bins, fmt: WK.fmt.int,
      oninput: function (v) { state.bins = v; draw(); } });
    var sS = WK.slider({ label: "LOESS span", min: 0.1, max: 1, step: 0.05, value: state.span,
      fmt: WK.fmt.num, oninput: function (v) { state.span = v; draw(); } });
    var again = WK.el("button", { type: "button", class: "widget-toggle-btn", text: "Draw a new sample",
      onclick: function () {
        state.seed += 1; data = CAL2.sample(state.seed); order = CAL2.orderByX(data, state.n); draw();
      } });
    [aS.root, nT.root, bS.root, sS.root, WK.el("div", {}, again)].forEach(function (c) { f.controls.appendChild(c); });

    var W = 520, H = 440, X0 = 56, X1 = 500, Y0 = 20, Y1 = 330, HB = 390;
    var svg = WK.svg(W, H, { "aria-label": "Reliability diagram with LOESS curve and histogram of forecasts" });
    f.body.appendChild(svg);
    var st = {};
    [["accuracy", "Accuracy at 0.5"], ["auc", "AUC"], ["ece", "ECE"], ["brier", "Brier score"],
     ["logloss", "Log loss"]].forEach(function (d) {
      st[d[0]] = WK.stat({ label: d[1], value: "" });
      f.stats.appendChild(st[d[0]].root);
    });
    root.appendChild(f.root);

    var sx = WK.lin([0, 1], [X0, X1]), sy = WK.lin([0, 1], [Y1, Y0]);

    function draw() {
      var r = CAL2.evaluate(data, order, state.alpha, state.bins, state.span);
      WK.clear(svg);
      WK.ticks(0, 1, 5).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: sx(t), x2: sx(t), y1: Y0, y2: Y1, stroke: "rule", "stroke-width": 0.6 }));
        svg.appendChild(WK.h("line", { x1: X0, x2: X1, y1: sy(t), y2: sy(t), stroke: "rule", "stroke-width": 0.6 }));
        svg.appendChild(WK.h("text", { x: sx(t), y: Y1 + 16, "font-size": 12, "text-anchor": "middle", fill: "muted" }, t.toFixed(1)));
        svg.appendChild(WK.h("text", { x: X0 - 6, y: sy(t) + 4, "font-size": 12, "text-anchor": "end", fill: "muted" }, t.toFixed(1)));
      });
      svg.appendChild(WK.h("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), stroke: "muted", "stroke-dasharray": "5 4" }));
      svg.appendChild(WK.h("text", { x: X0 + 4, y: Y0 + 12, "font-size": 12, fill: "muted" }, "observed frequency of Y = 1"));
      // LOESS curve, drawn inside a slightly padded box.
      var d = r.loess.x.map(function (xv, i) {
        var yv = Math.max(-0.05, Math.min(1.05, r.loess.y[i]));
        return (i ? "L" : "M") + sx(xv).toFixed(1) + "," + sy(yv).toFixed(1);
      }).join("");
      svg.appendChild(WK.h("path", { d: d, fill: "none", stroke: "c2", "stroke-width": 2.5 }));
      var maxN = Math.max.apply(null, r.bins.map(function (b) { return b.n; }));
      r.bins.forEach(function (b) {
        // Wilson interval: stays open when a bin's frequency is 0 or 1.
        var z2 = 1.96 * 1.96 / b.n, centre = (b.f + z2 / 2) / (1 + z2);
        var half = 1.96 / (1 + z2) * Math.sqrt(b.f * (1 - b.f) / b.n + z2 / (4 * b.n));
        svg.appendChild(WK.h("line", { x1: sx(b.p), x2: sx(b.p), y1: sy(Math.max(0, centre - half)),
          y2: sy(Math.min(1, centre + half)), stroke: "c1", "stroke-width": 1.2 }));
        svg.appendChild(WK.h("circle", { cx: sx(b.p), cy: sy(b.f), r: 3 + 8 * Math.sqrt(b.n / maxN),
          fill: "c1", "fill-opacity": 0.8, stroke: "surface", "stroke-width": 1 }));
      });
      // Histogram of forecasts.
      var counts = [], k;
      for (k = 0; k < 25; k++) counts.push(0);
      for (k = 0; k < r.n; k++) counts[Math.min(24, Math.floor(r.ps[k] * 25))]++;
      var cmax = Math.max.apply(null, counts);
      counts.forEach(function (c, i) {
        var hgt = 42 * c / cmax;
        svg.appendChild(WK.h("rect", { x: sx(i / 25) + 1, y: HB - hgt, width: (X1 - X0) / 25 - 2, height: hgt, fill: "muted", "fill-opacity": 0.45 }));
      });
      svg.appendChild(WK.h("line", { x1: X0, x2: X1, y1: HB, y2: HB, stroke: "rule" }));
      svg.appendChild(WK.h("text", { x: (X0 + X1) / 2, y: HB + 18, "font-size": 12, "text-anchor": "middle", fill: "muted" },
        "forecast probability (bars: histogram of forecasts)"));
      // Legend.
      svg.appendChild(WK.h("circle", { cx: X1 - 150, cy: Y1 - 40, r: 5, fill: "c1" }));
      svg.appendChild(WK.h("text", { x: X1 - 140, y: Y1 - 36, "font-size": 12, fill: "ink" }, "binned frequency"));
      svg.appendChild(WK.h("line", { x1: X1 - 158, x2: X1 - 142, y1: Y1 - 18, y2: Y1 - 18, stroke: "c2", "stroke-width": 2.5 }));
      svg.appendChild(WK.h("text", { x: X1 - 140, y: Y1 - 14, "font-size": 12, fill: "ink" }, "LOESS"));
      st.accuracy.set(r.accuracy.toFixed(3));
      st.auc.set(r.auc.toFixed(3));
      st.ece.set(r.ece.toFixed(3));
      st.brier.set(r.brier.toFixed(3));
      st.logloss.set(r.logloss.toFixed(3));
    }
    draw();
  });
}
