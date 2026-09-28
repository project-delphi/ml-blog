/* Widget for "Calibrated Decisions, Part 3: Recalibration and Decisions".
 *
 * widget-routing: refund requests with P(warranted | x) = p*(x) = sigmoid(2x),
 * X ~ N(0, 1). A model reports p_hat = sigmoid(alpha * 2x). Three actions:
 * decline (costs c_FN if the refund was warranted), send to review (costs
 * c_H), approve (costs c_FP if it was not). The Bayes thresholds follow from
 * the costs; they are applied to the raw p_hat and to exactly recalibrated
 * probabilities, which here equal p*. Expected cost is computed against p* by
 * quadrature over x on a fixed grid, so the numbers carry no sampling noise and
 * match the post's Python cell for the same settings.
 *
 * The model lives in `CAL3`, separate from the drawing, and is exported under
 * Node so the prose's claims can be checked over the whole control range.
 * Built on widget-kit (WK); colours are theme tokens.
 */
"use strict";

var CAL3 = (function () {
  var G = 4001, X = new Float64Array(G), W = new Float64Array(G), total = 0;
  for (var i = 0; i < G; i++) {
    X[i] = -8 + 16 * i / (G - 1);
    W[i] = Math.exp(-0.5 * X[i] * X[i]);
    total += W[i];
  }
  for (i = 0; i < G; i++) W[i] /= total;

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // Bayes thresholds for probability p: decline below lo, approve above hi,
  // review between. When review never pays, lo = hi = cFP / (cFP + cFN).
  function thresholds(cFP, cFN, cH) {
    var lo = cH / cFN, hi = 1 - cH / cFP;
    if (lo < hi) return { lo: lo, hi: hi, review: true };
    var p0 = cFP / (cFP + cFN);
    return { lo: p0, hi: p0, review: false };
  }

  function act(p, t) {
    if (t.review) return p < t.lo ? 0 : p > t.hi ? 2 : 1;
    return p > t.lo ? 2 : 0;
  }

  function evaluate(alpha, cFP, cFN, cH) {
    var t = thresholds(cFP, cFN, cH);
    var out = { thresholds: t, raw: { cost: 0, auto: 0, wrong: 0 }, cal: { cost: 0, auto: 0, wrong: 0 } };
    for (var i = 0; i < G; i++) {
      var ps = sigmoid(2 * X[i]);
      var pr = sigmoid(alpha * 2 * X[i]);
      [["raw", pr], ["cal", ps]].forEach(function (d) {
        var a = act(d[1], t), o = out[d[0]];
        var c = a === 0 ? ps * cFN : a === 1 ? cH : (1 - ps) * cFP;
        o.cost += W[i] * c;
        if (a !== 1) {
          o.auto += W[i];
          o.wrong += W[i] * (a === 0 ? ps : 1 - ps);
        }
      });
    }
    ["raw", "cal"].forEach(function (k) {
      out[k].cost *= 1000;
      out[k].wrong *= 1000;
    });
    out.regret = out.raw.cost - out.cal.cost;
    return out;
  }

  // Density of a forecast over [0, 1] in `bins` bins, from the quadrature grid.
  function density(alpha, bins) {
    var h = new Float64Array(bins);
    for (var i = 0; i < G; i++) {
      var p = sigmoid(alpha * 2 * X[i]);
      h[Math.min(bins - 1, Math.floor(p * bins))] += W[i];
    }
    return h;
  }

  return { thresholds: thresholds, evaluate: evaluate, density: density };
})();

if (typeof module === "object" && module.exports) module.exports = CAL3;

if (typeof WK !== "undefined") {
  WK.mount("widget-routing", function (root, WK) {
    var state = { alpha: 2.5, cFP: 10, cFN: 40, cH: 3 };
    var f = WK.frame({
      title: "Routing refunds on raw and recalibrated probabilities",
      note: "True P(warranted) = sigmoid(2x), x ~ N(0, 1); the model reports sigmoid(α·2x). " +
        "The shaded bands are the Bayes actions for a true probability. Costs are expectations " +
        "over x computed by quadrature, per 1,000 requests. \"Recalibrated\" means perfectly " +
        "recalibrated, which here recovers the true probability."
    });
    var sliders = [
      ["alpha", "Distortion α (1 = calibrated)", 0.25, 4, 0.05, WK.fmt.num],
      ["cFP", "Cost of a wrong approval", 1, 50, 1, WK.fmt.int],
      ["cFN", "Cost of a wrong decline", 1, 100, 1, WK.fmt.int],
      ["cH", "Cost of a review", 0.5, 30, 0.5, function (v) { return Number(v).toFixed(1); }]
    ].map(function (s) {
      var sl = WK.slider({ label: s[1], min: s[2], max: s[3], step: s[4], value: state[s[0]], fmt: s[5],
        oninput: function (v) { state[s[0]] = v; draw(); } });
      f.controls.appendChild(sl.root);
      return sl;
    });

    var W = 520, H = 300, X0 = 40, X1 = 500, Y0 = 34, Y1 = 230;
    var svg = WK.svg(W, H, { "aria-label": "Action regions over the distribution of forecasts" });
    f.body.appendChild(svg);
    var st = {};
    [["rawCost", "Cost per 1,000, raw"], ["calCost", "Cost per 1,000, recalibrated"],
     ["regret", "Extra cost of acting on raw"], ["rawAuto", "Automated, raw"],
     ["calAuto", "Automated, recalibrated"], ["rawWrong", "Wrong automatic decisions per 1,000, raw"]]
      .forEach(function (d) {
        st[d[0]] = WK.stat({ label: d[1], value: "" });
        f.stats.appendChild(st[d[0]].root);
      });
    root.appendChild(f.root);
    var sx = WK.lin([0, 1], [X0, X1]);

    function draw() {
      var r = CAL3.evaluate(state.alpha, state.cFP, state.cFN, state.cH), t = r.thresholds;
      WK.clear(svg);
      var bands = t.review
        ? [[0, t.lo, "c4", "decline"], [t.lo, t.hi, "c3", "review"], [t.hi, 1, "c6", "approve"]]
        : [[0, t.lo, "c4", "decline"], [t.lo, 1, "c6", "approve"]];
      bands.forEach(function (b) {
        svg.appendChild(WK.h("rect", { x: sx(b[0]), y: Y0, width: Math.max(0, sx(b[1]) - sx(b[0])),
          height: Y1 - Y0, fill: b[2], "fill-opacity": 0.13 }));
        if (sx(b[1]) - sx(b[0]) > 1) {
          svg.appendChild(WK.h("text", { x: Math.min(X1 - 26, Math.max(X0 + 24, (sx(b[0]) + sx(b[1])) / 2)), y: Y0 - 8, "font-size": 13,
            "text-anchor": "middle", "font-weight": 600, fill: b[2] }, b[3]));
        }
      });
      var BINS = 50, raw = CAL3.density(state.alpha, BINS), cal = CAL3.density(1, BINS);
      var top = Math.max(Math.max.apply(null, raw), Math.max.apply(null, cal));
      var sy = WK.lin([0, top * 1.05], [Y1, Y0 + 6]);
      [[cal, "c2", "recalibrated"], [raw, "c1", "raw forecast"]].forEach(function (d, k) {
        var path = "M" + sx(0) + "," + Y1;
        for (var i = 0; i < BINS; i++) {
          path += "L" + sx(i / BINS).toFixed(1) + "," + sy(d[0][i]).toFixed(1) +
                  "L" + sx((i + 1) / BINS).toFixed(1) + "," + sy(d[0][i]).toFixed(1);
        }
        path += "L" + sx(1) + "," + Y1;
        svg.appendChild(WK.h("path", { d: path, fill: "none", stroke: d[1], "stroke-width": k ? 2.2 : 1.6,
          "stroke-dasharray": k ? null : "5 3" }));
      });
      [t.lo, t.hi].forEach(function (v, k) {
        if (k === 1 && !t.review) return;
        svg.appendChild(WK.h("line", { x1: sx(v), x2: sx(v), y1: Y0, y2: Y1, stroke: "ink", "stroke-width": 1 }));
        svg.appendChild(WK.h("text", { x: sx(v) + (k ? -4 : 4), y: Y1 - 6, "font-size": 12,
          "text-anchor": k ? "end" : "start", fill: "ink" }, v.toFixed(3)));
      });
      svg.appendChild(WK.h("line", { x1: X0, x2: X1, y1: Y1, y2: Y1, stroke: "rule" }));
      [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
        svg.appendChild(WK.h("text", { x: sx(v), y: Y1 + 16, "font-size": 12, "text-anchor": "middle", fill: "muted" }, String(v)));
      });
      svg.appendChild(WK.h("text", { x: (X0 + X1) / 2, y: Y1 + 34, "font-size": 12, "text-anchor": "middle", fill: "muted" },
        "probability the refund is warranted"));
      svg.appendChild(WK.h("line", { x1: X0 + 6, x2: X0 + 26, y1: H - 14, y2: H - 14, stroke: "c1", "stroke-width": 2.2 }));
      svg.appendChild(WK.h("text", { x: X0 + 30, y: H - 10, "font-size": 12, fill: "ink" }, "raw forecasts"));
      svg.appendChild(WK.h("line", { x1: X0 + 150, x2: X0 + 170, y1: H - 14, y2: H - 14, stroke: "c2", "stroke-width": 1.6, "stroke-dasharray": "5 3" }));
      svg.appendChild(WK.h("text", { x: X0 + 174, y: H - 10, "font-size": 12, fill: "ink" }, "recalibrated (true) probabilities"));
      st.rawCost.set(r.raw.cost.toFixed(0));
      st.calCost.set(r.cal.cost.toFixed(0));
      st.regret.set("+" + Math.max(0, r.regret).toFixed(0));
      st.rawAuto.set(WK.fmt.pct(r.raw.auto));
      st.calAuto.set(WK.fmt.pct(r.cal.auto));
      st.rawWrong.set(r.raw.wrong.toFixed(0));
    }
    draw();
  });
}
