/* Part 2 widget: how many five-variable regressions are about as good as the best? */
"use strict";

var RashomonModel = (function () {
  // Pure functions of the tolerance; widgets.js and src/check_model.js both use them.
  function within(D, tol) {
    return D.subsets.filter(function (s) { return s.r <= 1 + tol + 1e-12; });
  }
  function inclusion(D, rows) {
    var c = D.names.map(function () { return 0; });
    rows.forEach(function (s) { s.v.forEach(function (j) { c[j] += 1; }); });
    return c.map(function (x) { return rows.length ? x / rows.length : 0; });
  }
  // Up to three subsets, best first, each sharing as few variables as possible
  // with those already picked (ties broken by RSS).
  function stories(rows) {
    var picked = [];
    for (var k = 0; k < 3 && k < rows.length; k++) {
      var best = null, bestOverlap = 99;
      rows.forEach(function (s) {
        if (picked.indexOf(s) >= 0) return;
        var ov = 0;
        picked.forEach(function (p) {
          s.v.forEach(function (j) { if (p.v.indexOf(j) >= 0) ov += 1; });
        });
        if (ov < bestOverlap) { bestOverlap = ov; best = s; }
      });
      picked.push(best);
    }
    return picked;
  }
  var api = { within: within, inclusion: inclusion, stories: stories };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();

if (typeof WK !== "undefined") WK.mount("widget-subsets", function (root, WK) {
  var D = WK.data("subsets"), M = RashomonModel;
  if (!D) throw new Error("widget data missing");
  var tol = 0.01;
  var f = WK.frame({
    title: "Five-variable regressions within a tolerance of the best",
    badge: "1976 LA ozone",
    note: "All " + D.n_subsets.toLocaleString("en") + " five-variable least-squares fits " +
      "of next-day ozone on 30 candidate predictors, trained January–August 1976. Left: the " +
      "spread of training RSS as a multiple of the best; the shaded part is within the " +
      "tolerance. Right: the share of those qualifying models that use each variable. " +
      "Lag 0 is day t, lag 1 the day before. Test MSE is on September–December."
  });
  var s = WK.slider({ label: "Tolerance above the lowest training RSS", min: 0.005, max: 0.10,
    step: 0.005, value: tol, fmt: function (v) { return WK.fmt.pct(v, 1); },
    oninput: function (v) { tol = v; draw(); } });
  f.controls.appendChild(s.root);
  var tN = WK.stat({ label: "models within the tolerance", value: "" });
  var tV = WK.stat({ label: "different variables they use", value: "" });
  var tT = WK.stat({ label: "their test MSE, lowest to highest", value: "" });
  [tN, tV, tT].forEach(function (t) { f.stats.appendChild(t.root); });
  var W = 640, H = 330, svg = WK.svg(W, H, { "aria-label":
    "Histogram of subset RSS ratios and bar chart of variable inclusion" });
  f.body.appendChild(svg);
  var list = WK.el("div", { class: "widget-note" });
  f.body.appendChild(list);
  root.appendChild(f.root);

  function label(name) { return name.replace("temp_sandburg", "temp").replace("_lag", " lag "); }

  function draw() {
    var rows = M.within(D, tol), inc = M.inclusion(D, rows);
    tN.set(rows.length.toLocaleString("en"));
    tV.set(String(inc.filter(function (x) { return x > 0; }).length) + " of " + D.names.length);
    var t = rows.map(function (r) { return r.t; });
    tT.set(WK.fmt.num(Math.min.apply(null, t), 1) + " – " + WK.fmt.num(Math.max.apply(null, t), 1));
    WK.clear(svg);
    // Left: histogram of the RSS ratio, 1.00 to 1.50.
    var L0 = 50, L1 = 290, T0 = 14, B0 = H - 40, nb = 50;
    var counts = D.hist.slice(0, nb), cmax = Math.max.apply(null, counts);
    var x = WK.lin([1, 1.5], [L0, L1]);
    var y = WK.lin([0, Math.log10(cmax + 1)], [B0, T0]);
    counts.forEach(function (c, i) {
      var lo = D.hist_edges[i], hi = D.hist_edges[i + 1];
      if (!c) return;
      svg.appendChild(WK.h("rect", { x: x(lo), width: Math.max(1, x(hi) - x(lo) - 0.5),
        y: y(Math.log10(c + 1)), height: B0 - y(Math.log10(c + 1)),
        fill: hi <= 1 + tol + 1e-9 ? "c1" : "rule" }));
    });
    svg.appendChild(WK.h("line", { x1: x(1 + tol), x2: x(1 + tol), y1: T0, y2: B0,
      stroke: "accent", "stroke-width": 1.5 }));
    [1, 1.1, 1.2, 1.3, 1.4, 1.5].forEach(function (v) {
      svg.appendChild(WK.h("text", { x: x(v), y: B0 + 14, "text-anchor": "middle",
        "font-size": 11, fill: "muted", text: v.toFixed(1) }));
    });
    [1, 10, 100, 1000].forEach(function (v) {
      if (v > cmax) return;
      svg.appendChild(WK.h("text", { x: L0 - 6, y: y(Math.log10(v + 1)) + 4, "text-anchor": "end",
        "font-size": 11, fill: "muted", text: String(v) }));
    });
    svg.appendChild(WK.h("text", { x: (L0 + L1) / 2, y: H - 10, "text-anchor": "middle",
      "font-size": 12, fill: "ink", text: "training RSS ÷ best RSS" }));
    svg.appendChild(WK.h("text", { x: 14, y: (T0 + B0) / 2, "text-anchor": "middle",
      "font-size": 12, fill: "ink", transform: "rotate(-90 14 " + (T0 + B0) / 2 + ")",
      text: "subsets per bin (log scale)" }));
    // Right: inclusion share, the 12 most used variables at this tolerance.
    var order = inc.map(function (v, j) { return [v, j]; })
      .sort(function (a, b) { return b[0] - a[0]; }).slice(0, 12);
    var R0 = 430, R1 = W - 14, rowH = (B0 - T0) / 12;
    var bx = WK.lin([0, 1], [R0, R1 - 34]);
    order.forEach(function (o, i) {
      var yy = T0 + i * rowH;
      svg.appendChild(WK.h("rect", { x: R0, y: yy + 2, width: Math.max(0, bx(o[0]) - R0),
        height: rowH - 5, fill: o[0] > 0 ? "c2" : "rule" }));
      svg.appendChild(WK.h("text", { x: R0 - 6, y: yy + rowH / 2 + 3, "text-anchor": "end",
        "font-size": 10.5, fill: "ink", text: label(D.names[o[1]]) }));
      svg.appendChild(WK.h("text", { x: bx(o[0]) + 4, y: yy + rowH / 2 + 3,
        "font-size": 10, fill: "muted", text: o[0] > 0 ? WK.fmt.pct(o[0]) : "" }));
    });
    svg.appendChild(WK.h("text", { x: (R0 + R1) / 2, y: H - 10, "text-anchor": "middle",
      "font-size": 12, fill: "ink", text: "share of qualifying models using it" }));
    // Three qualifying models that share as little as possible.
    WK.clear(list);
    list.appendChild(WK.el("strong", { text: "Three qualifying models that overlap least: " }));
    M.stories(rows).forEach(function (r, i) {
      var terms = r.v.map(function (j, m) {
        return (r.b[m] >= 0 ? "+ " : "− ") + Number(Math.abs(r.b[m]).toPrecision(2)) + "·" + label(D.names[j]);
      }).join(" ");
      list.appendChild(WK.el("div", { text: (i + 1) + ". RSS ×" + r.r.toFixed(3) +
        ", test MSE " + r.t.toFixed(1) + ":  " + terms }));
    });
  }
  draw();
});
