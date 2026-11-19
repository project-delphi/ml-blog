/* Part 5 widget: a predictive slope against an interventional one. */
"use strict";

WK.mount("widget-confound", function (root, WK) {
  var M = ConfoundModel;
  var p = { a: 0.9, b: -1.6, tau: 0.5 }, seed = 1;
  var f = WK.frame({
    title: "Severity drives both the dose and the outcome",
    badge: "synthetic",
    note: "400 synthetic patients. Dose T and outcome Y are standardised; colour is the " +
      "severity tercile. The solid line is the least-squares fit of Y on T, the line a " +
      "purely predictive model learns. The dashed line has the causal slope τ: what " +
      "raising everyone's dose by one unit would do to Y."
  });
  function sl(label, key, min, max) {
    return WK.slider({ label: label, min: min, max: max, step: 0.05, value: p[key],
      fmt: function (v) { return WK.fmt.num(v); },
      oninput: function (v) { p[key] = v; draw(); } });
  }
  [sl("Severity → dose (a)", "a", 0, 0.95),
   sl("Severity → outcome (b)", "b", -2, 0),
   sl("Dose → outcome, causal (τ)", "tau", -0.5, 1)].forEach(function (s) {
    f.controls.appendChild(s.root);
  });
  f.controls.appendChild(WK.el("button", { type: "button", class: "widget-toggle-btn",
    text: "New patients", onclick: function () { seed += 1; draw(); } }));
  var tNaive = WK.stat({ label: "Slope of Y on T (predictive)" , value: "" });
  var tR2 = WK.stat({ label: "R² of that prediction", value: "" });
  var tAdj = WK.stat({ label: "Slope of Y on T, adjusting for severity", value: "" });
  var tTau = WK.stat({ label: "Causal effect τ", value: "" });
  [tNaive, tR2, tAdj, tTau].forEach(function (s) { f.stats.appendChild(s.root); });
  var W = 600, H = 340, L = 48, R = 16, T = 14, B = 40;
  var svg = WK.svg(W, H, { "aria-label": "Scatter of outcome against dose with two lines" });
  f.body.appendChild(svg);
  root.appendChild(f.root);

  function tercile(rows) {
    var s = rows.map(function (r) { return r.s; }).sort(function (x, y) { return x - y; });
    return [s[Math.floor(s.length / 3)], s[Math.floor(2 * s.length / 3)]];
  }

  function draw() {
    var st = M.stats(p, seed), rows = st.rows;
    tNaive.set(WK.fmt.num(st.naiveSlope));
    tR2.set(WK.fmt.num(st.naiveR2));
    tAdj.set(WK.fmt.num(st.adjSlope));
    tTau.set(WK.fmt.num(p.tau));
    var ymax = Math.max.apply(null, rows.map(function (r) { return Math.abs(r.y); }));
    ymax = Math.ceil(ymax + 0.5);
    var x = WK.lin([-4, 4], [L, W - R]), y = WK.lin([-ymax, ymax], [H - B, T]);
    WK.clear(svg);
    WK.ticks(-4, 4, 8).forEach(function (v) {
      svg.appendChild(WK.h("line", { x1: x(v), x2: x(v), y1: T, y2: H - B, stroke: "rule" }));
      svg.appendChild(WK.h("text", { x: x(v), y: H - B + 14, "text-anchor": "middle",
        "font-size": 11, fill: "muted", text: String(v) }));
    });
    WK.ticks(-ymax, ymax, 6).forEach(function (v) {
      svg.appendChild(WK.h("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "rule" }));
      svg.appendChild(WK.h("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end",
        "font-size": 11, fill: "muted", text: String(v) }));
    });
    svg.appendChild(WK.h("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle",
      "font-size": 12, fill: "ink", text: "dose T" }));
    svg.appendChild(WK.h("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle",
      "font-size": 12, fill: "ink", transform: "rotate(-90 12 " + (T + H - B) / 2 + ")",
      text: "outcome Y" }));
    var cut = tercile(rows), tok = ["c2", "c3", "c4"];
    rows.forEach(function (r) {
      if (Math.abs(r.t) > 4) return;
      var k = r.s < cut[0] ? 0 : (r.s < cut[1] ? 1 : 2);
      svg.appendChild(WK.h("circle", { cx: x(r.t), cy: y(r.y), r: 2.4, fill: tok[k],
        "fill-opacity": 0.7 }));
    });
    // Clip each line to the plot box by trimming x where y leaves [-ymax, ymax].
    function clipped(slope, icpt, token, dash) {
      var xs = [-4, 4];
      if (Math.abs(slope) > 1e-9) {
        var a = (-ymax - icpt) / slope, b = (ymax - icpt) / slope;
        xs = [Math.max(-4, Math.min(a, b)), Math.min(4, Math.max(a, b))];
      }
      if (xs[0] >= xs[1]) return;
      svg.appendChild(WK.h("line", { x1: x(xs[0]), y1: y(icpt + slope * xs[0]),
        x2: x(xs[1]), y2: y(icpt + slope * xs[1]), stroke: token, "stroke-width": 2.5,
        "stroke-dasharray": dash || null }));
    }
    clipped(st.naiveSlope, st.naiveIntercept, "ink");
    clipped(p.tau, st.naiveIntercept, "accent", "7 5");
    var lx = L + 8, ly = T + 12;
    [["ink", "fit of Y on T", null], ["accent", "causal slope τ", "7 5"]].forEach(function (e, i) {
      svg.appendChild(WK.h("line", { x1: lx, x2: lx + 24, y1: ly + 16 * i, y2: ly + 16 * i,
        stroke: e[0], "stroke-width": 2.5, "stroke-dasharray": e[2] }));
      svg.appendChild(WK.h("text", { x: lx + 30, y: ly + 16 * i + 4, "font-size": 11,
        fill: "ink", text: e[1] }));
    });
    ["low", "mid", "high"].forEach(function (lab, i) {
      var xx = W - R - 150 + 50 * i;
      svg.appendChild(WK.h("circle", { cx: xx, cy: ly, r: 4, fill: tok[i] }));
      svg.appendChild(WK.h("text", { x: xx + 7, y: ly + 4, "font-size": 11, fill: "ink",
        text: lab }));
    });
    svg.appendChild(WK.h("text", { x: W - R - 150, y: ly + 18, "font-size": 11,
      fill: "muted", text: "severity tercile" }));
  }
  draw();
});
