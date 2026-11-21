/* Part 1 widget: goodness-of-fit power against what a forest gains. */
"use strict";

WK.mount("widget-gof", function (root, WK) {
  var D = WK.data("gof");
  if (!D) throw new Error("widget data missing");
  var shape = "regime", k = 2;
  var f = WK.frame({
    title: "Can a residual check see what a forest can use?",
    badge: "semi-synthetic",
    note: "Real 1976 LA meteorology (seven columns, 330 days); simulated response with a " +
      "nonlinear share η of the signal. Left: residuals against fitted values of the OLS fit " +
      "for one replicate. Right: the share of " + D.reps + " replicates in which each test " +
      "rejects linearity at the 5% level, with the current η marked. The tiles give the " +
      "forest's test error as a multiple of OLS's (median over replicates): below 1 means " +
      "the forest predicts better."
  });
  var tShape = WK.toggle({ label: "Shape of the nonlinearity", value: shape,
    options: [{ value: "regime", label: "hot and low inversion (fixed in advance)" },
              { value: "wiggle", label: "sin(3 × temp) (added afterwards)" }],
    onchange: function (v) { shape = v; draw(); } });
  var sEta = WK.slider({ label: "Nonlinear share of the signal, η", min: 0,
    max: D.etas.length - 1, step: 1, value: k,
    fmt: function (i) { return WK.fmt.num(D.etas[i], 2); },
    oninput: function (v) { k = v; draw(); } });
  f.controls.appendChild(tShape.root);
  f.controls.appendChild(sEta.root);
  var tiles = {
    reset: WK.stat({ label: "RESET rejects", value: "" }),
    quad: WK.stat({ label: "Quadratic F test rejects", value: "" }),
    rainbow: WK.stat({ label: "Rainbow test rejects", value: "" }),
    ratio: WK.stat({ label: "Forest test MSE ÷ OLS test MSE", value: "" })
  };
  Object.keys(tiles).forEach(function (key) { f.stats.appendChild(tiles[key].root); });
  var W = 640, H = 290, svg = WK.svg(W, H, { "aria-label":
    "Residual plot of one replicate and test rejection rates against nonlinearity" });
  f.body.appendChild(svg);
  root.appendChild(f.root);

  var series = [["reset", "c1", "RESET"], ["quad", "c2", "quadratic F"],
                ["rainbow", "c3", "rainbow"]];

  function axisText(x, y, s, anchor, extra) {
    return WK.h("text", Object.assign({ x: x, y: y, "font-size": 11, fill: "muted",
      "text-anchor": anchor || "middle", text: s }, extra || {}));
  }

  function draw() {
    var rows = D.shapes[shape], r = rows[k];
    tiles.reset.set(WK.fmt.pct(r.reset));
    tiles.quad.set(WK.fmt.pct(r.quad));
    tiles.rainbow.set(WK.fmt.pct(r.rainbow));
    tiles.ratio.set(WK.fmt.num(r.ratio_med, 2));
    WK.clear(svg);
    // Left: residuals against fitted values.
    var L0 = 44, L1 = 300, T0 = 16, B0 = H - 38;
    var fx = r.fit, rs = r.res;
    var fmin = Math.min.apply(null, fx), fmax = Math.max.apply(null, fx);
    var rmax = Math.max(3, Math.ceil(Math.max.apply(null, rs.map(Math.abs))));
    var x = WK.lin([Math.floor(fmin), Math.ceil(fmax)], [L0, L1]);
    var y = WK.lin([-rmax, rmax], [B0, T0]);
    WK.ticks(Math.floor(fmin), Math.ceil(fmax), 5).forEach(function (v) {
      svg.appendChild(axisText(x(v), B0 + 14, String(v)));
    });
    WK.ticks(-rmax, rmax, 5).forEach(function (v) {
      svg.appendChild(WK.h("line", { x1: L0, x2: L1, y1: y(v), y2: y(v), stroke: "rule" }));
      svg.appendChild(axisText(L0 - 5, y(v) + 4, String(v), "end"));
    });
    svg.appendChild(WK.h("line", { x1: L0, x2: L1, y1: y(0), y2: y(0), stroke: "muted" }));
    fx.forEach(function (v, i) {
      svg.appendChild(WK.h("circle", { cx: x(v), cy: y(rs[i]), r: 2, fill: "c5",
        "fill-opacity": 0.6 }));
    });
    svg.appendChild(axisText((L0 + L1) / 2, H - 8, "fitted value (OLS)", "middle",
      { fill: "ink", "font-size": 12 }));
    svg.appendChild(axisText(12, (T0 + B0) / 2, "residual", "middle",
      { fill: "ink", "font-size": 12, transform: "rotate(-90 12 " + (T0 + B0) / 2 + ")" }));
    // Right: rejection rate against eta.
    var R0 = 370, R1 = W - 14;
    var ex = WK.lin([0, D.etas[D.etas.length - 1]], [R0, R1]), ey = WK.lin([0, 1], [B0, T0]);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      svg.appendChild(WK.h("line", { x1: R0, x2: R1, y1: ey(v), y2: ey(v), stroke: "rule" }));
      svg.appendChild(axisText(R0 - 5, ey(v) + 4, Math.round(100 * v) + "%", "end"));
    });
    [0, 0.2, 0.4, 0.6, 0.8].forEach(function (v) {
      svg.appendChild(axisText(ex(v), B0 + 14, String(v)));
    });
    svg.appendChild(WK.h("line", { x1: ex(0), x2: ex(0.8), y1: ey(0.05), y2: ey(0.05),
      stroke: "muted", "stroke-dasharray": "3 3" }));
    svg.appendChild(WK.h("line", { x1: ex(r.eta), x2: ex(r.eta), y1: T0, y2: B0,
      stroke: "ink", "stroke-opacity": 0.5 }));
    series.forEach(function (s, j) {
      var pts = rows.map(function (q) { return ex(q.eta) + "," + ey(q[s[0]]); }).join(" ");
      svg.appendChild(WK.h("polyline", { points: pts, fill: "none", stroke: s[1],
        "stroke-width": 2.2 }));
      svg.appendChild(WK.h("circle", { cx: ex(r.eta), cy: ey(r[s[0]]), r: 4, fill: s[1] }));
      svg.appendChild(WK.h("line", { x1: R1 - 96, x2: R1 - 80, y1: T0 + 40 + 15 * j,
        y2: T0 + 40 + 15 * j, stroke: s[1], "stroke-width": 2.2 }));
      svg.appendChild(axisText(R1 - 76, T0 + 44 + 15 * j, s[2], "start", { fill: "ink" }));
    });
    svg.appendChild(axisText((R0 + R1) / 2, H - 8, "nonlinear share η", "middle",
      { fill: "ink", "font-size": 12 }));
    svg.appendChild(axisText(R0 - 36, (T0 + B0) / 2, "rejects linearity", "middle",
      { fill: "ink", "font-size": 12, transform: "rotate(-90 " + (R0 - 36) + " " +
        (T0 + B0) / 2 + ")" }));
  }
  draw();
});
