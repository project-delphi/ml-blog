/* Part 3 widget: test error against the number of random features (double descent). */
"use strict";

WK.mount("widget-dd", function (root, WK) {
  var D = WK.data("doubledescent");
  if (!D) throw new Error("widget data missing");
  var key = "minnorm", k = 0;
  var widths = D.series.minnorm.width;
  k = Math.max(0, widths.indexOf(D.start_width));  // the classical minimum
  var f = WK.frame({
    title: "More features than examples",
    badge: "USPS digits",
    note: "1,000 training digits, 2,007 test digits. Each digit's 256 pixels are mapped to p " +
      "random ReLU features; a least-squares fit to the one-hot labels predicts the class with " +
      "the largest output. Curves are means over " + D.seeds + " draws of the training set and " +
      "the random features. The dashed line marks p = n = 1,000."
  });
  var tFit = WK.toggle({ label: "Fit", value: key,
    options: [{ value: "minnorm", label: "minimum-norm least squares" },
              { value: "ridge", label: "small ridge penalty (λ = 0.1)" }],
    onchange: function (v) { key = v; draw(); } });
  var s = WK.slider({ label: "Random features p", min: 0, max: widths.length - 1, step: 1,
    value: k, fmt: function (i) { return widths[i].toLocaleString("en"); },
    oninput: function (v) { k = v; draw(); } });
  f.controls.appendChild(tFit.root);
  f.controls.appendChild(s.root);
  var tTe = WK.stat({ label: "test error", value: "" });
  var tTr = WK.stat({ label: "training error", value: "" });
  var tR = WK.stat({ label: "features per training digit, p ÷ n", value: "" });
  [tTe, tTr, tR].forEach(function (t) { f.stats.appendChild(t.root); });
  var W = 640, H = 290, svg = WK.svg(W, H, { "aria-label": "Train and test error against number of features" });
  f.body.appendChild(svg);
  root.appendChild(f.root);

  function draw() {
    var S = D.series[key];
    tTe.set(WK.fmt.pct(S.test[k], 1));
    tTr.set(WK.fmt.pct(S.train[k], 1));
    tR.set(WK.fmt.num(widths[k] / D.n_train, 2));
    WK.clear(svg);
    var L0 = 54, R0 = W - 14, T0 = 14, B0 = H - 40;
    var lx = function (p) { return Math.log10(p); };
    var x = WK.lin([lx(widths[0]), lx(widths[widths.length - 1])], [L0, R0]);
    var y = WK.lin([0, 0.6], [B0, T0]);
    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].forEach(function (v) {
      svg.appendChild(WK.h("line", { x1: L0, x2: R0, y1: y(v), y2: y(v), stroke: "rule" }));
      svg.appendChild(WK.h("text", { x: L0 - 6, y: y(v) + 4, "text-anchor": "end",
        "font-size": 11, fill: "muted", text: Math.round(100 * v) + "%" }));
    });
    [10, 100, 1000, 10000].forEach(function (v) {
      svg.appendChild(WK.h("text", { x: x(lx(v)), y: B0 + 15, "text-anchor": "middle",
        "font-size": 11, fill: "muted", text: v.toLocaleString("en") }));
    });
    svg.appendChild(WK.h("line", { x1: x(lx(D.n_train)), x2: x(lx(D.n_train)), y1: T0, y2: B0,
      stroke: "muted", "stroke-dasharray": "4 4" }));
    var band = S.width.map(function (p, i) { return x(lx(p)) + "," + y(S.test_hi[i]); })
      .concat(S.width.slice().reverse().map(function (p, i) {
        var j = S.width.length - 1 - i; return x(lx(p)) + "," + y(S.test_lo[j]);
      })).join(" ");
    svg.appendChild(WK.h("polygon", { points: band, fill: "c1", "fill-opacity": 0.15 }));
    [["test", "c1", "test"], ["train", "c2", "training"]].forEach(function (c, j) {
      var pts = S.width.map(function (p, i) { return x(lx(p)) + "," + y(S[c[0]][i]); }).join(" ");
      svg.appendChild(WK.h("polyline", { points: pts, fill: "none", stroke: c[1], "stroke-width": 2.2 }));
      svg.appendChild(WK.h("circle", { cx: x(lx(widths[k])), cy: y(S[c[0]][k]), r: 4.5, fill: c[1] }));
      svg.appendChild(WK.h("line", { x1: R0 - 110, x2: R0 - 92, y1: T0 + 10 + 16 * j,
        y2: T0 + 10 + 16 * j, stroke: c[1], "stroke-width": 2.2 }));
      svg.appendChild(WK.h("text", { x: R0 - 86, y: T0 + 14 + 16 * j, "font-size": 11,
        fill: "ink", text: c[2] + " error" }));
    });
    svg.appendChild(WK.h("text", { x: (L0 + R0) / 2, y: H - 8, "text-anchor": "middle",
      "font-size": 12, fill: "ink", text: "random features p (log scale)" }));
    svg.appendChild(WK.h("text", { x: 14, y: (T0 + B0) / 2, "text-anchor": "middle",
      "font-size": 12, fill: "ink", transform: "rotate(-90 14 " + (T0 + B0) / 2 + ")",
      text: "misclassification" }));
  }
  draw();
});
