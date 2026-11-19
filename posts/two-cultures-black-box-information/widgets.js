/* Part 4 widget: two importance measures across bootstrap resamples of hepatitis. */
"use strict";

var ImportanceModel = (function () {
  function rank(values, j) {
    var v = values[j], r = 1;
    values.forEach(function (x, k) { if (k !== j && x > v) r += 1; });
    return r;
  }
  var api = { rank: rank };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();

if (typeof WK !== "undefined") WK.mount("widget-importance", function (root, WK) {
  var D = WK.data("importance"), M = ImportanceModel;
  if (!D) throw new Error("widget data missing");
  var method = "perm", b = 0;
  var f = WK.frame({
    title: "Hepatitis importance under resampling",
    badge: "UCI hepatitis",
    note: "One forest of 500 trees per resample: resample 0 is the full 155 patients, 1–30 are " +
      "bootstrap resamples. Covariates are numbered as in Breiman's paper; N1 is a continuous " +
      "noise column and N2 a binary one, added to the data. Permutation: % rise in out-of-bag " +
      "error when the covariate is shuffled. Impurity: share of the forest's Gini decrease."
  });
  var tM = WK.toggle({ label: "Measure", value: method,
    options: [{ value: "perm", label: "permutation" }, { value: "mdi", label: "impurity" }],
    onchange: function (v) { method = v; draw(); } });
  var sB = WK.slider({ label: "Resample", min: 0, max: D.boots, step: 1, value: 0,
    fmt: function (v) { return v === 0 ? "0 (full data)" : String(v); },
    oninput: function (v) { b = v; draw(); } });
  f.controls.appendChild(tM.root);
  f.controls.appendChild(sB.root);
  var asc = D.names.indexOf("ASCITES"), alb = D.names.indexOf("ALBUMIN");
  var nc = D.names.indexOf("NOISE_CONT"), nb = D.names.indexOf("NOISE_BIN");
  var tA = WK.stat({ label: "rank of ascites (12)", value: "" });
  var tL = WK.stat({ label: "rank of albumin (17)", value: "" });
  var tN = WK.stat({ label: "rank of continuous noise (N1)", value: "" });
  [tA, tL, tN].forEach(function (t) { f.stats.appendChild(t.root); });
  var W = 640, H = 260, svg = WK.svg(W, H, { "aria-label": "Bar chart of importance per covariate" });
  f.body.appendChild(svg);
  root.appendChild(f.root);
  var labels = D.names.map(function (n, j) {
    return j === nc ? "N1" : (j === nb ? "N2" : String(j + 1));
  });

  function draw() {
    var v = D[method][b], n = v.length;
    tA.set(M.rank(v, asc) + " of " + n);
    tL.set(M.rank(v, alb) + " of " + n);
    tN.set(M.rank(v, nc) + " of " + n);
    WK.clear(svg);
    var L0 = 50, R0 = W - 10, T0 = 12, B0 = H - 34;
    var all = D[method].reduce(function (a, r) { return a.concat(r); }, []);
    var lo = Math.min(0, Math.min.apply(null, all)), hi = Math.max.apply(null, all);
    var y = WK.lin([lo, hi], [B0, T0]), bw = (R0 - L0) / n;
    WK.ticks(lo, hi, 5).forEach(function (t) {
      svg.appendChild(WK.h("line", { x1: L0, x2: R0, y1: y(t), y2: y(t), stroke: "rule" }));
      svg.appendChild(WK.h("text", { x: L0 - 6, y: y(t) + 4, "text-anchor": "end",
        "font-size": 11, fill: "muted", text: method === "perm" ? String(t) : t.toFixed(2) }));
    });
    v.forEach(function (x, j) {
      var tok = (j === asc || j === alb) ? "c3" : ((j === nc || j === nb) ? "c4" : "c1");
      var top = y(Math.max(0, x)), bot = y(Math.min(0, x));
      svg.appendChild(WK.h("rect", { x: L0 + j * bw + 2, width: bw - 4, y: top,
        height: Math.max(0.5, bot - top), fill: tok }));
      svg.appendChild(WK.h("text", { x: L0 + (j + 0.5) * bw, y: B0 + 14, "text-anchor": "middle",
        "font-size": 10.5, fill: "ink", text: labels[j] }));
    });
    svg.appendChild(WK.h("text", { x: (L0 + R0) / 2, y: H - 4, "text-anchor": "middle",
      "font-size": 12, fill: "ink", text: "covariate (Breiman's numbering)" }));
    svg.appendChild(WK.h("text", { x: 12, y: (T0 + B0) / 2, "text-anchor": "middle",
      "font-size": 12, fill: "ink", transform: "rotate(-90 12 " + (T0 + B0) / 2 + ")",
      text: method === "perm" ? "% rise in OOB error" : "impurity share" }));
  }
  draw();
});
