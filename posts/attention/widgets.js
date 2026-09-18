/* Widget for "Understanding Attention": why the scores are divided by sqrt(d).
 *
 * Six tokens get independent random query and key vectors of dimension d.
 * Their dot products have variance d, so without the divisor the softmax
 * saturates onto one key as d grows; with it the weights stay soft at every d.
 * Built on widget-kit (WK), colours by token so it follows the light/dark toggle.
 */
"use strict";

WK.mount("widget-attention", function (root, WK) {
  var TOKENS = ["The", "cat", "sat", "on", "the", "mat"];
  var DIMS = [4, 16, 64, 256];

  // Deterministic embeddings: the same picture every visit, at every d.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    var u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function matrix(d, seed) {
    var rng = mulberry32(seed + d), X = [];
    for (var i = 0; i < TOKENS.length; i++) {
      var x = [];
      for (var j = 0; j < d; j++) x.push(gaussian(rng));
      X.push(x);
    }
    return X;
  }

  // The model: Q and K are independent Gaussian matrices, as two different
  // projections of the same input would be; S = Q K^T, optionally divided by
  // sqrt(d), then softmax row by row.
  function model(d, scale) {
    var Q = matrix(d, 20250131), K = matrix(d, 31012025), n = TOKENS.length, S = [], A = [];
    var div = scale ? Math.sqrt(d) : 1;
    for (var i = 0; i < n; i++) {
      S.push([]);
      for (var j = 0; j < n; j++) {
        var s = 0;
        for (var k = 0; k < d; k++) s += Q[i][k] * K[j][k];
        S[i].push(s / div);
      }
    }
    for (i = 0; i < n; i++) {
      var m = Math.max.apply(null, S[i]), z = 0, row = [];
      for (j = 0; j < n; j++) { row.push(Math.exp(S[i][j] - m)); z += row[j]; }
      A.push(row.map(function (e) { return e / z; }));
    }
    return { S: S, A: A };
  }
  function entropyBits(p) {
    return -p.reduce(function (h, x) { return x > 0 ? h + x * Math.log2(x) : h; }, 0);
  }

  // ------------------------------------------------------------------ view
  var state = { d: 4, scale: true, q: 1 };
  var f = WK.frame({
    title: "Softmax attention and the divisor",
    note: "Six tokens with independent random query and key vectors of dimension d, " +
      "so the picture is about scale, not meaning. Each row of the grid is one query's " +
      "weights over the six keys."
  });
  var dim = WK.toggle({ label: "Embedding dimension d",
    options: DIMS.map(function (d) { return { value: d, label: String(d) }; }),
    value: state.d, onchange: function (v) { state.d = v; draw(); } });
  var sc = WK.toggle({ label: "Divide scores by",
    options: [{ value: true, label: "√d" }, { value: false, label: "nothing" }],
    value: state.scale, onchange: function (v) { state.scale = v; draw(); } });
  var qt = WK.toggle({ label: "Query token",
    options: TOKENS.map(function (t, i) { return { value: i, label: t }; }),
    value: state.q, onchange: function (v) { state.q = v; draw(); } });
  f.controls.appendChild(dim.root);
  f.controls.appendChild(sc.root);
  f.controls.appendChild(qt.root);

  var W = 640, H = 300, CELL = 40, GX = 90, GY = 40;
  var svg = WK.svg(W, H, { "aria-label": "Attention weight grid and one query row" });
  f.body.appendChild(svg);
  var maxStat = WK.stat({ label: "largest weight in the row", value: "" });
  var entStat = WK.stat({ label: "entropy of the row (bits, max 2.58)", value: "" });
  var rawStat = WK.stat({ label: "largest raw score", value: "" });
  f.stats.appendChild(maxStat.root);
  f.stats.appendChild(entStat.root);
  f.stats.appendChild(rawStat.root);
  root.appendChild(f.root);

  function draw() {
    var r = model(state.d, state.scale), n = TOKENS.length, i, j;
    WK.clear(svg);
    // Column labels (keys) and row labels (queries).
    for (j = 0; j < n; j++) {
      svg.appendChild(WK.h("text", { x: GX + j * CELL + CELL / 2, y: GY - 10,
        "text-anchor": "middle", "font-size": 12, fill: "muted" }, TOKENS[j]));
    }
    for (i = 0; i < n; i++) {
      var isQ = i === state.q;
      svg.appendChild(WK.h("text", { x: GX - 8, y: GY + i * CELL + CELL / 2 + 4,
        "text-anchor": "end", "font-size": 12, "font-weight": isQ ? 700 : 400,
        fill: isQ ? "accent" : "muted" }, TOKENS[i]));
      for (j = 0; j < n; j++) {
        var a = r.A[i][j];
        svg.appendChild(WK.h("rect", { x: GX + j * CELL, y: GY + i * CELL,
          width: CELL - 2, height: CELL - 2, rx: 3, fill: "c1",
          "fill-opacity": (0.08 + 0.92 * a).toFixed(3),
          stroke: isQ ? "accent" : "rule", "stroke-width": isQ ? 2 : 1 }));
        svg.appendChild(WK.h("text", { x: GX + j * CELL + CELL / 2 - 1,
          y: GY + i * CELL + CELL / 2 + 4, "text-anchor": "middle",
          "font-size": 11, fill: a > 0.5 ? "surface" : "ink" }, a.toFixed(2)));
      }
    }
    // The chosen query's row as bars, to the right of the grid.
    var bx = GX + n * CELL + 50, bw = 30, bh = 200, by = GY;
    svg.appendChild(WK.h("text", { x: bx, y: GY - 10, "font-size": 12, fill: "muted" },
      "weights for “" + TOKENS[state.q] + "”"));
    for (j = 0; j < n; j++) {
      var w = r.A[state.q][j], hgt = w * bh;
      svg.appendChild(WK.h("rect", { x: bx + j * (bw + 6), y: by + bh - hgt,
        width: bw, height: hgt, fill: j === state.q ? "accent" : "c2" }));
      svg.appendChild(WK.h("text", { x: bx + j * (bw + 6) + bw / 2, y: by + bh + 14,
        "text-anchor": "middle", "font-size": 11, fill: "muted" }, TOKENS[j]));
    }
    svg.appendChild(WK.h("line", { x1: bx, x2: bx + n * (bw + 6) - 6, y1: by + bh, y2: by + bh,
      stroke: "rule" }));
    var row = r.A[state.q];
    maxStat.set(Math.max.apply(null, row).toFixed(2));
    entStat.set(entropyBits(row).toFixed(2));
    var raw = r.S[state.q].map(function (s) { return state.scale ? s * Math.sqrt(state.d) : s; });
    rawStat.set(Math.max.apply(null, raw).toFixed(1));
  }
  draw();
});
