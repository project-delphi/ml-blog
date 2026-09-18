/* Widget for "Understanding Attention": why the scores are divided by sqrt(d).
 *
 * Six tokens get independent random query and key vectors. One 256-wide pair
 * is drawn once from fixed seeds and the widget uses its first d columns, so
 * stepping d adds coordinates to the same vectors rather than swapping in new
 * ones. Dot products of unit-variance vectors have variance d, so without the
 * divisor the softmax saturates onto one key as d grows; with it the weights
 * stay soft at every d. Because any single row can be close by chance, the
 * stats are averages over the six rows, and the prose quotes those. Built on
 * widget-kit (WK); colours are theme tokens.
 */
"use strict";

WK.mount("widget-attention", function (root, WK) {
  var TOKENS = ["The", "cat", "sat", "on", "the", "mat"];
  var DIMS = [4, 16, 64, 256];
  var D_MAX = 256;

  // Deterministic draws: the same picture every visit.
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
  function matrix(seed) {
    var rng = mulberry32(seed), X = [];
    for (var i = 0; i < TOKENS.length; i++) {
      var x = [];
      for (var j = 0; j < D_MAX; j++) x.push(gaussian(rng));
      X.push(x);
    }
    return X;
  }
  var Q = matrix(6), K = matrix(1006);

  // The model: S = Q K^T over the first d columns, optionally divided by
  // sqrt(d), then softmax row by row.
  function model(d, scale) {
    var n = TOKENS.length, S = [], A = [], div = scale ? Math.sqrt(d) : 1;
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
  function mean(xs) { return xs.reduce(function (a, b) { return a + b; }, 0) / xs.length; }

  // ------------------------------------------------------------------ view
  var state = { d: 4, scale: true, q: 1 };
  var f = WK.frame({
    title: "Softmax attention and the divisor",
    note: "Six tokens with independent random query and key vectors of dimension d, " +
      "so the picture is about scale, not meaning. Each row of the grid is one query's " +
      "weights over the six keys; the stats average over the six rows, because any one " +
      "row can be close by chance."
  });
  var dim = WK.toggle({ label: "Embedding dimension d",
    options: DIMS.map(function (d) { return { value: d, label: String(d) }; }),
    value: state.d, onchange: function (v) { state.d = v; draw(); } });
  var sc = WK.toggle({ label: "Divide scores by",
    options: [{ value: true, label: "√d" }, { value: false, label: "nothing" }],
    value: state.scale, onchange: function (v) { state.scale = v; draw(); } });
  var qt = WK.toggle({ label: "Query token (for the bars)",
    options: TOKENS.map(function (t, i) { return { value: i, label: t }; }),
    value: state.q, onchange: function (v) { state.q = v; draw(); } });
  f.controls.appendChild(dim.root);
  f.controls.appendChild(sc.root);
  f.controls.appendChild(qt.root);

  var W = 640, H = 300, CELL = 40, GX = 90, GY = 40;
  var svg = WK.svg(W, H, { "aria-label": "Attention weight grid and one query row" });
  f.body.appendChild(svg);
  var maxStat = WK.stat({ label: "largest weight per row, averaged", value: "" });
  var entStat = WK.stat({ label: "entropy per row, averaged (bits, max 2.58)", value: "" });
  var rawStat = WK.stat({ label: "largest raw score (before the divisor)", value: "" });
  f.stats.appendChild(maxStat.root);
  f.stats.appendChild(entStat.root);
  f.stats.appendChild(rawStat.root);
  root.appendChild(f.root);

  function draw() {
    var r = model(state.d, state.scale), n = TOKENS.length, i, j;
    WK.clear(svg);
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
    maxStat.set(mean(r.A.map(function (row) { return Math.max.apply(null, row); })).toFixed(2));
    entStat.set(mean(r.A.map(entropyBits)).toFixed(2));
    var raw = 0;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
      raw = Math.max(raw, Math.abs(r.S[i][j]) * (state.scale ? Math.sqrt(state.d) : 1));
    }
    rawStat.set(raw.toFixed(1));
  }
  draw();
});
