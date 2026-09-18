"use strict";

// Selecting features on the whole sample, then splitting, reports an accuracy
// that the same procedure done in the right order does not. The data here has
// no signal at all: every feature is noise and the labels are coin flips, so
// the honest accuracy is 50% by construction and anything above it is the test
// set leaking through the selection step.
WK.mount("widget-leakage", function (root, WK) {
  var REPEATS = 8;

  // ---------------------------------------------------------------- model
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** n samples of p pure-noise features, with labels that depend on nothing. */
  function sample(n, p, rand) {
    var X = new Float64Array(n * p), y = new Int8Array(n);
    for (var i = 0; i < n; i++) {
      y[i] = rand() < 0.5 ? -1 : 1;
      for (var j = 0; j < p; j++) X[i * p + j] = gauss(rand);
    }
    return { X: X, y: y, n: n, p: p };
  }

  /** Indices of the k features most correlated with y over the given rows. */
  function topK(d, rows, k) {
    var score = new Float64Array(d.p);
    for (var j = 0; j < d.p; j++) {
      var sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0, m = rows.length;
      for (var r = 0; r < m; r++) {
        var i = rows[r], x = d.X[i * d.p + j], yy = d.y[i];
        sx += x; sy += yy; sxy += x * yy; sxx += x * x; syy += yy * yy;
      }
      var cov = sxy - sx * sy / m;
      var den = Math.sqrt((sxx - sx * sx / m) * (syy - sy * sy / m));
      score[j] = den > 0 ? Math.abs(cov / den) : 0;
    }
    var order = [];
    for (var q = 0; q < d.p; q++) order.push(q);
    order.sort(function (a, b) { return score[b] - score[a]; });
    return order.slice(0, k);
  }

  /** Nearest-centroid accuracy on `test`, fitted on `train`, over `feats`. */
  function accuracy(d, feats, train, test) {
    var cA = new Float64Array(feats.length), cB = new Float64Array(feats.length);
    var nA = 0, nB = 0, t, i, f;
    for (t = 0; t < train.length; t++) {
      i = train[t];
      var into = d.y[i] === 1 ? cA : cB;
      for (f = 0; f < feats.length; f++) into[f] += d.X[i * d.p + feats[f]];
      if (d.y[i] === 1) nA++; else nB++;
    }
    if (!nA || !nB) return 0.5;
    for (f = 0; f < feats.length; f++) { cA[f] /= nA; cB[f] /= nB; }
    var right = 0;
    for (t = 0; t < test.length; t++) {
      i = test[t];
      var dA = 0, dB = 0;
      for (f = 0; f < feats.length; f++) {
        var x = d.X[i * d.p + feats[f]];
        dA += (x - cA[f]) * (x - cA[f]);
        dB += (x - cB[f]) * (x - cB[f]);
      }
      var pred = dA < dB ? 1 : -1;
      if (pred === d.y[i]) right++;
    }
    return right / test.length;
  }

  /** One trial: the same selection done before the split and after it. */
  function trial(n, p, k, seed) {
    var rand = rng(seed), d = sample(n, p, rand);
    var idx = [];
    for (var i = 0; i < n; i++) idx.push(i);
    for (var a = n - 1; a > 0; a--) {
      var b = Math.floor(rand() * (a + 1)), tmp = idx[a]; idx[a] = idx[b]; idx[b] = tmp;
    }
    var cut = Math.floor(0.8 * n), train = idx.slice(0, cut), test = idx.slice(cut);
    var all = [];
    for (var q = 0; q < n; q++) all.push(q);
    return {
      leaky: accuracy(d, topK(d, all, k), train, test),
      honest: accuracy(d, topK(d, train, k), train, test)
    };
  }

  /** REPEATS trials, so a reader never reads one seed as a result. */
  function run(n, p, k, seed) {
    var L = [], H = [];
    for (var r = 0; r < REPEATS; r++) {
      var t = trial(n, p, k, seed + r * 7919);
      L.push(t.leaky); H.push(t.honest);
    }
    var mean = function (a) { return a.reduce(function (s, x) { return s + x; }, 0) / a.length; };
    return {
      leaky: mean(L), honest: mean(H),
      leakyLo: Math.min.apply(null, L), leakyHi: Math.max.apply(null, L),
      honestLo: Math.min.apply(null, H), honestHi: Math.max.apply(null, H)
    };
  }

  // ----------------------------------------------------------------- view
  var f = WK.frame({
    title: "Selecting features before the split",
    note: "Every feature is noise and every label is a coin flip, so 50% is the truth. "
        + "The bars average " + REPEATS + " draws; the thin line is the range across them."
  });

  var nS = WK.slider({ label: "Rows", min: 40, max: 200, step: 20, value: 100,
                       fmt: WK.fmt.int, oninput: draw });
  var pS = WK.slider({ label: "Noise features", min: 50, max: 800, step: 50, value: 500,
                       fmt: WK.fmt.int, oninput: draw });
  var kS = WK.slider({ label: "Features kept", min: 1, max: 40, step: 1, value: 10,
                       fmt: WK.fmt.int, oninput: draw });
  var seedS = WK.slider({ label: "Seed", min: 1, max: 40, step: 1, value: 7,
                          fmt: WK.fmt.int, oninput: draw });
  [nS, pS, kS, seedS].forEach(function (c) { f.controls.appendChild(c.root); });

  var svg = WK.svg(640, 220);
  f.body.appendChild(svg);

  var statLeaky = WK.stat({ label: "Selected before the split", value: "--" });
  var statHonest = WK.stat({ label: "Selected inside the split", value: "--" });
  var statGap = WK.stat({ label: "Points of pure illusion", value: "--" });
  [statLeaky, statHonest, statGap].forEach(function (s) { f.stats.appendChild(s.root); });

  root.appendChild(f.root);

  function bar(x, label, m, lo, hi, token) {
    var scale = WK.lin([0.3, 1], [180, 40]);
    var y = scale(m), base = scale(0.3);
    svg.appendChild(WK.h("rect", { x: x, y: y, width: 90, height: base - y, fill: token }));
    svg.appendChild(WK.h("line", { x1: x + 45, x2: x + 45, y1: scale(lo), y2: scale(hi),
                                   stroke: "ink", "stroke-width": 2 }));
    // the label clears the whisker, which reaches above the mean whenever the
    // spread across draws does
    svg.appendChild(WK.h("text", { x: x + 45, y: Math.min(y, scale(hi)) - 10,
                                   "text-anchor": "middle",
                                   fill: "ink", "font-size": 15,
                                   text: WK.fmt.pct(m, 1) }));
    svg.appendChild(WK.h("text", { x: x + 45, y: 200, "text-anchor": "middle",
                                   fill: "muted", "font-size": 13, text: label }));
  }

  function draw() {
    var n = nS.get(), p = pS.get(), k = kS.get(), seed = seedS.get();
    if (k > n) { kS.set(n, true); k = n; }
    var r = run(n, p, k, seed);

    WK.clear(svg);
    var scale = WK.lin([0.3, 1], [180, 40]);
    // the chance line: the only honest answer for data with no signal
    svg.appendChild(WK.h("line", { x1: 30, x2: 610, y1: scale(0.5), y2: scale(0.5),
                                   stroke: "rule", "stroke-width": 1,
                                   "stroke-dasharray": "5 4" }));
    svg.appendChild(WK.h("text", { x: 616, y: scale(0.5) + 4, fill: "muted",
                                   "font-size": 12, text: "50%" }));
    bar(150, "before the split", r.leaky, r.leakyLo, r.leakyHi, "c4");
    bar(400, "inside the split", r.honest, r.honestLo, r.honestHi, "c2");

    statLeaky.set(WK.fmt.pct(r.leaky, 1));
    statHonest.set(WK.fmt.pct(r.honest, 1));
    statGap.set(WK.fmt.num(100 * (r.leaky - r.honest), 1));
  }

  draw();
});
