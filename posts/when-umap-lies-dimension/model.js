/* The numbers behind the widgets in "When UMAP Lies, Part 2".
 *
 * Kept apart from the drawing (widgets.js) so the prose can be checked against
 * it: `node posts/when-umap-lies/src/check_widgets.js dimension` sweeps every
 * control and prints what the widget shows. Exposes one global, `UMLDimModel`,
 * and `module.exports` under node. Needs UK (../when-umap-lies/umapkit.js) and
 * reads WK_DATA["uml-dimension"] (widget-data/dimension.js, written by
 * posts/when-umap-lies/src/export_widget_data.py).
 *
 * The experiment (cache/noise_dims): a base structure of p dimensions (Swiss
 * roll p = 3, Gaussian clusters p = 10) padded with pure-noise columns of sd
 * sigma to D columns, n = 2,000, five independent draws (data seed = algo seed
 * = 0..4). sigma is fixed per base so the noise index
 * eta = sigma^2 sqrt(8 (D - p)) / r10^2 is 1 at D - p = 300. Quality is k = 10
 * neighbour preservation against the truth: the share of each point's 10 true
 * nearest neighbours (in the clean structure) that are among its 10 nearest in
 * the map, or in the kNN graph a method starts from.
 */
"use strict";

var UMLDimModel = (function () {
  var K = typeof UK !== "undefined" ? UK : require("../when-umap-lies/umapkit.js");

  // Styles come from the series-wide method map (UK.METHOD), so a method looks
  // the same here as in every figure and widget.
  function styled(key, styleKey) {
    var m = K.method(styleKey || key);
    return { key: key, label: m.name, token: m.token, dash: m.dash, marker: m.marker };
  }
  var SERIES = ["tsne", "umap", "umap·pca50", "pca"].map(function (k) { return styled(k); });
  var GRAPHS = [styled("input"), styled("pca50", "pca50graph")];

  function ds(Data, key) { return Data.datasets[key]; }
  function index(Data, key, D) {
    var i = ds(Data, key).D.indexOf(D);
    if (i < 0) throw new Error("no D = " + D + " for " + key);
    return i;
  }

  /** Median and min-max over the five draws at one D, plus the seed-0 value
   *  (the run the panels show): {v, lo, hi, n, seed0}, or null where absent. */
  function draws(arr) {
    if (!arr) return null;
    var s = K.summary(arr, "median");
    s.seed0 = arr[0];
    return s;
  }
  function q10(Data, key, config, D) { return draws(ds(Data, key).q10[config][index(Data, key, D)]); }
  function graph(Data, key, space, D) { return draws(ds(Data, key).graph[space][index(Data, key, D)]); }
  function seconds(Data, key, D) { return draws(ds(Data, key).seconds.umap[index(Data, key, D)]); }
  function eta(Data, key, D) { return ds(Data, key).eta[index(Data, key, D)]; }

  /** One line of the chart: [{x: D, v, lo, hi}] over every D (v null where absent). */
  function curve(Data, key, kind, k) {
    return ds(Data, key).D.map(function (D) {
      var s = kind === "graph" ? graph(Data, key, k, D) : q10(Data, key, k, D);
      return s ? { x: D, v: s.v, lo: s.lo, hi: s.hi } : { x: D, v: null };
    });
  }

  var cache = {};
  /** The seed-0 map of `config` at D, Procrustes-aligned to the truth: interleaved xy. */
  function layout(Data, key, config, D) {
    var c = key + "/" + D + "/" + config;
    if (!cache[c]) cache[c] = K.decode(ds(Data, key).layouts[String(D)][config]);
    return cache[c];
  }
  function truth(Data, key) {
    var c = key + "/truth";
    if (!cache[c]) cache[c] = K.decode(ds(Data, key).truth);
    return cache[c];
  }
  /** {values} for a continuous colouring or {labels, classes, counts}. */
  function colour(Data, key) {
    var c = ds(Data, key).colour, k = key + "/colour";
    if (!cache[k]) {
      cache[k] = c.kind === "values" ? { values: K.decodeValues(c.values), label: c.label } :
        { labels: K.decodeLabels(c.labels), classes: c.classes, counts: c.counts, label: c.label };
    }
    return cache[k];
  }

  // ------------------------------------------------------- widget-signal
  // WK_DATA["uml-signal"] (widget-data/signal.js): six Gaussian clusters with d
  // internal dimensions, padded with noise columns of sd sigma to D columns, n
  // points, ONE draw (data seed 0) per setting. Each cell holds PCA, UMAP and
  // t-SNE (Procrustes-aligned to the truth), their k = 10 neighbour
  // preservation and trustworthiness against the truth, 15-NN class accuracy,
  // the input graph's k = 10 preservation, and eta.
  var SIGNAL_METHODS = ["pca", "umap", "tsne"].map(function (k) { return styled(k); });

  function signalKey(d, D, sigma, n) { return d + "/" + D + "/" + String(sigma) + "/" + n; }
  /** {eta, q10: {pca, umap, tsne, input}, tw10: {...}, acc: {...}, disparity} for one setting. */
  function signal(S, d, D, sigma, n) {
    var c = S.cells[signalKey(d, D, sigma, n)];
    if (!c) throw new Error("no signal cell " + signalKey(d, D, sigma, n));
    return c;
  }
  /** {truth, labels, classes, counts, shown, r10sq} for (d, n): the panel the maps are aligned to. */
  function signalBase(S, d, n) { return S.base[d + "/" + n]; }
  function signalLayout(S, d, D, sigma, n, m) {
    var k = "sig/" + signalKey(d, D, sigma, n) + "/" + m;
    if (!cache[k]) cache[k] = K.decode(signal(S, d, D, sigma, n).layouts[m]);
    return cache[k];
  }
  function signalTruth(S, d, n) {
    var k = "sig/" + d + "/" + n;
    if (!cache[k]) {
      var b = signalBase(S, d, n);
      cache[k] = { xy: K.decode(b.truth), labels: K.decodeLabels(b.labels), classes: b.classes, counts: b.counts };
    }
    return cache[k];
  }

  return { SERIES: SERIES, GRAPHS: GRAPHS, index: index, q10: q10, graph: graph, seconds: seconds,
    eta: eta, curve: curve, layout: layout, truth: truth, colour: colour,
    SIGNAL_METHODS: SIGNAL_METHODS, signal: signal, signalBase: signalBase,
    signalLayout: signalLayout, signalTruth: signalTruth };
})();

if (typeof module !== "undefined" && module.exports) module.exports = UMLDimModel;
