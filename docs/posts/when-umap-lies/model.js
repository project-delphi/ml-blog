/* The numbers behind the widgets in "When UMAP Lies, Part 1".
 *
 * Kept apart from the drawing (widgets.js) so the prose can be checked against
 * it: `node posts/when-umap-lies/src/check_widgets.js` sweeps every control and
 * prints what the widget shows. Exposes one global, `UMLModel`, and
 * `module.exports` under node. Needs UK (umapkit.js) for decoding layouts, and
 * reads its data from WK_DATA["uml-elephant"] (widget-data/elephant.js).
 *
 * The elephant (src/umaplab/elephant.py): at weight f the layout is the straight
 * blend (1 - f) * PCA + f * elephant, where "elephant" assigns each cell to a
 * point sampled uniformly inside a filled elephant silhouette (not the outline
 * Picasso used), by an optimal assignment from its PCA
 * position. f = 0 is exactly 2-D PCA. The "random" elephant assigns cells to the
 * same shape points at random. Scores are cached at f = 0, 0.1, ..., 1.
 */
"use strict";

var UMLModel = (function () {
  var K = typeof UK !== "undefined" ? UK : require("./umapkit.js");

  var VIEWS = [
    { key: "elephant", label: "Elephant at f" },
    { key: "random", label: "Random elephant" },
    { key: "umap", label: "UMAP" },
    { key: "tsne", label: "t-SNE" }
  ];

  // Metric metadata. `noun` is "cell type" or "cluster", from the data.
  function metricInfo(key, ds) {
    var t = ds.noun === "cluster" ? "cluster" : "type";
    var T = t.charAt(0).toUpperCase() + t.slice(1);
    var info = {
      inter_corr: { group: "chari", short: ["Between-" + t, ds.largeTypes ? "distances, all" : "distances"],
        tile: "between-" + t + " distance r" + (ds.largeTypes ? ", all " + ds.classes.length : ""),
        title: "Pearson r between the distances separating " + t + " centroids in the embedding and in the reference space" },
      inter_corr_large: { group: "chari", short: ["Between-" + t, "distances, ≥ " + ds.largeMin],
        tile: "between-" + t + " distance r, " + (ds.largeTypes || []).length + " types ≥ " + ds.largeMin + " cells",
        title: "As the all-types correlation, restricted to the " + (ds.largeTypes || []).length +
          " cell types with at least " + ds.largeMin + " cells" },
      intra_corr: { group: "chari", short: ["Within-" + t, "spread"], tile: "within-" + t + " spread r",
        title: "Pearson r, across " + t + "s, of the mean pairwise distance inside each " + t },
      knn_acc: { group: "lause", short: ["10-NN", "accuracy"], tile: "10-NN " + t + " accuracy",
        title: "Leave-one-out majority vote of the 10 nearest neighbours in the embedding: share of points whose " + t + " it gets right" },
      knn_recall: { group: "lause", short: ["10-NN", "recall"], tile: "10-NN recall",
        title: "Share of each point's 10 nearest neighbours in the reference space that are also among its 10 nearest in the embedding" },
      silhouette: { group: "lause", short: ["Silhouette", "on " + t + "s"], tile: "silhouette",
        title: "Mean silhouette of the " + t + " labels in the embedding" },
      max_ami: { group: "lause", short: ["Best", "HDBSCAN AMI"], tile: "best HDBSCAN AMI",
        title: "Best adjusted mutual information between HDBSCAN clusters of the embedding and the " + t + " labels, over nine settings" }
    };
    return Object.assign({ key: key, T: T }, info[key]);
  }

  function fIndex(D, f) {
    var best = 0;
    D.f.forEach(function (g, i) { if (Math.abs(g - f) < Math.abs(D.f[best] - f)) best = i; });
    return best;
  }

  /** Every series' score for one metric at elephant weight f:
   *  {elephant, pca, random, umap, tsne}, each {v, lo, hi, n} (UK.summary). */
  function elephantScores(D, dsKey, metric, f) {
    var s = D.datasets[dsKey].scores, i = fIndex(D, f);
    return {
      elephant: K.summary(s.elephant[metric][i]),
      pca: K.summary(s.elephant[metric][0]),
      random: K.summary(s.random[metric]),
      umap: K.summary(s.umap[metric]),
      tsne: K.summary(s.tsne[metric])
    };
  }

  // Metrics a dataset does not show, and why. The synthetic clusters have the
  // same spread by construction (unit sd, 200 points each), so a correlation
  // of within-cluster spreads measures only sampling noise.
  var HIDDEN = {
    clusters: { intra_corr: "Within-type spread is not shown here: the ten clusters have the same spread by construction." }
  };
  /** The metrics a dataset shows, in order. */
  function shownMetrics(D, dsKey) {
    var h = HIDDEN[dsKey] || {};
    return D.datasets[dsKey].metrics.filter(function (m) { return !h[m]; });
  }
  /** Why metrics are hidden for a dataset: [sentence]. */
  function hiddenNotes(dsKey) {
    var h = HIDDEN[dsKey] || {};
    return Object.keys(h).map(function (m) { return h[m]; });
  }

  /** The shared axis for a dataset's metric table: [lo, 1], lo a multiple of 0.25 below every shown score. */
  function domain(D, dsKey) {
    var s = D.datasets[dsKey].scores, lo = 0, shown = shownMetrics(D, dsKey);
    Object.keys(s).forEach(function (e) {
      Object.keys(s[e]).forEach(function (m) {
        if (shown.indexOf(m) < 0) return;
        [].concat(s[e][m]).forEach(function (v) { if (v !== null && v < lo) lo = v; });
      });
    });
    return [Math.floor(lo * 4) / 4, 1];
  }

  var cache = {};
  function decoded(D, dsKey, name) {
    var k = dsKey + "/" + name;
    if (!cache[k]) cache[k] = K.decode(D.datasets[dsKey].layouts[name]);
    return cache[k];
  }

  /** The layout the scatter shows: interleaved xy. */
  function elephantLayout(D, dsKey, view, f) {
    if (view === "elephant") return K.blend(decoded(D, dsKey, "pca"), decoded(D, dsKey, "elephant"), f);
    return decoded(D, dsKey, view);
  }

  /** A view extent that holds still while f moves: the union of PCA and the elephant. */
  function elephantExtent(D, dsKey, view) {
    if (view !== "elephant") return K.extent(decoded(D, dsKey, view));
    return K.extent(decoded(D, dsKey, "pca"), decoded(D, dsKey, "elephant"));
  }

  return { VIEWS: VIEWS, metricInfo: metricInfo, fIndex: fIndex, elephantScores: elephantScores,
    domain: domain, shownMetrics: shownMetrics, hiddenNotes: hiddenNotes,
    elephantLayout: elephantLayout, elephantExtent: elephantExtent };
})();

if (typeof module !== "undefined" && module.exports) module.exports = UMLModel;
