/* The numbers behind the widgets in "When UMAP Lies, Part 3".
 *
 * Kept apart from the drawing (widgets.js) so the prose can be checked against
 * it: `node posts/when-umap-lies/src/check_widgets.js umap` sweeps every
 * setting and prints what the widget shows. Exposes one global, `UMLUmapModel`,
 * and `module.exports` under node. Needs UK (../when-umap-lies/umapkit.js) and
 * reads WK_DATA["uml-umap"] (widget-data/umap.js, written by
 * posts/when-umap-lies/src/export_widget_data.py).
 *
 * The experiment (cache/umap_widget): UMAP on 1,000 points of two datasets at
 * n_neighbors x min_dist, three method seeds each. "Local" is k = 10
 * neighbour preservation against the truth; "global" is the Spearman rank
 * correlation of 50,000 sampled pairwise distances, map against truth. Every map is
 * Procrustes-aligned to the map at (15, 0.1, seed 0) of its dataset.
 */
"use strict";

var UMLUmapModel = (function () {
  var K = typeof UK !== "undefined" ? UK : require("../when-umap-lies/umapkit.js");

  function key(nn, md) { return nn + "/" + String(md); }

  /** {local: {v, lo, hi, n, seeds}, global: {...}} for one setting, v = the value at `seed`
   *  (or the median over seeds when seed is null), lo/hi = the range over seeds. */
  function setting(U, ds, nn, md, seed) {
    var d = U.datasets[ds], k = key(nn, md);
    if (!d.q10[k]) throw new Error("no setting " + ds + " " + k);
    function one(arr) {
      var s = K.summary(arr, "median"), i = seed === null || seed === undefined ? -1 : d.seeds.indexOf(seed);
      if (i >= 0) s.v = arr[i];
      s.seeds = arr.slice();
      return s;
    }
    return { local: one(d.q10[k]), global: one(d.dsp[k]) };
  }

  /** Grid of medians over seeds: [{nn, md, local, global}] for the heatmaps. */
  function grid(U, ds) {
    var d = U.datasets[ds], out = [];
    d.nn.forEach(function (nn) {
      d.md.forEach(function (md) {
        out.push({ nn: nn, md: md, local: K.median(d.q10[key(nn, md)]), global: K.median(d.dsp[key(nn, md)]) });
      });
    });
    return out;
  }

  var cache = {};
  /** The aligned map at one setting and seed: interleaved xy. */
  function layout(U, ds, nn, md, seed) {
    var d = U.datasets[ds], c = ds + "/" + key(nn, md) + "/" + seed;
    if (!cache[c]) cache[c] = K.decode(d.maps[key(nn, md)][d.seeds.indexOf(seed)]);
    return cache[c];
  }
  function truth(U, ds) {
    var c = ds + "/truth";
    if (!cache[c]) cache[c] = K.decode(U.datasets[ds].truth);
    return cache[c];
  }
  function colour(U, ds) {
    var c = U.datasets[ds].colour, k = ds + "/colour";
    if (!cache[k]) {
      cache[k] = c.kind === "values" ? { values: K.decodeValues(c.values), label: c.label } :
        { labels: K.decodeLabels(c.labels), classes: c.classes, counts: c.counts, label: c.label };
    }
    return cache[k];
  }

  return { key: key, setting: setting, grid: grid, layout: layout, truth: truth, colour: colour };
})();

if (typeof module !== "undefined" && module.exports) module.exports = UMLUmapModel;
