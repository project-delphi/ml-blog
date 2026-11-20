/* Widgets for "When UMAP Lies, Part 1". Built on widget-kit (WK) and the
 * series kit (UK, umapkit.js); every number comes from model.js (UMLModel),
 * and this file only draws.
 *
 *   widget-elephant   the elephant in the spirit of Picasso: a slider blends
 *                     2-D PCA into an elephant, and a table scores it, the
 *                     random elephant, UMAP and t-SNE with both papers' metrics
 */
"use strict";

(function () {
  // Read the model when a widget mounts, not at load: if the page forgot
  // <script src="model.js">, WK.mount shows the error in the frame.
  var M = typeof UMLModel !== "undefined" ? UMLModel : null;
  function needModel() { if (!M) throw new Error("model.js did not load"); }

  WK.mount("widget-elephant", function (root) {
    needModel();
    var D = WK.data("uml-elephant");
    if (!D) throw new Error("widget-data/elephant.js did not load");
    // Opens on the synthetic clusters at f = 1, the whole elephant: the post
    // tells readers to start there and drag f from 0 to 1.
    var state = { ds: "clusters", f: 1, view: "elephant" };
    var DATASETS = [{ value: "clusters", label: "Synthetic clusters" }, { value: "pbmc3k", label: "PBMC 3k" }];

    var fr = WK.frame({
      title: "An elephant, scored two ways",
      note: "In the spirit of Chari & Pachter's Picasso, not a reproduction of it: this elephant is " +
        "2-D PCA blended, by weight f, into an optimal assignment of the points to points sampled " +
        "uniformly inside a filled elephant silhouette (Picasso used the outline), so f = 0 is exactly " +
        "PCA. The random elephant assigns the same silhouette points at random. " +
        "UMAP and t-SNE marks are the mean of five seeds with the min–max range as whiskers; the " +
        "scatter shows seed 0. Hover or tap a legend entry to pick out one group."
    });
    var dsCtl = WK.toggle({ label: "Data", options: DATASETS, value: state.ds,
      onchange: function (v) { state.ds = v; draw({ dataset: true }); } });
    var fCtl = WK.slider({ label: "Elephant weight f", min: 0, max: 1, step: 0.1, value: state.f,
      fmt: function (v) { return v.toFixed(1) + (v === 0 ? " (PCA)" : v === 1 ? " (all elephant)" : ""); },
      oninput: function (v) {
        state.f = Math.round(v * 10) / 10;
        if (state.view !== "elephant") { state.view = "elephant"; viewCtl.set("elephant", true); }
        draw({});
      } });
    var viewCtl = WK.toggle({ label: "Scatter shows",
      options: M.VIEWS.map(function (v) { return { value: v.key, label: v.label }; }), value: state.view,
      onchange: function (v) { state.view = v; draw({ animate: true }); } });
    [dsCtl, fCtl, viewCtl].forEach(function (c) { fr.controls.appendChild(c.root); });
    UK.fitToggle(viewCtl, ["Elephant", "Random", "UMAP", "t-SNE"]);

    var sc = UK.scatter({ label: "Scatter of the chosen embedding, coloured by group", legendTitle: "" });
    var heads = {}, tables = {};
    var panel = WK.el("div", { style: { marginTop: "1.1rem" } });
    // Group headings, singular and plural: a dataset can hide a metric.
    var GROUPS = { chari: ["Chari & Pachter’s distance metric", "Chari & Pachter’s distance metrics"],
      lause: ["Lause et al.’s neighbourhood and class metric", "Lause et al.’s neighbourhood and class metrics"] };
    var hiddenNote = WK.el("p", { class: "widget-note", style: { marginTop: "0.2rem" } });
    ["chari", "lause"].forEach(function (g) {
      heads[g] = WK.el("div", { class: "widget-control-head", style: { marginTop: "0.6rem", fontWeight: "600" } },
        WK.el("span", { text: GROUPS[g][1] }));
      tables[g] = UK.dotTable({ rows: [], columns: [], domain: [0, 1], value: function () { return null; } });
      panel.appendChild(heads[g]); panel.appendChild(tables[g].root);
      if (g === "chari") panel.appendChild(hiddenNote);
    });
    fr.body.appendChild(sc.root);
    fr.body.appendChild(panel);

    // Tiles per dataset; the clusters hide within-type spread (model.js HIDDEN).
    var TILE_METRICS = { pbmc3k: ["inter_corr_large", "intra_corr", "knn_recall", "max_ami"],
      clusters: ["inter_corr", "knn_recall", "max_ami"] };
    var tiles = [0, 1, 2, 3].map(function () {
      var t = WK.stat({ label: "", value: "", hint: " " });
      fr.stats.appendChild(t.root);
      return t;
    });
    root.appendChild(fr.root);

    function rows() {
      var f = state.f.toFixed(1);
      // Methods take the series-wide style (UK.METHOD); the two elephants are
      // not methods, so they get shapes and neutrals no method uses (accent
      // would collide: it is c1, UMAP's colour, in both themes).
      function m(key, label) {
        var s = UK.method(key);
        return { key: key, label: label || s.name, mark: s.marker, token: s.token };
      }
      return [
        { key: "elephant", label: "Elephant, f = " + f, mark: "hexagon", token: "ink", emph: true },
        { key: "random", label: "Random elephant", mark: "ring", token: "muted" },
        m("pca", "PCA (f = 0)"), m("umap"), m("tsne")
      ];
    }

    function draw(o) {
      var ds = D.datasets[state.ds], fi = M.fIndex(D, state.f);
      var xy = M.elephantLayout(D, state.ds, state.view, D.f[fi]);
      var viewLabel = state.view === "elephant" ? "Elephant at f = " + state.f.toFixed(1) +
        (state.f === 0 ? ", which is 2-D PCA" : "") : M.VIEWS.filter(function (v) { return v.key === state.view; })[0].label +
        (state.view === "random" ? "" : ", seed 0");
      var d = { xy: xy, extent: M.elephantExtent(D, state.ds, state.view),
        caption: viewLabel + " · " + ds.n.toLocaleString("en-US") + " points" };
      if (o.dataset || !sc.state().n) {
        d.classes = ds.classes; d.counts = ds.counts; d.labels = UK.decodeLabels(ds.labels);
        d.legendTitle = ds.noun === "cluster" ? "Cluster" : "Cell type";
      }
      sc.set(d, { animate: o.animate ? 450 : 0 });
      sc.canvas.setAttribute("aria-label", viewLabel + ", coloured by " + ds.noun);

      var dom = M.domain(D, state.ds), r = rows(), notes = M.hiddenNotes(state.ds);
      hiddenNote.textContent = notes.join(" ");
      hiddenNote.style.display = notes.length ? "block" : "none";
      ["chari", "lause"].forEach(function (g) {
        var cols = M.shownMetrics(D, state.ds).map(function (m) { return M.metricInfo(m, ds); })
          .filter(function (mi) { return mi.group === g; });
        heads[g].firstChild.textContent = GROUPS[g][cols.length === 1 ? 0 : 1];
        tables[g].update({ rows: r, domain: dom, label: heads[g].textContent,
          columns: cols.map(function (mi) { return { key: mi.key, label: mi.short, title: mi.title }; }),
          value: function (rowKey, colKey) {
            var s = M.elephantScores(D, state.ds, colKey, state.f)[rowKey];
            var row = r.filter(function (x) { return x.key === rowKey; })[0];
            var mi = M.metricInfo(colKey, ds);
            s.title = row.label + ", " + mi.tile + ": " + UK.num(s.v, 3) +
              (s.n > 1 ? " (mean of " + s.n + " seeds, " + UK.num(s.lo, 3) + " to " + UK.num(s.hi, 3) + ")" : "");
            return s;
          } });
      });

      tiles.forEach(function (t, i) { t.root.style.display = i < TILE_METRICS[state.ds].length ? "" : "none"; });
      TILE_METRICS[state.ds].forEach(function (m, i) {
        var s = M.elephantScores(D, state.ds, m, state.f), mi = M.metricInfo(m, ds);
        tiles[i].set(UK.num(s.elephant.v));
        tiles[i].root.querySelector(".widget-stat-label").textContent = "Elephant: " + mi.tile;
        tiles[i].root.querySelector(".widget-note").textContent =
          "UMAP " + UK.num(s.umap.v) + " · t-SNE " + UK.num(s.tsne.v) + " · random " + UK.num(s.random.v);
      });
    }
    draw({ dataset: true });
  });
})();
