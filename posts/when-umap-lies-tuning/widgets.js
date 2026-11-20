/* Widgets for "When UMAP Lies, Part 3". Built on widget-kit (WK) and the
 * series kit (UK, ../when-umap-lies/umapkit.js); every number comes from
 * model.js (UMLUmapModel), and this file only draws.
 *
 *   widget-umap   UMAP's two settings: n_neighbors and min_dist sliders, a
 *                 seed toggle, the map beside the truth, local and global
 *                 tiles with their seed-to-seed range, and two heatmaps of
 *                 the grid medians (click a cell to go there)
 */
"use strict";

(function () {
  // Read the model when a widget mounts, not at load: if the page forgot
  // <script src="model.js">, WK.mount shows the error in the frame.
  var M = typeof UMLUmapModel !== "undefined" ? UMLUmapModel : null;
  function needModel() { if (!M) throw new Error("model.js did not load"); }

  WK.mount("widget-umap", function (root) {
    needModel();
    var U = WK.data("uml-umap");
    if (!U) throw new Error("widget-data/umap.js did not load");
    var state = { ds: "gaussian_clusters", nn: 15, md: 0.1, seed: 0, pinned: null };
    function D() { return U.datasets[state.ds]; }

    var fr = WK.frame({
      title: "UMAP's two settings",
      note: "UMAP on 1,000 points, three seeds at every setting. Every map is rotated, reflected and " +
        "scaled (Procrustes) onto the map at n_neighbors = 15, min_dist = 0.1, seed 0 of the same " +
        "dataset, so changing a setting does not spin the picture; alignment changes nothing about a " +
        "map's shape or its scores, and it cannot undo islands that trade places. The panels draw the " +
        "same 600 of the 1,000 points; the tiles score all 1,000. Local = share of each point's 10 true " +
        "neighbours kept; global = rank correlation of all pairwise distances, map against truth. Truth: " +
        "the 2-D PCA of the clean clusters, or the Swiss roll's unrolled sheet. The heatmaps show the " +
        "median of the three seeds; click a cell to go to that setting."
    });
    var dsCtl = WK.toggle({ label: "Data", value: state.ds,
      options: [{ value: "gaussian_clusters", label: "Gaussian clusters" }, { value: "swiss_roll", label: "Swiss roll" }],
      onchange: function (v) { state.ds = v; draw({ dataset: true }); } });
    var nnCtl = WK.slider({ label: "n_neighbors", min: 0, max: D().nn.length - 1, step: 1,
      value: D().nn.indexOf(state.nn), fmt: function (i) { return String(D().nn[i]); },
      oninput: function (i) { state.nn = D().nn[i]; draw({ animate: true }); } });
    var mdCtl = WK.slider({ label: "min_dist", min: 0, max: D().md.length - 1, step: 1,
      value: D().md.indexOf(state.md), fmt: function (i) { return String(D().md[i]); },
      oninput: function (i) { state.md = D().md[i]; draw({ animate: true }); } });
    var seedCtl = WK.toggle({ label: "Seed", value: state.seed,
      options: D().seeds.map(function (s) { return { value: s, label: String(s) }; }),
      onchange: function (v) { state.seed = v; draw({ animate: true }); } });
    [dsCtl, nnCtl, mdCtl, seedCtl].forEach(function (c) { fr.controls.appendChild(c.root); });

    // The map, and beside it the truth and the legend.
    var top = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "1rem 1.4rem", alignItems: "flex-start" } });
    var mainBox = WK.el("div", { style: { flex: "1 1 280px", maxWidth: "460px", minWidth: "0" } });
    var side = WK.el("div", { style: { flex: "1 1 180px", minWidth: "0" } });
    var truthBox = WK.el("div", { style: { maxWidth: "230px" } });
    var legendBox = WK.el("div", { style: { marginTop: "0.6rem" } });
    side.appendChild(truthBox); side.appendChild(legendBox);
    top.appendChild(mainBox); top.appendChild(side);
    var main = UK.scatter({ legend: false, directLabels: false, aspect: 0.85, title: "UMAP",
      label: "UMAP map", valueLabel: "s", onHover: function (k) { if (state.pinned === null) setHi(k); } });
    var truth = UK.scatter({ legend: false, directLabels: false, aspect: 0.85, title: "Truth",
      label: "Truth", valueLabel: "s", onHover: function (k) { if (state.pinned === null) setHi(k); } });
    mainBox.appendChild(main.root); truthBox.appendChild(truth.root);
    var legend = null;
    function setHi(k) {
      [main, truth].forEach(function (p) { p.highlight(k, true); });
      if (legend) legend.sync(k, state.pinned);
    }
    var heat = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "1rem 1.6rem", marginTop: "1rem" } });
    var hLocal = WK.el("div", { style: { flex: "1 1 260px", minWidth: "0" } });
    var hGlobal = WK.el("div", { style: { flex: "1 1 260px", minWidth: "0" } });
    heat.appendChild(hLocal); heat.appendChild(hGlobal);
    fr.body.appendChild(top); fr.body.appendChild(heat);
    var tLocal = WK.stat({ label: "", value: "", hint: " " });
    var tGlobal = WK.stat({ label: "", value: "", hint: " " });
    fr.stats.appendChild(tLocal.root); fr.stats.appendChild(tGlobal.root);
    root.appendChild(fr.root);

    function buildLegend() {
      WK.clear(legendBox);
      var c = M.colour(U, state.ds);
      state.pinned = null; legend = null;
      if (c.values) {
        var lo = Infinity, hi = -Infinity;
        for (var i = 0; i < c.values.length; i++) { lo = Math.min(lo, c.values[i]); hi = Math.max(hi, c.values[i]); }
        legendBox.appendChild(UK.rampLegend({ title: "Colour: arc length s along the roll",
          lo: lo, hi: hi, loLabel: UK.num(lo, 0), hiLabel: UK.num(hi, 0) }));
        return;
      }
      legend = UK.legend({ title: "Colour: cluster (n of all 1,000)", classes: c.classes, counts: c.counts, minItem: 110,
        onHover: function (k) { setHi(k === null ? state.pinned : k); },
        onPin: function (k) { state.pinned = k; main.pin(k); truth.pin(k); setHi(k); } });
      legendBox.appendChild(legend.root);
    }

    // A heatmap of the grid medians, laid out as the post's figure
    // (fig-umap-grid): columns n_neighbors, rows min_dist with 0.99 at the top.
    function heatmap(box, title, which) {
      var d = D(), g = M.grid(U, state.ds), w = box.clientWidth || 300;
      var LW = 34, TOP = 4, cw = Math.max(26, (w - LW) / d.nn.length), ch = 19;
      var H = TOP + d.md.length * ch + 32, W2 = LW + cw * d.nn.length;
      var vals = g.map(function (c) { return c[which]; });
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      var sv = WK.svg(W2, H, { width: W2, height: H, "aria-label": title });
      var rowOf = function (md) { return d.md.length - 1 - d.md.indexOf(md); };
      d.nn.forEach(function (nn, j) {
        sv.appendChild(WK.h("text", { x: LW + (j + 0.5) * cw, y: TOP + d.md.length * ch + 12, "text-anchor": "middle",
          "font-size": 9.5, fill: "muted" }, String(nn)));
      });
      d.md.forEach(function (md) {
        sv.appendChild(WK.h("text", { x: LW - 5, y: TOP + (rowOf(md) + 0.5) * ch + 3.5, "text-anchor": "end",
          "font-size": 9.5, fill: "muted" }, String(md)));
      });
      g.forEach(function (c) {
        var i = rowOf(c.md), j = d.nn.indexOf(c.nn), t = (c[which] - lo) / ((hi - lo) || 1);
        var pct = Math.round(12 + 80 * t), cur = c.nn === state.nn && c.md === state.md;
        var cell = WK.h("g", { style: { cursor: "pointer" },
          onclick: function () {
            state.nn = c.nn; state.md = c.md;
            nnCtl.set(d.nn.indexOf(c.nn), true); mdCtl.set(d.md.indexOf(c.md), true);
            draw({ animate: true });
          } });
        cell.appendChild(WK.h("rect", { x: LW + j * cw + 0.5, y: TOP + i * ch + 0.5, width: cw - 1, height: ch - 1,
          style: { fill: "color-mix(in srgb, " + WK.cssVar("c1") + " " + pct + "%, " + WK.cssVar("surface") + ")" } }));
        cell.appendChild(WK.h("text", { x: LW + (j + 0.5) * cw, y: TOP + (i + 0.5) * ch + 3.5, "text-anchor": "middle",
          "font-size": 9.5, fill: pct > 55 ? "surface" : "ink", style: { fontVariantNumeric: "tabular-nums" } },
          UK.num(c[which], 2)));
        cell.appendChild(WK.h("title", {}, "n_neighbors " + c.nn + ", min_dist " + c.md + ": median " + UK.num(c[which], 3)));
        sv.appendChild(cell);
        if (cur) sv.appendChild(WK.h("rect", { x: LW + j * cw + 0.5, y: TOP + i * ch + 0.5, width: cw - 1, height: ch - 1,
          fill: "none", stroke: "accent", "stroke-width": 2.2 }));
      });
      sv.appendChild(WK.h("text", { x: LW, y: H - 3, "font-size": 9.5, fill: "muted" },
        "columns n_neighbors, rows min_dist"));
      WK.clear(box);
      box.appendChild(WK.el("div", { class: "widget-control-head", style: { fontWeight: "600" } }, WK.el("span", { text: title })));
      box.appendChild(sv);
    }
    function heatmaps() {
      heatmap(hLocal, "Local: 10-NN kept (median of 3 seeds)", "local");
      heatmap(hGlobal, "Global: distance rank correlation (median of 3 seeds)", "global");
    }

    function draw(o) {
      var d = D(), c = M.colour(U, state.ds);
      if (o.dataset) {
        buildLegend();
        nnCtl.input.max = String(d.nn.length - 1); mdCtl.input.max = String(d.md.length - 1);
        nnCtl.set(d.nn.indexOf(state.nn), true); mdCtl.set(d.md.indexOf(state.md), true);
      }
      var s = M.setting(U, state.ds, state.nn, state.md, state.seed);
      [[main, M.layout(U, state.ds, state.nn, state.md, state.seed)], [truth, M.truth(U, state.ds)]].forEach(function (p, i) {
        var dd = { xy: p[1] };
        if (o.dataset) {
          p[0].pin(null);
          if (c.values) { dd.values = c.values; }
          else { dd.classes = c.classes; dd.counts = c.counts; dd.labels = c.labels; dd.values = null; }
        }
        if (i === 0) {
          dd.title = "UMAP: n_neighbors " + state.nn + ", min_dist " + state.md + ", seed " + state.seed;
          dd.caption = "local " + UK.num(s.local.v) + " · global " + UK.num(s.global.v);
        } else {
          dd.title = "Truth";
          dd.caption = state.ds === "swiss_roll" ? "the sheet unrolled (s, h)" : "clean data, first 2 PCs";
        }
        p[0].set(dd, { animate: o.animate && i === 0 ? 350 : 0 });
      });
      main.canvas.setAttribute("aria-label", "UMAP map of the " + d.title + ", n_neighbors " + state.nn +
        ", min_dist " + state.md + ", seed " + state.seed);
      tLocal.set(UK.num(s.local.v, 3));
      tLocal.root.querySelector(".widget-stat-label").textContent = "local: share of 10 true neighbours kept, seed " + state.seed;
      tLocal.root.querySelector(".widget-note").textContent = "3 seeds at this setting: " + UK.num(s.local.lo, 3) +
        "–" + UK.num(s.local.hi, 3);
      tGlobal.set(UK.num(s.global.v, 3));
      tGlobal.root.querySelector(".widget-stat-label").textContent = "global: distance rank correlation, seed " + state.seed;
      tGlobal.root.querySelector(".widget-note").textContent = "3 seeds at this setting: " + UK.num(s.global.lo, 3) +
        "–" + UK.num(s.global.hi, 3);
      heatmaps();
    }
    UK.onResize(heat, heatmaps);
    draw({ dataset: true });
  });
})();
