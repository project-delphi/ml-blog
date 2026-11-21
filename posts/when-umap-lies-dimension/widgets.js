/* Widgets for "When UMAP Lies, Part 2". Built on widget-kit (WK) and the
 * series kit (UK, ../when-umap-lies/umapkit.js); every number comes from
 * model.js (UMLDimModel), and this file only draws.
 *
 *   widget-dimension   pure-noise columns added to a fixed structure: the
 *                      truth, four maps at the chosen D, and k = 10 neighbour
 *                      preservation against D for the maps and their inputs
 *   widget-signal      six clusters with d real dimensions, padded with noise:
 *                      four controls (d, D, sigma, n), truth and three maps,
 *                      one draw per setting
 */
"use strict";

(function () {
  // Read the model when a widget mounts, not at load: if the page forgot
  // <script src="model.js">, WK.mount shows the error in the frame.
  var M = typeof UMLDimModel !== "undefined" ? UMLDimModel : null;
  function needModel() { if (!M) throw new Error("model.js did not load"); }

  function fmtD(D) { return D.toLocaleString("en-US"); }

  WK.mount("widget-dimension", function (root) {
    needModel();
    var Data = WK.data("uml-dimension");
    if (!Data) throw new Error("widget-data/dimension.js did not load");
    var state = { ds: "swiss_roll", i: 0, hi: null, pinned: null };
    function dsObj() { return Data.datasets[state.ds]; }
    function curD() { return dsObj().D[state.i]; }

    var fr = WK.frame({
      title: "Adding columns of pure noise",
      note: "The truth panel has no noise columns and is the same at every D. Each other panel is " +
        "one run (seed 0), drawn on the same 1,000 of the 2,000 points. A 2-D map's " +
        "orientation, reflection and scale are arbitrary, so every map was rotated, reflected and " +
        "scaled (Procrustes) onto the truth panel before drawing; nothing else was changed. The " +
        "chart scores each map, and the 10-NN graphs of the raw input and of its first 50 principal " +
        "components (dotted; PCA-50 exists only once D > 50), by the share of each point's 10 true " +
        "neighbours they keep: median of five independent draws, band = min to max. η is the " +
        "noise index, 1 at D − p = 300."
    });
    var dsCtl = WK.toggle({ label: "Data",
      options: [{ value: "swiss_roll", label: "Swiss roll" }, { value: "gaussian_clusters", label: "Gaussian clusters" }],
      value: state.ds, onchange: function (v) {
        var D = curD();
        state.ds = v;
        // keep the reader's D where the other base has it, else the nearest
        var Ds = dsObj().D, j = Ds.indexOf(D);
        state.i = j >= 0 ? j : Ds.reduce(function (b, x, k) {
          return Math.abs(Math.log(x / D)) < Math.abs(Math.log(Ds[b] / D)) ? k : b; }, 0);
        dSlider.input.max = String(Ds.length - 1);
        dSlider.set(state.i, true);
        draw({ dataset: true });
      } });
    var dSlider = WK.slider({ label: "Total dimension D", min: 0, max: dsObj().D.length - 1, step: 1, value: 0,
      fmt: function (i) {
        var D = dsObj().D[i];
        return "D = " + fmtD(D) + " · η = " + UK.num(M.eta(Data, state.ds, D), 2);
      },
      oninput: function (i) { state.i = i; draw({ animate: true }); } });
    fr.controls.appendChild(dsCtl.root);
    fr.controls.appendChild(dSlider.root);

    // The five panels: truth, then the four maps.
    var grid = WK.el("div", { style: { display: "grid", gap: "0.9rem 0.9rem",
      gridTemplateColumns: "repeat(auto-fill, minmax(124px, 1fr))" } });
    var PANELS = [{ key: "truth", label: "Truth" }].concat(M.SERIES.map(function (s) {
      return { key: s.key, label: s.label };
    }));
    var panels = PANELS.map(function (p) {
      var sc = UK.scatter({ legend: false, directLabels: false, aspect: 0.8, title: p.label,
        valueLabel: "s", label: p.label,
        onHover: function (k) { if (state.pinned === null) setHi(k); } });
      grid.appendChild(sc.root);
      return { key: p.key, label: p.label, sc: sc };
    });
    var legendBox = WK.el("div", { style: { marginTop: "0.7rem" } });
    var legend = null;
    function setHi(k) {
      state.hi = k;
      panels.forEach(function (p) { p.sc.highlight(k, true); });
      if (legend && legend.sync) legend.sync(k, state.pinned);
    }

    var chartHead = WK.el("div", { class: "widget-control-head", style: { marginTop: "1.1rem", fontWeight: "600" } },
      WK.el("span", { text: "Share of the 10 true neighbours kept (k = 10), against D" }));
    var chart = UK.lineChart({ series: [], points: function () { return []; }, yDomain: [0, 1] });
    fr.body.appendChild(grid);
    fr.body.appendChild(legendBox);
    fr.body.appendChild(chartHead);
    fr.body.appendChild(chart.root);

    var tiles = ["input", "umap", "tsne", "secs"].map(function () {
      var t = WK.stat({ label: "", value: "", hint: " " });
      fr.stats.appendChild(t.root);
      return t;
    });
    function tile(t, value, label, hint) {
      t.set(value);
      t.root.querySelector(".widget-stat-label").textContent = label;
      t.root.querySelector(".widget-note").textContent = hint;
    }
    root.appendChild(fr.root);

    function buildLegend() {
      WK.clear(legendBox);
      var c = M.colour(Data, state.ds);
      state.hi = null; state.pinned = null;
      if (c.values) {
        var lo = Infinity, hi = -Infinity;
        for (var i = 0; i < c.values.length; i++) { lo = Math.min(lo, c.values[i]); hi = Math.max(hi, c.values[i]); }
        legend = null;
        legendBox.appendChild(UK.rampLegend({ title: "Colour: arc length s along the roll (the truth panel's x)",
          lo: lo, hi: hi, loLabel: UK.num(lo, 0), hiLabel: UK.num(hi, 0) }));
        return;
      }
      legend = UK.legend({ title: "Colour: cluster (hover or tap to pick one out in every panel)",
        classes: c.classes, counts: c.counts, minItem: 120,
        onHover: function (k) { setHi(k === null ? state.pinned : k); },
        onPin: function (k) {
          state.pinned = k;
          panels.forEach(function (p) { p.sc.pin(k); });
          setHi(k);
        } });
      legendBox.appendChild(legend.root);
    }

    function draw(o) {
      var ds = dsObj(), D = curD(), c = M.colour(Data, state.ds);
      if (o.dataset) buildLegend();
      panels.forEach(function (p) {
        var xy = p.key === "truth" ? M.truth(Data, state.ds) : M.layout(Data, state.ds, p.key, D);
        var d = { xy: xy, title: p.label };
        if (o.dataset) {
          p.sc.pin(null);
          if (c.values) { d.classes = null; d.values = c.values; }
          else { d.classes = c.classes; d.counts = c.counts; d.labels = c.labels; d.values = null; }
        }
        if (p.key === "truth") d.caption = state.ds === "swiss_roll" ? "the sheet unrolled (s, h)" : "clean data, 2 PCs";
        else d.caption = "this run: " + UK.num(M.q10(Data, state.ds, p.key, D).seed0) + " kept";
        p.sc.set(d, { animate: o.animate && p.key !== "truth" ? 350 : 0 });
        p.sc.canvas.setAttribute("aria-label", d.title + " at D = " + fmtD(D));
      });

      var ticks = [ds.D[0], 10, 100, 1000, 5000].filter(function (x, k, a) {
        return a.indexOf(x) === k && x >= ds.D[0]; });
      chart.update({
        series: M.SERIES.map(function (s) {
          return { key: s.key, label: s.label, token: s.token, dash: s.dash, marker: s.marker }; })
          .concat(M.GRAPHS.map(function (g) { return { key: "graph:" + g.key, label: g.label, token: g.token,
            dash: g.dash, dots: false }; })),
        points: function (k) {
          return k.indexOf("graph:") === 0 ? M.curve(Data, state.ds, "graph", k.slice(6)) :
            M.curve(Data, state.ds, "map", k);
        },
        xLog: true, xTicks: ticks, xFormat: fmtD, xLabel: "total dimension D (log scale)",
        yDomain: [0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], marker: D, markerLabel: "D = " + fmtD(D),
        label: "k = 10 neighbour preservation against D for " + ds.title
      });

      var inp = M.graph(Data, state.ds, "input", D), um = M.q10(Data, state.ds, "umap", D),
        ts = M.q10(Data, state.ds, "tsne", D), secs = M.seconds(Data, state.ds, D);
      function rangeHint(s) { return "median of 5 draws; range " + UK.num(s.lo) + "–" + UK.num(s.hi); }
      tile(tiles[0], UK.num(inp.v), "input kNN graph, share of 10 true neighbours kept", rangeHint(inp));
      tile(tiles[1], UK.num(um.v), "UMAP map, share kept", rangeHint(um));
      tile(tiles[2], UK.num(ts.v), "t-SNE map, share kept", rangeHint(ts));
      tile(tiles[3], UK.num(secs.v, 1) + " s", "UMAP fit time, median",
        "range " + UK.num(secs.lo, 1) + "–" + UK.num(secs.hi, 1) + " s, n = " + ds.n.toLocaleString("en-US"));
    }
    draw({ dataset: true });
  });

  // ------------------------------------------------------------ widget-signal
  WK.mount("widget-signal", function (root) {
    needModel();
    var S = WK.data("uml-signal");
    if (!S) throw new Error("widget-data/signal.js did not load");
    var G = S.grid;
    var state = { d: 2, D: 1000, sigma: 0, n: 2000, pinned: null };

    var fr = WK.frame({
      title: "Noise columns versus real dimensions",
      note: "Controlled synthetic experiments, not predictions about real data: six Gaussian " +
        "clusters with d real dimensions (unit sd, nearest centres 6 sd apart), padded with " +
        "pure-noise columns of sd σ to D columns in all. Each setting is ONE draw (data seed 0), " +
        "so it illustrates rather than estimates: differences of a few hundredths between settings " +
        "are within draw-to-draw variation. The truth panel is the 2-D PCA of the clean clusters; " +
        "each map was rotated, reflected and scaled (Procrustes) onto it, since a map's orientation " +
        "is arbitrary. At n = 2,000 the panels show the same 500 points (so both sizes draw equally " +
        "dense panels); the scores use all 2,000. " +
        "η = σ²√(8(D − d)) / r₁₀², with r₁₀² the median squared " +
        "distance to the 10th neighbour in the clean data."
    });
    function ctl(label, key, values, fmt) {
      var t = WK.toggle({ label: label, value: state[key],
        options: values.map(function (v) { return { value: v, label: fmt ? fmt(v) : String(v) }; }),
        onchange: function (v) { state[key] = v; draw({ animate: true, base: key === "d" || key === "n" }); } });
      t.root.style.flex = "1 1 170px";
      fr.controls.appendChild(t.root);
      return t;
    }
    ctl("Real dimensions d", "d", G.d);
    ctl("Total columns D", "D", G.D, fmtD);
    ctl("Noise sd σ", "sigma", G.sigma);
    ctl("Points n", "n", G.n, fmtD);
    var readout = WK.el("div", { class: "widget-control-head", style: { marginBottom: "0.6rem" } });
    fr.body.appendChild(readout);

    var grid = WK.el("div", { style: { display: "grid", gap: "0.9rem",
      gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))" } });
    var panels = [{ key: "truth", label: "Truth" }].concat(M.SIGNAL_METHODS.map(function (m) {
      return { key: m.key, label: m.label };
    })).map(function (p) {
      var sc = UK.scatter({ legend: false, directLabels: false, aspect: 0.85, title: p.label, label: p.label,
        onHover: function (k) { if (state.pinned === null) setHi(k); } });
      grid.appendChild(sc.root);
      return { key: p.key, label: p.label, sc: sc };
    });
    fr.body.appendChild(grid);
    var legendBox = WK.el("div", { style: { marginTop: "0.7rem" } });
    fr.body.appendChild(legendBox);
    var legend = null;
    function setHi(k) {
      panels.forEach(function (p) { p.sc.highlight(k, true); });
      if (legend) legend.sync(k, state.pinned);
    }

    var tiles = M.SIGNAL_METHODS.concat([{ key: "input", label: "Input kNN graph" }]).map(function (m) {
      var t = WK.stat({ label: "", value: "", hint: " " });
      fr.stats.appendChild(t.root);
      return { key: m.key, label: m.label, t: t };
    });
    root.appendChild(fr.root);

    function draw(o) {
      var b = M.signalBase(S, state.d, state.n), c = M.signal(S, state.d, state.D, state.sigma, state.n);
      var tr = M.signalTruth(S, state.d, state.n), first = !legend;
      if (first || o.base) {
        state.pinned = null;
        WK.clear(legendBox);
        legend = UK.legend({ title: "Colour: cluster (hover or tap to pick one out in every panel)",
          classes: tr.classes, counts: tr.counts, minItem: 120,
          onHover: function (k) { setHi(k === null ? state.pinned : k); },
          onPin: function (k) { state.pinned = k; panels.forEach(function (p) { p.sc.pin(k); }); setHi(k); } });
        legendBox.appendChild(legend.root);
      }
      var noise = state.D - state.d;
      readout.textContent = "η = " + UK.num(c.eta, 2) + " · " +
        (noise ? fmtD(noise) + " noise columns" : "no noise columns (D = d), so σ has no effect") +
        " · showing " + fmtD(b.shown) + " of " + fmtD(state.n) + " points";
      panels.forEach(function (p) {
        var d = { xy: p.key === "truth" ? tr.xy : M.signalLayout(S, state.d, state.D, state.sigma, state.n, p.key) };
        if (first || o.base) { d.classes = tr.classes; d.counts = tr.counts; d.labels = tr.labels; d.values = null; }
        d.caption = p.key === "truth" ? "clean data, first 2 PCs" :
          "kept " + UK.num(c.q10[p.key]) + " · class acc. " + UK.num(c.acc[p.key]);
        p.sc.set(d, { animate: o.animate && !o.base && p.key !== "truth" ? 350 : 0 });
        p.sc.canvas.setAttribute("aria-label", p.label + ", d = " + state.d + ", D = " + state.D +
          ", sigma = " + state.sigma + ", n = " + state.n);
      });
      tiles.forEach(function (x) {
        x.t.set(UK.num(c.q10[x.key]));
        x.t.root.querySelector(".widget-stat-label").textContent = x.key === "input" ?
          "input kNN graph: share of 10 true neighbours kept" : x.label + ": share of 10 true neighbours kept";
        x.t.root.querySelector(".widget-note").textContent = x.key === "input" ?
          "10-NN graph of all D columns, before any map; η = " + UK.num(c.eta, 2) :
          "trustworthiness " + UK.num(c.tw10[x.key]) + " · 15-NN class accuracy " + UK.num(c.acc[x.key]);
      });
    }
    draw({});
  });
})();
