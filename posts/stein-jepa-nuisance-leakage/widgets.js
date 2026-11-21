/* Widgets for "Nuisance Leakage in JEPA Embeddings": the lambda sweep and the
 * configuration explorer. Both read the study's held-out summary (widget-data/
 * data.js, written by src/make_data.py) through model.js. Built on widget-kit
 * (WK); colours are theme tokens.
 */
"use strict";

WK.mount("widget-lambda", function (root, WK) {
  var M = window.SteinModel;
  var on = {};
  M.LAM_ARMS.forEach(function (a) { on[a] = true; });
  var f = WK.frame({
    title: "Leakage and content against regulariser weight",
    note: "Held-out seeds, mean and 95% interval. Dotted line: the tuned weight."
  });
  var checks = M.LAM_ARMS.map(function (a) {
    var c = WK.check({ label: a, value: true, onchange: function (v) { on[a] = v; draw(); } });
    c.root.insertBefore(WK.el("span", { class: "swatch", "aria-hidden": "true",
      style: { display: "inline-block", width: "0.7em", height: "0.7em", borderRadius: "50%",
               background: WK.cssVar(M.ARM_TOKEN[a]), marginRight: "0.3em" } }), c.root.firstChild);
    f.controls.appendChild(c.root);
    return c;
  });
  var W = 640, H = 300, svg = WK.svg(W, H, { "aria-label": "Linear leakage and content R-squared against lambda for six regularisers" });
  f.body.appendChild(svg);
  root.appendChild(f.root);

  var PANELS = [
    { key: "leak_r2", label: "linear leakage R² (lower is better)" },
    { key: "content_r2", label: "linear content R² (higher is better)" }
  ];
  var LO = Math.log(0.2), HI = Math.log(25);

  function draw() {
    WK.clear(svg);
    var left = 44, gap = 34, pw = (W - left - 14 - gap) / 2, top = 30, bot = 48, ph = H - top - bot;
    var shown = M.LAM_ARMS.filter(function (a) { return on[a]; });
    var all = [];
    shown.forEach(function (a) { M.lamSeries(window.WK_DATA, a).forEach(function (r) { all.push(r); }); });
    var ymin = 0;
    all.forEach(function (r) { ymin = Math.min(ymin, r.leak_r2_lo, r.content_r2_lo); });
    ymin = Math.floor(ymin * 10) / 10;
    PANELS.forEach(function (p, i) {
      var x0 = left + i * (pw + gap);
      var sx = WK.lin([LO, HI], [x0, x0 + pw]), sy = WK.lin([ymin, 1], [top + ph, top]);
      svg.appendChild(WK.h("text", { x: x0, y: top - 12, fill: "ink", "font-size": 12, text: p.label }));
      WK.ticks(ymin, 1, 5).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: x0, x2: x0 + pw, y1: sy(t), y2: sy(t), stroke: "rule", "stroke-width": 1 }));
        svg.appendChild(WK.h("text", { x: x0 - 6, y: sy(t) + 3, "text-anchor": "end", fill: "muted", "font-size": 11, text: WK.fmt.num(t, 1) }));
      });
      [[0.2, "1/5"], [1, "1"], [5, "5"], [25, "25"]].forEach(function (t) {
        var x = sx(Math.log(t[0]));
        svg.appendChild(WK.h("text", { x: x, y: top + ph + 14, "text-anchor": "middle", fill: "muted", "font-size": 11, text: t[1] }));
      });
      svg.appendChild(WK.h("line", { x1: sx(0), x2: sx(0), y1: top, y2: top + ph, stroke: "muted", "stroke-dasharray": "2,3" }));
      svg.appendChild(WK.h("text", { x: x0 + pw / 2, y: top + ph + 32, "text-anchor": "middle", fill: "muted", "font-size": 11, text: "λ ÷ tuned λ" }));
      shown.forEach(function (a) {
        var tok = M.ARM_TOKEN[a], jit = (M.LAM_ARMS.indexOf(a) - 2.5) / 25;
        var rows = M.lamSeries(window.WK_DATA, a);
        var pts = rows.map(function (r) { return [sx(Math.log(r.lam_rel) + jit), sy(r[p.key]), r]; });
        svg.appendChild(WK.h("polyline", { points: pts.map(function (q) { return q[0] + "," + q[1]; }).join(" "),
          fill: "none", stroke: tok, "stroke-width": 1.8 }));
        pts.forEach(function (q) {
          var r = q[2];
          svg.appendChild(WK.h("line", { x1: q[0], x2: q[0], y1: sy(r[p.key + "_lo"]), y2: sy(r[p.key + "_hi"]), stroke: tok, "stroke-width": 1.2 }));
          svg.appendChild(WK.h("circle", { cx: q[0], cy: q[1], r: 3.5, fill: tok },
            WK.h("title", { text: a + ", λ = " + r.lam + "\n" + p.label + ": " + WK.fmt.num(r[p.key], 3) +
              " [" + WK.fmt.num(r[p.key + "_lo"], 3) + ", " + WK.fmt.num(r[p.key + "_hi"], 3) + "]" })));
        });
      });
    });
    if (!shown.length) svg.appendChild(WK.h("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "muted", text: "Select at least one regulariser." }));
  }
  draw();
});

WK.mount("widget-explorer", function (root, WK) {
  var M = window.SteinModel;
  var st = { d: 12, sigma: 0, alpha: 1, knn: false, arms: {} };
  M.ARM_ORDER.forEach(function (a) { st.arms[a] = a !== "none"; });
  var f = WK.frame({
    title: "Leakage and content at the tuned weight",
    note: "Mean and 95% interval over 5 seeds. A setting that was not trained shows “not run”; nothing is interpolated."
  });
  function opts(vals) { return vals.map(function (v) { return { value: v, label: String(v) }; }); }
  f.controls.appendChild(WK.toggle({ label: "Embedding dimension d", options: opts([3, 6, 12, 24]), value: 12,
    onchange: function (v) { st.d = v; draw(); } }).root);
  f.controls.appendChild(WK.toggle({ label: "Noise σ", options: opts([0, 0.5, 1]), value: 0,
    onchange: function (v) { st.sigma = v; draw(); } }).root);
  f.controls.appendChild(WK.toggle({ label: "Nuisance strength α", options: opts([0, 0.5, 1]), value: 1,
    onchange: function (v) { st.alpha = v; draw(); } }).root);
  M.ARM_ORDER.forEach(function (a) {
    var c = WK.check({ label: a, value: st.arms[a], onchange: function (v) { st.arms[a] = v; draw(); } });
    c.root.insertBefore(WK.el("span", { "aria-hidden": "true",
      style: { display: "inline-block", width: "0.7em", height: "0.7em", borderRadius: "50%",
               background: WK.cssVar(M.ARM_TOKEN[a]), marginRight: "0.3em" } }), c.root.firstChild);
    f.controls.appendChild(c.root);
  });
  f.controls.appendChild(WK.check({ label: "Add kNN leakage", value: false,
    onchange: function (v) { st.knn = v; draw(); } }).root);
  var holder = WK.el("div"), tableHolder = WK.el("div", { style: { overflowX: "auto", marginTop: "0.6rem" } });
  f.body.appendChild(holder); f.body.appendChild(tableHolder);
  root.appendChild(f.root);

  function ci(r, m) {
    return WK.fmt.num(r[m], 3) + " [" + WK.fmt.num(r[m + "_lo"], 3) + ", " + WK.fmt.num(r[m + "_hi"], 3) + "]";
  }

  function draw() {
    WK.clear(holder); WK.clear(tableHolder);
    var where = "d = " + st.d + ", σ = " + st.sigma + ", α = " + st.alpha;
    var cellRows = M.cell(window.WK_DATA, st.d, st.sigma, st.alpha);
    if (!cellRows.length) {
      holder.appendChild(WK.el("p", { class: "widget-note", html: "<strong>Not run.</strong> No arm was trained at " + where +
        ". Move one setting back toward the base (d = 12, σ = 0, α = 1)." }));
      return;
    }
    var rows = cellRows.filter(function (r) { return st.arms[r.arm]; });
    var missing = M.ARM_ORDER.filter(function (a) {
      return st.arms[a] && !cellRows.some(function (r) { return r.arm === a; });
    });
    if (missing.length) holder.appendChild(WK.el("p", { class: "widget-note", text: "Not run at " + where + ": " + missing.join(", ") + "." }));
    if (!rows.length) return;
    var panels = [
      { key: "leak_r2", label: "linear leakage R² (lower is better)" },
      { key: "content_r2", label: "linear content R² (higher is better)" }
    ];
    if (st.knn) panels.push({ key: "leak_knn_r2", label: "kNN leakage R² (lower is better)" });
    var W = 640, left = 76, gap = 22, rowH = 26, top = 34, bot = 26;
    var H = top + rowH * rows.length + bot;
    var pw = (W - left - 10 - gap * (panels.length - 1)) / panels.length;
    var xmin = 0;
    rows.forEach(function (r) { panels.forEach(function (p) { if (r[p.key + "_lo"] !== null) xmin = Math.min(xmin, r[p.key + "_lo"]); }); });
    xmin = Math.floor(xmin * 10) / 10;
    var svg = WK.svg(W, H, { "aria-label": "Leakage and content R-squared for each regulariser at " + where });
    panels.forEach(function (p, i) {
      var x0 = left + i * (pw + gap), sx = WK.lin([xmin, 1], [x0, x0 + pw]);
      svg.appendChild(WK.h("text", { x: x0, y: top - 16, fill: "ink", "font-size": 12, text: p.label }));
      WK.ticks(xmin, 1, 4).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: sx(t), x2: sx(t), y1: top - 6, y2: top + rowH * rows.length - 6, stroke: "rule", "stroke-width": 1 }));
        svg.appendChild(WK.h("text", { x: sx(t), y: H - 8, "text-anchor": "middle", fill: "muted", "font-size": 11, text: WK.fmt.num(t, 1) }));
      });
      rows.forEach(function (r, j) {
        var y = top + rowH * j + 6, tok = M.ARM_TOKEN[r.arm];
        if (i === 0) svg.appendChild(WK.h("text", { x: left - 8, y: y + 3, "text-anchor": "end", fill: "ink", "font-size": 12, text: r.arm }));
        if (r[p.key] === null) return;
        svg.appendChild(WK.h("line", { x1: sx(r[p.key + "_lo"]), x2: sx(r[p.key + "_hi"]), y1: y, y2: y, stroke: tok, "stroke-width": 2 }));
        svg.appendChild(WK.h("circle", { cx: sx(r[p.key]), cy: y, r: 4, fill: tok },
          WK.h("title", { text: r.arm + "\n" + p.label + ": " + ci(r, p.key) })));
      });
    });
    holder.appendChild(svg);

    var cols = ["Arm", "λ (EMA: τ)", "Linear leakage", "Linear content"].concat(st.knn ? ["kNN leakage"] : []).concat(["Collapsed seeds"]);
    var table = WK.el("table", { class: "table table-sm", style: { fontSize: "0.8rem" } },
      WK.el("thead", {}, WK.el("tr", {}, cols.map(function (c) { return WK.el("th", { text: c }); }))),
      WK.el("tbody", {}, rows.map(function (r) {
        var cells = [r.arm, String(r.arm === "EMA" ? r.ema_tau : r.lam), ci(r, "leak_r2"), ci(r, "content_r2")];
        if (st.knn) cells.push(ci(r, "leak_knn_r2"));
        cells.push(Math.round(r.collapsed_frac * r.n_seeds) + " of " + r.n_seeds);
        return WK.el("tr", {}, cells.map(function (c) { return WK.el("td", { text: c }); }));
      })));
    tableHolder.appendChild(table);
  }
  draw();
});
