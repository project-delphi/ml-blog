/* Widgets for "When UMAP Lies, Part 4". Built on widget-kit (WK) and the
 * series kit (UK, ../when-umap-lies/umapkit.js); every number comes from
 * model.js (UMLRankModel), and this file only draws.
 *
 *   widget-ranking   which method is best depends on the ruler: one row per
 *                    case, every method's median score on the chosen metric,
 *                    the winners marked, and two tallies of wins
 */
"use strict";

(function () {
  // Read the model when a widget mounts, not at load: if the page forgot
  // <script src="model.js">, WK.mount shows the error in the frame.
  var M = typeof UMLRankModel !== "undefined" ? UMLRankModel : null;
  function needModel() { if (!M) throw new Error("model.js did not load"); }

  // Why a case has no score on a metric, for the row that says so.
  var NO_SCORE = {
    classes: "no class labels",
    gaps: "scored on the overlap case only"
  };

  WK.mount("widget-ranking", function (root) {
    needModel();
    var R = WK.data("uml-ranking");
    if (!R) throw new Error("widget-data/ranking.js did not load");
    var state = { metric: "neighbours", hi: null, pin: null };
    var name = function (m) { return UK.method(m).name; };
    var miss = M.missing(R);
    var notRun = R.cases.filter(function (c) { return c.notRun.length; })
      .map(function (c) { return c.title; });

    var fr = WK.frame({
      title: "Which method is best depends on the ruler",
      note: "Each mark is a method's median over draws (data seeds; for PBMC 3k, five method " +
        "seeds) at its default settings only; Part 3 tunes them. Best = highest median, and every " +
        "method within " + R.tie + " of it shares the win, so a shared win counts once for each " +
        "method in it. Large distances use the geodesic Spearman where the shape has a geodesic, " +
        "else the straight-line one; Density uses the known true density where it varies, else " +
        "kNN density. Real data have no ground truth: their reference is the input space itself. " +
        "Gaps: −|valley ratio of the map − valley ratio of the input| on the overlap case, " +
        "so 0 is perfect; scored for PCA, UMAP, t-SNE, PaCMAP and TriMap only. Classes is missing " +
        "for " + miss.classes.join(", ") + " (no labels). PHATE was not run on " + notRun.join(", ") +
        "; densMAP only on the variable-density case."
    });
    var metricCtl = UK.chips({ label: "Ruler (metric)", value: state.metric,
      options: R.metrics.map(function (m) { return { value: m.key, label: m.label }; }),
      onchange: function (v) { state.metric = v; draw(); } });
    fr.controls.appendChild(metricCtl.root);

    var legendBox = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.15rem 0.4rem",
      marginBottom: "0.6rem" } });
    var rows = WK.el("div", { style: { minWidth: "0" } });
    var tallies = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "1rem 1.6rem", marginTop: "1rem" } });
    var t1 = WK.el("div", { style: { flex: "1 1 260px", minWidth: "0" } });
    var t2 = WK.el("div", { style: { flex: "1 1 260px", minWidth: "0" } });
    tallies.appendChild(t1); tallies.appendChild(t2);
    fr.body.appendChild(legendBox); fr.body.appendChild(rows); fr.body.appendChild(tallies);
    var tileMetric = WK.stat({ label: "", value: "", hint: " " });
    var tileAll = WK.stat({ label: "", value: "", hint: " " });
    fr.stats.appendChild(tileMetric.root); fr.stats.appendChild(tileAll.root);
    root.appendChild(fr.root);

    // ------------------------------------------------------------ legend
    var legendItems = {};
    R.methods.forEach(function (m) {
      var s = UK.method(m), sv = WK.svg(14, 14, { width: 14, height: 14, "aria-hidden": "true" });
      sv.style.width = "14px"; sv.style.display = "inline-block";
      sv.appendChild(UK.mark(s.marker, 7, 7, 3.6, s.token));
      var b = WK.el("button", { type: "button", "aria-pressed": "false", title: "Hover or tap to pick out",
        style: { display: "inline-flex", alignItems: "center", gap: "0.3rem", border: "0", background: "transparent",
          color: WK.cssVar("ink"), font: "inherit", fontSize: "0.8rem", padding: "0.1rem 0.35rem",
          borderRadius: "6px", cursor: "pointer" } }, sv, WK.el("span", { text: s.name }));
      b.addEventListener("mouseenter", function () { if (!state.pin) highlight(m); });
      b.addEventListener("mouseleave", function () { if (!state.pin) highlight(null); });
      b.addEventListener("focus", function () { if (!state.pin) highlight(m); });
      b.addEventListener("blur", function () { if (!state.pin) highlight(null); });
      b.addEventListener("click", function () { state.pin = state.pin === m ? null : m; highlight(state.pin); });
      legendItems[m] = b;
      legendBox.appendChild(b);
    });
    function highlight(m) {
      state.hi = m;
      Object.keys(legendItems).forEach(function (k) {
        var b = legendItems[k];
        b.style.opacity = m === null || m === k ? "1" : "0.45";
        b.style.background = state.pin === k ? WK.cssVar("paper") : "transparent";
        b.setAttribute("aria-pressed", String(state.pin === k));
      });
      drawRows();
      drawTallies();
    }

    // -------------------------------------------------------------- rows
    function drawRows() {
      var w = rows.clientWidth || 600, metric = state.metric, dom = M.domain(R, metric);
      var wide = w >= 560, LW = wide ? 210 : 0, PAD = 10;
      var ROW = wide ? 44 : 64, AXIS_TOP = 18;
      var H = AXIS_TOP + R.cases.length * ROW + 4;
      var sv = WK.svg(w, H, { width: w, height: H, "aria-label": "Each case's methods on " +
        R.metrics.filter(function (m) { return m.key === metric; })[0].label });
      var x0 = LW + PAD, x1 = w - PAD, sx = WK.lin(dom, [x0, x1]);
      var ticks = metric === "gaps" ? [dom[0], dom[0] / 2, 0] : [dom[0], 0, 0.5, 1].filter(function (t, i, a) {
        return a.indexOf(t) === i; });
      ticks.forEach(function (t) {
        sv.appendChild(WK.h("line", { x1: sx(t), x2: sx(t), y1: AXIS_TOP - 3, y2: H - 4, stroke: "rule",
          "stroke-width": t === 0 ? 1 : 0.6, "stroke-dasharray": t === 0 ? null : "2 3" }));
        sv.appendChild(WK.h("text", { x: sx(t), y: AXIS_TOP - 6, "text-anchor": "middle", "font-size": 10,
          fill: "muted" }, UK.num(t, t === Math.round(t) ? 0 : Math.abs(t * 10 - Math.round(t * 10)) < 1e-9 ? 1 : 2)));
      });
      R.cases.forEach(function (c, i) {
        var top = AXIS_TOP + i * ROW, b = M.best(R, c.key, metric);
        if (i % 2 === 0) sv.appendChild(WK.h("rect", { x: 0, y: top, width: w, height: ROW, fill: "paper", "fill-opacity": 0.6 }));
        var bestText = !b ? "no score: " + (NO_SCORE[metric] || "not measured") :
          b.winners.length > 3 ? "best: " + b.winners.length + " methods within " + R.tie + " (" + UK.num(b.top) + ")" :
            "best: " + b.winners.map(name).join(", ") + " (" + UK.num(b.top) + ")";
        var ty = wide ? top + 18 : top + 15;
        sv.appendChild(WK.h("text", { x: 6, y: ty, "font-size": 12, "font-weight": 600, fill: "ink" }, c.title));
        sv.appendChild(WK.h("text", { x: 6, y: ty + 15, "font-size": 10.5, fill: "muted" }, bestText));
        if (!b) return;
        var cy = wide ? top + ROW / 2 : top + 46;
        // Dodge: methods sorted by score take the first lane with room.
        var LANES = [0, -7, 7, -14, 14], placed = LANES.map(function () { return []; });
        M.scored(R, c.key, metric).map(function (m) { return { m: m, s: M.score(R, c.key, metric, m) }; })
          .sort(function (a, z) { return a.s.v - z.s.v; })
          .forEach(function (o) {
            var x = sx(o.s.v), lane = 0;
            for (var L = 0; L < LANES.length; L++) {
              if (placed[L].every(function (px) { return Math.abs(px - x) > 7.5; })) { lane = L; break; }
              lane = L;
            }
            placed[lane].push(x);
            var st = UK.method(o.m), win = b.winners.indexOf(o.m) >= 0;
            var g = WK.h("g", { opacity: state.hi === null || state.hi === o.m ? 1 : 0.15 });
            if (win) g.appendChild(WK.h("circle", { cx: x, cy: cy + LANES[lane], r: 6.8, fill: "none", stroke: "ink", "stroke-width": 1.3 }));
            g.appendChild(UK.mark(st.marker, x, cy + LANES[lane], 3.3, st.token));
            g.appendChild(WK.h("title", {}, c.title + ", " + st.name + ": " + UK.num(o.s.v, 3) + " (median of " + o.s.n +
              (o.s.n > 1 ? "; " + UK.num(o.s.lo, 3) + " to " + UK.num(o.s.hi, 3) : "") + ")" + (win ? ", best" : "")));
            sv.appendChild(g);
          });
      });
      WK.clear(rows); rows.appendChild(sv);
    }

    // ----------------------------------------------------------- tallies
    function bars(box, title, wins, alone, of, unit) {
      var w = box.clientWidth || 300, LW = 146, ROW = 17, order = M.ranked(R, wins);
      var H = 4 + order.length * ROW;
      var sv = WK.svg(w, H, { width: w, height: H, "aria-label": title });
      var sx = WK.lin([0, of], [LW, w - 64]);
      order.forEach(function (m, i) {
        var st = UK.method(m), y = 2 + i * ROW, dim = state.hi !== null && state.hi !== m;
        var g = WK.h("g", { opacity: dim ? 0.3 : 1 });
        g.appendChild(UK.mark(st.marker, 6, y + ROW / 2, 3, st.token));
        g.appendChild(WK.h("text", { x: 15, y: y + ROW / 2 + 4, "font-size": 11, fill: "ink" }, st.name));
        g.appendChild(WK.h("rect", { x: LW, y: y + 4, width: Math.max(0, sx(wins[m]) - LW), height: ROW - 8,
          fill: st.token, "fill-opacity": 0.85 }));
        g.appendChild(WK.h("text", { x: sx(wins[m]) + 5, y: y + ROW / 2 + 4, "font-size": 11, fill: "ink",
          style: { fontVariantNumeric: "tabular-nums" } }, String(wins[m]) +
          (wins[m] ? " (" + alone[m] + " alone)" : "")));
        g.appendChild(WK.h("title", {}, st.name + ": " + wins[m] + " of " + of + " " + unit + ", " +
          alone[m] + " of them not shared"));
        sv.appendChild(g);
      });
      WK.clear(box);
      box.appendChild(WK.el("div", { class: "widget-control-head", style: { fontWeight: "600" } },
        WK.el("span", { text: title })));
      box.appendChild(sv);
    }
    function drawTallies() {
      var label = R.metrics.filter(function (m) { return m.key === state.metric; })[0].label;
      var t = M.tally(R, state.metric), a = M.tallyAll(R);
      bars(t1, "Wins on " + label + " (of " + t.cases + " case" + (t.cases === 1 ? "" : "s") + ")",
        t.wins, t.alone, t.cases, "cases");
      bars(t2, "Wins over all " + a.pairs + " (case, metric) pairs", a.wins, a.alone, a.pairs, "pairs");
      var top = M.ranked(R, t.wins)[0], topAll = M.ranked(R, a.wins)[0];
      var tiedTop = M.ranked(R, t.wins).filter(function (m) { return t.wins[m] === t.wins[top]; });
      tileMetric.set(String(t.wins[top]) + " of " + t.cases);
      tileMetric.root.querySelector(".widget-stat-label").textContent =
        "most wins on " + label + ": " + tiedTop.map(name).join(", ");
      tileMetric.root.querySelector(".widget-note").textContent =
        t.shared + " of the " + t.cases + " cases are shared wins";
      var tiedAll = M.ranked(R, a.wins).filter(function (m) { return a.wins[m] === a.wins[topAll]; });
      tileAll.set(String(a.wins[topAll]) + " of " + a.pairs);
      tileAll.root.querySelector(".widget-stat-label").textContent =
        "most wins over all pairs: " + tiedAll.map(name).join(", ");
      tileAll.root.querySelector(".widget-note").textContent =
        (tiedAll.length === 1 ? a.alone[topAll] + " of them not shared; " : "") +
        a.shared + " of the " + a.pairs + " pairs are shared wins";
    }

    function draw() { drawRows(); drawTallies(); }
    UK.onResize(rows, drawRows);
    UK.onResize(t1, drawTallies);
    draw();
  });
})();
