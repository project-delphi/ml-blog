/* Normal approximation at finite n. Numbers come from SteinCltModel (model.js)
 * and the exact distances in widget-data/data.js; this file only draws. */
"use strict";

(function () {
  var M = SteinCltModel;
  var DRAWS = 40000, LO = -4, HI = 4, BIN = 0.2;

  WK.mount("widget-clt-rate", function (root, WK) {
    var st = { law: "exponential", p: 0.3, n: 10, seed: 1 };
    var f = WK.frame({
      title: "A standardised sum against N(0, 1)",
      badge: "interactive · exact distances",
      note: "Left: a histogram of 40,000 seeded draws of W (the Bernoulli case shows the exact probabilities, one bar per atom) and the N(0, 1) density. " +
        "Right: the exact Wasserstein distance d_W(W, Z) for n = 1 to 200 (solid) and the leave-one-out bound (dashed), both on log scales. " +
        "The distances are computed without simulation; the dot marks the current n."
    });

    var lawT = WK.toggle({ label: "Summand law (mean 0, variance 1)", value: st.law,
      options: [{ value: "bernoulli", label: "Bernoulli(p)" }, { value: "exponential", label: "Exponential" },
                { value: "uniform", label: "Uniform" }, { value: "mixture", label: "Skewed mixture" }],
      onchange: function (v) { st.law = v; syncP(); schedule(); } });
    var pS = WK.slider({ label: "Bernoulli p", min: 0.05, max: 0.95, step: 0.05, value: st.p,
      fmt: function (v) { return Number(v).toFixed(2); }, oninput: function (v) { st.p = v; schedule(); } });
    var nS = WK.slider({ label: "Number of summands n", min: 1, max: 200, step: 1, value: st.n,
      fmt: WK.fmt.int, oninput: function (v) { st.n = Math.round(v); schedule(); } });
    var lawGroup = lawT.root.querySelector(".widget-toggle"); // wrap inside a narrow (phone-width) frame
    lawGroup.style.flexWrap = "wrap"; lawGroup.style.borderRadius = "12px";
    [lawT, pS, nS].forEach(function (c) { f.controls.appendChild(c.root); });
    function syncP() {
      pS.input.disabled = st.law !== "bernoulli";
      pS.root.style.opacity = st.law === "bernoulli" ? "1" : "0.45";
    }

    var row = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.8rem" } });
    var W = 420, H = 300;
    var svgH = WK.svg(W, H, { "aria-label": "Histogram of the standardised sum with the standard normal density." });
    var svgR = WK.svg(W, H, { "aria-label": "Log-log plot of the exact Wasserstein distance and the Stein bound against n." });
    [svgH, svgR].forEach(function (s) {
      row.appendChild(WK.el("div", { style: { flex: "1 1 300px", minWidth: "0" } }, s));
    });
    f.body.appendChild(row);

    var sDw = WK.stat({ label: "Exact d_W(W, Z)", value: "" });
    var sB = WK.stat({ label: "Stein bound", value: "" });
    var sR = WK.stat({ label: "Bound ÷ d_W", value: "" });
    var sAll = WK.stat({ label: "Bound ≥ d_W for n = 1…200", value: "" });
    [sDw, sB, sR, sAll].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    var pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; draw(); });
    }
    function text(svg, x, y, s, anchor, fill) {
      svg.appendChild(WK.h("text", { x: x, y: y, "text-anchor": anchor || "middle", fill: fill || "muted", "font-size": 12 }, s));
    }

    function drawHist(L) {
      WK.clear(svgH);
      var ml = 40, mr = 10, mt = 12, mb = 36;
      var x = WK.lin([LO, HI], [ml, W - mr]);
      var bars = [], ymax = 0.45;
      if (st.law === "bernoulli") {
        var b = M.bernoulliAtoms(st.n, L.p);
        b.atoms.forEach(function (a) {
          var dens = a.mass / b.spacing;
          bars.push({ x0: a.w - b.spacing / 2, x1: a.w + b.spacing / 2, y: dens });
          ymax = Math.max(ymax, dens);
        });
      } else {
        var hist = M.histogram(M.drawW(st.law, st.n, DRAWS, st.seed + st.n), LO, HI, BIN);
        hist.density.forEach(function (d, i) {
          bars.push({ x0: LO + i * BIN, x1: LO + (i + 1) * BIN, y: d });
          ymax = Math.max(ymax, d);
        });
      }
      ymax = Math.min(ymax * 1.08, 1.6);
      var y = WK.lin([0, ymax], [H - mb, mt]);
      svgH.appendChild(WK.h("defs", {}, WK.h("clipPath", { id: "clt-hclip" },
        WK.h("rect", { x: ml, y: mt, width: W - ml - mr, height: H - mt - mb }))));
      WK.ticks(0, ymax, 4).forEach(function (t) {
        svgH.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(t), y2: y(t), stroke: "rule" }));
        text(svgH, ml - 6, y(t) + 4, WK.fmt.num(t, 1), "end");
      });
      [-4, -2, 0, 2, 4].forEach(function (t) { text(svgH, x(t), H - mb + 16, String(t)); });
      text(svgH, (ml + W - mr) / 2, H - 6, "w");
      var g = WK.h("g", { "clip-path": "url(#clt-hclip)" });
      svgH.appendChild(g);
      bars.forEach(function (bar) {
        var x0 = Math.max(x(bar.x0), ml), x1 = Math.min(x(bar.x1), W - mr);
        if (x1 <= x0) return;
        g.appendChild(WK.h("rect", { x: x0, y: y(bar.y), width: Math.max(0.5, x1 - x0 - 0.4),
          height: Math.max(0, y(0) - y(bar.y)), fill: "c1", "fill-opacity": 0.45 }));
      });
      var pts = [];
      for (var t = LO; t <= HI + 1e-9; t += 0.05) pts.push(x(t) + "," + y(M.phi(t)));
      g.appendChild(WK.h("polyline", { points: pts.join(" "), fill: "none", stroke: "c3", "stroke-width": 2.2 }));
      svgH.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(0), y2: y(0), stroke: "muted" }));
      text(svgH, W - mr - 2, mt + 12, "N(0, 1) density", "end", "c3");
    }

    function drawRate(L) {
      WK.clear(svgR);
      var ml = 46, mr = 10, mt = 12, mb = 36;
      var lo = -4, hi = 1; // log10 range of the y axis
      var x = WK.lin([0, Math.log10(200)], [ml, W - mr]);
      var y = WK.lin([lo, hi], [H - mb, mt]);
      for (var e = lo; e <= hi; e++) {
        svgR.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(e), y2: y(e), stroke: "rule" }));
        text(svgR, ml - 6, y(e) + 4, e === 0 ? "1" : "10" + toSup(e), "end");
      }
      [1, 10, 100].forEach(function (t) { text(svgR, x(Math.log10(t)), H - mb + 16, String(t)); });
      text(svgR, x(Math.log10(200)), H - mb + 16, "200", "end");
      text(svgR, (ml + W - mr) / 2, H - 6, "n (log scale)");
      var dw = [], bd = [];
      for (var n = 1; n <= 200; n++) {
        var lx = x(Math.log10(n));
        dw.push(lx + "," + y(Math.log10(L.dw[n - 1])));
        bd.push(lx + "," + y(Math.log10(M.bound(L, n))));
      }
      svgR.appendChild(WK.h("polyline", { points: bd.join(" "), fill: "none", stroke: "c3", "stroke-width": 2, "stroke-dasharray": "6 4" }));
      svgR.appendChild(WK.h("polyline", { points: dw.join(" "), fill: "none", stroke: "c1", "stroke-width": 2.4 }));
      var cx = x(Math.log10(st.n));
      svgR.appendChild(WK.h("circle", { cx: cx, cy: y(Math.log10(L.dw[st.n - 1])), r: 4.5, fill: "c1" }));
      svgR.appendChild(WK.h("circle", { cx: cx, cy: y(Math.log10(M.bound(L, st.n))), r: 4, fill: "c3" }));
      text(svgR, W - mr - 2, mt + 12, "bound (dashed)", "end", "c3");
      text(svgR, W - mr - 2, mt + 28, "exact d_W (solid)", "end", "c1");
    }
    function toSup(e) {
      var map = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴" };
      return String(e).split("").map(function (c) { return map[c] || c; }).join("");
    }

    function draw() {
      var L = M.law(st.law, st.p);
      drawHist(L);
      drawRate(L);
      var d = L.dw[st.n - 1], b = M.bound(L, st.n);
      sDw.set(d < 0.001 ? d.toExponential(2) : WK.fmt.num(d, 4));
      sB.set(WK.fmt.num(b, 4));
      sR.set(WK.fmt.num(b / d, 1) + "×");
      var worst = M.worstRatio(L);
      sAll.set(worst > 1 ? "yes, ≥ " + WK.fmt.num(worst, 2) + "×" : "NO");
    }

    syncP();
    draw();
  });
})();
