/* Two charts for Part 6. Numbers come from SteinNetsModel (model.js) and from the
 * precomputed exact laws in widget-data/data.js; this file only draws. */
"use strict";

(function () {
  var M = SteinNetsModel;

  function text(svg, x, y, s, anchor, fill, size) {
    // a halo in the surface colour keeps labels legible where they cross a curve or band
    svg.appendChild(WK.h("text", { x: x, y: y, "text-anchor": anchor || "middle", fill: fill || "muted", "font-size": size || 12,
      stroke: "surface", "stroke-width": 3, "paint-order": "stroke" }, s));
  }
  function poly(pts, colour, width, dash) {
    return WK.h("polyline", { points: pts.join(" "), fill: "none", stroke: colour, "stroke-width": width || 2, "stroke-dasharray": dash || null });
  }
  function row2(f, W, H, labelA, labelB) {
    var row = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.8rem" } });
    var a = WK.svg(W, H, { "aria-label": labelA }), b = WK.svg(W, H, { "aria-label": labelB });
    [a, b].forEach(function (s) { row.appendChild(WK.el("div", { style: { flex: "1 1 300px", minWidth: "0" } }, s)); });
    f.body.appendChild(row);
    return [a, b];
  }
  function logScale(lo, hi, r0, r1) {
    var s = WK.lin([Math.log10(lo), Math.log10(hi)], [r0, r1]);
    var f = function (v) { return s(Math.log10(v)); };
    var t = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      var v = Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v);
    }
    f.ticks = t;
    return f;
  }
  function expLabel(v) {
    var e = Math.round(Math.log10(v));
    return e === 0 ? "1" : e === 1 ? "10" : e === 2 ? "100" : e === 3 ? "1000" : "1e" + String(e).replace("-", "−");
  }
  function sig(v, d) {
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e4)) return v.toExponential((d || 3) - 1);
    return v.toPrecision(d || 3);
  }
  function btn(label, onclick) {
    return WK.el("button", { type: "button", class: "widget-toggle-btn", text: label,
      style: { border: "1px solid var(--w-rule)", borderRadius: "999px" }, onclick: onclick });
  }

  // --------------------------------------------- 1. how Gaussian is a random network?
  WK.mount("widget-width", function (root, WK) {
    var D = WK.data("netGaussianity");
    if (!D) { root.textContent = "The chart's data file did not load."; return; }
    var st = { act: "relu", ni: D.n.indexOf(10), seed: 1 }, MS = 4000;
    var f = WK.frame({
      title: "A random one-hidden-layer network's output against N(0, 1)",
      badge: "interactive · random networks",
      note: "Each network has width n, input weights w_i ~ N(0, I/d), output weights v_i ~ N(0, 1), and is evaluated at one fixed input with |x|² = d; the output is scaled to variance 1. " +
        "Left: a histogram of 4,000 random networks' outputs, the exact density of the output, and the N(0, 1) density. " +
        "Right: the exact Wasserstein distance d_W from N(0, 1) against width, with Part 2's Theorem 4 bound, Corollary 2's bound, and the large-width limit 0.189 κ / n. Both axes are logarithmic."
    });
    var aT = WK.toggle({ label: "Activation", value: st.act,
      options: Object.keys(M.ACTS).map(function (k) { return { value: k, label: M.ACTS[k].label }; }),
      onchange: function (v) { st.act = v; resample(); } });
    var nS = WK.slider({ label: "Width n", min: 0, max: D.n.length - 1, step: 1, value: st.ni,
      fmt: function (v) { return String(D.n[v]); }, oninput: function (v) { st.ni = v; resample(); } });
    [aT, nS].forEach(function (c) { f.controls.appendChild(c.root); });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } },
      btn("Draw new networks", function () { st.seed += 1; resample(); })));
    var W = 420, H = 280, svgs = row2(f, W, H, "Histogram of network outputs with the exact and normal densities.", "Wasserstein distance from the normal against width, with two bounds.");
    var sK = WK.stat({ label: "κ = Var σ(Z)² / (E σ(Z)²)²", value: "" });
    var sD = WK.stat({ label: "Exact d_W at this width", value: "" });
    var s4 = WK.stat({ label: "Theorem 4 bound (Part 2)", value: "" });
    var sC = WK.stat({ label: "Corollary 2 bound", value: "" });
    var sN = WK.stat({ label: "n · d_W (limit 0.189 κ)", value: "" });
    [sK, sD, s4, sC, sN].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    var samples = null, m = { l: 54, r: 22, t: 22, b: 34 };
    function resample() { samples = M.sampleOutputs(st.act, D.n[st.ni], MS, 31 * st.seed + st.ni); draw(); }
    function draw() {
      var A = D.acts[st.act], n = D.n[st.ni], C = M.constants(st.act), B = M.bounds(st.act, n), dW = A.dW[st.ni];
      // left: histogram, exact density and N(0, 1)
      var svg = svgs[0]; WK.clear(svg);
      var lo = -4, hi = 4, bw = 0.2, nb = Math.round((hi - lo) / bw), counts = new Array(nb).fill(0), i;
      for (i = 0; i < samples.length; i++) { var k = Math.floor((samples[i] - lo) / bw); if (k >= 0 && k < nb) counts[k]++; }
      var dens = A.dens[st.ni], top = 0.6;
      var xs = WK.lin([lo, hi], [m.l, W - m.r]), ys = WK.lin([0, top], [H - m.b, m.t]);
      [0, 0.2, 0.4, 0.6].forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), stroke: "rule" }));
        text(svg, m.l - 6, ys(t) + 4, String(t), "end");
      });
      [-4, -2, 0, 2, 4].forEach(function (t) { text(svg, xs(t), H - m.b + 16, String(t)); });
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "muted" }));
      text(svg, (m.l + W - m.r) / 2, H - 6, "network output");
      text(svg, m.l, m.t - 6, "density", "start");
      for (k = 0; k < nb; k++) {
        var hgt = counts[k] / (samples.length * bw);
        if (hgt <= 0) continue;
        svg.appendChild(WK.h("rect", { x: xs(lo + k * bw) + 0.5, y: ys(Math.min(hgt, top)), width: xs(lo + bw) - xs(lo) - 1,
          height: ys(0) - ys(Math.min(hgt, top)), fill: "c3", "fill-opacity": 0.35 }));
      }
      var pn = [], pe = [];
      for (i = 0; i < D.w.length; i++) {
        var w = D.w[i];
        pn.push(xs(w) + "," + ys(Math.exp(-w * w / 2) / Math.sqrt(2 * Math.PI)));
        pe.push(xs(w) + "," + ys(Math.min(dens[i], top)));
      }
      svg.appendChild(poly(pn, "muted", 1.8, "5 4"));
      svg.appendChild(poly(pe, "c1", 2.4));
      var atom = A.atom[st.ni];
      if (atom > 1e-3) {
        svg.appendChild(WK.h("line", { x1: xs(0), x2: xs(0), y1: ys(0), y2: m.t + 4, stroke: "c1", "stroke-width": 3 }));
        text(svg, xs(0) + 6, m.t + 30, "atom at 0, mass " + atom.toPrecision(2), "start", "c1");
      }
      text(svg, W - m.r - 2, m.t + 10, "exact density", "end", "c1");
      text(svg, W - m.r - 2, m.t + 25, "N(0, 1)", "end", "muted");
      text(svg, W - m.r - 2, m.t + 40, MS + " networks", "end", "c3");

      // right: d_W against n
      svg = svgs[1]; WK.clear(svg);
      var N0 = D.n[0], N1 = D.n[D.n.length - 1], lx = logScale(N0, N1, m.l, W - m.r), ly = logScale(1e-5, 10, H - m.b, m.t);
      ly.ticks.forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ly(t), y2: ly(t), stroke: "rule" }));
        text(svg, m.l - 6, ly(t) + 4, expLabel(t), "end");
      });
      lx.ticks.forEach(function (t) { text(svg, lx(t), H - m.b + 16, String(t)); });
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "muted" }));
      text(svg, (m.l + W - m.r) / 2, H - 6, "width n (log scale)");
      text(svg, m.l, m.t - 6, "Wasserstein distance from N(0, 1)", "start");
      function curve(vals, colour, w, dash) {
        var p = [];
        D.n.forEach(function (nn, j) { var v = vals(nn, j); if (v > 0) p.push(lx(nn) + "," + Math.max(m.t, Math.min(H - m.b, ly(v)))); });
        if (p.length) svg.appendChild(poly(p, colour, w, dash));
      }
      curve(function (nn) { return M.bounds(st.act, nn).thm4; }, "c3", 1.8, "6 4");
      curve(function (nn) { return M.bounds(st.act, nn).mix; }, "c2", 1.8, "6 4");
      curve(function (nn) { return C.limit / nn; }, "muted", 1.4, "2 3");
      curve(function (nn, j) { return A.dW[j]; }, "c1", 2.4);
      if (dW > 0) svg.appendChild(WK.h("circle", { cx: lx(n), cy: ly(dW), r: 5, fill: "c1" }));
      svg.appendChild(WK.h("line", { x1: lx(n), x2: lx(n), y1: m.t, y2: H - m.b, stroke: "ink", "stroke-width": 1, "stroke-dasharray": "2 3" }));
      text(svg, W - m.r - 2, m.t + 10, "Theorem 4 (Part 2)", "end", "c3");
      text(svg, W - m.r - 2, m.t + 25, "Corollary 2", "end", "c2");
      text(svg, W - m.r - 2, m.t + 40, st.act === "sign" ? "exact: 0 at every width" : "exact d_W", "end", "c1");

      sK.set(C.kappa.toFixed(3));
      sD.set(dW > 0 ? sig(dW) : "0");
      s4.set(sig(B.thm4));
      sC.set(st.act === "sign" ? "0" : sig(B.mix));
      sN.set((n * dW).toFixed(3) + "  (" + C.limit.toFixed(3) + ")");
    }
    resample();
  });

  // ---------------------------------------------------- 2. finding a hidden direction
  WK.mount("widget-direction", function (root, WK) {
    var st = { link: "relu", d: 20, sigma: 0.5, ni: 4, seed: 1 }, K = 6, res = null;
    var f = WK.frame({
      title: "Recovering a hidden direction with Stein's lemma",
      badge: "interactive · synthetic data",
      note: "Synthetic data: inputs x ~ N(0, I_d), a hidden unit direction w, and y = g(w·x), standardised to mean 0 and variance 1, plus N(0, σ²) noise. " +
        "Left: the link's Hermite coefficients c_k = E[g⁽ᵏ⁾(Z)]. Right: |cos| between w and the first-order estimate mean(y x), and between w and the top eigenvector of mean(y (x xᵀ − I)), " +
        "against the number of samples n; lines are means over 6 data sets and bands their range. The dashed line is 1/√d, the root-mean-square overlap of a uniformly random direction."
    });
    var lT = WK.toggle({ label: "Link g(t)", value: st.link,
      options: ["linear", "tanh", "relu", "abs", "he2"].map(function (k) { return { value: k, label: M.LINKS[k].label }; }),
      onchange: function (v) { st.link = v; recompute(); } });
    var dT = WK.toggle({ label: "Dimension d", value: String(st.d),
      options: [10, 20, 40].map(function (d) { return { value: String(d), label: String(d) }; }),
      onchange: function (v) { st.d = +v; recompute(); } });
    var timer = null;
    var sS = WK.slider({ label: "Noise σ", min: 0, max: 1, step: 0.05, value: st.sigma, fmt: function (v) { return Number(v).toFixed(2); },
      oninput: function (v) { st.sigma = v; if (timer) clearTimeout(timer); timer = setTimeout(recompute, 120); } });
    var nS = WK.slider({ label: "Samples n", min: 0, max: M.N_LIST.length - 1, step: 1, value: st.ni,
      fmt: function (v) { return String(M.N_LIST[v]); }, oninput: function (v) { st.ni = v; draw(); } });
    lT.root.style.flex = "1 1 100%";
    [lT, dT, sS, nS].forEach(function (c) { f.controls.appendChild(c.root); });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } },
      btn("New data sets", function () { st.seed += K; recompute(); })));
    var W = 420, H = 280, svgs = row2(f, W, H, "Hermite coefficients of the link.", "Overlap of each estimate with the hidden direction against sample size.");
    var sC = WK.stat({ label: "c₁ = E g′(Z),  c₂ = E g″(Z)", value: "" });
    var sK = WK.stat({ label: "First order k with c_k ≠ 0", value: "" });
    var sO = WK.stat({ label: "Overlap at this n: first order / second order", value: "" });
    var sR = WK.stat({ label: "Random direction, RMS overlap 1/√d", value: "" });
    [sC, sK, sO, sR].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    var m = { l: 54, r: 22, t: 22, b: 34 };
    function recompute() { res = M.sweep(st.link, st.d, st.sigma, K, 1000 * st.seed + 7); draw(); }
    function draw() {
      var info = M.linkInfo(st.link), i;
      // left: coefficient bars
      var svg = svgs[0]; WK.clear(svg);
      var ys = WK.lin([-2, 2], [H - m.b, m.t]);
      [-2, -1, 0, 1, 2].forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), stroke: t === 0 ? "muted" : "rule" }));
        text(svg, m.l - 6, ys(t) + 4, String(t), "end");
      });
      var slot = (W - m.l - m.r) / 4, bw = slot * 0.5;
      info.c.forEach(function (c, k) {
        var x0 = m.l + slot * k + (slot - bw) / 2, colour = k === 0 ? "c1" : k === 1 ? "c2" : "muted";
        svg.appendChild(WK.h("rect", { x: x0, y: Math.min(ys(0), ys(c)), width: bw, height: Math.max(1, Math.abs(ys(c) - ys(0))), fill: colour, "fill-opacity": 0.85 }));
        text(svg, x0 + bw / 2, H - m.b + 16, "c" + "₁₂₃₄"[k], "middle", "ink");
        text(svg, x0 + bw / 2, c >= 0 ? ys(c) - 5 : ys(c) + 14, Math.abs(c) < 5e-4 ? "0" : c.toFixed(2), "middle", colour);
      });
      text(svg, (m.l + W - m.r) / 2, H - 6, "Hermite order k");
      text(svg, m.l, m.t - 6, "c_k = E[g⁽ᵏ⁾(Z)], standardised g", "start");

      // right: overlap against n
      svg = svgs[1]; WK.clear(svg);
      var NL = res.n, lx = logScale(NL[0], NL[NL.length - 1], m.l, W - m.r), yo = WK.lin([0, 1], [H - m.b, m.t]);
      [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: yo(t), y2: yo(t), stroke: "rule" }));
        text(svg, m.l - 6, yo(t) + 4, String(t), "end");
      });
      [25, 100, 400, 1600, 6400].forEach(function (t) { text(svg, lx(t), H - m.b + 16, String(t)); });
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "muted" }));
      text(svg, (m.l + W - m.r) / 2, H - 6, "samples n (log scale)");
      text(svg, m.l, m.t - 6, "|cos(estimate, w)|", "start");
      var rnd = 1 / Math.sqrt(st.d);
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: yo(rnd), y2: yo(rnd), stroke: "muted", "stroke-dasharray": "5 4" }));
      [["first", "c1"], ["second", "c2"]].forEach(function (p) {
        var S = res[p[0]], top = [], bot = [];
        for (i = 0; i < NL.length; i++) { top.push(lx(NL[i]) + "," + yo(S[i].max)); bot.unshift(lx(NL[i]) + "," + yo(S[i].min)); }
        svg.appendChild(WK.h("polygon", { points: top.concat(bot).join(" "), fill: p[1], "fill-opacity": 0.15, stroke: "none" }));
        svg.appendChild(poly(S.map(function (s, j) { return lx(NL[j]) + "," + yo(s.mean); }), p[1], 2.4));
      });
      var nCur = NL[st.ni];
      svg.appendChild(WK.h("line", { x1: lx(nCur), x2: lx(nCur), y1: m.t, y2: H - m.b, stroke: "ink", "stroke-width": 1, "stroke-dasharray": "2 3" }));
      text(svg, W - m.r - 2, H - m.b - 40, "first order: mean(y x)", "end", "c1");
      text(svg, W - m.r - 2, H - m.b - 25, "second order: mean(y (x xᵀ − I))", "end", "c2");
      text(svg, W - m.r - 2, H - m.b - 10, "random direction, RMS 1/√d", "end", "muted");

      var c3 = function (v) { return Math.abs(v) < 5e-4 ? "0" : v.toFixed(3); };
      sC.set(c3(info.c[0]) + ",  " + c3(info.c[1]));
      sK.set(String(info.kstar));
      sO.set(res.first[st.ni].mean.toFixed(2) + " / " + res.second[st.ni].mean.toFixed(2));
      sR.set(rnd.toFixed(2));
    }
    recompute();
  });
})();
