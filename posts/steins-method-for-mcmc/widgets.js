/* Two charts for Part 5. Numbers come from SteinMCMCModel (model.js); this file only draws. */
"use strict";

(function () {
  var M = SteinMCMCModel;

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
  /** A log10 scale with .ticks: powers of ten, plus 2 and 5 when the span is short. */
  function logScale(lo, hi, r0, r1) {
    var s = WK.lin([Math.log10(lo), Math.log10(hi)], [r0, r1]);
    var f = function (v) { return s(Math.log10(v)); };
    var t = [], e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi)), short = e1 - e0 <= 2;
    for (var e = e0; e <= e1; e++) [1, 2, 5].forEach(function (m) {
      var v = m * Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001 && (m === 1 || short)) t.push(v);
    });
    f.ticks = t;
    return f;
  }
  function fmtTick(v) {
    if (v >= 1000) return v.toExponential(0).replace("e+", "e");
    if (v >= 1) return String(+v.toPrecision(2));
    return String(+v.toPrecision(1));
  }
  function sig(v, d) {
    if (!isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(d === undefined ? 2 : d - 1);
    return v.toPrecision(d === undefined ? 3 : d);
  }

  // ---------------------------------------------------- 1. ULA's bias and the step size
  WK.mount("widget-ula", function (root, WK) {
    var EPS_LIST = [0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4];
    var st = { target: "gauss", h: 1, ei: EPS_LIST.indexOf(0.1), lN: 4 };
    var VIEW = { gauss: [-4, 4], heavy: [-8, 8], mixture: [-3.8, 4.4] };
    var f = WK.frame({
      title: "Unadjusted Langevin: bias, Stein's prediction, and the best step size",
      badge: "interactive · exact on a grid",
      note: "Each sampler's transition kernel is discretised on a grid of 241 points, so its stationary law, bias and asymptotic variance come from linear algebra with no Monte Carlo noise. " +
        "Left: ULA's stationary density minus the target density at this step size, Stein's first-order prediction of that difference, (ε/2) p (ψ − E ψ) with ψ = s' + s²/2, and the target's shape for reference. " +
        "Right: root mean squared error of an average over N steps, √(bias² + asymptotic variance / N), for ULA and MALA; ULA's bias on its own; and Stein's prediction of that bias, ε·c. Circles mark each sampler's best step at this N."
    });
    var tT = WK.toggle({ label: "Target", value: st.target,
      options: [{ value: "gauss", label: "Gaussian" }, { value: "heavy", label: "Heavier tails" }, { value: "mixture", label: "Two modes" }],
      onchange: function (v) { st.target = v; draw(); } });
    var hT = WK.toggle({ label: "Estimate", value: String(st.h),
      options: M.HS.map(function (h, i) { return { value: String(i), label: h.label }; }),
      onchange: function (v) { st.h = +v; draw(); } });
    var eS = WK.slider({ label: "Step size ε", min: 0, max: EPS_LIST.length - 1, step: 1, value: st.ei,
      fmt: function (v) { return String(EPS_LIST[v]); }, oninput: function (v) { st.ei = v; draw(); } });
    var nS = WK.slider({ label: "Budget N (steps)", min: 3, max: 6, step: 0.05, value: st.lN,
      fmt: function (v) { var n = Math.pow(10, v); return n >= 1e5 ? n.toExponential(1).replace("e+", "e") : String(Math.round(n)); },
      oninput: function (v) { st.lN = v; draw(); } });
    tT.root.style.flex = "1 1 300px"; hT.root.style.flex = "1 1 420px";
    [tT, hT, eS, nS].forEach(function (c) { f.controls.appendChild(c.root); });
    var W = 420, H = 280, svgs = row2(f, W, H, "Target density and ULA's stationary density.", "Error against step size for ULA and MALA.");
    var sBias = WK.stat({ label: "ULA bias in the estimate (exact)", value: "" });
    var sPred = WK.stat({ label: "Stein's first-order prediction ε·c", value: "" });
    var sErr = WK.stat({ label: "RMSE at this ε and N: ULA / MALA", value: "" });
    var sBest = WK.stat({ label: "Best step at this N: ULA / MALA", value: "" });
    var sBestR = WK.stat({ label: "RMSE at the best step: ULA / MALA", value: "" });
    var sAcc = WK.stat({ label: "MALA acceptance rate", value: "" });
    [sBias, sPred, sErr, sBest, sBestR, sAcc].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    var m = { l: 54, r: 22, t: 22, b: 34 };
    function draw() {
      var eps = EPS_LIST[st.ei], N = Math.pow(10, st.lN), G = M.setup(st.target);
      var ula = M.analyse(st.target, "ula", eps), mala = M.analyse(st.target, "mala", eps);
      var pred = M.predictedDensity(st.target, eps), v = VIEW[st.target], i;
      // left: ULA's stationary density minus the target, against Stein's first-order prediction
      var svg = svgs[0]; WK.clear(svg);
      var xs = WK.lin(v, [m.l, W - m.r]), amp = 0, pts = { p: [], u: [], q: [] }, pmax = 0;
      for (i = 0; i < G.N; i++) if (G.x[i] >= v[0] && G.x[i] <= v[1]) {
        amp = Math.max(amp, Math.abs(ula.pi[i] - G.w[i]) / G.dx, Math.abs(pred[i] - G.w[i]) / G.dx);
        pmax = Math.max(pmax, G.w[i] / G.dx);
      }
      amp = (amp || 1e-6) * 1.2;
      var ys = WK.lin([-amp, amp], [H - m.b, m.t]);
      WK.ticks(-amp, amp, 4).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), stroke: t === 0 ? "muted" : "rule" }));
        text(svg, m.l - 6, ys(t) + 4, t === 0 ? "0" : String(+t.toPrecision(2)), "end");
      });
      WK.ticks(v[0], v[1], 6).forEach(function (t) { text(svg, xs(t), H - m.b + 16, String(t)); });
      text(svg, (m.l + W - m.r) / 2, H - 6, "x");
      text(svg, m.l, m.t - 6, "density difference", "start");
      for (i = 0; i < G.N; i++) {
        if (G.x[i] < v[0] || G.x[i] > v[1]) continue;
        var X = xs(G.x[i]);
        // the target itself, rescaled to the panel's upper half, only to show where its mass is
        pts.p.push(X + "," + ys(amp * 0.9 * G.w[i] / G.dx / pmax));
        pts.u.push(X + "," + ys((ula.pi[i] - G.w[i]) / G.dx));
        pts.q.push(X + "," + ys((pred[i] - G.w[i]) / G.dx));
      }
      svg.appendChild(poly(pts.p, "c2", 1.4, "1 3"));
      svg.appendChild(poly(pts.u, "c1", 2.4));
      svg.appendChild(poly(pts.q, "c3", 1.8, "5 4"));
      text(svg, W - m.r - 2, m.t + 10, "ULA's density − target density", "end", "c1");
      text(svg, W - m.r - 2, m.t + 26, "Stein's first-order prediction", "end", "c3");
      text(svg, W - m.r - 2, m.t + 42, "target shape (rescaled)", "end", "c2");

      // right: error against step size
      svg = svgs[1]; WK.clear(svg);
      var cu = M.rmseCurve(st.target, "ula", st.h, N), cm = M.rmseCurve(st.target, "mala", st.h, N);
      var c = G.coef[st.h], hasBias = Math.abs(c) > 1e-9;
      var vals = cu.rmse.concat(cm.rmse);
      if (hasBias) vals = vals.concat(cu.bias.map(Math.abs));
      var lo = Math.min.apply(null, vals.filter(function (x) { return x > 0; })), hi = Math.max.apply(null, vals);
      lo = Math.pow(10, Math.floor(Math.log10(lo * 0.9))); hi = Math.pow(10, Math.ceil(Math.log10(hi * 1.1)));
      var lx = logScale(M.EPS_MIN, M.EPS_MAX, m.l, W - m.r), ly = logScale(lo, hi, H - m.b, m.t);
      ly.ticks.forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ly(t), y2: ly(t), stroke: "rule" }));
        text(svg, m.l - 6, ly(t) + 4, t.toExponential(0).replace("e+0", "").replace("e-", "e−").replace(/^1e/, "1e"), "end");
      });
      [0.01, 0.02, 0.05, 0.1, 0.2, 0.4].forEach(function (t) { text(svg, lx(t), H - m.b + 16, String(t)); });
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "muted" }));
      text(svg, (m.l + W - m.r) / 2, H - 6, "step size ε (log scale)");
      text(svg, m.l, m.t - 6, "error (log scale)", "start");
      function line(xsA, ysA, colour, w, dash) {
        var p = [];
        for (var k = 0; k < xsA.length; k++) if (ysA[k] > 0 && isFinite(ysA[k])) p.push(lx(xsA[k]) + "," + Math.max(m.t, Math.min(H - m.b, ly(ysA[k]))));
        svg.appendChild(poly(p, colour, w, dash));
      }
      if (hasBias) {
        line(cu.eps, cu.bias.map(Math.abs), "c1", 1.6, "6 4");
        line(cu.eps, cu.eps.map(function (e) { return Math.abs(c) * e; }), "c3", 1.6, "2 3");
      }
      line(cu.eps, cu.rmse, "c1", 2.4);
      line(cm.eps, cm.rmse, "c2", 2.4);
      [[cu.best, "c1"], [cm.best, "c2"]].forEach(function (b) {
        svg.appendChild(WK.h("circle", { cx: lx(b[0].eps), cy: ly(b[0].rmse), r: 5, fill: "none", stroke: b[1], "stroke-width": 2 }));
      });
      svg.appendChild(WK.h("line", { x1: lx(eps), x2: lx(eps), y1: m.t, y2: H - m.b, stroke: "ink", "stroke-width": 1, "stroke-dasharray": "2 3" }));
      var lyT = m.t + 10;
      text(svg, m.l + 6, lyT, "ULA RMSE", "start", "c1"); lyT += 15;
      text(svg, m.l + 6, lyT, "MALA RMSE", "start", "c2"); lyT += 15;
      if (hasBias) { text(svg, m.l + 6, lyT, "ULA |bias| (dashed)", "start", "c1"); lyT += 15; text(svg, m.l + 6, lyT, "Stein: |c| ε (dotted)", "start", "c3"); }
      else text(svg, m.l + 6, lyT, "ULA bias is 0 here, by symmetry", "start", "muted");

      sBias.set(hasBias ? sig(ula.bias[st.h]) : "0 (symmetry)");
      sPred.set(hasBias ? sig(c * eps) + "  (c = " + sig(c) + ")" : "0");
      sErr.set(sig(M.rmse(ula.bias[st.h], ula.avar[st.h], N)) + " / " + sig(M.rmse(mala.bias[st.h], mala.avar[st.h], N)));
      sBest.set(cu.best.eps.toFixed(3) + " / " + (cm.best.eps >= M.EPS_MAX * 0.999 ? "≥ " : "") + cm.best.eps.toFixed(3));
      sBestR.set(sig(cu.best.rmse) + " / " + sig(cm.best.rmse));
      sAcc.set(mala.acc.toFixed(3));
    }
    draw();
  });

  // ------------------------------------------------ 2. zero-variance control variates
  WK.mount("widget-zv", function (root, WK) {
    var EPS = 0.15, R = 20;
    var N_LIST = [500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000];
    var st = { h: 0, ni: N_LIST.indexOf(5000), seed: 1 };
    var f = WK.frame({
      title: "Zero-variance control variates on a MALA chain",
      badge: "interactive · synthetic data",
      note: "The posterior of a two-coefficient logistic regression with 30 synthetic observations and a N(0, 3²I) prior, sampled by MALA with step 0.15. " +
        "Left: posterior contours and the chain's last 400 states. Right: the running estimate after m states for the plain average and for zero-variance control variates of polynomial degree 1 and 2, refitted at each checkpoint. " +
        "The dashed line is the posterior expectation computed by quadrature on a 401 × 401 grid. Tiles report mean squared errors over 20 independent chains of the chosen length."
    });
    var hT = WK.toggle({ label: "Estimate", value: String(st.h),
      options: M.LR_H.map(function (h, i) { return { value: String(i), label: h.label }; }),
      onchange: function (v) { st.h = +v; recompute(); } });
    var timer = null;
    var nS = WK.slider({ label: "Chain length n", min: 0, max: N_LIST.length - 1, step: 1, value: st.ni,
      fmt: function (v) { return String(N_LIST[v]); },
      oninput: function (v) { st.ni = v; if (timer) clearTimeout(timer); timer = setTimeout(recompute, 120); } });
    var again = WK.el("button", { type: "button", class: "widget-toggle-btn", text: "New chain",
      style: { border: "1px solid var(--w-rule)", borderRadius: "999px" },
      onclick: function () { st.seed += 1; recompute(); } });
    hT.root.style.flex = "1 1 420px";
    [hT, nS].forEach(function (c) { f.controls.appendChild(c.root); });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } }, again));
    var W = 420, H = 300, svgs = row2(f, W, H, "Posterior contours with the chain's recent states.", "Running estimates with and without control variates.");
    var sOne = WK.stat({ label: "This chain's error: plain / degree 1 / degree 2", value: "" });
    var sMse = WK.stat({ label: "MSE of the plain average, 20 chains", value: "" });
    var s1 = WK.stat({ label: "MSE reduction, degree 1", value: "" });
    var s2 = WK.stat({ label: "MSE reduction, degree 2", value: "" });
    var sAcc = WK.stat({ label: "Acceptance rate", value: "" });
    [sOne, sMse, s1, s2, sAcc].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    // contours of the posterior, once
    var GD = 70, dens = M.lrDensityGrid(GD), V = M.LR_VIEW, segs = [];
    [0.05, 0.25, 0.5, 0.75].forEach(function (lev) {
      for (var j = 0; j < GD - 1; j++) for (var i = 0; i < GD - 1; i++) {
        var a = dens[j * GD + i], b = dens[j * GD + i + 1], c = dens[(j + 1) * GD + i + 1], d = dens[(j + 1) * GD + i], pts = [];
        var edge = function (p, q, x0, y0, x1, y1) { if ((p >= lev) !== (q >= lev)) { var t = (lev - p) / (q - p); pts.push([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]); } };
        edge(a, b, i, j, i + 1, j); edge(b, c, i + 1, j, i + 1, j + 1); edge(c, d, i + 1, j + 1, i, j + 1); edge(d, a, i, j + 1, i, j);
        if (pts.length >= 2) segs.push([pts[0], pts[1]]);
        if (pts.length === 4) segs.push([pts[2], pts[3]]);
      }
    });
    var m = { l: 54, r: 22, t: 22, b: 34 };
    var chain = null, rep = null;
    function recompute() {
      var n = N_LIST[st.ni];
      chain = M.mala(n, EPS, 7000 + st.seed);
      rep = M.replicate(st.h, n, EPS, R, 100 * st.seed);
      draw();
    }
    function draw() {
      var ref = M.lrReference()[st.h], n = chain.n;
      // left: contours and recent states
      var svg = svgs[0]; WK.clear(svg);
      var xs = WK.lin([V[0], V[1]], [m.l, W - m.r]), ys = WK.lin([V[2], V[3]], [H - m.b, m.t]);
      WK.ticks(V[2], V[3], 4).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), stroke: "rule" }));
        text(svg, m.l - 6, ys(t) + 4, String(t), "end");
      });
      WK.ticks(V[0], V[1], 5).forEach(function (t) { text(svg, xs(t), H - m.b + 16, String(t)); });
      text(svg, (m.l + W - m.r) / 2, H - 6, "θ₁");
      text(svg, m.l, m.t - 6, "θ₂", "start");
      var gx = function (u) { return xs(V[0] + (u + 0.5) / GD * (V[1] - V[0])); };
      var gy = function (u) { return ys(V[3] - (u + 0.5) / GD * (V[3] - V[2])); };
      var dpath = segs.map(function (s) { return "M" + gx(s[0][0]).toFixed(1) + "," + gy(s[0][1]).toFixed(1) + "L" + gx(s[1][0]).toFixed(1) + "," + gy(s[1][1]).toFixed(1); }).join("");
      svg.appendChild(WK.h("path", { d: dpath, fill: "none", stroke: "c2", "stroke-width": 1.4 }));
      for (var k = Math.max(0, n - 400); k < n; k++) {
        svg.appendChild(WK.h("circle", { cx: xs(chain.theta[2 * k]), cy: ys(chain.theta[2 * k + 1]), r: 1.8, fill: "c3", "fill-opacity": 0.7 }));
      }
      // right: running estimates
      svg = svgs[1]; WK.clear(svg);
      var cps = [], m0 = Math.min(100, n);
      for (var q = 0; q <= 40; q++) { var mm = Math.round(m0 * Math.pow(n / m0, q / 40)); if (!cps.length || mm > cps[cps.length - 1]) cps.push(mm); }
      var est = cps.map(function (mm) { return M.zv(chain, st.h, mm); });
      var all = [ref]; est.forEach(function (e) { all.push(e.plain, e.deg1, e.deg2); });
      var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pad = 0.08 * (hi - lo || 1);
      var lx = logScale(m0, n, m.l, W - m.r), ly = WK.lin([lo - pad, hi + pad], [H - m.b, m.t]);
      WK.ticks(lo - pad, hi + pad, 5).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ly(t), y2: ly(t), stroke: "rule" }));
        text(svg, m.l - 6, ly(t) + 4, String(+t.toFixed(3)), "end");
      });
      lx.ticks.forEach(function (t) { text(svg, lx(t), H - m.b + 16, fmtTick(t)); });
      text(svg, (m.l + W - m.r) / 2, H - 6, "states used, m (log scale)");
      text(svg, m.l, m.t - 6, M.LR_H[st.h].label, "start");
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ly(ref), y2: ly(ref), stroke: "ink", "stroke-width": 1.2, "stroke-dasharray": "5 4" }));
      [["plain", "muted"], ["deg1", "c3"], ["deg2", "c1"]].forEach(function (p) {
        svg.appendChild(poly(cps.map(function (mm, i) { return lx(mm) + "," + ly(est[i][p[0]]); }), p[1], 2.2));
      });
      text(svg, W - m.r - 2, m.t + 10, "plain average", "end", "muted");
      text(svg, W - m.r - 2, m.t + 25, "degree 1", "end", "c3");
      text(svg, W - m.r - 2, m.t + 40, "degree 2", "end", "c1");
      var last = est[est.length - 1];
      sOne.set([last.plain, last.deg1, last.deg2].map(function (e) { return sig(e - ref, 2); }).join(" / "));
      sMse.set(sig(rep.mse.plain, 3));
      s1.set("×" + sig(rep.vr1, 3));
      s2.set("×" + sig(rep.vr2, 3));
      sAcc.set(chain.acc.toFixed(2));
    }
    recompute();
  });
})();
