/* Two charts for Part 4. Numbers come from SteinModelsModel (model.js); this
 * file only draws. */
"use strict";

(function () {
  var M = SteinModelsModel;

  function text(svg, x, y, s, anchor, fill, size) {
    svg.appendChild(WK.h("text", { x: x, y: y, "text-anchor": anchor || "middle", fill: fill || "muted", "font-size": size || 12 }, s));
  }
  function axes(svg, W, H, m, xs, ys, xlab, ylab, yfmt) {
    ys.ticks.forEach(function (t) {
      svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: ys.f(t), y2: ys.f(t), stroke: "rule" }));
      text(svg, m.l - 6, ys.f(t) + 4, yfmt ? yfmt(t) : String(t), "end");
    });
    xs.ticks.forEach(function (t) { text(svg, xs.f(t), H - m.b + 16, String(t)); });
    svg.appendChild(WK.h("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "muted" }));
    text(svg, (m.l + W - m.r) / 2, H - 6, xlab);
    if (ylab) text(svg, m.l, m.t - 2, ylab, "start");
  }
  function poly(pts, colour, width, dash) {
    return WK.h("polyline", { points: pts.join(" "), fill: "none", stroke: colour, "stroke-width": width || 2, "stroke-dasharray": dash || null });
  }

  // ------------------------------------------------ 1. the constant does not matter
  WK.mount("widget-constant", function (root, WK) {
    var xs = M.sample(60, 4), logc = 0;
    var f = WK.frame({
      title: "Multiplying an unnormalised density by c",
      badge: "interactive · synthetic sample",
      note: "The density is p̃(x) = 0.6 exp(−(x + 1.5)² / 0.98) + 0.4 exp(−(x − 1.5)² / 2), written without its normalising constant, times c. " +
        "Ticks under the left panel are a fixed sample of 60 draws from the normalised density. Every tile is computed from c p̃ or from its log-derivative."
    });
    var cS = WK.slider({ label: "Multiplier c (log scale)", min: -3, max: 3, step: 0.05, value: logc,
      fmt: function (v) { var c = Math.pow(10, v); return c < 0.01 || c >= 100 ? c.toExponential(1) : c.toFixed(c < 1 ? 3 : 2); },
      oninput: function (v) { logc = v; draw(); } });
    f.controls.appendChild(cS.root);
    var row = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.8rem" } });
    var W = 420, H = 260;
    var svgD = WK.svg(W, H, { "aria-label": "The density c times p-tilde, with the fixed sample as ticks." });
    var svgS = WK.svg(W, H, { "aria-label": "The score of c times p-tilde, which does not change with c." });
    [svgD, svgS].forEach(function (s) { row.appendChild(WK.el("div", { style: { flex: "1 1 300px", minWidth: "0" } }, s)); });
    f.body.appendChild(row);
    var tMass = WK.stat({ label: "Total mass ∫ c p̃", value: "" });
    var tLL = WK.stat({ label: "Log-likelihood, treating c p̃ as a density", value: "" });
    var tK = WK.stat({ label: "KSD of the sample (IMQ)", value: "" });
    var tSM = WK.stat({ label: "Score-matching objective Ĵ", value: "" });
    [tMass, tLL, tK, tSM].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    var m = { l: 52, r: 10, t: 22, b: 34 }, X0 = -5, X1 = 5;
    function draw() {
      var c = Math.pow(10, logc), grid = [], i;
      for (i = 0; i <= 200; i++) grid.push(X0 + (X1 - X0) * i / 200);
      // left: c p~ on an axis that follows c, so its shape stays and its scale moves
      WK.clear(svgD);
      var peak = 0; grid.forEach(function (x) { peak = Math.max(peak, c * M.ptilde(x)); });
      var top = peak * 1.15, x = WK.lin([X0, X1], [m.l, W - m.r]), y = WK.lin([0, top], [H - m.b, m.t]);
      var yt = WK.ticks(0, top, 4);
      axes(svgD, W, H, m, { f: x, ticks: [-4, -2, 0, 2, 4] }, { f: y, ticks: yt }, "x", "c · p̃(x)",
        function (t) { return t === 0 ? "0" : (Math.abs(t) >= 1000 || Math.abs(t) < 0.01 ? t.toExponential(0) : String(+t.toPrecision(3))); });
      svgD.appendChild(poly(grid.map(function (g) { return x(g) + "," + y(c * M.ptilde(g)); }), "c1", 2.4));
      xs.forEach(function (v) { svgD.appendChild(WK.h("line", { x1: x(v), x2: x(v), y1: H - m.b, y2: H - m.b - 8, stroke: "c3", "stroke-width": 1.4 })); });
      // right: the score, on a fixed axis
      WK.clear(svgS);
      var ys = WK.lin([-5, 5], [H - m.b, m.t]), scoreC = M.scoreOf(c);
      axes(svgS, W, H, m, { f: x, ticks: [-4, -2, 0, 2, 4] }, { f: ys, ticks: [-4, -2, 0, 2, 4] }, "x", "score  d/dx log(c p̃)");
      svgS.appendChild(poly(grid.map(function (g) { return x(g) + "," + ys(Math.max(-5, Math.min(5, scoreC(g)))); }), "c2", 2.4));
      var st = M.constantStats(c, xs);
      tMass.set(st.mass < 0.01 || st.mass >= 1000 ? st.mass.toExponential(3) : st.mass.toFixed(3));
      tLL.set(st.loglik.toFixed(2));
      tK.set(st.ksd.toFixed(6));
      tSM.set(st.scoreMatching.toFixed(6));
    }
    draw();
  });

  // ------------------------------------------------ 2. SURE against the true loss
  WK.mount("widget-sure", function (root, WK) {
    var st = { kind: "soft", sigma: 1, lamFrac: 0.25, seed: 1 };
    var mu = M.signal(256, 25, 1), y = null, cur = null;
    var f = WK.frame({
      title: "Choosing λ by SURE, without the clean signal",
      badge: "interactive · synthetic signal",
      note: "A synthetic sparse signal: 256 coordinates, 25 of them nonzero with magnitudes between 2 and 6, observed once in Gaussian noise of known σ. " +
        "Left: the true signal (stems), the observation (dots) and the estimate at the current λ (line). Right: SURE(λ), computed from the observation alone, and the true loss |μ̂(λ) − μ|², which needs the clean signal. " +
        "Circles mark each curve's minimiser."
    });
    var kindT = WK.toggle({ label: "Estimator", value: st.kind,
      options: [{ value: "soft", label: "Soft threshold" }, { value: "ridge", label: "Ridge shrinkage y / (1 + λ)" }],
      onchange: function (v) { st.kind = v; recompute(); } });
    var sigS = WK.slider({ label: "Noise σ", min: 0.25, max: 2, step: 0.05, value: st.sigma,
      fmt: function (v) { return Number(v).toFixed(2); }, oninput: function (v) { st.sigma = v; recompute(); } });
    var lamS = WK.slider({ label: "λ", min: 0, max: 1, step: 0.0025, value: st.lamFrac,
      fmt: function (v) { return lamOf(v).toFixed(2); }, oninput: function (v) { st.lamFrac = v; draw(); } });
    var resample = WK.el("button", { type: "button", class: "widget-toggle-btn", text: "New noise draw",
      style: { border: "1px solid var(--w-rule)", borderRadius: "999px" },
      onclick: function () { st.seed += 1; recompute(); } });
    [kindT, sigS, lamS].forEach(function (c) { f.controls.appendChild(c.root); });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } }, resample));
    function lamMax() { return st.kind === "soft" ? 4 * st.sigma : 3; }
    function lamOf(frac) { return frac * lamMax(); }

    var row = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.8rem" } });
    var W = 420, H = 280;
    var svgSig = WK.svg(W, H, { "aria-label": "True signal, noisy observation and the estimate at the current lambda." });
    var svgC = WK.svg(W, H, { "aria-label": "SURE and the true loss against lambda." });
    [svgSig, svgC].forEach(function (s) { row.appendChild(WK.el("div", { style: { flex: "1 1 300px", minWidth: "0" } }, s)); });
    f.body.appendChild(row);
    var tCur = WK.stat({ label: "At this λ: SURE / true loss", value: "" });
    var tSure = WK.stat({ label: "λ chosen by SURE (true loss there)", value: "" });
    var tOr = WK.stat({ label: "Oracle λ (true loss there)", value: "" });
    var tRaw = WK.stat({ label: "True loss of the raw observation", value: "" });
    [tCur, tSure, tOr, tRaw].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    function recompute() {
      y = M.observe(mu, st.sigma, 100 + st.seed);
      cur = M.curves(st.kind, mu, y, st.sigma);
      lamS.set(st.lamFrac, true);
      draw();
    }
    var m = { l: 52, r: 12, t: 22, b: 34 };
    function draw() {
      var lam = lamOf(st.lamFrac), est = M.estimate(st.kind, y, lam);
      // left: the first 64 coordinates, so individual spikes stay readable
      WK.clear(svgSig);
      // the axis is ±9 unless the noise throws an observation further out; the estimate never
      // exceeds |y|, so dots, spikes and line all stay inside it
      var N = 64, amax = 9;
      for (var a = 0; a < N; a++) amax = Math.max(amax, Math.abs(y[a]) + 0.5, Math.abs(mu[a]) + 0.5);
      var ylo = -amax, yhi = amax, x = WK.lin([0, N - 1], [m.l, W - m.r]), yy = WK.lin([ylo, yhi], [H - m.b, m.t]);
      var yticks = []; for (var t4 = -4 * Math.floor(amax / 4); t4 <= amax; t4 += 4) yticks.push(t4);
      axes(svgSig, W, H, m, { f: x, ticks: [0, 16, 32, 48, 63] }, { f: yy, ticks: yticks }, "coordinate (first 64 of 256)", "value");
      for (var i = 0; i < N; i++) {
        if (mu[i] !== 0) svgSig.appendChild(WK.h("line", { x1: x(i), x2: x(i), y1: yy(0), y2: yy(mu[i]), stroke: "muted", "stroke-width": 3, "stroke-opacity": 0.55 }));
        svgSig.appendChild(WK.h("circle", { cx: x(i), cy: yy(y[i]), r: 2.4, fill: "c3" }));
      }
      var pts = [];
      for (i = 0; i < N; i++) pts.push(x(i) + "," + yy(est[i]));
      svgSig.appendChild(poly(pts, "c1", 1.8));
      // right: curves
      WK.clear(svgC);
      var g = cur.lam, hi = Math.max(cur.lossAtZero, Math.max.apply(null, cur.sure), Math.max.apply(null, cur.loss)) * 1.05;
      var lo = Math.min(0, Math.min.apply(null, cur.sure));
      var xl = WK.lin([0, lamMax()], [m.l, W - m.r]), yl = WK.lin([lo, hi], [H - m.b, m.t]);
      axes(svgC, W, H, m, { f: xl, ticks: WK.ticks(0, lamMax(), 4) }, { f: yl, ticks: WK.ticks(lo, hi, 4) }, "λ", "squared error");
      svgC.appendChild(poly(g.map(function (l, k) { return xl(l) + "," + yl(cur.loss[k]); }), "c3", 2.2, "6 3"));
      svgC.appendChild(poly(g.map(function (l, k) { return xl(l) + "," + yl(cur.sure[k]); }), "c1", 2.2));
      svgC.appendChild(WK.h("circle", { cx: xl(cur.lamSure), cy: yl(Math.min.apply(null, cur.sure)), r: 5, fill: "none", stroke: "c1", "stroke-width": 2 }));
      svgC.appendChild(WK.h("circle", { cx: xl(cur.lamOracle), cy: yl(cur.lossAtOracle), r: 5, fill: "none", stroke: "c3", "stroke-width": 2 }));
      svgC.appendChild(WK.h("line", { x1: xl(lam), x2: xl(lam), y1: m.t, y2: H - m.b, stroke: "ink", "stroke-width": 1, "stroke-dasharray": "2 3" }));
      text(svgC, W - m.r - 2, m.t + 12, "SURE (solid)", "end", "c1");
      text(svgC, W - m.r - 2, m.t + 28, "true loss (dashed)", "end", "c3");
      tCur.set(M.sure(st.kind, y, lam, st.sigma).toFixed(1) + " / " + M.loss(st.kind, y, mu, lam).toFixed(1));
      tSure.set(cur.lamSure.toFixed(2) + " (" + cur.lossAtSure.toFixed(1) + ")");
      tOr.set(cur.lamOracle.toFixed(2) + " (" + cur.lossAtOracle.toFixed(1) + ")");
      tRaw.set(cur.lossAtZero.toFixed(1));
    }
    recompute();
  });
})();
