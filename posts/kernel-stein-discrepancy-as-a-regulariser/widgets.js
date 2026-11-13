/* What a regulariser does to a free 2-D batch: the force it puts on every point,
 * how far it places the batch from a Gaussian batch of the same size, and where
 * gradient descent on the points alone takes them. Every number comes from
 * KsdRegModel (model.js); this file only draws. The scene is a canvas, so it reads
 * token colours at each draw and repaints on a theme change. */
"use strict";

(function () {
  var M = KsdRegModel;
  var MAX_STEPS = 3000, ETA = 0.02, NULL_REPS = 80;
  // V-statistics: a fixed set of points needs a loss that is bounded below, and the
  // V-statistic of KSD is the squared length of the SVGD step, never negative.
  var OPT = { ustat: false };

  WK.mount("widget-forces", function (root, WK) {
    var st = { law: "ring", reg: "moments", n: 150, seed: 1 };
    var sim = null, playing = false, raf = null, nullCache = {}, nullPending = {};

    function readPalette() {
      var cs = getComputedStyle(document.documentElement), p = {};
      WK.tokens.forEach(function (t) { p[t] = cs.getPropertyValue("--w-" + t).trim() || "#888888"; });
      return p;
    }

    var f = WK.frame({
      title: "What a regulariser does to a batch",
      badge: "interactive · synthetic 2-D batches",
      note: "Dots: a batch of codes. Arrows: the direction gradient descent on the selected regulariser moves each code, " +
        "scaled so that the longest arrow on a shrunk batch, N(0, 0.3² I), has the same length for every regulariser. " +
        "Dashed circles: radii 1 and 2 of the target N(0, I). " +
        "Bars: each regulariser's value on the current batch, in standard deviations above its mean on Gaussian batches of the same size; " +
        "the tick marks where 5% of Gaussian batches land. Every batch but “Shrunk” has mean 0 and covariance I."
    });

    function ctl(c) { f.controls.appendChild(c.root); return c; }
    ctl(WK.toggle({ label: "Batch", value: st.law,
      options: [{ value: "gaussian", label: "Gaussian" }, { value: "ring", label: "Ring" },
                { value: "clusters", label: "Four clusters" }, { value: "student", label: "Heavy tails" },
                { value: "square", label: "Square" }, { value: "shrunk", label: "Shrunk" }],
      onchange: function (v) { st.law = v; reset(); } }));
    var regT = ctl(WK.toggle({ label: "Regulariser", value: st.reg,
      options: M.REGS.map(function (r) { return { value: r, label: M.LABELS[r] }; }),
      onchange: function (v) { st.reg = v; sim.gref = gref(); render(); } }));
    regT.root.style.flex = "1 1 100%";
    ctl(WK.slider({ label: "Batch size n", min: 50, max: 300, step: 50, value: st.n,
      fmt: WK.fmt.int, oninput: function (v) { st.n = Math.round(v); reset(); } }));
    var btn = function (label, fn) {
      return WK.el("button", { type: "button", class: "widget-toggle-btn", text: label, onclick: fn,
        style: { border: "1px solid var(--w-rule)", borderRadius: "999px", marginRight: "0.4rem" } });
    };
    var playB = btn("Descend", function () { playing ? pause() : play(); });
    var stepB = btn("10 steps", function () { pause(); advance(10); render(); });
    var resetB = btn("New draw", function () { st.seed += 1; reset(); });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "1 1 100%" } }, playB, stepB, resetB));

    var row = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.8rem", alignItems: "flex-start" } });
    var CW = 460, VIEW = 3.4;
    var canvas = WK.el("canvas", { width: CW * 2, height: CW * 2, role: "img",
      "aria-label": "A two-dimensional batch of points with an arrow on each point showing the regulariser's descent direction.",
      style: { width: "100%", height: "auto", display: "block", borderRadius: "6px", border: "1px solid var(--w-rule)" } });
    var W = 400, H = 300;
    var chart = WK.svg(W, H, { "aria-label": "Each regulariser's value on the batch, in null standard deviations." });
    row.appendChild(WK.el("div", { style: { flex: "1 1 300px", minWidth: "0", maxWidth: "460px" } }, canvas));
    row.appendChild(WK.el("div", { style: { flex: "1 1 280px", minWidth: "0" } }, chart));
    f.body.appendChild(row);

    var sT = WK.stat({ label: "Descent steps", value: "" });
    var sR2 = WK.stat({ label: "Mean |z|² (Gaussian: 2)", value: "" });
    var sR4 = WK.stat({ label: "Mean |z|⁴ (Gaussian: 8)", value: "" });
    var sZ = WK.stat({ label: "Selected regulariser, null SDs", value: "" });
    [sT, sR2, sR4, sZ].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    // ------------------------------------------------------------ model glue
    function gref() {
      return M.maxNorm(M.evaluate(st.reg, M.sample("shrunk", st.n, 424242), Object.assign({ seed: 5 }, OPT)).g);
    }
    function nullFor(reg, n) {
      var key = reg + "|" + n;
      if (nullCache[key]) return nullCache[key];
      if (!nullPending[key]) {
        nullPending[key] = true;
        setTimeout(function () { nullCache[key] = M.nullStats(reg, n, NULL_REPS, 1, OPT); render(); }, 0);
      }
      return null;
    }
    function reset() {
      pause();
      sim = { X: M.sample(st.law, st.n, st.seed), t: 0, nrm: M.normalGen(M.rng(st.seed * 31 + 7)) };
      sim.gref = gref();
      render();
    }
    function advance(k) {
      for (var s = 0; s < k && sim.t < MAX_STEPS; s++) {
        M.descend(sim.X, st.reg, 1, ETA, sim.gref, Object.assign({ nrm: sim.nrm }, OPT));
        sim.t++;
      }
      if (sim.t >= MAX_STEPS) pause();
    }
    var frame = 0, lastZ = {};
    function loop() {
      if (!playing) return;
      advance(st.n <= 150 ? 4 : 2);
      // Scoring the batch under all five regularisers is the costly part of a frame:
      // refresh the bars every eighth frame while descending, and fully on pause.
      render(++frame % 8 === 0);
      raf = requestAnimationFrame(loop);
    }
    function play() { if (sim.t >= MAX_STEPS) reset(); playing = true; playB.textContent = "Pause"; raf = requestAnimationFrame(loop); }
    function pause() {
      var was = playing;
      playing = false; playB.textContent = "Descend"; if (raf) cancelAnimationFrame(raf); raf = null;
      if (was && sim) render();
    }

    // ------------------------------------------------------------- drawing
    function drawScene(g) {
      var pal = readPalette(), ctx = canvas.getContext("2d"), S = CW * 2;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = pal.surface; ctx.fillRect(0, 0, S, S);
      function px(x) { return (x + VIEW) / (2 * VIEW) * S; }
      function py(y) { return (VIEW - y) / (2 * VIEW) * S; }
      var sc = S / (2 * VIEW);
      ctx.strokeStyle = pal.rule; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px(0), 0); ctx.lineTo(px(0), S); ctx.moveTo(0, py(0)); ctx.lineTo(S, py(0)); ctx.stroke();
      ctx.setLineDash([10, 8]); ctx.strokeStyle = pal.muted; ctx.lineWidth = 2;
      [1, 2].forEach(function (r) { ctx.beginPath(); ctx.arc(px(0), py(0), r * sc, 0, 2 * Math.PI); ctx.stroke(); });
      ctx.setLineDash([]);
      var X = sim.X, n = X.length / 2, i, L = 70; // longest arrow on the shrunk batch, in canvas px
      ctx.strokeStyle = pal.c3; ctx.fillStyle = pal.c3; ctx.lineWidth = 2.4;
      for (i = 0; i < n; i++) {
        var x0 = px(X[2 * i]), y0 = py(X[2 * i + 1]);
        var dx = -g[2 * i] / sim.gref * L, dy = g[2 * i + 1] / sim.gref * L, len = Math.hypot(dx, dy);
        if (len < 1.5) continue;
        if (len > 3 * L) { dx *= 3 * L / len; dy *= 3 * L / len; len = 3 * L; }
        var x1 = x0 + dx, y1 = y0 + dy, ux = dx / len, uy = dy / len, hw = Math.min(9, 0.35 * len);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * hw * 0.8, y1 - uy * hw * 0.8); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - ux * hw - uy * hw * 0.5, y1 - uy * hw + ux * hw * 0.5);
        ctx.lineTo(x1 - ux * hw + uy * hw * 0.5, y1 - uy * hw - ux * hw * 0.5);
        ctx.closePath(); ctx.fill();
      }
      ctx.fillStyle = pal.c1;
      for (i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(px(X[2 * i]), py(X[2 * i + 1]), 5.5, 0, 2 * Math.PI); ctx.fill(); }
      ctx.fillStyle = pal.muted; ctx.font = "26px system-ui, sans-serif";
      ctx.fillText(M.LABELS[st.reg] + " · step " + sim.t, 18, 40);
    }

    function zOf(reg, X) {
      var ns = nullFor(reg, st.n);
      if (!ns) return null;
      var v = M.evaluate(reg, X, Object.assign({ seed: st.seed * 17 + 3 }, OPT)).v;
      return { z: (v - ns.mean) / ns.sd, z95: (ns.q95 - ns.mean) / ns.sd };
    }
    function sym(z) { return z >= 0 ? Math.log10(1 + z) : -Math.log10(1 - z); }
    function text(x, y, s, anchor, fill, extra) {
      chart.appendChild(WK.h("text", Object.assign({ x: x, y: y, "text-anchor": anchor || "middle", fill: fill || "muted", "font-size": 12 }, extra || {}), s));
    }
    function drawChart() {
      WK.clear(chart);
      var ml = 118, mr = 14, mt = 26, mb = 34, bh = (H - mt - mb) / M.REGS.length;
      var lo = -1, hi = 3, x = WK.lin([lo, hi], [ml, W - mr]);
      text(ml, 14, "null SDs above a Gaussian batch (log scale)", "start", "muted");
      [0, 1, 10, 100, 1000].forEach(function (t) {
        var xx = x(sym(t));
        chart.appendChild(WK.h("line", { x1: xx, x2: xx, y1: mt, y2: H - mb, stroke: "rule" }));
        text(xx, H - mb + 16, String(t));
      });
      var xm = x(sym(-3));
      text(xm, H - mb + 16, "−3");
      var out = {};
      M.REGS.forEach(function (reg, k) {
        var y = mt + k * bh, r = zOf(reg, sim.X), sel = reg === st.reg;
        text(ml - 8, y + bh / 2 + 4, M.LABELS[reg], "end", sel ? "ink" : "muted", sel ? { "font-weight": 700 } : null);
        if (!r) { text(ml + 6, y + bh / 2 + 4, "computing…", "start", "muted"); return; }
        out[reg] = r;
        var z = Math.max(-3, Math.min(999, r.z)), x0 = x(0), x1 = x(sym(z));
        var hit = r.z > r.z95;
        chart.appendChild(WK.h("rect", { x: Math.min(x0, x1), y: y + bh * 0.22, width: Math.max(1.5, Math.abs(x1 - x0)), height: bh * 0.56,
          rx: 3, fill: hit ? (sel ? "c1" : "c5") : "rule" }));
        var xt = x(sym(r.z95));
        chart.appendChild(WK.h("line", { x1: xt, x2: xt, y1: y + bh * 0.12, y2: y + bh * 0.88, stroke: "ink", "stroke-width": 1.5 }));
        text(Math.max(x0, x1) + 5, y + bh / 2 + 4, WK.fmt.num(r.z, r.z < 10 ? 1 : 0), "start", hit ? "ink" : "muted");
      });
      return out;
    }

    function render(withChart) {
      var cur = M.evaluate(st.reg, sim.X, Object.assign({ seed: st.seed * 17 + 3 }, OPT));
      drawScene(cur.g);
      if (withChart !== false) lastZ = drawChart();
      var zs = lastZ, rm = M.radialMoments(sim.X);
      sT.set(String(sim.t));
      sR2.set(WK.fmt.num(rm.r2, 2));
      sR4.set(WK.fmt.num(rm.r4, 2));
      sZ.set(zs[st.reg] ? WK.fmt.num(zs[st.reg].z, 1) : "…");
    }

    function repaintSoon() {
      [0, 150, 600].forEach(function (ms) { setTimeout(function () { if (sim && !playing) render(); }, ms); });
    }
    WK.onTheme(repaintSoon);
    window.addEventListener("load", repaintSoon);
    reset();
  });
})();
