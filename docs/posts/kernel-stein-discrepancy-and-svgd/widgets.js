/* SVGD against unadjusted Langevin on a 2-D target, with a live IMQ KSD trace.
 * All numbers come from SteinSvgdModel (model.js); this file only draws. The
 * scene is a canvas, so it reads token colours from WK.palette() and redraws
 * on a theme change. */
"use strict";

(function () {
  var M = SteinSvgdModel;
  var MAX_STEPS = 3000, STEPS_PER_FRAME = 2;

  WK.mount("widget-svgd", function (root, WK) {
    var EPS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2];
    var st = { target: "banana", sep: 4, n: 100, epsIdx: 4, median: true, ell: 0.5,
               spread: 1, cx: 0, showUla: true, seed: 1 };
    // Read the colour tokens fresh at each draw: WK.palette() can cache the light
    // values when the theme toggle fires before the dark stylesheet has applied.
    function readPalette() {
      var cs = getComputedStyle(document.documentElement), p = {};
      WK.tokens.forEach(function (t) { p[t] = cs.getPropertyValue("--w-" + t).trim() || "#888888"; });
      p.key = p.surface + p.c2 + p.muted;
      return p;
    }
    var sim = null, playing = false, raf = null;

    var f = WK.frame({
      title: "SVGD against unadjusted Langevin",
      badge: "interactive · synthetic targets",
      note: "Filled dots: SVGD particles. Hollow dots: independent unadjusted Langevin (ULA) chains, one per particle, run for the same number of steps, so both methods use the same number of score evaluations. " +
        "Shading and contours show the unnormalised target density; its normalising constant is never computed. " +
        "The right panel is the IMQ kernel Stein discrepancy (c = 1, β = −1/2, V-statistic) of each point set, on a log scale."
    });

    function ctl(c) { f.controls.appendChild(c.root); return c; }
    var targetT = ctl(WK.toggle({ label: "Target", value: st.target,
      options: [{ value: "banana", label: "Banana" }, { value: "mixture", label: "Mixture" }, { value: "donut", label: "Donut" }],
      onchange: function (v) { st.target = v; syncSep(); reset(); } }));
    var sepS = ctl(WK.slider({ label: "Mixture separation", min: 1, max: 8, step: 0.5, value: st.sep,
      fmt: function (v) { return Number(v).toFixed(1); }, oninput: function (v) { st.sep = v; reset(); } }));
    var nS = ctl(WK.slider({ label: "Particles n", min: 10, max: 300, step: 10, value: st.n,
      fmt: WK.fmt.int, oninput: function (v) { st.n = Math.round(v); reset(); } }));
    var epsS = ctl(WK.slider({ label: "Step size ε", min: 0, max: EPS.length - 1, step: 1, value: st.epsIdx,
      fmt: function (v) { return String(EPS[Math.round(v)]); }, oninput: function (v) { st.epsIdx = Math.round(v); } }));
    var medC = WK.check({ label: "Median-heuristic bandwidth", value: st.median,
      onchange: function (v) { st.median = v; syncEll(); } });
    var ellS = ctl(WK.slider({ label: "Bandwidth ℓ", min: 0.05, max: 3, step: 0.05, value: st.ell,
      fmt: function (v) { return Number(v).toFixed(2); }, oninput: function (v) { st.ell = v; } }));
    ellS.root.insertBefore(medC.root, ellS.root.firstChild);
    var spreadS = ctl(WK.slider({ label: "Initial spread", min: 0.1, max: 3, step: 0.1, value: st.spread,
      fmt: function (v) { return Number(v).toFixed(1); }, oninput: function (v) { st.spread = v; reset(); } }));
    var cxS = ctl(WK.slider({ label: "Initial centre x₀", min: -5, max: 5, step: 0.5, value: st.cx,
      fmt: function (v) { return Number(v).toFixed(1); }, oninput: function (v) { st.cx = v; reset(); } }));
    var ulaC = WK.check({ label: "Show Langevin chains", value: st.showUla,
      onchange: function (v) { st.showUla = v; render(); } });
    var btn = function (label, fn) {
      return WK.el("button", { type: "button", class: "widget-toggle-btn", text: label, onclick: fn,
        style: { border: "1px solid var(--w-rule)", borderRadius: "999px", marginRight: "0.4rem" } });
    };
    var playB = btn("Play", function () { playing ? pause() : play(); });
    var stepB = btn("Step", function () { pause(); advance(1); render(); });
    var resetB = btn("Reset", function () { st.seed += 1; reset(); });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "1 1 100%" } },
      playB, stepB, resetB, " ", ulaC.root));

    function syncSep() {
      sepS.input.disabled = st.target !== "mixture";
      sepS.root.style.opacity = st.target === "mixture" ? "1" : "0.45";
    }
    function syncEll() {
      ellS.input.disabled = st.median;
      ellS.input.style.opacity = st.median ? "0.45" : "1";
    }

    // ---------------------------------------------------------------- layout
    var row = WK.el("div", { style: { display: "flex", flexWrap: "wrap", gap: "0.8rem", alignItems: "flex-start" } });
    var CW = 520;
    var canvas = WK.el("canvas", { width: CW, height: CW, role: "img",
      "aria-label": "Particles over the contours of the target density.",
      style: { width: "100%", height: "auto", display: "block", borderRadius: "6px", border: "1px solid var(--w-rule)" } });
    var W = 420, H = 300;
    var chart = WK.svg(W, H, { "aria-label": "Kernel Stein discrepancy against score evaluations for SVGD and Langevin." });
    row.appendChild(WK.el("div", { style: { flex: "1 1 300px", minWidth: "0", maxWidth: "520px" } }, canvas));
    row.appendChild(WK.el("div", { style: { flex: "1 1 280px", minWidth: "0" } }, chart));
    f.body.appendChild(row);

    var sT = WK.stat({ label: "Iteration (score evaluations each)", value: "" });
    var sA = WK.stat({ label: "KSD, SVGD", value: "" });
    var sB = WK.stat({ label: "KSD, Langevin", value: "" });
    var sL = WK.stat({ label: "Bandwidth ℓ in use", value: "" });
    var sM = WK.stat({ label: "Share of SVGD particles with x > 0 (target)", value: "" });
    [sT, sA, sB, sL, sM].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    // ------------------------------------------------------------ simulation
    function reset() {
      var T = M.target(st.target, { sep: st.sep });
      var A = M.init(st.n, st.spread, st.seed, [st.cx, 0]);
      sim = { T: T, A: A, B: Float64Array.from(A), nrm: M.normalGen(M.rng(st.seed + 7919)), t: 0,
              ulaDead: false, svgdDead: false, trace: [], ell: st.median ? M.medianEll(A) : st.ell, bg: null };
      record();
      buildBackground();
      sM.root.style.display = st.target === "mixture" ? "" : "none";
      render();
    }
    function record() {
      var a = sim.svgdDead ? NaN : M.ksd(sim.A, sim.T), b = sim.ulaDead ? NaN : M.ksd(sim.B, sim.T);
      if (!isFinite(a)) sim.svgdDead = true;
      if (!isFinite(b)) sim.ulaDead = true;
      sim.trace.push({ t: sim.t, a: a, b: b });
    }
    function advance(k) {
      var eps = EPS[st.epsIdx];
      for (var s = 0; s < k && sim.t < MAX_STEPS; s++) {
        // A large step can throw particles past the score's steep walls: stop that sampler and say so.
        if (!sim.svgdDead) {
          sim.ell = st.median ? M.medianEll(sim.A) : st.ell;
          M.svgdStep(sim.A, sim.T, eps, sim.ell);
          for (var j = 0; j < sim.A.length; j++) if (!isFinite(sim.A[j]) || Math.abs(sim.A[j]) > 1e6) { sim.svgdDead = true; break; }
        }
        if (!sim.ulaDead) {
          M.ulaStep(sim.B, sim.T, eps, sim.nrm);
          for (var i = 0; i < sim.B.length; i++) if (!isFinite(sim.B[i]) || Math.abs(sim.B[i]) > 1e6) { sim.ulaDead = true; break; }
        }
        sim.t++;
      }
      record();
      if (sim.t >= MAX_STEPS || (sim.svgdDead && sim.ulaDead)) pause();
    }
    function loop() {
      if (!playing) return;
      advance(STEPS_PER_FRAME);
      render();
      raf = requestAnimationFrame(loop);
    }
    function play() { if (sim.t >= MAX_STEPS) reset(); playing = true; playB.textContent = "Pause"; raf = requestAnimationFrame(loop); }
    function pause() { playing = false; playB.textContent = "Play"; if (raf) cancelAnimationFrame(raf); raf = null; }

    // ------------------------------------------------------------- drawing
    var G = 130; // density grid
    function buildBackground() {
      var v = sim.T.view, lp = new Float64Array(G * G), mx = -Infinity, i, j;
      for (j = 0; j < G; j++) for (i = 0; i < G; i++) {
        var x = v[0] + (i + 0.5) / G * (v[1] - v[0]), y = v[3] - (j + 0.5) / G * (v[3] - v[2]);
        var l = sim.T.logp(x, y); lp[j * G + i] = l; if (l > mx) mx = l;
      }
      var dens = new Float64Array(G * G);
      for (i = 0; i < G * G; i++) dens[i] = Math.exp(lp[i] - mx); // relative density in [0, 1]
      sim.bg = { dens: dens, segs: contours(dens, [0.05, 0.25, 0.5, 0.75]) };
      sim.bgCanvas = null;
    }
    // marching squares on cell centres; returns segments in grid units
    function contours(D, levels) {
      var out = [];
      levels.forEach(function (lev) {
        for (var j = 0; j < G - 1; j++) for (var i = 0; i < G - 1; i++) {
          var a = D[j * G + i], b = D[j * G + i + 1], c = D[(j + 1) * G + i + 1], d = D[(j + 1) * G + i];
          var pts = [];
          function edge(p, q, x0, y0, x1, y1) {
            if ((p >= lev) !== (q >= lev)) { var t = (lev - p) / (q - p); pts.push([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]); }
          }
          edge(a, b, i, j, i + 1, j); edge(b, c, i + 1, j, i + 1, j + 1);
          edge(c, d, i + 1, j + 1, i, j + 1); edge(d, a, i, j + 1, i, j);
          if (pts.length >= 2) out.push([pts[0], pts[1]]);
          if (pts.length === 4) out.push([pts[2], pts[3]]);
        }
      });
      return out;
    }
    function hexToRgb(c) {
      var m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(c.trim());
      if (m) return [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)];
      var r = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(c);
      return r ? [+r[1], +r[2], +r[3]] : [128, 128, 128];
    }
    function background(pal) {
      if (sim.bgCanvas && sim.bgPal === pal.key) return sim.bgCanvas;
      var off = document.createElement("canvas"); off.width = G; off.height = G;
      var g = off.getContext("2d"), img = g.createImageData(G, G);
      var s = hexToRgb(pal.surface), t = hexToRgb(pal.c2);
      for (var k = 0; k < G * G; k++) {
        var w = 0.55 * Math.sqrt(sim.bg.dens[k]);
        img.data[4 * k] = s[0] + w * (t[0] - s[0]);
        img.data[4 * k + 1] = s[1] + w * (t[1] - s[1]);
        img.data[4 * k + 2] = s[2] + w * (t[2] - s[2]);
        img.data[4 * k + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      sim.bgCanvas = off; sim.bgPal = pal.key;
      return off;
    }
    function drawScene() {
      var pal = readPalette(), g = canvas.getContext("2d"), v = sim.T.view;
      g.imageSmoothingEnabled = true;
      g.drawImage(background(pal), 0, 0, CW, CW);
      var sx = CW / G;
      g.strokeStyle = pal.muted; g.globalAlpha = 0.7; g.lineWidth = 1;
      g.beginPath();
      sim.bg.segs.forEach(function (sgm) {
        g.moveTo((sgm[0][0] + 0.5) * sx, (sgm[0][1] + 0.5) * sx);
        g.lineTo((sgm[1][0] + 0.5) * sx, (sgm[1][1] + 0.5) * sx);
      });
      g.stroke(); g.globalAlpha = 1;
      function px(x) { return (x - v[0]) / (v[1] - v[0]) * CW; }
      function py(y) { return (v[3] - y) / (v[3] - v[2]) * CW; }
      // axes through the origin, for orientation
      g.strokeStyle = pal.rule; g.beginPath();
      g.moveTo(px(0), 0); g.lineTo(px(0), CW); g.moveTo(0, py(0)); g.lineTo(CW, py(0)); g.stroke();
      var n = sim.A.length / 2, i;
      if (st.showUla && !sim.ulaDead) {
        g.strokeStyle = pal.c3; g.lineWidth = 1.4;
        for (i = 0; i < n; i++) { g.beginPath(); g.arc(px(sim.B[2 * i]), py(sim.B[2 * i + 1]), 3.2, 0, 2 * Math.PI); g.stroke(); }
      }
      g.fillStyle = pal.c1;
      if (!sim.svgdDead) for (i = 0; i < n; i++) { g.beginPath(); g.arc(px(sim.A[2 * i]), py(sim.A[2 * i + 1]), 3.4, 0, 2 * Math.PI); g.fill(); }
      g.fillStyle = pal.ink; g.font = "13px system-ui, sans-serif";
      g.fillText("t = " + sim.t, 10, 20);
    }
    function text(x, y, s, anchor, fill) {
      chart.appendChild(WK.h("text", { x: x, y: y, "text-anchor": anchor || "middle", fill: fill || "muted", "font-size": 12 }, s));
    }
    function drawChart() {
      WK.clear(chart);
      var ml = 44, mr = 10, mt = 14, mb = 38;
      var tr = sim.trace, tmax = Math.max(50, sim.t), lo = -2.5, hi = 1;
      var x = WK.lin([0, tmax * st.n], [ml, W - mr]), y = WK.lin([lo, hi], [H - mb, mt]);
      for (var e = Math.ceil(lo); e <= hi; e++) {
        chart.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(e), y2: y(e), stroke: "rule" }));
        text(ml - 6, y(e) + 4, e === 0 ? "1" : e === 1 ? "10" : "10" + (e === -1 ? "⁻¹" : "⁻²"), "end");
      }
      WK.ticks(0, tmax * st.n, 4).forEach(function (t) { text(x(t), H - mb + 16, t >= 1000 ? (t / 1000) + "k" : String(t)); });
      text((ml + W - mr) / 2, H - 6, "score evaluations (n × iterations)");
      function line(key, colour, dash) {
        var pts = [];
        tr.forEach(function (p) { if (isFinite(p[key]) && p[key] > 0) pts.push(x(p.t * st.n) + "," + y(Math.max(lo, Math.min(hi, Math.log10(p[key]))))); });
        if (pts.length > 1) chart.appendChild(WK.h("polyline", { points: pts.join(" "), fill: "none", stroke: colour, "stroke-width": 2.2, "stroke-dasharray": dash || null }));
      }
      line("b", "c3", "5 3");
      line("a", "c1");
      text(W - mr - 2, mt + 12, sim.svgdDead ? "SVGD (diverged)" : "SVGD", "end", "c1");
      text(W - mr - 2, mt + 28, sim.ulaDead ? "Langevin (diverged)" : "Langevin (dashed)", "end", "c3");
    }
    function render() {
      drawScene();
      drawChart();
      var last = sim.trace[sim.trace.length - 1];
      sT.set(sim.t + " (" + (sim.t * st.n).toLocaleString() + ")");
      sA.set(sim.svgdDead ? "diverged" : WK.fmt.num(last.a, 3));
      sB.set(sim.ulaDead ? "diverged" : WK.fmt.num(last.b, 3));
      sL.set(sim.svgdDead ? "—" : WK.fmt.num(sim.ell, 2));
      if (st.target === "mixture") sM.set(sim.svgdDead ? "—" : WK.fmt.num(M.rightFraction(sim.A), 2) + " (" + WK.fmt.num(M.mixtureRightMass(st.sep), 2) + ")");
    }

    // Repaint after a theme change, and again once the new stylesheet has applied;
    // the delayed passes also catch a page that loads straight into dark mode.
    function repaintSoon() {
      [0, 150, 600].forEach(function (ms) { setTimeout(function () { if (sim && !playing) render(); }, ms); });
    }
    WK.onTheme(repaintSoon);
    window.addEventListener("load", repaintSoon);
    syncSep(); syncEll(); reset(); repaintSoon();
  });
})();
