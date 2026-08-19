/* Widget for "The Anatomy of a Volcano Plot".
 *
 * One widget: a screen of 3,000 hypothesis tests that the reader re-runs and
 * re-filters in the browser. Dependency-free -- no CDN, no module loader, no
 * framework -- so it works from a plain file:// page as well as over HTTP.
 * Observable JS does not work with the Quarto installed here (1.6.40), which is
 * why this and the other widgets on the blog are hand-rolled.
 *
 * The widget ships no data. It simulates its own: three populations of
 * features, a real two-sample t-test per feature, and Benjamini-Hochberg
 * q-values, all recomputed as the reader moves a control.
 *
 * The simulation core (VolcanoSim) touches no DOM, so `node` can exercise it
 * directly -- which is how its t-tail and its BH adjustment were checked
 * against scipy and statsmodels.
 */
"use strict";

/* ---------------------------------------------------------------------------
 * Simulation core. No DOM, no globals beyond the export at the end.
 * ------------------------------------------------------------------------ */
const VolcanoSim = (function () {
  /* Seeded PRNG, so a draw is reproducible and re-rolling is deliberate. */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randn(rng) {
    // Box-Muller. The second variate is discarded: the loops here are short
    // and keeping a cached spare made the seeding harder to reason about.
    let u = 0;
    while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  const uniform = (rng, lo, hi) => lo + (hi - lo) * rng();

  /* --- the t-distribution tail ------------------------------------------- */

  const LG_C = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];

  function logGamma(x) {
    let y = x;
    let tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += LG_C[j] / ++y;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }

  /* Continued fraction for the incomplete beta, evaluated by Lentz's method. */
  function betacf(a, b, x) {
    const FPMIN = 1e-300;
    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      h *= d * c;
      aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d;
      if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c;
      if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-16) break;
    }
    return h;
  }

  /* Regularised incomplete beta I_x(a, b). */
  function betainc(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const front = Math.exp(
      logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)
    );
    return x < (a + 1) / (a + b + 2)
      ? (front * betacf(a, b, x)) / a
      : 1 - (front * betacf(b, a, 1 - x)) / b;
  }

  /* Two-sided p-value for a Student t statistic on df degrees of freedom. */
  function tTwoSided(t, df) {
    const x = df / (df + t * t);
    return betainc(df / 2, 0.5, x);
  }

  /* Benjamini-Hochberg: q_i = min over k >= i of m * p_(k) / k, sorted. */
  function bhAdjust(p) {
    const m = p.length;
    const order = Array.from({ length: m }, (_, i) => i).sort((i, j) => p[i] - p[j]);
    const q = new Float64Array(m);
    let running = 1;
    for (let k = m; k >= 1; k--) {
      const i = order[k - 1];
      running = Math.min(running, (m * p[i]) / k);
      q[i] = running;
    }
    return q;
  }

  /* --- the screen --------------------------------------------------------- */

  const N_FEATURES = 3000;
  const N_TRUE = 150;
  const N_MICRO = 300;

  /* Draw the population each feature belongs to: its true effect and its
   * per-replicate noise SD. Four populations, because the post's argument needs
   * all four: the inert background, low-count features whose noise fakes a large
   * fold change, high-abundance features whose real shift is far too small to
   * matter, and the genuine responders.
   *
   * No measurement happens here -- that is `test`'s job, because what a feature
   * looks like depends on how many replicates you spent on it. The SD ranges are
   * quoted so that the standard error of the difference, sd*sqrt(2/n), lands in
   * the same place at the default n = 4 that the post's static figure uses. */
  function draw(seed, noiseFrac) {
    const rng = mulberry32(seed);
    const pool = N_FEATURES - N_TRUE - N_MICRO;
    const nNoisy = Math.round(noiseFrac * pool);
    const nNull = pool - nNoisy;

    const trueLfc = new Float64Array(N_FEATURES);
    const sd = new Float64Array(N_FEATURES);
    // 0 background null, 1 low-count noise, 2 genuine responder, 3 micro-shift
    const kind = new Uint8Array(N_FEATURES);

    let i = 0;
    for (let k = 0; k < nNull; k++, i++) {
      kind[i] = 0;
      trueLfc[i] = 0;
      sd[i] = uniform(rng, 0.21, 0.64);
    }
    for (let k = 0; k < nNoisy; k++, i++) {
      kind[i] = 1;
      trueLfc[i] = 0;
      sd[i] = uniform(rng, 1.13, 2.55);
    }
    for (let k = 0; k < N_MICRO; k++, i++) {
      // Deeply sequenced, so the standard error collapses and a 7% shift
      // clears any significance bar you care to set.
      kind[i] = 3;
      trueLfc[i] = (rng() < 0.5 ? -1 : 1) * uniform(rng, 0.08, 0.4);
      sd[i] = uniform(rng, 0.03, 0.10);
    }
    for (let k = 0; k < N_TRUE; k++, i++) {
      kind[i] = 2;
      trueLfc[i] = (rng() < 0.5 ? -1 : 1) * uniform(rng, 1.5, 4.0);
      sd[i] = uniform(rng, 0.28, 0.71);
    }

    return { trueLfc: trueLfc, sd: sd, kind: kind, seed: seed,
             n: N_FEATURES, nTrue: N_TRUE };
  }

  /* Measure every feature in nRep replicates per group and run a real
   * two-sample t-test on the draws.
   *
   * The replicates are drawn here rather than in `draw` because the whole point
   * of the replicate control is that spending more of them buys a tighter
   * estimate. Testing one fixed observation against a shrinking denominator
   * would instead inflate every t by sqrt(n) and flood the plot with noise --
   * the opposite of what more data does.
   *
   * Each feature gets its own stream, seeded from the experiment's seed, and the
   * replicates come off it in order. So raising the slider from 4 to 6 keeps the
   * first four measurements and adds two: the same experiment continued, not a
   * fresh one. Estimating the variance from those draws is also what puts the
   * small-n variance lottery on screen -- a null feature can still clear the bar
   * on a sample SD that came out small by luck. */
  function test(sample, nRep) {
    const df = 2 * nRep - 2;
    const m = sample.n;
    const p = new Float64Array(m);
    const lfc = new Float64Array(m);

    for (let j = 0; j < m; j++) {
      const rngj = mulberry32((sample.seed * 0x9e3779b1) ^ ((j + 1) * 0x85ebca6b));
      const sdj = sample.sd[j];
      let mC = 0, mT = 0, sC = 0, sT = 0;
      for (let k = 0; k < nRep; k++) {
        // Welford, so a long replicate stream stays numerically honest.
        const c = sdj * randn(rngj);
        const t = sample.trueLfc[j] + sdj * randn(rngj);
        let d = c - mC; mC += d / (k + 1); sC += d * (c - mC);
        d = t - mT; mT += d / (k + 1); sT += d * (t - mT);
      }
      lfc[j] = mT - mC;
      // Pooled two-sample t, equal group sizes.
      const sPooled = Math.sqrt((sC + sT) / df);
      const seDiff = sPooled * Math.sqrt(2 / nRep);
      const t = seDiff > 0 ? lfc[j] / seDiff : 0;
      p[j] = Math.max(tTwoSided(t, df), 1e-16);
    }

    const q = bhAdjust(p);
    const negLogP = new Float64Array(m);
    const negLogQ = new Float64Array(m);
    for (let j = 0; j < m; j++) {
      negLogP[j] = -Math.log10(p[j]);
      negLogQ[j] = -Math.log10(Math.max(q[j], 1e-16));
    }
    return { lfc: lfc, p: p, q: q, negLogP: negLogP, negLogQ: negLogQ, df: df };
  }

  /* Cumulative genuine-hit curves for the two one-dimensional rankings.
   * cum[k] is how many of the top k features by that ranking are real, so any
   * budget can be answered in constant time while a slider is being dragged.
   * Benjamini-Hochberg is monotone in p, so ranking by q gives the same order
   * as ranking by p and one significance curve covers both. */
  function rankCurves(sample, tested) {
    const m = sample.n;
    const idx = Array.from({ length: m }, (_, i) => i);
    const byEffect = idx.slice().sort((i, j) => Math.abs(tested.lfc[j]) - Math.abs(tested.lfc[i]));
    const bySig = idx.slice().sort((i, j) => tested.p[i] - tested.p[j]);
    const cumEffect = new Int32Array(m + 1);
    const cumSig = new Int32Array(m + 1);
    for (let k = 0; k < m; k++) {
      cumEffect[k + 1] = cumEffect[k] + (sample.kind[byEffect[k]] === 2 ? 1 : 0);
      cumSig[k + 1] = cumSig[k] + (sample.kind[bySig[k]] === 2 ? 1 : 0);
    }
    return { effect: cumEffect, significance: cumSig };
  }

  /* The dual filter, and what it caught. */
  function score(sample, tested, lfcCut, sigCut, useQ) {
    const y = useQ ? tested.negLogQ : tested.negLogP;
    const selected = new Uint8Array(sample.n);
    let nSel = 0;
    let nHit = 0;
    for (let j = 0; j < sample.n; j++) {
      if (Math.abs(tested.lfc[j]) >= lfcCut && y[j] >= sigCut) {
        selected[j] = 1;
        nSel++;
        if (sample.kind[j] === 2) nHit++;
      }
    }
    return {
      selected: selected,
      nSelected: nSel,
      nGenuine: nHit,
      precision: nSel > 0 ? nHit / nSel : 0,
      recall: nHit / sample.nTrue,
    };
  }

  return {
    mulberry32: mulberry32,
    randn: randn,
    tTwoSided: tTwoSided,
    bhAdjust: bhAdjust,
    draw: draw,
    test: test,
    score: score,
    rankCurves: rankCurves,
    N_FEATURES: N_FEATURES,
    N_TRUE: N_TRUE,
    N_MICRO: N_MICRO,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = VolcanoSim;

/* ---------------------------------------------------------------------------
 * The widget. A canvas scatter of all 3,000 tests, plain DOM controls around
 * it. Points are drawn as small squares through fillRect rather than arcs:
 * 3,000 of them are redrawn on every slider step and squares are several times
 * cheaper.
 * ------------------------------------------------------------------------ */
(function () {
  const C = {
    ink: "#171C1B",
    muted: "#67706E",
    rule: "#D2D6D1",
    paper: "#FFFFFF",
    primary: "#1D5C6E",
    amber: "#C98A12",
    neutral: "#A0AAB2",
    plum: "#7A3B6B",
    up: "#D9381E",
    down: "#2070B4",
  };

  function h(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) {
      if (k === "text") e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    for (const c of kids || []) e.appendChild(c);
    return e;
  }

  function slider(label, min, max, step, value, fmt, onchange) {
    const out = h("span", { class: "wv-value" });
    const input = h("input", {
      type: "range", min: min, max: max, step: step, value: value, class: "wv-range",
    });
    const sync = () => {
      const v = parseFloat(input.value);
      out.textContent = fmt(v);
      onchange(v);
    };
    input.addEventListener("input", sync);
    const node = h("label", { class: "wv-control" }, [
      h("span", { class: "wv-label", text: label }),
      input,
      out,
    ]);
    return { node: node, input: input, sync: sync };
  }

  function toggle(labels, initial, onchange) {
    let current = initial;
    const buttons = labels.map((t, i) =>
      h("button", { type: "button", class: "wv-toggle", text: t })
    );
    const paint = () =>
      buttons.forEach((b, i) => b.setAttribute("class", "wv-toggle" + (i === current ? " wv-on" : "")));
    buttons.forEach((b, i) =>
      b.addEventListener("click", () => {
        current = i;
        paint();
        onchange(i);
      })
    );
    paint();
    return { node: h("span", { class: "wv-toggle-group" }, buttons), get: () => current };
  }

  function readout(label) {
    const value = h("span", { class: "wv-stat-value", text: "—" });
    const node = h("span", { class: "wv-stat" }, [
      h("span", { class: "wv-stat-label", text: label }),
      value,
    ]);
    return { node: node, set: (t) => (value.textContent = t) };
  }

  function initScreenWidget(mountId) {
    const root = document.getElementById(mountId);
    if (!root) return;
    const S = VolcanoSim;

    const W = 720;
    const H = 400;
    const PAD = { l: 58, r: 16, t: 16, b: 44 };

    const state = {
      seed: 42,
      noiseFrac: 0.3,
      nRep: 4,
      lfcCut: 1.0,
      sigCut: 2.0,
      useQ: 0,
      rawAxis: 0,
    };

    let sample = null;
    let tested = null;
    let curves = null;

    const canvas = h("canvas", { class: "wv-canvas" });
    const ctx = canvas.getContext("2d");

    const rSelected = readout("selected");
    const rGenuine = readout("genuine hits");
    const rPrecision = readout("precision");
    const rRecall = readout("recall");
    const rTopFc = readout("same budget, |log₂FC| alone");
    const rTopP = readout("same budget, significance alone");
    const note = h("p", { class: "wv-note" });

    const sLfc = slider("effect-size gate |log₂FC| ≥", 0, 3, 0.1, state.lfcCut,
      (v) => v.toFixed(1),
      (v) => { state.lfcCut = v; refilter(); });

    const sSig = slider("significance gate −log₁₀ ≥", 0, 8, 0.25, state.sigCut,
      (v) => v.toFixed(2),
      (v) => { state.sigCut = v; refilter(); });

    const sRep = slider("replicates per group", 2, 12, 1, state.nRep,
      (v) => String(v),
      (v) => { state.nRep = v; retest(); });

    const sNoise = slider("low-count features", 0, 0.4, 0.02, state.noiseFrac,
      (v) => (v * 100).toFixed(0) + "%",
      (v) => { state.noiseFrac = v; redraw(); });

    const tAxis = toggle(["log₂ fold change", "raw ratio"], 0, (i) => {
      state.rawAxis = i;
      render();
    });

    const tSig = toggle(["raw p", "BH q"], 0, (i) => {
      state.useQ = i;
      refilter();
    });

    const reroll = h("button", { type: "button", class: "wv-button", text: "re-roll the experiment" });
    reroll.addEventListener("click", () => {
      state.seed = (state.seed * 1103515245 + 12345) >>> 0;
      redraw();
    });

    root.appendChild(
      h("div", { class: "wv-controls" }, [sLfc.node, sSig.node, sRep.node, sNoise.node])
    );
    root.appendChild(
      h("div", { class: "wv-toggles" }, [
        h("span", { class: "wv-label", text: "horizontal axis" }), tAxis.node,
        h("span", { class: "wv-label", text: "vertical axis" }), tSig.node,
        reroll,
      ])
    );
    root.appendChild(canvas);
    root.appendChild(
      h("div", { class: "wv-stats" }, [
        rSelected.node, rGenuine.node, rPrecision.node, rRecall.node,
      ])
    );
    root.appendChild(
      h("div", { class: "wv-stats wv-stats-compare" }, [rTopFc.node, rTopP.node])
    );
    root.appendChild(note);

    /* --- the three levels of recomputation ------------------------------- */

    function redraw() {
      sample = S.draw(state.seed, state.noiseFrac);
      retest();
    }

    function retest() {
      tested = S.test(sample, state.nRep);
      curves = S.rankCurves(sample, tested);
      refilter();
    }

    function refilter() {
      render();
    }

    /* --- drawing ---------------------------------------------------------- */

    function xDomain() {
      return state.rawAxis ? [0, 16] : [-6, 6];
    }

    function xValue(lfc) {
      return state.rawAxis ? Math.pow(2, lfc) : lfc;
    }

    function render() {
      const y = state.useQ ? tested.negLogQ : tested.negLogP;
      const res = S.score(sample, tested, state.lfcCut, state.sigCut, state.useQ);

      let yMax = 6;
      for (let i = 0; i < y.length; i++) if (y[i] > yMax) yMax = y[i];
      yMax = Math.min(Math.ceil(yMax) + 1, 18);

      const [x0, x1] = xDomain();
      const px = (v) => PAD.l + ((Math.min(Math.max(v, x0), x1) - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
      const py = (v) => H - PAD.b - (Math.min(v, yMax) / yMax) * (H - PAD.t - PAD.b);

      const dpr = window.devicePixelRatio || 1;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = "100%";
      canvas.style.maxWidth = W + "px";
      canvas.style.height = "auto";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.paper;
      ctx.fillRect(0, 0, W, H);

      // Shade the selected regions, so the filter is a place on the plot and
      // not just a pair of numbers.
      const yCut = py(state.sigCut);
      ctx.fillStyle = "rgba(29, 92, 110, 0.07)";
      const rightX = px(xValue(state.lfcCut));
      const leftX = px(xValue(-state.lfcCut));
      ctx.fillRect(rightX, PAD.t, W - PAD.r - rightX, yCut - PAD.t);
      ctx.fillRect(PAD.l, PAD.t, leftX - PAD.l, yCut - PAD.t);

      // Axes.
      ctx.strokeStyle = C.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD.l, PAD.t);
      ctx.lineTo(PAD.l, H - PAD.b);
      ctx.lineTo(W - PAD.r, H - PAD.b);
      ctx.stroke();

      ctx.fillStyle = C.muted;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      const xTicks = state.rawAxis ? [0, 1, 2, 4, 8, 12, 16] : [-6, -4, -2, 0, 2, 4, 6];
      for (const t of xTicks) {
        const X = px(t);
        ctx.beginPath();
        ctx.moveTo(X, H - PAD.b);
        ctx.lineTo(X, H - PAD.b + 4);
        ctx.stroke();
        ctx.fillText(String(t), X, H - PAD.b + 16);
      }
      ctx.textAlign = "right";
      const yStep = yMax <= 8 ? 2 : 4;
      for (let t = 0; t <= yMax; t += yStep) {
        const Y = py(t);
        ctx.beginPath();
        ctx.moveTo(PAD.l - 4, Y);
        ctx.lineTo(PAD.l, Y);
        ctx.stroke();
        ctx.fillText(String(t), PAD.l - 7, Y + 4);
      }

      ctx.fillStyle = C.ink;
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        state.rawAxis ? "fold change (raw ratio, treated / control)" : "log₂ fold change",
        PAD.l + (W - PAD.l - PAD.r) / 2,
        H - 8
      );
      ctx.save();
      ctx.translate(14, PAD.t + (H - PAD.t - PAD.b) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(state.useQ ? "−log₁₀ q (BH-adjusted)" : "−log₁₀ p", 0, 0);
      ctx.restore();

      // Points, painted background-first so genuine hits are never buried.
      const order = [0, 1, 3, 2];
      const fill = [C.neutral, C.amber, C.primary, C.plum];
      const alpha = [0.45, 0.6, 0.9, 0.75];
      for (const k of order) {
        ctx.fillStyle = fill[k];
        ctx.globalAlpha = alpha[k];
        const size = k === 2 ? 4 : 3;
        for (let i = 0; i < sample.n; i++) {
          if (sample.kind[i] !== k) continue;
          ctx.fillRect(px(xValue(tested.lfc[i])) - size / 2, py(y[i]) - size / 2, size, size);
        }
      }
      ctx.globalAlpha = 1;

      // Ring every selected point, so precision is countable by eye.
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 0.9;
      for (let i = 0; i < sample.n; i++) {
        if (!res.selected[i]) continue;
        ctx.beginPath();
        ctx.arc(px(xValue(tested.lfc[i])), py(y[i]), 4.2, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // The gates themselves.
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 4]);
      for (const X of [leftX, rightX]) {
        ctx.beginPath();
        ctx.moveTo(X, PAD.t);
        ctx.lineTo(X, H - PAD.b);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.moveTo(PAD.l, yCut);
      ctx.lineTo(W - PAD.r, yCut);
      ctx.stroke();
      ctx.setLineDash([]);

      // Legend.
      const legend = [
        ["genuine responders (150)", C.primary],
        ["low-count noise", C.amber],
        ["tiny but certain shifts", C.plum],
        ["unchanged background", C.neutral],
      ];
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(255, 255, 255, 0.82)";
      ctx.fillRect(PAD.l + 4, PAD.t + 2, 148, legend.length * 15 + 6);
      legend.forEach((entry, i) => {
        const Y = PAD.t + 12 + i * 15;
        ctx.fillStyle = entry[1];
        ctx.fillRect(PAD.l + 10, Y - 5, 8, 8);
        ctx.fillStyle = C.muted;
        ctx.fillText(entry[0], PAD.l + 24, Y + 2);
      });

      const k = res.nSelected;
      const fcHits = curves.effect[k];
      const sigHits = curves.significance[k];

      rSelected.set(String(k));
      rGenuine.set(res.nGenuine + " / 150");
      rPrecision.set(k ? (res.precision * 100).toFixed(0) + "%" : "—");
      rRecall.set((res.recall * 100).toFixed(0) + "%");
      rTopFc.set(fcHits + " genuine");
      rTopP.set(sigHits + " genuine");

      note.textContent = commentary(res, fcHits, sigHits);
    }

    function commentary(res, fcHits, sigHits) {
      const k = res.nSelected;
      if (state.rawAxis) {
        return (
          "On the raw ratio axis every repressed feature — half the genuine signal — is " +
          "crushed into the sliver between 0 and 1, while induced ones sprawl off the end " +
          "of the axis: anything past 16× is stacked against the right edge. Switch back " +
          "to log₂ and the two halves become mirror images."
        );
      }
      if (k === 0) {
        return "Both gates are now strict enough that nothing passes at all.";
      }
      if (state.sigCut === 0 && state.lfcCut === 0) {
        return "Both gates are open, so all 3,000 features are selected and precision is just the share of the screen that was ever real.";
      }
      if (state.sigCut === 0) {
        return (
          "With the significance gate at zero this is fold-change ranking: " + k +
          " features pass on effect size alone, and " + (k - res.nGenuine) +
          " of them are noise that happened to move."
        );
      }
      if (state.lfcCut === 0) {
        return (
          "With the effect-size gate at zero this is significance ranking: " + k +
          " features clear the bar, including deeply sequenced ones whose shift is real, " +
          "certain, and far too small to act on."
        );
      }
      return (
        "The dual filter selected " + k + " features and caught " + res.nGenuine +
        " of the 150 real responders. Hand a one-dimensional ranking the same budget of " +
        k + " and it finds " + fcHits + " (effect size alone) or " + sigHits +
        " (significance alone)."
      );
    }

    redraw();
    sLfc.sync();
    sSig.sync();
    sRep.sync();
    sNoise.sync();
  }

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", () => initScreenWidget("widget-screen"));
  }
})();
