/* Widgets for "Why the Bayesian Bootstrap?"
 *
 * Dependency-free: no CDN, no module loading, no framework. Everything is
 * built from the DOM and inline SVG, so the widgets work from a plain
 * file:// page as well as from a web server.
 *
 * Data arrives inlined in the page as <script id="bb-data" type="application/json">,
 * written from the committed widget-data/*.json files at render time.
 */
"use strict";

const BB = (function () {
  const NS = "http://www.w3.org/2000/svg";
  const C = {
    efron: "#1D5C6E", bayes: "#C98A12", bayesText: "#8A5E06",
    data: "#7A3B6B", ink: "#171C1B", muted: "#67706E", rule: "#D2D6D1",
  };

  // ---- DOM helpers -------------------------------------------------------
  function el(tag, attrs, kids) {
    const e = document.createElementNS(NS, tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    for (const c of (kids || [])) e.appendChild(c);
    return e;
  }
  function h(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const k in (attrs || {})) {
      if (k === "html") e.innerHTML = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    for (const c of (kids || [])) e.appendChild(c);
    return e;
  }
  function svg(w, hgt) {
    return el("svg", {
      viewBox: `0 0 ${w} ${hgt}`, width: w, height: hgt,
      style: "max-width:100%;height:auto;display:block", "font-family": "inherit",
    });
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // A labelled control that reports changes through `onchange`.
  function slider(label, min, max, step, value, fmt, onchange) {
    const out = h("span", { style: "font-variant-numeric:tabular-nums;font-weight:600" });
    const input = h("input", { type: "range", min, max, step, value,
      style: "vertical-align:middle;width:190px" });
    const sync = () => { out.textContent = fmt(parseFloat(input.value)); onchange(parseFloat(input.value)); };
    input.addEventListener("input", sync);
    const wrap = h("label", { style: "display:inline-block;margin:.15rem 1rem .15rem 0;font-size:.88rem" },
      [h("span", { text: label + " ", style: "color:#67706E" }), input, h("span", { text: " " }), out]);
    return { node: wrap, input, sync, get: () => parseFloat(input.value) };
  }
  function radio(name, label, options, value, onchange) {
    const wrap = h("span", { style: "display:inline-block;margin:.15rem 1rem .15rem 0;font-size:.88rem" },
      [h("span", { text: label + " ", style: "color:#67706E" })]);
    const inputs = [];
    for (const o of options) {
      const id = name + "-" + o.replace(/\W+/g, "");
      const inp = h("input", { type: "radio", name, id, value: o, style: "margin-left:.5rem" });
      if (o === value) inp.checked = true;
      inp.addEventListener("change", () => onchange(o));
      inputs.push(inp);
      wrap.appendChild(inp);
      wrap.appendChild(h("label", { for: id, text: " " + o, style: "margin-right:.3rem" }));
    }
    return { node: wrap, get: () => inputs.find((i) => i.checked).value };
  }
  function select(label, options, value, onchange) {
    const sel = h("select", { style: "font-size:.88rem;padding:.1rem" });
    for (const o of options) {
      const op = h("option", { value: String(o), text: String(o) });
      if (String(o) === String(value)) op.selected = true;
      sel.appendChild(op);
    }
    sel.addEventListener("change", () => onchange(sel.value));
    const wrap = h("label", { style: "display:inline-block;margin:.15rem 1rem .15rem 0;font-size:.88rem" },
      [h("span", { text: label + " ", style: "color:#67706E" }), sel]);
    return { node: wrap, get: () => sel.value };
  }

  // ---- special functions -------------------------------------------------
  const LG = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(z) {
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1;
    let a = 0.99999999999980993;
    const t = z + 7.5;
    for (let i = 0; i < 8; i++) a += LG[i] / (z + i + 1);
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
  }
  function lbeta(a, b) { return lgamma(a) + lgamma(b) - lgamma(a + b); }
  function betaPdf(x, a, b) {
    if (x <= 0 || x >= 1) return 0;
    return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b));
  }
  // Regularised incomplete beta, Lentz continued fraction.
  function betacf(x, a, b) {
    const EPS = 3e-14, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let hh = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; hh *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c; hh *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return hh;
  }
  function betaCdf(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta(a, b));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(x, a, b) / a
                                     : 1 - bt * betacf(1 - x, b, a) / b;
  }
  function betaInv(p, a, b) {           // bisection: robust, plenty fast here
    let lo = 0, hi = 1;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (betaCdf(mid, a, b) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }
  function lchoose(n, k) { return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); }
  function binomPmf(k, n, p) {
    if (p <= 0) return k === 0 ? 1 : 0;
    if (p >= 1) return k === n ? 1 : 0;
    return Math.exp(lchoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p));
  }
  // Central interval of Binomial(n,p)/n, by summing the exact pmf.
  function binomInterval(n, p, lo, hi) {
    let c = 0, a = 0, b = n;
    for (let k = 0; k <= n; k++) { c += binomPmf(k, n, p); if (c >= lo) { a = k; break; } }
    c = 0;
    for (let k = 0; k <= n; k++) { c += binomPmf(k, n, p); if (c >= hi) { b = k; break; } }
    return [a / n, b / n];
  }

  // ---- rng ---------------------------------------------------------------
  function lcg(seed) {                       // deterministic, so widgets are stable
    let s = seed >>> 0;
    return function () { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  }

  // ---- tiny chart primitives --------------------------------------------
  function axes(s, box, xd, yd, xlab, ylab, opts) {
    opts = opts || {};
    const { l, t, w, hh } = box;
    const g = el("g");
    const xs = (v) => l + (v - xd[0]) / (xd[1] - xd[0]) * w;
    const ys = (v) => t + hh - (v - yd[0]) / (yd[1] - yd[0]) * hh;
    const nx = opts.nx || 5, ny = opts.ny || 4;
    for (let i = 0; i <= ny; i++) {
      const v = yd[0] + (yd[1] - yd[0]) * i / ny;
      g.appendChild(el("line", { x1: l, x2: l + w, y1: ys(v), y2: ys(v),
        stroke: C.rule, "stroke-width": 0.7, opacity: 0.7 }));
      g.appendChild(text(l - 6, ys(v) + 3, fmtTick(v), { anchor: "end", size: 9, fill: C.muted }));
    }
    for (let i = 0; i <= nx; i++) {
      const v = xd[0] + (xd[1] - xd[0]) * i / nx;
      g.appendChild(text(xs(v), t + hh + 14, fmtTick(v), { anchor: "middle", size: 9, fill: C.muted }));
    }
    g.appendChild(el("line", { x1: l, x2: l + w, y1: t + hh, y2: t + hh, stroke: C.muted, "stroke-width": 0.9 }));
    if (xlab) g.appendChild(text(l + w / 2, t + hh + 30, xlab, { anchor: "middle", size: 10, fill: C.muted }));
    if (ylab) {
      const tx = text(0, 0, ylab, { anchor: "middle", size: 10, fill: C.muted });
      tx.setAttribute("transform", `translate(${l - 34},${t + hh / 2}) rotate(-90)`);
      g.appendChild(tx);
    }
    s.appendChild(g);
    return { xs, ys };
  }
  function fmtTick(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1000) return v.toExponential(0);
    if (a >= 10) return v.toFixed(0);
    if (a >= 1) return v.toFixed(1);
    if (a >= 0.01) return v.toFixed(2);
    return v.toExponential(1);
  }
  function text(x, y, str, o) {
    o = o || {};
    return el("text", { x, y, "text-anchor": o.anchor || "start",
      "font-size": o.size || 10, fill: o.fill || C.ink,
      "font-weight": o.weight || "normal" }, [document.createTextNode(str)]);
  }
  function path(d, o) {
    o = o || {};
    return el("path", { d, fill: o.fill || "none", stroke: o.stroke || "none",
      "stroke-width": o.w || 1.5, "fill-opacity": o.fillOpacity == null ? 1 : o.fillOpacity,
      "stroke-dasharray": o.dash || "none" });
  }
  function polyline(pts, sc, o) {
    let d = "";
    pts.forEach((p, i) => { d += (i ? "L" : "M") + sc.xs(p[0]) + "," + sc.ys(p[1]); });
    return path(d, o);
  }
  function areaUnder(pts, sc, y0, o) {
    let d = "M" + sc.xs(pts[0][0]) + "," + sc.ys(y0);
    pts.forEach((p) => { d += "L" + sc.xs(p[0]) + "," + sc.ys(p[1]); });
    d += "L" + sc.xs(pts[pts.length - 1][0]) + "," + sc.ys(y0) + "Z";
    return path(d, o);
  }
  // Histogram of `vals` as an SVG group, normalised to a density-like height.
  function histogram(vals, nbins, lo, hi) {
    const counts = new Array(nbins).fill(0);
    for (const v of vals) {
      let b = Math.floor((v - lo) / (hi - lo) * nbins);
      if (b < 0) b = 0; if (b >= nbins) b = nbins - 1;
      counts[b]++;
    }
    const mx = Math.max(...counts) || 1;
    return { counts, mx, width: (hi - lo) / nbins, lo };
  }
  function readout(rows) {
    const d = h("div", { style: "font-size:.9rem;line-height:1.75;margin-top:.4rem" });
    for (const r of rows) d.appendChild(h("div", { html: r }));
    return d;
  }
  function sd(a) {
    const m = a.reduce((s, v) => s + v, 0) / a.length;
    return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length);
  }

  return { NS, C, el, h, svg, clear, slider, radio, select, lgamma, betaPdf,
    betaCdf, betaInv, binomPmf, binomInterval, lcg, axes, text, path, polyline,
    areaUnder, histogram, readout, sd };
})();

const BBDATA = JSON.parse(document.getElementById("bb-data").textContent);

/* ------------------------------------------------------------------ W1 --
 * Simplex explorer: a draggable point in a barycentric triangle.
 */
(function () {
  const root = document.getElementById("w1");
  if (!root) return;
  const x = BBDATA.toy.triple;
  let w = [1 / 3, 1 / 3, 1 / 3], snap = false;

  const controls = BB.h("div");
  const toggle = BB.h("input", { type: "checkbox", id: "w1-snap" });
  toggle.addEventListener("change", () => { snap = toggle.checked; w = maybeSnap(w); draw(); });
  controls.appendChild(BB.h("label", { style: "font-size:.88rem" },
    [toggle, BB.h("span", { text: " snap to Efron's lattice" })]));

  const panel = BB.h("div", { style: "display:flex;gap:1.4rem;flex-wrap:wrap;align-items:flex-start" });
  const left = BB.h("div"), right = BB.h("div");
  panel.appendChild(left); panel.appendChild(right);
  root.appendChild(controls); root.appendChild(panel);

  const S = 300, PAD = 38;
  const A = [PAD, S - PAD], Bc = [S - PAD, S - PAD], Cc = [S / 2, PAD];
  const toXY = (v) => [v[0] * A[0] + v[1] * Bc[0] + v[2] * Cc[0],
                       v[0] * A[1] + v[1] * Bc[1] + v[2] * Cc[1]];
  function toW(px, py) {
    const den = (Bc[1] - Cc[1]) * (A[0] - Cc[0]) + (Cc[0] - Bc[0]) * (A[1] - Cc[1]);
    let a = ((Bc[1] - Cc[1]) * (px - Cc[0]) + (Cc[0] - Bc[0]) * (py - Cc[1])) / den;
    let b = ((Cc[1] - A[1]) * (px - Cc[0]) + (A[0] - Cc[0]) * (py - Cc[1])) / den;
    let c = 1 - a - b;
    let v = [a, b, c].map((t) => Math.max(0, t));
    const s = v[0] + v[1] + v[2];
    return s > 0 ? v.map((t) => t / s) : [1 / 3, 1 / 3, 1 / 3];
  }
  function maybeSnap(v) {
    if (!snap) return v;
    let best = v, bd = Infinity;
    for (let a = 0; a <= 3; a++) for (let b = 0; a + b <= 3; b++) {
      const cand = [a / 3, b / 3, (3 - a - b) / 3];
      const d = cand.reduce((s, ci, i) => s + (ci - v[i]) ** 2, 0);
      if (d < bd) { bd = d; best = cand; }
    }
    return best;
  }
  const fact = [1, 1, 2, 6];

  function draw() {
    BB.clear(left); BB.clear(right);
    const s = BB.svg(S, S);
    s.appendChild(BB.el("polygon", {
      points: [A, Bc, Cc].map((p) => p.join(",")).join(" "),
      fill: BB.C.bayes, "fill-opacity": 0.16, stroke: BB.C.muted }));
    if (snap) {
      for (let a = 0; a <= 3; a++) for (let b = 0; a + b <= 3; b++) {
        const q = toXY([a / 3, b / 3, (3 - a - b) / 3]);
        s.appendChild(BB.el("circle", { cx: q[0], cy: q[1], r: 3.4,
          fill: BB.C.efron, "fill-opacity": 0.65 }));
      }
    }
    [["x = 1", A, 0, 17], ["x = 3", Bc, 0, 17], ["x = 10", Cc, 0, -11]].forEach(
      ([t, p, dx, dy]) => s.appendChild(BB.text(p[0] + dx, p[1] + dy, t,
        { anchor: "middle", size: 11, fill: BB.C.muted })));
    const p = toXY(w);
    s.appendChild(BB.el("circle", { cx: p[0], cy: p[1], r: 7.5, fill: BB.C.ink }));
    s.style.cursor = "grab"; s.style.touchAction = "none";

    function move(ev) {
      const r = s.getBoundingClientRect();
      const cx = (ev.touches ? ev.touches[0].clientX : ev.clientX);
      const cy = (ev.touches ? ev.touches[0].clientY : ev.clientY);
      w = maybeSnap(toW((cx - r.left) / r.width * S, (cy - r.top) / r.height * S));
      draw();
      if (ev.cancelable) ev.preventDefault();
    }
    const up = () => { window.removeEventListener("pointermove", move);
                       window.removeEventListener("pointerup", up); };
    s.addEventListener("pointerdown", (ev) => {
      move(ev);
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    left.appendChild(s);

    // the distribution the point *is*
    const W = 250, H = 210, box = { l: 46, t: 12, w: W - 62, hh: H - 52 };
    const s2 = BB.svg(W, H);
    const sc = BB.axes(s2, box, [0, 11], [0, 1], "x", "mass", { nx: 4, ny: 4 });
    x.forEach((xi, i) => {
      s2.appendChild(BB.el("line", { x1: sc.xs(xi), x2: sc.xs(xi), y1: sc.ys(0),
        y2: sc.ys(w[i]), stroke: BB.C.ink, "stroke-width": 3.2 }));
      s2.appendChild(BB.el("circle", { cx: sc.xs(xi), cy: sc.ys(w[i]), r: 4, fill: BB.C.ink }));
    });
    right.appendChild(s2);

    const mean = w.reduce((s3, wi, i) => s3 + wi * x[i], 0);
    let cum = 0, med = x[0];
    for (let i = 0; i < 3; i++) { cum += w[i]; if (cum >= 0.5) { med = x[i]; break; } }
    let maxi = 0; for (let i = 0; i < 3; i++) if (w[i] > 0) maxi = i;
    const mode = x[w.indexOf(Math.max.apply(null, w))];
    const rows = [
      "<b>w</b> = (" + w.map((v) => v.toFixed(3)).join(", ") + ")",
      "mean <b>" + mean.toFixed(3) + "</b>",
      "median <b>" + med + "</b>",
      "max <b>" + x[maxi] + "</b>",
      "mode <b>" + mode + "</b>",
    ];
    if (snap) {
      const c = w.map((v) => Math.round(v * 3));
      const pr = (6 / (fact[c[0]] * fact[c[1]] * fact[c[2]])) / 27;
      rows.push('<span style="color:' + BB.C.efron + '">pattern (' + c.join(",") +
        ") &middot; multinomial probability " + pr.toFixed(4) + "</span>");
    }
    right.appendChild(BB.readout(rows));
  }
  draw();
})();

/* ------------------------------------------------------------------ W2 --
 * Dirichlet explorer: sweep alpha, watch the cloud move.
 */
(function () {
  const root = document.getElementById("w2");
  if (!root) return;
  const n = 3;
  let alpha = 1;

  const controls = BB.h("div");
  const sl = BB.slider("α (log scale)", Math.log(0.05), Math.log(20), 0.01, 0,
    (v) => Math.exp(v).toFixed(2), (v) => { alpha = Math.exp(v); draw(); });
  controls.appendChild(sl.node);
  for (const [lab, val] of [["α = 1 (Bayesian bootstrap)", 1], ["α = 1 − 1/n (matches Efron)", 1 - 1 / n]]) {
    const b = BB.h("button", { text: lab, style: "font-size:.82rem;margin-right:.4rem;cursor:pointer" });
    b.addEventListener("click", () => { alpha = val; sl.input.value = Math.log(val); sl.sync(); });
    controls.appendChild(b);
  }
  const body = BB.h("div");
  root.appendChild(controls); root.appendChild(body);

  const S = 310, PAD = 26;
  const A = [PAD, S - PAD], Bc = [S - PAD, S - PAD], Cc = [S / 2, PAD];
  const toXY = (v) => [v[0] * A[0] + v[1] * Bc[0] + v[2] * Cc[0],
                       v[0] * A[1] + v[1] * Bc[1] + v[2] * Cc[1]];
  const fact = [1, 1, 2, 6];

  function draw() {
    BB.clear(body);
    const rnd = BB.lcg(20260726);
    // Gamma(alpha) by the boost identity, in logs: stable for tiny alpha.
    function gammaDraw(a) {
      const d = a + 1 - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let xg, v;
        do { const u1 = rnd(), u2 = rnd();
             xg = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
             v = 1 + c * xg; } while (v <= 0);
        v = v * v * v;
        const u = rnd();
        if (u < 1 - 0.0331 * xg * xg * xg * xg) return d * v;
        if (Math.log(u + 1e-12) < 0.5 * xg * xg + d * (1 - v + Math.log(v))) return d * v;
      }
    }
    const pts = [];
    for (let b = 0; b < 2200; b++) {
      const lg = [];
      for (let i = 0; i < n; i++) lg.push(Math.log(gammaDraw(alpha)) + Math.log(rnd() + 1e-300) / alpha);
      const m = Math.max.apply(null, lg);
      const g = lg.map((v) => Math.exp(v - m));
      const s = g[0] + g[1] + g[2];
      pts.push(toXY(g.map((v) => v / s)));
    }
    const s = BB.svg(S, S);
    s.appendChild(BB.el("polygon", { points: [A, Bc, Cc].map((p) => p.join(",")).join(" "),
      fill: "none", stroke: BB.C.muted }));
    for (const p of pts)
      s.appendChild(BB.el("circle", { cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: 1.2,
        fill: BB.C.bayes, "fill-opacity": 0.36 }));
    for (let a = 0; a <= n; a++) for (let b = 0; a + b <= n; b++) {
      const q = toXY([a / n, b / n, (n - a - b) / n]);
      const pr = (6 / (fact[a] * fact[b] * fact[n - a - b])) / 27;
      s.appendChild(BB.el("circle", { cx: q[0], cy: q[1], r: 3 + 22 * pr,
        fill: BB.C.efron, "fill-opacity": 0.9 }));
    }
    body.appendChild(s);

    const vD = (n - 1) / (n * n * (n * alpha + 1)), vE = (n - 1) / n ** 3;
    body.appendChild(BB.readout([
      "α = <b>" + alpha.toFixed(3) + "</b>",
      "Var(wᵢ) Dirichlet <b>" + vD.toFixed(5) + "</b> &nbsp;·&nbsp; Efron <b>" + vE.toFixed(5) +
        "</b> &nbsp;·&nbsp; ratio <b>" + (vD / vE).toFixed(4) + "</b>",
    ]));
  }
  draw();
})();

/* ------------------------------------------------------------------ W3 --
 * The ladder: three denominators against n.
 */
(function () {
  const root = document.getElementById("w3");
  if (!root) return;
  let n = 59;
  const marks = BBDATA.ladder.marked;
  const controls = BB.h("div");
  controls.appendChild(BB.slider("n", 2, 500, 1, 59, (v) => String(v), (v) => { n = v; draw(); }).node);
  const body = BB.h("div");
  root.appendChild(controls); root.appendChild(body);

  function draw() {
    BB.clear(body);
    const rows = [
      { name: "Bayes  s̃²/(n+1)", v: 1 / (n + 1), c: BB.C.bayes },
      { name: "Efron  s̃²/n", v: 1 / n, c: BB.C.efron },
      { name: "classical s̃²/(n−1)", v: 1 / (n - 1), c: BB.C.data },
    ];
    const W = 520, H = 160, box = { l: 150, t: 10, w: W - 170, hh: H - 46 };
    const s = BB.svg(W, H);
    const mx = rows[2].v * 1.08;
    const sc = BB.axes(s, box, [0, mx], [0, 3], "variance of the mean (s̃² = 1)", null, { nx: 4, ny: 0 });
    rows.forEach((r, i) => {
      const y = box.t + (i + 0.5) * box.hh / 3;
      s.appendChild(BB.el("rect", { x: box.l, y: y - 13, width: Math.max(1, sc.xs(r.v) - box.l),
        height: 26, fill: r.c, "fill-opacity": 0.85 }));
      s.appendChild(BB.text(box.l - 8, y + 4, r.name, { anchor: "end", size: 10 }));
    });
    body.appendChild(s);
    const ratio = Math.sqrt(n / (n + 1));
    let note = "";
    if (n === marks.short) note = " ← the short S&P window";
    else if (n === marks.head) note = " ← the head-to-head window";
    else if (n === marks.long) note = " ← the long S&P window";
    body.appendChild(BB.readout([
      "n = <b>" + n + "</b>" + note,
      "sd ratio √(n/(n+1)) = <b>" + ratio.toFixed(4) + "</b> &nbsp;·&nbsp; Bayes is <b>" +
        ((1 - ratio) * 100).toFixed(2) + "%</b> tighter than Efron",
    ]));
  }
  draw();
})();

/* ------------------------------------------------------------------ W4 --
 * Live returns bootstrap: both methods, run in the browser.
 */
(function () {
  const root = document.getElementById("w4");
  if (!root) return;
  let series = "sp500", fun = "mean", q = 0.25, thr = -5, B = 10000;

  const controls = BB.h("div");
  controls.appendChild(BB.radio("w4s", "series", ["sp500", "ibm"], series,
    (v) => { series = v; draw(); }).node);
  controls.appendChild(BB.select("functional",
    ["mean", "median", "max", "min", "quantile", "P(R ≤ t)"], fun,
    (v) => { fun = v; draw(); }).node);
  controls.appendChild(BB.slider("quantile level", 0.01, 0.99, 0.01, 0.25,
    (v) => v.toFixed(2), (v) => { q = v; if (fun === "quantile") draw(); }).node);
  controls.appendChild(BB.slider("threshold t", -13, 8, 0.5, -5,
    (v) => v.toFixed(1), (v) => { thr = v; if (fun === "P(R ≤ t)") draw(); }).node);
  controls.appendChild(BB.select("B", [1000, 10000, 50000], B,
    (v) => { B = +v; draw(); }).node);
  const body = BB.h("div");
  root.appendChild(controls); root.appendChild(body);

  function run(x, efron, reps) {
    const n = x.length, out = new Array(reps), w = new Float64Array(n);
    const rnd = BB.lcg(efron ? 11 : 23);
    for (let b = 0; b < reps; b++) {
      w.fill(0);
      if (efron) { for (let j = 0; j < n; j++) w[(rnd() * n) | 0] += 1 / n; }
      else {
        let s = 0;
        for (let i = 0; i < n; i++) { const e = -Math.log(rnd() + 1e-300); w[i] = e; s += e; }
        for (let i = 0; i < n; i++) w[i] /= s;
      }
      out[b] = evaluate(x, w, n);
    }
    return out;
  }
  function evaluate(x, w, n) {
    if (fun === "mean") { let m = 0; for (let i = 0; i < n; i++) m += w[i] * x[i]; return m; }
    if (fun === "P(R ≤ t)") { let c = 0; for (let i = 0; i < n && x[i] <= thr; i++) c += w[i]; return c; }
    if (fun === "max") { for (let i = n - 1; i >= 0; i--) if (w[i] > 0) return x[i]; }
    if (fun === "min") { for (let i = 0; i < n; i++) if (w[i] > 0) return x[i]; }
    const lvl = fun === "median" ? 0.5 : q;
    let c = 0;
    for (let i = 0; i < n; i++) { c += w[i]; if (c >= lvl) return x[i]; }
    return x[n - 1];
  }

  function draw() {
    BB.clear(body);
    const x = BBDATA.returns[series].returns.slice().sort((a, b) => a - b);
    const n = x.length;
    const E = run(x, true, B), Bs = run(x, false, B);
    const sdE = BB.sd(E), sdB = BB.sd(Bs);
    // Under Bayes every weight is strictly positive, so max and min are point
    // masses. Compare against a tolerance rather than exact zero: the spread is
    // genuinely nil but floating-point accumulation need not give a literal 0.
    const degenerate = (fun === "max" || fun === "min") && sdB < 1e-12;

    const all = degenerate ? E : E.concat(Bs);
    let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;

    const W = 580, H = 250, box = { l: 52, t: 12, w: W - 70, hh: H - 56 };
    const s = BB.svg(W, H);
    const hE = BB.histogram(E, 46, lo, hi);
    const hB = degenerate ? null : BB.histogram(Bs, 46, lo, hi);
    const ymax = Math.max(hE.mx, hB ? hB.mx : 0) * 1.1;
    const sc = BB.axes(s, box, [lo, hi], [0, ymax], fun, "count", { nx: 5, ny: 4 });
    const bars = (hist, col) => {
      hist.counts.forEach((c, i) => {
        if (!c) return;
        const x0 = sc.xs(hist.lo + i * hist.width), x1 = sc.xs(hist.lo + (i + 1) * hist.width);
        s.appendChild(BB.el("rect", { x: x0, y: sc.ys(c), width: Math.max(0.6, x1 - x0),
          height: sc.ys(0) - sc.ys(c), fill: col, "fill-opacity": 0.55 }));
      });
    };
    bars(hE, BB.C.efron);
    if (hB) bars(hB, BB.C.bayes);
    if (degenerate)
      s.appendChild(BB.el("line", { x1: sc.xs(Bs[0]), x2: sc.xs(Bs[0]), y1: sc.ys(0),
        y2: sc.ys(ymax * 0.95), stroke: BB.C.bayes, "stroke-width": 3.5 }));
    s.appendChild(BB.text(box.l + 6, box.t + 12, "■ Efron", { size: 10, fill: BB.C.efron, weight: "600" }));
    s.appendChild(BB.text(box.l + 62, box.t + 12, "■ Bayes", { size: 10, fill: BB.C.bayesText, weight: "600" }));
    body.appendChild(s);

    const rows = [];
    if (degenerate)
      rows.push('<span style="color:' + BB.C.bayesText + '"><b>Bayes posterior is a point mass at ' +
        Bs[0].toFixed(3) + "</b> — drawn as a spike, not a histogram.</span>");
    rows.push("sd Efron <b>" + sdE.toFixed(4) + "</b> &nbsp;·&nbsp; sd Bayes <b>" + sdB.toFixed(4) +
      "</b> &nbsp;·&nbsp; ratio <b>" + (sdE > 0 ? (sdB / sdE).toFixed(4) : "—") +
      "</b> &nbsp;·&nbsp; √(n/(n+1)) = <b>" + Math.sqrt(n / (n + 1)).toFixed(4) + "</b>");
    body.appendChild(BB.readout(rows));
  }
  draw();
})();

/* ------------------------------------------------------------------ W5 --
 * CDF and tail explorer.
 */
(function () {
  const root = document.getElementById("w5");
  if (!root) return;
  let series = "sp500", t = -5;

  const controls = BB.h("div");
  controls.appendChild(BB.radio("w5s", "series", ["sp500", "ibm"], series,
    (v) => { series = v; draw(); }).node);
  controls.appendChild(BB.slider("threshold t (%)", -13, 8, 0.1, -5,
    (v) => v.toFixed(1), (v) => { t = v; draw(); }).node);
  const stairBtn = BB.h("button", { text: "redraw 40 candidate distributions",
    style: "font-size:.82rem;cursor:pointer" });
  let stairSeed = 1;
  stairBtn.addEventListener("click", () => { stairSeed++; draw(); });
  controls.appendChild(stairBtn);
  const body = BB.h("div", { style: "display:flex;gap:1.2rem;flex-wrap:wrap;align-items:flex-start" });
  root.appendChild(controls); root.appendChild(body);

  function draw() {
    BB.clear(body);
    const x = BBDATA.returns[series].returns.slice().sort((a, b) => a - b);
    const n = x.length;
    const k = x.filter((v) => v <= t).length;

    // left: ecdf with 40 posterior draws and the threshold
    const W = 350, H = 250, box = { l: 46, t: 12, w: W - 62, hh: H - 56 };
    const s = BB.svg(W, H);
    const sc = BB.axes(s, box, [x[0] - 1, x[n - 1] + 1], [0, 1], "monthly return, %", "F(t)");
    const rnd = BB.lcg(97 * stairSeed);
    for (let b = 0; b < 40; b++) {
      let sum = 0; const w = new Float64Array(n);
      for (let i = 0; i < n; i++) { const e = -Math.log(rnd() + 1e-300); w[i] = e; sum += e; }
      let c = 0; let d = "M" + sc.xs(x[0]) + "," + sc.ys(0);
      for (let i = 0; i < n; i++) { c += w[i] / sum;
        d += "L" + sc.xs(x[i]) + "," + sc.ys(c) + "L" + sc.xs(i + 1 < n ? x[i + 1] : x[n - 1]) + "," + sc.ys(c); }
      s.appendChild(BB.path(d, { stroke: BB.C.bayes, w: 0.7, fillOpacity: 0 }));
      s.lastChild.setAttribute("opacity", 0.3);
    }
    let d2 = "M" + sc.xs(x[0]) + "," + sc.ys(0);
    for (let i = 0; i < n; i++) { const c = (i + 1) / n;
      d2 += "L" + sc.xs(x[i]) + "," + sc.ys(c) + "L" + sc.xs(i + 1 < n ? x[i + 1] : x[n - 1]) + "," + sc.ys(c); }
    s.appendChild(BB.path(d2, { stroke: BB.C.ink, w: 1.8 }));
    s.appendChild(BB.el("line", { x1: sc.xs(t), x2: sc.xs(t), y1: sc.ys(0), y2: sc.ys(1),
      stroke: BB.C.data, "stroke-width": 2 }));
    body.appendChild(s);

    // right: the exact posterior for F(t)
    const right = BB.h("div");
    if (k > 0 && k < n) {
      const W2 = 330, H2 = 250, box2 = { l: 50, t: 12, w: W2 - 66, hh: H2 - 56 };
      const s2 = BB.svg(W2, H2);
      const loQ = BB.betaInv(0.001, k, n - k), hiQ = BB.betaInv(0.999, k, n - k);
      const pts = [];
      for (let i = 0; i <= 200; i++) { const p = loQ + (hiQ - loQ) * i / 200;
        pts.push([p, BB.betaPdf(p, k, n - k)]); }
      const ymax = Math.max.apply(null, pts.map((p) => p[1])) * 1.12;
      const sc2 = BB.axes(s2, box2, [loQ, hiQ], [0, ymax], "F(t)", "posterior density");
      s2.appendChild(BB.areaUnder(pts, sc2, 0, { fill: BB.C.bayes, fillOpacity: 0.45 }));
      s2.appendChild(BB.polyline(pts, sc2, { stroke: BB.C.bayesText, w: 1.6 }));
      // Efron's Bin(n,k/n)/n pmf beside it
      for (let j = 0; j <= n; j++) {
        const p = j / n;
        if (p < loQ || p > hiQ) continue;
        const pm = BB.binomPmf(j, n, k / n);
        if (pm < 1e-4) continue;
        s2.appendChild(BB.el("line", { x1: sc2.xs(p), x2: sc2.xs(p), y1: sc2.ys(0),
          y2: sc2.ys(pm * ymax / 0.35), stroke: BB.C.efron, "stroke-width": 2.2, opacity: 0.8 }));
      }
      right.appendChild(s2);
    } else {
      right.appendChild(BB.h("div", { style: "width:330px;font-size:.9rem;color:#67706E",
        text: "k = " + k + ": the posterior is degenerate at " + (k === 0 ? 0 : 1) + "." }));
    }
    const atom = Math.pow((n - k) / n, n);
    const bi = k > 0 && k < n ? [BB.betaInv(0.025, k, n - k), BB.betaInv(0.975, k, n - k)] : null;
    const ei = BB.binomInterval(n, k / n, 0.025, 0.975);
    right.appendChild(BB.readout([
      "k = <b>" + k + "</b> of " + n + " &nbsp;·&nbsp; plug-in <b>" + (k / n).toFixed(4) + "</b>",
      bi ? '<span style="color:' + BB.C.bayesText + '">Bayes <b>Beta(' + k + ", " + (n - k) +
        ")</b> 95% [" + bi[0].toFixed(5) + ", " + bi[1].toFixed(4) + "]</span>" : "",
      '<span style="color:' + BB.C.efron + '">Efron 95% [' + ei[0].toFixed(5) + ", " + ei[1].toFixed(4) + "]</span>",
      '<span style="color:' + BB.C.efron + '">P(Efron reports exactly 0) = <b>' +
        (atom < 1e-4 ? atom.toExponential(2) : atom.toFixed(4)) + "</b></span>",
    ].filter(Boolean)));
    body.appendChild(right);
  }
  draw();
})();

/* ------------------------------------------------------------------ W6 --
 * Rare-event / rare-class atom.
 */
(function () {
  const root = document.getElementById("w6");
  if (!root) return;
  let n = 59, k = 1;

  const controls = BB.h("div");
  const sn = BB.slider("n", 5, 500, 1, 59, (v) => String(v), (v) => { n = v; if (k > n) k = n; draw(); });
  const sk = BB.slider("k", 0, 30, 1, 1, (v) => String(v), (v) => { k = v; draw(); });
  controls.appendChild(sn.node); controls.appendChild(sk.node);
  for (const [lab, nn, kk] of [["S&P tail (n=59, k=1)", 59, 1], ["MNIST rare class (n=50, k=1)", 50, 1]]) {
    const b = BB.h("button", { text: lab, style: "font-size:.82rem;margin-right:.4rem;cursor:pointer" });
    b.addEventListener("click", () => {
      n = nn; k = kk; sn.input.value = nn; sk.input.value = kk; sn.sync(); sk.sync();
    });
    controls.appendChild(b);
  }
  const body = BB.h("div", { style: "display:flex;gap:1.4rem;flex-wrap:wrap;align-items:center" });
  root.appendChild(controls); root.appendChild(body);

  function draw() {
    BB.clear(body);
    const kk = Math.min(k, n);
    const atom = Math.pow((n - kk) / n, n);
    const W = 390, H = 230, box = { l: 56, t: 12, w: W - 74, hh: H - 56 };
    const s = BB.svg(W, H);
    const kmax = Math.min(30, n);
    const sc = BB.axes(s, box, [0, kmax], [0, 1], "k", "P(Efron reports exactly 0)");
    const pts = [];
    for (let j = 0; j <= kmax; j++) pts.push([j, Math.pow((n - j) / n, n)]);
    s.appendChild(BB.polyline(pts, sc, { stroke: BB.C.efron, w: 2 }));
    s.appendChild(BB.el("line", { x1: box.l, x2: box.l + box.w, y1: sc.ys(Math.exp(-1)),
      y2: sc.ys(Math.exp(-1)), stroke: BB.C.muted, "stroke-dasharray": "3,3" }));
    s.appendChild(BB.text(box.l + box.w - 4, sc.ys(Math.exp(-1)) - 5, "1/e", { anchor: "end", size: 9, fill: BB.C.muted }));
    s.appendChild(BB.el("circle", { cx: sc.xs(Math.min(kk, kmax)), cy: sc.ys(atom), r: 5, fill: BB.C.data }));
    body.appendChild(s);

    const rows = ["n = <b>" + n + "</b>, k = <b>" + kk + "</b>",
      "plug-in estimate <b>" + (kk / n).toFixed(4) + "</b>",
      '<span style="color:' + BB.C.efron + '">P(Efron = 0) = <b>' + (atom * 100).toFixed(1) + "%</b></span>"];
    if (kk > 0 && kk < n) {
      const bi = [BB.betaInv(0.025, kk, n - kk), BB.betaInv(0.975, kk, n - kk)];
      const ei = BB.binomInterval(n, kk / n, 0.025, 0.975);
      rows.push('<span style="color:' + BB.C.bayesText + '">Bayes <b>Beta(' + kk + ", " + (n - kk) +
        ")</b> 95% [" + bi[0].toFixed(5) + ", " + bi[1].toFixed(4) + "]</span>");
      rows.push('<span style="color:' + BB.C.efron + '">Efron 95% [' + ei[0].toFixed(5) + ", " + ei[1].toFixed(4) + "]</span>");
      rows.push('<span style="color:#67706E;font-size:.88rem">the Bayesian lower endpoint is never zero</span>');
    } else {
      rows.push('<span style="color:#67706E">with k = 0 both methods are degenerate at 0</span>');
    }
    body.appendChild(BB.readout(rows));
  }
  draw();
})();

/* ------------------------------------------------------------------ W7 --
 * MNIST metric explorer.
 */
(function () {
  const root = document.getElementById("w7");
  if (!root) return;
  let metric = "per-class recall", cls = 5;

  const controls = BB.h("div");
  controls.appendChild(BB.select("metric",
    ["overall accuracy", "per-class recall", "per-class precision"], metric,
    (v) => { metric = v; draw(); }).node);
  controls.appendChild(BB.slider("class", 0, 9, 1, 5, (v) => String(v), (v) => { cls = v; draw(); }).node);
  const body = BB.h("div", { style: "display:flex;gap:1.4rem;flex-wrap:wrap;align-items:center" });
  root.appendChild(controls); root.appendChild(body);

  function draw() {
    BB.clear(body);
    const M = BBDATA.mnist, pc = M.per_class[String(cls)];
    let k, n, label;
    if (metric === "overall accuracy") { k = M.k; n = M.n; label = "accuracy"; }
    else if (metric === "per-class recall") { k = pc.correct; n = pc.support; label = "recall, class " + cls; }
    else { k = pc.correct; n = pc.predicted; label = "precision, class " + cls; }
    const a = k, b = Math.max(n - k, 1e-9);

    const W = 390, H = 230, box = { l: 52, t: 12, w: W - 70, hh: H - 56 };
    const s = BB.svg(W, H);
    const lo = BB.betaInv(0.0005, a, b), hi = Math.min(1, BB.betaInv(0.9995, a, b));
    const pts = [];
    for (let i = 0; i <= 220; i++) { const p = lo + (hi - lo) * i / 220; pts.push([p, BB.betaPdf(p, a, b)]); }
    const ymax = Math.max.apply(null, pts.map((p) => p[1])) * 1.12;
    const sc = BB.axes(s, box, [lo, hi], [0, ymax], label, "posterior density");
    s.appendChild(BB.areaUnder(pts, sc, 0, { fill: BB.C.bayes, fillOpacity: 0.45 }));
    s.appendChild(BB.polyline(pts, sc, { stroke: BB.C.bayesText, w: 1.6 }));
    body.appendChild(s);

    const sdv = Math.sqrt(a * b / ((a + b) * (a + b) * (a + b + 1)));
    const ci = [BB.betaInv(0.025, a, b), BB.betaInv(0.975, a, b)];
    body.appendChild(BB.readout([
      "<b>" + label + "</b>",
      "k = <b>" + k + "</b> / n = <b>" + n + "</b> &nbsp;(the effective denominator)",
      "point estimate <b>" + (k / n).toFixed(4) + "</b>",
      "Beta(" + k + ", " + (n - k) + ") sd <b>" + sdv.toFixed(5) + "</b>",
      "95% [" + ci[0].toFixed(4) + ", " + ci[1].toFixed(4) + "]",
      '<span style="color:#67706E;font-size:.88rem">headline accuracy uses n = ' + M.n +
        "; a single class uses about a tenth of that</span>",
    ]));
  }
  draw();
})();
