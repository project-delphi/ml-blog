/* umapkit.js -- shared pieces for the "When UMAP Lies" widgets. One global, `UK`.
 *
 * Loaded after widget-kit/kit.js (`WK`) and before a part's model.js and
 * widgets.js. Parts 2-4 load it from Part 1's folder:
 *
 *   <script src="../when-umap-lies/umapkit.js"></script>
 *
 * What it adds to the kit:
 *
 *   - the embedding format ("uk-embed-1") written by src/widgetdata.py, and its
 *     decoder: UK.decode, UK.decodeLabels, UK.decodeValues, UK.blend, UK.extent;
 *   - UK.legend (categorical, hover/pin, can drive several linked scatters) and
 *     UK.rampLegend (the continuous ramp);
 *   - UK.lineChart: thin lines with lo-hi bands, direct labels at the right
 *     end, a marker at the current x, drawn at the container's pixel width;
 *   - UK.fitToggle: short button labels when a toggle row would overflow;
 *   - UK.scatter: a canvas scatter for a few thousand points (equal aspect, no
 *     axes, devicePixelRatio-aware), coloured by category or by a continuous
 *     value, with a legend that highlights one category on hover, direct labels
 *     at each category's middle, and hover on the canvas itself;
 *   - UK.dotTable: an SVG table of dot marks (rows = embeddings, columns =
 *     metrics) with min-max whiskers and printed values, laid out at the
 *     container's real pixel width so text never shrinks on a phone;
 *   - small helpers: UK.num (signed number with a real minus), UK.summary,
 *     UK.palette (fresh read, never stale), UK.onTheme, UK.onResize.
 *
 * Colour is always a kit token (`ink`, `muted`, `rule`, `paper`, `surface`,
 * `accent`, `c1`..`c6`). Categories take c1..c6, then ink and muted: eight
 * distinct colours. Categories 9-16 reuse those eight drawn as hollow rings,
 * in the canvas and in the legend; more than 16 is refused. The site's dark
 * categorical tokens do not all separate for colour-blind readers, so the
 * scatter never relies on colour alone: the legend is always shown, each
 * category is labelled where it sits, and hovering a legend entry or a point
 * greys out everything else.
 *
 * The data functions (decode, blend, extent, summary, num) have no DOM
 * dependency and work under node: `const UK = require("./umapkit.js")`.
 */
"use strict";

var UK = (function () {
  var MINUS = "−";
  var CATEGORY_TOKENS = ["c1", "c2", "c3", "c4", "c5", "c6", "ink", "muted"];
  // A direct label is drawn only for a category whose median distance from its
  // own middle is under this share of the whole layout's.
  var LABEL_SPREAD = 0.6;
  // A continuous colouring runs from `rule` mixed this far towards the ramp
  // token, up to the token itself: pure `rule` vanishes on a light panel.
  var RAMP_FLOOR = 0.25;
  var MAX_CATEGORIES = 16;
  var hasDOM = typeof document !== "undefined";
  var W = typeof WK !== "undefined" ? WK : null;

  // ------------------------------------------------------------ the format
  function bytes(b64) {
    if (typeof atob === "function") {
      var s = atob(b64), out = new Uint8Array(s.length);
      for (var i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }

  /** uk-embed-1 -> Float32Array [x0, y0, x1, y1, ...] in the original units.
   *  e = {n, bits (8 or 16), cx, cy, scale, xy: base64}; see src/widgetdata.py. */
  function decode(e) {
    var b = bytes(e.xy), n = e.n, bits = e.bits || 8;
    var half = (Math.pow(2, bits) - 1) / 2, s = e.scale, out = new Float32Array(2 * n), i, q;
    if (b.length !== 2 * n * (bits / 8)) throw new Error("embedding: expected " + n + " points");
    for (i = 0; i < 2 * n; i++) {
      q = bits === 8 ? b[i] : (b[2 * i] | (b[2 * i + 1] << 8));
      out[i] = (q - half) * s + ((i & 1) ? e.cy : e.cx);
    }
    return out;
  }

  /** Base64 of one uint8 per point -> Uint8Array. */
  function decodeLabels(b64) { return bytes(b64); }

  /** A quantised per-point value, {n, bits, lo, hi, q: base64} (widgetdata.encode_values),
   *  -> Float32Array, within (hi - lo) / (2 * (2**bits - 1)) of the original. */
  function decodeValues(e) {
    var b = bytes(e.q), bits = e.bits || 8, top = Math.pow(2, bits) - 1, out = new Float32Array(e.n);
    for (var i = 0; i < e.n; i++) {
      var q = bits === 8 ? b[i] : (b[2 * i] | (b[2 * i + 1] << 8));
      out[i] = e.lo + (e.hi - e.lo) * q / top;
    }
    return out;
  }

  /** Median of an array of numbers. */
  function median(a) {
    var s = Array.prototype.slice.call(a).sort(function (x, y) { return x - y; }), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /** (1 - t) * a + t * b, into `out` if given. */
  function blend(a, b, t, out) {
    out = out || new Float32Array(a.length);
    for (var i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * t;
    return out;
  }

  /** Bounding box {x0, x1, y0, y1} of one or more interleaved xy arrays. */
  function extent() {
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (var a = 0; a < arguments.length; a++) {
      var xy = arguments[a];
      for (var i = 0; i < xy.length; i += 2) {
        if (xy[i] < x0) x0 = xy[i]; if (xy[i] > x1) x1 = xy[i];
        if (xy[i + 1] < y0) y0 = xy[i + 1]; if (xy[i + 1] > y1) y1 = xy[i + 1];
      }
    }
    return { x0: x0, x1: x1, y0: y0, y1: y1 };
  }

  // ---------------------------------------------------------- small numbers
  /** {v: mean, lo: min, hi: max, n} of a number or an array of numbers
   *  ({v: median, ...} with UK.summary(x, "median")). */
  function summary(x, centre) {
    if (x === null || x === undefined) return null;
    if (typeof x === "number") return { v: x, lo: x, hi: x, n: 1 };
    var s = 0, lo = Infinity, hi = -Infinity;
    x.forEach(function (v) { s += v; if (v < lo) lo = v; if (v > hi) hi = v; });
    return { v: centre === "median" ? median(x) : s / x.length, lo: lo, hi: hi, n: x.length };
  }

  /** Fixed decimals with a typographic minus and no "-0.00". */
  function num(x, d) {
    if (x === null || x === undefined || isNaN(x)) return "–";
    d = d === undefined ? 2 : d;
    var s = Math.abs(x).toFixed(d);
    return (x < 0 && Number(s) !== 0 ? MINUS : "") + s;
  }

  /** Token and ring flag for category k of n (see the header for the scheme). */
  function category(k, n) {
    if (n > MAX_CATEGORIES) throw new Error("UK: at most " + MAX_CATEGORIES + " categories");
    return { token: CATEGORY_TOKENS[k % 8], ring: k >= 8 };
  }

  // ----------------------------------------------------------------- theme
  /** Token colours read now. WK.palette caches; a redraw must never see a stale one. */
  function palette() {
    var cs = getComputedStyle(document.documentElement), p = {};
    (W ? W.tokens : CATEGORY_TOKENS).forEach(function (t) {
      p[t] = cs.getPropertyValue("--w-" + t).trim() || "#888888";
    });
    return p;
  }

  /** cb() on a light/dark flip, and again a frame later in case the stylesheet
   *  swap lands after the body class changes. */
  function onTheme(cb) {
    if (!W) return;
    W.onTheme(function () { cb(); requestAnimationFrame(cb); });
  }

  /** cb(width) when an element's content width changes (debounced to a frame). */
  function onResize(node, cb) {
    var last = -1, pending = false;
    function check() {
      pending = false;
      var w = node.clientWidth;
      if (w && w !== last) { last = w; cb(w); }
    }
    function schedule() { if (!pending) { pending = true; requestAnimationFrame(check); } }
    if (window.ResizeObserver) new ResizeObserver(schedule).observe(node);
    else window.addEventListener("resize", schedule);
  }

  /** Keep a WK.toggle on one line: its buttons never break inside a label, and
   *  when the row is wider than its control the buttons take `short` labels
   *  (an array in option order) instead. */
  function fitToggle(toggle, short) {
    var group = toggle.root.querySelector(".widget-toggle");
    var buttons = Array.prototype.slice.call(group.querySelectorAll(".widget-toggle-btn"));
    var long = buttons.map(function (b) { b.style.whiteSpace = "nowrap"; return b.textContent; });
    // The control asks for the long labels' width (so it takes a line of its
    // own when it needs one) but may shrink below it; without min-width 0 the
    // flex item would grow to its unbreakable content and push the page sideways.
    var natural = 0;
    toggle.root.style.minWidth = "0";
    onResize(toggle.root, function (w) {
      buttons.forEach(function (b, i) { b.textContent = long[i]; });
      if (!natural) {
        natural = group.scrollWidth;
        toggle.root.style.flexBasis = natural + "px";
        w = toggle.root.clientWidth;
      }
      if (group.scrollWidth > w) buttons.forEach(function (b, i) { b.textContent = short[i]; });
    });
  }

  /** A segmented control that wraps: chips({label, options: [{value, label}],
   *  value, onchange}) -> {root, get, set}. Same look and aria-pressed as
   *  WK.toggle, but each option is its own pill, so a long list wraps onto
   *  several lines on a phone instead of overflowing. */
  function chips(opts) {
    var current = opts.value, buttons = [];
    var group = W.el("div", { role: "group", "aria-label": opts.label,
      style: { display: "flex", flexWrap: "wrap", gap: "0.35rem" } });
    opts.options.forEach(function (o) {
      var b = W.el("button", { type: "button", class: "widget-toggle-btn", text: o.label,
        "aria-pressed": String(o.value === current),
        style: { border: "1px solid " + W.cssVar("rule"), borderRadius: "999px" },
        onclick: function () { set(o.value); } });
      b._value = o.value; buttons.push(b); group.appendChild(b);
    });
    function set(v, silent) {
      current = v;
      buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b._value === v)); });
      if (!silent && opts.onchange) opts.onchange(v);
    }
    var root = W.el("div", { class: "widget-control", style: { flex: "1 1 100%" } },
      opts.label ? W.el("div", { class: "widget-control-head" }, W.el("span", { text: opts.label })) : null,
      group);
    return { root: root, get: function () { return current; }, set: set };
  }

  var rgbCtx = null;
  /** Any CSS colour -> [r, g, b], via a canvas for the forms we do not parse. */
  function rgb(c) {
    var m = /^#([0-9a-f]{6})$/i.exec(c);
    if (m) return [0, 2, 4].map(function (i) { return parseInt(m[1].substr(i, 2), 16); });
    m = /^#([0-9a-f]{3})$/i.exec(c);
    if (m) return [0, 1, 2].map(function (i) { return 17 * parseInt(m[1][i], 16); });
    m = /^rgba?\(([^)]+)\)/.exec(c);
    if (m) return m[1].split(/[ ,/]+/).slice(0, 3).map(Number);
    rgbCtx = rgbCtx || document.createElement("canvas").getContext("2d");
    rgbCtx.fillStyle = "#000"; rgbCtx.fillStyle = c;
    return rgbCtx.fillStyle.charAt(0) === "#" ? rgb(rgbCtx.fillStyle) : [128, 128, 128];
  }
  function mix(a, b, t) {
    var p = rgb(a), q = rgb(b);
    return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(p[i] + (q[i] - p[i]) * t); }).join(",") + ")";
  }

  function reducedMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  // Deterministic shuffle, so the drawing order interleaves the categories and
  // no category always sits on top.
  function permutation(n, seed) {
    var p = new Uint32Array(n), a = seed | 0, i, j, t;
    for (i = 0; i < n; i++) p[i] = i;
    for (i = n - 1; i > 0; i--) {
      a = (a + 0x6D2B79F5) | 0;
      t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      j = (((t ^ (t >>> 14)) >>> 0) / 4294967296 * (i + 1)) | 0;
      var s = p[i]; p[i] = p[j]; p[j] = s;
    }
    return p;
  }

  // --------------------------------------------------------------- legends
  /**
   * UK.legend({title, classes, counts, onHover(k|null), onPin(k|null)}) -> {root, sync(hi, pinned)}
   *
   * One button per category: a swatch (filled, or a ring for categories 9-16),
   * the name and the count. Hover or focus calls onHover(k); leaving calls
   * onHover(null); a click calls onPin(k), or onPin(null) on the pinned entry.
   * The caller owns the state and reports it back through sync(), so one legend
   * can drive several linked scatters.
   */
  function legend(opts) {
    var el = W.el, items = [], pinned = null;
    var root = el("div", { class: "uk-legend", style: { minWidth: "0" } });
    if (opts.title) root.appendChild(el("div", { class: "widget-control-head", text: opts.title }));
    var list = el("div", { role: "list", style: { display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(" + (opts.minItem || 150) + "px, 1fr))", gap: "0.05rem 0.6rem" } });
    opts.classes.forEach(function (name, k) {
      var c = category(k, opts.classes.length);
      var sw = el("span", { "aria-hidden": "true", style: { flex: "0 0 auto", width: "10px", height: "10px",
        borderRadius: "50%", boxSizing: "border-box",
        background: c.ring ? "transparent" : W.cssVar(c.token),
        border: c.ring ? "2px solid " + W.cssVar(c.token) : "0" } });
      var b = el("button", { type: "button", role: "listitem", "aria-pressed": "false",
        title: "Hover to pick out, click to pin", style: { display: "flex", alignItems: "center",
          gap: "0.45rem", width: "100%", border: "0", background: "transparent", color: W.cssVar("ink"),
          font: "inherit", fontSize: "0.82rem", lineHeight: "1.3", padding: "0.15rem 0.3rem",
          borderRadius: "6px", cursor: "pointer", textAlign: "left" } },
        sw, el("span", { text: name, style: { minWidth: "0" } }),
        opts.counts ? el("span", { text: opts.counts[k].toLocaleString("en-US"),
          style: { marginLeft: "auto", color: W.cssVar("muted"), fontVariantNumeric: "tabular-nums" } }) : null);
      function enter() { if (pinned === null && opts.onHover) opts.onHover(k); }
      function leave() { if (pinned === null && opts.onHover) opts.onHover(null); }
      b.addEventListener("mouseenter", enter);
      b.addEventListener("mouseleave", leave);
      b.addEventListener("focus", enter);
      b.addEventListener("blur", leave);
      b.addEventListener("click", function () { if (opts.onPin) opts.onPin(pinned === k ? null : k); });
      items.push(b); list.appendChild(b);
    });
    root.appendChild(list);
    function sync(hi, pin) {
      pinned = pin === undefined ? null : pin;
      items.forEach(function (b, k) {
        b.style.opacity = hi === null || hi === undefined || hi === k ? "1" : "0.45";
        b.style.background = pinned === k ? W.cssVar("paper") : "transparent";
        b.setAttribute("aria-pressed", String(pinned === k));
      });
    }
    sync(null, null);
    return { root: root, sync: sync };
  }

  var rampId = 0;
  /** UK.rampLegend({title, lo, hi, ramp, loLabel, hiLabel}) -> root: the
   *  single-hue ramp the scatter uses for a continuous value, `rule` to `ramp`. */
  function rampLegend(opts) {
    var id = "uk-ramp-" + (++rampId), sv = W.svg(180, 34, { "aria-label": (opts.title || "Colour scale") +
      ", from " + num(opts.lo) + " to " + num(opts.hi) });
    sv.style.maxWidth = "180px";
    sv.appendChild(W.h("defs", {}, W.h("linearGradient", { id: id },
      W.h("stop", { offset: "0", style: { stopColor: "color-mix(in srgb, " + W.cssVar(opts.ramp || "c1") + " " +
        Math.round(100 * RAMP_FLOOR) + "%, " + W.cssVar("rule") + ")" } }),
      W.h("stop", { offset: "1", style: { stopColor: W.cssVar(opts.ramp || "c1") } }))));
    sv.appendChild(W.h("rect", { x: 0, y: 4, width: 180, height: 10, rx: 2, fill: "url(#" + id + ")" }));
    sv.appendChild(W.h("text", { x: 0, y: 30, "font-size": 11, fill: "muted" }, opts.loLabel || num(opts.lo)));
    sv.appendChild(W.h("text", { x: 180, y: 30, "font-size": 11, fill: "muted", "text-anchor": "end" },
      opts.hiLabel || num(opts.hi)));
    return W.el("div", { class: "uk-legend", style: { minWidth: "0" } },
      W.el("div", { class: "widget-control-head", text: opts.title || "Value" }), sv);
  }

  // --------------------------------------------------------------- scatter
  /**
   * UK.scatter(opts) -> {root, canvas, set, highlight, pin, redraw, state}
   *
   * opts: label (aria text), legendTitle, legend (default true; false for a
   *       panel whose legend is shared, see UK.legend), title (text above the
   *       canvas), aspect (height / width, default 1), maxWidth (CSS, default
   *       "460px" with a legend, none without), radius (px, default by width),
   *       alpha (default 0.8), directLabels (default true), ramp (token for a
   *       continuous colouring, default "c1"), valueLabel (its legend title and
   *       hover text), onHover(k | null) (a category hovered on the canvas).
   * set(d, o): d = {xy, labels?, classes?, counts?, values?, extent?, caption?,
   *       title?, legendTitle?}. xy is interleaved [x0, y0, ...]; labels a
   *       Uint8Array of category indices into classes; values (instead of
   *       labels) a continuous colouring; extent {x0, x1, y0, y1} holds the view
   *       fixed (default: fit the points). o = {animate: ms} tweens from the
   *       previous points when the count is unchanged.
   * highlight(k | null, silent): grey out every category but k; silent skips onHover.
   * pin(k | null): hold k highlighted; canvas hover is ignored while pinned.
   */
  function scatter(opts) {
    opts = opts || {};
    var el = W.el, aspect = opts.aspect || 1, alpha = opts.alpha === undefined ? 0.8 : opts.alpha;
    var hasLegend = opts.legend !== false;
    var root = el("div", { class: "uk-scatter", style: {
      display: "flex", flexWrap: "wrap", gap: "0.8rem 1.4rem", alignItems: "flex-start", minWidth: "0" } });
    var plot = el("div", { style: { position: "relative", flex: hasLegend ? "1 1 260px" : "1 1 auto",
      maxWidth: opts.maxWidth || (hasLegend ? "460px" : "none"), minWidth: "0", width: hasLegend ? null : "100%" } });
    var title = el("div", { class: "widget-control-head", style: { display: opts.title ? "flex" : "none" } });
    if (opts.title) title.textContent = opts.title;
    var canvas = el("canvas", { role: "img", "aria-label": opts.label || "Scatter plot",
      style: { display: "block", width: "100%", touchAction: "pan-y" } });
    var tip = el("div", { "aria-hidden": "true", style: { position: "absolute", left: "6px",
      top: opts.title ? "1.6rem" : "4px",
      fontSize: "0.8rem", fontWeight: "600", color: W.cssVar("ink"), pointerEvents: "none",
      background: W.cssVar("surface"), padding: "0 0.3rem", borderRadius: "4px", display: "none" } });
    var caption = el("div", { class: "widget-note", style: { marginTop: "0.35rem" } });
    plot.appendChild(title); plot.appendChild(canvas); plot.appendChild(tip); plot.appendChild(caption);
    var legendBox = el("div", { style: { flex: "1 1 170px", minWidth: "0" } });
    root.appendChild(plot);
    if (hasLegend) root.appendChild(legendBox);
    // A CPU-backed canvas: a few thousand dots raster in well under a frame, and
    // several panels redrawn every frame during a tween do not queue work on
    // the GPU process (in headless Chrome that stalled the page for seconds).
    var ctx = canvas.getContext("2d", { willReadFrequently: true });

    var S = { xy: null, labels: null, values: null, classes: null, counts: null, groups: null,
      ext: null, fixedExt: null, hi: null, pinned: null, px: null, anim: null, n: 0 };
    var leg = null;

    function fitExtent(xy) { return S.fixedExt || extent(xy); }

    function buildGroups() {
      // Point indices per category, in shuffled order, so the draw can batch
      // by colour and still interleave (see draw()).
      var k = S.classes && !S.values ? S.classes.length : 1, g = [], perm = permutation(S.n, 20261120), i;
      for (i = 0; i < k; i++) g.push([]);
      for (i = 0; i < S.n; i++) {
        var p = perm[i];
        g[k > 1 ? S.labels[p] : 0].push(p);
      }
      S.groups = g.map(function (a) { return Uint32Array.from(a); });
    }

    function buildLegend() {
      leg = null;
      if (!hasLegend) return;
      W.clear(legendBox);
      if (S.values) {
        legendBox.appendChild(rampLegend({ title: opts.valueLabel, lo: S.vrange[0], hi: S.vrange[1], ramp: opts.ramp }));
        return;
      }
      leg = legend({ title: opts.legendTitle, classes: S.classes, counts: S.counts,
        onHover: function (k) { highlight(k === null ? S.pinned : k); },
        onPin: function (k) { S.pinned = k; highlight(k); syncLegend(); } });
      legendBox.appendChild(leg.root);
    }

    function syncLegend() { if (leg) leg.sync(S.hi, S.pinned); }

    function highlight(k, silent) {
      k = k === undefined ? null : k;
      if (k === S.hi) return;
      S.hi = k; syncLegend(); draw();
      if (!silent && opts.onHover) opts.onHover(k);
    }

    function pin(k) { S.pinned = k === undefined ? null : k; highlight(S.pinned, true); syncLegend(); }

    function set(d, o) {
      var same = S.xy && d.xy.length === S.xy.length && !d.classes;
      var from = S.xy, fromExt = S.ext;
      if (d.extent !== undefined) S.fixedExt = d.extent;
      S.n = d.xy.length / 2;
      var regroup = !S.groups || !same && !!S.xy && d.xy.length !== S.xy.length;
      if (d.classes) { S.classes = d.classes; S.counts = d.counts || null; S.pinned = null; S.hi = null; S.values = null; regroup = true; }
      if (d.labels) { S.labels = d.labels; regroup = true; }
      if (S.labels && S.labels.length !== S.n) {
        throw new Error("UK.scatter.set: " + S.n + " points but " + S.labels.length +
          " labels; pass labels whenever the point count changes");
      }
      if (d.values !== undefined) {
        S.values = d.values; S.vrange = d.values ? (d.valueRange || valueRange(d.values)) : null; regroup = true;
      }
      if (d.caption !== undefined) caption.textContent = d.caption;
      if (d.title !== undefined) { title.textContent = d.title; title.style.display = d.title ? "flex" : "none"; }
      if (d.legendTitle !== undefined) opts.legendTitle = d.legendTitle;
      S.n = d.xy.length / 2;
      if (regroup) { buildGroups(); buildLegend(); syncLegend(); }
      var target = d.xy, targetExt = fitExtent(target);
      var ms = o && o.animate && same && !reducedMotion() ? o.animate : 0;
      if (!ms) { S.xy = target; S.ext = targetExt; S.anim = null; draw(); return; }
      var t0 = performance.now(), buf = new Float32Array(target.length), token = {};
      S.anim = token;
      (function step(now) {
        if (S.anim !== token) return;
        var t = Math.min(1, (now - t0) / ms), e = t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
        S.xy = t < 1 ? blend(from, target, e, buf) : target;
        S.ext = { x0: fromExt.x0 + (targetExt.x0 - fromExt.x0) * e, x1: fromExt.x1 + (targetExt.x1 - fromExt.x1) * e,
          y0: fromExt.y0 + (targetExt.y0 - fromExt.y0) * e, y1: fromExt.y1 + (targetExt.y1 - fromExt.y1) * e };
        draw();
        if (t < 1) requestAnimationFrame(step); else S.anim = null;
      })(t0);
    }

    function draw() {
      var w = canvas.clientWidth;
      if (!w || !S.xy) return;
      var hgt = Math.round(w * aspect), dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hgt * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(hgt * dpr);
        canvas.style.height = hgt + "px";
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, hgt);
      var P = palette(), xy = S.xy, n = S.n, e = S.ext;
      var r = opts.radius || Math.max(1.3, Math.min(2.6, w / 190));
      var pad = 8 + r, dx = (e.x1 - e.x0) || 1, dy = (e.y1 - e.y0) || 1;
      var s = Math.min((w - 2 * pad) / dx, (hgt - 2 * pad) / dy);
      var mx = (e.x0 + e.x1) / 2, my = (e.y0 + e.y1) / 2;
      if (!S.px || S.px.length !== 2 * n) S.px = new Float32Array(2 * n);
      var px = S.px, i;
      for (i = 0; i < n; i++) {  // y up
        px[2 * i] = w / 2 + s * (xy[2 * i] - mx);
        px[2 * i + 1] = hgt / 2 - s * (xy[2 * i + 1] - my);
      }
      var big = n > 20000, TAU = Math.PI * 2;
      function mark(p, rad, ring) {
        var x = px[2 * p], y = px[2 * p + 1];
        if (big && !ring) { ctx.rect(x - rad, y - rad, 2 * rad, 2 * rad); return; }
        ctx.moveTo(x + rad, y); ctx.arc(x, y, rad, 0, TAU);
      }
      function paint(idx, from, to, colour, rad, ring) {
        ctx.beginPath();
        for (var j = from; j < to; j++) mark(idx[j], ring ? rad * 0.85 : rad, ring);
        if (ring) { ctx.strokeStyle = colour; ctx.lineWidth = Math.max(1, rad * 0.55); ctx.stroke(); }
        else { ctx.fillStyle = colour; ctx.fill(); }
      }
      var K = S.groups.length, ROUNDS = 8, round, k, len;
      if (S.values) {
        // Continuous: 24 bins from `rule` to the ramp token, drawn low to high.
        var lo = S.vrange[0], hi = S.vrange[1], B = 24, bins = [];
        for (k = 0; k < B; k++) bins.push([]);
        for (i = 0; i < n; i++) bins[Math.max(0, Math.min(B - 1, Math.floor((S.values[i] - lo) / ((hi - lo) || 1) * B)))].push(i);
        ctx.globalAlpha = alpha;
        bins.forEach(function (b, k) { paint(b, 0, b.length, mix(P.rule, P[opts.ramp || "c1"], RAMP_FLOOR + (1 - RAMP_FLOOR) * (k + 0.5) / B), r, false); });
      } else if (S.hi === null) {
        // Draw in eight rounds, each taking the next eighth of every category,
        // so categories interleave and the canvas still batches by colour.
        ctx.globalAlpha = alpha;
        for (round = 0; round < ROUNDS; round++) {
          for (k = 0; k < K; k++) {
            len = S.groups[k].length;
            var c = S.classes ? category(k, K) : { token: "c1", ring: false };
            paint(S.groups[k], Math.floor(len * round / ROUNDS), Math.floor(len * (round + 1) / ROUNDS),
              P[c.token], r, c.ring);
          }
        }
      } else {
        ctx.globalAlpha = 0.22;
        for (k = 0; k < K; k++) if (k !== S.hi) paint(S.groups[k], 0, S.groups[k].length, P.muted, r, false);
        ctx.globalAlpha = 1;
        var ch = category(S.hi, K);
        paint(S.groups[S.hi], 0, S.groups[S.hi].length, P[ch.token], r * 1.2, ch.ring);
      }
      ctx.globalAlpha = 1;
      if (S.classes && !S.values && opts.directLabels !== false) drawLabels(P, w, hgt);
    }

    // Each category's name at the point nearest its median, nudged off
    // labels already placed; a label with nowhere to go is left out.
    function drawLabels(P, w, hgt) {
      var px = S.px, placed = [], order = S.groups.map(function (g, k) { return k; });
      var ax = [], ay = [], ad = [], i;
      for (i = 0; i < S.n; i++) { ax.push(px[2 * i]); ay.push(px[2 * i + 1]); }
      var amx = median(ax), amy = median(ay);
      for (i = 0; i < S.n; i++) ad.push((ax[i] - amx) * (ax[i] - amx) + (ay[i] - amy) * (ay[i] - amy));
      var allSpread = Math.sqrt(median(ad));
      order.sort(function (a, b) { return S.groups[b].length - S.groups[a].length; });
      ctx.font = "600 11px " + (getComputedStyle(canvas).fontFamily || "system-ui, sans-serif");
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.lineJoin = "round";
      order.forEach(function (k) {
        if (S.hi !== null && k !== S.hi) return;
        var g = S.groups[k];
        if (!g.length) return;
        var xs = [], ys = [], j;
        for (j = 0; j < g.length; j++) { xs.push(px[2 * g[j]]); ys.push(px[2 * g[j] + 1]); }
        var mxs = median(xs), mys = median(ys), best = 0, bd = Infinity, ds = [];
        for (j = 0; j < g.length; j++) {
          var dd = (xs[j] - mxs) * (xs[j] - mxs) + (ys[j] - mys) * (ys[j] - mys);
          ds.push(dd);
          if (dd < bd) { bd = dd; best = j; }
        }
        // A category spread over most of the layout has no place to name: its
        // median is a point in the middle that says nothing (the random elephant).
        if (Math.sqrt(median(ds)) > LABEL_SPREAD * allSpread) return;
        var text = S.classes[k], tw = ctx.measureText(text).width + 6, th = 15;
        var offs = [[0, 0], [0, -16], [0, 16], [0, -32], [0, 32], [-tw / 2, -16], [tw / 2, -16], [-tw / 2, 16], [tw / 2, 16]];
        for (var o = 0; o < offs.length; o++) {
          var x = Math.max(tw / 2 + 2, Math.min(w - tw / 2 - 2, xs[best] + offs[o][0]));
          var y = Math.max(th / 2 + 2, Math.min(hgt - th / 2 - 2, ys[best] + offs[o][1]));
          var box = [x - tw / 2, y - th / 2, x + tw / 2, y + th / 2];
          if (placed.every(function (b) { return box[2] < b[0] || box[0] > b[2] || box[3] < b[1] || box[1] > b[3]; })) {
            placed.push(box);
            ctx.strokeStyle = P.surface; ctx.lineWidth = 3.5; ctx.strokeText(text, x, y);
            ctx.fillStyle = P.ink; ctx.fillText(text, x, y);
            return;
          }
        }
      });
    }

    // Hover or tap on the canvas: the nearest point within 12 px picks its
    // category, or, under a continuous colouring, reports its value.
    function nearest(ev) {
      if (!S.px) return null;
      var rect = canvas.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
      var best = -1, bd = 144, px = S.px;
      for (var i = 0; i < S.n; i++) {
        var dx = px[2 * i] - x, dy = px[2 * i + 1] - y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      return best < 0 ? null : best;
    }
    var hoverPending = null;
    canvas.addEventListener("pointermove", function (ev) {
      if (hoverPending) return;
      hoverPending = requestAnimationFrame(function () {
        hoverPending = null;
        if (S.pinned !== null) return;
        var i = nearest(ev);
        tip.style.display = i === null ? "none" : "block";
        if (S.values) {
          if (i !== null) tip.textContent = (opts.valueLabel || "value") + " " + num(S.values[i], 1);
          return;
        }
        if (!S.classes) return;
        var k = i === null ? null : S.labels[i];
        highlight(k);
        if (k !== null) tip.textContent = S.classes[k] + (S.counts ? " (" + S.counts[k].toLocaleString("en-US") + ")" : "");
      });
    });
    canvas.addEventListener("pointerleave", function () {
      tip.style.display = "none";
      highlight(S.pinned);
    });

    onResize(plot, draw);
    onTheme(draw);
    return { root: root, canvas: canvas, set: set, highlight: highlight, pin: pin, redraw: draw,
      state: function () { return { hi: S.hi, pinned: S.pinned, n: S.n, extent: S.ext }; } };
  }

  function valueRange(v) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < v.length; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
    return [lo, hi];
  }


  // -------------------------------------------------------------- dotTable
  /**
   * UK.dotTable(opts) -> {root, update(newOpts)}
   *
   * A grid of dot marks: one row per series (an embedding), one column per
   * metric, all columns on one shared axis `domain`. Columns that do not fit
   * the container's width wrap into further bands, each repeating the row
   * labels, so a phone gets two columns at a time at full text size.
   *
   * opts: rows [{key, label, mark: "dot"|"ring"|"square"|"diamond"|"cross",
   *              token, emph}],
   *       columns [{key, label: string | [line, line], title}],
   *       value(rowKey, colKey) -> null | {v, lo, hi, title},
   *       domain [lo, hi], ticks (default [lo, 0, hi]), digits (default 2),
   *       label (aria), minCol (px, default 92), labelWidth (px, default 124).
   */
  function dotTable(opts) {
    var root = W.el("div", { class: "uk-dot-table", style: { minWidth: "0" } });
    function render() {
      var w = root.clientWidth || 600;
      var LW = Math.min(opts.labelWidth || 124, Math.max(96, w * 0.34)), MINC = opts.minCol || 92;
      var per = Math.max(1, Math.min(opts.columns.length, Math.floor((w - LW) / MINC)));
      var colW = Math.min(170, (w - LW) / per), ROW = 22, TXT = 34;
      var lines = Math.max.apply(null, opts.columns.map(function (c) { return Array.isArray(c.label) ? c.label.length : 1; }));
      var HEAD = 13 * lines + 8, AX = 18, GAP = 14;
      var bands = [];
      for (var b = 0; b < opts.columns.length; b += per) bands.push(opts.columns.slice(b, b + per));
      var bandH = HEAD + ROW * opts.rows.length + AX, H = bands.length * bandH + (bands.length - 1) * GAP;
      var lo = opts.domain[0], hi = opts.domain[1];
      var ticks = opts.ticks || (lo < 0 && hi > 0 ? [lo, 0, hi] : [lo, hi]);
      var sv = W.svg(w, H, { "aria-label": opts.label || "Metric table", width: w, height: H });
      bands.forEach(function (cols, bi) {
        var top = bi * (bandH + GAP), y0 = top + HEAD;
        opts.rows.forEach(function (r, ri) {
          var cy = y0 + ri * ROW + ROW / 2;
          if (r.emph) sv.appendChild(W.h("rect", { x: 0, y: y0 + ri * ROW, width: LW + cols.length * colW,
            height: ROW, rx: 4, fill: "paper" }));
          sv.appendChild(markEl(r.mark, 7, cy, 4, r.token));
          sv.appendChild(W.h("text", { x: 16, y: cy + 4, "font-size": 11.5, fill: "ink",
            "font-weight": r.emph ? 700 : 400 }, r.label));
        });
        cols.forEach(function (c, ci) {
          var x0 = LW + ci * colW, ax0 = x0 + 6, ax1 = x0 + colW - TXT - 4;
          var sc = W.lin([lo, hi], [ax0, ax1]);
          var head = Array.isArray(c.label) ? c.label : [c.label];
          var ht = W.h("text", { x: (ax0 + ax1) / 2, y: top + 12, "text-anchor": "middle", "font-size": 11,
            fill: "ink", "font-weight": 600 });
          head.forEach(function (line, li) {
            ht.appendChild(W.h("tspan", { x: (ax0 + ax1) / 2, dy: li ? 13 : 0 }, line));
          });
          if (c.title) ht.appendChild(W.h("title", {}, c.title));
          sv.appendChild(ht);
          var yb = y0 + ROW * opts.rows.length;
          sv.appendChild(W.h("line", { x1: ax0, x2: ax1, y1: yb, y2: yb, stroke: "rule" }));
          ticks.forEach(function (t) {
            var tx = sc(t);
            sv.appendChild(W.h("line", { x1: tx, x2: tx, y1: y0, y2: yb + 3, stroke: "rule",
              "stroke-width": t === 0 ? 1 : 0.6, "stroke-dasharray": t === 0 ? null : "2 3" }));
          });
          // Tick labels by priority (0, then the ends); one that would touch a
          // label already placed is left out, so a narrow column never shows
          // "-0.250".
          var placedTicks = [];
          ticks.slice().sort(function (a, b) { return (a === 0 ? 0 : 1) - (b === 0 ? 0 : 1) || b - a; })
            .forEach(function (t) {
              var tx = sc(t), label = num(t, t === Math.round(t) ? 0 : 2), half = label.length * 2.9 + 2;
              if (placedTicks.some(function (p) { return Math.abs(p.x - tx) < p.half + half; })) return;
              placedTicks.push({ x: tx, half: half });
              sv.appendChild(W.h("text", { x: tx, y: yb + 14, "text-anchor": "middle", "font-size": 9.5,
                fill: "muted" }, label));
            });
          opts.rows.forEach(function (r, ri) {
            var v = opts.value(r.key, c.key);
            if (!v || v.v === null || v.v === undefined) return;
            var cy = y0 + ri * ROW + ROW / 2, g = W.h("g", {});
            if (v.lo !== undefined && v.hi !== undefined && v.hi > v.lo) {
              g.appendChild(W.h("line", { x1: sc(v.lo), x2: sc(v.hi), y1: cy, y2: cy, stroke: "muted", "stroke-width": 1.5 }));
              [v.lo, v.hi].forEach(function (e) {
                g.appendChild(W.h("line", { x1: sc(e), x2: sc(e), y1: cy - 3, y2: cy + 3, stroke: "muted", "stroke-width": 1.2 }));
              });
            }
            g.appendChild(markEl(r.mark, sc(Math.max(lo, Math.min(hi, v.v))), cy, r.emph ? 5 : 4, r.token));
            g.appendChild(W.h("text", { x: x0 + colW - 2, y: cy + 4, "text-anchor": "end", "font-size": 11,
              fill: r.emph ? "ink" : "muted", "font-weight": r.emph ? 700 : 400,
              style: { fontVariantNumeric: "tabular-nums" } }, num(v.v, opts.digits)));
            g.appendChild(W.h("title", {}, v.title || (r.label + ": " + num(v.v, 3))));
            sv.appendChild(g);
          });
        });
      });
      W.clear(root); root.appendChild(sv);
    }
    onResize(root, render);
    return { root: root, update: function (o) { Object.assign(opts, o || {}); render(); } };
  }

  // ------------------------------------------------------------- lineChart
  /**
   * UK.lineChart(opts) -> {root, update(newOpts)}
   *
   * Thin lines on one y-axis, each with an optional lo-hi band, labelled
   * directly at its right-hand end (no legend), drawn at the container's real
   * pixel width. A marker line picks out the current x.
   *
   * opts: series [{key, label, token, dash ("2 3" for a reference line),
   *                band (default true), dots (default true)}],
   *       points(key) -> [{x, v, lo, hi}] (v null to leave a gap),
   *       xLog (bool), xTicks [x], xFormat(x), xLabel,
   *       yDomain [lo, hi], yTicks [y], yLabel, digits (default 2),
   *       marker (an x or null), markerLabel, height (px, default 260), label (aria).
   */
  function lineChart(opts) {
    var root = W.el("div", { class: "uk-line-chart", style: { minWidth: "0" } });
    function render() {
      var w = root.clientWidth || 600, H = opts.height || 260;
      var labelW = 20 + 6.3 * Math.max.apply(null, opts.series.map(function (s) { return s.label.length; }));
      var ML = 40, MR = Math.min(labelW, w * 0.4), MT = 22, MB = 38;
      var tx = opts.xLog ? function (x) { return Math.log10(x); } : function (x) { return x; };
      var xs = [];
      opts.series.forEach(function (s) { opts.points(s.key).forEach(function (p) { xs.push(tx(p.x)); }); });
      var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
      var sx = W.lin([x0, x1], [ML, w - MR - 8]), sy = W.lin(opts.yDomain, [H - MB, MT]);
      var X = function (x) { return sx(tx(x)); };
      var sv = W.svg(w, H, { "aria-label": opts.label || "Line chart", width: w, height: H });
      var d = opts.digits === undefined ? 2 : opts.digits;
      (opts.yTicks || W.ticks(opts.yDomain[0], opts.yDomain[1], 4)).forEach(function (t) {
        sv.appendChild(W.h("line", { x1: ML, x2: w - MR - 8, y1: sy(t), y2: sy(t), stroke: "rule", "stroke-width": 0.8 }));
        sv.appendChild(W.h("text", { x: ML - 6, y: sy(t) + 3.5, "text-anchor": "end", "font-size": 10, fill: "muted" },
          num(t, t === Math.round(t) ? 0 : Math.abs(t * 10 - Math.round(t * 10)) < 1e-9 ? 1 : 2)));
      });
      var fx = opts.xFormat || function (x) { return String(x); };
      var lastTick = -Infinity;
      (opts.xTicks || []).forEach(function (t) {
        if (X(t) - lastTick < 28) return;  // a tick label with no room is left out
        lastTick = X(t);
        sv.appendChild(W.h("line", { x1: X(t), x2: X(t), y1: H - MB, y2: H - MB + 4, stroke: "muted" }));
        sv.appendChild(W.h("text", { x: X(t), y: H - MB + 15, "text-anchor": "middle", "font-size": 10, fill: "muted" }, fx(t)));
      });
      if (opts.xLabel) sv.appendChild(W.h("text", { x: (ML + w - MR - 8) / 2, y: H - 4, "text-anchor": "middle",
        "font-size": 11, fill: "muted" }, opts.xLabel));
      if (opts.yLabel) sv.appendChild(W.h("text", { x: 2, y: 11, "font-size": 11, fill: "muted" }, opts.yLabel));
      if (opts.marker !== null && opts.marker !== undefined) {
        var mx = X(opts.marker);
        sv.appendChild(W.h("line", { x1: mx, x2: mx, y1: MT - 4, y2: H - MB, stroke: "accent", "stroke-width": 1.2 }));
        sv.appendChild(W.h("text", { x: mx, y: MT - 8, "text-anchor": mx > w - MR - 40 ? "end" : "middle",
          "font-size": 10.5, fill: "accent", "font-weight": 700 }, opts.markerLabel || fx(opts.marker)));
      }
      var ends = [];
      opts.series.forEach(function (s) {
        var pts = opts.points(s.key).filter(function (p) { return p.v !== null && p.v !== undefined; });
        if (!pts.length) return;
        if (s.band !== false && pts[0].lo !== undefined) {
          var up = pts.map(function (p) { return X(p.x) + "," + sy(p.hi); });
          var dn = pts.slice().reverse().map(function (p) { return X(p.x) + "," + sy(p.lo); });
          sv.appendChild(W.h("polygon", { points: up.concat(dn).join(" "), fill: s.token, "fill-opacity": 0.16 }));
        }
        sv.appendChild(W.h("polyline", { points: pts.map(function (p) { return X(p.x) + "," + sy(p.v); }).join(" "),
          fill: "none", stroke: s.token, "stroke-width": s.dash ? 1.3 : 1.7, "stroke-dasharray": s.dash || null,
          "stroke-linejoin": "round" }));
        pts.forEach(function (p) {
          var on = p.x === opts.marker;
          if (s.dots === false && !on) return;
          var c = W.h("g", {}, markEl(s.marker || "dot", X(p.x), sy(p.v), on ? 3.8 : 2.1, s.token));
          c.appendChild(W.h("title", {}, s.label + " at " + fx(p.x) + ": " + num(p.v, 3) +
            (p.lo !== undefined && p.hi > p.lo ? " (" + num(p.lo, 3) + " to " + num(p.hi, 3) + ")" : "")));
          sv.appendChild(c);
        });
        var last = pts[pts.length - 1];
        ends.push({ s: s, y: sy(last.v), x: X(last.x) });
      });
      // Direct labels: in end order, pushed apart to 12 px and kept in the plot.
      ends.sort(function (a, b) { return a.y - b.y; });
      var GAP = 12, i;
      for (i = 1; i < ends.length; i++) ends[i].ly = Math.max(ends[i].y, (ends[i - 1].ly || ends[i - 1].y) + GAP);
      if (ends.length) ends[0].ly = ends[0].ly || ends[0].y;
      for (i = ends.length - 1; i >= 0; i--) {
        var cap = i === ends.length - 1 ? H - MB : ends[i + 1].ly - GAP;
        ends[i].ly = Math.min(ends[i].ly, cap);
      }
      ends.forEach(function (e) {
        var lx = w - MR - 2;
        sv.appendChild(W.h("line", { x1: lx, x2: lx + 16, y1: e.ly, y2: e.ly, stroke: e.s.token,
          "stroke-width": 1.8, "stroke-dasharray": e.s.dash || null }));
        if (e.s.marker) sv.appendChild(markEl(e.s.marker, lx + 8, e.ly, 2.4, e.s.token));
        sv.appendChild(W.h("text", { x: lx + 20, y: e.ly + 3.5, "font-size": 11, fill: "ink" }, e.s.label));
      });
      W.clear(root); root.appendChild(sv);
    }
    onResize(root, render);
    return { root: root, update: function (o) { Object.assign(opts, o || {}); render(); } };
  }

  /** One SVG mark at (x, y) of radius r in a token colour. */
  function markEl(kind, x, y, r, token) {
    if (kind === "ring") return W.h("circle", { cx: x, cy: y, r: r - 0.5, fill: "surface", stroke: token, "stroke-width": 1.6 });
    if (kind === "square") return W.h("rect", { x: x - r + 0.5, y: y - r + 0.5, width: 2 * r - 1, height: 2 * r - 1, fill: token });
    if (kind === "diamond") return W.h("path", { d: "M" + x + " " + (y - r - 1) + "L" + (x + r + 1) + " " + y + "L" + x + " " +
      (y + r + 1) + "L" + (x - r - 1) + " " + y + "Z", fill: token });
    if (kind === "cross") return W.h("path", { d: "M" + (x - r) + " " + (y - r) + "L" + (x + r) + " " + (y + r) + "M" + (x + r) + " " +
      (y - r) + "L" + (x - r) + " " + (y + r), stroke: token, "stroke-width": 1.8, fill: "none" });
    if (kind === "plus") return W.h("path", { d: "M" + (x - r - 1) + " " + y + "H" + (x + r + 1) + "M" + x + " " + (y - r - 1) +
      "V" + (y + r + 1), stroke: token, "stroke-width": 1.8, fill: "none" });
    if (kind === "plus-filled") return W.h("path", { d: "M" + (x - r - 0.5) + " " + y + "H" + (x + r + 0.5) + "M" + x + " " +
      (y - r - 0.5) + "V" + (y + r + 0.5), stroke: token, "stroke-width": Math.max(2.4, r * 0.9), fill: "none" });
    if (kind === "triangle" || kind === "triangle-down") {
      var s = kind === "triangle" ? 1 : -1, R = r + 1.2;
      return W.h("path", { d: "M" + x + " " + (y - s * R) + "L" + (x + R * 0.87) + " " + (y + s * R * 0.5) + "L" +
        (x - R * 0.87) + " " + (y + s * R * 0.5) + "Z", fill: token });
    }
    if (kind === "hexagon") {  // no method uses it: for series that are not methods
      var hx = [];
      for (var j = 0; j < 6; j++) {
        var b = Math.PI / 6 + j * Math.PI / 3;
        hx.push((x + (r + 1) * Math.cos(b)).toFixed(2) + "," + (y + (r + 1) * Math.sin(b)).toFixed(2));
      }
      return W.h("polygon", { points: hx.join(" "), fill: token });
    }
    if (kind === "star") {
      var pts = [];
      for (var i = 0; i < 10; i++) {
        var a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? (r + 1.4) * 0.45 : r + 1.4;
        pts.push((x + rr * Math.cos(a)).toFixed(2) + "," + (y + rr * Math.sin(a)).toFixed(2));
      }
      return W.h("polygon", { points: pts.join(" "), fill: token });
    }
    return W.h("circle", { cx: x, cy: y, r: r, fill: token });
  }

  // ---------------------------------------------------------- method style
  /**
   * UK.METHOD: one style per method, the same as the matplotlib figures
   * (umaplab/plotting.py METHOD_STYLE), so a method keeps its colour, line and
   * marker in every figure and widget of the series. {token, dash, marker, name}:
   * dash is an SVG stroke-dasharray (null = solid); marker a UK.mark kind.
   * A "·pca50" arm takes its parent's token with a dashed line. The spectral
   * family shares `muted` and is told apart by dash and marker. The two
   * reference lines (kNN graphs a method starts from) are `ink`, dotted, with
   * sparser dots for the PCA-50 graph. UK.method(key) falls back to muted.
   */
  var DASHED = "5 3";
  var METHOD = {
    "umap": { token: "c1", dash: null, marker: "dot", name: "UMAP" },
    "umap·pca50": { token: "c1", dash: DASHED, marker: "dot", name: "UMAP on PCA-50" },
    // densMAP is UMAP with a density term: UMAP's hue, dotted, open marker.
    // Not in plotting.py's METHOD_STYLE; add it there if a figure shows it.
    "umap·densmap": { token: "c1", dash: "1.5 2", marker: "ring", name: "densMAP" },
    "tsne": { token: "c3", dash: null, marker: "square", name: "t-SNE" },
    "tsne·pca50": { token: "c3", dash: DASHED, marker: "square", name: "t-SNE on PCA-50" },
    "pca": { token: "c5", dash: null, marker: "diamond", name: "PCA" },
    "pacmap": { token: "c4", dash: null, marker: "triangle", name: "PaCMAP" },
    "trimap": { token: "c6", dash: null, marker: "triangle-down", name: "TriMap" },
    "phate": { token: "c2", dash: null, marker: "plus-filled", name: "PHATE" },
    "isomap": { token: "muted", dash: null, marker: "cross", name: "Isomap" },
    "spectral": { token: "muted", dash: "1.5 2.5", marker: "plus", name: "Laplacian eigenmaps" },
    "diffmap": { token: "muted", dash: "6 2.5 1.5 2.5", marker: "star", name: "Diffusion map" },
    "input": { token: "ink", dash: "1 1.6", marker: null, name: "input kNN graph" },
    "pca50graph": { token: "ink", dash: "1 4", marker: null, name: "PCA-50 kNN graph" }
  };
  function method(key) {
    return METHOD[key] || { token: "muted", dash: null, marker: "dot", name: key };
  }

  var api = {
    decode: decode, decodeLabels: decodeLabels, decodeValues: decodeValues, blend: blend, extent: extent,
    median: median, METHOD: METHOD, method: method,
    summary: summary, num: num, category: category, categoryTokens: CATEGORY_TOKENS.slice(),
    MAX_CATEGORIES: MAX_CATEGORIES
  };
  if (hasDOM) {
    Object.assign(api, { palette: palette, onTheme: onTheme, onResize: onResize, mix: mix, fitToggle: fitToggle, chips: chips,
      scatter: scatter, dotTable: dotTable, lineChart: lineChart, legend: legend, rampLegend: rampLegend,
      mark: markEl });
  }
  return api;
})();

if (typeof module !== "undefined" && module.exports) module.exports = UK;
