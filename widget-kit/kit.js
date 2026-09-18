/* widget-kit/kit.js -- the shared runtime for this blog's interactive widgets.
 *
 * Loaded on every post through posts/_metadata.yml (include-in-header), before
 * a post's own widgets.js. Exposes one global, `WK`. Dependency-free: no CDN,
 * no module loader, no framework, so a page works from file:// as well as over
 * HTTP, and a widget file is plain JavaScript that calls these helpers.
 *
 * Colour never appears as a hex value in a widget. A fill or stroke is named by
 * token (`ink`, `muted`, `rule`, `paper`, `surface`, `accent`, `c1`..`c6`) and
 * the kit writes it as `var(--w-<token>)`, which theme/light.scss and
 * theme/dark.scss define, so the reader's light/dark choice repaints every
 * widget with no redraw. `WK.palette()` returns the same tokens as resolved
 * colours for the rare canvas widget; `WK.onTheme` tells it when to repaint.
 *
 * Markup contract for the frame is in widget-kit/chrome.scss.
 */
"use strict";

var WK = (function () {
  var SVG_NS = "http://www.w3.org/2000/svg";
  var TOKENS = ["ink", "muted", "rule", "paper", "surface", "accent",
                "c1", "c2", "c3", "c4", "c5", "c6"];
  var COLOUR_ATTRS = { fill: 1, stroke: 1, color: 1 };

  function isToken(v) { return typeof v === "string" && TOKENS.indexOf(v) >= 0; }
  function cssVar(token) { return "var(--w-" + token + ")"; }

  // -------------------------------------------------------------- elements
  // Shared attribute handling for `el` and `h`: `text`, `html`, a `style`
  // object, `class`, `on<event>` handlers, and token colours.
  function apply(e, attrs, svg) {
    var k, v;
    for (k in (attrs || {})) {
      v = attrs[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") e.textContent = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else if (k.slice(0, 2) === "on" && typeof v === "function") e.addEventListener(k.slice(2), v);
      else if (COLOUR_ATTRS[k] && isToken(v)) e.style[k] = cssVar(v);
      else if (k === "class" && svg) e.setAttribute("class", v);
      else e.setAttribute(k, v === true ? "" : v);
    }
  }

  function append(e, kids) {
    kids.forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      if (Array.isArray(c)) append(e, c);
      else e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
  }

  /** HTML element: el("div", {class: "x", text: "hi"}, child, ...). */
  function el(tag, attrs) {
    var e = document.createElement(tag);
    apply(e, attrs, false);
    append(e, Array.prototype.slice.call(arguments, 2));
    return e;
  }

  /** SVG element in the SVG namespace: h("circle", {r: 4, fill: "c1"}). */
  function h(tag, attrs) {
    var e = document.createElementNS(SVG_NS, tag);
    apply(e, attrs, true);
    append(e, Array.prototype.slice.call(arguments, 2));
    return e;
  }

  /** A responsive <svg> with the given viewBox; width follows the container. */
  function svg(w, hgt, attrs) {
    var a = Object.assign({
      viewBox: "0 0 " + w + " " + hgt,
      preserveAspectRatio: "xMidYMid meet",
      role: "img"
    }, attrs || {});
    return h("svg", a);
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  // ---------------------------------------------------------------- theme
  var paletteCache = null;

  /** The tokens resolved to colours, for canvas or anything that cannot use CSS. */
  function palette() {
    if (paletteCache) return paletteCache;
    var cs = getComputedStyle(document.documentElement), p = {};
    TOKENS.forEach(function (t) { p[t] = cs.getPropertyValue("--w-" + t).trim() || "#888888"; });
    paletteCache = p;
    return p;
  }

  /** Runs `cb(palette)` whenever the light/dark toggle flips. */
  function onTheme(cb) {
    if (!window.MutationObserver) return;
    new MutationObserver(function () {
      paletteCache = null;
      cb(palette());
    }).observe(document.body, { attributes: true, attributeFilter: ["class"] });
  }

  // --------------------------------------------------------------- scales
  /** Linear scale from domain [d0, d1] to range [r0, r1], with .invert. */
  function lin(domain, range) {
    var d0 = domain[0], d1 = domain[1], r0 = range[0], r1 = range[1];
    var f = function (x) { return r0 + (x - d0) / (d1 - d0) * (r1 - r0); };
    f.invert = function (y) { return d0 + (y - r0) / (r1 - r0) * (d1 - d0); };
    f.domain = domain; f.range = range;
    return f;
  }

  /** "Nice" tick values for a linear axis. */
  function ticks(lo, hi, n) {
    n = n || 5;
    var span = hi - lo, raw = span / n;
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 5, 10].map(function (m) { return m * mag; })
      .reduce(function (best, s) { return Math.abs(s - raw) < Math.abs(best - raw) ? s : best; });
    var out = [], t = Math.ceil(lo / step) * step;
    for (; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }

  var fmt = {
    num: function (x, d) { return Number(x).toFixed(d === undefined ? 2 : d); },
    pct: function (x, d) { return (100 * x).toFixed(d === undefined ? 0 : d) + "%"; },
    int: function (x) { return String(Math.round(x)); }
  };

  // ------------------------------------------------------------- controls
  /** A labelled range input with a live value readout. */
  function slider(opts) {
    var f = opts.fmt || function (v) { return String(v); };
    var value = el("span", { class: "widget-control-value", text: f(opts.value) });
    var input = el("input", {
      type: "range", min: opts.min, max: opts.max,
      step: opts.step === undefined ? "any" : opts.step, value: opts.value,
      "aria-label": opts.label
    });
    var root = el("div", { class: "widget-control" },
      el("div", { class: "widget-control-head" }, el("span", { text: opts.label }), value),
      input);
    function get() { return Number(input.value); }
    function set(v, silent) {
      input.value = v; value.textContent = f(Number(input.value));
      if (!silent && opts.oninput) opts.oninput(get());
    }
    input.addEventListener("input", function () { value.textContent = f(get()); if (opts.oninput) opts.oninput(get()); });
    return { root: root, input: input, get: get, set: set };
  }

  /** A row of exclusive buttons: toggle({label, options: [{value, label}], value, onchange}). */
  function toggle(opts) {
    var current = opts.value, buttons = [];
    var group = el("div", { class: "widget-toggle", role: "group", "aria-label": opts.label });
    opts.options.forEach(function (o) {
      var b = el("button", { type: "button", class: "widget-toggle-btn", text: o.label,
        "aria-pressed": String(o.value === current),
        onclick: function () { set(o.value); } });
      b._value = o.value; buttons.push(b); group.appendChild(b);
    });
    function set(v, silent) {
      current = v;
      buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b._value === v)); });
      if (!silent && opts.onchange) opts.onchange(v);
    }
    var root = el("div", { class: "widget-control" },
      opts.label ? el("div", { class: "widget-control-head" }, el("span", { text: opts.label })) : null,
      group);
    return { root: root, get: function () { return current; }, set: set };
  }

  /** A checkbox: check({label, value, onchange}). */
  function check(opts) {
    var input = el("input", { type: "checkbox", checked: !!opts.value });
    input.checked = !!opts.value;
    input.addEventListener("change", function () { if (opts.onchange) opts.onchange(input.checked); });
    var root = el("label", { class: "widget-check" }, input, " ", el("span", { text: opts.label }));
    return { root: root, get: function () { return input.checked; },
      set: function (v) { input.checked = !!v; if (opts.onchange) opts.onchange(!!v); } };
  }

  /** A stat tile: stat({label, value, hint}) with .set(value). */
  function stat(opts) {
    var v = el("div", { class: "widget-stat-value", text: opts.value });
    var root = el("div", { class: "widget-stat" }, v,
      el("div", { class: "widget-stat-label", text: opts.label }),
      opts.hint ? el("div", { class: "widget-note", text: opts.hint }) : null);
    return { root: root, set: function (x) { v.textContent = x; } };
  }

  /** The frame from chrome.scss: frame({title, badge, note}) -> {root, controls, body, stats, note}. */
  function frame(opts) {
    var controls = el("div", { class: "widget-controls" });
    var body = el("div", { class: "widget-body" });
    var stats = el("div", { class: "widget-stats" });
    var note = el("p", { class: "widget-note", text: opts.note || "" });
    var root = el("div", { class: "widget-container" },
      el("div", { class: "widget-header" },
        el("span", { class: "widget-title", text: opts.title }),
        el("span", { class: "widget-badge", text: opts.badge || "interactive" })),
      controls, body, stats, note);
    return { root: root, controls: controls, body: body, stats: stats, note: note };
  }

  // ---------------------------------------------------------------- input
  /** Pointer dragging in SVG user units: drag(target, svgEl, {start, move, end}). */
  function drag(target, svgEl, handlers) {
    function point(ev) {
      var pt = svgEl.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
      var m = svgEl.getScreenCTM();
      return m ? pt.matrixTransform(m.inverse()) : pt;
    }
    target.style.cursor = "grab";
    target.addEventListener("pointerdown", function (ev) {
      ev.preventDefault();
      target.setPointerCapture(ev.pointerId);
      target.style.cursor = "grabbing";
      var p = point(ev);
      if (handlers.start) handlers.start(p.x, p.y, ev);
      function move(e) { var q = point(e); if (handlers.move) handlers.move(q.x, q.y, e); }
      function up(e) {
        target.removeEventListener("pointermove", move);
        target.removeEventListener("pointerup", up);
        target.removeEventListener("pointercancel", up);
        target.style.cursor = "grab";
        if (handlers.end) handlers.end(e);
      }
      target.addEventListener("pointermove", move);
      target.addEventListener("pointerup", up);
      target.addEventListener("pointercancel", up);
    });
  }

  // ----------------------------------------------------------------- data
  /** Precomputed data: a <script type="application/json" id> block, or window.WK_DATA[id]. */
  function data(id) {
    var node = document.getElementById(id);
    if (node && node.type === "application/json") return JSON.parse(node.textContent);
    if (window.WK_DATA && window.WK_DATA[id] !== undefined) return window.WK_DATA[id];
    return null;
  }

  // ---------------------------------------------------------------- mount
  /** mount("widget-x", function (root, WK) {...}): runs once the DOM is ready,
   *  does nothing when the element is absent, and keeps one failing widget
   *  from taking the page's other scripts down with it. */
  function mount(id, build) {
    function go() {
      var root = document.getElementById(id);
      if (!root) return;
      try { build(root, api); }
      catch (err) {
        root.appendChild(el("p", { class: "widget-note", text: "This widget failed to load: " + err.message }));
        if (window.console) console.error("widget " + id, err);
      }
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go);
    else go();
  }

  var api = {
    el: el, h: h, svg: svg, clear: clear,
    palette: palette, onTheme: onTheme, cssVar: cssVar, tokens: TOKENS.slice(),
    lin: lin, ticks: ticks, fmt: fmt,
    slider: slider, toggle: toggle, check: check, stat: stat, frame: frame,
    drag: drag, data: data, mount: mount
  };
  return api;
})();
