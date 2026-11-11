/* Stein's identity tester. All numbers come from SteinLemmaModel (model.js);
 * this file only draws them. Colours are kit tokens, so the theme toggle repaints. */
"use strict";

(function () {
  var M = SteinLemmaModel;
  var N = 100000;

  WK.mount("widget-stein-identity", function (root, WK) {
    var st = { dist: "gauss", nu: 10, fn: "sin", op: "gauss", seed: 1, custom: "x*exp(-x^2/4)" };

    var f = WK.frame({
      title: "Stein's identity by Monte Carlo",
      badge: "interactive · synthetic draws",
      note: "Each run draws 100,000 pseudo-random values from the chosen law, standardised to mean 0 and variance 1. " +
        "The line is the running mean of the operator applied to f, the band is ±2 standard errors, and the dashed line is the exact expectation. " +
        "A custom f accepts x, numbers, + − * / ^, pi, e and sin cos tan exp log sqrt abs tanh sinh cosh atan; its derivative is a central difference, and whether its expectation exists is judged from its growth in the tails."
    });

    var distT = WK.toggle({ label: "Law of X (mean 0, variance 1)", value: st.dist,
      options: [{ value: "gauss", label: "Gaussian" }, { value: "laplace", label: "Laplace" },
                { value: "t", label: "Student-t" }, { value: "unif", label: "Uniform" }],
      onchange: function (v) { st.dist = v; syncNu(); schedule(); } });
    var nuS = WK.slider({ label: "Student-t degrees of freedom ν", min: 2.5, max: 30, step: 0.5, value: st.nu,
      fmt: function (v) { return Number(v).toFixed(1); }, oninput: function (v) { st.nu = v; schedule(); } });
    var fnT = WK.toggle({ label: "Test function f", value: st.fn,
      options: [{ value: "sin", label: "sin x" }, { value: "tanh", label: "tanh x" },
                { value: "cube", label: "x³" }, { value: "custom", label: "custom" }],
      onchange: function (v) { st.fn = v; syncCustom(); schedule(); } });
    var opT = WK.toggle({ label: "Operator", value: st.op,
      options: [{ value: "gauss", label: "Gaussian  f′ − x f" }, { value: "langevin", label: "Langevin  f′ + sₚ f" }],
      onchange: function (v) { st.op = v; schedule(); } });
    var customIn = WK.el("input", { type: "text", value: st.custom, "aria-label": "Custom test function f of x",
      spellcheck: "false", autocomplete: "off",
      style: { width: "100%", font: "inherit", fontSize: "0.9rem", padding: "0.3rem 0.5rem",
               border: "1px solid var(--w-rule)", borderRadius: "6px",
               background: "var(--w-paper)", color: "var(--w-ink)" } });
    var customErr = WK.el("div", { class: "widget-note", style: { marginTop: "0.3rem" } });
    var customBox = WK.el("div", { class: "widget-control" },
      WK.el("div", { class: "widget-control-head" }, WK.el("span", { text: "Custom f(x)" })), customIn, customErr);
    var timer = null;
    customIn.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { st.custom = customIn.value; update(); }, 350);
    });
    var resample = WK.el("button", { type: "button", class: "widget-toggle-btn", text: "Resample",
      style: { border: "1px solid var(--w-rule)", borderRadius: "999px" },
      onclick: function () { st.seed += 1; schedule(); } });
    var resampleBox = WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } }, resample);

    [distT, fnT, opT].forEach(function (t) { // let a toggle wrap inside a narrow (phone-width) frame
      var g = t.root.querySelector(".widget-toggle");
      g.style.flexWrap = "wrap"; g.style.borderRadius = "12px";
    });
    [distT, nuS, fnT].forEach(function (c) { f.controls.appendChild(c.root); });
    f.controls.appendChild(customBox);
    f.controls.appendChild(opT.root);
    f.controls.appendChild(resampleBox);

    function syncNu() {
      nuS.input.disabled = st.dist !== "t";
      nuS.root.style.opacity = st.dist === "t" ? "1" : "0.45";
    }
    function syncCustom() { customBox.style.display = st.fn === "custom" ? "" : "none"; }

    var W = 600, H = 300, ml = 62, mr = 18, mt = 16, mb = 46, FS = 14;
    var svg = WK.svg(W, H, { "aria-label": "Running Monte Carlo mean against sample size, with a two standard error band and the exact expectation." });
    f.body.appendChild(svg);

    var sEst = WK.stat({ label: "Estimate at n = 100,000", value: "" });
    var sSe = WK.stat({ label: "± 2 standard errors", value: "" });
    var sExact = WK.stat({ label: "Exact expectation", value: "" });
    [sEst, sSe, sExact].forEach(function (s) { f.stats.appendChild(s.root); });
    var condNote = WK.el("p", { class: "widget-note", style: { marginTop: "0.2rem", color: "var(--w-ink)" } });
    f.stats.parentNode.insertBefore(condNote, f.stats.nextSibling);
    root.appendChild(f.root);

    function label(x, y, text, anchor, cls) {
      return WK.h("text", { x: x, y: y, "text-anchor": anchor || "middle", fill: cls || "muted",
        "font-size": FS }, text);
    }

    // One redraw per animation frame however fast the slider fires; the exact value and the
    // conditions do not depend on the seed, so Resample reuses them.
    var pending = false, cache = {};
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; update(); });
    }
    function exactAndConditions() {
      var key = [st.op, st.dist, st.dist === "t" ? st.nu : "", st.fn, st.fn === "custom" ? st.custom : ""].join("|");
      if (!(key in cache)) cache[key] = { exact: M.exact(st.op, st.dist, st.nu, st.fn, st.custom),
        cond: M.conditions(st.op, st.dist, st.nu, st.fn, st.custom) };
      return cache[key];
    }

    function update() {
      var result, exact, cond;
      customErr.textContent = "";
      try {
        if (st.fn === "custom") M.parse(st.custom);
        result = M.run({ dist: st.dist, nu: st.nu, fn: st.fn, op: st.op, seed: st.seed, N: N, custom: st.custom });
        var ec = exactAndConditions();
        exact = ec.exact; cond = ec.cond;
      } catch (err) {
        customErr.textContent = "Cannot read this f: " + err.message + ".";
        WK.clear(svg);
        [sEst, sSe, sExact].forEach(function (s) { s.set("—"); });
        condNote.textContent = "";
        return;
      }
      draw(result, exact, cond);
    }

    function draw(result, exact, cond) {
      WK.clear(svg);
      if (!cond.domain || result.nonFinite) {
        // f is undefined at some draws: a running mean over the rest would look like an answer
        [sEst, sSe].forEach(function (s) { s.set("—"); });
        sExact.set("undefined");
        condNote.textContent = cond.domain
          ? "f or its derivative is not finite at " + result.nonFinite.toLocaleString("en") + " of the " + N.toLocaleString("en") + " draws, so the expectation is not defined."
          : cond.note;
        return;
      }
      var path = result.path, fin = result.final;
      var lo = Infinity, hi = -Infinity;
      path.forEach(function (p) {
        if (p.n < 300) return;
        var w = cond.variance ? 2 * p.se : 0;
        lo = Math.min(lo, p.mean - w); hi = Math.max(hi, p.mean + w);
      });
      if (exact !== null && isFinite(exact)) { lo = Math.min(lo, exact); hi = Math.max(hi, exact); }
      lo = Math.min(lo, 0); hi = Math.max(hi, 0);
      var pad = Math.max(0.08, 0.15 * (hi - lo));
      lo -= pad; hi += pad;

      var x = WK.lin([1, Math.log10(N)], [ml, W - mr]);
      var y = WK.lin([lo, hi], [H - mb, mt]);
      var clipId = "stein-id-clip";
      svg.appendChild(WK.h("defs", {}, WK.h("clipPath", { id: clipId },
        WK.h("rect", { x: ml, y: mt, width: W - ml - mr, height: H - mt - mb }))));

      WK.ticks(lo, hi, 5).forEach(function (t) {
        svg.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(t), y2: y(t), stroke: "rule", "stroke-width": 1 }));
        svg.appendChild(label(ml - 8, y(t) + 4, WK.fmt.num(t, Math.abs(hi - lo) < 1 ? 2 : 1), "end"));
      });
      ["10", "100", "1k", "10k", "100k"].forEach(function (s, i) {
        svg.appendChild(label(x(i + 1), H - mb + 18, s));
      });
      svg.appendChild(label((ml + W - mr) / 2, H - 6, "sample size n (log scale)"));
      svg.appendChild(WK.h("line", { x1: ml, x2: ml, y1: mt, y2: H - mb, stroke: "muted", "stroke-width": 1 }));
      svg.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: H - mb, y2: H - mb, stroke: "muted", "stroke-width": 1 }));

      var g = WK.h("g", { "clip-path": "url(#" + clipId + ")" });
      svg.appendChild(g);
      if (cond.variance) {
        var up = path.map(function (p) { return x(Math.log10(p.n)) + "," + y(p.mean + 2 * p.se); });
        var dn = path.slice().reverse().map(function (p) { return x(Math.log10(p.n)) + "," + y(p.mean - 2 * p.se); });
        g.appendChild(WK.h("polygon", { points: up.concat(dn).join(" "), fill: "c1", "fill-opacity": 0.18 }));
      }
      if (exact !== 0) g.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(0), y2: y(0), stroke: "muted", "stroke-width": 1 }));
      if (exact !== null && isFinite(exact)) {
        g.appendChild(WK.h("line", { x1: ml, x2: W - mr, y1: y(exact), y2: y(exact), stroke: "c3",
          "stroke-width": 2, "stroke-dasharray": "6 4" }));
      }
      g.appendChild(WK.h("polyline", { fill: "none", stroke: "c1", "stroke-width": 2,
        points: path.map(function (p) { return x(Math.log10(p.n)) + "," + y(p.mean); }).join(" ") }));
      svg.appendChild(label(W - mr - 4, mt + 14, "running mean of " + (st.op === "gauss" ? "f′(X) − X f(X)" : "f′(X) + sₚ(X) f(X)"), "end", "ink"));

      sEst.set(WK.fmt.num(Math.abs(fin.mean) < 5e-4 ? 0 : fin.mean, 3));
      sSe.set(cond.variance ? "± " + WK.fmt.num(2 * fin.se, 3) : "not valid");
      sExact.set(exact === null ? "undefined" : WK.fmt.num(Math.abs(exact) < 5e-4 ? 0 : exact, 3));
      var msg = cond.note;
      if (!msg && exact !== null) msg = Math.abs(exact) < 5e-4
        ? "The identity holds for this law and operator: the expectation is 0."
        : "The identity fails for this law and operator: the expectation is " + WK.fmt.num(exact, 3) + ", not 0.";
      condNote.textContent = msg;
    }

    syncNu(); syncCustom(); update();
  });
})();
