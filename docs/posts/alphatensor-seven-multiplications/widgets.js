/* Widgets for "AlphaTensor, Part 2": a factorization of the multiplication
 * tensor, read as an algorithm.
 *
 *   widget-build      add the blocks of a factorization one at a time and watch
 *                     the cube fill in until it equals the tensor
 *   widget-run        type two matrices and run the factorization on them
 *   widget-recursion  what one saved multiplication is worth when the recipe
 *                     is applied to the quarters of a matrix, again and again
 *
 * Numbers come from Part 1's model.js (MatmulModel) and the cube from its
 * cube.js (TensorCube); this file only draws.
 */
"use strict";

(function () {
  var M = MatmulModel;
  var RECIPES = {
    strassen: { name: "Strassen's 7", F: M.STRASSEN },
    school: { name: "Schoolbook's 8", F: M.schoolbook(2) }
  };
  var T = M.tensor(2), CN = M.names("c", 2);

  function button(text, onclick) {
    return WK.el("button", { type: "button", class: "at-chip", text: text, onclick: onclick });
  }
  function list(xs) {
    return xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];
  }
  function int(x) { return Math.round(x).toLocaleString("en-US"); }
  function minus(x) { return x < 0 ? "−" + (-x) : String(x); }

  // ---------------------------------------------------------------- build
  WK.mount("widget-build", function (root) {
    var state = { recipe: "strassen", k: 0 };
    var stage = null, cube = null, flat = TensorCube.flat();

    var f = WK.frame({
      title: "Building the cube out of blocks",
      note: "A block u ∘ v ∘ w fills the cells where an entry of u, an entry of v and an entry of w " +
        "are all nonzero. Purple cells hold +1 and amber cells hold −1. The translucent shell marks the block " +
        "just added, and a hollow cell held something earlier and has since cancelled to zero. The target is " +
        "the multiplication tensor from Part 1: eight cells at +1, the other 56 at zero."
    });
    var recipe = WK.toggle({ label: "Factorization", value: "strassen",
      options: [{ value: "strassen", label: RECIPES.strassen.name }, { value: "school", label: RECIPES.school.name }],
      onchange: function (v) { state.recipe = v; state.k = 0; slider.input.max = RECIPES[v].F.U.length; draw(); } });
    recipe.root.style.flex = "0 1 auto";
    f.controls.appendChild(recipe.root);
    var slider = WK.slider({ label: "Blocks added", min: 0, max: 7, step: 1, value: 0,
      fmt: function (v) { return v + " of " + RECIPES[state.recipe].F.U.length; },
      oninput: function (v) { state.k = v; draw(); } });
    f.controls.appendChild(slider.root);
    var next = button("Add the next block", function () {
      if (state.k < RECIPES[state.recipe].F.U.length) { state.k++; draw(); }
    });
    var reset = button("Start over", function () { state.k = 0; draw(); });
    f.controls.appendChild(WK.el("div", { class: "widget-control at-actions", style: { flex: "0 0 auto", alignSelf: "flex-end", margin: "0" } },
      next, reset));

    var readout = WK.el("p", { class: "at-readout", "aria-live": "polite" });
    f.body.appendChild(readout);
    var stats = {
      k: WK.stat({ label: "blocks added, one multiplication each", value: "" }),
      nnz: WK.stat({ label: "nonzero cells in the sum so far", value: "" }),
      wrong: WK.stat({ label: "cells that differ from the target", value: "" }),
      done: WK.stat({ label: "the sum equals the multiplication tensor", value: "" })
    };
    Object.keys(stats).forEach(function (key) { f.stats.appendChild(stats[key].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "The multiplication cube being assembled from rank-one blocks.",
      view: TensorCube.view(2)
    }, function (s) {
      stage = s;
      cube = TensorCube.create(s, 2);
      draw();
    });
    f.body.appendChild(WK.el("div", { class: "at-flat" }, flat.root));

    function describe(F, k) {
      var text = M.formulas(F, 2), R = F.U.length;
      if (k === 0) return "Nothing added yet: every cell is zero. The first block is " + text.products[0] + ".";
      var w = F.W[k - 1], plus = [], less = [];
      w.forEach(function (x, c) { if (x > 0) plus.push(CN[c]); if (x < 0) less.push(CN[c]); });
      var block = M.nnz(M.rankOne(F.U[k - 1], F.V[k - 1], w));
      return "Block " + k + " of " + R + ": " + text.products[k - 1] + ", one multiplication. Its product is " +
        (plus.length ? "added to " + list(plus) : "") + (plus.length && less.length ? " and " : "") +
        (less.length ? "subtracted from " + list(less) : "") + ". The block covers " + block +
        (block === 1 ? " cell." : " cells.");
    }

    function draw() {
      var F = RECIPES[state.recipe].F, k = state.k, X = M.sumOfTerms(F, k), touched = X.map(function () { return false; });
      for (var r = 1; r <= k; r++) M.sumOfTerms(F, r).forEach(function (x, i) { if (x) touched[i] = true; });
      var wrong = X.filter(function (x, i) { return x !== T[i]; }).length;
      slider.set(k, true);
      stats.k.set(String(k));
      stats.nnz.set(String(M.nnz(X)));
      stats.wrong.set(String(wrong));
      stats.done.set(wrong === 0 ? "yes" : "not yet");
      readout.textContent = describe(F, k);
      next.disabled = k >= F.U.length;
      var st = { n: 2, values: X, touched: touched, slice: null, cell: null,
        preview: k > 0 ? M.rankOne(F.U[k - 1], F.V[k - 1], F.W[k - 1]) : null };
      flat.set(st);
      if (stage) cube.set(st);
    }
    draw();
  });

  // ------------------------------------------------------------------ run
  WK.mount("widget-run", function (root) {
    var EXAMPLE = { A: [1, 2, 3, 4], B: [5, 6, 7, 8] };
    var state = { recipe: "strassen", A: EXAMPLE.A.slice(), B: EXAMPLE.B.slice() };
    var rand = M.mulberry32(20261002);
    var inputs = { A: [], B: [] }, outs = [];

    var f = WK.frame({
      title: "Run the recipe",
      note: "Type any integers into A and B. The first rows of the table are the blocks of the factorization: a sum " +
        "of A's entries, a sum of B's entries, and their product. The last four rows assemble C from the " +
        "products with additions and subtractions only."
    });
    var recipe = WK.toggle({ label: "Recipe", value: "strassen",
      options: [{ value: "strassen", label: RECIPES.strassen.name }, { value: "school", label: RECIPES.school.name }],
      onchange: function (v) { state.recipe = v; draw(); } });
    recipe.root.style.flex = "0 1 auto";
    f.controls.appendChild(recipe.root);
    f.controls.appendChild(WK.el("div", { class: "widget-control at-actions", style: { flex: "0 0 auto", alignSelf: "flex-end", margin: "0" } },
      button("Random matrices", function () {
        state.A = M.randomMatrix(rand, 4, -9, 9); state.B = M.randomMatrix(rand, 4, -9, 9); fill(); draw();
      }),
      button("Back to the example", function () { state.A = EXAMPLE.A.slice(); state.B = EXAMPLE.B.slice(); fill(); draw(); })));

    function matrix(key) {
      var grid = WK.el("div", { class: "at-matrix", role: "group", "aria-label": "Matrix " + key });
      for (var i = 0; i < 4; i++) (function (i) {
        var input = WK.el("input", { class: "at-num", type: "number", step: 1, min: -99, max: 99, inputmode: "numeric",
          "aria-label": M.name(key.toLowerCase(), 2, i),
          oninput: function () {
            var v = Math.round(Number(input.value));
            state[key][i] = isFinite(v) ? Math.max(-99, Math.min(99, v)) : 0;
            draw();
          } });
        inputs[key].push(input);
        grid.appendChild(input);
      })(i);
      return grid;
    }
    var cGrid = WK.el("div", { class: "at-matrix", role: "group", "aria-label": "The product C" });
    for (var i = 0; i < 4; i++) { outs.push(WK.el("div", { class: "at-out" })); cGrid.appendChild(outs[i]); }
    f.body.appendChild(WK.el("div", { class: "at-matrices" },
      WK.el("span", { class: "at-matrix-name", text: "A =" }), matrix("A"),
      WK.el("span", { class: "at-matrix-name", text: "B =" }), matrix("B"),
      WK.el("span", { class: "at-matrix-name", text: "AB =" }), cGrid));
    var table = WK.el("table", { class: "at-table" });
    f.body.appendChild(WK.el("div", { class: "at-scroll" }, table));

    var stats = {
      mults: WK.stat({ label: "multiplications this recipe used", value: "" }),
      adds: WK.stat({ label: "additions and subtractions it used", value: "" }),
      other: WK.stat({ label: "multiplications the other recipe needs", value: "" }),
      agree: WK.stat({ label: "matches the schoolbook product", value: "" })
    };
    Object.keys(stats).forEach(function (key) { f.stats.appendChild(stats[key].root); });
    root.appendChild(f.root);

    function fill() {
      ["A", "B"].forEach(function (key) { inputs[key].forEach(function (input, i) { input.value = state[key][i]; }); });
    }
    function row(cells, head) {
      return WK.el("tr", {}, cells.map(function (c, j) {
        return WK.el(head ? "th" : "td", { class: j ? "at-right" : null, text: c, scope: head ? "col" : null });
      }));
    }
    function draw() {
      var F = RECIPES[state.recipe].F, other = RECIPES[state.recipe === "strassen" ? "school" : "strassen"].F;
      var run = M.applyAlgorithm(F, state.A, state.B), text = M.formulas(F, 2), truth = M.matmul(2, state.A, state.B);
      WK.clear(table);
      table.appendChild(WK.el("thead", {}, row(["step", "sum from A", "sum from B", "value"], true)));
      var body = WK.el("tbody");
      run.m.forEach(function (m, r) {
        body.appendChild(row([text.products[r], minus(run.left[r]), minus(run.right[r]), minus(m)]));
      });
      text.outputs.forEach(function (s, c) { body.appendChild(row([s, "", "", minus(run.C[c])])); });
      table.appendChild(body);
      run.C.forEach(function (x, c) { outs[c].textContent = minus(x); });
      stats.mults.set(String(F.U.length));
      stats.adds.set(String(M.additions(F)));
      stats.other.set(String(other.U.length));
      stats.agree.set(M.equal(run.C, truth) ? "yes" : "no");
    }
    fill();
    draw();
  });

  // ------------------------------------------------------------ recursion
  WK.mount("widget-recursion", function (root) {
    var KMAX = 12, state = { k: 3 };
    var W = 600, H = 300, x = WK.lin([0, KMAX], [58, 560]), y = WK.lin([0, 11.5], [262, 18]);

    var f = WK.frame({
      title: "One multiplication saved, at every level",
      note: "Multiplying two n × n matrices with n = 2ᵏ by cutting each into quarters, then cutting the " +
        "quarters, k times over. The schoolbook rule spends 8 products of quarters per level, Strassen's 7. " +
        "The vertical axis is logarithmic: each gridline is ten times the one under it."
    });
    var slider = WK.slider({ label: "Matrix size", min: 1, max: KMAX, step: 1, value: state.k,
      fmt: function (k) { var n = Math.pow(2, k); return int(n) + " × " + int(n) + "  (k = " + k + ")"; },
      oninput: function (k) { state.k = k; draw(); } });
    f.controls.appendChild(slider.root);
    var svg = WK.svg(W, H, { "aria-label": "Multiplications against matrix size for the schoolbook rule and for Strassen's." });
    f.body.appendChild(svg);
    var stats = {
      school: WK.stat({ label: "multiplications, schoolbook (8ᵏ = n³)", value: "" }),
      strassen: WK.stat({ label: "multiplications, Strassen (7ᵏ)", value: "" }),
      saved: WK.stat({ label: "of the multiplications saved", value: "" }),
      ratio: WK.stat({ label: "schoolbook multiplications per Strassen one", value: "" })
    };
    Object.keys(stats).forEach(function (key) { f.stats.appendChild(stats[key].root); });
    root.appendChild(f.root);

    function log10(v) { return Math.log(v) / Math.LN10; }
    function path(key) {
      var d = "";
      for (var k = 0; k <= KMAX; k++) d += (k ? "L" : "M") + x(k).toFixed(1) + " " + y(log10(M.recursion(k)[key])).toFixed(1);
      return d;
    }
    function draw() {
      var r = M.recursion(state.k), e, k;
      WK.clear(svg);
      for (e = 0; e <= 10; e += 2) {
        svg.appendChild(WK.h("line", { x1: x(0), x2: x(KMAX), y1: y(e), y2: y(e), stroke: "rule" }));
        svg.appendChild(WK.h("text", { x: x(0) - 8, y: y(e) + 4, "text-anchor": "end", "font-size": 11, fill: "muted",
          text: e === 0 ? "1" : "10" + String(e).split("").map(function (d) { return "⁰¹²³⁴⁵⁶⁷⁸⁹".charAt(+d); }).join("") }));
      }
      for (k = 0; k <= KMAX; k += 2)
        svg.appendChild(WK.h("text", { x: x(k), y: y(0) + 18, "text-anchor": "middle", "font-size": 11, fill: "muted",
          text: int(Math.pow(2, k)) }));
      svg.appendChild(WK.h("text", { x: x(KMAX / 2), y: H - 4, "text-anchor": "middle", "font-size": 12, fill: "muted",
        text: "matrix size n" }));
      svg.appendChild(WK.h("line", { x1: x(state.k), x2: x(state.k), y1: y(0), y2: y(11.5), stroke: "muted", "stroke-dasharray": "4 3" }));
      svg.appendChild(WK.h("path", { d: path("school"), fill: "none", stroke: "c3", "stroke-width": 2.5 }));
      svg.appendChild(WK.h("path", { d: path("strassen"), fill: "none", stroke: "c1", "stroke-width": 2.5 }));
      svg.appendChild(WK.h("circle", { cx: x(state.k), cy: y(log10(r.school)), r: 5, fill: "c3" }));
      svg.appendChild(WK.h("circle", { cx: x(state.k), cy: y(log10(r.strassen)), r: 5, fill: "c1" }));
      svg.appendChild(WK.h("text", { x: x(KMAX) - 4, y: y(log10(M.recursion(KMAX).school)) - 9, "text-anchor": "end",
        "font-size": 12, "font-weight": 700, fill: "c3", text: "schoolbook, n³" }));
      svg.appendChild(WK.h("text", { x: x(KMAX) - 4, y: y(log10(M.recursion(KMAX).strassen)) + 20, "text-anchor": "end",
        "font-size": 12, "font-weight": 700, fill: "c1", text: "Strassen, n^" + M.OMEGA.toFixed(2) }));
      stats.school.set(int(r.school));
      stats.strassen.set(int(r.strassen));
      stats.saved.set((100 * r.saving).toFixed(1) + "%");
      stats.ratio.set((r.school / r.strassen).toFixed(2));
    }
    draw();
  });
})();
