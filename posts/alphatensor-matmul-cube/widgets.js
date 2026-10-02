/* Widget for "AlphaTensor, Part 1": the matrix multiplication tensor as a
 * cube, and the schoolbook rule read off it one cell at a time.
 *
 *   widget-cube   pick an output entry and its tray lifts out; pick a product
 *                 and the cell that records it lights up
 *
 * Numbers come from model.js (MatmulModel) and the picture from cube.js
 * (TensorCube). Selection lives in the buttons, not the scene, so the widget
 * reads the same by keyboard, on a phone, and when three.js does not load.
 */
"use strict";

(function () {
  var M = MatmulModel;
  // The series' worked example: A = [1 2; 3 4], B = [5 6; 7 8].
  var A = [1, 2, 3, 4], B = [5, 6, 7, 8];

  function chip(text, label, onclick) {
    return WK.el("button", { type: "button", class: "at-chip", text: text, "aria-pressed": "false",
      "aria-label": label, onclick: onclick });
  }

  WK.mount("widget-cube", function (root) {
    var state = { n: 2, slice: null, cell: null };
    var stage = null, cube = null, flat = TensorCube.flat();
    var chips = [];  // {node, slice, cell}

    var f = WK.frame({
      title: "The multiplication rule, stacked into a cube",
      note: "Each tray is one entry of C = AB. Inside a tray, rows are the entries of A and columns the " +
        "entries of B; a filled cell means that row's entry times that column's entry is one of the " +
        "products that entry of C adds up. The grids under the scene are the same trays seen from above. " +
        "The example numbers use A = [1 2; 3 4] and B = [5 6; 7 8]."
    });
    var size = WK.toggle({ label: "Matrices", value: 2,
      options: [{ value: 2, label: "2 × 2" }, { value: 3, label: "3 × 3" }],
      onchange: function (n) { state = { n: n, slice: null, cell: null }; formulas(); draw(); } });
    size.root.style.flex = "0 1 auto";
    f.controls.appendChild(size.root);
    var all = WK.el("button", { type: "button", class: "widget-toggle-btn", text: "Show the whole cube",
      onclick: function () { state.slice = null; state.cell = null; draw(); } });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } },
      WK.el("div", { class: "widget-toggle" }, all)));

    var panel = WK.el("div", { class: "at-formulas", role: "group",
      "aria-label": "The schoolbook rule. Choose an entry of C or one of its products." });
    var readout = WK.el("p", { class: "at-readout", "aria-live": "polite" });
    f.body.appendChild(panel);
    f.body.appendChild(readout);

    var stats = {
      cells: WK.stat({ label: "cells in the cube", value: "" }),
      ones: WK.stat({ label: "cells holding a 1", value: "" }),
      mults: WK.stat({ label: "multiplications, reading one cell at a time", value: "" }),
      fill: WK.stat({ label: "of the cube is nonzero", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "A stack of trays, one per entry of the product matrix, with a cube in every cell that holds a 1.",
      view: TensorCube.view(2)
    }, function (s) {
      stage = s;
      cube = TensorCube.create(s, 3);
      draw();
    });
    f.body.appendChild(WK.el("div", { class: "at-flat" }, flat.root));

    function formulas() {
      var n = state.n, N = n * n, terms = M.terms(n);
      WK.clear(panel);
      chips = [];
      panel.setAttribute("data-n", n);
      for (var c = 0; c < N; c++) (function (c) {
        var row = WK.el("div", { class: "at-eq" });
        var head = chip(M.name("c", n, c), "The slice for " + M.name("c", n, c), function () {
          state.cell = null;
          state.slice = state.slice === c ? null : c;
          draw();
        });
        chips.push({ node: head, slice: c, cell: null });
        row.appendChild(head);
        row.appendChild(WK.el("span", { class: "at-op", text: "=" }));
        terms.filter(function (t) { return t.c === c; }).forEach(function (t, j) {
          if (j) row.appendChild(WK.el("span", { class: "at-op", text: "+" }));
          var text = M.name("a", n, t.a) + " · " + M.name("b", n, t.b);
          var b = chip(text, text + ", a product in " + M.name("c", n, c), function () {
            var same = state.cell && state.cell.a === t.a && state.cell.b === t.b;
            state.cell = same ? null : t;
            state.slice = c;
            draw();
          });
          chips.push({ node: b, slice: c, cell: t });
          row.appendChild(b);
        });
        panel.appendChild(row);
      })(c);
    }

    function describe() {
      var n = state.n, t = state.cell, c = state.slice, sum;
      if (t) {
        return "The 1 at (" + M.name("a", n, t.a) + ", " + M.name("b", n, t.b) + ", " + M.name("c", n, t.c) +
          ") says: multiply " + M.name("a", n, t.a) + " by " + M.name("b", n, t.b) + " and add the product to " +
          M.name("c", n, t.c) + "." + (n === 2 ? " In the example, " + A[t.a] + " · " + B[t.b] + " = " + A[t.a] * B[t.b] + "." : "");
      }
      if (c !== null) {
        var mine = M.terms(n).filter(function (x) { return x.c === c; });
        var text = "Slice " + M.name("c", n, c) + " holds " + mine.length + " ones, so " + M.name("c", n, c) + " = " +
          mine.map(function (x) { return M.name("a", n, x.a) + "·" + M.name("b", n, x.b); }).join(" + ") + ".";
        if (n === 2) {
          sum = mine.reduce(function (s, x) { return s + A[x.a] * B[x.b]; }, 0);
          text += " In the example, " + mine.map(function (x) { return A[x.a] + "·" + B[x.b]; }).join(" + ") + " = " + sum + ".";
        }
        return text;
      }
      return "Choose an entry of C to lift its tray, or a product to light its cell.";
    }

    function draw() {
      var n = state.n, N = n * n, T = M.tensor(n);
      var st = { n: n, values: T, touched: null, preview: null, slice: state.slice, cell: state.cell };
      stats.cells.set(N + " × " + N + " × " + N + " = " + (N * N * N));
      stats.ones.set(String(M.nnz(T)));
      stats.mults.set(String(M.nnz(T)));
      stats.fill.set((100 * M.nnz(T) / (N * N * N)).toFixed(1) + "%");
      chips.forEach(function (ch) {
        var on = ch.cell ? (state.cell && state.cell.a === ch.cell.a && state.cell.b === ch.cell.b)
                         : (state.slice === ch.slice && !state.cell);
        ch.node.setAttribute("aria-pressed", String(!!on));
      });
      readout.textContent = describe();
      flat.set(st);
      if (!stage) return;
      cube.frame(n);
      cube.set(st);
    }

    formulas();
    draw();
  });
})();
