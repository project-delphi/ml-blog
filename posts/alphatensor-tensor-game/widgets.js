/* Widgets for "AlphaTensor, Part 3": the search for a factorization, played
 * as a game.
 *
 *   widget-play   TensorGame on the 2 x 2 cube: choose u, v, w, subtract the
 *                 block, and try to reach the zero tensor in few moves
 *   widget-size   how many moves there are at one turn, and how many games
 *
 * Numbers come from Part 1's model.js (MatmulModel) and the cube from its
 * cube.js (TensorCube); this file only draws.
 */
"use strict";

(function () {
  var M = MatmulModel;
  var N = 4, T = M.tensor(2), SCHOOL = M.schoolbook(2);
  var SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";

  function button(text, onclick) {
    return WK.el("button", { type: "button", class: "at-chip", text: text, onclick: onclick });
  }
  function sup(k) { return String(k).split("").map(function (d) { return d === "-" ? "⁻" : SUP.charAt(+d); }).join(""); }
  function list(xs) {
    return xs.length < 2 ? xs.join("") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];
  }

  // ----------------------------------------------------------------- play
  WK.mount("widget-play", function (root) {
    var state = { R: T.slice(), history: [], u: [0, 0, 0, 0], v: [0, 0, 0, 0], w: [0, 0, 0, 0], note: "" };
    var stage = null, cube = null, flat = TensorCube.flat(), cells = { u: [], v: [], w: [] };

    var f = WK.frame({
      title: "TensorGame on the 2 × 2 cube",
      note: "The game starts from the multiplication tensor. A move is three vectors u, v, w with entries " +
        "−1, 0 or +1; it subtracts the block u ∘ v ∘ w from what is left. Reach the zero tensor and the moves " +
        "played are an algorithm, one multiplication per move. Purple cells hold a positive value and amber " +
        "cells a negative one; a hollow cell has been cleared. The translucent shell is the block the move " +
        "being set up would subtract. The greedy button plays the move that leaves the fewest nonzero cells."
    });

    function vector(key, title, letter) {
      var grid = WK.el("div", { class: "at-vector-grid", role: "group", "aria-label": title });
      for (var i = 0; i < 4; i++) (function (i) {
        var name = M.name(letter, 2, i);
        var value = WK.el("b"), b = WK.el("button", { type: "button", class: "at-cellbtn" }, value, WK.el("small", { text: name }));
        function step(d) {
          var order = [0, 1, -1], at = order.indexOf(state[key][i]);
          state[key][i] = order[(at + d + 3) % 3];
          state.note = "";
          draw();
        }
        b.addEventListener("click", function () { step(1); });
        b.addEventListener("keydown", function (ev) {
          if (ev.key === "ArrowUp") { ev.preventDefault(); step(1); }
          if (ev.key === "ArrowDown") { ev.preventDefault(); step(-1); }
        });
        cells[key].push({ node: b, value: value, name: name });
        grid.appendChild(b);
      })(i);
      return WK.el("div", { class: "at-vector" }, WK.el("span", { class: "at-vector-name", text: title }), grid);
    }
    f.body.appendChild(WK.el("div", { class: "at-picker" },
      vector("u", "u: which entries of A", "a"), vector("v", "v: which entries of B", "b"),
      vector("w", "w: which entries of C", "c")));

    var subtract = button("Subtract this block", function () { play(state.u, state.v, state.w, ""); });
    var undo = button("Undo", function () {
      var last = state.history.pop();
      if (!last) return;
      state.R = M.add(state.R, M.rankOne(last.u, last.v, last.w));
      state.note = "Took the last move back.";
      draw();
    });
    var restart = button("Start over", function () {
      state.R = T.slice(); state.history = []; state.u = [0, 0, 0, 0]; state.v = [0, 0, 0, 0]; state.w = [0, 0, 0, 0];
      state.note = "";
      draw();
    });
    f.body.appendChild(WK.el("div", { class: "at-actions" }, subtract, undo, restart));
    f.body.appendChild(WK.el("div", { class: "at-actions" },
      button("Strassen's next move", function () { scripted(M.STRASSEN, "Strassen's"); }),
      button("Schoolbook's next move", function () { scripted(SCHOOL, "The schoolbook rule's"); }),
      button("Greedy move", function () {
        var best = M.bestMove(state.R, N);
        if (M.isZero(state.R)) { state.note = "Nothing left to subtract."; draw(); return; }
        if (!M.better(best, M.score(state.R))) { state.note = "No single move leaves fewer nonzero cells. Greedy is stuck."; draw(); return; }
        play(best.u, best.v, best.w, "Greedy's move.");
      })));
    var readout = WK.el("p", { class: "at-readout", "aria-live": "polite" });
    f.body.appendChild(readout);

    var stats = {
      moves: WK.stat({ label: "moves played, one multiplication each", value: "" }),
      left: WK.stat({ label: "nonzero cells left", value: "" }),
      after: WK.stat({ label: "nonzero cells if this block is subtracted", value: "" }),
      best: WK.stat({ label: "fewest moves that can win", value: "7" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "What is left of the multiplication cube after the moves played so far.",
      view: TensorCube.view(2)
    }, function (s) {
      stage = s;
      cube = TensorCube.create(s, 2);
      draw();
    });
    f.body.appendChild(WK.el("div", { class: "at-flat" }, flat.root));

    function ready() {
      return ["u", "v", "w"].every(function (k) { return state[k].some(function (x) { return x !== 0; }); });
    }
    function play(u, v, w, note) {
      var block = M.rankOne(u, v, w);
      if (M.isZero(block)) return;
      u = u.slice(); v = v.slice(); w = w.slice();
      var before = M.nnz(state.R), text = M.formulas({ U: [u], V: [v], W: [w] }, 2), into = [];
      w.forEach(function (x, c) { if (x) into.push((x < 0 ? "−" : "+") + M.name("c", 2, c)); });
      state.R = M.subtract(state.R, block);
      state.history.push({ u: u.slice(), v: v.slice(), w: w.slice() });
      // Empty the picker, so the shell never shows a block already subtracted.
      state.u = [0, 0, 0, 0]; state.v = [0, 0, 0, 0]; state.w = [0, 0, 0, 0];
      state.note = (note ? note + " " : "") + "It multiplied (" + text.left[0] + ") by (" + text.right[0] +
        ") and sent the product to " + list(into) + ": nonzero cells went from " + before + " to " + M.nnz(state.R) + ".";
      draw();
    }
    // The first term of a factorization that has not been played yet.
    // A move counts as played by its net effect: one that was later taken back
    // (its negative subtracted, as greedy does) is due again.
    function scripted(F, whose) {
      if (M.isZero(state.R)) { state.note = "Nothing left to subtract."; draw(); return; }
      var played = state.history.map(function (h) { return M.rankOne(h.u, h.v, h.w); });
      for (var r = 0; r < F.U.length; r++) {
        var block = M.rankOne(F.U[r], F.V[r], F.W[r]);
        var back = block.map(function (x) { return -x; }), net = 0;
        played.forEach(function (p) { if (M.equal(p, block)) net++; else if (M.equal(p, back)) net--; });
        if (net <= 0) {
          play(F.U[r], F.V[r], F.W[r], whose + " move " + (r + 1) + " of " + F.U.length + ".");
          return;
        }
      }
      state.note = whose + " moves have all been played.";
      draw();
    }
    function describe(before, after) {
      if (M.isZero(state.R)) {
        var k = state.history.length;
        return "Zero tensor reached in " + k + " moves: an algorithm with " + k + " multiplications." +
          (k === 7 ? " That is the fewest possible for 2 × 2 matrices." : k === 8 ? " That matches the schoolbook rule; 7 is possible." : "");
      }
      if (!ready()) return state.note || "Set at least one entry in each of u, v and w to see the block a move would subtract.";
      var text = M.formulas({ U: [state.u], V: [state.v], W: [state.w] }, 2);
      var into = [];
      state.w.forEach(function (x, c) { if (x) into.push((x < 0 ? "−" : "+") + M.name("c", 2, c)); });
      return "The block shown multiplies (" + text.left[0] + ") by (" + text.right[0] +
        ") and sends the product to " + list(into) + ". Subtracting it takes the nonzero cells from " + before + " to " + after + ".";
    }

    function draw() {
      var before = M.nnz(state.R), block = ready() ? M.rankOne(state.u, state.v, state.w) : null;
      var after = block ? M.nnz(M.subtract(state.R, block)) : null;
      // Cells that have held a nonzero value at any point of this game.
      var touched = T.map(function (x) { return x !== 0; }), X = T.slice();
      state.history.forEach(function (h) {
        X = M.subtract(X, M.rankOne(h.u, h.v, h.w));
        X.forEach(function (x, i) { if (x) touched[i] = true; });
      });
      ["u", "v", "w"].forEach(function (k) {
        cells[k].forEach(function (c, i) {
          var x = state[k][i];
          c.node.setAttribute("data-v", x);
          c.value.textContent = x > 0 ? "+1" : x < 0 ? "−1" : "0";
          c.node.setAttribute("aria-label", k + " at " + c.name + ": " + c.value.textContent + ". Press to change.");
        });
      });
      subtract.disabled = !block || M.isZero(state.R);
      undo.disabled = !state.history.length;
      stats.moves.set(String(state.history.length));
      stats.left.set(String(before));
      stats.after.set(after === null ? "–" : after + (after === before ? "" : " (" + (after > before ? "+" : "−") + Math.abs(after - before) + ")"));
      readout.textContent = describe(before, after);
      var st = { n: 2, values: state.R, touched: touched, preview: M.isZero(state.R) ? null : block, slice: null, cell: null };
      flat.set(st);
      if (stage) cube.set(st);
    }
    draw();
  });

  // ----------------------------------------------------------------- size
  WK.mount("widget-size", function (root) {
    var state = { n: 2, f: 5, depth: 7 };
    var YEAR = 31557600;  // seconds in a Julian year

    var f = WK.frame({
      title: "How many moves, how many games",
      note: "A move is three vectors of length n², each entry one of the allowed coefficients, so one turn " +
        "has (coefficients)^(3n²) choices, counting the few that subtract nothing. A game of d moves is a " +
        "sequence of d such choices. The last tile assumes a machine that writes down a billion games a second."
    });
    var size = WK.toggle({ label: "Matrices", value: 2,
      options: [2, 3, 4, 5].map(function (n) { return { value: n, label: n + " × " + n }; }),
      onchange: function (n) { state.n = n; draw(); } });
    size.root.style.flex = "0 1 auto";
    var coeff = WK.toggle({ label: "Coefficients allowed", value: 5,
      options: [{ value: 3, label: "−1, 0, 1" }, { value: 5, label: "−2, −1, 0, 1, 2" }],
      onchange: function (v) { state.f = v; draw(); } });
    coeff.root.style.flex = "0 1 auto";
    var depth = WK.slider({ label: "Moves in the game", min: 1, max: 125, step: 1, value: state.depth,
      fmt: function (d) { return d + (d === 1 ? " move" : " moves"); },
      oninput: function (d) { state.depth = d; draw(); } });
    [size.root, coeff.root, depth.root].forEach(function (node) { f.controls.appendChild(node); });
    var exact = WK.el("p", { class: "at-readout", "aria-live": "polite" });
    f.body.appendChild(exact);
    var stats = {
      one: WK.stat({ label: "choices at one move", value: "" }),
      go: WK.stat({ label: "choices at one move of Go, at most", value: "361" }),
      games: WK.stat({ label: "games of this length", value: "" }),
      years: WK.stat({ label: "to write them all down, at a billion a second", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    /** "3.6 × 10³³" from a base-10 logarithm. */
    function sci(L) {
      if (L < 6) return Math.round(Math.pow(10, L)).toLocaleString("en-US");
      var e = Math.floor(L), m = Math.pow(10, L - e);
      if (m >= 9.95) { m = 1; e++; }
      return m.toFixed(1) + " × 10" + sup(e);
    }
    function duration(L) {  // L = log10(seconds)
      var s = Math.pow(10, L);
      if (L < 0) return "under a second";
      if (s < 120) return Math.round(s) + " seconds";
      if (s < 7200) return Math.round(s / 60) + " minutes";
      if (s < 172800) return Math.round(s / 3600) + " hours";
      if (s < 2 * YEAR) return Math.round(s / 86400).toLocaleString("en-US") + " days";
      return sci(L - Math.log(YEAR) / Math.LN10) + " years";
    }
    function draw() {
      var one = M.log10Moves(state.n, state.f), all = M.log10Games(state.n, state.f, state.depth);
      stats.one.set(sci(one));
      stats.games.set(sci(all));
      stats.years.set(duration(all - 9));
      var digits = "";
      if (typeof BigInt === "function") {
        var big = BigInt(1), base = BigInt(state.f), k;
        for (k = 0; k < 3 * state.n * state.n; k++) big = big * base;
        digits = " Written out, one move has " + big.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",") + " choices.";
      }
      exact.textContent = "A move for " + state.n + " × " + state.n + " matrices fills 3 × " + (state.n * state.n) + " = " +
        (3 * state.n * state.n) + " entries, each with " + state.f + " options." + digits +
        " The number of games of " + state.depth + (state.depth === 1 ? " move" : " moves") + " has " +
        (Math.floor(all + 1e-9) + 1).toLocaleString("en-US") + " digits.";
    }
    draw();
  });
})();
