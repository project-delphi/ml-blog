/* The picture the three AlphaTensor posts share: the matrix multiplication
 * tensor as a stack of trays, one tray per entry of C, drawn on widget-kit's
 * stage (WKStage) and again as a flat strip of grids (plain SVG).
 *
 * Browser-only. Lives beside model.js in Part 1's folder; Parts 2 and 3 load
 * both by relative path. Exposes one global, `TensorCube`.
 *
 * Layout. Tray c holds the 4 x 4 slice T[:, :, c] laid out as a matrix: rows
 * are the entries of A, running away from the viewer, and columns are the
 * entries of B, running left to right. c11 is the top tray. At the default
 * camera every 1 of the tensor is in view, for 2 x 2 and for 3 x 3.
 *
 * Both views take the same state:
 *   { n, values, touched, preview, slice, cell }
 * `values` is the flat tensor to draw. `touched` marks cells that held a
 * nonzero value earlier, drawn hollow once they are back at zero. `preview`
 * is a second flat tensor shown translucent over the first. `slice` is the
 * selected tray (an index into C, or null) and `cell` the selected {a, b, c}.
 */
"use strict";

var TensorCube = (function () {
  var M = MatmulModel;
  var GAP = 1.3, LIFT = 1.0, BOX = 0.66, MAX_LABELS = 64;  // a label for every cell of the 2 x 2 cube

  function half(n) { return (n * n - 1) / 2; }
  function view(n) {
    return n === 2 ? { az: -108, el: 27, dist: 16, target: [0, 0, -0.5] }
                   : { az: -108, el: 27, dist: 37, target: [0, 0, -0.4] };
  }
  function valueText(v) { return v < 0 ? "−" + (-v) : String(v); }
  // Digits alone for 3 x 3: nine full names along one axis collide on a phone.
  function tick(letter, n, i) { return n === 2 ? M.name(letter, n, i) : M.name("", n, i); }

  /** The 3D view. `create(stage, nMax)` allocates for the largest n it will be
   *  asked to draw, since a stage has no way to remove an object. */
  function create(stage, nMax) {
    var NM = nMax * nMax, CELLS = NM * NM * NM;
    var zero = stage.dots("muted", { radius: 0.05, max: CELLS, opacity: 0.45 });
    var hollow = stage.line("muted", { pairs: true, opacity: 0.85 });
    var trays = stage.line("muted", { pairs: true, opacity: 0.3 });
    var floor = stage.quad("c2", { opacity: 0.16 });
    var guides = stage.line("ink", { pairs: true, dashed: true, dash: 0.14, gap: 0.1, opacity: 0.75 });
    var solid = stage.boxes(CELLS);
    var ghost = stage.boxes(CELLS, { opacity: 0.4 });
    var ticks = { a: [], b: [], c: [] }, values = [], i;
    for (i = 0; i < NM; i++) {
      ticks.a.push(stage.label("", "muted"));
      ticks.b.push(stage.label("", "muted"));
      ticks.c.push(stage.label("", "muted"));
    }
    var titles = { a: stage.label("A", "muted"), b: stage.label("B", "muted"), c: stage.label("C", "muted") };
    for (i = 0; i < MAX_LABELS; i++) values.push(stage.label("", "ink"));
    var callout = stage.label("", "ink", { dy: -30 });

    var state = null, lift = [], cancel = null;
    for (i = 0; i < NM; i++) lift.push(0);

    function targetLift(st) {
      var N = st.n * st.n, out = [], c;
      // Trays above the selected one rise and trays under it sink, which opens
      // it to the camera and clears room beside it for the tick labels.
      for (c = 0; c < NM; c++)
        out.push(st.slice === null || st.slice === undefined || c >= N || c === st.slice ? 0 : c < st.slice ? LIFT : -LIFT);
      return out;
    }
    function at(n, a, b, c) {
      var h = half(n);
      return [b - h, h - a, (h - c) * GAP + lift[c]];
    }
    function edges(p, s, out) {
      // The twelve edges of a cube: four parallel to each axis.
      var r = s / 2, x, y;
      for (x = -1; x <= 1; x += 2) for (y = -1; y <= 1; y += 2) {
        out.push([p[0] + x * r, p[1] + y * r, p[2] - r], [p[0] + x * r, p[1] + y * r, p[2] + r]);
        out.push([p[0] + x * r, p[1] - r, p[2] + y * r], [p[0] + x * r, p[1] + r, p[2] + y * r]);
        out.push([p[0] - r, p[1] + x * r, p[2] + y * r], [p[0] + r, p[1] + x * r, p[2] + y * r]);
      }
      return out;
    }

    function paint() {
      var st = state, n = st.n, N = n * n, h = half(n), picked = st.slice !== null && st.slice !== undefined;
      var solids = [], ghosts = [], zeros = [], wire = [], rim = [], used = 0, a, b, c, v, p, idx, dim;
      for (c = 0; c < N; c++) {
        dim = picked && c !== st.slice;
        var z = (h - c) * GAP + lift[c] - 0.42, e = h + 0.5;
        rim.push([-e, -e, z], [e, -e, z], [e, -e, z], [e, e, z], [e, e, z], [-e, e, z], [-e, e, z], [-e, -e, z]);
        for (a = 0; a < N; a++) for (b = 0; b < N; b++) {
          idx = M.cell(N, a, b, c); v = st.values[idx]; p = at(n, a, b, c);
          var chosen = st.cell && st.cell.a === a && st.cell.b === b && st.cell.c === c;
          if (v) {
            solids.push({ at: p, size: chosen ? 0.92 : Math.abs(v) > 1 ? 0.86 : BOX,
              token: v > 0 ? "c1" : "c3", mix: dim && !chosen ? 0.74 : 0 });
            if ((v < 0 || Math.abs(v) > 1) && used < MAX_LABELS) {
              values[used].set(p, valueText(v)); values[used].show(true); used++;
            }
          } else if (st.touched && st.touched[idx]) edges(p, BOX, wire);
          else zeros.push(p);
          if (st.preview && st.preview[idx])
            ghosts.push({ at: p, size: 0.84, token: st.preview[idx] > 0 ? "c1" : "c3" });
        }
      }
      for (; used < MAX_LABELS; used++) values[used].show(false);
      // Keep the stack centred while trays spread around a selected one.
      var mean = 0;
      for (c = 0; c < N; c++) mean += lift[c] / N;
      stage.view.target = [0, 0, view(n).target[2] + mean];
      solid.set(solids); ghost.set(ghosts); zero.set(zeros); hollow.set(wire); trays.set(rim);

      // The A and B ticks ride with the selected tray, so they sit beside the
      // cells they name; with nothing selected they sit under the stack.
      var base = picked ? st.slice : N - 1, zt = (h - base) * GAP + lift[base] - 0.42, rail = h + 1.15, k;
      for (k = 0; k < NM; k++) {
        ticks.a[k].show(k < N); ticks.b[k].show(k < N); ticks.c[k].show(k < N);
        if (k >= N) continue;
        ticks.a[k].set([-rail, h - k, zt], tick("a", n, k));
        ticks.b[k].set([k - h, -rail, zt], tick("b", n, k));
        ticks.c[k].set([rail + 0.2, -h - 0.5, (h - k) * GAP + lift[k] - 0.1], tick("c", n, k));
      }
      titles.a.show(n > 2); titles.b.show(n > 2); titles.c.show(n > 2);
      titles.a.set([-rail - 1.1, 0, zt]);
      titles.b.set([0, -rail - 1.1, zt]);
      titles.c.set([rail + 0.2, -h - 0.5, h * GAP + lift[0] + 0.9]);

      floor.show(picked);
      if (picked) {
        var zf = (h - st.slice) * GAP + lift[st.slice] - 0.42, f = h + 0.5;
        floor.set([-f, -f, zf], [f, -f, zf], [f, f, zf], [-f, f, zf]);
      }
      if (st.cell) {
        p = at(n, st.cell.a, st.cell.b, st.cell.c);
        guides.set([p, [-rail + 0.35, p[1], p[2]], p, [p[0], -rail + 0.35, p[2]]]);
        callout.set(p, M.name("a", n, st.cell.a) + " · " + M.name("b", n, st.cell.b) + " → " + M.name("c", n, st.cell.c));
      }
      guides.show(!!st.cell); callout.show(!!st.cell);
    }

    return {
      /** Draw a state. Trays glide to their new heights; all else changes at once. */
      set: function (st) {
        var from = lift.slice(), to = targetLift(st), moved = false, c;
        state = st;
        for (c = 0; c < NM; c++) if (from[c] !== to[c]) moved = true;
        if (cancel) { cancel(); cancel = null; }
        if (!moved) { paint(); stage.render(); return; }
        cancel = stage.animate(420, function (t) {
          for (c = 0; c < NM; c++) lift[c] = from[c] + (to[c] - from[c]) * t;
          paint();
        });
      },
      /** Point the camera at an n x n cube. */
      frame: function (n) {
        var v = view(n);
        stage.view.dist = v.dist; stage.view.target = v.target;
      }
    };
  }

  /** The flat view: one grid per tray, rows a11.. and columns b11... It needs
   *  no WebGL, so it is also what a reader sees when the 3D scene cannot load. */
  function flat() {
    var svg = WK.svg(600, 170, { "aria-label": "The slices of the tensor, one grid per entry of C." });
    function set(st) {
      var n = st.n, N = n * n, perRow = n === 2 ? 4 : 5, rowsOf = Math.ceil(N / perRow);
      var S = n === 2 ? 30 : 13, G = S * N, left = n === 2 ? 40 : 8, top = 34, gapX = n === 2 ? 22 : 18, gapY = 34;
      var W = left + perRow * G + (perRow - 1) * gapX + 6, H = top + rowsOf * G + (rowsOf - 1) * gapY + 8;
      var picked = st.slice !== null && st.slice !== undefined, c, a, b, v, idx;
      WK.clear(svg);
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      for (c = 0; c < N; c++) {
        var gx = left + (c % perRow) * (G + gapX), gy = top + Math.floor(c / perRow) * (G + gapY);
        var g = WK.h("g", { opacity: picked && c !== st.slice ? 0.4 : 1 });
        g.appendChild(WK.h("text", { x: gx + G / 2, y: gy - (n === 2 ? 19 : 8), "text-anchor": "middle",
          "font-size": 13, "font-weight": 700, fill: picked && c === st.slice ? "c2" : "ink",
          text: "slice " + M.name("c", n, c) }));
        for (a = 0; a < N; a++) for (b = 0; b < N; b++) {
          idx = M.cell(N, a, b, c); v = st.values[idx];
          var x = gx + b * S, y = gy + a * S, was = !v && st.touched && st.touched[idx];
          g.appendChild(WK.h("rect", { x: x + 1, y: y + 1, width: S - 2, height: S - 2, rx: 3,
            fill: v > 0 ? "c1" : v < 0 ? "c3" : "surface", stroke: v ? null : was ? "muted" : "rule",
            "stroke-dasharray": was ? "3 2" : null }));
          if (v && n === 2) g.appendChild(WK.h("text", { x: x + S / 2, y: y + S / 2 + 4.5, "text-anchor": "middle",
            "font-size": 13, "font-weight": 700, fill: "surface", text: valueText(v) }));
          if (st.preview && st.preview[idx]) g.appendChild(WK.h("rect", { x: x + 3.5, y: y + 3.5, width: S - 7,
            height: S - 7, rx: 2, fill: "none", stroke: st.preview[idx] > 0 ? "c1" : "c3", "stroke-width": 2.5,
            "stroke-dasharray": v ? "4 2" : null }));
          if (st.cell && st.cell.a === a && st.cell.b === b && st.cell.c === c)
            g.appendChild(WK.h("rect", { x: x - 1, y: y - 1, width: S + 2, height: S + 2, rx: 4, fill: "none",
              stroke: "ink", "stroke-width": 2.5 }));
        }
        if (n === 2) for (b = 0; b < N; b++) g.appendChild(WK.h("text", { x: gx + b * S + S / 2, y: gy - 4,
          "text-anchor": "middle", "font-size": 10.5, fill: "muted", text: M.name("b", n, b) }));
        if (picked && c === st.slice) g.appendChild(WK.h("rect", { x: gx - 3, y: gy - 3, width: G + 6, height: G + 6,
          rx: 6, fill: "none", stroke: "c2", "stroke-width": 2 }));
        svg.appendChild(g);
      }
      // Row names once per row of grids, at the left.
      if (n === 2) for (a = 0; a < N; a++) svg.appendChild(WK.h("text", { x: left - 6, y: top + a * S + S / 2 + 4,
        "text-anchor": "end", "font-size": 10.5, fill: "muted", text: M.name("a", n, a) }));
    }
    return { root: svg, set: set };
  }

  return { create: create, flat: flat, view: view };
})();
