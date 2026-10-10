/* The 3D scenes in slides.qmd: two "paper in layers" scenes and a citation graph.
 *
 * One paper is drawn as four stacked sheets, from the skim layer on top
 * (abstract & headings) down to what only a reconstruction reaches (proofs,
 * code, data). Each reading pass lights the sheets it reads, filled with the
 * colour of whoever leads it (you = c1, AI = c3) and edged with the colour of
 * whoever seconds it. The AI-mode scene adds the rest of the field: six smaller
 * candidate papers, so triage visibly reads the top sheet of many papers while
 * the deep pass reads the bottom of one.
 *
 * The citation graph (graphScene, below) draws graph.js's Semantic Scholar
 * snapshot; its own comment says how. The survey chart (surveyChart) is plain SVG,
 * not three.js, and steps the same way.
 *
 * The pass rows come from model.js. Reveal fragments drive every scene: each 3D
 * slide carries `.fragment[data-pass]` steps, and the scene shows the pass of
 * the last visible one, so forwards, backwards and jumping all land in the
 * right state. The readout under the scene is plain HTML, so the slide still
 * names the pass, its lead and its time when three.js does not load.
 *
 * Keyboard belongs to Reveal here, not to the scene: a presenter's clicker
 * sends arrow keys, and a focused canvas would orbit and advance at once. The
 * canvas therefore loses its tabindex once built, and dragging still orbits.
 */
"use strict";

(function () {
  var M = PaperLayersModel;
  var W = 2.1, H = 2.97;          // an A4 sheet, in scene units
  var GAP = 0.8;                  // space between layers
  var LIFT = 0.22;                // how far a lit sheet rises
  var TINT = 0.5;                 // opacity of a lit sheet's fill
  var FADE_MS = 450;

  // ------------------------------------------------------- page furniture
  // Segments in sheet coordinates: u across, v up the page, both in [-1, 1].
  function rect(u0, v0, u1, v1) {
    return [[u0, v0, u1, v0], [u1, v0, u1, v1], [u1, v1, u0, v1], [u0, v1, u0, v0]];
  }
  // n lines of text from vTop down, ragged on the right by a fixed pattern.
  function text(vTop, n, step, u0, u1) {
    var rag = [0, 0.08, 0.02, 0.14, 0.05, 0.1, 0, 0.18, 0.04, 0.12];
    var out = [];
    for (var i = 0; i < n; i++) out.push([u0, vTop - i * step, u1 - rag[i % rag.length] * (u1 - u0), vTop - i * step]);
    return out;
  }
  var FURNITURE = {
    // Title, authors, abstract, section headings, conclusions.
    1: [].concat(
      [[-0.6, 0.86, 0.6, 0.86], [-0.55, 0.82, 0.55, 0.82], [-0.3, 0.73, 0.3, 0.73]],
      rect(-0.78, 0.22, 0.78, 0.64), text(0.57, 5, 0.075, -0.7, 0.7),
      [[-0.85, 0.08, -0.35, 0.08], [-0.85, -0.3, -0.4, -0.3], [-0.85, -0.62, -0.3, -0.62]],
      text(-0.74, 2, 0.08, -0.85, 0.75)),
    // Three panels: bars, a curve, a trend.
    2: [].concat(
      rect(-0.86, 0.1, -0.04, 0.86), rect(0.04, 0.1, 0.86, 0.86), rect(-0.86, -0.86, 0.86, -0.02),
      [0.3, 0.55, 0.42, 0.66, 0.2].map(function (h, i) {
        var u = -0.72 + i * 0.14; return [u, 0.16, u, 0.16 + h];
      }),
      [0, 1, 2, 3, 4, 5, 6].map(function (i) {
        var a = 0.12 + i * 0.1, b = a + 0.1;
        return [a, 0.2 + 0.6 * Math.pow(i / 7, 2), b, 0.2 + 0.6 * Math.pow((i + 1) / 7, 2)];
      }),
      [[-0.78, -0.78, 0.78, -0.12]]),
    // Paragraphs of argument.
    3: [].concat(text(0.86, 6, 0.1, -0.86, 0.86), text(0.2, 5, 0.1, -0.86, 0.86), text(-0.42, 5, 0.1, -0.86, 0.86)),
    // A displayed equation, indented code, a data table.
    4: [].concat(
      rect(-0.6, 0.58, 0.6, 0.88), [[-0.4, 0.73, 0.4, 0.73]],
      [[-0.86, 0.4, 0.2], [-0.7, 0.3, 0.5], [-0.54, 0.2, 0.3], [-0.54, 0.1, 0.6], [-0.7, 0, 0.1],
       [-0.86, -0.1, 0.4], [-0.7, -0.2, 0.7]].map(function (r) { return [r[0], r[1], r[2], r[1]]; }),
      rect(-0.86, -0.88, 0.86, -0.4), [[-0.86, -0.56, 0.86, -0.56], [-0.86, -0.72, 0.86, -0.72],
        [-0.3, -0.88, -0.3, -0.4], [0.28, -0.88, 0.28, -0.4]]),
    // Your note: a heading and four lines.
    note: [].concat([[-0.7, 0.6, 0.2, 0.6]], text(0.25, 4, 0.25, -0.7, 0.75))
  };

  // ------------------------------------------------------------- scene
  function scene(stage, mode) {
    var sheets = [], layerSheets = {}, field = [], cites = [], extras = {}, cancel = null;

    function sheet(c, hw, hh, z, o) {
      o = o || {};
      var s = {
        c: c, hw: hw, hh: hh, z: z, lift: 0, tint: { you: 0, ai: 0 },
        base: stage.quad("surface", { opacity: o.paper === undefined ? 0.92 : o.paper }),
        fill: { you: stage.quad("c1", { opacity: 0 }), ai: stage.quad("c3", { opacity: 0 }) },
        edge: { none: stage.line("muted", { opacity: o.edge === undefined ? 0.55 : o.edge }),
          you: stage.line("c1"), ai: stage.line("c3") },
        furniture: o.furniture ? stage.line("muted", { pairs: true, opacity: 0.45 }) : null,
        segs: o.furniture || null
      };
      s.setEdge = function (who) {
        ["none", "you", "ai"].forEach(function (k) { s.edge[k].show(k === (who || "none")); });
      };
      s.show = function (on) {
        s.base.show(on); s.fill.you.show(on); s.fill.ai.show(on);
        if (s.furniture) s.furniture.show(on);
        if (!on) ["none", "you", "ai"].forEach(function (k) { s.edge[k].show(false); });
      };
      s.setEdge(null);
      sheets.push(s);
      return s;
    }

    function place(s) {
      var z = s.z + s.lift, x = s.c[0], y = s.c[1];
      var p = [[x - s.hw, y - s.hh], [x + s.hw, y - s.hh], [x + s.hw, y + s.hh], [x - s.hw, y + s.hh]];
      function at(dz) { return p.map(function (q) { return [q[0], q[1], z + dz]; }); }
      var c0 = at(0), c1 = at(0.004);
      s.base.set(c0[0], c0[1], c0[2], c0[3]);
      s.fill.you.set(c1[0], c1[1], c1[2], c1[3]); s.fill.you.opacity(s.tint.you);
      s.fill.ai.set(c1[0], c1[1], c1[2], c1[3]); s.fill.ai.opacity(s.tint.ai);
      var ring = at(0.008); ring.push(ring[0]);
      ["none", "you", "ai"].forEach(function (k) { s.edge[k].set(ring); });
      if (s.furniture) {
        var pts = [], ku = s.hw * 0.86, kv = s.hh * 0.9;
        s.segs.forEach(function (g) {
          pts.push([x + g[0] * ku, y + g[1] * kv, z + 0.006], [x + g[2] * ku, y + g[3] * kv, z + 0.006]);
        });
        s.furniture.set(pts);
      }
    }

    // The paper: layer 1 on top.
    var top = 3 * GAP;
    M.LAYERS.forEach(function (l, i) {
      var s = sheet([0, 0], W / 2, H / 2, top - i * GAP, { furniture: FURNITURE[l.id] });
      // One label per colour, since a stage label keeps the colour it was made with.
      var anchor = [W / 2 + 0.12, -H / 2 + 0.25, s.z];
      var dx = 10 + l.short.length * 4.2;
      s.labels = { none: stage.label(l.short, "muted", { dx: dx }),
        you: stage.label(l.short, "c1", { dx: dx }), ai: stage.label(l.short, "c3", { dx: dx }) };
      s.anchor = anchor;
      layerSheets[l.id] = s;
    });

    // Your note, above the stack, only when integrating.
    var note = sheet([0.15, 0.25], 0.8, 0.55, top + 1.1, { furniture: FURNITURE.note });
    note.labelNote = stage.label("your note", "c1", { dx: 46 });
    layerSheets.note = note;

    // Who seconds the pass, pinned to the left of the sheets it reads.
    var second = {
      ai: { text: "AI as tutor", label: stage.label("AI as tutor", "c3", { dx: -58 }) },
      you: { text: "then you judge", label: stage.label("then you judge", "c1", { dx: -68 }) }
    };

    // The rest of the field: smaller candidate papers on an arc behind.
    if (mode === "ai") {
      var k = 0.5;
      // Three a side, leaving the space straight behind the paper open so no
      // candidate hides behind it from the starting view.
      [38, 62, 86, 150, 174, 198].forEach(function (deg) {
        var a = deg * Math.PI / 180, c = [5 * Math.cos(a), 5 * Math.sin(a)], stack = [];
        for (var i = 0; i < 4; i++) {
          stack.push(sheet(c, k * W / 2, k * H / 2, (3 - i) * 0.3, { paper: 0.8, edge: 0.4 }));
        }
        field.push(stack);
      });
      cites = [1, 4].map(function (i) {
        return { to: field[i], line: stage.line("c3", { dashed: true, dash: 0.14, gap: 0.09 }) };
      });
    }

    function placeAll() {
      sheets.forEach(place);
      M.LAYERS.forEach(function (l) {
        var s = layerSheets[l.id];
        ["none", "you", "ai"].forEach(function (key) {
          s.labels[key].set([s.anchor[0], s.anchor[1], s.z + s.lift]);
        });
      });
      note.labelNote.set([note.c[0] + note.hw, note.c[1], note.z + note.lift]);
      cites.forEach(function (ct) {
        var from = layerSheets[3], t = ct.to[0];
        ct.line.set([[from.c[0], from.c[1] + from.hh * 0.3, from.z + from.lift + 0.01],
          [t.c[0], t.c[1], t.z + t.lift + 0.01]]);
      });
      if (extras.second) {
        var s = extras.second.sheet;
        extras.second.label.set([s.c[0] - s.hw - 0.08, s.c[1], s.z + s.lift]);
      }
    }

    /** Light the sheets for pass row `p` (null: every sheet neutral). */
    function show(p, animated) {
      var lit = {}, targets = [];
      if (p) p.layers.forEach(function (id) { lit[id] = true; });
      var litField = p && p.candidates === "top" ? field
        : p && p.candidates === "cited" ? cites.map(function (ct) { return ct.to; }) : [];

      function aim(s, who, edgeWho) {
        s.setEdge(who ? edgeWho || who : null);
        targets.push({ s: s, lift: [s.lift, who ? LIFT : 0],
          you: [s.tint.you, who === "you" ? TINT : 0], ai: [s.tint.ai, who === "ai" ? TINT : 0] });
      }
      M.LAYERS.forEach(function (l) {
        var s = layerSheets[l.id], who = lit[l.id] ? p.lead : null;
        aim(s, who, p && p.second);
        ["none", "you", "ai"].forEach(function (key) { s.labels[key].show(key === (who || "none")); });
      });
      aim(note, lit.note ? "you" : null);
      note.show(!!lit.note);  // after aim, which would show its neutral edge
      note.labelNote.show(!!lit.note);
      field.forEach(function (stack) {
        stack.forEach(function (s, i) { aim(s, i === 0 && litField.indexOf(stack) >= 0 ? "ai" : null); });
      });
      cites.forEach(function (ct) { ct.line.show(p && p.candidates === "cited"); });

      second.ai.label.show(false); second.you.label.show(false);
      extras.second = null;
      if (p && p.second) {
        // Pinned beside the lowest lit layer, where the help or the check lands.
        var ids = p.layers.filter(function (id) { return typeof id === "number"; });
        extras.second = { label: second[p.second].label, sheet: layerSheets[ids[ids.length - 1]] };
        extras.second.label.show(true);
      }

      if (cancel) cancel();
      function step(t) {
        targets.forEach(function (g) {
          g.s.lift = g.lift[0] + (g.lift[1] - g.lift[0]) * t;
          g.s.tint.you = g.you[0] + (g.you[1] - g.you[0]) * t;
          g.s.tint.ai = g.ai[0] + (g.ai[1] - g.ai[0]) * t;
        });
        placeAll();
      }
      if (animated) cancel = stage.animate(FADE_MS, step);
      else { step(1); stage.render(); }
    }

    show(null, false);
    return show;
  }

  // ---------------------------------------------------- the citation graph
  // Papers Semantic Scholar says cite Keshav's "How to read a paper", from
  // graph.js. Height is publication year, so a citation should point down; a
  // faint floor at 2007 marks Keshav's year, and anything under it predates
  // the paper it supposedly cites. Colour is Semantic Scholar's own field tag.
  // On the dark ground: Keshav in ink (also labelled), and three field colours
  // that the dataviz validator separates for normal and colour-deficient vision.
  var FIELD = { seed: "ink", cs: "c5", med: "c3", other: "c6" };

  function graphScene(stage) {
    var G = PaperGraph, T = stage.THREE, n = G.nodes.length;
    var P = G.pos;
    var radius = G.nodes.map(function (nd) { return 0.06 + 0.035 * Math.log10(1 + nd.cites); });
    var zLo = (G.yearRange[0] - G.Y0) * G.PER_YEAR, zHi = (G.yearRange[1] - G.Y0) * G.PER_YEAR;

    // Keshav's year as a floor, and a year axis at one corner.
    stage.grid([-3, 3], [-3, 3], 1, 0, "muted", 0.18);
    var ax = [-3.2, 0];
    stage.line("muted", { opacity: 0.6 }).set([[ax[0], ax[1], zLo], [ax[0], ax[1], zHi]]);
    [G.yearRange[0], G.Y0, 2015, G.yearRange[1]].forEach(function (y) {
      stage.label(String(y), y === G.Y0 ? "c1" : "muted", { dx: -24 })
        .set([ax[0], ax[1], (y - G.Y0) * G.PER_YEAR]);
    });

    // Papers: one instanced mesh, a colour and a size each.
    var mesh = new T.InstancedMesh(new T.SphereGeometry(1, 18, 12), new T.MeshLambertMaterial(), n);
    mesh.frustumCulled = false;
    stage.scene.add(mesh);
    var m4 = new T.Matrix4(), q = new T.Quaternion(), s = new T.Vector3();
    function paint(focus) {
      var paper = stage.colour("paper");
      G.nodes.forEach(function (nd, i) {
        var dim = focus && focus.indexOf(i) < 0, r = radius[i] * (focus && !dim ? 1.3 : 1);
        mesh.setMatrixAt(i, m4.compose(stage.V(P[i]), q, s.set(r, r, r)));
        var c = stage.colour(FIELD[nd.field]);
        mesh.setColorAt(i, dim ? c.clone().lerp(paper, 0.75) : c);
      });
      mesh.instanceMatrix.needsUpdate = true;
      mesh.instanceColor.needsUpdate = true;
    }

    // Citations: thin lines when they point down in time, arrows when they point up.
    function shorten(a, b, by) {
      var d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], len = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      var k = (len - by) / len;
      return [a[0] + d[0] * k, a[1] + d[1] * k, a[2] + d[2] * k];
    }
    var up = G.upLinks;
    // Dim on purpose: lightness, not hue, is what keeps the up arrows apart from
    // these for every kind of colour vision.
    var down = stage.line("rule", { pairs: true });
    down.set([].concat.apply([], G.citation.filter(function (e) { return up.indexOf(e) < 0; })
      .map(function (e) { return [P[e[0]], P[e[1]]]; })));
    var arrows = up.map(function (e) {
      var a = stage.arrow("c4", { radius: 0.03 });
      a.set(P[e[0]], shorten(P[e[0]], P[e[1]], radius[e[1]] + 0.03));
      return a;
    });
    // Connected links are the step's whole point, so they are rods, not 1px lines.
    function rod(token, r) {
      var geo = new T.CylinderGeometry(r, r, 1, 10);
      geo.translate(0, 0.5, 0);
      var mesh = new T.Mesh(geo, stage.material("lambert", token));
      mesh.frustumCulled = false;
      stage.scene.add(mesh);
      return {
        set: function (a, b) {
          var A = stage.V(a), d = stage.V(b).sub(A), len = d.length();
          mesh.position.copy(A);
          mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), d.normalize());
          mesh.scale.set(1, len, 1);
        },
        show: function (v) { mesh.visible = !!v; }
      };
    }
    var linked = G.connected.map(function (p) { var l = rod("ink", 0.016); l.set(P[p.a], P[p.b]); return l; });
    var inLinks = [];
    G.connected.forEach(function (p) { [p.a, p.b].forEach(function (i) { if (inLinks.indexOf(i) < 0) inLinks.push(i); }); });

    // Labels: Keshav always, the most-cited when ranking, any paper under the pointer.
    function trim(t, k) { return t.length > k ? t.slice(0, k - 1).trim() + "…" : t; }
    stage.label("Keshav 2007", "ink", { dy: -18 }).set(P[0]);
    var ranked = G.topCited.map(function (i) {
      // Counts only: the most-cited cluster near 2007, where titles would collide.
      // Hovering a paper still gives its title.
      var l = stage.label(G.nodes[i].cites + " citations", "ink", { dy: -16 });
      l.set(P[i]);
      return l;
    });
    var hover = stage.label("", "ink", { dy: -18 });
    hover.show(false);

    function show(view) {
      down.show(view === "cites");
      arrows.forEach(function (a) { a.show(view === "cites"); });
      linked.forEach(function (l) { l.show(view === "connected"); });
      ranked.forEach(function (l) { l.show(view === "ranked"); });
      // Keshav stays lit when ranking, as the paper they all supposedly cite.
      paint(view === "ranked" ? G.topCited.concat([0]) : view === "connected" ? inLinks : null);
      stage.render();
    }
    show(null);

    return {
      show: show,
      hover: function (canvasEl) {
        // Redraw only when the paper under the pointer changes, not on every move.
        var last = -1;
        canvasEl.addEventListener("pointermove", function (ev) {
          if (ev.buttons) return;
          var rect = canvasEl.getBoundingClientRect(), best = -1, bestD = 14 * 14;
          P.forEach(function (p, i) {
            var v = stage.V(p).project(stage.camera);
            var x = rect.left + (v.x * 0.5 + 0.5) * rect.width, y = rect.top + (-v.y * 0.5 + 0.5) * rect.height;
            var d = (x - ev.clientX) * (x - ev.clientX) + (y - ev.clientY) * (y - ev.clientY);
            if (d < bestD) { bestD = d; best = i; }
          });
          if (best === last) return;
          last = best;
          if (best < 0) { hover.show(false); stage.render(); return; }
          var nd = G.nodes[best];
          hover.set(P[best], trim(nd.title, 56) + " (" + nd.year + ") · " + nd.cites + " citations");
          hover.show(true);
          stage.render();
        });
        canvasEl.addEventListener("pointerleave", function () {
          if (last < 0) return;
          last = -1;
          hover.show(false);
          stage.render();
        });
      }
    };
  }

  // ------------------------------------------------------- the survey chart
  // Hours to survey a field and then read five papers properly, by hand and
  // with AI triage: arithmetic on the post's per-paper times (model.js,
  // budget). Colour is whose time it is, as on the layer slides: you = c1,
  // AI-led = c3. The five Pass 2 reads (~1 h each) come first, so the part AI does not
  // shorten sits on the axis and the part it does grows to the right. A range
  // the post gives (5–10 min a paper) is solid to its low end and faded to its
  // high end. One axis, fixed across steps, so a bigger field visibly costs more.
  function surveyChart(root) {
    // X1 leaves room past the axis for the longest end label ("+ 8.3–16.7 h").
    var W = 640, H = 250, X0 = 140, X1 = 505, MAXH = 22;
    var x = WK.lin([0, MAXH], [X0, X1]);
    var svg = WK.svg(W, H, { class: "pl-chart-svg", "aria-label":
      "Hours to survey a field and read five papers in depth, by hand and with AI triage." });
    WK.ticks(0, MAXH, 4).forEach(function (t) {
      svg.appendChild(WK.h("line", { x1: x(t), x2: x(t), y1: 18, y2: 200, stroke: "rule" }));
      svg.appendChild(WK.h("text", { x: x(t), y: 222, "text-anchor": "middle", fill: "muted", "font-size": 15,
        text: t + " h" }));
    });
    var rows = [
      { y: 40, label: "By hand", sub: "5–10 min a paper" },
      { y: 125, label: "With AI triage", sub: "~2 min a paper" }
    ];
    var BAR = 46, GAP = 2;
    rows.forEach(function (r) {
      svg.appendChild(WK.h("text", { x: 0, y: r.y + 20, fill: "ink", "font-size": 17, "font-weight": 700, text: r.label }));
      svg.appendChild(WK.h("text", { x: 0, y: r.y + 40, fill: "muted", "font-size": 13, text: r.sub }));
      r.deep = WK.h("rect", { y: r.y, height: BAR, rx: 3, fill: "c1" });
      r.solid = WK.h("rect", { y: r.y, height: BAR, rx: 3, fill: r === rows[0] ? "c1" : "c3" });
      r.faded = WK.h("rect", { y: r.y, height: BAR, rx: 3, fill: "c1", "fill-opacity": 0.38 });
      r.inDeep = WK.h("text", { x: x(0) + 7, y: r.y + 28, "font-size": 13, "font-weight": 700, fill: "surface",
        text: M.SURVEY.deep + " × Pass 2" });
      r.end = WK.h("text", { y: r.y + 29, "font-size": 16, "font-weight": 700, fill: "ink" });
      r.inSurvey = WK.h("text", { y: r.y + 28, "font-size": 13, "font-weight": 700, fill: "surface",
        text: r === rows[0] ? "first passes" : "triage" });
      [r.deep, r.solid, r.faded, r.inDeep, r.end, r.inSurvey].forEach(function (n) { svg.appendChild(n); });
    });
    root.appendChild(svg);

    var h = M.hours;
    function span(node, a, b) {
      node.setAttribute("x", x(a));
      node.setAttribute("width", Math.max(0, x(b) - x(a)));
      node.style.display = b > a ? "" : "none";
    }
    return function (n) {
      var b = M.budget(n || 0), deep = b.deep;
      rows.forEach(function (r, i) {
        span(r.deep, 0, deep);
        var lo = i === 0 ? b.hand[0] : b.ai, hi = i === 0 ? b.hand[1] : b.ai;
        var start = deep + GAP / (x(1) - x(0));
        span(r.solid, start, start + lo);
        span(r.faded, start + lo, start + hi);
        // Name the survey segment inside it when it is wide enough to hold the words.
        r.inSurvey.setAttribute("x", x(start) + 7);
        r.inSurvey.style.display = x(start + lo) - x(start) > (i === 0 ? 92 : 52) ? "" : "none";
        r.end.setAttribute("x", x(start + hi) + 8);
        r.end.textContent = n ? "+ " + (lo === hi ? h(lo) : h(lo) + "–" + h(hi)) + " h" : deep + " h";
      });
    };
  }
  // --------------------------------------------------------------- mount
  // A scene on a slide: the stage, a readout of a name and two tiles under it,
  // and a set(pass) that the fragment sync below calls. `describe(pass)` gives
  // the readout for a pass (null before the first step); `build(stage)` returns
  // draw(pass, animated).
  var mounted = [];

  function mountScene(id, o) {
    WK.mount(id, function (root) {
      root.setAttribute("data-prevent-swipe", "");
      var host = WK.el("div", { class: "pl-scene" });
      var name = WK.el("div", { class: "pl-name" });
      var tiles = [WK.stat({ label: "", value: "—" }), WK.stat({ label: "", value: "—" })];
      root.appendChild(host);
      root.appendChild(WK.el("div", { class: "pl-readout" }, name,
        WK.el("div", { class: "widget-stats" }, tiles[0].root, tiles[1].root)));

      var entry = { section: root.closest("section"), pass: undefined, draw: null };
      entry.set = function (pass) {
        if (pass === entry.pass) return;
        var first = entry.pass === undefined;
        entry.pass = pass;
        var d = o.describe(pass);
        name.textContent = d.name;
        d.tiles.forEach(function (t, i) {
          tiles[i].root.querySelector(".widget-stat-label").textContent = t[0];
          tiles[i].set(t[1]);
        });
        if (entry.draw) entry.draw(pass, !first);
      };

      var box = WKStage.create(host, { label: o.label, view: o.view }, function (stage) {
        var canvas = box.querySelector("canvas");
        canvas.removeAttribute("tabindex");
        canvas.setAttribute("aria-label", o.label + " Drag to rotate.");
        entry.draw = o.build(stage, canvas);
        entry.draw(entry.pass === undefined ? null : entry.pass, false);
      });

      mounted.push(entry);
      sync();
    });
  }

  // A scene shows the pass of the last visible step on its slide.
  function sync() {
    mounted.forEach(function (e) {
      var steps = e.section ? e.section.querySelectorAll(".fragment.visible[data-pass]") : [];
      e.set(steps.length ? steps[steps.length - 1].getAttribute("data-pass") : null);
    });
  }
  ["ready", "slidechanged", "fragmentshown", "fragmenthidden"].forEach(function (ev) {
    document.addEventListener(ev, sync);
  });

  function layers(id, mode, view, label) {
    mountScene(id, {
      view: view, label: label,
      describe: function (pass) {
        var p = pass === null ? null : M.find(mode, pass);
        return { name: p ? p.name : "→ steps through the passes",
          tiles: [["who leads", p ? M.who(p) : "—"], ["time", p ? p.time : "—"]] };
      },
      build: function (stage) {
        var show = scene(stage, mode);
        return function (pass, animated) { show(pass === null ? null : M.find(mode, pass), animated); };
      }
    });
  }

  layers("layers-classic", "classic", { az: -62, el: 24, dist: 8.6, target: [0.45, 0, 1.25] },
    "A research paper drawn as four stacked sheets, lit by Keshav's three reading passes.");
  layers("layers-ai", "ai", { az: -62, el: 22, dist: 15, target: [-0.5, 1.0, 1.0] },
    "A research paper drawn as four stacked sheets among six candidate papers, lit by the six-step protocol.");

  if (typeof PaperGraph !== "undefined") {
    var G = PaperGraph;
    // The slide's prose numbers come from the same snapshot as its tiles, so a
    // re-fetch cannot leave the two disagreeing. The typed values are the fallback.
    var GV = { citingTotal: G.citingTotal, predates: G.predates, medicine: G.tagged("Medicine"), fetched: G.fetched };
    WK.mount("citation-graph", function (root) {
      var section = root.closest("section") || document;
      section.querySelectorAll("[data-g]").forEach(function (n) {
        var v = GV[n.getAttribute("data-g")];
        if (v !== undefined) n.textContent = String(v);
      });
    });
    mountScene("citation-graph", {
      view: { az: -58, el: 16, dist: 12.5, target: [0, 0, 1.0] },
      label: "Papers that Semantic Scholar lists as citing Keshav's How to Read a Paper, with height as publication year.",
      describe: function (pass) {
        return {
          null: { name: G.nodes.length + " papers, by year",
            tiles: [["citing papers drawn", String(G.nodes.length - 1)], ["years", G.yearRange.join("–")]] },
          cites: { name: "Citations · height is the year",
            tiles: [["citation links", String(G.citation.length)], ["pointing up in time", String(G.upLinks.length)]] },
          connected: { name: "Connected · shared references",
            tiles: [["connected links", String(G.connected.length)],
              ["where neither cites the other", String(G.connectedNotCiting)]] },
          ranked: { name: "Ranked by citations",
            tiles: [["of the top " + G.TOP + " citing papers are medical", String(G.topCitedMed)],
              ["also among the " + G.TOP + " best connected", String(G.overlap)]] }
        }[pass === null ? "null" : pass];
      },
      build: function (stage, canvas) {
        var g = graphScene(stage);
        g.hover(canvas);
        return function (pass) { g.show(pass); };
      }
    });
  }

  // The survey chart: an SVG, not a stage, on the same fragment sync.
  WK.mount("survey-chart", function (root) {
    var name = WK.el("div", { class: "pl-name" });
    var tiles = [WK.stat({ label: "first passes, by hand", value: "—" }),
      WK.stat({ label: "AI triage", value: "—" }), WK.stat({ label: "Pass 2 on five, either way", value: "—" })];
    var draw = surveyChart(root);
    root.appendChild(WK.el("div", { class: "pl-readout" }, name,
      WK.el("div", { class: "widget-stats" }, tiles[0].root, tiles[1].root, tiles[2].root)));
    var entry = { section: root.closest("section"), pass: undefined };
    entry.set = function (pass) {
      if (pass === entry.pass) return;
      entry.pass = pass;
      var n = pass === null ? 0 : Number(pass), b = M.budget(n);
      var h = M.hours;
      draw(n);
      name.textContent = n ? n + " candidates, then five read properly" : "Five papers read properly";
      tiles[0].set(n ? h(b.hand[0]) + "–" + h(b.hand[1]) + " h" : "—");
      tiles[1].set(n ? h(b.ai) + " h" : "—");
      tiles[2].set(h(b.deep) + " h");
    };
    mounted.push(entry);
    sync();
  });
})();
