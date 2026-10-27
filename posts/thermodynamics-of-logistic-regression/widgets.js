/* Two views of one ten-row conditional model. Numeric readouts do not depend on WebGL. */
"use strict";

(function () {
  var M = ThermoModel;
  var LIMIT = 2;
  function number(x, n) { return x.toFixed(n === undefined ? 2 : n); }
  function control(label, value, min, max, change) {
    return WK.slider({ label: label, value: value, min: min, max: max, step: 0.05,
      fmt: function (v) { return number(v); }, oninput: change });
  }
  function boundary(p) {
    // Intersect w1*x1 + w2*x2 + b = 0 with the four edges of the square.
    var pts = [];
    function add(x, y) {
      if (Math.abs(x) > LIMIT + 1e-8 || Math.abs(y) > LIMIT + 1e-8) return;
      if (!pts.some(function (q) { return Math.hypot(q[0] - x, q[1] - y) < 1e-7; })) pts.push([x, y, 0]);
    }
    if (Math.abs(p.w2) > 1e-9) {
      add(-LIMIT, -(p.b - p.w1 * LIMIT) / p.w2);
      add(LIMIT, -(p.b + p.w1 * LIMIT) / p.w2);
    }
    if (Math.abs(p.w1) > 1e-9) {
      add(-(p.b - p.w2 * LIMIT) / p.w1, -LIMIT);
      add(-(p.b + p.w2 * LIMIT) / p.w1, LIMIT);
    }
    return pts.length >= 2 ? [pts[0], pts[1]] : [];
  }

  WK.mount("thermo-energy", function (root) {
    var p = { w1: 0.8, w2: -0.65, b: 0.2 }, beta = 1;
    var stage, parts;
    var f = WK.frame({ title: "Two label energies at fixed inputs", badge: "3D · synthetic",
      note: "Ten fixed synthetic feature pairs. The sloped sheet is E₁ = −η; the flat sheet is E₀ = 0. " +
        "Small dots mark observed labels on their respective sheets. The line where the sheets cross is their equal-energy boundary. " +
        "The larger dot is the fixed input (0.55, −1.25) with observed label 1. Drag to orbit; arrow keys work when the scene has focus." });
    [["w₁", "w1", -1.2, 1.2], ["w₂", "w2", -1.2, 1.2], ["Bias b", "b", -1, 1]].forEach(function (spec) {
      f.controls.appendChild(control(spec[0], p[spec[1]], spec[2], spec[3], function (v) {
        p[spec[1]] = v; draw();
      }).root);
    });
    f.controls.appendChild(control("Inverse temperature β", beta, 0, 3, function (v) { beta = v; draw(); }).root);
    var gap = WK.stat({ label: "Gap ΔE at highlighted input", value: "" });
    var prob = WK.stat({ label: "P(y = 1 | x) at highlighted input", value: "" });
    var surprise = WK.stat({ label: "Observed-label surprisal (nats)", value: "" });
    [gap, prob, surprise].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    WKStage.create(f.body, { label: "Two label energy sheets over the x1 and x2 feature plane.",
      view: { az: -125, el: 25, dist: 13.5, target: [0, 0, 0], fov: 38 } }, function (s) {
      stage = s;
      var clip = s.clipBox([-LIMIT, LIMIT], [-LIMIT, LIMIT], [-5, 5]);
      parts = {
        floor: s.grid([-LIMIT, LIMIT], [-LIMIT, LIMIT], 0.5, 0, "muted", 0.2),
        zero: s.quad("muted", { opacity: 0.13 }),
        one: s.quad("c1", { opacity: 0.32, clip: clip }),
        mesh: [], boundary: s.line("accent"),
        label0: s.dots("c2", { radius: 0.075, max: M.samples.length }),
        label1: s.dots("c1", { radius: 0.075, max: M.samples.length }),
        stems0: s.line("c2", { pairs: true, opacity: 0.65 }),
        stems1: s.line("c1", { pairs: true, opacity: 0.65 }),
        probe: s.dots("c3", { radius: 0.13, max: 1 })
      };
      for (var k = -LIMIT; k <= LIMIT + 0.001; k += 0.5) {
        parts.mesh.push({ k: k, x: s.line("c1", { opacity: 0.32 }), y: s.line("c1", { opacity: 0.32 }) });
      }
      s.label("x₁", "ink").set([LIMIT + 0.3, -LIMIT, 0]);
      s.label("x₂", "ink").set([-LIMIT, LIMIT + 0.3, 0]);
      s.label("E₀ = 0", "muted").set([-LIMIT, -LIMIT, 0.2]);
      s.label("E₁ = −η", "c1").set([LIMIT, LIMIT, 0.4]);
      draw();
    });

    function draw() {
      var q = M.probeStats(p, beta);
      gap.set(number(q.eta));
      prob.set(number(q.probability, 3));
      surprise.set(number(q.surprisal, 3));
      if (!stage) return;
      var E = function (x, y) { return -M.score(x, y, p); };
      parts.zero.set([-LIMIT, -LIMIT, 0], [LIMIT, -LIMIT, 0], [LIMIT, LIMIT, 0], [-LIMIT, LIMIT, 0]);
      parts.one.set([-LIMIT, -LIMIT, E(-LIMIT, -LIMIT)], [LIMIT, -LIMIT, E(LIMIT, -LIMIT)],
        [LIMIT, LIMIT, E(LIMIT, LIMIT)], [-LIMIT, LIMIT, E(-LIMIT, LIMIT)]);
      parts.mesh.forEach(function (m) {
        m.x.set([[-LIMIT, m.k, E(-LIMIT, m.k)], [LIMIT, m.k, E(LIMIT, m.k)]]);
        m.y.set([[m.k, -LIMIT, E(m.k, -LIMIT)], [m.k, LIMIT, E(m.k, LIMIT)]]);
      });
      var b = boundary(p);
      parts.boundary.set(b.map(function (q) { return [q[0], q[1], 0.035]; }));
      parts.boundary.show(beta > 0 && b.length > 0);
      var dots = [[], []], stems = [[], []];
      M.samples.forEach(function (row) {
        var x = row[0], y = row[1], label = row[2], e = E(x, y);
        dots[label].push([x, y, label ? e : 0]);
        stems[label].push([x, y, 0], [x, y, e]);
      });
      parts.label0.set(dots[0]); parts.label1.set(dots[1]);
      parts.stems0.set(stems[0]); parts.stems1.set(stems[1]);
      var probe = M.samples[M.probe];
      parts.probe.set([[probe[0], probe[1], E(probe[0], probe[1])]]);
      stage.render();
    }
    draw();
  });

  WK.mount("thermo-loss", function (root) {
    var p = { w1: 0.8, w2: -0.65, b: 0.2 }, flip = false;
    var stage, parts;
    var f = WK.frame({ title: "Cross-entropy over coefficients", badge: "3D · synthetic",
      note: "Height is mean binary cross-entropy in nats for the same ten inputs, at the chosen bias. " +
        "The large dot marks the current (w₁, w₂); its stem reaches the coefficient plane. " +
        "Flipping the highlighted observation changes only its label. The surface is recomputed at each setting." });
    [["w₁", "w1"], ["w₂", "w2"], ["Bias b", "b"]].forEach(function (spec) {
      f.controls.appendChild(control(spec[0], p[spec[1]], spec[1] === "b" ? -1 : -LIMIT,
        spec[1] === "b" ? 1 : LIMIT, function (v) { p[spec[1]] = v; draw(); }).root);
    });
    var toggle = WK.check({ label: "Flip the highlighted label from 1 to 0", value: false,
      onchange: function (v) { flip = v; draw(); } });
    f.controls.appendChild(toggle.root);
    var loss = WK.stat({ label: "Mean BCE at marked coefficients (nats)", value: "" });
    var probe = WK.stat({ label: "Highlighted observation's loss (nats)", value: "" });
    [loss, probe].forEach(function (s) { f.stats.appendChild(s.root); });
    root.appendChild(f.root);

    WKStage.create(f.body, { label: "Mean cross-entropy surface over the w1 and w2 coefficient plane.",
      view: { az: -130, el: 28, dist: 11.5, target: [0, 0, 1.3], fov: 38 } }, function (s) {
      stage = s;
      parts = {
        floor: s.grid([-LIMIT, LIMIT], [-LIMIT, LIMIT], 0.5, 0, "muted", 0.3),
        surface: s.surface(32, 32, { wire: 4, wireOpacity: 0.17, wireToken: "ink" }),
        point: s.dots("c3", { radius: 0.13, max: 1 }),
        stem: s.line("c3", { opacity: 0.85 }),
        ramp: s.ramp(["c1", "c2", "c4"])
      };
      s.label("w₁", "ink").set([LIMIT + 0.3, -LIMIT, 0]);
      s.label("w₂", "ink").set([-LIMIT, LIMIT + 0.3, 0]);
      s.label("mean BCE", "ink").set([-LIMIT, -LIMIT, 3.4]);
      draw();
      s.onTheme(draw);
    });

    function draw() {
      var current = M.loss(p, flip);
      var row = M.samples[M.probe], eta = M.score(row[0], row[1], p);
      var y = flip ? 0 : 1;
      loss.set(number(current, 3));
      probe.set(number(M.softplus(eta) - y * eta, 3));
      if (!stage) return;
      parts.surface.set(function (u, v) {
        var w1 = -LIMIT + 2 * LIMIT * u, w2 = -LIMIT + 2 * LIMIT * v;
        return [w1, w2, M.loss({ w1: w1, w2: w2, b: p.b }, flip)];
      }, function (u, v, at) { return parts.ramp((at[2] - 0.35) / 2.6); });
      parts.point.set([[p.w1, p.w2, current]]);
      parts.stem.set([[p.w1, p.w2, 0], [p.w1, p.w2, current]]);
      stage.render();
    }
    draw();
  });
})();
