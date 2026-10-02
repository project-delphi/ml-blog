/* Widgets for "Collinearity, Part 1". Four 3D scenes on widget-kit's stage
 * (WKStage, three.js) with the kit's frame and controls (WK). Every number
 * comes from model.js (CollinearityModel); this file only draws.
 *
 *   widget-fence   data space: the fitted plane over (x1, x2), refitted on
 *                  fresh noise, pivoting about the line the data lie along
 *   widget-nudge   column space, three observations: the same 2% nudge of y
 *                  along, across, and out of the plane of the columns
 *   widget-valley  coefficient space: the loss surface and where its minimum
 *                  lands over 200 noise draws
 *   widget-three   three columns flattening into one plane
 */
"use strict";

(function () {
  var M = CollinearityModel;

  function sig(x, d) {
    var a = Math.abs(x);
    if (a < 0.005) x = 0;  // no "-0.00"
    if (a >= 1000) return Math.round(x).toLocaleString("en-US");
    if (a >= 100) return x.toFixed(0);
    if (a >= 10) return x.toFixed(1);
    return x.toFixed(d === undefined ? 2 : d);
  }
  function pct(x) { return sig(100 * x, 1) + "%"; }
  function angleLabel(theta) {
    return "θ = " + (theta >= 10 ? theta.toFixed(0) : theta.toFixed(theta >= 2 ? 1 : 2)) + "°";
  }
  function corrLabel(theta) {
    var r = M.corr(theta);
    return r > 0.9995 ? r.toFixed(5) : r > 0.995 ? r.toFixed(4) : r.toFixed(3);
  }
  // The angle control: a log-scale slider from 90° down to 0.5°, and presets
  // at the correlations the post quotes, so a reader lands on them exactly.
  var PRESETS = [
    { label: "r = 0", theta: 90 }, { label: "0.9", theta: Math.acos(0.9) * 180 / Math.PI },
    { label: "0.99", theta: Math.acos(0.99) * 180 / Math.PI },
    { label: "0.999", theta: Math.acos(0.999) * 180 / Math.PI }, { label: "θ = 0.5°", theta: 0.5 }];
  function angleControl(label, start, onchange) {
    var slider = WK.slider({ label: label, min: 0, max: 100, step: 0.1, value: M.sliderOf(start),
      fmt: function (v) { return angleLabel(M.thetaOf(v)) + "  ·  r = " + corrLabel(M.thetaOf(v)); },
      oninput: function (v) { presets.set(null, true); onchange(M.thetaOf(v)); } });
    var presets = WK.toggle({ label: "Jump to a correlation", value: start,
      options: PRESETS.map(function (p) { return { value: p.theta, label: p.label }; }),
      onchange: function (theta) { slider.set(M.sliderOf(theta), true); onchange(theta); } });
    presets.root.style.flex = "0 1 auto";
    return [slider.root, presets.root];
  }
  function button(text, onclick) {
    var b = WK.el("button", { type: "button", class: "widget-toggle-btn", text: text, onclick: onclick });
    return WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } },
      WK.el("div", { class: "widget-toggle" }, b));
  }
  function lerp(a, b, t) { return a + (b - a) * t; }

  // ---------------------------------------------------------------- fence
  WK.mount("widget-fence", function (root) {
    var SHOWN = 25, L = 2.8, H = 2.4, ZS = 0.5;  // L covers every row: max |x| over the slider is 2.77
    var state = { theta: PRESETS[2].theta, k: 0, fan: true };
    var stage = null, parts = null, cancel = null;
    // What is on screen for the highlighted fit; tweened when the noise changes.
    var shown = { beta: M.fit(state.theta, 0), noise: M.NOISE[0].e.slice() };

    var f = WK.frame({
      title: "A plane balanced on a fence",
      note: "40 synthetic rows: y = x₁ + x₂ + noise, the two predictors at correlation r. Amber dots are " +
        "one sample and the solid sheet is its least-squares plane. The faint sheets are the planes fitted " +
        "to 24 other noise draws on the same x values. The two vertical lines mark where each plane " +
        "predicts y at (1.5, 1.5), where the predictors agree, and at (1.5, −1.5), where they disagree. " +
        "Stats are over 200 noise draws."
    });
    angleControl("Angle between the predictor columns", state.theta,
      function (theta) { state.theta = theta; settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    f.controls.appendChild(button("New noise", function () { renoise(); }));
    var fan = WK.check({ label: "Show the 24 other fits", value: true,
      onchange: function (v) { state.fan = v; draw(); } });
    f.controls.appendChild(WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } },
      fan.root));

    var stats = {
      fit: WK.stat({ label: "this fit's coefficients", value: "" }),
      kappa: WK.stat({ label: "condition number κ", value: "" }),
      b1: WK.stat({ label: "sd of the x₁ coefficient", value: "" }),
      sum: WK.stat({ label: "sd of the two coefficients' sum", value: "" }),
      on: WK.stat({ label: "sd of the prediction at (1.5, 1.5)", value: "" }),
      off: WK.stat({ label: "sd of the prediction at (1.5, −1.5)", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "A 3D scatter of y against two predictors with fitted planes through it.",
      view: { az: -122, el: 13, dist: 14.6, target: [0, 0, 0.05] }
    }, function (s) {
      stage = s;
      var clip = s.clipBox([-L, L], [-L, L], [-H, H]);
      parts = { fan: [], floor: s.grid([-L, L], [-L, L], L / 4, -H, "muted", 0.28) };
      for (var i = 0; i < SHOWN; i++) parts.fan.push(s.quad("c1", { opacity: 0.04, clip: clip }));
      parts.plane = s.quad("c1", { opacity: 0.34, clip: clip });
      // The rows are never clipped: one that pokes out of the box is still data.
      parts.points = s.dots("c3", { radius: 0.055, max: M.N });
      parts.shadow = s.dots("muted", { radius: 0.035, max: M.N, opacity: 0.55 });
      parts.frame = s.line("muted", { pairs: true, opacity: 0.45 });
      parts.frame.set([[-L, -L, -H], [-L, -L, H], [L, -L, -H], [L, -L, H], [L, L, -H], [L, L, H], [-L, L, -H], [-L, L, H],
        [-L, -L, H], [L, -L, H], [L, -L, H], [L, L, H], [L, L, H], [-L, L, H], [-L, L, H], [-L, -L, H]]);
      parts.onLine = s.line("c2", { opacity: 0.6 });
      parts.offLine = s.line("c4", { opacity: 0.6 });
      parts.onLine.set([[M.QUERY.on[0], M.QUERY.on[1], -H], [M.QUERY.on[0], M.QUERY.on[1], H]]);
      parts.offLine.set([[M.QUERY.off[0], M.QUERY.off[1], -H], [M.QUERY.off[0], M.QUERY.off[1], H]]);
      parts.onDots = s.dots("c2", { radius: 0.045, max: SHOWN + 1, clip: clip });
      parts.offDots = s.dots("c4", { radius: 0.045, max: SHOWN + 1, clip: clip });
      s.label("x₁", "muted").set([0, -L - 0.4, -H]);
      s.label("x₂", "muted").set([-L - 0.4, 0, -H]);
      s.label("y", "muted").set([-L, -L, H + 0.25]);
      s.label("agree", "c2").set([M.QUERY.on[0], M.QUERY.on[1], H + 0.25]);
      s.label("disagree", "c4").set([M.QUERY.off[0], M.QUERY.off[1], H + 0.25]);
      // The rail every plane shares: y = 2t above the diagonal x1 = x2 = t.
      parts.rail = s.line("ink", { clip: clip });
      parts.rail.set([[-L, -L, ZS * M.predict(M.BETA, [-L, -L])], [L, L, ZS * M.predict(M.BETA, [L, L])]]);
      draw();
    });

    function corners(b) {
      return [[-L, -L], [L, -L], [L, L], [-L, L]].map(function (c) {
        return [c[0], c[1], ZS * M.predict(b, c)];
      });
    }
    function settle() {
      if (cancel) { cancel(); cancel = null; }
      shown.beta = M.fit(state.theta, state.k);
      shown.noise = M.NOISE[state.k].e.slice();
    }
    function renoise() {
      var from = { beta: shown.beta.slice(), noise: shown.noise.slice() };
      state.k = (state.k + 1) % SHOWN;
      var to = { beta: M.fit(state.theta, state.k), noise: M.NOISE[state.k].e };
      if (cancel) cancel();
      stats.fit.set("(" + sig(to.beta[0]) + ", " + sig(to.beta[1]) + ")");
      if (!stage) { settle(); return; }
      cancel = stage.animate(650, function (t) {
        shown.beta = [lerp(from.beta[0], to.beta[0], t), lerp(from.beta[1], to.beta[1], t)];
        shown.noise = from.noise.map(function (e, i) { return lerp(e, to.noise[i], t); });
        scene();
      });
    }
    function scene() {
      if (!stage) return;
      var X = M.columns(state.theta), i, q;
      for (i = 0; i < SHOWN; i++) {
        q = corners(M.fit(state.theta, i));
        parts.fan[i].set(q[0], q[1], q[2], q[3]);
        parts.fan[i].show(state.fan && i !== state.k);
      }
      q = corners(shown.beta);
      parts.plane.set(q[0], q[1], q[2], q[3]);
      var pts = [], sh = [];
      for (i = 0; i < M.N; i++) {
        pts.push([X.x1[i], X.x2[i], ZS * (M.BETA[0] * X.x1[i] + M.BETA[1] * X.x2[i] + shown.noise[i])]);
        sh.push([X.x1[i], X.x2[i], -H]);
      }
      parts.points.set(pts);
      parts.shadow.set(sh);
      var on = [[M.QUERY.on[0], M.QUERY.on[1], ZS * M.predict(shown.beta, M.QUERY.on)]];
      var off = [[M.QUERY.off[0], M.QUERY.off[1], ZS * M.predict(shown.beta, M.QUERY.off)]];
      if (state.fan) for (i = 0; i < SHOWN; i++) {
        var b = M.fit(state.theta, i);
        on.push([M.QUERY.on[0], M.QUERY.on[1], ZS * M.predict(b, M.QUERY.on)]);
        off.push([M.QUERY.off[0], M.QUERY.off[1], ZS * M.predict(b, M.QUERY.off)]);
      }
      parts.onDots.set(on);
      parts.offDots.set(off);
      stage.render();
    }
    function draw() {
      var s = M.spread(state.theta, M.DRAWS);
      var b = M.fit(state.theta, state.k);
      stats.fit.set("(" + sig(b[0]) + ", " + sig(b[1]) + ")");
      stats.kappa.set(sig(M.kappa(state.theta)));
      stats.b1.set(sig(s.b1));
      stats.sum.set(sig(s.sum));
      stats.on.set(sig(s.on));
      stats.off.set(sig(s.off));
      scene();
    }
    draw();
  });

  // ---------------------------------------------------------------- nudge
  WK.mount("widget-nudge", function (root) {
    var state = { theta: PRESETS[2].theta, which: "none" };
    var stage = null, parts = null, cancel = null;
    var shownY = M.Y0.slice();

    var f = WK.frame({
      title: "Which 2%",
      note: "Three observations, so the two predictor columns and y are arrows in 3D. The columns x₁ and x₂ " +
        "have length 1 and span the floor. The fitted vector ŷ is the shadow of y on the floor, and the " +
        "coefficients are the route to it: β₁ steps along x₁, then β₂ steps along x₂. The buttons move y by " +
        "2% of its length in one of three directions."
    });
    angleControl("Angle between the columns", state.theta,
      function (theta) { state.theta = theta; settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    var which = WK.toggle({ label: "Move y by 2% of its length", value: "none",
      options: [{ value: "none", label: "no nudge" }, { value: "strong", label: "along the columns" },
        { value: "weak", label: "across them" }, { value: "out", label: "out of the floor" }],
      onchange: function (v) { state.which = v; move(); } });
    f.controls.appendChild(which.root);
    var stats = {
      kappa: WK.stat({ label: "condition number κ", value: "" }),
      beta: WK.stat({ label: "coefficients (β₁, β₂)", value: "" }),
      dbeta: WK.stat({ label: "change in the coefficients", value: "" }),
      dfit: WK.stat({ label: "change in the fitted vector ŷ", value: "" }),
      drss: WK.stat({ label: "change in the squared residual", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "Two predictor columns in a plane, the target y above it, and the fitted vector on the plane.",
      view: { az: -96, el: 46, dist: 3.4, target: [0.7, 0.12, 0.2] }
    }, function (s) {
      stage = s;
      parts = {
        grid: s.grid([-4, 5], [-1.5, 1.5], 0.5, 0, "muted", 0.22),
        x1: s.arrow("c2", { radius: 0.022 }), x2: s.arrow("c1", { radius: 0.022 }),
        y: s.arrow("c3", { radius: 0.018 }), fit: s.arrow("ink", { radius: 0.012 }),
        step1: s.arrow("c2", { radius: 0.011 }), step2: s.arrow("c1", { radius: 0.011 }),
        drop: s.line("muted", { dashed: true, dash: 0.05, gap: 0.04 }),
        ghost: s.line("c3", { dashed: true, dash: 0.05, gap: 0.05, opacity: 0.6 }),
        lx1: s.label("x₁", "c2"), lx2: s.label("x₂", "c1"), ly: s.label("y", "c3"),
        lfit: s.label("ŷ", "ink", { dy: 14 }), l1: s.label("", "c2", { dy: -12 }), l2: s.label("", "c1", { dy: 12 })
      };
      parts.ghost.set([[0, 0, 0], M.Y0]);
      draw();
    });

    function settle() {
      if (cancel) { cancel(); cancel = null; }
      shownY = M.nudge(state.theta, state.which).y;
    }
    function move() {
      var from = shownY.slice(), to = M.nudge(state.theta, state.which).y;
      if (cancel) cancel();
      stats_();
      if (!stage) { shownY = to; return; }
      cancel = stage.animate(700, function (t) {
        shownY = from.map(function (v, i) { return lerp(v, to[i], t); });
        scene();
      });
    }
    function scene() {
      if (!stage) return;
      var r = M.nudge(state.theta, state.which), p = M.project(state.theta, shownY);
      var a = [p.beta[0], 0, 0];  // the corner of the route: beta1 * x1
      parts.x1.set([0, 0, 0], r.x1);
      parts.x2.set([0, 0, 0], r.x2);
      parts.y.set([0, 0, 0], shownY);
      parts.fit.set([0, 0, 0], p.fitted);
      parts.step1.set([0, 0, 0.03], [a[0], 0, 0.03]);
      parts.step2.set([a[0], 0, 0.06], [p.fitted[0], p.fitted[1], 0.06]);
      parts.drop.set([p.fitted, shownY]);
      parts.lx1.set([1.12, -0.1, 0]);
      parts.lx2.set([r.x2[0] * 1.12, r.x2[1] * 1.12 + 0.1, 0]);
      parts.ly.set([shownY[0], shownY[1], shownY[2] + 0.14]);
      parts.lfit.set(p.fitted);
      var far = Math.abs(p.beta[1]) > 0.05;
      parts.l1.show(far); parts.l2.show(far);
      parts.l1.set([a[0] / 2, -0.02, 0], "β₁x₁ = " + sig(p.beta[0]) + "·x₁");
      parts.l2.set([(a[0] + p.fitted[0]) / 2, p.fitted[1] / 2 + 0.02, 0], "β₂x₂ = " + sig(p.beta[1]) + "·x₂");
      // Pull the camera back as the route leaves the unit-sized scene.
      var reach = Math.max(1.5, Math.abs(p.beta[0]), Math.abs(p.beta[1]));
      stage.view.dist = 3.4 + 2.0 * (reach - 1.5);
      stage.view.target = [0.7 - 0.5 * (reach - 1.5), 0.12, 0.2];
      stage.render();
    }
    function stats_() {
      var r = M.nudge(state.theta, state.which);
      stats.kappa.set(sig(M.kappa(state.theta)));
      stats.beta.set("(" + sig(r.beta[0]) + ", " + sig(r.beta[1]) + ")");
      stats.dbeta.set(pct(r.betaChange));
      stats.dfit.set(pct(r.fittedChange));
      stats.drss.set(pct(r.rssChange));
    }
    function draw() { stats_(); scene(); }
    draw();
  });

  // --------------------------------------------------------------- valley
  WK.mount("widget-valley", function (root) {
    var R = 4.5, CAP = 4, HS = 0.6, LIFT = 0.05, NU = 96, NV = 28, SPAN = 6.4;
    var state = { theta: PRESETS[2].theta, k: 0 };
    var stage = null, parts = null, cancel = null;
    var shown = M.fit(state.theta, 0);

    var f = WK.frame({
      title: "The loss surface grows a valley",
      note: "The surface is the mean squared error of every candidate pair (β₁, β₂) on one sample, measured " +
        "from its minimum (amber) and cut off at a height of 4. Each grey dot is the minimum of another " +
        "noise draw, placed on this sample's surface, so its height is what this sample thinks of that " +
        "answer. The dark dot is the pair that generated the data, (1, 1)."
    });
    angleControl("Angle between the predictor columns", state.theta,
      function (theta) { state.theta = theta; settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    f.controls.appendChild(button("New noise", function () { renoise(); }));
    var stats = {
      at: WK.stat({ label: "this draw's minimum", value: "" }),
      kappa: WK.stat({ label: "condition number κ", value: "" }),
      across: WK.stat({ label: "curvature across the valley", value: "" }),
      along: WK.stat({ label: "curvature along the valley", value: "" }),
      b1: WK.stat({ label: "sd of the x₁ coefficient", value: "" }),
      sum: WK.stat({ label: "sd of the two coefficients' sum", value: "" }),
      min: WK.stat({ label: "minimum loss, this draw", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    // Display coordinates: coefficients relative to the truth, loss scaled.
    function D(b, h) { return [b[0] - M.BETA[0], b[1] - M.BETA[1], h]; }

    WKStage.create(f.body, {
      label: "A 3D loss surface over the two coefficients, with the fitted coefficients of 200 samples on the floor.",
      view: { az: -58, el: 34, dist: 18, target: [0, 0, -0.1] }
    }, function (s) {
      stage = s;
      var clip = s.clipBox([-R, R], [-R, R], [-0.05, CAP * HS]);
      var dotClip = s.clipBox([-R, R], [-R, R], [-0.05, CAP * HS + 0.2]);
      parts = {
        grid: s.grid([-R, R], [-R, R], 1.5, -0.03, "muted", 0.3),
        surface: s.surface(NU, NV, { clip: clip, wire: 4, wireOpacity: 0.22 }),
        cloud: s.dots("muted", { radius: 0.055, max: M.DRAWS, clip: dotClip }),
        here: s.dots("c3", { radius: 0.12, max: 1, clip: dotClip }),
        truth: s.dots("ink", { radius: 0.09, max: 1, clip: dotClip }),
        ramp: s.ramp(["c3", "c2", "c1"]),
        lmin: s.label("", "c3", { dy: 22 }),
        ltruth: s.label("truth (1, 1)", "ink", { dy: -22 })
      };
      s.label("β₁", "muted").set([0, -R - 0.5, 0]);
      s.label("β₂", "muted").set([R + 0.5, 0, 0]);
      s.label("loss", "muted").set([-R, -R, CAP * HS + 0.3]);
      s.onTheme(function () { scene(); });
      draw();
    });

    function settle() {
      if (cancel) { cancel(); cancel = null; }
      shown = M.fit(state.theta, state.k);
    }
    function renoise() {
      var from = shown.slice();
      state.k = (state.k + 1) % M.DRAWS;
      var to = M.fit(state.theta, state.k);
      if (cancel) cancel();
      stats_();
      if (!stage) { shown = to; return; }
      cancel = stage.animate(650, function (t) {
        shown = [lerp(from[0], to[0], t), lerp(from[1], to[1], t)];
        scene();
      });
    }
    function scene() {
      if (!stage) return;
      var c = M.corr(state.theta), q = Math.SQRT1_2, b = shown;
      // The grid runs along the valley's own axes, v1 = (1, 1)/sqrt2 across
      // it and v2 = (-1, 1)/sqrt2 along it, so the trough never cuts cells
      // diagonally.
      parts.surface.set(function (u, v) {
        var t1 = (2 * u - 1) * SPAN, t2 = (2 * v - 1) * SPAN;
        var b1 = M.BETA[0] + q * (t1 - t2), b2 = M.BETA[1] + q * (t1 + t2);
        var d1 = b1 - b[0], d2 = b2 - b[1];
        var loss = d1 * d1 + 2 * c * d1 * d2 + d2 * d2;
        return D([b1, b2], HS * Math.min(loss, CAP * 1.5));
      }, function (u, v, p) { return parts.ramp(p[2] / (CAP * HS)); });
      // This sample's loss, above its minimum, at any coefficient pair.
      function height(p) {
        var d1 = p[0] - b[0], d2 = p[1] - b[1];
        return HS * (d1 * d1 + 2 * c * d1 * d2 + d2 * d2) + LIFT;
      }
      var cloud = [];
      for (var k = 0; k < M.DRAWS; k++) {
        var bk = M.fit(state.theta, k);
        cloud.push(D(bk, height(bk)));
      }
      parts.cloud.set(cloud);
      parts.here.set([D(b, LIFT)]);
      var hTruth = height(M.BETA);
      parts.truth.set([D(M.BETA, hTruth)]);
      parts.ltruth.show(hTruth < CAP * HS);
      parts.ltruth.set(D(M.BETA, hTruth));
      var inside = Math.abs(b[0] - M.BETA[0]) < R && Math.abs(b[1] - M.BETA[1]) < R;
      parts.lmin.show(inside);
      parts.lmin.set(D(b, LIFT), "(" + sig(b[0]) + ", " + sig(b[1]) + ")");
      stage.render();
    }
    function stats_() {
      var s = M.spread(state.theta, M.DRAWS), sv = M.singular(state.theta), b = M.fit(state.theta, state.k);
      stats.at.set("(" + sig(b[0]) + ", " + sig(b[1]) + ")");
      stats.kappa.set(sig(M.kappa(state.theta)));
      stats.across.set(sig(2 * sv[0] * sv[0]));
      stats.along.set((2 * sv[1] * sv[1]).toPrecision(2));
      stats.b1.set(sig(s.b1));
      stats.sum.set(sig(s.sum));
      stats.min.set(M.sample(state.theta, state.k).mse.toFixed(3));
    }
    function draw() { stats_(); scene(); }
    draw();
  });

  // ---------------------------------------------------------------- three
  WK.mount("widget-three", function (root) {
    var state = { alpha: 45 };
    var stage = null, parts = null;
    var f = WK.frame({
      title: "Three columns, no two alike",
      note: "x₁ and x₂ sit at a right angle on the floor. x₃ points above their diagonal, lifted off the floor " +
        "by the slider's angle. The box they span has volume 1 when all three are at right angles and " +
        "volume 0 when x₃ lies in the floor, where x₃ = (x₁ + x₂)/√2 and the three columns are dependent."
    });
    var lift = WK.slider({ label: "Lift of x₃ out of the plane of x₁ and x₂", min: 0, max: 100, step: 0.5,
      value: M.sliderOf(state.alpha),
      fmt: function (v) { var a = M.thetaOf(v); return (a >= 10 ? a.toFixed(0) : a.toFixed(a >= 2 ? 1 : 2)) + "°"; },
      oninput: function (v) { state.alpha = M.thetaOf(v); draw(); } });
    f.controls.appendChild(lift.root);
    var stats = {
      pair: WK.stat({ label: "largest pairwise correlation", value: "" }),
      kappa: WK.stat({ label: "condition number κ", value: "" }),
      vol: WK.stat({ label: "volume of the box", value: "" }),
      vif3: WK.stat({ label: "VIF of x₃", value: "" }),
      vif1: WK.stat({ label: "VIF of x₁ and of x₂", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "Three unit vectors and the box they span, flattening as the third drops into the plane of the other two.",
      view: { az: -112, el: 20, dist: 4.2, target: [0.6, 0.6, 0.35] }
    }, function (s) {
      stage = s;
      parts = {
        grid: s.grid([-0.5, 2], [-0.5, 2], 0.5, 0, "muted", 0.22),
        faces: [0, 1, 2, 3, 4, 5].map(function () { return s.quad("c5", { opacity: 0.13 }); }),
        edges: s.line("ink", { pairs: true, opacity: 0.55 }),
        x1: s.arrow("c2"), x2: s.arrow("c1"), x3: s.arrow("c3"),
        drop: s.line("muted", { dashed: true, dash: 0.04, gap: 0.035 }),
        l1: s.label("x₁", "c2"), l2: s.label("x₂", "c1"), l3: s.label("x₃", "c3", { dy: -12 })
      };
      draw();
    });

    function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
    function draw() {
      var r = M.three(state.alpha);
      stats.pair.set(r.pairwise.toFixed(3));
      stats.kappa.set(sig(r.kappa));
      stats.vol.set(r.volume.toFixed(r.volume < 0.1 ? 4 : 2));
      stats.vif3.set(sig(r.vif[2], 1));
      stats.vif1.set(sig(r.vif[0], 1));
      if (!stage) return;
      var o = [0, 0, 0], a = r.x1, b = r.x2, c = r.x3;
      var ab = add(a, b), ac = add(a, c), bc = add(b, c), abc = add(ab, c);
      parts.x1.set(o, a); parts.x2.set(o, b); parts.x3.set(o, c);
      [[o, a, ab, b], [c, ac, abc, bc], [o, a, ac, c], [b, ab, abc, bc], [o, b, bc, c], [a, ab, abc, ac]]
        .forEach(function (q, i) { parts.faces[i].set(q[0], q[1], q[2], q[3]); });
      parts.edges.set([o, a, o, b, o, c, a, ab, a, ac, b, ab, b, bc, c, ac, c, bc, ab, abc, ac, abc, bc, abc]);
      parts.drop.set([c, [c[0], c[1], 0]]);
      parts.l1.set([1.13, 0, 0]); parts.l2.set([0, 1.13, 0]);
      parts.l3.set(c);
      stage.render();
    }
    draw();
  });
})();
