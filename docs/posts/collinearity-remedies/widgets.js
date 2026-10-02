/* Widgets for "Collinearity, Part 2". Two 3D scenes on widget-kit's stage
 * (WKStage, three.js) with the kit's frame and controls (WK). Every number
 * comes from model.js (RemedyModel); this file only draws.
 *
 *   widget-remedy-valley  coefficient space: the surface each method
 *                         minimises, and where its answer lands over 200
 *                         noise draws
 *   widget-remedy-fence   data space: the planes each method fits, and what
 *                         they predict where the predictors agree and where
 *                         they disagree
 */
"use strict";

(function () {
  var M = RemedyModel;

  function sig(x, d) {
    var a = Math.abs(x);
    if (a < 0.005) x = 0;  // no "-0.00"
    if (a >= 1000) return Math.round(x).toLocaleString("en-US");
    if (a >= 100) return x.toFixed(0);
    if (a >= 10) return x.toFixed(1);
    return x.toFixed(d === undefined ? 2 : d);
  }
  function pair(b) { return "(" + sig(b[0]) + ", " + sig(b[1]) + ")"; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function angleLabel(theta) {
    return "θ = " + (theta >= 10 ? theta.toFixed(0) : theta.toFixed(theta >= 2 ? 1 : 2)) + "°";
  }
  function corrLabel(theta) {
    var r = M.corr(theta);
    return r > 0.9995 ? r.toFixed(5) : r > 0.995 ? r.toFixed(4) : r.toFixed(3);
  }
  function lambdaLabel(lambda) { return "λ = " + lambda.toPrecision(2); }

  // The same angle control as Part 1: a log-scale slider and presets at the
  // correlations the post quotes.
  var PRESETS = [
    { label: "r = 0", theta: 90 }, { label: "0.9", theta: Math.acos(0.9) * 180 / Math.PI },
    { label: "0.99", theta: Math.acos(0.99) * 180 / Math.PI },
    { label: "0.999", theta: Math.acos(0.999) * 180 / Math.PI }];
  function angleControl(start, onchange) {
    var slider = WK.slider({ label: "Angle between the predictor columns", min: 0, max: 100, step: 0.1,
      value: M.sliderOf(start),
      fmt: function (v) { return angleLabel(M.thetaOf(v)) + "  ·  r = " + corrLabel(M.thetaOf(v)); },
      oninput: function (v) { presets.set(null, true); onchange(M.thetaOf(v)); } });
    var presets = WK.toggle({ label: "Jump to a correlation", value: start,
      options: PRESETS.map(function (p) { return { value: p.theta, label: p.label }; }),
      onchange: function (theta) { slider.set(M.sliderOf(theta), true); onchange(theta); } });
    presets.root.style.flex = "0 1 auto";
    return [slider.root, presets.root];
  }
  // The method toggle and the penalty slider, which greys out for the two
  // methods that have no penalty weight.
  function methodControl(state, onchange) {
    var lambda = WK.slider({ label: "Penalty weight", min: 0, max: M.LAMBDA_STOPS, step: 1,
      value: M.lambdaStop(state.lambda), fmt: function (v) { return lambdaLabel(M.lambdaOf(v)); },
      oninput: function (v) { state.lambda = M.lambdaOf(v); onchange(); } });
    function sync() {
      var on = M.METHODS[state.method].lambda;
      lambda.input.disabled = !on;
      lambda.root.style.opacity = on ? "1" : "0.45";
    }
    var method = WK.toggle({ label: "Method", value: state.method,
      options: Object.keys(M.METHODS).map(function (k) { return { value: k, label: M.METHODS[k].label }; }),
      onchange: function (v) { state.method = v; sync(); onchange(); } });
    method.root.style.flex = "1 1 100%";
    sync();
    return [method.root, lambda.root];
  }
  function button(text, onclick) {
    var b = WK.el("button", { type: "button", class: "widget-toggle-btn", text: text, onclick: onclick });
    return WK.el("div", { class: "widget-control", style: { flex: "0 0 auto", alignSelf: "flex-end" } },
      WK.el("div", { class: "widget-toggle" }, b));
  }

  // --------------------------------------------------------------- valley
  WK.mount("widget-remedy-valley", function (root) {
    var R = 4.5, CAP = 4, HS = 0.6, LIFT = 0.05, NU = 96, NV = 28, SPAN = 6.4, CENTRE = [1, 1];
    var state = { theta: PRESETS[2].theta, k: 0, method: "ols", lambda: 0.1 };
    var stage = null, parts = null, cancel = null;
    // Least squares for the sample on screen; tweened when the noise changes.
    var shown = M.ols(state.theta, 0);

    var f = WK.frame({
      title: "What each method minimises",
      note: "The surface is what the chosen method minimises on one sample: the mean squared error plus its " +
        "penalty, measured from its lowest point and cut off at a height of 4. Amber is the method's answer " +
        "for this sample and pink is plain least squares on the same sample. Each grey dot is the method's " +
        "answer on another of the 200 noise draws. The dark dot is the pair that generated the data, (1.5, 0.5)."
    });
    angleControl(state.theta, function (theta) { state.theta = theta; settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    methodControl(state, function () { settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    f.controls.appendChild(button("New noise", function () { renoise(); }));
    var stats = {
      ols: WK.stat({ label: "least squares, this sample", value: "" }),
      est: WK.stat({ label: "this method, this sample", value: "" }),
      centre: WK.stat({ label: "average answer over 200 samples", value: "" }),
      sd: WK.stat({ label: "sd of the x₁ coefficient", value: "" }),
      dist: WK.stat({ label: "typical distance from the truth", value: "" }),
      zero: WK.stat({ label: "samples with x₁ / x₂ dropped", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    function D(b, h) { return [b[0] - CENTRE[0], b[1] - CENTRE[1], h]; }

    WKStage.create(f.body, {
      label: "A 3D surface over the two coefficients showing the penalised loss, with each method's answers on it.",
      view: { az: -58, el: 34, dist: 18, target: [0, 0, -0.1] }
    }, function (s) {
      stage = s;
      var clip = s.clipBox([-R, R], [-R, R], [-0.05, CAP * HS]);
      var dotClip = s.clipBox([-R, R], [-R, R], [-0.05, CAP * HS + 0.2]);
      parts = {
        grid: s.grid([-R, R], [-R, R], 1.5, -0.03, "muted", 0.3),
        surface: s.surface(NU, NV, { clip: clip, wire: 4, wireOpacity: 0.22 }),
        axis1: s.line("ink", { clip: dotClip, opacity: 0.75 }),
        axis2: s.line("ink", { clip: dotClip, opacity: 0.75 }),
        equal: s.line("c3", { clip: dotClip }),
        cloud: s.dots("muted", { radius: 0.055, max: M.DRAWS, clip: dotClip }),
        olsDot: s.dots("c4", { radius: 0.1, max: 1, clip: dotClip }),
        pull: s.line("c4", { clip: dotClip }),
        here: s.dots("c3", { radius: 0.12, max: 1, clip: dotClip }),
        truth: s.dots("ink", { radius: 0.09, max: 1, clip: dotClip }),
        ramp: s.ramp(["c3", "c2", "c1"]),
        lhere: s.label("", "c3", { dy: 22 }),
        ltruth: s.label("truth (1.5, 0.5)", "ink", { dy: -22 }),
        l0a: s.label("β₂ = 0", "ink"), l0b: s.label("β₁ = 0", "ink")
      };
      s.label("β₁", "muted").set([0, -R - 0.5, 0]);
      s.label("β₂", "muted").set([R + 0.5, 0, 0]);
      s.label("height", "muted").set([-R, -R, CAP * HS + 0.3]);
      s.onTheme(function () { scene(); });
      draw();
    });

    function settle() {
      if (cancel) { cancel(); cancel = null; }
      shown = M.ols(state.theta, state.k);
    }
    function renoise() {
      var from = shown.slice();  // what is on screen, so a second click mid-tween does not jump
      state.k = (state.k + 1) % M.DRAWS;
      var to = M.ols(state.theta, state.k);
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
      var q = Math.SQRT1_2, m = M.METHODS[state.method], bhat = shown;
      var est = M.estimateFrom(bhat, state.theta, state.method, state.lambda);
      var value = M.objectiveFrom(bhat, state.theta, state.method, state.lambda);
      // For one component the answer is the lowest point on the line
      // beta1 = beta2, not of the surface, so heights are measured from the
      // surface's own minimum (least squares) and the line is drawn on it.
      var base = state.method === "pcr" ? 0 : value(est);
      function height(p) { return HS * (value(p) - base) + LIFT; }
      parts.surface.set(function (u, v) {
        var t1 = (2 * u - 1) * SPAN, t2 = (2 * v - 1) * SPAN;
        var p = [CENTRE[0] + q * (t1 - t2), CENTRE[1] + q * (t1 + t2)];
        return D(p, Math.min(HS * (value(p) - base), CAP * HS * 1.5));
      }, function (u, v, p) { return parts.ramp(p[2] / (CAP * HS)); });
      function path(fn) {
        var pts = [], i;
        for (i = 0; i <= 80; i++) { var p = fn(-3.5 + 9 * i / 80); pts.push(D(p, height(p))); }
        return pts;
      }
      var showAxes = m.l1 > 0;
      parts.axis1.show(showAxes); parts.axis2.show(showAxes);
      parts.l0a.show(showAxes); parts.l0b.show(showAxes);
      if (showAxes) {
        parts.axis1.set(path(function (t) { return [t, 0]; }));
        parts.axis2.set(path(function (t) { return [0, t]; }));
        parts.l0a.set(D([4.6, 0], 0.3)); parts.l0b.set(D([0, 4.6], 0.3));
      }
      parts.equal.show(state.method === "pcr");
      if (state.method === "pcr") parts.equal.set(path(function (t) { return [t, t]; }));
      var cloud = [], k;
      for (k = 0; k < M.DRAWS; k++) {
        var bk = M.estimate(state.theta, k, state.method, state.lambda);
        cloud.push(D(bk, height(bk)));
      }
      parts.cloud.set(cloud);
      parts.here.set([D(est, height(est))]);
      var moved = state.method !== "ols";
      parts.olsDot.show(moved); parts.pull.show(moved);
      if (moved) {
        parts.olsDot.set([D(bhat, height(bhat))]);
        var pull = [], i;
        for (i = 0; i <= 24; i++) {
          var w = [lerp(bhat[0], est[0], i / 24), lerp(bhat[1], est[1], i / 24)];
          pull.push(D(w, height(w)));
        }
        parts.pull.set(pull);
      }
      var hTruth = height(M.BETA);
      parts.truth.set([D(M.BETA, hTruth)]);
      parts.ltruth.show(hTruth < CAP * HS);
      parts.ltruth.set(D(M.BETA, hTruth));
      var inside = Math.abs(est[0] - CENTRE[0]) < R && Math.abs(est[1] - CENTRE[1]) < R;
      parts.lhere.show(inside);
      parts.lhere.set(D(est, height(est)), pair(est));
      stage.render();
    }
    function stats_() {
      var s = M.summary(state.theta, state.method, state.lambda);
      stats.ols.set(pair(M.ols(state.theta, state.k)));
      stats.est.set(pair(M.estimate(state.theta, state.k, state.method, state.lambda)));
      stats.centre.set(pair(s.mean));
      stats.sd.set(sig(s.sd[0]));
      stats.dist.set(sig(s.rmse));
      stats.zero.set(s.zero1 + " / " + s.zero2);
    }
    function draw() { stats_(); scene(); }
    draw();
  });

  // ---------------------------------------------------------------- fence
  WK.mount("widget-remedy-fence", function (root) {
    var SHOWN = 25, L = 2.8, H = 2.4, ZS = 0.5;  // L covers every row: max |x| over the slider is 2.77
    var state = { theta: PRESETS[2].theta, k: 0, method: "ols", lambda: 0.1 };
    var stage = null, parts = null, cancel = null;
    var shown = { beta: M.estimate(state.theta, 0, state.method, state.lambda), noise: M.NOISE[0].e.slice() };

    var f = WK.frame({
      title: "The same planes, held",
      note: "Part 1's first scene with a method switch. 40 synthetic rows, y = 1.5·x₁ + 0.5·x₂ + noise. The " +
        "solid sheet is the chosen method's plane for one sample and the faint sheets are its planes for 24 " +
        "other noise draws. Dots on the two vertical lines are each plane's prediction at (1.5, 1.5) and at " +
        "(1.5, −1.5); the dark dot on each line is the true value. Stats are over 200 noise draws."
    });
    angleControl(state.theta, function (theta) { state.theta = theta; settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    methodControl(state, function () { settle(); draw(); })
      .forEach(function (node) { f.controls.appendChild(node); });
    f.controls.appendChild(button("New noise", function () { renoise(); }));
    var stats = {
      fit: WK.stat({ label: "this fit's coefficients", value: "" }),
      onErr: WK.stat({ label: "typical error at (1.5, 1.5)", value: "" }),
      offMean: WK.stat({ label: "average prediction at (1.5, −1.5); truth 1.50", value: "" }),
      offSd: WK.stat({ label: "sd of the prediction at (1.5, −1.5)", value: "" }),
      offErr: WK.stat({ label: "typical error at (1.5, −1.5)", value: "" })
    };
    Object.keys(stats).forEach(function (k) { f.stats.appendChild(stats[k].root); });
    root.appendChild(f.root);

    WKStage.create(f.body, {
      label: "A 3D scatter of y against two predictors with the planes one method fits to 25 samples.",
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
      parts.truth = s.dots("ink", { radius: 0.085, max: 2 });
      parts.truth.set([[M.QUERY.on[0], M.QUERY.on[1], ZS * M.predict(M.BETA, M.QUERY.on)],
        [M.QUERY.off[0], M.QUERY.off[1], ZS * M.predict(M.BETA, M.QUERY.off)]]);
      s.label("x₁", "muted").set([0, -L - 0.4, -H]);
      s.label("x₂", "muted").set([-L - 0.4, 0, -H]);
      s.label("y", "muted").set([-L, -L, H + 0.25]);
      s.label("agree", "c2").set([M.QUERY.on[0], M.QUERY.on[1], H + 0.25]);
      s.label("disagree", "c4").set([M.QUERY.off[0], M.QUERY.off[1], H + 0.25]);
      draw();
    });

    function fit(k) { return M.estimate(state.theta, k, state.method, state.lambda); }
    function corners(b) {
      return [[-L, -L], [L, -L], [L, L], [-L, L]].map(function (c) {
        return [c[0], c[1], ZS * M.predict(b, c)];
      });
    }
    function settle() {
      if (cancel) { cancel(); cancel = null; }
      shown.beta = fit(state.k);
      shown.noise = M.NOISE[state.k].e.slice();
    }
    function renoise() {
      var a = { beta: shown.beta.slice(), noise: shown.noise.slice() };
      state.k = (state.k + 1) % SHOWN;
      var b = { beta: fit(state.k), noise: M.NOISE[state.k].e };
      if (cancel) cancel();
      stats.fit.set(pair(b.beta));
      if (!stage) { settle(); return; }
      cancel = stage.animate(650, function (t) {
        shown.beta = [lerp(a.beta[0], b.beta[0], t), lerp(a.beta[1], b.beta[1], t)];
        shown.noise = a.noise.map(function (e, i) { return lerp(e, b.noise[i], t); });
        scene();
      });
    }
    function scene() {
      if (!stage) return;
      var X = M.columns(state.theta), i, q;
      var on = [[M.QUERY.on[0], M.QUERY.on[1], ZS * M.predict(shown.beta, M.QUERY.on)]];
      var off = [[M.QUERY.off[0], M.QUERY.off[1], ZS * M.predict(shown.beta, M.QUERY.off)]];
      for (i = 0; i < SHOWN; i++) {
        var b = fit(i);
        q = corners(b);
        parts.fan[i].set(q[0], q[1], q[2], q[3]);
        parts.fan[i].show(i !== state.k);
        on.push([M.QUERY.on[0], M.QUERY.on[1], ZS * M.predict(b, M.QUERY.on)]);
        off.push([M.QUERY.off[0], M.QUERY.off[1], ZS * M.predict(b, M.QUERY.off)]);
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
      parts.onDots.set(on);
      parts.offDots.set(off);
      stage.render();
    }
    function draw() {
      var s = M.summary(state.theta, state.method, state.lambda);
      stats.fit.set(pair(fit(state.k)));
      stats.onErr.set(sig(s.on.rmse));
      stats.offMean.set(sig(s.off.mean));
      stats.offSd.set(sig(s.off.sd));
      stats.offErr.set(sig(s.off.rmse));
      scene();
    }
    draw();
  });
})();
