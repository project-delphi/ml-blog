"use strict";

// Two panels over the same three-variable world. On the left the treatment is
// observed, so the confounder reaches the outcome by two routes and the fitted
// slope mixes them. On the right the treatment is assigned at random, which
// cuts the confounder's route into it, and the same fit returns the effect.
WK.mount("widget-seeing-doing", function (root, WK) {
  var N = 300;

  // ---------------------------------------------------------------- model
  function rng(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gauss(rand) {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /**
   * One world. `assign` true is the intervention: the treatment is drawn
   * independently of the confounder rather than caused by it.
   */
  function world(n, effect, zToX, zToY, assign, rand) {
    // Assignment cuts the arrow and nothing else. Keeping the treatment's
    // marginal variance at zToX^2 + 1 under both regimes matters: drawing it
    // as plain noise would also shrink the spread, and the narrower cloud
    // would read as randomisation compressing the treatment, which it does
    // not do.
    var spread = Math.sqrt(zToX * zToX + 1);
    var pts = [];
    for (var i = 0; i < n; i++) {
      var z = gauss(rand);
      var x = assign ? spread * gauss(rand) : zToX * z + gauss(rand);
      var y = effect * x + zToY * z + gauss(rand);
      pts.push({ x: x, y: y, z: z });
    }
    return pts;
  }

  /** Least-squares slope of y on x, which is what a one-feature model reports. */
  function slope(pts) {
    var n = pts.length, sx = 0, sy = 0, sxy = 0, sxx = 0, i;
    for (i = 0; i < n; i++) { sx += pts[i].x; sy += pts[i].y; }
    var mx = sx / n, my = sy / n;
    for (i = 0; i < n; i++) {
      sxy += (pts[i].x - mx) * (pts[i].y - my);
      sxx += (pts[i].x - mx) * (pts[i].x - mx);
    }
    return sxx > 0 ? sxy / sxx : 0;
  }

  // ----------------------------------------------------------------- view
  var f = WK.frame({
    title: "Seeing and doing, on one world",
    note: "Both panels share the same causal effect. Only the left one lets the "
        + "confounder decide who gets treated."
  });

  var effS = WK.slider({ label: "True effect of the treatment", min: -1, max: 1, step: 0.1,
                         value: 0.3, fmt: WK.fmt.num, oninput: draw });
  var zxS = WK.slider({ label: "Confounder to treatment", min: -1.5, max: 1.5, step: 0.1,
                        value: 1.2, fmt: WK.fmt.num, oninput: draw });
  var zyS = WK.slider({ label: "Confounder to outcome", min: -1.5, max: 1.5, step: 0.1,
                        value: -1.2, fmt: WK.fmt.num, oninput: draw });
  var seedS = WK.slider({ label: "Seed", min: 1, max: 30, step: 1, value: 3,
                          fmt: WK.fmt.int, oninput: draw });
  [effS, zxS, zyS, seedS].forEach(function (c) { f.controls.appendChild(c.root); });

  var svg = WK.svg(660, 300, {
    "aria-label": "Two scatter plots of the same world, one with the treatment "
      + "observed and one with it assigned at random, each with its fitted slope"
  });
  f.body.appendChild(svg);

  var statSee = WK.stat({ label: "Slope from seeing", value: "--" });
  var statDo = WK.stat({ label: "Slope from doing", value: "--" });
  var statTrue = WK.stat({ label: "True effect", value: "--" });
  [statSee, statDo, statTrue].forEach(function (s) { f.stats.appendChild(s.root); });

  root.appendChild(f.root);

  /** One panel, drawn inside its own clip so nothing can stray into the other. */
  function panel(id, x0, title, pts, m, xs, ys) {
    var clip = "wsd-clip-" + id;
    svg.appendChild(WK.h("clipPath", { id: clip },
      WK.h("rect", { x: x0, y: 30, width: 280, height: 245 })));
    var g = WK.h("g", { "clip-path": "url(#" + clip + ")" });

    svg.appendChild(WK.h("text", { x: x0 + 140, y: 18, "text-anchor": "middle",
                                   fill: "ink", "font-size": 14, text: title }));
    // axes, with ticks so the reader can see the two panels share one scale
    g.appendChild(WK.h("line", { x1: x0, x2: x0 + 280, y1: ys(0), y2: ys(0),
                                 stroke: "rule", "stroke-width": 1 }));
    g.appendChild(WK.h("line", { x1: xs(0), x2: xs(0), y1: 30, y2: 275,
                                 stroke: "rule", "stroke-width": 1 }));
    WK.ticks(xs.domain[0], xs.domain[1], 4).forEach(function (t) {
      if (t === 0) return;
      g.appendChild(WK.h("line", { x1: xs(t), x2: xs(t), y1: ys(0) - 3, y2: ys(0) + 3,
                                   stroke: "rule", "stroke-width": 1 }));
      g.appendChild(WK.h("text", { x: xs(t), y: ys(0) + 15, "text-anchor": "middle",
                                   fill: "muted", "font-size": 10, text: WK.fmt.num(t, 0) }));
    });
    WK.ticks(ys.domain[0], ys.domain[1], 4).forEach(function (t) {
      if (t === 0) return;
      g.appendChild(WK.h("line", { x1: xs(0) - 3, x2: xs(0) + 3, y1: ys(t), y2: ys(t),
                                   stroke: "rule", "stroke-width": 1 }));
      g.appendChild(WK.h("text", { x: xs(0) - 7, y: ys(t) + 3, "text-anchor": "end",
                                   fill: "muted", "font-size": 10, text: WK.fmt.num(t, 0) }));
    });

    pts.forEach(function (p) {
      // colour carries the confounder, so its role is visible in the left panel
      g.appendChild(WK.h("circle", { cx: xs(p.x), cy: ys(p.y), r: 2.4,
                                     fill: p.z > 0 ? "c1" : "c3", opacity: 0.55 }));
    });

    var my = 0, mx = 0, i;
    for (i = 0; i < pts.length; i++) { mx += pts[i].x; my += pts[i].y; }
    mx /= pts.length; my /= pts.length;
    var xa = xs.domain[0], xb = xs.domain[1];
    g.appendChild(WK.h("line", {
      x1: xs(xa), y1: ys(my + m * (xa - mx)),
      x2: xs(xb), y2: ys(my + m * (xb - mx)),
      stroke: "accent", "stroke-width": 2.5
    }));
    svg.appendChild(g);
    svg.appendChild(WK.h("text", { x: x0 + 140, y: 292, "text-anchor": "middle",
                                   fill: "muted", "font-size": 12,
                                   text: "slope " + WK.fmt.num(m, 2) }));
  }

  function draw() {
    var eff = effS.get(), zx = zxS.get(), zy = zyS.get(), seed = seedS.get();
    var seen = world(N, eff, zx, zy, false, rng(seed));
    var done = world(N, eff, zx, zy, true, rng(seed + 991));
    var mSee = slope(seen), mDo = slope(done);

    WK.clear(svg);
    // One pair of domains for both panels, wide enough for the data actually
    // drawn: a fixed window silently dropped points at the stronger settings,
    // and shared limits are what makes the two clouds comparable at a glance.
    var all = seen.concat(done), xm = 1, ym = 1, i;
    for (i = 0; i < all.length; i++) {
      xm = Math.max(xm, Math.abs(all[i].x));
      ym = Math.max(ym, Math.abs(all[i].y));
    }
    xm = Math.ceil(xm); ym = Math.ceil(ym);
    var xsL = WK.lin([-xm, xm], [30, 310]), xsR = WK.lin([-xm, xm], [360, 640]);
    var ys = WK.lin([-ym, ym], [270, 40]);
    panel("see", 30, "Seeing: who happens to be treated", seen, mSee, xsL, ys);
    panel("do", 360, "Doing: treatment assigned at random", done, mDo, xsR, ys);

    statSee.set(WK.fmt.num(mSee, 2));
    statDo.set(WK.fmt.num(mDo, 2));
    statTrue.set(WK.fmt.num(eff, 2));
  }

  draw();
});
