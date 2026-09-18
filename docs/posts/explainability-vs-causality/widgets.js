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
    var pts = [];
    for (var i = 0; i < n; i++) {
      var z = gauss(rand);
      var x = (assign ? 0 : zToX * z) + gauss(rand);
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

  var svg = WK.svg(660, 300);
  f.body.appendChild(svg);

  var statSee = WK.stat({ label: "Slope from seeing", value: "--" });
  var statDo = WK.stat({ label: "Slope from doing", value: "--" });
  var statTrue = WK.stat({ label: "True effect", value: "--" });
  [statSee, statDo, statTrue].forEach(function (s) { f.stats.appendChild(s.root); });

  root.appendChild(f.root);

  function panel(x0, title, pts, m, xs, ys) {
    svg.appendChild(WK.h("text", { x: x0 + 140, y: 20, "text-anchor": "middle",
                                   fill: "ink", "font-size": 14, text: title }));
    // axes
    svg.appendChild(WK.h("line", { x1: x0, x2: x0 + 280, y1: ys(0), y2: ys(0),
                                   stroke: "rule", "stroke-width": 1 }));
    svg.appendChild(WK.h("line", { x1: xs(0), x2: xs(0), y1: 40, y2: 270,
                                   stroke: "rule", "stroke-width": 1 }));
    pts.forEach(function (p) {
      // colour carries the confounder, so its role is visible in the left panel
      svg.appendChild(WK.h("circle", { cx: xs(p.x), cy: ys(p.y), r: 2.4,
                                       fill: p.z > 0 ? "c1" : "c3", opacity: 0.55 }));
    });
    var my = 0, mx = 0, i;
    for (i = 0; i < pts.length; i++) { mx += pts[i].x; my += pts[i].y; }
    mx /= pts.length; my /= pts.length;
    var xa = -3.5, xb = 3.5;
    svg.appendChild(WK.h("line", {
      x1: xs(xa), y1: ys(my + m * (xa - mx)),
      x2: xs(xb), y2: ys(my + m * (xb - mx)),
      stroke: "accent", "stroke-width": 2.5
    }));
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
    var xsL = WK.lin([-4, 4], [30, 310]), xsR = WK.lin([-4, 4], [360, 640]);
    var ys = WK.lin([-5, 5], [270, 40]);
    panel(30, "Seeing: who happens to be treated", seen, mSee, xsL, ys);
    panel(360, "Doing: treatment assigned at random", done, mDo, xsR, ys);

    statSee.set(WK.fmt.num(mSee, 2));
    statDo.set(WK.fmt.num(mDo, 2));
    statTrue.set(WK.fmt.num(eff, 2));
  }

  draw();
});
