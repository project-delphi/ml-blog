/* Widget for "Below the Prediction Iceberg": intrinsic neural timescales as
 * temporal priors that filter fast sensory prediction errors.
 *
 * A 2,000 ms stream (dt = 10 ms) combines a slow underlying baseline with
 * seeded zero-mean sensory noise and a 120 ms surprise burst starting at
 * t = 900 ms. Two leaky integrators track the stream: a fast sensory rung
 * fixed at tau_fast = 40 ms, and a slow transmodal/self-prior rung whose
 * window tau_slow the reader steps from 40 ms to 600 ms. Built on widget-kit
 * (WK); colours are theme tokens.
 */
"use strict";

WK.mount("widget-timescales", function (root, WK) {
  var DT = 10;          // ms per step
  var T_MAX = 2000;     // total duration in ms
  var N = T_MAX / DT + 1;
  var TAU_FAST = 40;    // sensory cortex window (ms)
  var TAUS = [40, 120, 300, 600];
  var BURSTS = [
    { value: 0, label: "none (0.0)" },
    { value: 1.5, label: "moderate (1.5)" },
    { value: 2.5, label: "sharp (2.5)" }
  ];
  var BURST_START = 900;
  var BURST_END = 1020; // 120 ms burst window

  // Deterministic PRNG so the sensory stream is identical every visit.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    var u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  var rng = mulberry32(20261013);
  var times = [], baseline = [], noise = [];
  for (var i = 0; i < N; i++) {
    var t = i * DT;
    times.push(t);
    // Slow underlying drift: gentle self/context trajectory
    baseline.push(0.35 * Math.sin(2 * Math.PI * t / 2000));
    noise.push(0.45 * gaussian(rng));
  }

  function integrate(signal, tau, initVal) {
    var alpha = 1 - Math.exp(-DT / tau);
    var out = [initVal !== undefined ? initVal : signal[0]];
    for (var k = 1; k < signal.length; k++) {
      out.push(out[k - 1] + alpha * (signal[k] - out[k - 1]));
    }
    return out;
  }

  function rms(a) {
    var s = 0;
    for (var k = 0; k < a.length; k++) s += a[k] * a[k];
    return Math.sqrt(s / a.length);
  }

  function model(tauSlow, burstAmp) {
    var raw = [], k;
    for (k = 0; k < N; k++) {
      var t = times[k];
      var b = (t >= BURST_START && t <= BURST_END) ? burstAmp : 0;
      raw.push(baseline[k] + noise[k] + b);
    }
    var rawNoBurst = [];
    for (k = 0; k < N; k++) rawNoBurst.push(baseline[k] + noise[k]);

    var fast = integrate(raw, TAU_FAST, baseline[0]);
    var slow = integrate(raw, tauSlow, baseline[0]);
    var slowNoBurst = integrate(rawNoBurst, tauSlow, baseline[0]);
    var noiseSlow = integrate(noise, tauSlow, 0);

    // Peak deflection from the surprise burst on the slow state
    var peakDeflect = 0;
    for (k = 0; k < N; k++) {
      var diff = Math.abs(slow[k] - slowNoBurst[k]);
      if (diff > peakDeflect) peakDeflect = diff;
    }

    // Fast jitter passed through: RMS of filtered zero-mean noise vs raw noise RMS
    var jitterRatio = rms(noiseSlow) / rms(noise);

    return {
      raw: raw,
      fast: fast,
      slow: slow,
      halfLife: tauSlow * Math.LN2,
      jitterRatio: jitterRatio,
      peakDeflect: peakDeflect
    };
  }

  // ------------------------------------------------------------------ view
  var state = { tauSlow: 300, burst: 2.5 };
  var f = WK.frame({
    title: "Timescale hierarchy: autocorrelation window and prediction stream",
    note: "Top: intrinsic autocorrelation decay R(Δt) = exp(−Δt / τ) for a fast sensory " +
      "rung (τ = 40 ms) and a slow transmodal prior (τ_slow). Bottom: a 2,000 ms sensory " +
      "stream with a 120 ms surprise burst at 900 ms. The fast rung chases every spike; " +
      "the slow prior pools across the window and holds the baseline."
  });

  var tauCtrl = WK.toggle({
    label: "Slow prior timescale τ_slow (ms)",
    options: TAUS.map(function (v) { return { value: v, label: v + " ms" }; }),
    value: state.tauSlow,
    onchange: function (v) { state.tauSlow = v; draw(); }
  });
  var burstCtrl = WK.toggle({
    label: "Surprise burst at 900 ms",
    options: BURSTS,
    value: state.burst,
    onchange: function (v) { state.burst = v; draw(); }
  });
  f.controls.appendChild(tauCtrl.root);
  f.controls.appendChild(burstCtrl.root);

  var W = 680, H = 370;
  var svg = WK.svg(W, H, {
    "aria-label": "Autocorrelation decay curves and filtered sensory prediction traces"
  });
  f.body.appendChild(svg);

  var halfStat = WK.stat({ label: "slow prior half-life (τ_slow · ln 2)", value: "" });
  var jitStat = WK.stat({ label: "fast noise passed into slow state", value: "" });
  var burstStat = WK.stat({ label: "peak deflection from surprise burst", value: "" });
  f.stats.appendChild(halfStat.root);
  f.stats.appendChild(jitStat.root);
  f.stats.appendChild(burstStat.root);
  root.appendChild(f.root);

  function pathFromPts(pts) {
    return pts.map(function (p, idx) {
      return (idx === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1);
    }).join(" ");
  }

  function draw() {
    var r = model(state.tauSlow, state.burst);
    WK.clear(svg);

    var L = 52, R = W - 20;

    // ---------------- Top panel: Autocorrelation R(dt) = exp(-dt / tau)
    var T1 = 22, B1 = 132;
    var xLag = WK.lin([0, 800], [L, R]);
    var yAcf = WK.lin([0, 1], [B1, T1]);

    svg.appendChild(WK.h("text", {
      x: L, y: T1 - 7, "font-size": 11.5, "font-weight": 600, fill: "ink"
    }, "Autocorrelation window R(Δt) = exp(−Δt / τ)"));

    // Horizontal rules at 0, 0.5, 1.0
    [0, 0.5, 1].forEach(function (v) {
      var y = yAcf(v);
      svg.appendChild(WK.h("line", {
        x1: L, x2: R, y1: y, y2: y,
        stroke: "rule", "stroke-dasharray": v === 0.5 ? "3,3" : null
      }));
      svg.appendChild(WK.h("text", {
        x: L - 6, y: y + 4, "text-anchor": "end", "font-size": 10.5, fill: "muted"
      }, v.toFixed(1)));
    });

    // Lag ticks
    [0, 200, 400, 600, 800].forEach(function (lag) {
      var x = xLag(lag);
      svg.appendChild(WK.h("text", {
        x: x, y: B1 + 14, "text-anchor": "middle", "font-size": 10.5, fill: "muted"
      }, lag + " ms"));
    });

    var acfFast = [], acfSlow = [];
    for (var lag = 0; lag <= 800; lag += 10) {
      acfFast.push([xLag(lag), yAcf(Math.exp(-lag / TAU_FAST))]);
      acfSlow.push([xLag(lag), yAcf(Math.exp(-lag / state.tauSlow))]);
    }
    svg.appendChild(WK.h("path", {
      d: pathFromPts(acfFast), fill: "none", stroke: "c2",
      "stroke-width": 2, "stroke-dasharray": "4,3"
    }));
    svg.appendChild(WK.h("path", {
      d: pathFromPts(acfSlow), fill: "none", stroke: "c1", "stroke-width": 2.6
    }));

    // Legend top-right
    svg.appendChild(WK.h("line", {
      x1: R - 230, x2: R - 210, y1: T1 + 8, y2: T1 + 8,
      stroke: "c2", "stroke-width": 2, "stroke-dasharray": "4,3"
    }));
    svg.appendChild(WK.h("text", {
      x: R - 204, y: T1 + 12, "font-size": 11, fill: "ink"
    }, "sensory rung (τ = 40 ms)"));

    svg.appendChild(WK.h("line", {
      x1: R - 230, x2: R - 210, y1: T1 + 24, y2: T1 + 24,
      stroke: "c1", "stroke-width": 2.6
    }));
    svg.appendChild(WK.h("text", {
      x: R - 204, y: T1 + 28, "font-size": 11, fill: "ink"
    }, "slow prior rung (τ_slow = " + state.tauSlow + " ms)"));

    // ---------------- Bottom panel: Time series traces
    var T2 = 178, B2 = 342;
    var xTime = WK.lin([0, T_MAX], [L, R]);
    var yVal = WK.lin([-1.4, 3.2], [B2, T2]);

    svg.appendChild(WK.h("text", {
      x: L, y: T2 - 8, "font-size": 11.5, "font-weight": 600, fill: "ink"
    }, "Sensory stream vs. filtered state across a 120 ms surprise burst"));

    // Burst shaded band
    if (state.burst > 0) {
      var bx1 = xTime(BURST_START), bx2 = xTime(BURST_END);
      svg.appendChild(WK.h("rect", {
        x: bx1, y: T2, width: bx2 - bx1, height: B2 - T2,
        fill: "accent", "fill-opacity": 0.12
      }));
      svg.appendChild(WK.h("text", {
        x: (bx1 + bx2) / 2, y: T2 + 12, "text-anchor": "middle",
        "font-size": 10, fill: "accent"
      }, "surprise burst"));
    }

    // Zero line & y ticks
    [-1, 0, 1, 2, 3].forEach(function (v) {
      var y = yVal(v);
      svg.appendChild(WK.h("line", {
        x1: L, x2: R, y1: y, y2: y,
        stroke: "rule", "stroke-dasharray": v === 0 ? null : "2,3"
      }));
      svg.appendChild(WK.h("text", {
        x: L - 6, y: y + 4, "text-anchor": "end", "font-size": 10.5, fill: "muted"
      }, v.toFixed(0)));
    });

    // Time ticks
    [0, 500, 1000, 1500, 2000].forEach(function (t) {
      var x = xTime(t);
      svg.appendChild(WK.h("text", {
        x: x, y: B2 + 16, "text-anchor": "middle", "font-size": 10.5, fill: "muted"
      }, t + " ms"));
    });

    var ptsRaw = [], ptsFast = [], ptsSlow = [];
    for (var k = 0; k < N; k++) {
      var x = xTime(times[k]);
      ptsRaw.push([x, yVal(r.raw[k])]);
      ptsFast.push([x, yVal(r.fast[k])]);
      ptsSlow.push([x, yVal(r.slow[k])]);
    }

    svg.appendChild(WK.h("path", {
      d: pathFromPts(ptsRaw), fill: "none", stroke: "muted",
      "stroke-width": 1, "stroke-opacity": 0.55
    }));
    svg.appendChild(WK.h("path", {
      d: pathFromPts(ptsFast), fill: "none", stroke: "c2",
      "stroke-width": 1.6, "stroke-dasharray": "4,2"
    }));
    svg.appendChild(WK.h("path", {
      d: pathFromPts(ptsSlow), fill: "none", stroke: "c1", "stroke-width": 2.6
    }));

    halfStat.set(Math.round(r.halfLife) + " ms");
    jitStat.set(WK.fmt.pct(r.jitterRatio, 0));
    burstStat.set(r.peakDeflect.toFixed(2));
  }

  draw();
});
