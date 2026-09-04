/* Widgets for "Why So Many Matrix Factorizations".
 *
 * Dependency-free: no CDN, no module loader, no framework. Both widgets are
 * inline SVG plus ordinary form controls, so they work from a file:// page.
 * Observable JS does not work with the Quarto installed here (1.6.40).
 *
 * W1 draws one synthetic parabola, marks the feature each algebraic form
 * makes cheap, and eases the frame onto that feature. W2 is a job-first flop
 * chart: which methods apply, and
 * what they cost. Parallel n^3 lines would make Cholesky look universally
 * cheapest; that is the wrong lesson.
 */
"use strict";

const MFCW = (function () {
  const NS = "http://www.w3.org/2000/svg";

  const C = {
    ink: "#1F2430",
    muted: "#5F6672",
    rule: "#D8DBE2",
    purple: "#4A3AA7",
    teal: "#1D6E6E",
    amber: "#9A5B00",
    paper: "#F4F5F7",
  };

  const METHODS = {
    chol: { label: "Cholesky", color: C.purple },
    lu: { label: "LU", color: C.teal },
    qr: { label: "QR", color: C.amber },
    eig: { label: "Eigen", color: "#7A3B6B" },
    svd: { label: "SVD", color: "#1D5C6E" },
    nmf: { label: "NMF", color: "#888888" },
    ne: { label: "X^T X + Cholesky", color: "#C45C26" },
  };

  function n3(n) {
    return n * n * n;
  }

  const JOBS = [
    {
      id: "spd",
      label: "SPD sampling",
      extra: null,
      blurb:
        "Seven-name return covariance. Cholesky is defined and cheapest. LU, eigen, and SVD also factor Sigma; they cost more for the same draws.",
      rows: function (n) {
        return [
          { id: "chol", flops: n3(n) / 3, pick: true },
          { id: "lu", flops: (2 * n3(n)) / 3 },
          { id: "eig", flops: 9 * n3(n) },
          { id: "svd", flops: 21 * n3(n) },
          { id: "qr", flops: null, reason: "X is not a least-squares design" },
          { id: "nmf", flops: null, reason: "signed returns" },
        ];
      },
    },
    {
      id: "ols",
      label: "Tall least squares",
      extra: "aspect",
      blurb:
        "Diabetes-shaped job. QR of X and (form X^T X, then Cholesky) have similar flop counts. QR is the one that does not square kappa.",
      rows: function (n, opt) {
        const m = opt.ratio * n;
        return [
          { id: "qr", flops: 2 * m * n * n - (2 * n3(n)) / 3, pick: true },
          { id: "ne", flops: 2 * m * n * n + n3(n) / 3 },
          { id: "svd", flops: 4 * m * n * n - (4 * n3(n)) / 3 },
          { id: "chol", flops: null, reason: "X is not SPD" },
          { id: "lu", flops: null, reason: "X is not square" },
          { id: "eig", flops: null, reason: "spectrum of X^T X squares kappa" },
        ];
      },
    },
    {
      id: "kkt",
      label: "Square, not SPD",
      extra: null,
      blurb:
        "Mean-variance KKT. The matrix is symmetric indefinite, so Cholesky is not defined. LU is the square solve.",
      rows: function (n) {
        return [
          { id: "lu", flops: (2 * n3(n)) / 3, pick: true },
          { id: "svd", flops: 21 * n3(n) },
          { id: "chol", flops: null, reason: "not SPD" },
          { id: "qr", flops: 4 * n3(n) / 3 },
          { id: "eig", flops: null, reason: "does not solve Kx = b" },
          { id: "nmf", flops: null, reason: "signed KKT block" },
        ];
      },
    },
    {
      id: "any",
      label: "Rank-k / any shape",
      extra: null,
      blurb:
        "Astronaut photograph, or any inverse. SVD is defined for every real matrix. Cholesky and LU are not.",
      rows: function (n) {
        return [
          { id: "svd", flops: 21 * n3(n), pick: true },
          { id: "qr", flops: (4 * n3(n)) / 3 },
          { id: "chol", flops: null, reason: "not SPD" },
          { id: "lu", flops: null, reason: "need square full rank" },
          { id: "eig", flops: null, reason: "need square symmetric" },
          { id: "nmf", flops: null, reason: "allows negatives" },
        ];
      },
    },
    {
      id: "nmf",
      label: "Nonnegative parts",
      extra: "nmf",
      blurb:
        "Counts or intensities. NMF is iterative; SVD is a signed alternative. Move k and t: NMF can undercut or overshoot a full SVD.",
      rows: function (n, opt) {
        return [
          { id: "nmf", flops: 2 * opt.t * opt.k * n * n, pick: true },
          { id: "svd", flops: 21 * n3(n) },
          { id: "chol", flops: null, reason: "not SPD" },
          { id: "lu", flops: null, reason: "not a square solve" },
          { id: "qr", flops: null, reason: "not least squares" },
          { id: "eig", flops: null, reason: "signed spectrum" },
        ];
      },
    },
  ];

  function el(tag, attrs, kids) {
    const e = document.createElementNS(NS, tag);
    for (const key in attrs || {}) e.setAttribute(key, attrs[key]);
    for (const c of kids || []) e.appendChild(c);
    return e;
  }

  function h(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const key in attrs || {}) {
      if (key === "text") e.textContent = attrs[key];
      else e.setAttribute(key, attrs[key]);
    }
    for (const c of kids || []) e.appendChild(c);
    return e;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function labelStyle() {
    return "font: 12px system-ui, sans-serif; color:" + C.muted;
  }

  function controlRow() {
    return h("div", {
      style:
        "display:flex;flex-wrap:wrap;gap:12px 18px;align-items:center;margin:0 0 10px 0",
    });
  }

  // ---- quadratic widget --------------------------------------------------

  function yOf(a, b, c, x) {
    return a * x * x + b * x + c;
  }

  function vertex(a, b, c) {
    if (Math.abs(a) < 1e-12) return { h: 0, k: c };
    const hv = -b / (2 * a);
    return { h: hv, k: yOf(a, b, c, hv) };
  }

  function roots(a, b, c) {
    if (Math.abs(a) < 1e-12) {
      if (Math.abs(b) < 1e-12) return [];
      return [-c / b];
    }
    const d = b * b - 4 * a * c;
    if (d < -1e-12) return [];
    if (d < 1e-12) return [-b / (2 * a)];
    const s = Math.sqrt(d);
    return [(-b - s) / (2 * a), (-b + s) / (2 * a)];
  }

  function initQuad(mountId) {
    const root = document.getElementById(mountId);
    if (!root) return;

    const state = { a: 1, b: -2, c: -3, form: "standard" };

    const formSel = h("select", { id: "mfc-form" }, [
      h("option", { value: "standard", text: "Standard (intercept)" }),
      h("option", { value: "vertex", text: "Vertex (min / max)" }),
      h("option", { value: "factored", text: "Factored (roots)" }),
      h("option", { value: "matrix", text: "Matrix (2x2)" }),
    ]);

    function slider(key, lo, hi, step) {
      const input = h("input", {
        type: "range",
        min: String(lo),
        max: String(hi),
        step: String(step),
        value: String(state[key]),
      });
      const read = h("span", {
        style: "min-width:3.2em;display:inline-block;font:12px ui-monospace,monospace;color:" + C.ink,
      });
      input.addEventListener("input", function () {
        state[key] = Number(input.value);
        draw({ ms: 220, pulse: false });
      });
      return { input: input, read: read, key: key };
    }

    const sa = slider("a", -2, 2, 0.05);
    const sb = slider("b", -6, 6, 0.1);
    const sc = slider("c", -6, 6, 0.1);
    const eq = h("div", {
      style: "min-height:2.1em;margin:8px 0 10px 0;color:" + C.ink,
    });
    const note = h("div", {
      style: "font:13px system-ui,sans-serif;color:" + C.muted + ";min-height:2.4em",
    });

    const W = 640;
    const H = 320;
    const pad = { l: 40, r: 16, t: 16, b: 28 };
    const svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      width: String(W),
      height: String(H),
      style: "max-width:100%;height:auto;display:block;background:" + C.paper,
    });

    const row = controlRow();
    row.appendChild(h("label", { style: labelStyle(), text: "Form " }));
    row.appendChild(formSel);
    function addSlider(name, s) {
      const wrap = h("label", { style: labelStyle() + ";display:flex;align-items:center;gap:6px" });
      wrap.appendChild(h("span", { text: name }));
      wrap.appendChild(s.input);
      wrap.appendChild(s.read);
      row.appendChild(wrap);
    }
    addSlider("a", sa);
    addSlider("b", sb);
    addSlider("c", sc);

    root.appendChild(row);
    root.appendChild(eq);
    root.appendChild(svg);
    root.appendChild(note);

    formSel.addEventListener("change", function () {
      state.form = formSel.value;
      draw({ ms: 560, pulse: true });
    });

    const cam = { xMin: -6, xMax: 6, yMin: -8, yMax: 10 };
    let dest = null;
    let easeFrom = null;
    let easeT0 = 0;
    let easeMs = 420;
    let pulse = 0;
    let ticking = false;
    let booted = false;
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function xpx(x) {
      return pad.l + ((x - cam.xMin) / (cam.xMax - cam.xMin)) * (W - pad.l - pad.r);
    }
    function ypx(y) {
      return pad.t + (1 - (y - cam.yMin) / (cam.yMax - cam.yMin)) * (H - pad.t - pad.b);
    }

    function easeInOut(t) {
      return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    }

    function padBox(xs, ys, minW, minH, margin) {
      let x0 = Math.min.apply(null, xs) - margin;
      let x1 = Math.max.apply(null, xs) + margin;
      let y0 = Math.min.apply(null, ys) - margin;
      let y1 = Math.max.apply(null, ys) + margin;
      if (x1 - x0 < minW) {
        const mid = (x0 + x1) / 2;
        x0 = mid - minW / 2;
        x1 = mid + minW / 2;
      }
      if (y1 - y0 < minH) {
        const mid = (y0 + y1) / 2;
        y0 = mid - minH / 2;
        y1 = mid + minH / 2;
      }
      const lim = 24;
      if (x0 < -lim) {
        x1 += -lim - x0;
        x0 = -lim;
      }
      if (x1 > lim) {
        x0 -= x1 - lim;
        x1 = lim;
      }
      if (y0 < -lim) {
        y1 += -lim - y0;
        y0 = -lim;
      }
      if (y1 > lim) {
        y0 -= y1 - lim;
        y1 = lim;
      }
      return { xMin: x0, xMax: x1, yMin: y0, yMax: y1 };
    }

    function sampleY(a, b, c, x) {
      const y = yOf(a, b, c, x);
      return Math.max(-24, Math.min(24, y));
    }

    function focusView(form, a, b, c, hv, k, rs) {
      const hClamp = Math.max(-18, Math.min(18, hv));
      if (form === "standard") {
        return padBox(
          [0, -1.6, 1.6],
          [c, 0, sampleY(a, b, c, -1.6), sampleY(a, b, c, 1.6)],
          4.2,
          5.2,
          0.85
        );
      }
      if (form === "vertex") {
        const span = 2.2;
        return padBox(
          [hClamp, hClamp - span, hClamp + span],
          [k, sampleY(a, b, c, hClamp - span), sampleY(a, b, c, hClamp + span)],
          4.6,
          5.4,
          0.95
        );
      }
      if (form === "factored") {
        if (rs.length === 0) {
          return padBox(
            [hClamp, hClamp - 2.4, hClamp + 2.4],
            [k, 0, sampleY(a, b, c, hClamp - 2.4), sampleY(a, b, c, hClamp + 2.4)],
            5.2,
            5.8,
            1.0
          );
        }
        const xs = rs.slice();
        const ys = rs.map(function () {
          return 0;
        });
        const mid = rs.length === 2 ? (rs[0] + rs[1]) / 2 : rs[0];
        xs.push(mid);
        ys.push(sampleY(a, b, c, mid));
        return padBox(xs, ys, 5.2, 5.6, 1.05);
      }
      const xs = [0, hClamp, -4.5, 4.5];
      const ys = [c, k, sampleY(a, b, c, -4.5), sampleY(a, b, c, 4.5)];
      rs.forEach(function (r) {
        xs.push(r);
        ys.push(0);
      });
      return padBox(xs, ys, 11, 14, 1.3);
    }

    function goTo(target, ms) {
      dest = target;
      if (reduceMotion || !booted || ms <= 0) {
        cam.xMin = target.xMin;
        cam.xMax = target.xMax;
        cam.yMin = target.yMin;
        cam.yMax = target.yMax;
        easeFrom = null;
        dest = null;
        return;
      }
      easeFrom = {
        xMin: cam.xMin,
        xMax: cam.xMax,
        yMin: cam.yMin,
        yMax: cam.yMax,
      };
      easeT0 = performance.now();
      easeMs = ms;
      requestTick();
    }

    function requestTick() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(frame);
    }

    function frame(now) {
      ticking = false;
      if (easeFrom && dest) {
        const u = Math.min(1, (now - easeT0) / easeMs);
        const e = easeInOut(u);
        cam.xMin = easeFrom.xMin + (dest.xMin - easeFrom.xMin) * e;
        cam.xMax = easeFrom.xMax + (dest.xMax - easeFrom.xMax) * e;
        cam.yMin = easeFrom.yMin + (dest.yMin - easeFrom.yMin) * e;
        cam.yMax = easeFrom.yMax + (dest.yMax - easeFrom.yMax) * e;
        if (u >= 1) {
          cam.xMin = dest.xMin;
          cam.xMax = dest.xMax;
          cam.yMin = dest.yMin;
          cam.yMax = dest.yMax;
          easeFrom = null;
          dest = null;
        }
      }
      if (pulse > 0.02) pulse *= 0.88;
      else pulse = 0;
      paint();
      if (easeFrom || pulse > 0) requestTick();
    }

    function fmt(v) {
      return String(Math.round(v * 100) / 100);
    }

    function texNum(v) {
      const r = Math.round(v * 100) / 100;
      return String(r);
    }

    function near(v, t) {
      return Math.abs(v - t) < 1e-12;
    }

    function leadTimes(v, monom) {
      if (Math.abs(v) < 1e-12) return "";
      if (near(v, 1)) return monom;
      if (near(v, -1)) return "-" + monom;
      return texNum(v) + monom;
    }

    function signedTimes(v, monom) {
      if (Math.abs(v) < 1e-12) return "";
      const mag = near(Math.abs(v), 1) && monom ? monom : texNum(Math.abs(v)) + monom;
      return (v < 0 ? " - " : " + ") + mag;
    }

    function parenShift(h) {
      if (Math.abs(h) < 1e-12) return "x";
      return h < 0 ? "(x + " + texNum(-h) + ")" : "(x - " + texNum(h) + ")";
    }

    function setEq(tex, display) {
      const wrap = display ? "\\[" + tex + "\\]" : "\\(" + tex + "\\)";
      const cls = display ? "math display" : "math inline";
      eq.innerHTML = '<span class="' + cls + '">' + wrap + "</span>";
      if (window.Quarto && window.Quarto.typesetMath) {
        window.Quarto.typesetMath(eq);
      } else if (window.MathJax && window.MathJax.typeset) {
        window.MathJax.typeset([eq]);
      }
    }

    function writeCopy(a, b, c, hv, k, rs) {
      if (state.form === "standard") {
        let body = leadTimes(a, "x^{2}");
        body += signedTimes(b, "x");
        body += signedTimes(c, "");
        setEq("y = " + (body || "0"));
        note.textContent = "Standard form. The constant term is the y-intercept (" + fmt(c) + ").";
      } else if (state.form === "vertex") {
        let body = leadTimes(a, parenShift(hv) + "^{2}");
        if (!body) body = parenShift(hv) + "^{2}";
        if (Math.abs(a) < 1e-12) body = "0";
        body += signedTimes(k, "");
        setEq("y = " + body);
        note.textContent =
          "Vertex form. The turning point is (" + fmt(hv) + ", " + fmt(k) + ").";
      } else if (state.form === "factored") {
        if (rs.length === 0) {
          eq.textContent = "No real roots.";
          note.textContent = "Factored form needs real roots. Move a, b, or c until the curve crosses the axis.";
        } else if (rs.length === 1) {
          setEq("y = " + (leadTimes(a, parenShift(rs[0]) + "^{2}") || parenShift(rs[0]) + "^{2}"));
          note.textContent = "One real root (repeated) at x = " + fmt(rs[0]) + ".";
        } else {
          const pair = parenShift(rs[0]) + parenShift(rs[1]);
          setEq("y = " + (leadTimes(a, pair) || pair));
          note.textContent = "Factored form. The roots sit at x = " + fmt(rs[0]) + " and x = " + fmt(rs[1]) + ".";
        }
      } else {
        setEq(
          "y = \\begin{bmatrix} x & 1 \\end{bmatrix}" +
            "\\begin{bmatrix} " +
            texNum(a) +
            " & " +
            texNum(b / 2) +
            " \\\\ " +
            texNum(b / 2) +
            " & " +
            texNum(c) +
            " \\end{bmatrix}" +
            "\\begin{bmatrix} x \\\\ 1 \\end{bmatrix}",
          true
        );
        note.textContent =
          "Matrix form. The 2x2 already holds (a, b, c). Cholesky of that 2x2 is completing the square.";
      }
    }

    function paint() {
      const a = state.a;
      const b = state.b;
      const c = state.c;
      const { h: hv, k } = vertex(a, b, c);
      const rs = roots(a, b, c);
      const innerX = pad.l;
      const innerY = pad.t;
      const innerW = W - pad.l - pad.r;
      const innerH = H - pad.t - pad.b;

      clear(svg);
      svg.appendChild(
        el("rect", { x: "0", y: "0", width: String(W), height: String(H), fill: C.paper })
      );
      const clip = el("clipPath", { id: "mfc-quad-clip" });
      clip.appendChild(
        el("rect", {
          x: String(innerX),
          y: String(innerY),
          width: String(innerW),
          height: String(innerH),
        })
      );
      svg.appendChild(el("defs", {}, [clip]));
      const g = el("g", { "clip-path": "url(#mfc-quad-clip)" });

      function axis(x1, y1, x2, y2) {
        g.appendChild(
          el("line", {
            x1: String(x1),
            x2: String(x2),
            y1: String(y1),
            y2: String(y2),
            stroke: C.rule,
            "stroke-width": "1",
          })
        );
      }
      if (cam.yMin <= 0 && cam.yMax >= 0) {
        axis(xpx(cam.xMin), ypx(0), xpx(cam.xMax), ypx(0));
      }
      if (cam.xMin <= 0 && cam.xMax >= 0) {
        axis(xpx(0), ypx(cam.yMin), xpx(0), ypx(cam.yMax));
      }

      const pts = [];
      for (let i = 0; i <= 240; i++) {
        const x = cam.xMin + (i / 240) * (cam.xMax - cam.xMin);
        pts.push(xpx(x) + "," + ypx(yOf(a, b, c, x)));
      }
      g.appendChild(
        el("polyline", {
          points: pts.join(" "),
          fill: "none",
          stroke: C.purple,
          "stroke-width": "2.2",
        })
      );

      function mark(x, y, color, label) {
        const cx = xpx(x);
        const cy = ypx(y);
        if (pulse > 0.02) {
          g.appendChild(
            el("circle", {
              cx: String(cx),
              cy: String(cy),
              r: String(5 + pulse * 16),
              fill: "none",
              stroke: color,
              "stroke-width": "1.6",
              opacity: String(Math.max(0, pulse)),
            })
          );
        }
        g.appendChild(
          el("circle", {
            cx: String(cx),
            cy: String(cy),
            r: "5",
            fill: color,
          })
        );
        if (label) {
          const tx = el("text", {
            x: String(cx + 8),
            y: String(cy - 8),
            fill: C.ink,
            "font-family": "system-ui,sans-serif",
            "font-size": "11",
          });
          tx.textContent = label;
          g.appendChild(tx);
        }
      }

      if (state.form === "standard") {
        mark(0, c, C.amber, "c");
      } else if (state.form === "vertex") {
        mark(hv, k, C.teal, "(h, k)");
      } else if (state.form === "factored") {
        rs.forEach(function (r, i) {
          mark(r, 0, C.amber, rs.length === 1 ? "r" : i === 0 ? "r1" : "r2");
        });
      }
      svg.appendChild(g);

      if (state.form === "matrix") {
        const boxX = W - 150;
        const boxY = 22;
        const fade = reduceMotion ? 1 : Math.max(0.35, 1 - pulse * 0.4);
        svg.appendChild(
          el("rect", {
            x: String(boxX),
            y: String(boxY),
            width: "132",
            height: "64",
            fill: "#fff",
            stroke: C.rule,
            opacity: String(fade),
          })
        );
        const t1 = el("text", {
          x: String(boxX + 16),
          y: String(boxY + 26),
          fill: C.ink,
          "font-family": "ui-monospace,monospace",
          "font-size": "12",
          opacity: String(fade),
        });
        t1.textContent = fmt(a) + "    " + fmt(b / 2);
        const t2 = el("text", {
          x: String(boxX + 16),
          y: String(boxY + 48),
          fill: C.ink,
          "font-family": "ui-monospace,monospace",
          "font-size": "12",
          opacity: String(fade),
        });
        t2.textContent = fmt(b / 2) + "    " + fmt(c);
        svg.appendChild(t1);
        svg.appendChild(t2);
      }
    }

    function draw(opts) {
      opts = opts || {};
      sa.read.textContent = fmt(state.a);
      sb.read.textContent = fmt(state.b);
      sc.read.textContent = fmt(state.c);
      const a = state.a;
      const b = state.b;
      const c = state.c;
      const { h: hv, k } = vertex(a, b, c);
      const rs = roots(a, b, c);
      writeCopy(a, b, c, hv, k, rs);
      if (opts.pulse && !reduceMotion) pulse = 1;
      goTo(focusView(state.form, a, b, c, hv, k, rs), opts.ms || 420);
      booted = true;
      paint();
      if (easeFrom || pulse > 0) requestTick();
    }

    draw({ ms: 0, pulse: false });
  }

  // ---- cost widget -------------------------------------------------------

  function initCost(mountId) {
    const root = document.getElementById(mountId);
    if (!root) return;

    const state = { n: 500, job: "spd", k: 8, t: 200, ratio: 5 };

    const nSlider = h("input", {
      type: "range",
      min: "50",
      max: "2000",
      step: "50",
      value: String(state.n),
    });
    const nRead = h("span", {
      style: "min-width:3.2em;font:12px ui-monospace,monospace;color:" + C.ink,
    });
    const jobSel = h("select");
    JOBS.forEach(function (j) {
      jobSel.appendChild(h("option", { value: j.id, text: j.label }));
    });
    const ratioSlider = h("input", {
      type: "range",
      min: "2",
      max: "20",
      step: "1",
      value: String(state.ratio),
    });
    const ratioRead = h("span", {
      style: "font:12px ui-monospace,monospace;color:" + C.ink,
    });
    const kSlider = h("input", {
      type: "range",
      min: "2",
      max: "64",
      step: "1",
      value: String(state.k),
    });
    const tSlider = h("input", {
      type: "range",
      min: "20",
      max: "1000",
      step: "20",
      value: String(state.t),
    });
    const kRead = h("span", { style: "font:12px ui-monospace,monospace;color:" + C.ink });
    const itersRead = h("span", { style: "font:12px ui-monospace,monospace;color:" + C.ink });
    const mapping = h("div", {
      style:
        "font:13px system-ui,sans-serif;color:" +
        C.ink +
        ";background:" +
        C.paper +
        ";padding:10px 12px;margin:8px 0 0 0",
    });

    const W = 680;
    const H = 360;
    const svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      width: String(W),
      height: String(H),
      style: "max-width:100%;height:auto;display:block;background:" + C.paper,
    });

    const row = controlRow();
    const jobLab = h("label", { style: labelStyle() + ";display:flex;align-items:center;gap:6px" });
    jobLab.appendChild(h("span", { text: "Job" }));
    jobLab.appendChild(jobSel);
    const nLab = h("label", { style: labelStyle() + ";display:flex;align-items:center;gap:6px" });
    nLab.appendChild(h("span", { text: "n (size)" }));
    nLab.appendChild(nSlider);
    nLab.appendChild(nRead);
    const ratioLab = h("label", { style: labelStyle() + ";display:flex;align-items:center;gap:6px" });
    ratioLab.appendChild(h("span", { text: "m/n" }));
    ratioLab.appendChild(ratioSlider);
    ratioLab.appendChild(ratioRead);
    const kLab = h("label", { style: labelStyle() + ";display:flex;align-items:center;gap:6px" });
    kLab.appendChild(h("span", { text: "NMF k" }));
    kLab.appendChild(kSlider);
    kLab.appendChild(kRead);
    const tLab = h("label", { style: labelStyle() + ";display:flex;align-items:center;gap:6px" });
    tLab.appendChild(h("span", { text: "NMF t" }));
    tLab.appendChild(tSlider);
    tLab.appendChild(itersRead);
    row.appendChild(jobLab);
    row.appendChild(nLab);
    row.appendChild(ratioLab);
    row.appendChild(kLab);
    row.appendChild(tLab);

    root.appendChild(row);
    root.appendChild(svg);
    root.appendChild(mapping);

    function sci(v) {
      return v.toExponential(2);
    }

    function draw() {
      state.n = Number(nSlider.value);
      state.job = jobSel.value;
      state.k = Number(kSlider.value);
      state.t = Number(tSlider.value);
      state.ratio = Number(ratioSlider.value);
      nRead.textContent = String(state.n);
      kRead.textContent = String(state.k);
      itersRead.textContent = String(state.t);
      ratioRead.textContent = String(state.ratio);
      const job = JOBS.find(function (j) {
        return j.id === state.job;
      });
      ratioLab.style.display = job.extra === "aspect" ? "flex" : "none";
      kLab.style.display = job.extra === "nmf" ? "flex" : "none";
      tLab.style.display = job.extra === "nmf" ? "flex" : "none";

      const rows = job.rows(state.n, { k: state.k, t: state.t, ratio: state.ratio });
      const defined = rows.filter(function (r) {
        return r.flops != null;
      });
      const max = Math.max.apply(
        null,
        defined.map(function (r) {
          return r.flops;
        })
      );

      clear(svg);
      svg.appendChild(
        el("rect", { x: "0", y: "0", width: String(W), height: String(H), fill: C.paper })
      );

      const left = 156;
      const right = 92;
      const top = 56;
      const bottom = 48;
      const rowH = (H - top - bottom) / rows.length;
      const avail = W - left - right;

      const title = el("text", {
        x: String(W / 2),
        y: "22",
        fill: C.ink,
        "font-size": "14",
        "font-family": "system-ui,sans-serif",
        "font-weight": "600",
        "text-anchor": "middle",
      });
      title.textContent = "Estimated flops to do: " + job.label;
      svg.appendChild(title);
      const sub = el("text", {
        x: String(W / 2),
        y: "40",
        fill: C.muted,
        "font-size": "11",
        "font-family": "system-ui,sans-serif",
        "text-anchor": "middle",
      });
      sub.textContent = "Bar length is cost. Color names the method, so it stays put when you change Job.";
      svg.appendChild(sub);

      const yHead = el("text", {
        x: String(left - 10),
        y: String(top - 8),
        fill: C.muted,
        "font-size": "11",
        "font-family": "system-ui,sans-serif",
        "text-anchor": "end",
      });
      yHead.textContent = "method";
      svg.appendChild(yHead);

      const axisY = H - bottom;
      svg.appendChild(
        el("line", {
          x1: String(left),
          x2: String(left + avail),
          y1: String(axisY),
          y2: String(axisY),
          stroke: C.rule,
          "stroke-width": "1",
        })
      );
      const zero = el("text", {
        x: String(left),
        y: String(axisY + 12),
        fill: C.muted,
        "font-size": "10",
        "font-family": "ui-monospace,monospace",
      });
      zero.textContent = "0";
      svg.appendChild(zero);
      const xmax = el("text", {
        x: String(left + avail),
        y: String(axisY + 12),
        fill: C.muted,
        "font-size": "10",
        "font-family": "ui-monospace,monospace",
        "text-anchor": "end",
      });
      xmax.textContent = defined.length ? sci(max) : "";
      svg.appendChild(xmax);
      const xlab = el("text", {
        x: String(left + avail / 2),
        y: String(H - 8),
        fill: C.muted,
        "font-size": "11",
        "font-family": "system-ui,sans-serif",
        "text-anchor": "middle",
      });
      xlab.textContent = "flops (leading-term model, n = " + state.n + ")";
      svg.appendChild(xlab);

      rows.forEach(function (r, i) {
        const y = top + i * rowH;
        const meta = METHODS[r.id];
        const name = el("text", {
          x: String(left - 10),
          y: String(y + rowH * 0.62),
          fill: r.pick ? C.ink : C.muted,
          "font-size": r.pick ? "13" : "12",
          "font-family": "system-ui,sans-serif",
          "font-weight": r.pick ? "600" : "400",
          "text-anchor": "end",
        });
        name.textContent = (r.pick ? "▸ " : "") + meta.label;
        svg.appendChild(name);

        if (r.flops == null) {
          const miss = el("text", {
            x: String(left + 4),
            y: String(y + rowH * 0.62),
            fill: C.muted,
            "font-size": "12",
            "font-family": "system-ui,sans-serif",
          });
          miss.textContent = r.reason || "not defined";
          svg.appendChild(miss);
          return;
        }

        const w = Math.max(4, (r.flops / max) * avail);
        svg.appendChild(
          el("rect", {
            x: String(left),
            y: String(y + 8),
            width: String(w),
            height: String(rowH - 16),
            fill: meta.color,
            opacity: r.pick ? "1" : "0.55",
          })
        );
        const val = el("text", {
          x: String(left + w + 6),
          y: String(y + rowH * 0.62),
          fill: C.ink,
          "font-size": "11",
          "font-family": "ui-monospace,monospace",
        });
        val.textContent = sci(r.flops);
        svg.appendChild(val);
      });

      mapping.textContent = job.blurb;
    }

    nSlider.addEventListener("input", draw);
    jobSel.addEventListener("change", draw);
    kSlider.addEventListener("input", draw);
    tSlider.addEventListener("input", draw);
    ratioSlider.addEventListener("input", draw);
    draw();
  }

  function boot() {
    initQuad("quad-widget");
    initCost("cost-widget");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  return { boot: boot };
})();
