/* Widgets for "The Matrix That Rotates, Stretches, and Rotates Again".
 *
 * Dependency-free: no CDN, no module loader, no framework. W1 is inline SVG
 * built through the DOM; W2 is a <canvas> fed by ImageData. Both therefore work
 * from a plain file:// page as well as over HTTP.
 *
 * Data arrives inlined in the page as <script id="svd-data" type="application/json">,
 * written from the committed widget-data/*.json files at render time.
 *
 * W1 needs no data at all: it computes the SVD of the reader's own 2x2 matrix
 * in the browser, in closed form. W2 ships the leading 100 singular triplets of
 * the photograph as quantised int16 and rebuilds any rank from them.
 */
"use strict";

const SVDW = (function () {
  const NS = "http://www.w3.org/2000/svg";

  const C = {
    ink: "#1F2430",
    muted: "#5F6672",
    rule: "#D8DBE2",
    purple: "#4A3AA7",
    teal: "#1D6E6E",
    amber: "#9A5B00",
    circle: "#B9BEC9",
  };

  // ---- DOM helpers -------------------------------------------------------

  function el(tag, attrs, kids) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
    for (const c of kids || []) e.appendChild(c);
    return e;
  }

  function h(tag, attrs, kids) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) {
      if (k === "html") e.innerHTML = attrs[k];
      else if (k === "text") e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    for (const c of kids || []) e.appendChild(c);
    return e;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function svg(w, hgt) {
    return el("svg", {
      viewBox: `0 0 ${w} ${hgt}`,
      width: w,
      height: hgt,
      style: "max-width:100%;height:auto;display:block;margin:0 auto",
      "font-family": "inherit",
    });
  }

  // ---- 2x2 linear algebra, in closed form --------------------------------

  /* Eigen-decomposition of a symmetric 2x2 [[a, b], [b, c]], descending. */
  function symEig2(a, b, c) {
    const mid = (a + c) / 2;
    const gap = Math.hypot((a - c) / 2, b);
    const l1 = mid + gap;
    const l2 = mid - gap;
    // atan2 keeps the angle right when a === c, where atan would divide by zero.
    const theta = 0.5 * Math.atan2(2 * b, a - c);
    return {
      lambda: [l1, l2],
      v: [
        [Math.cos(theta), Math.sin(theta)],
        [-Math.sin(theta), Math.cos(theta)],
      ],
    };
  }

  /* SVD of a 2x2 given row-wise as [[m00, m01], [m10, m11]].
   *
   * Exactly the argument the post makes in prose: diagonalise A^T A to get the
   * right singular vectors, then read the left ones off the images A v_i. */
  function svd2(m) {
    const [[m00, m01], [m10, m11]] = m;
    const a = m00 * m00 + m10 * m10;
    const b = m00 * m01 + m10 * m11;
    const c = m01 * m01 + m11 * m11;

    const { lambda, v } = symEig2(a, b, c);
    const sigma = lambda.map((l) => Math.sqrt(Math.max(l, 0)));

    const u = [];
    for (let i = 0; i < 2; i++) {
      const av = [m00 * v[i][0] + m01 * v[i][1], m10 * v[i][0] + m11 * v[i][1]];
      if (sigma[i] > 1e-9) {
        u.push([av[0] / sigma[i], av[1] / sigma[i]]);
      } else {
        // A singular direction has no image to normalise. Complete the basis
        // with the perpendicular of whatever u_1 turned out to be.
        const prev = u[0] || [1, 0];
        u.push([-prev[1], prev[0]]);
      }
    }
    return { u, sigma, v };
  }

  function apply(m, p) {
    return [m[0][0] * p[0] + m[0][1] * p[1], m[1][0] * p[0] + m[1][1] * p[1]];
  }

  function matmul(x, y) {
    return [
      [
        x[0][0] * y[0][0] + x[0][1] * y[1][0],
        x[0][0] * y[0][1] + x[0][1] * y[1][1],
      ],
      [
        x[1][0] * y[0][0] + x[1][1] * y[1][0],
        x[1][0] * y[0][1] + x[1][1] * y[1][1],
      ],
    ];
  }

  // ---- controls ----------------------------------------------------------

  function numberBox(value, onchange) {
    const input = h("input", {
      type: "number",
      step: "0.1",
      value: value,
      style:
        "width:4.6rem;padding:.2rem .3rem;font:inherit;font-variant-numeric:tabular-nums;" +
        "border:1px solid " + C.rule + ";border-radius:4px;text-align:right",
    });
    input.addEventListener("input", () => onchange(parseFloat(input.value)));
    return input;
  }

  function slider(label, min, max, step, value, fmt, onchange) {
    const out = h("span", {
      style: "font-variant-numeric:tabular-nums;font-weight:600;min-width:3.5rem;display:inline-block",
    });
    const input = h("input", {
      type: "range",
      min: min,
      max: max,
      step: step,
      value: value,
      style: "vertical-align:middle;width:min(60vw,260px)",
    });
    const sync = () => {
      const v = parseFloat(input.value);
      out.textContent = fmt(v);
      onchange(v);
    };
    input.addEventListener("input", sync);
    const wrap = h(
      "label",
      { style: "display:inline-flex;align-items:center;gap:.4rem;font-size:.88rem" },
      [h("span", { text: label, style: "color:" + C.muted }), input, out]
    );
    return { node: wrap, input: input, sync: sync };
  }

  function readout(label, initial) {
    const value = h("span", {
      text: initial,
      style: "font-weight:700;font-variant-numeric:tabular-nums;color:" + C.ink,
    });
    const node = h(
      "span",
      {
        style:
          "display:inline-flex;flex-direction:column;gap:.05rem;padding:.3rem .7rem;" +
          "border-left:3px solid " + C.rule + ";line-height:1.25",
      },
      [
        h("span", { text: label, style: "font-size:.72rem;letter-spacing:.04em;color:" + C.muted }),
        value,
      ]
    );
    return { node: node, set: (t) => (value.textContent = t) };
  }

  return {
    NS: NS, C: C, el: el, h: h, clear: clear, svg: svg,
    svd2: svd2, apply: apply, matmul: matmul,
    numberBox: numberBox, slider: slider, readout: readout,
  };
})();

/* ---------------------------------------------------------------------------
 * W1 -- a 2x2 matrix, its unit circle, and the ellipse it becomes.
 *
 * The reader edits the four entries. Everything else is recomputed from a
 * genuine SVD of what they typed: the ellipse, the two singular values, the
 * condition number, and the four basis vectors.
 * ------------------------------------------------------------------------ */
function initEllipseWidget(mountId) {
  const root = document.getElementById(mountId);
  if (!root) return;

  const S = SVDW;
  const C = S.C;
  const W = 460;
  const H = 320;
  const CX = W / 2;
  const CY = H / 2 - 6;
  const UNIT = 68; // pixels per unit length

  // A shear: distinct singular values, visibly not symmetric, and its two
  // rotations are different, which is the whole point of the picture.
  let M = [
    [1.0, 1.2],
    [0.0, 1.0],
  ];
  let stage = 3;
  let timer = null;

  const STAGES = [
    { label: "unit circle", note: "Start: every input vector of length 1." },
    { label: "after Vᵀ", note: "A rotation. The circle is unmoved; the marked directions are not." },
    { label: "after ΣVᵀ", note: "The stretch. Now it is an ellipse, aligned to the axes." },
    { label: "after A = UΣVᵀ", note: "The second rotation turns the ellipse to its final pose." },
  ];

  const box = S.h("div", {});
  const controls = S.h("div", {
    style: "display:flex;flex-wrap:wrap;gap:1rem;align-items:center;margin-bottom:.6rem",
  });
  const grid = S.h("div", {
    style: "display:grid;grid-template-columns:auto auto;gap:.35rem",
  });
  const stageRow = S.h("div", {
    style: "display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;margin:.5rem 0",
  });
  const stats = S.h("div", {
    style: "display:flex;flex-wrap:wrap;gap:.2rem;margin:.5rem 0 .3rem",
  });
  const caption = S.h("div", {
    style: "font-size:.85rem;color:" + C.muted + ";min-height:2.4em;margin-top:.3rem",
  });
  const plot = S.h("div", {});

  const inputs = [];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const inp = S.numberBox(M[i][j], (v) => {
        M[i][j] = Number.isFinite(v) ? v : 0;
        draw();
      });
      inputs.push(inp);
      grid.appendChild(inp);
    }
  }

  const matrixLabel = S.h("div", {
    style: "display:flex;align-items:center;gap:.5rem",
  }, [
    S.h("span", { text: "A =", style: "font-style:italic;font-size:1.05rem" }),
    grid,
  ]);
  controls.appendChild(matrixLabel);

  const presets = S.h("div", { style: "display:flex;flex-wrap:wrap;gap:.35rem" });
  const PRESETS = [
    ["shear", [[1, 1.2], [0, 1]]],
    ["rotation", [[0.6, -0.8], [0.8, 0.6]]],
    ["near-singular", [[1, 1], [1, 1.02]]],
    ["pure stretch", [[2.2, 0], [0, 0.5]]],
  ];
  for (const [name, mat] of PRESETS) {
    const b = S.h("button", {
      text: name,
      type: "button",
      style:
        "font:inherit;font-size:.8rem;padding:.22rem .6rem;cursor:pointer;" +
        "border:1px solid " + C.rule + ";border-radius:999px;background:#fff;color:" + C.ink,
    });
    b.addEventListener("click", () => {
      M = mat.map((r) => r.slice());
      inputs[0].value = M[0][0];
      inputs[1].value = M[0][1];
      inputs[2].value = M[1][0];
      inputs[3].value = M[1][1];
      draw();
    });
    presets.appendChild(b);
  }
  controls.appendChild(presets);

  const stageButtons = [];
  STAGES.forEach((s, i) => {
    const b = S.h("button", {
      text: s.label,
      type: "button",
      style: "font:inherit;font-size:.8rem;padding:.22rem .6rem;cursor:pointer;border-radius:4px",
    });
    b.addEventListener("click", () => {
      stop();
      stage = i;
      draw();
    });
    stageButtons.push(b);
    stageRow.appendChild(b);
  });

  const playBtn = S.h("button", {
    text: "▶ play",
    type: "button",
    style:
      "font:inherit;font-size:.8rem;padding:.22rem .7rem;cursor:pointer;margin-left:.4rem;" +
      "border:1px solid " + C.purple + ";border-radius:4px;background:" + C.purple + ";color:#fff",
  });
  playBtn.addEventListener("click", () => (timer ? stop() : start()));
  stageRow.appendChild(playBtn);

  function start() {
    timer = setInterval(() => {
      stage = (stage + 1) % 4;
      draw();
    }, 1100);
    playBtn.textContent = "⏸ pause";
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    playBtn.textContent = "▶ play";
  }

  const rSigma1 = S.readout("σ₁ (max stretch)", "");
  const rSigma2 = S.readout("σ₂ (min stretch)", "");
  const rCond = S.readout("condition number", "");
  const rDet = S.readout("|det A| = σ₁σ₂", "");
  [rSigma1, rSigma2, rCond, rDet].forEach((r) => stats.appendChild(r.node));

  box.appendChild(controls);
  box.appendChild(stageRow);
  box.appendChild(plot);
  box.appendChild(stats);
  box.appendChild(caption);
  root.appendChild(box);

  const MARKERS = 4;

  function px(p) {
    return [CX + p[0] * UNIT, CY - p[1] * UNIT];
  }

  function arrow(from, to, colour, width) {
    const a = px(from);
    const b = px(to);
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    const g = S.el("g", {});
    g.appendChild(
      S.el("line", {
        x1: a[0], y1: a[1], x2: b[0], y2: b[1],
        stroke: colour, "stroke-width": width, "stroke-linecap": "round",
      })
    );
    if (len > 6) {
      const ux = dx / len;
      const uy = dy / len;
      const head = 9;
      g.appendChild(
        S.el("polygon", {
          points: [
            `${b[0]},${b[1]}`,
            `${b[0] - head * ux + head * 0.45 * uy},${b[1] - head * uy - head * 0.45 * ux}`,
            `${b[0] - head * ux - head * 0.45 * uy},${b[1] - head * uy + head * 0.45 * ux}`,
          ].join(" "),
          fill: colour,
        })
      );
    }
    return g;
  }

  function draw() {
    const { u, sigma, v } = S.svd2(M);

    // The three stages, as actual matrices. Nothing here is interpolated:
    // each button shows the exact image of the circle under a real product.
    const vt = [
      [v[0][0], v[0][1]],
      [v[1][0], v[1][1]],
    ];
    const sig = [
      [sigma[0], 0],
      [0, sigma[1]],
    ];
    const stages = [
      [[1, 0], [0, 1]],
      vt,
      S.matmul(sig, vt),
      M,
    ];
    const T = stages[stage];

    const g = S.svg(W, H);

    // Axes.
    g.appendChild(S.el("line", { x1: 0, y1: CY, x2: W, y2: CY, stroke: C.rule, "stroke-width": 1 }));
    g.appendChild(S.el("line", { x1: CX, y1: 0, x2: CX, y2: H, stroke: C.rule, "stroke-width": 1 }));

    // The image of the unit circle under T, sampled densely.
    const pts = [];
    for (let i = 0; i <= 180; i++) {
      const t = (i / 180) * 2 * Math.PI;
      pts.push(px(S.apply(T, [Math.cos(t), Math.sin(t)])));
    }
    g.appendChild(
      S.el("polygon", {
        points: pts.map((p) => p.join(",")).join(" "),
        fill: C.purple,
        "fill-opacity": 0.09,
        stroke: C.purple,
        "stroke-width": 2.2,
      })
    );

    // Faint reference: where the unit circle started.
    if (stage > 0) {
      g.appendChild(
        S.el("circle", {
          cx: CX, cy: CY, r: UNIT,
          fill: "none", stroke: C.circle, "stroke-width": 1.2, "stroke-dasharray": "3 4",
        })
      );
    }

    // Marked directions, so a pure rotation is visibly a rotation.
    for (let i = 0; i < MARKERS; i++) {
      const t = (i / MARKERS) * 2 * Math.PI + 0.35;
      const p = px(S.apply(T, [Math.cos(t), Math.sin(t)]));
      g.appendChild(S.el("circle", { cx: p[0], cy: p[1], r: 4.2, fill: C.purple, "fill-opacity": 0.85 }));
    }

    // The right singular vectors, drawn on whatever the current stage is.
    // At stage 0 they sit on the circle; by stage 3 they are the ellipse axes.
    const colours = [C.teal, C.amber];
    for (let i = 0; i < 2; i++) {
      const image = S.apply(T, v[i]);
      g.appendChild(arrow([0, 0], image, colours[i], 2.6));
      const tip = px(image);
      const label = stage === 0 ? (i === 0 ? "v₁" : "v₂") : stage === 3 ? (i === 0 ? "σ₁u₁" : "σ₂u₂") : "";
      if (label) {
        const t = S.el("text", {
          x: tip[0] + (image[0] >= 0 ? 8 : -8),
          y: tip[1] + (image[1] >= 0 ? -8 : 16),
          fill: colours[i],
          "font-size": 14,
          "font-weight": 700,
          "text-anchor": image[0] >= 0 ? "start" : "end",
        });
        t.textContent = label;
        g.appendChild(t);
      }
    }

    S.clear(plot);
    plot.appendChild(g);

    const kappa = sigma[1] > 1e-12 ? sigma[0] / sigma[1] : Infinity;
    rSigma1.set(sigma[0].toFixed(3));
    rSigma2.set(sigma[1].toFixed(3));
    rCond.set(Number.isFinite(kappa) ? (kappa > 1e4 ? kappa.toExponential(1) : kappa.toFixed(2)) : "∞");
    rDet.set((sigma[0] * sigma[1]).toFixed(3));

    stageButtons.forEach((b, i) => {
      const on = i === stage;
      b.setAttribute(
        "style",
        "font:inherit;font-size:.8rem;padding:.22rem .6rem;cursor:pointer;border-radius:4px;" +
          "border:1px solid " + (on ? C.purple : C.rule) + ";" +
          "background:" + (on ? C.purple : "#fff") + ";color:" + (on ? "#fff" : C.ink)
      );
    });

    let note = STAGES[stage].note;
    if (sigma[1] < 1e-9) {
      note += " σ₂ is zero here: this matrix flattens the plane onto a line, and nothing recovers what it lost.";
    }
    caption.textContent = note;
  }

  draw();
}

/* ---------------------------------------------------------------------------
 * W2 -- the rank dial on a real photograph.
 *
 * The page ships the leading 100 singular triplets as quantised int16. Moving
 * the slider by one rank is a single rank-one update of an accumulator, so the
 * reconstruction keeps up with a dragged slider on any machine.
 * ------------------------------------------------------------------------ */
function initRankWidget(mountId, payload) {
  const root = document.getElementById(mountId);
  if (!root || !payload) return;

  const S = SVDW;
  const C = S.C;

  const m = payload.m;
  const n = payload.n;
  const maxRank = payload.maxRank;
  const scale = payload.quant;

  function decode(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Int16Array(bytes.buffer);
  }

  // Dequantise once into float arrays. Dividing inside the update loop cost a
  // division per pixel per rank step, which is what made a dragged slider lag.
  function dequantise(b64) {
    const q = decode(b64);
    const out = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = q[i] / scale;
    return out;
  }

  const U = dequantise(payload.u); // rank-major: U[j * m + i]
  const V = dequantise(payload.v); // rank-major: V[j * n + i]
  const sigma = payload.sigma;

  const acc = new Float32Array(m * n);
  let current = 0;

  /* Add or subtract one rank-one outer product. */
  function applyRank(j, sign) {
    const s = sigma[j] * sign;
    const uOff = j * m;
    const vOff = j * n;
    for (let i = 0; i < m; i++) {
      const ui = U[uOff + i] * s;
      if (ui === 0) continue;
      const row = i * n;
      for (let k = 0; k < n; k++) {
        acc[row + k] += ui * V[vOff + k];
      }
    }
  }

  const canvas = S.h("canvas", {
    width: n,
    height: m,
    style:
      "width:100%;max-width:300px;height:auto;display:block;border-radius:4px;border:1px solid " + C.rule,
  });
  const ctx = canvas.getContext("2d");
  const frame = ctx.createImageData(n, m);
  for (let i = 3; i < frame.data.length; i += 4) frame.data[i] = 255;

  function paint() {
    const d = frame.data;
    for (let i = 0; i < m * n; i++) {
      let value = acc[i];
      value = value < 0 ? 0 : value > 255 ? 255 : value;
      const o = i * 4;
      d[o] = d[o + 1] = d[o + 2] = value;
    }
    ctx.putImageData(frame, 0, 0);
  }

  const original = S.h("img", {
    src: payload.original,
    alt: "The original photograph, before any truncation.",
    style:
      "width:100%;max-width:300px;height:auto;display:block;border-radius:4px;border:1px solid " + C.rule,
  });

  function panel(node, title) {
    return S.h("figure", { style: "margin:0;flex:1 1 220px;min-width:180px" }, [
      node,
      S.h("figcaption", {
        text: title,
        style: "font-size:.8rem;color:" + C.muted + ";margin-top:.35rem;text-align:center",
      }),
    ]);
  }

  const captionRank = S.h("figcaption", {
    style: "font-size:.8rem;color:" + C.muted + ";margin-top:.35rem;text-align:center",
  });
  const rankPanel = S.h("figure", { style: "margin:0;flex:1 1 220px;min-width:180px" }, [
    canvas,
    captionRank,
  ]);

  const images = S.h("div", {
    style: "display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;margin:.4rem 0 .8rem",
  }, [panel(original, "original — " + payload.rawBytes.toLocaleString() + " bytes"),
      rankPanel]);

  const stats = S.h("div", { style: "display:flex;flex-wrap:wrap;gap:.2rem;margin:.4rem 0" });
  const rPsnr = S.readout("PSNR", "");
  const rStore = S.readout("stored (int16)", "");
  const rRatio = S.readout("compression", "");
  const rEnergy = S.readout("energy kept", "");
  [rPsnr, rStore, rRatio, rEnergy].forEach((r) => stats.appendChild(r.node));

  const note = S.h("div", {
    style: "font-size:.85rem;color:" + C.muted + ";min-height:2.4em",
  });

  const controls = S.h("div", { style: "margin:.2rem 0 .4rem" });
  const dial = S.slider("rank k", 1, maxRank, 1, 20, (v) => String(v), (v) => setRank(v));
  controls.appendChild(dial.node);

  root.appendChild(controls);
  root.appendChild(images);
  root.appendChild(stats);
  root.appendChild(note);

  function setRank(k) {
    k = Math.max(1, Math.min(maxRank, Math.round(k)));
    while (current < k) applyRank(current++, +1);
    while (current > k) applyRank(--current, -1);
    paint();

    const i = k - 1;
    const storedBytes = 2 * k * (m + n + 1);
    const ratio = payload.rawBytes / storedBytes;

    captionRank.textContent = `rank ${k} — ${storedBytes.toLocaleString()} bytes`;
    rPsnr.set(payload.psnr[i].toFixed(2) + " dB");
    rStore.set((payload.fracI16[i] * 100).toFixed(1) + "% of raw");
    rRatio.set(ratio >= 1 ? ratio.toFixed(2) + "×" : "none (" + ratio.toFixed(2) + "×)");
    rEnergy.set((payload.energy[i] * 100).toFixed(2) + "%");

    if (payload.fracI16[i] >= 1) {
      note.textContent =
        "Past this rank the factors take more space than the pixels they approximate. " +
        "The dial has run out of road.";
    } else if (k <= 5) {
      note.textContent =
        "At this rank the picture is a handful of horizontal and vertical bands — " +
        "the strongest shared structure in the photograph, and nothing else.";
    } else if (k <= payload.elbow) {
      note.textContent =
        "Every extra rank still buys visible detail. This is the steep part of the spectrum.";
    } else {
      note.textContent =
        "Ranks past the elbow at k = " + payload.elbow +
        " sharpen edges and add texture, and each one costs the same " +
        (2 * (m + n + 1)).toLocaleString() + " bytes as the first.";
    }
  }

  dial.sync();
}

document.addEventListener("DOMContentLoaded", () => {
  const script = document.getElementById("svd-data");
  let data = {};
  if (script) {
    try {
      data = JSON.parse(script.textContent);
    } catch (e) {
      console.error("could not parse the SVD widget payload", e);
    }
  }
  initEllipseWidget("widget-ellipse");
  initRankWidget("widget-rank", data.image);
});
