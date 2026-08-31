/* The t-inverse, taken apart stage by stage.
 *
 * Dependency-free: no CDN, no module loader, no framework. Everything is
 * inline SVG built through the DOM, so the page works from a file:// copy as
 * well as over HTTP.
 *
 * Lookup-only. Every state was precomputed by src/export_widget_data.py and
 * arrives inlined as <script id="ti-data" type="application/json">. There is
 * no linear algebra in the browser.
 */
"use strict";

const TIW = (function () {
  const NS = "http://www.w3.org/2000/svg";
  const C = {
    ink: "#1F2430",
    muted: "#5F6672",
    rule: "#D8DBE2",
    purple: "#4A3AA7",
    teal: "#2A9D8F",
    coral: "#E07A5F",
    gold: "#E8A33D",
    paper: "#F4F5F7",
    white: "#FFFFFF",
  };

  const STAGES = [
    {
      label: "1. Frontal slices",
      blurb:
        "The tensor as it arrives: six 8x8 frontal slices. Nothing has been " +
        "transformed yet, and the slices are not independent of one another.",
    },
    {
      label: "2. FFT along mode 3",
      blurb:
        "The same tensor after a Fourier transform along the third mode. This " +
        "is the step that makes the whole thing work: after the transform the " +
        "slices decouple, so one tensor inverse becomes six independent matrix " +
        "inverses. The condition number under each slice says how well behaved " +
        "each of those six problems is.",
    },
    {
      label: "3. Invert each slice",
      blurb:
        "Six separate matrix inverses, one per Fourier slice. Slices 2 and 4 " +
        "are a conjugate pair, so the conditioning control moves both together.",
    },
    {
      label: "4. Back, and verify",
      blurb:
        "Inverse FFT returns a real tensor, the t-inverse. The lower strip is " +
        "the check: A * A-inverse should be the identity tensor, which is the " +
        "identity matrix in frontal slice 0 and zeros in the other five.",
    },
  ];

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
      else if (k === "style" && typeof attrs[k] === "object") {
        Object.assign(e.style, attrs[k]);
      } else e.setAttribute(k, attrs[k]);
    }
    for (const c of kids || []) e.appendChild(c);
    return e;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function fmtSci(x) {
    if (!isFinite(x)) return "-";
    if (x === 0) return "0";
    if (x >= 0.01 && x < 10000) return (Math.round(x * 100) / 100).toString();
    const e = Math.floor(Math.log10(Math.abs(x)));
    const m = x / Math.pow(10, e);
    return (Math.round(m * 10) / 10) + "e" + e;
  }

  /* Diverging ramp for signed values, sequential for magnitudes. Both are
   * legible in greyscale, which matters because the strips are small. */
  function colourSigned(v) {
    const t = Math.max(-1, Math.min(1, v));
    if (t >= 0) return mix([244, 245, 247], [74, 58, 167], t);
    return mix([244, 245, 247], [224, 122, 95], -t);
  }

  function colourMag(v) {
    const t = Math.max(0, Math.min(1, Math.abs(v)));
    return mix([247, 247, 244], [42, 157, 143], Math.sqrt(t));
  }

  function mix(a, b, t) {
    const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
    return "rgb(" + c.join(",") + ")";
  }

  function ensureCss() {
    if (document.getElementById("ti-widget-css")) return;
    const s = document.createElement("style");
    s.id = "ti-widget-css";
    s.textContent = [
      ".widget-container{border:1px solid #D8DBE2;border-radius:10px;background:#fff;padding:1.6rem 1.5rem 1.4rem;margin:2rem 0}",
      ".widget-header{display:flex;justify-content:space-between;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.9rem;padding-bottom:0.6rem;border-bottom:1px solid #E9EBEF}",
      ".widget-title{font-weight:700;font-size:1.15rem;color:#1F2430}",
      ".widget-badge{font-size:0.75rem;text-transform:uppercase;font-weight:700;letter-spacing:0.05em;padding:0.2rem 0.6rem;border-radius:12px;background:#4A3AA7;color:#fff}",
      ".widget-note{font-size:0.9rem;color:#5F6672;margin-top:0.85rem;line-height:1.45}",
      ".ti-wrap{font-size:1rem;color:#1F2430}",
      ".ti-tabs{display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap}",
      ".ti-tab{border:1px solid #D8DBE2;background:#fff;color:#1F2430;padding:0.45rem 0.95rem;border-radius:999px;cursor:pointer;font:inherit;font-size:0.95rem}",
      ".ti-tab[aria-pressed='true']{background:#4A3AA7;color:#fff;border-color:#4A3AA7}",
      ".ti-blurb{font-size:0.95rem;color:#5F6672;line-height:1.5;margin-bottom:1rem;min-height:3.2em}",
      ".ti-controls{display:flex;flex-wrap:wrap;gap:1.4rem;align-items:flex-end;margin-bottom:1.1rem}",
      ".ti-control{flex:1 1 16rem;min-width:14rem}",
      ".ti-control label{display:block;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em;color:#5F6672;margin-bottom:0.35rem}",
      ".ti-slider{width:100%;accent-color:#4A3AA7;height:1.35rem}",
      ".ti-radio{display:flex;gap:0.5rem}",
      ".ti-radio button{border:1px solid #D8DBE2;background:#fff;color:#1F2430;padding:0.4rem 0.9rem;border-radius:6px;cursor:pointer;font:inherit;font-size:0.95rem}",
      ".ti-radio button[aria-pressed='true']{background:#2A9D8F;color:#fff;border-color:#2A9D8F}",
      ".ti-stats{display:flex;flex-wrap:wrap;gap:1.15rem;background:#F4F5F7;border:1px solid #D8DBE2;border-radius:8px;padding:0.75rem 1rem;margin-bottom:1rem}",
      ".ti-stat span{display:block;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em;color:#5F6672}",
      ".ti-stat strong{font-size:1.1rem;font-variant-numeric:tabular-nums}",
      ".ti-strip{overflow-x:auto;margin-bottom:0.6rem}",
      ".ti-striplabel{font-size:0.8rem;color:#5F6672;margin-bottom:0.3rem}",
      "@media (max-width:720px){.ti-controls{flex-direction:column;align-items:stretch}}",
    ].join("\n");
    document.head.appendChild(s);
  }

  /* One 8x8 slice as an SVG group of cells. */
  function sliceSvg(values, n, px, signed) {
    const g = el("g", {});
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const v = values[r * n + c];
        g.appendChild(
          el("rect", {
            x: c * px,
            y: r * px,
            width: px,
            height: px,
            fill: signed ? colourSigned(v) : colourMag(v),
          })
        );
      }
    }
    g.appendChild(
      el("rect", {
        x: 0,
        y: 0,
        width: n * px,
        height: n * px,
        fill: "none",
        stroke: C.rule,
        "stroke-width": 1,
      })
    );
    return g;
  }

  /* A row of six slices, each captioned. */
  function stripSvg(block, n, opts) {
    const px = 13;
    const side = n * px;
    const gap = 18;
    const top = 16;
    const bottom = opts.captions ? 34 : 18;
    const w = block.slices.length * (side + gap) - gap + 2;
    const svg = el("svg", {
      width: w,
      height: side + top + bottom,
      viewBox: `0 0 ${w} ${side + top + bottom}`,
      role: "img",
      "aria-label": opts.alt || "",
    });
    block.slices.forEach((values, k) => {
      const x = k * (side + gap) + 1;
      const g = sliceSvg(values, n, px, opts.signed);
      g.setAttribute("transform", `translate(${x},${top})`);
      svg.appendChild(g);
      const head = el("text", {
        x: x,
        y: 11,
        fill: opts.highlight && opts.highlight.indexOf(k) >= 0 ? C.coral : C.muted,
        "font-size": 10,
        "font-weight": opts.highlight && opts.highlight.indexOf(k) >= 0 ? 700 : 400,
      });
      head.textContent = opts.headPrefix + k;
      svg.appendChild(head);
      if (opts.captions) {
        const cap = el("text", {
          x: x,
          y: top + side + 13,
          fill: C.muted,
          "font-size": 10,
        });
        cap.textContent = opts.captions[k];
        svg.appendChild(cap);
      }
    });
    return svg;
  }

  function labelledStrip(label, svg) {
    return h("div", { class: "ti-strip" }, [
      h("div", { class: "ti-striplabel", text: label }),
      svg,
    ]);
  }

  function init(mountId, data) {
    const root = document.getElementById(mountId);
    if (!root || !data || !data.levels) return;
    ensureCss();

    const n = data.n;
    const state = { stage: 0, level: 0, solver: "inv" };

    const wrap = h("div", { class: "ti-wrap" });
    const tabs = h("div", { class: "ti-tabs" });
    const blurb = h("div", { class: "ti-blurb" });
    const controls = h("div", { class: "ti-controls" });
    const stats = h("div", { class: "ti-stats" });
    const canvas = h("div", {});

    const tabButtons = STAGES.map((s, i) =>
      h("button", { class: "ti-tab", type: "button", text: s.label }, [])
    );
    tabButtons.forEach((b, i) => {
      b.addEventListener("click", () => {
        state.stage = i;
        render();
      });
      tabs.appendChild(b);
    });

    const slider = h("input", {
      class: "ti-slider",
      type: "range",
      min: 0,
      max: data.levels.length - 1,
      step: 1,
      value: 0,
    });
    slider.addEventListener("input", () => {
      state.level = Number(slider.value);
      render();
    });
    const sliderLabel = h("label", { text: "Conditioning" });
    controls.appendChild(
      h("div", { class: "ti-control" }, [sliderLabel, slider])
    );

    const solverButtons = data.solvers.map((s) =>
      h("button", {
        type: "button",
        text: s === "inv" ? "inv (true inverse)" : "pinv (pseudoinverse)",
      })
    );
    const radio = h("div", { class: "ti-radio" }, solverButtons);
    solverButtons.forEach((b, i) => {
      b.addEventListener("click", () => {
        state.solver = data.solvers[i];
        render();
      });
    });
    controls.appendChild(
      h("div", { class: "ti-control" }, [
        h("label", { text: "Solver" }),
        radio,
      ])
    );

    wrap.appendChild(tabs);
    wrap.appendChild(blurb);
    wrap.appendChild(controls);
    wrap.appendChild(stats);
    wrap.appendChild(canvas);
    clear(root);
    root.appendChild(wrap);

    function statTile(name, value, tone) {
      return h("div", { class: "ti-stat" }, [
        h("span", { text: name }),
        h("strong", { text: value, style: tone ? { color: tone } : {} }),
      ]);
    }

    function render() {
      const lvl = data.by_level[state.level];
      const st = data.by_state[state.solver][state.level];
      const bent = data.bent_slice;
      const mirror = (data.slices - bent) % data.slices;
      const highlight = [bent, mirror];
      const kappa = lvl.conds[bent];

      tabButtons.forEach((b, i) =>
        b.setAttribute("aria-pressed", String(i === state.stage))
      );
      solverButtons.forEach((b, i) =>
        b.setAttribute("aria-pressed", String(data.solvers[i] === state.solver))
      );
      slider.setAttribute("value", String(state.level));
      sliderLabel.textContent =
        "Conditioning — slice " + bent + " and " + mirror + ", kappa " + fmtSci(kappa);
      blurb.textContent = STAGES[state.stage].blurb;

      clear(stats);
      stats.appendChild(
        statTile(
          "kappa, slice " + bent,
          fmtSci(kappa),
          kappa > 1e8 ? C.coral : C.ink
        )
      );
      stats.appendChild(
        statTile(
          "identity residual",
          fmtSci(st.residual),
          st.residual > 1e-6 ? C.coral : C.ink
        )
      );
      stats.appendChild(
        statTile(
          "largest entry of X",
          fmtSci(st.max_abs),
          st.max_abs > 1e3 ? C.coral : C.ink
        )
      );
      stats.appendChild(
        statTile(
          "solve error",
          fmtSci(st.solve_error),
          st.solve_error > 1 ? C.coral : C.teal
        )
      );

      clear(canvas);
      const conds = lvl.conds.map((c) => "k " + fmtSci(c));
      if (state.stage === 0) {
        canvas.appendChild(
          labelledStrip(
            "A — frontal slices, scale " + fmtSci(lvl.spatial.scale),
            stripSvg(lvl.spatial, n, {
              signed: true,
              headPrefix: "slice ",
              alt: "Six frontal slices of the tensor A.",
            })
          )
        );
      } else if (state.stage === 1) {
        canvas.appendChild(
          labelledStrip(
            "FFT(A) along mode 3 — magnitudes, scale " + fmtSci(lvl.fourier.scale),
            stripSvg(lvl.fourier, n, {
              signed: false,
              headPrefix: "slice ",
              captions: conds,
              highlight: highlight,
              alt: "Six Fourier slices with their condition numbers.",
            })
          )
        );
      } else if (state.stage === 2) {
        canvas.appendChild(
          labelledStrip(
            "Each Fourier slice inverted — magnitudes, scale " +
              fmtSci(st.finverse.scale),
            stripSvg(st.finverse, n, {
              signed: false,
              headPrefix: "slice ",
              captions: conds,
              highlight: highlight,
              alt: "The six inverted Fourier slices.",
            })
          )
        );
      } else {
        canvas.appendChild(
          labelledStrip(
            "A-inverse — back in the spatial domain, scale " +
              fmtSci(st.inverse.scale),
            stripSvg(st.inverse, n, {
              signed: true,
              headPrefix: "slice ",
              alt: "The t-inverse of A, six frontal slices.",
            })
          )
        );
        canvas.appendChild(
          labelledStrip(
            "A * A-inverse — should be the identity tensor, scale " +
              fmtSci(st.product.scale),
            stripSvg(st.product, n, {
              signed: true,
              headPrefix: "slice ",
              alt: "A times its t-inverse, six frontal slices.",
            })
          )
        );
      }
    }

    render();
  }

  return { init: init };
})();

document.addEventListener("DOMContentLoaded", function () {
  const script = document.getElementById("ti-data");
  let data = {};
  if (script) {
    try {
      data = JSON.parse(script.textContent);
    } catch (err) {
      console.error("could not parse tensor-inverse widget payload", err);
    }
  }
  TIW.init("ti-widget", data);
});
