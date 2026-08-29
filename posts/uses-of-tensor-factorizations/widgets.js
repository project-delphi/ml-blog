/* Seven-rank slider for "Uses of Tensor Factorizations".
 *
 * Replaces the static CP / TT figures. Seven discrete ranks, lookup-only
 * (no factorization in the browser). Data is inlined as #tf-data.
 */
"use strict";

const TFW = (function () {
  const NS = "http://www.w3.org/2000/svg";
  const C = {
    ink: "#1F2430",
    muted: "#5F6672",
    rule: "#D8DBE2",
    purple: "#4A3AA7",
    teal: "#2A9D8F",
    coral: "#E07A5F",
    paper: "#F4F5F7",
    white: "#FFFFFF",
  };

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

  function fmtInt(n) {
    return Math.round(n).toLocaleString("en-US");
  }

  function fmtRatio(n) {
    return (Math.round(n * 10) / 10).toFixed(1) + "x";
  }

  function ensureCss() {
    if (document.getElementById("tf-widget-css")) return;
    const s = document.createElement("style");
    s.id = "tf-widget-css";
    s.textContent = [
      ".widget-container{border:1px solid #D8DBE2;border-radius:10px;background:#fff;padding:1.6rem 1.5rem 1.4rem;margin:2rem 0}",
      ".widget-header{display:flex;justify-content:space-between;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.9rem;padding-bottom:0.6rem;border-bottom:1px solid #E9EBEF}",
      ".widget-title{font-weight:700;font-size:1.15rem;color:#1F2430}",
      ".widget-badge{font-size:0.75rem;text-transform:uppercase;font-weight:700;letter-spacing:0.05em;padding:0.2rem 0.6rem;border-radius:12px;background:#4A3AA7;color:#fff}",
      ".widget-note{font-size:0.9rem;color:#5F6672;margin-top:0.85rem;line-height:1.45}",
      ".tf-wrap{font-size:1rem;color:#1F2430}",
      ".tf-tabs{display:flex;gap:0.5rem;margin-bottom:1rem;flex-wrap:wrap}",
      ".tf-tab{border:1px solid #D8DBE2;background:#fff;color:#1F2430;padding:0.45rem 0.95rem;border-radius:999px;cursor:pointer;font:inherit;font-size:0.95rem}",
      ".tf-tab[aria-pressed='true']{background:#4A3AA7;color:#fff;border-color:#4A3AA7}",
      ".tf-stats{display:flex;flex-wrap:wrap;gap:1.15rem;background:#F4F5F7;border:1px solid #D8DBE2;border-radius:8px;padding:0.75rem 1rem;margin-bottom:1rem}",
      ".tf-stat span{display:block;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em;color:#5F6672}",
      ".tf-stat strong{font-size:1.15rem}",
      ".tf-slider{width:100%;accent-color:#4A3AA7;height:1.35rem}",
      ".tf-slider-row{display:flex;align-items:center;gap:0.85rem;margin:0.5rem 0 0.2rem}",
      ".tf-ticks{display:flex;justify-content:space-between;font-size:0.9rem;color:#5F6672;margin:0 0 1.1rem;padding-left:5.5rem}",
      ".tf-tick{cursor:pointer;border:0;background:none;font:inherit;color:#5F6672;padding:0.2rem 0.25rem}",
      ".tf-tick[aria-current='true']{color:#4A3AA7;font-weight:700}",
      ".tf-charts{display:grid;grid-template-columns:minmax(240px,0.9fr) minmax(280px,1.25fr);gap:1.25rem;align-items:stretch}",
      ".tf-charts svg{background:#fff;border:1px solid #E2E6E1;border-radius:8px;width:100%;height:auto}",
      "@media (max-width:720px){.tf-charts{grid-template-columns:1fr}}",
    ].join("");
    document.head.appendChild(s);
  }

  function fmtErr(e) {
    return e.toFixed(3);
  }

  function barChart(dense, used, color, usedLabel) {
    const w = 420, ht = 260, pad = { l: 14, r: 108, t: 36, b: 18 };
    const inner = w - pad.l - pad.r;
    const svg = el("svg", {
      viewBox: "0 0 " + w + " " + ht, width: w, height: ht,
      role: "img", "aria-label": "Weights stored, dense versus factorized",
    });
    svg.appendChild(el("text", {
      x: pad.l, y: 18, fill: C.ink, "font-size": 13, "font-weight": 700,
    }, [document.createTextNode("Weights stored")]));
    const lo = Math.log(Math.max(used * 0.45, 1));
    const hi = Math.log(dense);
    const rows = [
      { label: "uncompressed", value: dense, fill: "#B9BEC9" },
      { label: usedLabel, value: used, fill: color },
    ];
    rows.forEach(function (row, i) {
      const y = 48 + i * 88;
      const bw = Math.max(8, ((Math.log(row.value) - lo) / (hi - lo)) * inner);
      svg.appendChild(el("text", {
        x: pad.l, y: y - 10, fill: C.muted, "font-size": 12,
      }, [document.createTextNode(row.label)]));
      svg.appendChild(el("rect", {
        x: pad.l, y: y, width: bw, height: 28, rx: 4, fill: row.fill,
      }));
      svg.appendChild(el("text", {
        x: pad.l + bw + 8, y: y + 20, fill: C.ink, "font-size": 14, "font-weight": 600,
      }, [document.createTextNode(fmtInt(row.value))]));
    });
    svg.appendChild(el("text", {
      x: pad.l, y: ht - 6, fill: C.muted, "font-size": 11,
    }, [document.createTextNode("log scale, so both bars read")]));
    return svg;
  }

  function errorChart(ranks, errors, idx, trueRank, color) {
    const w = 640, ht = 280;
    const pad = { l: 58, r: 36, t: 40, b: 48 };
    const innerW = w - pad.l - pad.r;
    const innerH = ht - pad.t - pad.b;
    const xMax = ranks.length - 1;
    const yMax = 1;
    function xPos(i) {
      return pad.l + (i / (xMax || 1)) * innerW;
    }
    function yPos(e) {
      return pad.t + innerH - (e / yMax) * innerH;
    }
    const svg = el("svg", {
      viewBox: "0 0 " + w + " " + ht, width: w, height: ht,
      role: "img",
      "aria-label": "Relative error at seven ranks",
    });
    svg.appendChild(el("text", {
      x: pad.l, y: 18, fill: C.ink, "font-size": 13, "font-weight": 700,
    }, [document.createTextNode("Toy relative error")]));

    [0, 0.25, 0.5, 0.75, 1].forEach(function (tick) {
      const y = yPos(tick);
      svg.appendChild(el("line", {
        x1: pad.l, x2: pad.l + innerW, y1: y, y2: y,
        stroke: C.paper, "stroke-width": 1,
      }));
      svg.appendChild(el("text", {
        x: pad.l - 8, y: y + 4, fill: C.muted,
        "font-size": 11, "text-anchor": "end",
      }, [document.createTextNode(tick.toFixed(2))]));
    });
    svg.appendChild(el("line", {
      x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + innerH,
      stroke: C.rule, "stroke-width": 1,
    }));
    svg.appendChild(el("line", {
      x1: pad.l, x2: pad.l + innerW, y1: pad.t + innerH, y2: pad.t + innerH,
      stroke: C.rule, "stroke-width": 1,
    }));

    const trueIdx = ranks.indexOf(trueRank);
    if (trueIdx >= 0) {
      const tx = xPos(trueIdx);
      svg.appendChild(el("rect", {
        x: tx - 8, y: pad.t, width: 16, height: innerH,
        fill: "#E9C46A", opacity: "0.22",
      }));
      svg.appendChild(el("line", {
        x1: tx, x2: tx, y1: pad.t, y2: pad.t + innerH,
        stroke: "#B08918", "stroke-width": 1.75, "stroke-dasharray": "5 4",
      }));
      const built = "toy built at rank " + trueRank;
      const textX = trueIdx <= ranks.length - 3 ? tx + 12 : tx - 12;
      svg.appendChild(el("text", {
        x: textX, y: pad.t + 14, fill: "#8A6B12",
        "font-size": 12, "font-weight": 700,
        "text-anchor": trueIdx <= ranks.length - 3 ? "start" : "end",
      }, [document.createTextNode(built)]));
    }

    let line = "";
    let area = "M " + xPos(0) + " " + (pad.t + innerH) + " ";
    ranks.forEach(function (_r, i) {
      line += (i === 0 ? "M" : "L") + xPos(i) + " " + yPos(errors[i]) + " ";
      area += "L " + xPos(i) + " " + yPos(errors[i]) + " ";
    });
    area += "L " + xPos(xMax) + " " + (pad.t + innerH) + " Z";
    svg.appendChild(el("path", { d: area, fill: color, opacity: "0.10" }));
    svg.appendChild(el("path", {
      d: line.trim(), fill: "none", stroke: color, "stroke-width": 2.5,
    }));

    ranks.forEach(function (r, i) {
      const x = xPos(i);
      const y = yPos(errors[i]);
      const selected = i === idx;
      svg.appendChild(el("circle", {
        cx: x, cy: y, r: selected ? 7 : 4.5,
        fill: selected ? color : C.white, stroke: color, "stroke-width": 2,
      }));
      let ly;
      if (selected) {
        ly = y < pad.t + 24 ? y + 22 : y - 16;
      } else if (i % 2 === 1) {
        ly = Math.min(y + 18, pad.t + innerH - 6);
      } else {
        ly = y < pad.t + 18 ? y + 18 : y - 12;
      }
      svg.appendChild(el("text", {
        x: x, y: ly, fill: selected ? color : C.ink,
        "font-size": selected ? 14 : 11,
        "font-weight": selected ? 700 : 500,
        "text-anchor": "middle",
      }, [document.createTextNode(fmtErr(errors[i]))]));
      svg.appendChild(el("text", {
        x: x, y: pad.t + innerH + 18, fill: selected ? C.ink : C.muted,
        "font-size": 12, "font-weight": selected ? 700 : 400,
        "text-anchor": "middle",
      }, [document.createTextNode(String(r))]));
    });

    svg.appendChild(el("text", {
      x: pad.l + innerW / 2, y: ht - 6, fill: C.muted,
      "font-size": 12, "text-anchor": "middle",
    }, [document.createTextNode("rank")]));
    svg.appendChild(el("text", {
      x: 14, y: pad.t + innerH / 2, fill: C.muted, "font-size": 12,
      transform: "rotate(-90 14 " + (pad.t + innerH / 2) + ")",
      "text-anchor": "middle",
    }, [document.createTextNode("relative error")]));
    return svg;
  }

  function init(containerId, payload) {
    const root = document.getElementById(containerId);
    if (!root || !payload || !payload.cp || !payload.tt) return;
    ensureCss();
    clear(root);
    root.classList.add("tf-wrap");

    let mode = "cp";
    const series = { cp: payload.cp, tt: payload.tt };
    const idx0 = {
      cp: payload.cp.ranks.indexOf(payload.cp.true_rank),
      tt: payload.tt.ranks.indexOf(payload.tt.true_rank),
    };
    const idx = { cp: idx0.cp < 0 ? 0 : idx0.cp, tt: idx0.tt < 0 ? 0 : idx0.tt };

    const tabs = h("div", { class: "tf-tabs" });
    const tabCp = h("button", {
      class: "tf-tab", type: "button", text: "CP convolution",
      "aria-pressed": "true",
    });
    const tabTt = h("button", {
      class: "tf-tab", type: "button", text: "TT-matrix",
      "aria-pressed": "false",
    });
    tabs.appendChild(tabCp);
    tabs.appendChild(tabTt);

    const stats = h("div", { class: "tf-stats" });
    const sliderRow = h("div", { class: "tf-slider-row" });
    const sliderLabel = h("label", { text: "rank" });
    sliderLabel.setAttribute("for", "tf-rank");
    const slider = h("input", {
      id: "tf-rank", class: "tf-slider", type: "range", min: "0", step: "1",
    });
    sliderRow.appendChild(sliderLabel);
    sliderRow.appendChild(slider);
    const ticks = h("div", { class: "tf-ticks" });

    const charts = h("div", { class: "tf-charts" });
    const note = h("div", { class: "widget-note" });

    root.appendChild(tabs);
    root.appendChild(stats);
    root.appendChild(sliderRow);
    root.appendChild(ticks);
    root.appendChild(charts);
    root.appendChild(note);

    function current() {
      const s = series[mode];
      const i = idx[mode];
      const head = s.headline;
      const headUsed = (head.params && head.params[i] != null)
        ? head.params[i]
        : (mode === "cp" ? head.cp_params : head.tt_params);
      return {
        rank: s.ranks[i],
        params: s.params[i],
        error: s.rel_error[i],
        dense: s.dense_params,
        headDense: head.dense_params,
        headUsed: headUsed,
        trueRank: s.true_rank,
        ranks: s.ranks,
        errors: s.rel_error,
        color: mode === "cp" ? C.purple : C.teal,
        name: mode === "cp" ? "CP" : "TT",
        headLabel: mode === "cp" ? "VGG-16 conv5" : "4096×4096 W_O",
      };
    }

    function render() {
      const cur = current();
      slider.max = String(cur.ranks.length - 1);
      slider.value = String(idx[mode]);
      sliderLabel.textContent = cur.name + " rank " + cur.rank;
      tabCp.setAttribute("aria-pressed", mode === "cp" ? "true" : "false");
      tabTt.setAttribute("aria-pressed", mode === "tt" ? "true" : "false");
      clear(stats);
      [
        ["rank", String(cur.rank)],
        [cur.headLabel, fmtInt(cur.headUsed) + " / " + fmtInt(cur.headDense)],
        ["how much smaller", fmtRatio(cur.headDense / cur.headUsed)],
        ["toy relative error", cur.error.toFixed(3)],
      ].forEach(function (pair) {
        stats.appendChild(h("div", { class: "tf-stat" }, [
          h("span", { text: pair[0] }),
          h("strong", { text: pair[1] }),
        ]));
      });
      clear(ticks);
      cur.ranks.forEach(function (r, i) {
        const b = h("button", {
          class: "tf-tick", type: "button", text: String(r),
        });
        if (i === idx[mode]) b.setAttribute("aria-current", "true");
        b.addEventListener("click", function () {
          idx[mode] = i;
          render();
        });
        ticks.appendChild(b);
      });
      clear(charts);
      charts.appendChild(barChart(
        cur.headDense, cur.headUsed, cur.color,
        mode === "cp" ? "after CP" : "after TT"
      ));
      charts.appendChild(errorChart(
        cur.ranks, cur.errors, idx[mode], cur.trueRank, cur.color
      ));
      note.textContent = mode === "cp"
        ? "Weight bar: VGG-16 conv5, closed form. Error curve: 64-channel toy. Gold band: the toy was generated at rank 16 — that is where leftover error hits the noise."
        : "Weight bar: transformer W_O at width 4096, closed form. Error curve: 256×256 toy. Gold band: the toy was generated at rank 4 — that is where leftover error hits the noise.";
    }

    tabCp.addEventListener("click", function () { mode = "cp"; render(); });
    tabTt.addEventListener("click", function () { mode = "tt"; render(); });
    slider.addEventListener("input", function () {
      idx[mode] = parseInt(slider.value, 10);
      render();
    });
    render();
  }

  return { init: init };
})();

document.addEventListener("DOMContentLoaded", function () {
  const script = document.getElementById("tf-data");
  let data = {};
  if (script) {
    try { data = JSON.parse(script.textContent); }
    catch (err) { console.error("could not parse tensor-factorization widget payload", err); }
  }
  TFW.init("tf-widget", data);
});
