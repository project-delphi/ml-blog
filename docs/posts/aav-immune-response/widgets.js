/* Widget for "What the Immune System Sees in an AAV Vector".
 *
 * One widget: the route a recombinant AAV particle takes from an intravenous
 * infusion to a hepatocyte nucleus, drawn as attrition across six stations,
 * with the four host immune arms switchable on and off.
 *
 * Dependency-free -- no CDN, no module loader, no framework -- so it works
 * from a plain file:// page as well as over HTTP. Observable JS does not work
 * with the Quarto installed here (1.6.40), which is why this and the other
 * widgets on the blog are hand-rolled.
 *
 * The model (AAVModel) touches no DOM, so `node` can exercise it directly.
 * The numbers are illustrative, not fitted: see the "Model" section of the post.
 */
"use strict";

var AAV = (function () {
  var NS = "http://www.w3.org/2000/svg";

  var C = {
    ink: "#16202B",
    muted: "#5C6B7A",
    rule: "#D5DCE3",
    paper: "#F7F9FB",
    white: "#FFFFFF",
    vector: "#4A3AA7",
    innate: "#E69F00",
    antibody: "#0072B2",
    tcell: "#A8437A",
    complement: "#009E73",
    traffic: "#9AA6B2",
    blood: "#FDEEF0",
    bloodEdge: "#E4B4BC",
    nucleus: "#EFEAFB"
  };

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, kids) {
    var e = document.createElementNS(NS, tag), k;
    for (k in (attrs || {})) e.setAttribute(k, attrs[k]);
    (kids || []).forEach(function (c) { e.appendChild(c); });
    return e;
  }

  function h(tag, attrs, kids) {
    var e = document.createElement(tag), k;
    for (k in (attrs || {})) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else if (k === "style" && typeof attrs[k] === "object") Object.assign(e.style, attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { e.appendChild(c); });
    return e;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  // Small deterministic PRNG, so the same settings always draw the same
  // picture and a reader comparing two states is comparing the model, not
  // the noise.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ------------------------------------------------------------------ model
  var DOSES = [1e11, 3e11, 1e12, 3e12, 1e13, 3e13, 1e14];
  var DOSE_REF_INDEX = 4; // 1e13 vg/kg, the reference for the expression index
  var TITRES = ["seronegative", "1:5", "1:20", "1:100", "1:1000"];
  var TITRE_F = [0.05, 0.15, 0.35, 0.65, 1.0];

  // Baseline trafficking survival, one per station, in a patient with no
  // active immune response. The product is the "only a few percent of the
  // dose reaches a nucleus" figure the post quotes.
  var BASE = [0.92, 0.72, 0.58, 0.46, 0.60, 0.72];

  var STATIONS = [
    {
      key: "circulation", name: "Bloodstream",
      x: 96, y: 96,
      sub: "First minutes after infusion",
      body: "The full dose is in plasma as intact particles. Nothing has been sensed from the inside yet, because nothing has opened. The only surface on offer is the capsid shell.",
      arms: ["complement"],
      detail: "Complement settles on the capsid surface and the cascade begins to amplify. Antibody already bound to a capsid starts the classical pathway here too."
    },
    {
      key: "sinusoid", name: "Sinusoid wall",
      x: 258, y: 96,
      sub: "Endothelium and Kupffer cells",
      body: "An intravenous dose meets the liver first. Kupffer cells, the macrophages lining the sinusoid, take up a large share of any particle passing through, immune response or not.",
      arms: ["complement", "antibody"],
      detail: "Binding antibodies opsonise capsids and Fc-gamma receptors on Kupffer cells grip the Fc tail, so clearance rises steeply with total antibody. C5b-9 pores open on the endothelium itself."
    },
    {
      key: "surface", name: "Cell surface",
      x: 450, y: 250,
      sub: "Glycan attachment, then AAVR",
      body: "The capsid binds a cell-surface glycan, then the protein receptor AAVR, and is pulled in by endocytosis. This is the step that decides tropism.",
      arms: ["antibody"],
      detail: "Neutralising antibodies work here and only here. They sit physically over the attachment sites, so the particle is intact and simply cannot dock."
    },
    {
      key: "endosome", name: "Endosome",
      x: 580, y: 330, labelDy: 74,
      sub: "Acidification and escape",
      body: "The compartment acidifies. The VP1 unique region flips out and its phospholipase activity breaks the endosomal membrane so the particle can escape. Most particles fail here.",
      arms: ["innate"],
      detail: "As the capsid begins to open, TLR9 in the endosomal membrane reaches the unmethylated CpG in the single-stranded genome and signals through MyD88 to IRF7, driving type I interferon."
    },
    {
      key: "cytosol", name: "Cytosol",
      x: 710, y: 400,
      sub: "Proteasome against trafficking",
      body: "An escaped capsid races the ubiquitin-proteasome system to the nuclear pore. Whatever loses is shredded into peptides, and those peptides are what the T-cell arm will later use.",
      arms: ["innate"],
      detail: "Interferon raised at the previous station suppresses transgene expression and matures antigen-presenting cells, which is how the innate arm licenses the adaptive one."
    },
    {
      key: "nucleus", name: "Nucleus",
      x: 855, y: 405, labelDy: 88,
      sub: "Uncoating and second-strand synthesis",
      body: "The genome is released, the complementary strand is made, and the ends join into a circular episome that stays outside the chromosome. Only now is there a working copy of the therapeutic gene.",
      arms: [],
      detail: "No immune arm acts inside the nucleus. Everything that reduces expression from here on acts on the cell, not on the genome."
    }
  ];

  var TCELL_CARD = {
    key: "tcell", name: "CD8⁺ T cells",
    sub: "Four to eight weeks later",
    body: "Capsid peptides from every particle the proteasome destroyed are displayed on MHC class I. Capsid-specific CD8⁺ T cells expand, recognise them, and kill the cell.",
    detail: "This removes no particles. It removes cells that already received them, which is why the nucleus count barely moves while the expression index collapses."
  };

  var ARMS = [
    { key: "innate", label: "Innate sensing", note: "TLR2 / TLR9", color: C.innate },
    { key: "antibody", label: "Antibodies", note: "NAb / BAb", color: C.antibody },
    { key: "tcell", label: "CD8⁺ T cells", note: "MHC class I", color: C.tcell },
    { key: "complement", label: "Complement", note: "C3a / C5a / MAC", color: C.complement }
  ];

  /* Survival fractions per station, plus the severities that drive the lamps.
     Pure function of state: no DOM, no globals. */
  function AAVModel(state) {
    var d = state.doseIndex / (DOSES.length - 1);          // 0 at 1e11, 1 at 1e14
    var titre = state.arms.antibody ? TITRE_F[state.titreIndex] : 0;

    // Complement amplifies, so it is steeply non-linear in dose, and a bound
    // antibody starts the classical pathway at a much lower threshold.
    var sevComp = state.arms.complement
      ? clamp(0.12 + 0.72 * Math.pow(d, 1.7) + 0.30 * titre, 0, 1) : 0;
    // Innate sensing scales with particles sensed per cell: roughly linear.
    var sevInn = state.arms.innate ? clamp(0.22 + 0.62 * d, 0, 1) : 0;
    // T-cell killing needs enough capsid presented before it takes hold.
    var sevT = state.arms.tcell ? clamp(0.08 + 0.92 * Math.pow(d, 1.25), 0, 1) : 0;

    // Per station, an ordered list of independent survival factors. The order
    // only affects which arm gets the credit for a given lost particle.
    var factors = [
      [{ cause: "traffic", keep: BASE[0] },
       { cause: "complement", keep: 1 - 0.30 * sevComp }],
      [{ cause: "traffic", keep: BASE[1] },
       { cause: "antibody", keep: 1 - 0.60 * titre },
       { cause: "complement", keep: 1 - 0.35 * sevComp }],
      [{ cause: "traffic", keep: BASE[2] },
       { cause: "antibody", keep: 1 - 0.95 * titre }],
      [{ cause: "traffic", keep: BASE[3] },
       { cause: "innate", keep: 1 - 0.25 * sevInn }],
      [{ cause: "traffic", keep: BASE[4] },
       { cause: "innate", keep: 1 - 0.20 * sevInn }],
      [{ cause: "traffic", keep: BASE[5] }]
    ];

    var perStation = factors.map(function (fs) {
      return fs.reduce(function (a, f) { return a * f.keep; }, 1);
    });
    var reaching = perStation.reduce(function (a, p) { return a * p; }, 1);

    var baseProduct = BASE.reduce(function (a, p) { return a * p; }, 1);
    var exprSurvival = 1 - 0.88 * sevT;
    var doseScale = DOSES[state.doseIndex] / DOSES[DOSE_REF_INDEX];
    var expression = 100 * doseScale * (reaching / baseProduct) * exprSurvival;

    return {
      factors: factors,
      perStation: perStation,
      reaching: reaching,
      baseProduct: baseProduct,
      expression: expression,
      absolute: DOSES[state.doseIndex] * reaching,
      sev: { complement: sevComp, innate: sevInn, tcell: sevT },
      lamps: [
        { key: "ifn", label: "IFN-α/β", level: sevInn },
        { key: "cyto", label: "TNF-α, IL-6", level: sevInn * 0.9 },
        { key: "alt", label: "ALT / AST", level: sevT },
        { key: "comp", label: "C3a, C5a, platelets", level: sevComp }
      ]
    };
  }

  /* Assign every particle a fate: the station it dies at and the arm that
     killed it, or station 6 meaning it reached the nucleus. */
  function assignFates(model, n, seed) {
    var rand = mulberry32(seed);
    var fates = [];
    for (var i = 0; i < n; i++) {
      var died = false;
      for (var s = 0; s < model.factors.length && !died; s++) {
        var fs = model.factors[s];
        for (var f = 0; f < fs.length; f++) {
          if (rand() > fs[f].keep) {
            fates.push({ station: s, cause: fs[f].cause, jitter: rand() });
            died = true;
            break;
          }
        }
      }
      if (!died) fates.push({ station: 6, cause: "arrived", jitter: rand() });
    }
    return fates;
  }

  // --------------------------------------------------------------- geometry
  var W = 960, H = 540;
  var ROUTE = STATIONS.map(function (s) { return [s.x, s.y]; });

  function routeLengths() {
    var segs = [], total = 0;
    for (var i = 1; i < ROUTE.length; i++) {
      var dx = ROUTE[i][0] - ROUTE[i - 1][0];
      var dy = ROUTE[i][1] - ROUTE[i - 1][1];
      var L = Math.sqrt(dx * dx + dy * dy);
      segs.push(L); total += L;
    }
    return { segs: segs, total: total };
  }

  /* Position at a continuous station coordinate u, where u = 2.5 means
     halfway between station 2 and station 3. */
  function pointAt(u) {
    var i = Math.floor(u);
    if (i < 0) return [ROUTE[0][0] - 70 - (-u) * 20, ROUTE[0][1]];
    if (i >= ROUTE.length - 1) {
      var last = ROUTE[ROUTE.length - 1];
      return [last[0], last[1]];
    }
    var t = u - i;
    return [
      ROUTE[i][0] + t * (ROUTE[i + 1][0] - ROUTE[i][0]),
      ROUTE[i][1] + t * (ROUTE[i + 1][1] - ROUTE[i][1])
    ];
  }

  function causeColor(cause) {
    if (cause === "arrived") return C.vector;
    if (cause === "traffic") return C.traffic;
    return C[cause] || C.traffic;
  }

  // ----------------------------------------------------------------- scene
  function txt(x, y, s, opts) {
    opts = opts || {};
    var t = el("text", {
      x: x, y: y,
      "font-size": opts.size || 14,
      "font-family": "system-ui, -apple-system, Segoe UI, sans-serif",
      "font-weight": opts.weight || 400,
      fill: opts.fill || C.muted,
      "text-anchor": opts.anchor || "start"
    });
    t.textContent = s;
    return t;
  }

  function buildScene(svg, state, refs) {
    var g = el("g", {});

    // --- sinusoid: a blood channel running the width of the panel --------
    g.appendChild(el("rect", { x: 0, y: 40, width: W, height: 110, fill: C.blood }));
    g.appendChild(el("path", { d: "M0 40 H" + W, stroke: C.bloodEdge, "stroke-width": 3 }));
    g.appendChild(txt(14, 30, "sinusoid lumen", { size: 14 }));

    // endothelium: the floor of the channel and the first barrier crossed
    g.appendChild(el("path", {
      d: "M0 150 H" + W, stroke: C.ink, "stroke-width": 6, "stroke-linecap": "round"
    }));
    g.appendChild(txt(14, 172, "endothelium", { size: 14 }));
    refs.macPores = el("g", { opacity: 0 });
    for (var p = 0; p < 12; p++) {
      refs.macPores.appendChild(el("path", {
        d: "M" + (48 + p * 76) + " 142 v16",
        stroke: C.complement, "stroke-width": 5, "stroke-linecap": "round"
      }));
    }
    g.appendChild(refs.macPores);

    // Kupffer cell, sitting in the lumen where the dose passes
    g.appendChild(el("path", {
      d: "M356 112 q20 -32 50 -14 q30 -16 40 14 q16 26 -14 34 q-32 16 -58 -2 q-28 -8 -18 -32 z",
      fill: C.white, stroke: C.muted, "stroke-width": 2
    }));
    g.appendChild(txt(402, 140, "Kupffer cell", { size: 14, anchor: "middle" }));

    // circulating antibodies, shown only when the humoral arm is on
    refs.antibodies = el("g", { opacity: 0 });
    [[54, 92], [126, 128], [188, 74], [346, 118], [420, 82], [520, 126],
     [604, 76], [700, 120], [790, 84], [886, 122], [148, 66], [462, 138]
    ].forEach(function (s) {
      refs.antibodies.appendChild(el("path", {
        d: "M" + s[0] + " " + s[1] + " v-11 M" + s[0] + " " + (s[1] - 11) +
           " l-8 -9 M" + s[0] + " " + (s[1] - 11) + " l8 -9",
        stroke: C.antibody, "stroke-width": 2.6, fill: "none", "stroke-linecap": "round"
      }));
    });
    g.appendChild(refs.antibodies);

    // --- extracellular space, then the target cell -----------------------
    g.appendChild(txt(14, 496, "space of Disse", { size: 14 }));
    g.appendChild(el("rect", {
      x: 330, y: 200, width: 616, height: 320, rx: 28,
      fill: C.paper, stroke: C.ink, "stroke-width": 4
    }));
    g.appendChild(txt(934, 508, "hepatocyte", { size: 14, anchor: "end" }));

    // entry receptors on the membrane, at the point the route crosses it
    g.appendChild(el("path", {
      d: "M424 200 v-17 M442 200 v-23 M460 200 v-17",
      stroke: C.vector, "stroke-width": 3, "stroke-linecap": "round", opacity: 0.6
    }));

    // TLR2 on the membrane, lit with the innate arm
    refs.tlr2 = el("g", { opacity: 0 });
    refs.tlr2.appendChild(el("path", {
      d: "M556 200 v-20 M580 200 v-20",
      stroke: C.innate, "stroke-width": 6, "stroke-linecap": "round"
    }));
    refs.tlr2.appendChild(txt(568, 173, "TLR2",
      { size: 15, weight: 700, anchor: "middle", fill: C.innate }));
    g.appendChild(refs.tlr2);

    // endosome
    g.appendChild(el("circle", {
      cx: 580, cy: 330, r: 54, fill: C.white, stroke: C.ink, "stroke-width": 3.5
    }));
    refs.tlr9 = el("g", { opacity: 0 });
    refs.tlr9.appendChild(el("path", {
      d: "M545 291 l-14 -14 M615 291 l14 -14",
      stroke: C.innate, "stroke-width": 6, "stroke-linecap": "round"
    }));
    refs.tlr9.appendChild(txt(662, 282, "TLR9",
      { size: 15, weight: 700, anchor: "middle", fill: C.innate }));
    g.appendChild(refs.tlr9);

    // proteasome, fed by whatever loses the race to the nuclear pore
    g.appendChild(el("rect", {
      x: 548, y: 438, width: 104, height: 40, rx: 8,
      fill: C.white, stroke: C.muted, "stroke-width": 2
    }));
    g.appendChild(txt(600, 464, "proteasome", { size: 14, anchor: "middle" }));
    g.appendChild(el("path", {
      d: "M690 418 Q650 438 654 456", stroke: C.muted, "stroke-width": 2,
      "stroke-dasharray": "4 4", fill: "none"
    }));

    // MHC class I on the cell wall, and the T cell that reads it
    refs.tcellArt = el("g", { opacity: 0 });
    refs.tcellArt.appendChild(el("circle", {
      cx: 166, cy: 356, r: 54, fill: C.white, stroke: C.tcell, "stroke-width": 3
    }));
    refs.tcellArt.appendChild(txt(166, 352, "CD8⁺",
      { size: 15, weight: 700, anchor: "middle", fill: C.tcell }));
    refs.tcellArt.appendChild(txt(166, 372, "T cell",
      { size: 12, anchor: "middle", fill: C.tcell }));
    refs.tcellArt.appendChild(el("path", {
      d: "M330 336 h-16 M314 336 v-9 M314 336 v9",
      stroke: C.tcell, "stroke-width": 3, fill: "none", "stroke-linecap": "round"
    }));
    refs.tcellArt.appendChild(el("rect", {
      x: 300, y: 331, width: 12, height: 10, rx: 2, fill: C.vector
    }));
    refs.tcellArt.appendChild(txt(340, 316, "MHC I",
      { size: 14, weight: 700, fill: C.tcell }));
    refs.tcellArt.appendChild(el("path", {
      d: "M228 348 h64", stroke: C.tcell, "stroke-width": 3, "stroke-linecap": "round"
    }));
    refs.tcellArt.appendChild(el("path", {
      d: "M244 396 q52 30 96 22", stroke: C.tcell, "stroke-width": 2,
      "stroke-dasharray": "5 4", fill: "none"
    }));
    refs.tcellArt.appendChild(txt(258, 448, "perforin, granzyme",
      { size: 14, fill: C.tcell }));
    g.appendChild(refs.tcellArt);

    // nucleus
    g.appendChild(el("ellipse", {
      cx: 855, cy: 400, rx: 82, ry: 66,
      fill: C.nucleus, stroke: C.vector, "stroke-width": 3.5
    }));
    refs.episomes = el("g", {});
    g.appendChild(refs.episomes);

    // the route itself, drawn faintly under the particles
    var dpath = "M" + ROUTE.map(function (pt) { return pt[0] + " " + pt[1]; }).join(" L ");
    g.appendChild(el("path", {
      d: dpath, fill: "none", stroke: C.rule, "stroke-width": 2, "stroke-dasharray": "6 7"
    }));

    svg.appendChild(g);
    return g;
  }

  // ---------------------------------------------------------------- widget
  function initRoute(containerId) {
    var root = document.getElementById(containerId);
    if (!root) return;
    clear(root);

    var N = 220;
    var reduceMotion = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    var state = {
      arms: { innate: false, antibody: false, tcell: false, complement: false },
      doseIndex: DOSE_REF_INDEX,
      titreIndex: 3,
      selected: null
    };

    var refs = {};
    var model, fates;

    // --- controls --------------------------------------------------------
    var armRow = h("div", { class: "aav-arms" });
    var armButtons = {};
    ARMS.forEach(function (arm) {
      var b = h("button", {
        type: "button", class: "aav-arm", "data-arm": arm.key,
        "aria-pressed": "false"
      }, [
        h("span", { class: "aav-dot", style: { background: arm.color } }),
        h("span", { class: "aav-arm-label", text: arm.label }),
        h("span", { class: "aav-arm-note", text: arm.note })
      ]);
      b.addEventListener("click", function () {
        state.arms[arm.key] = !state.arms[arm.key];
        recompute();
      });
      armButtons[arm.key] = b;
      armRow.appendChild(b);
    });

    function slider(id, label, max, value, fmt) {
      var out = h("span", { class: "aav-slider-value" });
      var input = h("input", {
        type: "range", min: 0, max: String(max), step: "1",
        value: String(value), id: containerId + "-" + id,
        class: "aav-range", "aria-label": label
      });
      var wrap = h("div", { class: "aav-slider" }, [
        h("label", { class: "aav-slider-head", for: containerId + "-" + id }, [
          h("span", { text: label }), out
        ]),
        input
      ]);
      return { wrap: wrap, input: input, out: out, fmt: fmt };
    }

    var doseS = slider("dose", "Dose", DOSES.length - 1, state.doseIndex, function (i) {
      var m = DOSES[i] / Math.pow(10, Math.floor(Math.log10(DOSES[i])));
      return (m === 1 ? "10" : m + " × 10") +
        "^" + Math.floor(Math.log10(DOSES[i])) + " vg/kg";
    });
    doseS.input.addEventListener("input", function () {
      state.doseIndex = +doseS.input.value; recompute();
    });

    var titreS = slider("titre", "Pre-existing NAb titre", TITRES.length - 1,
      state.titreIndex, function (i) { return TITRES[i]; });
    titreS.input.addEventListener("input", function () {
      state.titreIndex = +titreS.input.value; recompute();
    });

    var sliderRow = h("div", { class: "aav-sliders" }, [doseS.wrap, titreS.wrap]);

    // --- readouts --------------------------------------------------------
    function stat(label, hint) {
      var v = h("div", { class: "aav-stat-value" });
      var box = h("div", { class: "aav-stat" }, [
        v,
        h("div", { class: "aav-stat-label", text: label }),
        h("div", { class: "aav-stat-hint", text: hint })
      ]);
      return { box: box, v: v };
    }
    var sReach = stat("reach a nucleus", "share of the injected dose");
    var sAbs = stat("genomes delivered", "vector genomes per kg");
    var sExpr = stat("expression, week 8", "index, 100 = no immune response at 10¹³");
    var statRow = h("div", { class: "aav-stats" }, [sReach.box, sAbs.box, sExpr.box]);

    var lampRow = h("div", { class: "aav-lamps" });
    var lampEls = {};
    ["ifn", "cyto", "alt", "comp"].forEach(function (k) {
      var led = h("span", { class: "aav-led" });
      var lab = h("span", { class: "aav-led-label" });
      var box = h("div", { class: "aav-lamp" }, [led, lab]);
      lampEls[k] = { led: led, lab: lab };
      lampRow.appendChild(box);
    });

    // --- svg -------------------------------------------------------------
    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H,
      width: "100%", role: "img",
      "aria-label": "Route of a recombinant AAV particle from the bloodstream to a hepatocyte nucleus, with switchable immune responses removing particles at each station.",
      style: "max-width:100%;height:auto;display:block;"
    });
    buildScene(svg, state, refs);

    var dotLayer = el("g", {});
    svg.appendChild(dotLayer);
    var dots = [];
    for (var i = 0; i < N; i++) {
      var c = el("circle", { r: 3.6, fill: C.vector, cx: -20, cy: -20 });
      dotLayer.appendChild(c);
      dots.push(c);
    }

    // station markers, clickable
    var markerLayer = el("g", {});
    svg.appendChild(markerLayer);
    var markers = [];
    STATIONS.forEach(function (st, idx) {
      var grp = el("g", { class: "aav-station", tabindex: "0", role: "button",
        "aria-label": "Station " + (idx + 1) + ": " + st.name });
      var ring = el("circle", {
        cx: st.x, cy: st.y, r: 21, fill: "none",
        stroke: C.ink, "stroke-width": 2, opacity: 0.28
      });
      var badge = el("circle", { cx: st.x, cy: st.y - 34, r: 13, fill: C.ink });
      var num = txt(st.x, st.y - 30, String(idx + 1),
        { size: 15, weight: 700, anchor: "middle", fill: C.white });
      var name = txt(st.x, st.y + (st.labelDy || 44), st.name,
        { size: 15, weight: 700, anchor: "middle", fill: C.ink });
      grp.appendChild(ring); grp.appendChild(badge); grp.appendChild(num); grp.appendChild(name);
      grp.style.cursor = "pointer";
      function pick() { state.selected = idx; renderPanel(); renderMarkers(); }
      grp.addEventListener("click", pick);
      grp.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
      });
      markerLayer.appendChild(grp);
      markers.push({ grp: grp, ring: ring, badge: badge, name: name, st: st });
    });

    // a seventh card for the T-cell arm, which is not a station
    var tcellBtn = el("g", { class: "aav-station", tabindex: "0", role: "button",
      "aria-label": "CD8 T cells, four to eight weeks later" });
    tcellBtn.appendChild(el("rect", {
      x: 108, y: 264, width: 116, height: 26, rx: 13,
      fill: C.white, stroke: C.tcell, "stroke-width": 2
    }));
    tcellBtn.appendChild(txt(166, 282, "weeks later",
      { size: 14, weight: 700, anchor: "middle", fill: C.tcell }));
    tcellBtn.style.cursor = "pointer";
    function pickT() { state.selected = 6; renderPanel(); renderMarkers(); }
    tcellBtn.addEventListener("click", pickT);
    tcellBtn.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickT(); }
    });
    markerLayer.appendChild(tcellBtn);

    var panel = h("div", { class: "aav-panel" });
    var legend = h("div", { class: "aav-legend" });

    root.appendChild(armRow);
    root.appendChild(sliderRow);
    root.appendChild(statRow);
    root.appendChild(lampRow);
    root.appendChild(svg);
    root.appendChild(legend);
    root.appendChild(panel);

    // --- rendering -------------------------------------------------------
    function fmtPow(x) {
      var e = Math.floor(Math.log10(x));
      var m = x / Math.pow(10, e);
      return m.toFixed(1) + " × 10<sup>" + e + "</sup>";
    }

    function renderArms() {
      ARMS.forEach(function (arm) {
        var on = state.arms[arm.key];
        var b = armButtons[arm.key];
        b.setAttribute("aria-pressed", on ? "true" : "false");
        b.classList.toggle("is-on", on);
        b.style.borderColor = on ? arm.color : C.rule;
        b.style.background = on ? arm.color + "14" : C.white;
      });
      titreS.wrap.classList.toggle("is-muted", !state.arms.antibody);
    }

    function renderStats() {
      sReach.v.innerHTML = (model.reaching * 100).toFixed(2) + "%";
      sAbs.v.innerHTML = fmtPow(model.absolute);
      sExpr.v.innerHTML = model.expression < 1
        ? model.expression.toFixed(2)
        : Math.round(model.expression);

      model.lamps.forEach(function (l) {
        var e = lampEls[l.key];
        var col = l.level <= 0.001 ? C.rule
          : (l.level < 0.35 ? "#8DBF6A" : (l.level < 0.68 ? C.innate : "#C6362F"));
        e.led.style.background = col;
        e.lab.textContent = l.label;
        e.lab.style.color = l.level <= 0.001 ? C.muted : C.ink;
      });
    }

    function renderScene() {
      refs.antibodies.setAttribute("opacity", state.arms.antibody ? 1 : 0);
      refs.tlr2.setAttribute("opacity", state.arms.innate ? 1 : 0);
      refs.tlr9.setAttribute("opacity", state.arms.innate ? 1 : 0);
      refs.tcellArt.setAttribute("opacity", state.arms.tcell ? 1 : 0);
      tcellBtn.setAttribute("opacity", state.arms.tcell ? 1 : 0.35);
      refs.macPores.setAttribute("opacity",
        state.arms.complement ? clamp(0.25 + model.sev.complement, 0, 1) : 0);

      // episomes accumulate in the nucleus in proportion to what arrived
      clear(refs.episomes);
      var arrived = fates.filter(function (f) { return f.station === 6; }).length;
      var rings = Math.min(9, Math.round(arrived / 4));
      var rr = mulberry32(99);
      for (var k = 0; k < rings; k++) {
        refs.episomes.appendChild(el("circle", {
          cx: 855 + (rr() - 0.5) * 112, cy: 400 + (rr() - 0.5) * 90,
          r: 6 + rr() * 3, fill: "none", stroke: C.vector,
          "stroke-width": 2.2, opacity: 0.75
        }));
      }
    }

    function renderMarkers() {
      markers.forEach(function (m, idx) {
        var active = state.selected === idx;
        var acting = m.st.arms.filter(function (a) { return state.arms[a]; });
        var col = acting.length ? C[acting[0]] : C.ink;
        m.badge.setAttribute("fill", active ? C.vector : col);
        m.ring.setAttribute("stroke", col);
        m.ring.setAttribute("opacity", acting.length ? 0.85 : 0.28);
        m.ring.setAttribute("stroke-width", acting.length ? 3.5 : 2);
        m.name.setAttribute("fill", active ? C.vector : C.ink);
      });
      tcellBtn.setAttribute("opacity", state.arms.tcell ? 1 : 0.35);
    }

    function renderLegend() {
      clear(legend);
      var items = [{ key: "arrived", label: "reached a nucleus" },
                   { key: "traffic", label: "lost to trafficking" }];
      ARMS.forEach(function (a) {
        if (state.arms[a.key] && a.key !== "tcell") {
          items.push({ key: a.key, label: "removed by " + a.label.toLowerCase() });
        }
      });
      items.forEach(function (it) {
        legend.appendChild(h("span", { class: "aav-legend-item" }, [
          h("span", { class: "aav-dot", style: { background: causeColor(it.key) } }),
          h("span", { text: it.label })
        ]));
      });
    }

    function renderPanel() {
      clear(panel);
      if (state.selected === null) {
        var onArms = ARMS.filter(function (a) { return state.arms[a.key]; });
        panel.appendChild(h("div", { class: "aav-panel-title",
          text: onArms.length ? "Arms acting: " + onArms.map(function (a) { return a.label; }).join(", ")
                              : "No immune response active" }));
        panel.appendChild(h("p", { class: "aav-panel-body", text:
          onArms.length
            ? "Every particle that disappears now carries the colour of the arm that removed it. Click a numbered station to read what happens there."
            : "This is trafficking loss alone: the attrition a particle meets in a patient with no response to it at all. Click a numbered station to read what happens there." }));
        return;
      }
      var card = state.selected === 6 ? TCELL_CARD : STATIONS[state.selected];
      var head = state.selected === 6
        ? card.name
        : (state.selected + 1) + ". " + card.name;
      panel.appendChild(h("div", { class: "aav-panel-title", text: head }));
      panel.appendChild(h("div", { class: "aav-panel-sub", text: card.sub }));
      panel.appendChild(h("p", { class: "aav-panel-body", text: card.body }));
      panel.appendChild(h("p", { class: "aav-panel-body aav-panel-arm", text: card.detail }));
      if (state.selected !== 6) {
        var kept = model.perStation[state.selected];
        panel.appendChild(h("div", { class: "aav-panel-num",
          html: "Particles surviving this station: <strong>" +
            (kept * 100).toFixed(0) + "%</strong>" }));
      }
      panel.appendChild(h("button", { type: "button", class: "aav-clear", text: "Back to summary" }))
        .addEventListener("click", function () { state.selected = null; renderPanel(); renderMarkers(); });
    }

    // --- animation -------------------------------------------------------
    var t0 = null, running = false;

    function placeDots(progress) {
      for (var i = 0; i < N; i++) {
        var f = fates[i];
        var lane = (f.jitter - 0.5);
        // Each particle enters on its own stagger so the flow reads as a stream.
        var u = progress * 6.6 - (i / N) * 1.6;
        var limit = f.station === 6 ? 5 : f.station;
        var uu = Math.min(u, limit);
        var c = dots[i];
        if (u < -0.2) { c.setAttribute("opacity", 0); continue; }
        var pt = pointAt(uu);
        var x, y;
        if (f.station < 6 && u > limit) {
          // Removed here. Settle into a ring around the station and keep the
          // colour of the arm that did it, so the picture reads as attrition
          // rather than as a flow chart.
          var over = clamp(u - limit, 0, 1);
          var ang = i * 2.3999632 + f.jitter * 0.8;
          var rad = (15 + Math.sqrt(f.jitter) * 46) * over;
          x = pt[0] + Math.cos(ang) * rad;
          y = pt[1] + Math.sin(ang) * rad * 0.72;
          c.setAttribute("fill", causeColor(f.cause));
          c.setAttribute("opacity", String(0.30 + 0.32 * over));
          c.setAttribute("r", "3.2");
        } else if (f.station === 6 && uu >= 5) {
          var aa = i * 2.3999632;
          var rr2 = 14 + Math.sqrt(f.jitter) * 62;
          x = pt[0] + Math.cos(aa) * rr2;
          y = pt[1] + Math.sin(aa) * rr2 * 0.74;
          c.setAttribute("fill", C.vector);
          c.setAttribute("opacity", "0.95");
          c.setAttribute("r", "4");
        } else {
          x = pt[0];
          y = pt[1] + 20 * lane;
          c.setAttribute("fill", C.vector);
          c.setAttribute("opacity", "0.9");
          c.setAttribute("r", "3.6");
        }
        c.setAttribute("cx", x.toFixed(1));
        c.setAttribute("cy", y.toFixed(1));
      }
    }
    function frame(ts) {
      if (t0 === null) t0 = ts;
      var progress = Math.min(1, (ts - t0) / 3600);
      placeDots(progress);
      if (progress < 1) requestAnimationFrame(frame);
      else running = false;
    }

    function restart() {
      if (reduceMotion) { placeDots(1); return; }
      t0 = null;
      if (!running) { running = true; requestAnimationFrame(frame); }
    }

    function recompute() {
      model = AAVModel(state);
      fates = assignFates(model, N, 20260904);
      doseS.out.innerHTML = doseS.fmt(state.doseIndex)
        .replace(/\^(-?\d+)/, "<sup>$1</sup>");
      titreS.out.textContent = titreS.fmt(state.titreIndex);
      renderArms(); renderStats(); renderScene(); renderMarkers();
      renderLegend(); renderPanel(); restart();
    }

    recompute();
  }

  return { init: initRoute, AAVModel: AAVModel, DOSES: DOSES, TITRES: TITRES, BASE: BASE };
})();

if (typeof module !== "undefined" && module.exports) module.exports = AAV;

document.addEventListener("DOMContentLoaded", function () {
  if (document.getElementById("aav-route")) AAV.init("aav-route");
});
