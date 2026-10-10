/* The pass table behind the 3D "paper in layers" scenes in the slide deck.
 *
 * Kept apart from the drawing (widgets.js) so the deck can be checked against
 * the post: every time, lead and layer a scene shows is a row here, and
 * `node posts/how-to-read-a-paper-in-the-age-of-ai/src/check_model.js` prints
 * the table for a side-by-side read with the post. Plain JavaScript with no
 * dependencies; exposes one global, `PaperLayersModel`, and `module.exports`
 * under node.
 *
 * Two modes. `classic` is Keshav's three passes as the post's first table
 * gives them, with his own Pass 3 time from the paper ("about four or five hours
 * for beginners, and about an hour for an experienced reader"). `ai` is the
 * post's six-step protocol. A time neither gives is "—", never an estimate.
 *
 * Layers count down from the top of the stack: 1 is what a skim reads, 4 is
 * what only a reconstruction reaches. `candidates` says what the scene does to
 * the other papers in the field: "top" lights the top sheet of every one
 * (triage reads wide and shallow), "cited" draws lines to two of them (the
 * citations a skeptical read checks).
 */
"use strict";

var PaperLayersModel = (function () {
  var LAYERS = [
    { id: 1, short: "abstract & headings", long: "title, abstract, intro, headings, conclusions" },
    { id: 2, short: "figures", long: "every figure" },
    { id: 3, short: "argument", long: "the main text: each claim and its evidence" },
    { id: 4, short: "proofs, code, data", long: "proofs, derivations, code and data" }
  ];

  var WHO = { you: "you", ai: "AI" };

  var MODES = {
    // Post, first table: "Pass 1 (5–10 min): title, abstract, intro, headings,
    // conclusions ... Pass 2 (~1 h): figures and argument, skip proofs.
    // Pass 3: virtually re-implement the work."
    classic: [
      { pass: "1", name: "Pass 1 · the five Cs", lead: "you", second: null,
        time: "5–10 min", layers: [1], candidates: null },
      { pass: "2", name: "Pass 2 · figures and argument", lead: "you", second: null,
        time: "~1 h", layers: [2, 3], candidates: null },
      { pass: "3", name: "Pass 3 · virtually re-implement", lead: "you", second: null,
        time: "4–5 h (~1 h expert)", layers: [4], candidates: null }
    ],
    // Post, "A protocol: the three passes, updated for AI", steps 1–6.
    ai: [
      { pass: "0", name: "Pass 0 · triage", lead: "ai", second: null,
        time: "~2 min per paper", layers: [1], candidates: "top" },
      { pass: "1", name: "Pass 1 · the bird's-eye view", lead: "you", second: null,
        time: "5–10 min", layers: [1, 2], candidates: null },
      { pass: "2", name: "Pass 2 · the argument", lead: "you", second: "ai",
        role: "you, with AI as tutor", time: "~1 h", layers: [2, 3], candidates: null },
      { pass: "3", name: "Pass 3 · the field-specific deep read", lead: "you", second: null,
        time: "hours", layers: [4], candidates: null },
      { pass: "4", name: "Pass 4 · adversarial check", lead: "ai", second: "you",
        role: "AI as skeptic, then you", time: "—", layers: [3], candidates: "cited" },
      { pass: "integrate", name: "Integrate", lead: "you", second: null,
        time: "—", layers: ["note"], candidates: null }
    ]
  };

  function passes(mode) { return MODES[mode] || []; }
  function find(mode, pass) {
    return passes(mode).filter(function (p) { return p.pass === pass; })[0] || null;
  }
  /** Who the readout names: the role line when there is one, else the lead. */
  function who(p) { return p.role || WHO[p.lead]; }

  // ---------------------------------------------------------- survey time
  // The "scale and speed" slide is arithmetic on the post's own per-paper times:
  // a first pass by hand takes 5–10 min (Keshav), AI triage ~2 min (protocol,
  // Pass 0), and the triage workflow picks five papers to read properly, each a
  // Pass 2 of ~1 h (Pass 3, the deep read, takes hours more and is left out).
  // Hours, as [low, high] where the post gives a range.
  var SURVEY = { handMin: [5, 10], aiMin: 2, deep: 5, deepH: 1, fields: [30, 100] };
  function budget(n) {
    return { n: n, hand: [n * SURVEY.handMin[0] / 60, n * SURVEY.handMin[1] / 60], ai: n * SURVEY.aiMin / 60,
      deep: SURVEY.deep * SURVEY.deepH };
  }
  /** Hours to one decimal: the one rounding rule the chart, its tiles and the
   *  check script all print with. */
  function hours(x) { return String(Math.round(x * 10) / 10); }

  var api = { LAYERS: LAYERS, WHO: WHO, MODES: Object.keys(MODES), passes: passes, find: find, who: who,
    SURVEY: SURVEY, budget: budget, hours: hours };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
