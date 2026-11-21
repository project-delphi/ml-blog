/* The numbers behind the widgets in "When UMAP Lies, Part 4".
 *
 * Kept apart from the drawing (widgets.js) so the prose can be checked against
 * it: `node posts/when-umap-lies/src/check_widgets.js ranking` prints every
 * winner and tally. Exposes one global, `UMLRankModel`, and `module.exports`
 * under node. Reads WK_DATA["uml-ranking"] (widget-data/ranking.js, written by
 * posts/when-umap-lies/src/export_widget_data.py).
 *
 * Each case holds, per metric and method, the median, min and max over draws
 * (data seeds; method seeds for PBMC 3k) of one score, default settings only.
 * "Best" is the highest median; every method within `tie` (0.01) of the best
 * shares the win, and a shared win counts as a win for each method in it.
 */
"use strict";

var UMLRankModel = (function () {
  function caseOf(R, key) {
    for (var i = 0; i < R.cases.length; i++) if (R.cases[i].key === key) return R.cases[i];
    throw new Error("no case " + key);
  }

  /** {v, lo, hi, n} for one (case, metric, method), or null when not scored. */
  function score(R, caseKey, metric, method) {
    var s = caseOf(R, caseKey).scores[metric];
    return s && s[method] ? s[method] : null;
  }

  /** The methods scored for (case, metric), in the series' method order. */
  function scored(R, caseKey, metric) {
    var c = caseOf(R, caseKey), s = c.scores[metric];
    if (!s) return [];
    return R.methods.filter(function (m) { return s[m] && s[m].v !== null; });
  }

  /** {top, winners: [methods within R.tie of top]} for (case, metric), or null. */
  function best(R, caseKey, metric) {
    var ms = scored(R, caseKey, metric);
    if (!ms.length) return null;
    var top = Math.max.apply(null, ms.map(function (m) { return score(R, caseKey, metric, m).v; }));
    return { top: top, winners: ms.filter(function (m) { return score(R, caseKey, metric, m).v >= top - R.tie - 1e-12; }) };
  }

  /** Wins per method on one metric over the cases that have it:
   *  {wins: {method: n}, alone: {method: n wins not shared}, cases: n,
   *   shared: n (cases with more than one winner)}. */
  function tally(R, metric) {
    var wins = {}, alone = {}, cases = 0, shared = 0;
    R.methods.forEach(function (m) { wins[m] = 0; alone[m] = 0; });
    R.cases.forEach(function (c) {
      var b = best(R, c.key, metric);
      if (!b) return;
      cases++;
      if (b.winners.length > 1) shared++;
      else alone[b.winners[0]]++;
      b.winners.forEach(function (m) { wins[m]++; });
    });
    return { wins: wins, alone: alone, cases: cases, shared: shared };
  }

  /** The same over every (case, metric) pair that has a score: {wins, alone, pairs, shared}. */
  function tallyAll(R) {
    var wins = {}, alone = {}, pairs = 0, shared = 0;
    R.methods.forEach(function (m) { wins[m] = 0; alone[m] = 0; });
    R.metrics.forEach(function (mt) {
      var t = tally(R, mt.key);
      pairs += t.cases; shared += t.shared;
      R.methods.forEach(function (m) { wins[m] += t.wins[m]; alone[m] += t.alone[m]; });
    });
    return { wins: wins, alone: alone, pairs: pairs, shared: shared };
  }

  /** Methods sorted by wins, most first (the series' method order breaks ties). */
  function ranked(R, wins) {
    return R.methods.slice().sort(function (a, b) {
      return wins[b] - wins[a] || R.methods.indexOf(a) - R.methods.indexOf(b);
    });
  }

  /** The axis for a metric: [lo, hi] covering every score (hi = 1, or 0 for gaps). */
  function domain(R, metric) {
    var lo = 0, hi = metric === "gaps" ? 0 : 1;
    R.cases.forEach(function (c) {
      var s = c.scores[metric];
      if (!s) return;
      Object.keys(s).forEach(function (m) { if (s[m].v < lo) lo = s[m].v; });
    });
    return [Math.floor(lo * 4 + 1e-9) / 4, hi];
  }

  /** Which cases lack each metric: {metric: [case titles]}. */
  function missing(R) {
    var out = {};
    R.metrics.forEach(function (mt) {
      out[mt.key] = R.cases.filter(function (c) { return !c.scores[mt.key]; }).map(function (c) { return c.title; });
    });
    return out;
  }

  return { caseOf: caseOf, score: score, scored: scored, best: best, tally: tally, tallyAll: tallyAll,
    ranked: ranked, domain: domain, missing: missing };
})();

if (typeof module !== "undefined" && module.exports) module.exports = UMLRankModel;
