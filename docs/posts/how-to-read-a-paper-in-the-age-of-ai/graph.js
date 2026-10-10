/* The citation-graph slide's model: which papers it draws, the two kinds of
 * link between them, and where each one sits.
 *
 * Data is graph-data.js, a Semantic Scholar snapshot around Keshav's "How to
 * read a paper" (2007). The slide draws Keshav plus every citing paper whose
 * reference list the API returned -- the only ones a link between two citing
 * papers can be computed for -- so there is no threshold to choose. A paper with
 * no year is left out rather than placed on a guessed one, since height is year.
 *
 * Two kinds of link:
 * - a citation link joins a paper to a paper it cites;
 * - a "connected" link joins two papers that share references (bibliographic
 *   coupling, half of what Connected Papers' similarity is built from). Every
 *   drawn paper cites Keshav, so Keshav is left out of the shared count. Each
 *   paper adds its KEEP strongest such links to one shared set, so a paper can
 *   end up with more than KEEP (its neighbours' picks count too). A link needs
 *   at least MIN_SHARED shared references.
 *
 * Height is publication year, so a citation should point down. The ones that
 * point up (`upLinks`) are wrong data, and they are the slide's point. The other two
 * coordinates come from a small seeded force layout over both kinds of link,
 * computed once here so the slide and `src/check_model.js` see the same
 * picture. Plain JavaScript; one global, `PaperGraph`, and module.exports
 * under node.
 */
"use strict";

var PaperGraph = (function () {
  var D = typeof PaperGraphData !== "undefined" ? PaperGraphData : require("./graph-data.js");
  var MIN_SHARED = 2, KEEP = 2, TOP = 5;
  var Y0 = 2007, PER_YEAR = 0.26, REACH = 2.7;

  var pool = D.papers;
  // One field per paper for colour: Semantic Scholar's own tags, computer science
  // first, since that is Keshav's field and the question is who else got counted.
  function fieldOf(p) {
    if (p.k === "seed") return "seed";
    if (p.f.indexOf("Computer Science") >= 0) return "cs";
    if (p.f.indexOf("Medicine") >= 0) return "med";
    return "other";
  }
  var seed = 0;
  var citingAll = pool.filter(function (p) { return p.k === "citing"; });
  var drawn = [seed].concat(pool.map(function (p, i) { return i; })
    .filter(function (i) { return pool[i].k === "citing" && pool[i].r.length && pool[i].y; }));
  var at = {};  // pool index -> drawn index
  drawn.forEach(function (pi, i) { at[pi] = i; });
  var nodes = drawn.map(function (pi) {
    var p = pool[pi];
    return { id: p.id, title: p.t, year: p.y, cites: p.c, seed: pi === seed, field: fieldOf(p),
      refs: p.r.filter(function (r) { return r !== seed; }) };
  });

  // -------------------------------------------------------- citation links
  var citation = [];
  drawn.forEach(function (pi, a) {
    pool[pi].r.forEach(function (r) { if (at[r] !== undefined && at[r] !== a) citation.push([a, at[r]]); });
  });
  function cites(a, b) {
    return citation.some(function (e) { return (e[0] === a && e[1] === b) || (e[0] === b && e[1] === a); });
  }

  // ------------------------------------------------------- connected links
  var sets = nodes.map(function (n) { var s = {}; n.refs.forEach(function (r) { s[r] = 1; }); return s; });
  function shared(a, b) {
    var k = 0;
    nodes[a].refs.forEach(function (r) { if (sets[b][r]) k++; });
    return k;
  }
  var pairs = [];
  for (var a = 1; a < nodes.length; a++) {
    for (var b = a + 1; b < nodes.length; b++) {
      var k = shared(a, b);
      if (k >= MIN_SHARED) {
        pairs.push({ a: a, b: b, shared: k, w: k / Math.sqrt(nodes[a].refs.length * nodes[b].refs.length) });
      }
    }
  }
  var keep = {};
  nodes.forEach(function (n, i) {
    pairs.filter(function (p) { return p.a === i || p.b === i; })
      .sort(function (x, y) { return y.w - x.w || y.shared - x.shared; })
      .slice(0, KEEP).forEach(function (p) { keep[p.a + "-" + p.b] = p; });
  });
  var connected = Object.keys(keep).map(function (k) { return keep[k]; })
    .sort(function (x, y) { return x.a - y.a || x.b - y.b; });

  // ------------------------------------------------------------ rankings
  // Both rankings are of the citing papers; Keshav is what they cite, not a rival.
  var byCites = nodes.map(function (n, i) { return i; }).filter(function (i) { return !nodes[i].seed; })
    .sort(function (x, y) { return nodes[y].cites - nodes[x].cites || x - y; });
  var strength = nodes.map(function () { return 0; });
  connected.forEach(function (p) { strength[p.a] += p.w; strength[p.b] += p.w; });
  var byConnection = nodes.map(function (n, i) { return i; }).filter(function (i) { return !nodes[i].seed; })
    .sort(function (x, y) { return strength[y] - strength[x] || x - y; });
  var topCited = byCites.slice(0, TOP);
  var topConnected = byConnection.slice(0, TOP);

  // -------------------------------------------------------------- layout
  function mulberry32(s) {
    return function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rng = mulberry32(2007);
  var xy = nodes.map(function (n) {
    if (n.seed) return [0, 0];
    var a = rng() * 2 * Math.PI, r = 1.5 + rng() * 1.5;
    return [r * Math.cos(a), r * Math.sin(a)];
  });
  var springs = connected.map(function (p) { return [p.a, p.b, 0.6 * p.w + 0.05, 1.1]; })
    .concat(citation.filter(function (e) { return e[1] !== 0; }).map(function (e) { return [e[0], e[1], 0.08, 1.4]; }));
  for (var it = 0; it < 600; it++) {
    var f = xy.map(function () { return [0, 0]; }), i, j;
    for (i = 0; i < xy.length; i++) {
      for (j = i + 1; j < xy.length; j++) {
        var dx = xy[i][0] - xy[j][0], dy = xy[i][1] - xy[j][1], d2 = dx * dx + dy * dy + 0.01;
        var rep = 0.35 / d2;
        f[i][0] += dx * rep; f[i][1] += dy * rep; f[j][0] -= dx * rep; f[j][1] -= dy * rep;
      }
      // A weak pull to the middle keeps unlinked papers on the stage.
      f[i][0] -= 0.02 * xy[i][0]; f[i][1] -= 0.02 * xy[i][1];
    }
    springs.forEach(function (s) {
      var dx = xy[s[1]][0] - xy[s[0]][0], dy = xy[s[1]][1] - xy[s[0]][1], d = Math.sqrt(dx * dx + dy * dy) + 1e-6;
      var pull = s[2] * (d - s[3]) / d;
      f[s[0]][0] += dx * pull; f[s[0]][1] += dy * pull; f[s[1]][0] -= dx * pull; f[s[1]][1] -= dy * pull;
    });
    var step = 0.5 * (1 - it / 600) + 0.02;
    for (i = 1; i < xy.length; i++) {  // Keshav stays at the centre
      xy[i][0] += Math.max(-0.3, Math.min(0.3, f[i][0] * step));
      xy[i][1] += Math.max(-0.3, Math.min(0.3, f[i][1] * step));
    }
  }
  // Scale the spread to fit inside the 2007 floor the slide draws (radius REACH).
  var far = Math.max.apply(null, xy.map(function (p) { return Math.sqrt(p[0] * p[0] + p[1] * p[1]); })) || 1;
  var pos = nodes.map(function (n, i) {
    return [xy[i][0] * REACH / far, xy[i][1] * REACH / far, (n.year - Y0) * PER_YEAR];
  });

  var years = nodes.map(function (n) { return n.year; });
  var api = {
    fetched: D.fetched, source: D.source, MIN_SHARED: MIN_SHARED, KEEP: KEEP, TOP: TOP, Y0: Y0, PER_YEAR: PER_YEAR,
    citingTotal: citingAll.length,
    citingWithRefs: citingAll.filter(function (p) { return p.r.length; }).length,
    seedHasRefs: pool[seed].r.length > 0,
    predates: citingAll.filter(function (p) { return p.y && p.y < pool[seed].y; }).length,
    tagged: function (f) { return citingAll.filter(function (p) { return p.f.indexOf(f) >= 0; }).length; },
    drawnBy: function (f) { return nodes.filter(function (n) { return n.field === f; }).length; },
    upLinks: citation.filter(function (e) { return nodes[e[0]].year < nodes[e[1]].year; }),
    topCitedMed: topCited.filter(function (i) { return nodes[i].field === "med"; }).length,
    nodes: nodes, pos: pos, citation: citation, connected: connected,
    connectedNotCiting: connected.filter(function (p) { return !cites(p.a, p.b); }).length,
    topCited: topCited, topConnected: topConnected,
    overlap: topCited.filter(function (i) { return topConnected.indexOf(i) >= 0; }).length,
    yearRange: [Math.min.apply(null, years), Math.max.apply(null, years)]
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
