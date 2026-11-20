/* Sweeps the widget controls and prints every number the post can quote from a
 * widget, so the prose is checked against the model rather than one screenshot.
 * Run from the repo root (after export_widget_data.py):
 *
 *   node posts/when-umap-lies/src/check_widgets.js            # every widget
 *   node posts/when-umap-lies/src/check_widgets.js elephant   # one
 *
 * One function per widget; add the next widget's check beside the elephant's.
 */
"use strict";
const path = require("path");

global.window = global;  // the data files assign window.WK_DATA
const POST = path.join(__dirname, "..");
const UK = require(path.join(POST, "umapkit.js"));
const M = require(path.join(POST, "model.js"));
const n = (x, d = 3) => UK.num(x, d).replace("−", "-");
const pad = (s, w) => String(s).padStart(w);

function elephant() {
  require(path.join(POST, "widget-data", "elephant.js"));
  const D = window.WK_DATA["uml-elephant"];
  for (const [key, ds] of Object.entries(D.datasets)) {
    console.log(`\n== ${key}: ${ds.n} points, ${ds.classes.length} ${ds.noun}s; axis ${M.domain(D, key).join(" to ")}`);
    console.log("   counts: " + ds.classes.map((c, i) => `${c} ${ds.counts[i]}`).join(", "));
    if (ds.largeTypes) console.log(`   types >= ${ds.largeMin} cells: ${ds.largeTypes.join(", ")}`);
    const head = "   " + "metric".padEnd(18) + D.f.map((f) => pad("f" + f.toFixed(1), 7)).join("") +
      pad("random", 8) + pad("UMAP mean [min, max]", 24) + pad("t-SNE mean [min, max]", 24);
    console.log(head);
    for (const m of ds.metrics) {
      const row = D.f.map((f) => pad(n(M.elephantScores(D, key, m, f).elephant.v), 7)).join("");
      const s = M.elephantScores(D, key, m, 0);
      const rng = (x) => `${n(x.v)} [${n(x.lo)}, ${n(x.hi)}]`;
      const hidden = M.shownMetrics(D, key).includes(m) ? "" : "  (not shown in the widget)";
      console.log("   " + m.padEnd(18) + row + pad(n(s.random.v), 8) + pad(rng(s.umap), 24) + pad(rng(s.tsne), 24) + hidden);
    }
    // Where the elephant sits against UMAP and t-SNE, metric by metric, at each f.
    console.log("   f values where the elephant scores at or above the lower of UMAP's and t-SNE's seed minimum:");
    for (const m of ds.metrics) {
      const ok = D.f.filter((f) => {
        const s = M.elephantScores(D, key, m, f);
        return s.elephant.v >= Math.min(s.umap.lo, s.tsne.lo);
      });
      console.log(`     ${m.padEnd(18)} ${ok.length ? ok.map((f) => f.toFixed(1)).join(" ") : "none"}`);
    }
    const lay = M.elephantLayout(D, key, "elephant", 0.5), e = UK.extent(lay);
    console.log(`   layout at f 0.5: ${lay.length / 2} points, extent x ${n(e.x0)}..${n(e.x1)}, y ${n(e.y0)}..${n(e.y1)}`);
  }
}

// Part 2: what widget-dimension shows at every D, for both bases. Each cell is
// "median [min, max]" over the five draws; "s0" is the seed-0 run the panels draw.
function dimension() {
  const PART2 = path.join(POST, "..", "when-umap-lies-dimension");
  require(path.join(PART2, "widget-data", "dimension.js"));
  const M2 = require(path.join(PART2, "model.js"));
  const Data = window.WK_DATA["uml-dimension"];
  const cell = (s) => (s ? `${n(s.v, 2)} [${n(s.lo, 2)}, ${n(s.hi, 2)}]` : "-");
  for (const [key, ds] of Object.entries(Data.datasets)) {
    console.log(`\n== ${ds.title} (p = ${ds.p}, sigma ${n(ds.sigma, 4)}, r10^2 ${n(ds.r10sq, 4)}), ` +
      `${ds.shown} of ${ds.n} points shown`);
    const cols = ["input", "pca50", ...M2.SERIES.map((s) => s.key)];
    console.log("   " + "D".padStart(5) + "  " + "eta".padStart(6) + cols.map((c) => pad(c, 19)).join("") +
      pad("UMAP s", 8) + pad("UMAP s0", 9) + pad("t-SNE s0", 9));
    for (const D of ds.D) {
      const row = cols.map((c) => pad(cell(c === "input" || c === "pca50" ? M2.graph(Data, key, c, D) :
        M2.q10(Data, key, c, D)), 19)).join("");
      console.log("   " + pad(D, 5) + "  " + pad(n(M2.eta(Data, key, D), 2), 6) + row +
        pad(n(M2.seconds(Data, key, D).v, 1), 8) + pad(n(M2.q10(Data, key, "umap", D).seed0, 2), 9) +
        pad(n(M2.q10(Data, key, "tsne", D).seed0, 2), 9));
    }
    // Which maps keep more true neighbours than their own input graph (medians)?
    const above = (c) => ds.D.filter((D) => M2.q10(Data, key, c, D).v > M2.graph(Data, key, "input", D).v);
    for (const s of M2.SERIES) console.log(`   ${s.label.padEnd(16)} above the input graph at D: ${above(s.key).join(" ") || "none"}`);
    const maxDis = Math.max(...ds.D.flatMap((D) => Object.values(ds.disparity[String(D)])));
    console.log(`   largest Procrustes disparity of a shown panel: ${n(maxDis, 3)}`);
  }
}

// Part 2: what widget-signal shows at every setting (one draw each). Columns:
// k = 10 preservation (q), trustworthiness (tw), 15-NN class accuracy (acc).
function signal(argv) {
  const PART2 = path.join(POST, "..", "when-umap-lies-dimension");
  require(path.join(PART2, "widget-data", "signal.js"));
  const M2 = require(path.join(PART2, "model.js"));
  const S = window.WK_DATA["uml-signal"];
  const G = S.grid;
  const line = (d, D, s, nn) => {
    const c = M2.signal(S, d, D, s, nn);
    const m = (k) => `${k.padEnd(4)} q ${n(c.q10[k], 2)} tw ${n(c.tw10[k], 2)} acc ${n(c.acc[k], 2)}`;
    return `   d ${pad(d, 2)} D ${pad(D, 4)} sigma ${pad(s, 4)} n ${pad(nn, 4)}  eta ${pad(n(c.eta, 2), 5)}  ` +
      `input q ${n(c.q10.input, 2)}  |  ${["pca", "umap", "tsne"].map(m).join("  |  ")}`;
  };
  console.log(`\n== signal: grid d ${G.d}, D ${G.D}, sigma ${G.sigma}, n ${G.n}; one draw each`);
  for (const nn of G.n) {
    console.log(`   r10^2 at n = ${nn}: ` + G.d.map((d) => `d ${d}: ${n(M2.signalBase(S, d, nn).r10sq, 2)}`).join(", "));
  }
  // The settings the prose asks about, then (with "all") every cell.
  const asked = [[2, 1000, 0, 2000], [2, 1000, 0.6, 2000], [20, 20, 0, 2000], [2, 20, 0, 2000],
    [5, 1000, 0.3, 500], [5, 1000, 0.3, 2000]];
  console.log("   -- asked about:");
  asked.forEach((a) => console.log(line(...a)));
  if (argv.includes("all")) {
    console.log("   -- every setting:");
    for (const d of G.d) for (const D of G.D) for (const s of G.sigma) for (const nn of G.n) console.log(line(d, D, s, nn));
  }
}

// Part 4: every case's scores and winners per metric, and the two tallies.
function ranking() {
  const PART4 = path.join(POST, "..", "when-umap-lies-gaps-density-holes");
  require(path.join(PART4, "widget-data", "ranking.js"));
  const M4 = require(path.join(PART4, "model.js"));
  const R = window.WK_DATA["uml-ranking"];
  const f = (x) => (x === null ? "-" : n(x, 3));
  console.log(`\n== ranking: ${R.cases.length} cases, ties within ${R.tie} share the win`);
  for (const mt of R.metrics) {
    console.log(`\n   ${mt.label} (axis ${M4.domain(R, mt.key).join(" to ")})`);
    for (const c of R.cases) {
      const b = M4.best(R, c.key, mt.key);
      if (!b) { console.log(`     ${c.title.padEnd(36)} no score`); continue; }
      const all = M4.scored(R, c.key, mt.key).map((m) => `${m} ${f(M4.score(R, c.key, mt.key, m).v)}`).join(", ");
      console.log(`     ${c.title.padEnd(36)} best ${b.winners.join("+")} ${f(b.top)} [${c.sources[mt.key]}]  ${all}`);
    }
    const t = M4.tally(R, mt.key);
    console.log(`     wins over ${t.cases} cases (${t.shared} shared): ` +
      M4.ranked(R, t.wins).filter((m) => t.wins[m]).map((m) => `${m} ${t.wins[m]}`).join(", "));
  }
  const a = M4.tallyAll(R);
  console.log(`\n   all ${a.pairs} (case, metric) pairs, ${a.shared} with a shared win: ` +
    M4.ranked(R, a.wins).map((m) => `${m} ${a.wins[m]}`).join(", "));
  const top = M4.ranked(R, a.wins)[0];
  console.log(`   most wins: ${top} with ${a.wins[top]} of ${a.pairs} (${a.wins[top] > a.pairs / 2 ? "more" : "not more"} than half)`);
  console.log("   not run: " + R.cases.filter((c) => c.notRun.length).map((c) => `${c.title}: ${c.notRun.join(", ")}`).join("; "));
  const miss = M4.missing(R);
  console.log("   missing metrics: " + Object.keys(miss).filter((k) => miss[k].length).map((k) => `${k}: ${miss[k].join(", ")}`).join(" | "));
}

// Part 3: every UMAP setting's local and global tile for each seed, the
// seed-to-seed ranges, and how much the island layout moves between seeds.
function umap() {
  const PART3 = path.join(POST, "..", "when-umap-lies-tuning");
  require(path.join(PART3, "widget-data", "umap.js"));
  const M3 = require(path.join(PART3, "model.js"));
  const U = window.WK_DATA["uml-umap"];
  const pearson = (a, b) => {
    const ma = a.reduce((s, x) => s + x, 0) / a.length, mb = b.reduce((s, x) => s + x, 0) / b.length;
    let sab = 0, saa = 0, sbb = 0;
    a.forEach((x, i) => { sab += (x - ma) * (b[i] - mb); saa += (x - ma) ** 2; sbb += (b[i] - mb) ** 2; });
    return sab / Math.sqrt(saa * sbb);
  };
  // Distances between cluster centres in one shown map, as a flat vector.
  const centreDists = (xy, labels, k) => {
    const c = Array.from({ length: k }, () => [0, 0, 0]);
    for (let i = 0; i < labels.length; i++) { c[labels[i]][0] += xy[2 * i]; c[labels[i]][1] += xy[2 * i + 1]; c[labels[i]][2]++; }
    const p = c.map((v) => [v[0] / v[2], v[1] / v[2]]), out = [];
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) out.push(Math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1]));
    return out;
  };
  for (const [ds, d] of Object.entries(U.datasets)) {
    console.log(`\n== umap: ${d.title}, n = ${d.n} (${d.shown} shown), seeds ${d.seeds}; local = 10-NN kept, global = distance Spearman`);
    console.log("   " + "nn".padStart(4) + "  md".padEnd(7) + "local s0/s1/s2 (range)".padEnd(32) + "global s0/s1/s2 (range)".padEnd(34) +
      (ds === "gaussian_clusters" ? "centre-distance r between seeds 0-1, 0-2" : ""));
    let maxLocal = 0, maxGlobal = 0, localWithin = 0, total = 0;
    for (const nn of d.nn) for (const md of d.md) {
      const s = M3.setting(U, ds, nn, md, null), L = s.local, G = s.global;
      const lr = L.hi - L.lo, gr = G.hi - G.lo;
      maxLocal = Math.max(maxLocal, lr); maxGlobal = Math.max(maxGlobal, gr); total++;
      if (lr <= 0.01 + 1e-9) localWithin++;
      let extra = "";
      if (ds === "gaussian_clusters") {
        const c = M3.colour(U, ds), k = c.classes.length;
        const cd = d.seeds.map((sd) => centreDists(M3.layout(U, ds, nn, md, sd), c.labels, k));
        extra = `${n(pearson(cd[0], cd[1]), 2)}, ${n(pearson(cd[0], cd[2]), 2)}`;
      }
      console.log("   " + pad(nn, 4) + "  " + String(md).padEnd(5) +
        `${L.seeds.map((x) => n(x, 3)).join("/")} (${n(lr, 3)})`.padEnd(32) +
        `${G.seeds.map((x) => n(x, 3)).join("/")} (${n(gr, 3)})`.padEnd(34) + extra);
    }
    console.log(`   largest seed range: local ${n(maxLocal, 3)}, global ${n(maxGlobal, 3)}; ` +
      `local range <= 0.01 at ${localWithin} of ${total} settings`);
  }
}

const CHECKS = { elephant, dimension, signal, ranking, umap };
// `node check_widgets.js signal all` prints every signal setting.
const args = process.argv.slice(2).filter((a) => a in CHECKS);
for (const name of args.length ? args : Object.keys(CHECKS)) CHECKS[name](process.argv.slice(2));
