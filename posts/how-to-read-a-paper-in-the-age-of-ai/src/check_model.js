/* Prints the pass table behind the deck's two "paper in layers" scenes, so the
 * slides can be read against the post rather than against a screenshot
 * (compare with the post's first table and its protocol section), and every
 * number the citation-graph slide quotes from its Semantic Scholar snapshot.
 * Run from the repo root:
 *
 *   node posts/how-to-read-a-paper-in-the-age-of-ai/src/check_model.js
 */
"use strict";
const M = require("../model.js");

const layerName = (id) => (id === "note" ? "your note" : M.LAYERS.find((l) => l.id === id).short);

for (const mode of M.MODES) {
  console.log(`\n${mode}`);
  for (const p of M.passes(mode)) {
    const lit = p.layers.map(layerName).join(" + ");
    const field = p.candidates ? `  [field: ${p.candidates}]` : "";
    console.log(`  ${p.name.padEnd(38)} ${M.who(p).padEnd(24)} ${p.time.padEnd(17)} ${lit}${field}`);
  }
}

console.log("\nsurvey time (hours): first pass by hand vs AI triage, then the deep reads");
const h = (x) => (Math.round(x * 10) / 10).toString();
for (const n of M.SURVEY.fields) {
  const b = M.budget(n);
  console.log(`  ${String(n).padStart(3)} candidates: by hand ${h(b.hand[0])}–${h(b.hand[1])} h, with AI triage ${h(b.ai)} h; ` +
    `then ${M.SURVEY.deep} papers × Pass 2 = ${h(b.deep)} h either way`);
}

const G = require("../graph.js");
const short = (i) => {
  const n = G.nodes[i];
  return `${n.title.slice(0, 58).padEnd(58)} ${String(n.year).padEnd(5)} ${String(n.cites).padStart(6)} cites`;
};
console.log(`\ncitation graph (snapshot ${G.fetched}, ${G.source})`);
console.log(`  papers citing Keshav: ${G.citingTotal}; with a reference list from the API: ${G.citingWithRefs}; ` +
  `Keshav's own list returned: ${G.seedHasRefs}`);
console.log(`  drawn: ${G.nodes.length} papers, years ${G.yearRange[0]}–${G.yearRange[1]}`);
console.log(`  citation links: ${G.citation.length} (${G.citation.filter((e) => e[1] === 0).length} to Keshav)`);
console.log(`  connected links (>= ${G.MIN_SHARED} shared references, top ${G.KEEP} per paper): ${G.connected.length}, ` +
  `of which ${G.connectedNotCiting} join papers where neither cites the other`);
console.log(`  top ${G.TOP} by citations:`);
G.topCited.forEach((i) => console.log("    " + short(i)));
console.log(`  top ${G.TOP} by connection strength:`);
G.topConnected.forEach((i) => console.log("    " + short(i)));
console.log(`  in both lists: ${G.overlap} of ${G.TOP}`);
console.log(`  citing papers dated before Keshav (${G.Y0}): ${G.predates}; tagged Medicine: ${G.tagged("Medicine")}; ` +
  `tagged Computer Science: ${G.tagged("Computer Science")}`);
console.log(`  drawn by colour: cs ${G.drawnBy("cs")}, med ${G.drawnBy("med")}, other ${G.drawnBy("other")}`);
console.log(`  citation links pointing up in time on the map: ${G.upLinks.length}`);
console.log(`  of the top ${G.TOP} by citations, tagged Medicine (and not CS): ${G.topCitedMed}`);
