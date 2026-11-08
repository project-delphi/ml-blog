// Tests each sentence the post aims at the reader against the widget data.
// Run: node posts/stein-jepa-nuisance-leakage/src/check_claims.js
"use strict";
const fs = require("fs"), path = require("path");
const root = path.join(__dirname, "..");
global.window = {};
new Function("window", fs.readFileSync(path.join(root, "widget-data/data.js"), "utf8"))(global.window);
const D = window.WK_DATA, M = require(path.join(root, "model.js"));
let bad = 0;
function claim(text, ok) { console.log((ok ? "ok   " : "FAIL ") + text); if (!ok) bad++; }
const get = (d, s, a, arm) => M.cell(D, d, s, a).find(r => r.arm === arm);
const six = ["VICReg", "SIGReg", "sliced KSD", "KSD", "MMD", "EMA"];

// lambda explorer
const lowHigh = M.LAM_ARMS.map(a => { const s = M.lamSeries(D, a); return [a, s[0], s[s.length - 1], s.find(r => r.lam_rel === 1)]; });
claim("leakage higher at the highest lambda than the lowest, every arm", lowHigh.every(x => x[2].leak_r2 > x[1].leak_r2));
claim("content at the highest lambda below content at the tuned lambda, every arm", lowHigh.every(x => x[2].content_r2 < x[3].content_r2));
claim("KSD and sliced KSD curves start at 1", ["KSD", "sliced KSD"].every(a => M.lamSeries(D, a)[0].lam_rel === 1));

// explorer: d = 3 vs d = 24 at sigma 0, alpha 1
const arms7 = ["VICReg", "SIGReg", "sliced KSD", "KSD", "MMD", "EMA", "oracle"];
for (const a of arms7) {
  const lo = get(3, 0, 1, a), hi = get(24, 0, 1, a);
  console.log("  d=3 vs 24 leak", a, lo && lo.leak_r2, hi && hi.leak_r2, "collapsed@3", lo && lo.collapsed_frac);
}
claim("d=24 leakage above d=3 for every arm except none", arms7.every(a => get(24, 0, 1, a).leak_r2 > get(3, 0, 1, a).leak_r2));
// sigma 0 vs 1 at d = 12 and 24
for (const d of [12, 24]) {
  const up = six.filter(a => get(d, 1, 1, a).leak_r2 > get(d, 0, 1, a).leak_r2);
  claim("d=" + d + ": leakage higher at sigma=1 for all but VICReg (got " + up.join(",") + ")", up.length === 5 && !up.includes("VICReg"));
  claim("d=" + d + ": content lower at sigma=1 for every arm", six.every(a => get(d, 1, 1, a).content_r2 < get(d, 0, 1, a).content_r2));
}
// alpha 0 vs 1
const lower = six.filter(a => get(12, 0, 1, a).leak_r2 < get(12, 0, 0, a).leak_r2);
claim("d=12: leakage lower at alpha=1 than alpha=0 for every arm (got " + lower.join(",") + ")", lower.length === 6);
// knn at the base setting
for (const a of ["KSD", "sliced KSD", "MMD"]) {
  const r = get(12, 0, 1, a);
  claim(a + " holds nuisance a linear probe misses: knn " + r.leak_knn_r2 + " vs linear " + r.leak_r2, r.leak_knn_r2 > r.leak_r2 + 0.05);
}
// cells outside the three slices are absent
claim("d=3 sigma=1 is not run", M.cell(D, 3, 1, 1).length === 0);
// width table claims: narrowing d from 24 to 6 lowers leakage and content, all six arms
claim("d 6 vs 24 lowers leakage, six arms", six.every(a => get(6, 0, 1, a).leak_r2 < get(24, 0, 1, a).leak_r2));
claim("d 6 vs 24 lowers content, six arms", six.every(a => get(6, 0, 1, a).content_r2 < get(24, 0, 1, a).content_r2));
process.exit(bad ? 1 : 0);
