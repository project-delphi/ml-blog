// Sweep the subset explorer's tolerance and print what its tiles show.
// Run: node posts/two-cultures-rashomon/src/check_model.js
"use strict";
const path = require("node:path");
const assert = require("node:assert/strict");
global.window = {};
require(path.join(__dirname, "../widget-data/data.js"));
const D = window.WK_DATA.subsets;
const M = require(path.join(__dirname, "../widgets.js"));
let prev = 0;
for (let k = 1; k <= 20; k++) {
  const tol = +(0.005 * k).toFixed(3);
  const rows = M.within(D, tol), inc = M.inclusion(D, rows);
  const t = rows.map((r) => r.t);
  assert.ok(rows.length >= prev, "count must grow with tolerance");
  prev = rows.length;
  const used = inc.filter((x) => x > 0).length;
  const st = M.stories(rows).map((r) => r.v.map((j) => D.names[j]).join(","));
  console.log(`tol ${(100 * tol).toFixed(1)}%: ${rows.length} models, ${used} variables, ` +
    `test MSE ${Math.min(...t).toFixed(1)}-${Math.max(...t).toFixed(1)}`);
  if ([0.01, 0.02, 0.05].includes(tol)) st.forEach((s) => console.log("    " + s));
}
console.log("ok");
