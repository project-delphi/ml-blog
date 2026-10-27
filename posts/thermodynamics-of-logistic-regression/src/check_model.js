"use strict";

const assert = require("node:assert/strict");
const M = require("../model.js");

assert.equal(M.samples.length, 10);
for (const w1 of [-1.2, 0, 0.8, 1.2]) {
  for (const w2 of [-1.2, -0.65, 0, 1.2]) {
    for (const b of [-1, 0, 0.2, 1]) {
      const p = { w1, w2, b };
      const s = M.probeStats(p, 1);
      assert.ok(Math.abs(s.eta - M.score(0.55, -1.25, p)) < 1e-12);
      assert.ok(Math.abs(s.surprisal + Math.log(s.probability)) < 1e-12);
      assert.equal(M.probeStats(p, 0).probability, 0.5);
      const direct = M.samples.reduce((sum, row) => {
        const probability = M.sigmoid(M.score(row[0], row[1], p));
        return sum - (row[2] ? Math.log(probability) : Math.log1p(-probability));
      }, 0) / M.samples.length;
      assert.ok(Math.abs(M.loss(p, false) - direct) < 1e-12);
      const eta = M.score(0.55, -1.25, p);
      assert.ok(Math.abs(M.loss(p, true) - M.loss(p, false) - eta / 10) < 1e-12);
    }
  }
}
console.log("Thermodynamics widget model: 64 settings and both label states pass.");
