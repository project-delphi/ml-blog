/* Asserts and prints every number "AlphaTensor, Part 1" quotes.
 *
 *   node posts/alphatensor-matmul-cube/src/check_model.js
 *
 * Run it after touching model.js: Parts 2 and 3 load that file too, so run
 * their check scripts as well.
 */
"use strict";

const assert = require("node:assert/strict");
const M = require("../model.js");

for (const n of [2, 3]) {
  const N = n * n, T = M.tensor(n);
  assert.equal(T.length, N ** 3);
  assert.ok(T.every((x) => x === 0 || x === 1), "entries are 0 or 1");
  assert.equal(M.nnz(T), n ** 3, "n^3 ones");
  // Each output entry's slice holds n ones: the n products of one row-times-column.
  for (let c = 0; c < N; c++) {
    let ones = 0;
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) ones += T[M.cell(N, a, b, c)];
    assert.equal(ones, n, `slice ${c} has ${n} ones`);
  }
  // Reading the product off the tensor is matrix multiplication, at one
  // multiplication per 1.
  const rand = M.mulberry32(20261002 + n);
  for (let trial = 0; trial < 200; trial++) {
    const A = M.randomMatrix(rand, N, -9, 9), B = M.randomMatrix(rand, N, -9, 9);
    const got = M.readOff(T, A, B);
    assert.deepEqual(got.C, M.matmul(n, A, B));
    assert.equal(got.mults, n ** 3);
  }
  console.log(`n = ${n}: ${N} x ${N} x ${N} = ${N ** 3} cells, ${M.nnz(T)} ones, ` +
    `${(100 * M.nnz(T) / N ** 3).toFixed(1)}% full`);
}

console.log("\nthe eight products of the 2 x 2 rule, as cells (a, b, c):");
for (const t of M.terms(2))
  console.log(`  ${M.name("a", 2, t.a)} · ${M.name("b", 2, t.b)} → ${M.name("c", 2, t.c)}` +
    `   cell (${t.a + 1}, ${t.b + 1}, ${t.c + 1})`);

console.log("\nthe four slices, one per output entry (rows a11..a22, columns b11..b22):");
const T2 = M.tensor(2);
for (let c = 0; c < 4; c++) {
  console.log(`  ${M.name("c", 2, c)}:`);
  for (let a = 0; a < 4; a++)
    console.log("    " + [0, 1, 2, 3].map((b) => T2[M.cell(4, a, b, c)]).join(" "));
}

// The worked example the series carries.
const A = [1, 2, 3, 4], B = [5, 6, 7, 8];
console.log(`\nworked example: A = [${A}], B = [${B}], AB = [${M.matmul(2, A, B)}]`);
assert.deepEqual(M.matmul(2, A, B), [19, 22, 43, 50]);
assert.deepEqual(M.readOff(T2, A, B).C, [19, 22, 43, 50]);

console.log("\nsizes for larger n:");
for (const n of [2, 3, 4, 5])
  console.log(`  n = ${n}: side ${n * n}, cells ${(n * n) ** 3}, ones ${n ** 3}`);

console.log("\nall Part 1 assertions passed");
