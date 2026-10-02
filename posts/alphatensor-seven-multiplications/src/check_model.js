/* Asserts and prints every number "AlphaTensor, Part 2" quotes.
 *
 *   node posts/alphatensor-seven-multiplications/src/check_model.js
 *
 * The model is Part 1's: this post loads ../alphatensor-matmul-cube/model.js.
 */
"use strict";

const assert = require("node:assert/strict");
const M = require("../../alphatensor-matmul-cube/model.js");

const T = M.tensor(2), S = M.STRASSEN;

// A factorization is exact, not approximate: the terms sum to the tensor.
assert.equal(S.U.length, 7);
assert.ok(M.equal(M.sumOfTerms(S), T), "Strassen's seven terms sum to T");
for (const n of [2, 3]) {
  const F = M.schoolbook(n);
  assert.equal(F.U.length, n ** 3);
  assert.ok(M.equal(M.sumOfTerms(F), M.tensor(n)), `schoolbook(${n}) sums to T`);
}
for (const part of [S.U, S.V, S.W])
  assert.ok(part.flat().every((x) => x === -1 || x === 0 || x === 1), "entries in {-1, 0, 1}");

// Building the cube term by term: what the "build" scene shows.
const build = [];
let touched = new Set();
for (let r = 0; r <= 7; r++) {
  const X = M.sumOfTerms(S, r);
  assert.ok(X.every((x) => x >= -1 && x <= 1), "running sum stays in {-1, 0, 1}");
  X.forEach((x, i) => { if (x) touched.add(i); });
  const wrong = X.filter((x, i) => x !== T[i]).length;
  build.push({ r, nonzero: M.nnz(X), wrong });
}
assert.deepEqual(build.map((b) => b.nonzero), [0, 8, 10, 12, 12, 12, 10, 8]);
assert.equal(build[7].wrong, 0);
console.log("Strassen, term by term (nonzero cells in the running sum, cells that differ from T):");
for (const b of build) console.log(`  after ${b.r} terms: ${b.nonzero} nonzero, ${b.wrong} differ from T`);
console.log(`  cells ever nonzero: ${touched.size}; of those, ${touched.size - 8} end at zero`);
assert.equal(touched.size, 20);

// Running it as an algorithm.
const rand = M.mulberry32(20261002);
for (let trial = 0; trial < 1000; trial++) {
  const A = M.randomMatrix(rand, 4, -99, 99), B = M.randomMatrix(rand, 4, -99, 99);
  assert.deepEqual(M.applyAlgorithm(S, A, B).C, M.matmul(2, A, B));
  assert.deepEqual(M.applyAlgorithm(M.schoolbook(2), A, B).C, M.matmul(2, A, B));
}
const ex = M.applyAlgorithm(S, [1, 2, 3, 4], [5, 6, 7, 8]);
assert.deepEqual(ex.C, [19, 22, 43, 50]);
console.log("\nworked example, A = [1 2; 3 4], B = [5 6; 7 8]:");
const f = M.formulas(S, 2);
ex.m.forEach((m, r) => console.log(`  ${f.products[r]} = (${ex.left[r]})(${ex.right[r]}) = ${m}`));
f.outputs.forEach((s, c) => console.log(`  ${s} = ${ex.C[c]}`));

console.log("\nschoolbook as a factorization:");
const fs = M.formulas(M.schoolbook(2), 2);
fs.products.forEach((s) => console.log("  " + s));
fs.outputs.forEach((s) => console.log("  " + s));

console.log(`\nadditions and subtractions per pass: Strassen ${M.additions(S)}, ` +
  `schoolbook ${M.additions(M.schoolbook(2))}`);
assert.equal(M.additions(S), 18);
assert.equal(M.additions(M.schoolbook(2)), 4);

// Recursion on blocks: the algorithm never swaps the order of a product, so
// it runs unchanged when the entries are matrices.
function rows(rand, n) {
  return Array.from({ length: n }, () => M.randomMatrix(rand, n, -9, 9));
}
function plain(A, B) {
  const n = A.length;
  return A.map((row, i) => B[0].map((_, k) => { let s = 0; for (let j = 0; j < n; j++) s += row[j] * B[j][k]; return s; }));
}
for (const k of [1, 2, 3]) {
  const n = 2 ** k, A = rows(rand, n), B = rows(rand, n);
  const fast = M.multiplyRecursive(S, A, B), slow = M.multiplyRecursive(M.schoolbook(2), A, B);
  assert.deepEqual(fast.C, plain(A, B));
  assert.deepEqual(slow.C, plain(A, B));
  assert.equal(fast.mults, 7 ** k);
  assert.equal(slow.mults, 8 ** k);
  assert.equal(M.recursion(k).strassen, fast.mults);
  assert.equal(M.recursion(k).school, slow.mults);
}
for (let k = 0; k <= 17; k++) {
  assert.equal(BigInt(M.recursion(k).school), 8n ** BigInt(k));
  assert.equal(BigInt(M.recursion(k).strassen), 7n ** BigInt(k));
}

console.log(`\nlog2(7) = ${M.OMEGA.toFixed(4)}`);
console.log("recursion on 2 x 2 blocks (k levels, n = 2^k):");
console.log("   k        n        schoolbook 8^k          Strassen 7^k    saved");
for (const k of [1, 2, 3, 4, 5, 6, 8, 10, 12]) {
  const r = M.recursion(k);
  console.log(`  ${String(k).padStart(2)} ${String(r.n).padStart(8)} ${r.school.toLocaleString("en-US").padStart(22)} ` +
    `${r.strassen.toLocaleString("en-US").padStart(22)}   ${(100 * r.saving).toFixed(1)}%`);
}

console.log("\nall Part 2 assertions passed");
