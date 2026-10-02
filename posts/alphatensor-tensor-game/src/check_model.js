/* Asserts and prints every number "AlphaTensor, Part 3" quotes.
 *
 *   node posts/alphatensor-tensor-game/src/check_model.js
 *
 * The model is Part 1's: this post loads ../alphatensor-matmul-cube/model.js.
 */
"use strict";

const assert = require("node:assert/strict");
const M = require("../../alphatensor-matmul-cube/model.js");

const N = 4, T = M.tensor(2), S = M.STRASSEN;

// ------------------------------------------------------------- the moves
const mv = M.moves(N);
assert.equal(M.vectors(N, -1, 1).length, 80);          // 3^4 - 1
assert.equal(mv.u.length, 40);
assert.equal(mv.count, 128000);                         // 80^3 / 4
const seen = new Set();
for (const u of mv.u) for (const v of mv.v) for (const w of mv.w) seen.add(M.rankOne(u, v, w).join(","));
assert.equal(seen.size, 128000, "the 128,000 moves are 128,000 different tensors");
console.log(`moves with entries in {-1, 0, 1} on the 2 x 2 cube: ${3 ** 12} triples (u, v, w), ` +
  `${80 ** 3} with no zero vector, ${mv.count} distinct rank-one tensors`);

// -------------------------------------------- Strassen's path goes uphill
const trail = M.trajectory(T, S);
assert.deepEqual(trail, [8, 12, 12, 12, 10, 8, 4, 0]);
console.log(`\nnonzero cells left as Strassen's seven moves are played in order: ${trail.join(" → ")}`);

function permutations(xs) {
  if (xs.length <= 1) return [xs];
  return xs.flatMap((x, i) => permutations(xs.slice(0, i).concat(xs.slice(i + 1))).map((p) => [x, ...p]));
}
let lowPeak = Infinity, highPeak = 0, orders = 0, neverUp = 0;
for (const order of permutations([0, 1, 2, 3, 4, 5, 6])) {
  const t = M.trajectory(T, S, order), peak = Math.max(...t);
  orders++;
  lowPeak = Math.min(lowPeak, peak); highPeak = Math.max(highPeak, peak);
  if (t.every((x, i) => i === 0 || x <= t[i - 1])) neverUp++;
}
assert.equal(orders, 5040);
assert.equal(neverUp, 0, "no ordering of Strassen's moves is downhill all the way");
assert.ok(lowPeak >= 10);
console.log(`over all ${orders} orderings of those seven moves, the peak number of nonzero cells is ` +
  `between ${lowPeak} and ${highPeak}; orderings that never go up: ${neverUp}`);

// ------------------------------------------------------- the first move
const first = M.census(T, N);
assert.equal(first.total, 128000);
assert.equal(first.lower, 8);
console.log(`\nfirst move from T: ${first.lower} moves lower the nonzero count, ${first.same} leave it ` +
  `unchanged, ${first.raise} raise it (by up to ${first.maxRise})`);

// --------------------------------------------------------------- greedy
const g = M.greedyPeel(T, N);
assert.ok(g.solved);
assert.equal(g.moves.length, 8);
for (const m of g.moves) {
  const X = M.rankOne(m.u, m.v, m.w);
  assert.equal(M.nnz(X), 1, "each greedy move is a single cell");
  assert.ok(X.every((x, i) => !x || T[i] === 1), "and that cell is a 1 of T");
}
console.log(`\ngreedy from T: ${g.moves.length} moves, nonzero cells ${g.trail.join(" → ")}`);

// The same result under other tie-breaks: every way of playing the 8
// count-lowering moves is the schoolbook rule in some order.
const lowering = [];
for (const u of mv.u) for (const v of mv.v) for (const w of mv.w) {
  const X = M.rankOne(u, v, w);
  if (M.nnz(M.subtract(T, X)) < 8) lowering.push(X);
}
assert.equal(lowering.length, 8);
assert.ok(lowering.every((X) => M.nnz(X) === 1));
console.log("the 8 lowering first moves are the 8 single cells of T: any greedy tie-break plays the schoolbook rule");

console.log("\ngreedy after k of Strassen's moves (moves played in total, solved?):");
let R = T.slice();
for (let k = 0; k <= 7; k++) {
  if (k > 0) R = M.subtract(R, M.rankOne(S.U[k - 1], S.V[k - 1], S.W[k - 1]));
  const gk = M.greedyPeel(R, N);
  console.log(`  k = ${k}: ${k} + ${gk.moves.length} = ${k + gk.moves.length} moves, solved ${gk.solved}`);
  if (k === 0) assert.equal(gk.moves.length, 8);
  if (k === 1) {
    // Greedy's answer to Strassen's first move is to take it back.
    const back = M.rankOne(gk.moves[0].u, gk.moves[0].v, gk.moves[0].w);
    const fwd = M.rankOne(S.U[0], S.V[0], S.W[0]);
    assert.ok(M.equal(back, fwd.map((x) => -x)), "greedy undoes Strassen's first move");
    console.log("         its first reply is the negative of Strassen's move: it takes the move back");
  }
}

// --------------------------------------------------------- search space
function digits(big) { return big.toString().length; }
console.log("\nchoices at one move, f^(3 n^2), and move sequences of a given length:");
console.log("   n   f   depth   log10(one move)   log10(sequences)");
const rows = [[2, 3, 7], [2, 5, 7], [3, 5, 23], [4, 5, 49], [4, 5, 47], [5, 5, 98]];
for (const [n, f, depth] of rows) {
  const one = M.log10Moves(n, f), all = M.log10Games(n, f, depth);
  const exactOne = BigInt(f) ** BigInt(3 * n * n), exactAll = exactOne ** BigInt(depth);
  assert.equal(Math.floor(one + 1e-9), digits(exactOne) - 1);
  assert.equal(Math.floor(all + 1e-9), digits(exactAll) - 1);
  console.log(`  ${String(n).padStart(2)}  ${String(f).padStart(2)}  ${String(depth).padStart(6)}  ` +
    `${one.toFixed(2).padStart(16)}  ${all.toFixed(2).padStart(17)}`);
}
console.log(`  2 x 2, f = 5, one move: ${(5n ** 12n).toLocaleString("en-US")}`);
console.log(`  3 x 3, f = 5, one move: ${(5n ** 27n).toLocaleString("en-US")}`);
console.log(`  4 x 4, f = 5, one move: ${(5n ** 48n).toLocaleString("en-US")}`);

console.log(`  3 x 3, f = 3, one move: ${(3n ** 27n).toLocaleString("en-US")}`);
console.log(`  4 x 4, f = 3, one move: ${(3n ** 48n).toLocaleString("en-US")}`);

// The widget's "years at a billion games a second" tile, for the settings the prose quotes.
const YEAR = 31557600;
for (const [n, f, depth] of [[2, 5, 7], [4, 5, 49]]) {
  const logYears = M.log10Games(n, f, depth) - 9 - Math.log10(YEAR);
  console.log(`  n = ${n}, f = ${f}, ${depth} moves: 10^${logYears.toFixed(2)} years at 1e9 games a second ` +
    `(${(10 ** (logYears - Math.floor(logYears))).toFixed(1)} x 10^${Math.floor(logYears)})`);
}

// The most generous count for the 2 x 2 game: entries in {-1, 0, 1}, each
// block listed once, and the order of the seven moves ignored.
function choose(n, k) { let r = 1n; for (let i = 0n; i < k; i++) r = r * (n - i) / (i + 1n); return r; }
const sets = choose(128000n, 7n);
const logSets = digits(sets) - 1 + Math.log10(Number(sets.toString().slice(0, 15)) / 1e14);
console.log(`\nunordered sets of 7 distinct {-1, 0, 1} blocks on the 2 x 2 cube: ${sets.toLocaleString("en-US")}`);
console.log(`  = 10^${logSets.toFixed(2)}; at 1e9 a second: 10^${(logSets - 9 - Math.log10(YEAR)).toFixed(2)} years`);
assert.equal(digits(sets), 33);

console.log("\nall Part 3 assertions passed");
