/* The numbers behind the three AlphaTensor posts: the matrix multiplication
 * tensor, the algorithms that are its factorizations, the game of subtracting
 * rank-one tensors from it, and the size of that game.
 *
 * One file serves the series. It lives in Part 1's folder; Parts 2 and 3 load
 * it by relative path, and all three `src/check_model.js` scripts require it.
 * Nothing here draws. The widgets read these functions, and the check scripts
 * assert and print every figure the prose quotes.
 *
 * Conventions. For n x n matrices, N = n * n. A matrix is a flat row-major
 * array of length N, so entry (i, j) sits at i * n + j. The tensor is a flat
 * array of length N^3 with cell (a, b, c) at (a * N + b) * N + c: `a` indexes
 * an entry of A, `b` an entry of B, `c` an entry of C = AB. A factorization
 * is {U, V, W}, each a list of R vectors of length N; term r is
 * U[r] o V[r] o W[r].
 */
"use strict";

var MatmulModel = (function () {
  var SUB = "₀₁₂₃₄₅₆₇₈₉";

  function zeros(len) { var x = [], i; for (i = 0; i < len; i++) x.push(0); return x; }
  function cell(N, a, b, c) { return (a * N + b) * N + c; }

  /** The n x n matrix multiplication tensor: T[a, b, c] = 1 when the product
   *  A_a * B_b is one of the terms of C_c, that is when a = (i, j), b = (j, k)
   *  and c = (i, k). It has n^3 ones among N^3 cells. */
  function tensor(n) {
    var N = n * n, T = zeros(N * N * N), i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) for (k = 0; k < n; k++)
      T[cell(N, i * n + j, j * n + k, i * n + k)] = 1;
    return T;
  }

  /** The n^3 products of the schoolbook rule, as {a, b, c}, grouped by output entry. */
  function terms(n) {
    var out = [], i, j, k;
    for (i = 0; i < n; i++) for (k = 0; k < n; k++) for (j = 0; j < n; j++)
      out.push({ a: i * n + j, b: j * n + k, c: i * n + k });
    return out;
  }

  /** "a₁₂" for letter "a" and flat index 1 of a 2 x 2 matrix. */
  function name(letter, n, idx) {
    return letter + SUB.charAt(Math.floor(idx / n) + 1) + SUB.charAt(idx % n + 1);
  }
  function names(letter, n) {
    var out = [], i;
    for (i = 0; i < n * n; i++) out.push(name(letter, n, i));
    return out;
  }

  // ------------------------------------------------------------ tensors
  /** u o v o w: the tensor whose (a, b, c) cell is u[a] * v[b] * w[c]. */
  function rankOne(u, v, w) {
    var N = u.length, X = zeros(N * N * N), a, b, c;
    for (a = 0; a < N; a++) if (u[a]) for (b = 0; b < N; b++) if (v[b]) for (c = 0; c < N; c++)
      X[cell(N, a, b, c)] = u[a] * v[b] * w[c];
    return X;
  }
  function add(X, Y) { return X.map(function (x, i) { return x + Y[i]; }); }
  function subtract(X, Y) { return X.map(function (x, i) { return x - Y[i]; }); }
  function nnz(X) { var k = 0, i; for (i = 0; i < X.length; i++) if (X[i]) k++; return k; }
  function isZero(X) { return nnz(X) === 0; }
  function equal(X, Y) {
    if (X.length !== Y.length) return false;
    for (var i = 0; i < X.length; i++) if (X[i] !== Y[i]) return false;
    return true;
  }

  // --------------------------------------------------------- algorithms
  /** Strassen's seven products (Strassen 1969), one row per product. Rows of
   *  U weight (a11, a12, a21, a22), rows of V weight (b11, b12, b21, b22), and
   *  rows of W say which of (c11, c12, c21, c22) the product is added to. */
  var STRASSEN = {
    U: [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]],
    V: [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
    W: [[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
  };

  /** The schoolbook rule as a factorization: one term per 1 in the tensor,
   *  each term a single cell. */
  function schoolbook(n) {
    var N = n * n, F = { U: [], V: [], W: [] };
    terms(n).forEach(function (t) {
      var u = zeros(N), v = zeros(N), w = zeros(N);
      u[t.a] = 1; v[t.b] = 1; w[t.c] = 1;
      F.U.push(u); F.V.push(v); F.W.push(w);
    });
    return F;
  }

  /** The sum of the first `count` terms of a factorization (all of them by default). */
  function sumOfTerms(F, count) {
    var N = F.U[0].length, X = zeros(N * N * N), r;
    if (count === undefined) count = F.U.length;
    for (r = 0; r < count; r++) X = add(X, rankOne(F.U[r], F.V[r], F.W[r]));
    return X;
  }

  function dot(x, y) { var s = 0, i; for (i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }

  /** Run a factorization as an algorithm on flat matrices A and B. Term r
   *  costs one multiplication: m[r] = (U[r] . A) * (V[r] . B). */
  function applyAlgorithm(F, A, B) {
    var R = F.U.length, N = A.length, left = [], right = [], m = [], C = zeros(N), r, c;
    for (r = 0; r < R; r++) {
      left.push(dot(F.U[r], A));
      right.push(dot(F.V[r], B));
      m.push(left[r] * right[r]);
      for (c = 0; c < N; c++) C[c] += F.W[r][c] * m[r];
    }
    return { left: left, right: right, m: m, C: C };
  }

  /** The schoolbook product of two flat n x n matrices. */
  function matmul(n, A, B) {
    var C = zeros(n * n), i, j, k;
    for (i = 0; i < n; i++) for (k = 0; k < n; k++) for (j = 0; j < n; j++)
      C[i * n + k] += A[i * n + j] * B[j * n + k];
    return C;
  }

  /** The product read straight off the tensor: C_c = sum over (a, b) of
   *  T[a, b, c] * A_a * B_b. Also returns how many multiplications it did. */
  function readOff(T, A, B) {
    var N = A.length, C = zeros(N), mults = 0, a, b, c;
    for (a = 0; a < N; a++) for (b = 0; b < N; b++) for (c = 0; c < N; c++)
      if (T[cell(N, a, b, c)]) { C[c] += T[cell(N, a, b, c)] * A[a] * B[b]; mults++; }
    return { C: C, mults: mults };
  }

  /** "a₁₁ + a₂₂" for a weight vector over named entries; a positive term leads. */
  function combo(vec, labels) {
    var parts = [], i;
    for (i = 0; i < vec.length; i++) if (vec[i] > 0) parts.push({ k: vec[i], s: labels[i] });
    for (i = 0; i < vec.length; i++) if (vec[i] < 0) parts.push({ k: vec[i], s: labels[i] });
    if (!parts.length) return "0";
    return parts.map(function (p, j) {
      var mag = Math.abs(p.k) === 1 ? "" : String(Math.abs(p.k));
      if (j === 0) return (p.k < 0 ? "−" : "") + mag + p.s;
      return (p.k < 0 ? " − " : " + ") + mag + p.s;
    }).join("");
  }

  /** The algorithm a factorization spells out, as text: the two sums each
   *  product multiplies, and the products each output entry adds up. */
  function formulas(F, n) {
    var R = F.U.length, N = n * n, an = names("a", n), bn = names("b", n), cn = names("c", n);
    var mn = [], r, c, out = { left: [], right: [], products: [], outputs: [] };
    for (r = 0; r < R; r++) mn.push("m" + String(r + 1).split("").map(function (d) { return SUB.charAt(+d); }).join(""));
    for (r = 0; r < R; r++) {
      out.left.push(combo(F.U[r], an));
      out.right.push(combo(F.V[r], bn));
      out.products.push(mn[r] + " = (" + out.left[r] + ")(" + out.right[r] + ")");
    }
    for (c = 0; c < N; c++)
      out.outputs.push(cn[c] + " = " + combo(F.W.map(function (w) { return w[c]; }), mn));
    return out;
  }

  /** Additions and subtractions one pass of the algorithm does. */
  function additions(F) {
    var N = F.U[0].length, total = 0, c;
    function extra(vec) { return Math.max(0, vec.filter(function (x) { return x !== 0; }).length - 1); }
    F.U.forEach(function (u) { total += extra(u); });
    F.V.forEach(function (v) { total += extra(v); });
    for (c = 0; c < N; c++) total += extra(F.W.map(function (w) { return w[c]; }));
    return total;
  }

  // Matrices as arrays of rows, for the recursion.
  function madd(X, Y, s) {
    return X.map(function (row, i) { return row.map(function (x, j) { return x + s * Y[i][j]; }); });
  }
  function mzero(n) { var X = [], i; for (i = 0; i < n; i++) X.push(zeros(n)); return X; }
  function blocks(X) {
    var h = X.length / 2, out = [], bi, bj;
    for (bi = 0; bi < 2; bi++) for (bj = 0; bj < 2; bj++)
      out.push(X.slice(bi * h, bi * h + h).map(function (row) { return row.slice(bj * h, bj * h + h); }));
    return out;
  }

  /** Multiply two 2^k x 2^k matrices (arrays of rows) by applying a 2 x 2
   *  factorization to their four blocks, and again inside each block product.
   *  Returns the product and how many scalar multiplications it took. */
  function multiplyRecursive(F, A, B) {
    var n = A.length;
    if (n === 1) return { C: [[A[0][0] * B[0][0]]], mults: 1 };
    var h = n / 2, Ab = blocks(A), Bb = blocks(B), Cb = [mzero(h), mzero(h), mzero(h), mzero(h)], mults = 0;
    F.U.forEach(function (u, r) {
      var L = mzero(h), Rt = mzero(h), i;
      for (i = 0; i < 4; i++) {
        if (u[i]) L = madd(L, Ab[i], u[i]);
        if (F.V[r][i]) Rt = madd(Rt, Bb[i], F.V[r][i]);
      }
      var p = multiplyRecursive(F, L, Rt);
      mults += p.mults;
      for (i = 0; i < 4; i++) if (F.W[r][i]) Cb[i] = madd(Cb[i], p.C, F.W[r][i]);
    });
    var C = [], i;
    for (i = 0; i < h; i++) C.push(Cb[0][i].concat(Cb[1][i]));
    for (i = 0; i < h; i++) C.push(Cb[2][i].concat(Cb[3][i]));
    return { C: C, mults: mults };
  }

  /** Multiplications to multiply two 2^k x 2^k matrices by recursion on 2 x 2
   *  blocks: 8 per level for the schoolbook rule, 7 for Strassen's. Exact in a
   *  double up to k = 17. */
  function recursion(k) {
    var school = 1, strassen = 1, i;
    for (i = 0; i < k; i++) { school *= 8; strassen *= 7; }
    return { k: k, n: Math.pow(2, k), school: school, strassen: strassen, saving: 1 - strassen / school };
  }
  var OMEGA = Math.log(7) / Math.log(2);

  // --------------------------------------------------------------- game
  /** Every nonzero vector of length N with entries in lo..hi, in base-(hi-lo+1)
   *  order with the first entry most significant. */
  function vectors(N, lo, hi) {
    var f = hi - lo + 1, total = Math.pow(f, N), out = [], t, i, x, v, any;
    for (t = 0; t < total; t++) {
      v = zeros(N); x = t; any = false;
      for (i = N - 1; i >= 0; i--) { v[i] = x % f + lo; x = Math.floor(x / f); if (v[i]) any = true; }
      if (any) out.push(v);
    }
    return out;
  }
  /** Keep the vectors whose first nonzero entry is positive: one of each +/- pair. */
  function canonical(vs) {
    return vs.filter(function (v) {
      for (var i = 0; i < v.length; i++) if (v[i]) return v[i] > 0;
      return false;
    });
  }
  /** The moves with entries in {-1, 0, 1}, each rank-one tensor listed once:
   *  flipping the sign of two of u, v, w leaves the tensor unchanged, so u and
   *  v are taken with a positive leading entry and w runs over both signs. */
  function moves(N) {
    var all = vectors(N, -1, 1), half = canonical(all);
    return { u: half, v: half, w: all, count: half.length * half.length * all.length };
  }

  /** The score a greedy player minimises: nonzero cells first, then the sum
   *  of absolute values, so a cell holding 2 counts as further from zero. */
  function score(X) {
    var k = 0, s = 0, i;
    for (i = 0; i < X.length; i++) if (X[i]) { k++; s += Math.abs(X[i]); }
    return { nnz: k, abs: s };
  }
  function better(p, q) { return p.nnz < q.nnz || (p.nnz === q.nnz && p.abs < q.abs); }

  /** Visit every {-1, 0, 1} move with the residual it would leave scored.
   *  `visit(u, v, w, nnz, abs)`. */
  function scan(R, N, visit) {
    var mv = moves(N), P = zeros(N * N), ui, vi, wi, a, b, c, u, v, w, k, s, x, p;
    for (ui = 0; ui < mv.u.length; ui++) {
      u = mv.u[ui];
      for (vi = 0; vi < mv.v.length; vi++) {
        v = mv.v[vi];
        for (a = 0; a < N; a++) for (b = 0; b < N; b++) P[a * N + b] = u[a] * v[b];
        for (wi = 0; wi < mv.w.length; wi++) {
          w = mv.w[wi]; k = 0; s = 0;
          for (p = 0; p < N * N; p++) for (c = 0; c < N; c++) {
            x = R[p * N + c] - P[p] * w[c];
            if (x) { k++; s += x < 0 ? -x : x; }
          }
          visit(u, v, w, k, s);
        }
      }
    }
  }

  /** The move that leaves the best-scoring residual; the first one found wins ties. */
  function bestMove(R, N) {
    var best = null;
    scan(R, N, function (u, v, w, k, s) {
      if (!best || better({ nnz: k, abs: s }, best)) best = { u: u, v: v, w: w, nnz: k, abs: s };
    });
    return best;
  }

  /** Play greedily from R: take the best move while it improves the score. */
  function greedyPeel(R, N, maxMoves) {
    var played = [], trail = [nnz(R)], now = score(R), best;
    while (now.nnz > 0 && played.length < (maxMoves || 64)) {
      best = bestMove(R, N);
      if (!better(best, now)) break;
      R = subtract(R, rankOne(best.u, best.v, best.w));
      now = { nnz: best.nnz, abs: best.abs };
      played.push({ u: best.u, v: best.v, w: best.w });
      trail.push(now.nnz);
    }
    return { moves: played, trail: trail, residual: R, solved: now.nnz === 0 };
  }

  /** How the {-1, 0, 1} moves change the number of nonzero cells of R. */
  function census(R, N) {
    var before = nnz(R), out = { lower: 0, same: 0, raise: 0, maxRise: 0, total: 0 };
    scan(R, N, function (u, v, w, k) {
      out.total++;
      if (k < before) out.lower++; else if (k === before) out.same++; else out.raise++;
      if (k - before > out.maxRise) out.maxRise = k - before;
    });
    return out;
  }

  /** Nonzero cells left after each term of F is subtracted from R, in `order`. */
  function trajectory(R, F, order) {
    var trail = [nnz(R)];
    (order || F.U.map(function (_, r) { return r; })).forEach(function (r) {
      R = subtract(R, rankOne(F.U[r], F.V[r], F.W[r]));
      trail.push(nnz(R));
    });
    return trail;
  }

  // ------------------------------------------------------- search space
  function log10(x) { return Math.log(x) / Math.LN10; }
  /** log10 of the choices at one move: three vectors of length n^2, each
   *  entry one of f values, so f^(3 n^2) triples (all-zero vectors included). */
  function log10Moves(n, f) { return 3 * n * n * log10(f); }
  /** log10 of the number of move sequences of a given length. */
  function log10Games(n, f, depth) { return depth * log10Moves(n, f); }

  /** Seeded integers, for the checks and the "random matrices" button. */
  function mulberry32(seed) {
    var s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) >>> 0;
      var t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function randomMatrix(rand, len, lo, hi) {
    var out = [], i;
    for (i = 0; i < len; i++) out.push(lo + Math.floor(rand() * (hi - lo + 1)));
    return out;
  }

  var api = {
    cell: cell, tensor: tensor, terms: terms, name: name, names: names,
    rankOne: rankOne, add: add, subtract: subtract, nnz: nnz, isZero: isZero, equal: equal,
    STRASSEN: STRASSEN, schoolbook: schoolbook, sumOfTerms: sumOfTerms,
    applyAlgorithm: applyAlgorithm, matmul: matmul, readOff: readOff,
    combo: combo, formulas: formulas, additions: additions,
    multiplyRecursive: multiplyRecursive, recursion: recursion, OMEGA: OMEGA,
    vectors: vectors, canonical: canonical, moves: moves, score: score, better: better,
    bestMove: bestMove, greedyPeel: greedyPeel, census: census, trajectory: trajectory,
    log10Moves: log10Moves, log10Games: log10Games,
    mulberry32: mulberry32, randomMatrix: randomMatrix
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  return api;
})();
