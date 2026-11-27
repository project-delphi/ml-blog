// Model behind the post's OJS widget: three ways of testing features against
// k-means clusters, run on synthetic data in the browser.
//
// Pure functions of the control values (p, snr, seed). The OJS cells import
// `simulate` and only draw; `src/check_model.mjs` imports the same file in node
// and sweeps the slider grid, so every number the prose quotes comes from here.

export const N = 400; // rows per data set
export const DRIVERS = 5; // features that carry the cluster shift
export const NOMINAL = 0.05;

// ---------------------------------------------------------------- random numbers
function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalSource(rand) {
  let spare = null;
  return function () {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    while (u === 0) u = rand();
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

// ---------------------------------------------------------- Student t tail area
function lgamma(x) {
  const g = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += g[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(a, b, x) {
  const TINY = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}

// Regularised incomplete beta I_x(a, b).
export function betainc(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  if (x < (a + 1) / (a + b + 2)) return Math.exp(lbt) * betacf(a, b, x) / a;
  return 1 - (Math.exp(lbt) * betacf(b, a, 1 - x)) / b;
}

// Two-sided p-value of Student's t with df degrees of freedom.
export function tTwoSided(t, df) {
  return betainc(df / (df + t * t), df / 2, 0.5);
}

// Welch's two-sample t-test, feature by feature. X is row-major n x p.
function welchPvalues(X, n, p, rows, labels) {
  const out = new Float64Array(p);
  const s1 = new Float64Array(p), s0 = new Float64Array(p);
  const q1 = new Float64Array(p), q0 = new Float64Array(p);
  let n1 = 0, n0 = 0;
  for (const i of rows) {
    const base = i * p;
    if (labels[i] === 1) {
      n1++;
      for (let j = 0; j < p; j++) { const v = X[base + j]; s1[j] += v; q1[j] += v * v; }
    } else {
      n0++;
      for (let j = 0; j < p; j++) { const v = X[base + j]; s0[j] += v; q0[j] += v * v; }
    }
  }
  if (n1 < 2 || n0 < 2) { out.degenerate = true; return out.fill(1); }
  for (let j = 0; j < p; j++) {
    const m1 = s1[j] / n1, m0 = s0[j] / n0;
    const v1 = (q1[j] - n1 * m1 * m1) / (n1 - 1), v0 = (q0[j] - n0 * m0 * m0) / (n0 - 1);
    const a = v1 / n1, b = v0 / n0;
    const se = Math.sqrt(a + b);
    if (!(se > 0)) { out[j] = 1; continue; }
    const t = (m1 - m0) / se;
    const df = ((a + b) * (a + b)) / ((a * a) / (n1 - 1) + (b * b) / (n0 - 1));
    out[j] = tTwoSided(t, df);
  }
  return out;
}

// ---------------------------------------------------------------- 2-means
function sqdist(X, base, C, cbase, p) {
  let s = 0;
  for (let j = 0; j < p; j++) { const d = X[base + j] - C[cbase + j]; s += d * d; }
  return s;
}

// Lloyd's algorithm with k-means++ seeding, best of `restarts`. Returns the
// centroids (2 x p) fitted on `rows` of X.
function twoMeans(X, p, rows, rand, restarts = 2, iters = 25) {
  let best = null, bestInertia = Infinity;
  const m = rows.length;
  for (let r = 0; r < restarts; r++) {
    const C = new Float64Array(2 * p);
    const first = rows[Math.floor(rand() * m)];
    for (let j = 0; j < p; j++) C[j] = X[first * p + j];
    // k-means++: second centre drawn with probability proportional to d^2
    const d2 = rows.map((i) => sqdist(X, i * p, C, 0, p));
    let total = d2.reduce((a, b) => a + b, 0), u = rand() * total, pick = rows[m - 1];
    for (let k = 0; k < m; k++) { u -= d2[k]; if (u <= 0) { pick = rows[k]; break; } }
    for (let j = 0; j < p; j++) C[p + j] = X[pick * p + j];
    const lab = new Uint8Array(m);
    let inertia = Infinity;
    for (let it = 0; it < iters; it++) {
      let changed = 0;
      inertia = 0;
      for (let k = 0; k < m; k++) {
        const base = rows[k] * p;
        const a = sqdist(X, base, C, 0, p), b = sqdist(X, base, C, p, p);
        const l = b < a ? 1 : 0;
        inertia += l ? b : a;
        if (l !== lab[k] || it === 0) { changed++; lab[k] = l; }
      }
      const cnt = [0, 0];
      C.fill(0);
      for (let k = 0; k < m; k++) {
        const base = rows[k] * p, cb = lab[k] * p;
        cnt[lab[k]]++;
        for (let j = 0; j < p; j++) C[cb + j] += X[base + j];
      }
      for (let c = 0; c < 2; c++) {
        if (cnt[c] === 0) continue;
        for (let j = 0; j < p; j++) C[c * p + j] /= cnt[c];
      }
      if (changed === 0 && it > 0) break;
    }
    if (inertia < bestInertia) { bestInertia = inertia; best = C; }
  }
  return best;
}

function assign(X, p, rows, C, labels) {
  for (const i of rows) labels[i] = sqdist(X, i * p, C, p, p) < sqdist(X, i * p, C, 0, p) ? 1 : 0;
}

// ---------------------------------------------------------------- one replicate
// Two groups of equal expected size; group 1 is shifted by `snr` noise standard
// deviations on the first DRIVERS features. Noise variance is 1 and known, which
// is what data fission needs.
function replicate(p, snr, rand, normal) {
  const X = new Float64Array(N * p);
  const Z = new Float64Array(N * p);
  for (let i = 0; i < N; i++) {
    const g = rand() < 0.5 ? 1 : 0;
    for (let j = 0; j < p; j++) {
      X[i * p + j] = normal() + (g && j < DRIVERS ? snr : 0);
      Z[i * p + j] = normal();
    }
  }
  const all = Array.from({ length: N }, (_, i) => i);
  const half = N / 2;
  const train = all.slice(0, half), held = all.slice(half);
  const labels = new Uint8Array(N);

  // 1. Double dipping: cluster all rows, test all rows.
  assign(X, p, all, twoMeans(X, p, all, rand), labels);
  const dd = welchPvalues(X, N, p, all, labels);

  // 2. Sample split: cluster the first half, assign and test the second half.
  assign(X, p, held, twoMeans(X, p, train, rand), labels);
  const split = welchPvalues(X, N, p, held, labels);

  // 3. Data fission: f = X + Z clusters, g = X - Z tests. With known unit noise
  //    variance, f and g are independent given the means.
  const F = new Float64Array(N * p), G = new Float64Array(N * p);
  for (let k = 0; k < N * p; k++) { F[k] = X[k] + Z[k]; G[k] = X[k] - Z[k]; }
  assign(F, p, all, twoMeans(F, p, all, rand), labels);
  const fission = welchPvalues(G, N, p, all, labels);

  return { dd, split, fission };
}

export const METHODS = [
  { key: "dd", label: "Double dipping", detail: "cluster all rows, test the same rows" },
  { key: "split", label: "Sample split", detail: "cluster half, test the other half" },
  { key: "fission", label: "Data fission", detail: "cluster X + Z, test X − Z" },
];

// Replicates needed so roughly 2,000 null features enter each rate.
export function replicatesFor(p) {
  return Math.min(200, Math.max(1, Math.ceil(2000 / p)));
}

// Everything the widget draws, as a function of the three controls.
export function simulate({ p, snr, seed = 1 }) {
  p = Math.round(p);
  const rand = mulberry32(seed * 7919 + p * 31 + Math.round(snr * 1000));
  const normal = normalSource(rand);
  const reps = replicatesFor(p);
  const bonferroni = NOMINAL / p;
  const tally = {};
  for (const m of METHODS) tally[m.key] = { fp: 0, nulls: 0, hits: 0, drivers: 0, degenerate: 0 };
  let first = null;
  for (let r = 0; r < reps; r++) {
    const res = replicate(p, snr, rand, normal);
    if (r === 0) first = res;
    for (const m of METHODS) {
      const pv = res[m.key], t = tally[m.key];
      if (pv.degenerate) { t.degenerate++; continue; } // no test possible
      for (let j = 0; j < p; j++) {
        const isDriver = snr > 0 && j < DRIVERS;
        if (isDriver) { t.drivers++; if (pv[j] < bonferroni) t.hits++; }
        else { t.nulls++; if (pv[j] < NOMINAL) t.fp++; }
      }
    }
  }
  const points = [];
  for (const m of METHODS) {
    const pv = first[m.key];
    for (let j = 0; j < p; j++) {
      points.push({
        method: m.label,
        feature: j + 1,
        kind: snr > 0 && j < DRIVERS ? "driver" : "noise",
        p: pv[j],
        score: -Math.log10(Math.max(pv[j], 1e-300)),
      });
    }
  }
  const rates = METHODS.map((m) => {
    const t = tally[m.key];
    return {
      method: m.label,
      detail: m.detail,
      falsePositive: t.nulls ? t.fp / t.nulls : null,
      driverPower: t.drivers ? t.hits / t.drivers : null,
      driverHits: t.hits,
      driverTests: t.drivers,
      // replicates whose tested rows all fell in one cluster (no test possible)
      oneCluster: t.degenerate,
    };
  });
  return { p, snr, reps, n: N, bonferroni, points, rates };
}
