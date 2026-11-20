# umaplab — interface and protocol spec

The shared laboratory behind the four-part series *When UMAP Lies*. Every part imports
`umaplab` from `posts/when-umap-lies/src/`. This file is the contract: implementers fill
in modules to it, and must not change a signature, a column name or a protocol without
the lead's agreement.

Environment: `.venv-umap` (Python 3.12; numpy 2.5, scipy, scikit-learn 1.9, umap-learn
0.6.0, openTSNE 1.0.4, pacmap 0.9.1, trimap 1.2.0, phate 2.0.0, ripser 0.6.15, pandas,
pyarrow, scanpy). Run code with `.venv-umap/bin/python` — never bare `python`/`pip`.
Tests: `.venv-umap/bin/python -m pytest posts/when-umap-lies/tests -q`.
Lint: `.venv/bin/ruff check <files>` and `.venv/bin/ruff format <files>` (Google
docstrings; every public function needs a one-line docstring at least).

Style: type hints everywhere, short meaningful docstrings, pure functions, no global
RNG — every random function takes `seed: int` or `rng: np.random.Generator` and uses
`np.random.default_rng`. No printing. float64 for data, float32 for stored embeddings.

## Package layout

```
posts/when-umap-lies/
  src/umaplab/
    __init__.py
    generators.py   # synthetic data with ground truth, plus ambient transforms
    embeddings.py   # one interface over every DR method
    metrics.py      # local / global / density / cluster / graph metrics
    topology.py     # persistent homology via ripser
    stability.py    # Procrustes alignment and seed-to-seed comparisons
    benchmark.py    # grid runner, caching, results schema
    realdata.py     # PBMC 3k and MNIST loaders (lead writes this)
    widgetdata.py   # export of quantised embeddings for widgets (lead writes this)
  tests/            # pytest; unit tests per module + one integration test
  cache/            # committed results (parquet / npz), read by the posts
  src/run_experiments.py  # CLI: regenerate every cached result
```

## 1. `generators.py`

```python
@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray  # (n, D) float64 — what the DR method sees
    latent: np.ndarray  # (n, q) float64 — ground-truth intrinsic coordinates
    clean: np.ndarray  # (n, p) float64 — the noiseless structured data before any
    #   ambient transform (p = base ambient dim). For a dataset
    #   with no transform, clean is X.
    labels: np.ndarray | None  # (n,) int — class / cluster / branch membership
    density: np.ndarray | None  # (n,) float — true sampling density at each point
    #   (relative, any positive scale), if known
    extra: dict[str, np.ndarray]  # e.g. "pseudotime", "branch", "posterior"
    meta: dict[str, Any]  # generator name, all params, seed, intrinsic_dim,
    #   betti (tuple, true Betti numbers if known),
    #   ambient_dim, transform history (list of dicts)

    def geodesic(self, i: np.ndarray, j: np.ndarray) -> np.ndarray | None:
        """Ground-truth geodesic distances for index pairs, or None if unknown."""
```

Implement `geodesic` by dispatching on `meta["generator"]` (a module-level registry of
geodesic functions keyed by generator name is fine). It must be vectorised over pairs.

### Base generators (signature `fn(n: int, seed: int, **params) -> Dataset`)

| name | params (defaults) | X | latent | truth extras | betti |
|---|---|---|---|---|---|
| `gaussian_clusters` | `n_clusters=10, dim=10, separation=6.0, imbalance=1.0, anisotropy=1.0, within_sd=1.0` | mixture in R^dim | = clean | labels; `extra["centres"]` | (n_clusters,) components |
| `swiss_roll` | `turns=1.5, height=21.0, noise=0.0` | R^3 | (arc length s, height h) — **arc-length parametrised so latent Euclidean distance = intrinsic geodesic** | `extra["s"]` | (1, 0) |
| `circle` | `radius=1.0, noise=0.0, dim=2` | circle in first 2 coords of R^dim | angle (n,1) | — | (1, 1) |
| `sphere` | `m=2, radius=1.0, noise=0.0` | uniform on S^m ⊂ R^{m+1} | the points themselves (unit vectors); geodesic = great-circle | — | S^2: (1,0,1) |
| `branching` | `n_branches=3, dim=10, branch_length=1.0, noise=0.03, levels=1` | a tree of straight segments in R^dim: one trunk, then `n_branches` children from its tip (levels=2: each child splits again in 2) | (pseudotime,) | labels=branch id; `extra["pseudotime"]`; geodesic = path length along the tree (projected positions) | (1, 0) |
| `variable_density` | `dim=3, density_ratio=10.0, shape="strip"` | a 2-D strip (or swiss roll) sampled with density varying smoothly by `density_ratio` along s | (s, h) | `density` = true sampling density | (1, 0) |
| `overlapping` | `dim=10, delta=1.0, weights=(0.5,0.5)` | two Gaussians N(±delta/2 · e1, I_dim) | = clean | labels; `extra["posterior"]` = exact P(class 1 | x) | (1,) |
| `latent_factor` | `d=5, D=None, noise=0.0` | Z ~ N(0, I_d) mapped by a random orthonormal D×d matrix (D defaults to d) | Z | — | (1,) |
| `uniform_ball` | `d=5` | uniform in the unit d-ball | = X | — | (1,) |
| `elephant_target` | `n_points` | 2-D points sampled uniformly inside an elephant silhouette (a hand-made polygon in code, no image files) | = X | — | — |

Scale convention: each base generator documents the typical nearest-neighbour spacing
of its output in its docstring; nothing is standardised silently. `gaussian_clusters`
`separation` is the distance between neighbouring centres in units of `within_sd`;
centres are placed by a seeded random draw scaled so that the **minimum** pairwise
centre distance equals `separation` (rescale after drawing). `imbalance` = ratio of
largest to smallest cluster size (geometric spacing of sizes). `anisotropy` = ratio of
largest to smallest within-cluster sd along random axes.

### Ambient transforms (return a new Dataset; `latent`, `clean`, labels unchanged)

```python
def add_noise_dims(ds: Dataset, D: int, sigma: float, seed: int) -> Dataset
    """X <- [X, sigma * N(0,1)^(n x (D - p))]. D is the TOTAL ambient dim; raise if D < p."""
def add_isotropic_noise(ds: Dataset, sigma: float, seed: int) -> Dataset
    """X <- X + sigma * N(0, I). Used for noise on the structured dims themselves."""
def random_isometry(ds: Dataset, D: int, seed: int) -> Dataset
    """X <- X Q^T with Q a (D x p) matrix with orthonormal columns (QR of a Gaussian).
    Pairwise distances are preserved exactly. D total; raise if D < p."""
```

Note the identity the post relies on: isotropic Gaussian noise is rotation invariant, so
`add_noise_dims(ds, D, s)` and `add_isotropic_noise(random_isometry(ds, D), s)` restricted
to the noise-free structured dims are equal in distribution up to a rotation. Test it.

`meta["transforms"]` appends `{"op": ..., **params}` for every transform.

### Tests (`tests/test_generators.py`)
- shapes, dtypes, determinism under seed, different output under different seed;
- swiss roll: latent distance between near neighbours matches 3-D Euclidean distance
  to < 2% (arc-length param check); `geodesic` ≥ Euclidean for all sampled pairs;
- sphere: points have unit norm; geodesic in [0, π]; circle geodesic ≤ π;
- branching: pseudotime monotone along each branch; geodesic is a metric on samples
  (symmetry, triangle inequality on random triples);
- variable_density: empirical histogram of s matches `density` to within sampling error;
- overlapping: posterior is in [0,1] and is well calibrated (binned frequencies);
- gaussian_clusters: min centre distance == separation; imbalance ratio respected;
- transforms: random_isometry preserves all pairwise distances (atol 1e-8);
  add_noise_dims leaves the first p columns equal to the input.

## 2. `embeddings.py`

```python
@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str              # "linear" | "spectral" | "neighbour-graph SGD" | ...
    preserves: str           # one line: what it tries to preserve
    defaults: dict[str, Any] # the PACKAGE defaults we pass explicitly, recorded
    complexity: str          # e.g. "O(n D k) kNN + O(n) per epoch"
    package: str             # "umap-learn 0.6.0" etc. (read from importlib.metadata)

METHODS: dict[str, MethodSpec]

@dataclass(frozen=True)
class EmbedResult:
    Y: np.ndarray            # (n, 2) float32
    seconds: float           # wall time of the fit only (preprocessing PCA included)
    params: dict[str, Any]   # the full effective params (defaults merged with overrides)
    info: dict[str, Any]     # e.g. kl divergence, n_iter, warnings caught

def embed(method: str, X: np.ndarray, seed: int, params: dict | None = None,
          n_jobs: int = 1) -> EmbedResult
```

Methods (names are the registry keys):

| key | implementation | notes |
|---|---|---|
| `pca` | sklearn PCA(2) | seed irrelevant (use svd_solver="full" for n·D small, "randomized" with seed otherwise — record which) |
| `umap` | umap-learn UMAP, defaults n_neighbors=15, min_dist=0.1, init="spectral" | random_state=seed (this disables UMAP's parallelism; that is acceptable and must be recorded) |
| `tsne` | openTSNE TSNE, defaults perplexity=30, init="pca", learning_rate="auto", early_exaggeration=12 | random_state=seed, n_jobs |
| `pacmap` | pacmap.PaCMAP defaults (n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, init="pca") | random_state=seed |
| `trimap` | trimap.TRIMAP defaults | trimap has no seed arg in 1.2.0 → set `np.random.seed(seed)` immediately before fit inside a local block and document; check determinism in a test and record the result either way |
| `isomap` | sklearn Isomap(n_neighbors=10) | deterministic; O(n^2) memory → registry marks `max_n=5000` |
| `spectral` | sklearn SpectralEmbedding(affinity="nearest_neighbors", n_neighbors=max(n//10,1) default) — pass sklearn's default explicitly | Laplacian eigenmaps |
| `diffmap` | own implementation: kNN Gaussian kernel with adaptive bandwidth ε_i = distance to k-th NN (k=15), symmetric kernel K_ij = exp(-d_ij^2/(ε_i ε_j)), α=1 density normalisation (Coifman–Lafon), Markov normalisation, top non-trivial 2 eigenvectors scaled by λ^t, t=1, sparse eigensolver | document the formula in the docstring |
| `phate` | phate.PHATE defaults (knn=5, decay=40, t="auto") | random_state=seed; verbose=0 |

Every method accepts a `pca_dims: int | None` param (default None): if set and
< X.shape[1], X is first reduced by PCA(pca_dims) (seeded randomized solver), timed as
part of the fit. This is the standard single-cell pipeline step and is a first-class
experimental arm (`umap` with `pca_dims=50` is reported as its own configuration).

`MethodSpec` also carries `max_n: int | None` and `max_D: int | None` (feasibility
limits); `embed` raises `Infeasible` (a custom exception) beyond them, so the runner can
record status="infeasible" instead of crashing.

Thread control: `embed` must not oversubscribe. The runner runs one process per worker
with `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` set before imports; `n_jobs=1` is passed to every method.

Tests (`tests/test_embeddings.py`): every method on a 300-point swiss roll returns
finite (n,2) float32; determinism under the same seed for every method that claims to be
seeded (record trimap's actual behaviour); `pca_dims` path works; Infeasible raised.

## 3. `metrics.py`

All neighbourhood metrics use **exact** Euclidean kNN (sklearn NearestNeighbors,
algorithm="brute" when D > 20), self excluded. `ref` is the reference point cloud
(either `latent`, `clean`, or `X`), `emb` the (n,2) embedding.

```python
def knn_indices(Z: np.ndarray, k: int) -> np.ndarray            # (n, k), self excluded
def knn_preservation(ref, emb, ks: Sequence[int]) -> dict[int, float]
    """Q_NX(k): mean_i |N_k^ref(i) ∩ N_k^emb(i)| / k. Equals kNN precision = recall."""
def rnx(ref, emb, ks) -> dict[int, float]
    """Chance-adjusted R_NX(k) = ((n-1) Q_NX(k) - k) / (n - 1 - k) (Lee & Verleysen)."""
def trustworthiness(ref, emb, k: int) -> float     # Venna & Kaski; may wrap sklearn
def continuity(ref, emb, k: int) -> float          # = trustworthiness(emb, ref, k)
def sample_pairs(n: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]
    """m distinct unordered pairs i<j, reproducible; m capped at n(n-1)/2."""
def distance_correlation(ref_d: np.ndarray, emb_d: np.ndarray) -> dict[str, float]
    """{'spearman': ..., 'pearson': ...} over the sampled pair distances."""
def pair_distances(Z, i, j) -> np.ndarray          # Euclidean, vectorised
def normalized_stress(ref_d, emb_d) -> float
    """sqrt(sum (d - a e)^2 / sum d^2) with the optimal scale a = <d,e>/<e,e>."""
def local_radius(Z, k: int) -> np.ndarray          # distance to k-th neighbour
def density_preservation(ref, emb, k: int = 15) -> float
    """Spearman correlation of log local radius in ref vs emb (densMAP-style)."""
def true_density_correlation(density: np.ndarray, emb, k: int = 15) -> float
    """Spearman of the known sampling density vs 1/r_k^2 in the 2-D embedding."""
def knn_graph(Z, k) -> scipy.sparse.csr_matrix     # symmetrised (union) adjacency
def graph_edge_jaccard(G1, G2) -> float
def n_components(G) -> int
def cluster_scores(Z, labels, seed: int, n_clusters: int | None = None) -> dict
    """In the space Z (embedding or original): silhouette (true labels), 5-fold
    stratified CV accuracy of a 15-NN classifier, and ARI/AMI vs labels of (a) KMeans
    with k = n_true_clusters (n_init=10, seeded) and (b) sklearn HDBSCAN(min_cluster_size=
    max(10, n//100)) with noise points counted as their own label. Return flat dict."""
def evaluate(ds: Dataset, emb: np.ndarray, *, ks=(5, 10, 30, 100), n_pairs=50_000,
             seed: int = 0, refs=("truth", "input")) -> list[dict]
    """Every applicable metric, as long-format rows {metric, k, reference, value}.
    reference="truth" uses ds.latent (and ds.geodesic for geodesic correlation),
    reference="input" uses ds.X. Cluster metrics only when ds.labels is not None;
    true_density only when ds.density is not None. k values >= n are skipped."""
```

Metric names in rows (exact strings — the posts and widgets key on them):
`knn_preservation`, `rnx`, `trustworthiness`, `continuity`, `dist_spearman`,
`dist_pearson`, `geodesic_spearman`, `stress`, `density_spearman`,
`true_density_spearman`, `silhouette`, `knn_accuracy`, `kmeans_ari`, `kmeans_ami`,
`hdbscan_ari`, `hdbscan_ami`, `hdbscan_n_clusters`, `graph_jaccard`, `n_components`.
k is NaN where not applicable. `graph_jaccard` compares the k=15 kNN graph of the
reference with that of the embedding.

Also `input_fidelity(ds, ks) -> list[dict]`: the same local metrics comparing
`ds.latent` (truth) to `ds.X` (input) — the quality of the neighbour graph the method is
handed, before any embedding. Rows get method="input" in the runner.

Validation tests (`tests/test_metrics.py`) with known answers:
- identity embedding of 2-D data → knn_preservation = trustworthiness = continuity = 1,
  dist correlations = 1, stress = 0; a rotation+scale+reflection → same;
- random-permutation embedding → knn_preservation ≈ k/(n-1) (within 3 SE), rnx ≈ 0;
- a hand-built 6-point example where trustworthiness is computed by hand from the
  Venna–Kaski formula, checked against our function and sklearn;
- continuity(ref, emb) == trustworthiness(emb, ref);
- normalized_stress invariant to scaling emb; =0 for a scaled copy;
- sample_pairs: distinct, i<j, deterministic;
- density_preservation = 1 for identity, ≈ 0 for permutation;
- cluster_scores on two far-apart blobs → ARI = 1, accuracy = 1;
- graph_edge_jaccard(G, G) = 1; n_components of two far blobs with k=5 is 2.

## 4. `topology.py`

```python
def subsample(Z, m: int, seed: int, method="maxmin") -> np.ndarray   # indices
def persistence(Z, maxdim: int = 1, m: int = 400, seed: int = 0) -> list[np.ndarray]
    """ripser diagrams on a reproducible maxmin subsample, Z rescaled to unit median
    nearest-neighbour distance... (document the normalisation)."""
def betti_estimate(dgms, threshold: float) -> tuple[int, ...]
    """Number of bars with persistence (death-birth) > threshold × (diameter scale)
    per dimension; infinite H0 bar counts as 1."""
def top_persistence(dgms, dim: int, r: int = 3) -> list[float]   # largest r lifetimes, normalised
def topology_rows(ds, emb, seed) -> list[dict]
    """Rows {metric: 'h1_top1'|'h1_top2'|'h1_ratio'|'betti0'|'betti1', reference: 'input'|'emb', value}."""
```
Tests: noisy circle → one dominant H1 bar (ratio top1/top2 > 3); two blobs → betti0=2;
filled disk → no dominant H1 bar.

## 5. `stability.py`

```python
def procrustes_align(A, B) -> tuple[np.ndarray, float]
    """Align B to A by translation, uniform scale, rotation and reflection; return the
    aligned B and the Procrustes disparity (scipy.spatial.procrustes semantics)."""
def seed_pair_rows(ref_emb, other_emb, ks=(10, 30)) -> list[dict]
    """{metric: 'procrustes_disparity'|'seed_knn_overlap'|'seed_dist_spearman', k, value}"""
```
Tests: a rotated/reflected/scaled copy has disparity < 1e-10 and knn overlap 1.

## 6. `benchmark.py`

```python
@dataclass(frozen=True)
class Cell:
    experiment: str
    generator: str
    gen_params: dict          # passed to the base generator
    transforms: tuple[dict]   # applied in order, each {"op": name, **params}
    n: int
    data_seed: int
    method: str
    method_params: dict
    algo_seed: int
    keep_embedding: bool = False

def cell_id(c: Cell) -> str      # stable sha1 over a canonical JSON of the cell + CODE_VERSION
def build_dataset(c: Cell) -> Dataset   # generator + transforms; transform seeds derive
                                         # from data_seed (never from algo_seed)
def run_cell(c: Cell) -> tuple[dict, list[dict], np.ndarray | None]
    """Generate (cached per process by data key), embed, evaluate (+ topology when the
    generator's meta['betti'] has an H1 entry or the experiment asks), return
    (run_row, metric_rows, embedding_or_None). Never raises: failures become
    status='failed' with the message."""
def run_grid(cells: list[Cell], out: Path, n_workers: int = 8, resume: bool = True) -> None
    """Run cells in a process pool (threads pinned to 1, see embeddings), appending to
    out/runs.parquet, out/metrics.parquet, and out/embeddings.npz (float16, key=cell_id)
    for keep_embedding cells. Resume skips cell_ids already in runs.parquet. Writes are
    batched and atomic (write tmp then rename). Also writes input_fidelity rows once per
    dataset key with method='input'."""
```

### Result schema (long format; the posts and widgets read exactly these columns)

`runs.parquet`: `cell_id, experiment, generator, gen_params (json str), transforms
(json str), D (int, final ambient dim), intrinsic_dim (int), noise_sigma (float, NaN if
none), n, data_seed, method, method_params (json str), config (str: method plus a short
param label, e.g. "umap", "umap·pca50", "tsne·perp5"), algo_seed, seconds, peak_rss_mb
(float, NaN unless measured), status ("ok"|"failed"|"infeasible"), error (str)`.

`metrics.parquet`: `cell_id, metric, k (float, NaN if n/a), reference ("truth"|"input"|
"emb"), value (float)`. Input-fidelity rows use a synthetic cell_id = data key + ":input".

Seeds: `data_seed` indexes an independent draw from the data-generating process;
`algo_seed` the method's RNG. The main sweeps use data_seed ∈ {0..4} with
algo_seed = data_seed (one stochastic run per independent dataset — this estimates total
variability). The nested experiment (Part 3) crosses data_seed ∈ {0,1,2} with
algo_seed ∈ {0..4} to split the variance into data-sampling and algorithm parts.

`CODE_VERSION = "v1"` constant in benchmark.py, included in cell_id.

Integration test (`tests/test_benchmark.py`): a 6-cell grid (2 methods × 3 dims, n=200)
in a tmp dir runs end to end with 2 workers, produces both parquet files with the
schema above, resume re-runs nothing, and a failing method (monkeypatched) yields
status="failed" without killing the grid.
