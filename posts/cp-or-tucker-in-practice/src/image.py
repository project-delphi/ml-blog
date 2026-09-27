r"""CP and Tucker on an image folded into a four-way tensor: tune, then transfer.

An RGB image is cut into 8 x 8 blocks and folded into

    block row x block column x pixel within block (64) x colour (3)

so the 512 x 512 astronaut becomes 64 x 64 x 64 x 3. The same tensor serves
both jobs the post compares:

- compression: fit every pixel, and find the smallest model whose PSNR clears
  a quality threshold;
- prediction: hide pixels, fit the rest, and find the ranks whose fit best
  predicts pixels held out for validation.

The tuned models then go to a second image, scikit-learn's `china.jpg`. The
pixel-within-block and colour factors learned on the astronaut are held fixed,
and only the image-specific factors are fitted.

Usage (the sweeps write CSVs into data/; `panels` writes image crops):
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/image.py compress
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/image.py predict 0
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/image.py transfer
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import tensorly as tl
from scipy.ndimage import convolve
from skimage import data as skdata
from sklearn.datasets import load_sample_image
from tensorly.decomposition import parafac

DATA = Path(__file__).resolve().parent.parent / "data"
BLOCK = 8
SPLIT = (0.5, 0.25, 0.25)  # train, validation, test shares of pixels
SEEDS = (0, 1, 2)
CP_ITERS = 300
TUCKER_ITERS = 100
PSNR_TARGET = 30.0  # dB, the compression threshold

COMPRESS_CP = (25, 50, 100, 150, 200, 300, 400, 600)
COMPRESS_TUCKER = [
    (r, r, p, c)
    for r, p, c in itertools.product(
        (8, 16, 24, 32, 40, 48, 56, 64), (4, 8, 12, 16, 24, 32, 48, 64), (1, 2, 3)
    )
]
PREDICT_CP = (5, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400)
PREDICT_TUCKER = [
    (r, r, p, c)
    for r, p, c in itertools.product(
        (8, 16, 24, 32, 48, 56, 64), (4, 8, 16, 32, 48), (1, 2, 3)
    )
]


# --- images and the four-way fold -------------------------------------------


def astronaut() -> np.ndarray:
    return skdata.astronaut().astype(np.float64) / 255


def china() -> np.ndarray:
    return load_sample_image("china.jpg").astype(np.float64) / 255


def crop(img: np.ndarray) -> np.ndarray:
    h, w = (s - s % BLOCK for s in img.shape[:2])
    return img[:h, :w]


def fold(img: np.ndarray) -> np.ndarray:
    """H x W x C image -> (H/8) x (W/8) x 64 x C tensor."""
    img = crop(img)
    h, w, c = img.shape
    blocks = img.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, c)
    return blocks.transpose(0, 2, 1, 3, 4).reshape(h // BLOCK, w // BLOCK, BLOCK**2, c)


def unfold_image(x: np.ndarray) -> np.ndarray:
    """Inverse of fold."""
    bh, bw, _, c = x.shape
    blocks = x.reshape(bh, bw, BLOCK, BLOCK, c).transpose(0, 2, 1, 3, 4)
    return blocks.reshape(bh * BLOCK, bw * BLOCK, c)


def psnr(truth: np.ndarray, estimate: np.ndarray, where=None) -> float:
    """Peak signal-to-noise ratio in dB for images scaled to [0, 1]."""
    err = (np.clip(estimate, 0, 1) - truth) ** 2
    if where is not None:
        err = err[where]
    return float(10 * np.log10(1 / err.mean()))


def pixel_split(shape_hw: tuple[int, int], seed: int) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test, drawn per pixel (all colours together)."""
    rng = np.random.default_rng(seed)
    return rng.choice(3, size=shape_hw, p=SPLIT)


def entries(labels: np.ndarray, which: int) -> np.ndarray:
    """Pixel labels -> boolean mask over the folded tensor's entries."""
    hit = (labels == which)[..., None].repeat(3, axis=2)
    return fold(hit.astype(np.float64)) > 0.5


# --- storage -----------------------------------------------------------------


def cp_storage(rank: int, shape: tuple[int, ...], shared=()) -> int:
    return rank * sum(n for m, n in enumerate(shape) if m not in shared)


def tucker_storage(ranks: tuple[int, ...], shape: tuple[int, ...], shared=()) -> int:
    factors = sum(
        n * r for m, (n, r) in enumerate(zip(shape, ranks)) if m not in shared
    )
    return int(np.prod(ranks)) + factors


# --- fitting -------------------------------------------------------------------


def fit_cp(x, rank, mask=None, fixed=None, seed=0, iters=CP_ITERS):
    """CP-ALS with line search; missing entries are re-imputed from the fit each step.

    `fixed` maps a mode to a factor matrix that is held, not fitted.
    """
    fixed = fixed or {}
    rng = np.random.default_rng(seed)
    init = [
        fixed[m] if m in fixed else rng.standard_normal((n, rank))
        for m, n in enumerate(x.shape)
    ]
    cp = parafac(
        x,
        rank,
        init=tl.cp_tensor.CPTensor((np.ones(rank), init)),
        mask=None if mask is None else mask.astype(np.float64),
        fixed_modes=sorted(fixed),
        n_iter_max=iters,
        tol=1e-9,
        linesearch=not fixed,
    )
    return cp


def fit_tucker(x, ranks, mask=None, fixed=None, iters=TUCKER_ITERS, tol=1e-7):
    """HOOI from HOSVD; missing entries re-imputed from the fit each sweep (EM).

    `fixed` maps a mode to an orthonormal factor that is held, not fitted.
    """
    fixed = fixed or {}
    filled = x.copy() if mask is None else np.where(mask, x, x[mask].mean())
    factors = [
        fixed[m] if m in fixed else _leading(tl.unfold(filled, m), r)
        for m, r in enumerate(ranks)
    ]
    scale, previous = np.linalg.norm(x if mask is None else x[mask]), np.inf
    for _ in range(iters):
        for m in range(x.ndim):
            if m in fixed:
                continue
            y = tl.tenalg.multi_mode_dot(filled, factors, skip=m, transpose=True)
            factors[m] = _leading(tl.unfold(y, m), ranks[m])
        core = tl.tenalg.multi_mode_dot(filled, factors, transpose=True)
        estimate = tl.tenalg.multi_mode_dot(core, factors)
        if mask is None:
            error = np.linalg.norm(x - estimate)
        else:
            filled = np.where(mask, x, estimate)
            error = np.linalg.norm((x - estimate)[mask])
        if abs(previous - error) < tol * scale:
            break
        previous = error
    return core, factors


def _leading(matrix: np.ndarray, rank: int) -> np.ndarray:
    return np.linalg.svd(matrix, full_matrices=False)[0][:, :rank]


def interpolate(img: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Baseline: fill each missing pixel with the mean of its observed 3 x 3 neighbours."""
    known, estimate = observed.copy(), img * observed[..., None]
    kernel = np.ones((3, 3))
    while not known.all():
        weight = convolve(known.astype(float), kernel, mode="nearest")
        fresh = (weight > 0) & ~known
        for c in range(img.shape[2]):
            total = convolve(estimate[..., c] * known, kernel, mode="nearest")
            estimate[..., c][fresh] = total[fresh] / weight[fresh]
        known |= fresh
    return estimate


# --- sweeps --------------------------------------------------------------------


def write_rows(name: str, header: list[str], rows) -> None:
    with (DATA / name).open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            fh.flush()
            print(*row, flush=True)


def done_rows(name: str) -> set[tuple[str, str, str]]:
    """(model, rank, ranks) keys already written, so an interrupted sweep resumes.

    Every fit is deterministic given its seed, so a resumed row equals a rerun one.
    """
    path = DATA / name
    if not path.exists():
        return set()
    with path.open() as fh:
        return {(r["model"], r["rank"], r["ranks"]) for r in csv.DictReader(fh)}


def sweep_compress():
    x = fold(astronaut())

    def rows():
        for rank in COMPRESS_CP:
            for seed in SEEDS:
                est = tl.cp_to_tensor(fit_cp(x, rank, seed=seed))
                yield [
                    "cp",
                    rank,
                    "",
                    seed,
                    cp_storage(rank, x.shape),
                    f"{psnr(x, est):.3f}",
                ]
        for ranks in COMPRESS_TUCKER:
            est = tl.tucker_to_tensor(fit_tucker(x, ranks))
            label = "-".join(map(str, ranks))
            yield [
                "tucker",
                "",
                label,
                0,
                tucker_storage(ranks, x.shape),
                f"{psnr(x, est):.3f}",
            ]

    write_rows(
        "image_compress.csv",
        ["model", "rank", "ranks", "seed", "storage", "psnr"],
        rows(),
    )


def sweep_predict(seed: int):
    """One pixel split per process, so the three splits can run side by side."""
    img = crop(astronaut())
    x = fold(img)
    labels = pixel_split(img.shape[:2], seed)
    train, val, test = (entries(labels, k) for k in range(3))
    name = f"image_predict_seed{seed}.csv"
    done = done_rows(name)

    def rows():
        if ("interpolate", "", "") not in done:
            base = interpolate(img, labels == 0)
            yield ["interpolate", "", "", seed, 0, *_scores(x, fold(base), val, test)]
        for rank in PREDICT_CP:
            if ("cp", str(rank), "") in done:
                continue
            est = tl.cp_to_tensor(fit_cp(x, rank, mask=train, seed=seed))
            storage = cp_storage(rank, x.shape)
            yield ["cp", rank, "", seed, storage, *_scores(x, est, val, test)]
        for ranks in PREDICT_TUCKER:
            label = "-".join(map(str, ranks))
            if ("tucker", "", label) in done:
                continue
            est = tl.tucker_to_tensor(fit_tucker(x, ranks, mask=train))
            storage = tucker_storage(ranks, x.shape)
            yield ["tucker", "", label, seed, storage, *_scores(x, est, val, test)]

    header = ["model", "rank", "ranks", "seed", "storage", "val_psnr", "test_psnr"]
    append_rows(name, header, rows())


def append_rows(name: str, header: list[str], rows) -> None:
    path = DATA / name
    fresh = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        if fresh:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            fh.flush()
            print(*row, flush=True)


def predict_runs():
    import pandas as pd

    paths = sorted(DATA.glob("image_predict_seed*.csv"))
    return pd.concat(pd.read_csv(path) for path in paths)


def _scores(x, est, val, test):
    return f"{psnr(x, est, val):.3f}", f"{psnr(x, est, test):.3f}"


# --- choosing ------------------------------------------------------------------


def config(row) -> tuple[int, ...] | int:
    return (
        int(row["rank"])
        if row["model"] == "cp"
        else tuple(map(int, row["ranks"].split("-")))
    )


def choose_compression(runs, target=PSNR_TARGET):
    """Smallest storage whose PSNR clears the target on every seed, per model."""
    runs = runs.fillna({"ranks": ""})
    worst = runs.groupby(["model", "rank", "ranks"], dropna=False).agg(
        storage=("storage", "first"), psnr=("psnr", "min")
    )
    passing = worst[worst.psnr >= target].reset_index()
    return {m: g.sort_values("storage").iloc[0] for m, g in passing.groupby("model")}


def choose_prediction(runs):
    """One-standard-error rule on validation PSNR across the three pixel splits.

    Take the configuration with the best mean validation PSNR, then the smallest
    model whose mean is within one standard error of it.
    """
    runs = runs[runs.model != "interpolate"].fillna({"ranks": ""})
    stats = (
        runs.groupby(["model", "rank", "ranks"], dropna=False)
        .agg(
            storage=("storage", "first"),
            val=("val_psnr", "mean"),
            se=("val_psnr", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
            test=("test_psnr", "mean"),
        )
        .reset_index()
    )
    chosen = {}
    for model, g in stats.groupby("model"):
        best = g.loc[g.val.idxmax()]
        near = g[g.val >= best.val - best.se]
        chosen[model] = near.sort_values("storage").iloc[0]
    return chosen, stats


# --- transfer to a second image ------------------------------------------------

SHARED = (2, 3)  # pixel-within-block and colour modes carry over from the astronaut


def _cp_fixed(cp):
    weights, factors = cp
    factors = [factors[0] * weights, *factors[1:]]
    return {m: factors[m] for m in SHARED}


def _fit(model, x, cfg, mask=None, fixed=None):
    if model == "cp":
        return fit_cp(x, cfg, mask=mask, fixed=fixed)
    return fit_tucker(x, cfg, mask=mask, fixed=fixed)


def _dense(model, fitted):
    return tl.cp_to_tensor(fitted) if model == "cp" else tl.tucker_to_tensor(fitted)


def _shared(model, fitted):
    return _cp_fixed(fitted) if model == "cp" else {m: fitted[1][m] for m in SHARED}


def _storage(model, cfg, shape, shared=()):
    if model == "cp":
        return cp_storage(cfg, shape, shared)
    return tucker_storage(cfg, shape, shared)


def _largest_term(cp, x) -> float:
    """CP's largest rank-one term, relative to the tensor's norm: the degeneracy tell."""
    weights, factors = cp
    sizes = np.prod([np.linalg.norm(f, axis=0) for f in factors], axis=0) * weights
    return float(np.abs(sizes).max() / np.linalg.norm(x))


def transfer():
    import pandas as pd

    squeeze = choose_compression(pd.read_csv(DATA / "image_compress.csv"))
    predict, _ = choose_prediction(predict_runs())

    src_img, dst_img = crop(astronaut()), crop(china())
    src, dst = fold(src_img), fold(dst_img)
    src_labels, dst_labels = (
        pixel_split(src_img.shape[:2], 0),
        pixel_split(dst_img.shape[:2], 0),
    )
    src_train = entries(src_labels, 0)
    dst_train, dst_test = entries(dst_labels, 0), entries(dst_labels, 2)

    rows, panels, terms = [], {}, {}
    for model in ("cp", "tucker"):
        # Compression: learn on every astronaut pixel, reuse two modes on china.
        cfg = config(squeeze[model])
        learned = _fit(model, src, cfg)
        panels[f"astronaut_compress_{model}"] = _dense(model, learned)
        if model == "cp":
            terms["compress"] = _largest_term(learned, src)
        moved = _fit(model, dst, cfg, fixed=_shared(model, learned))
        fresh = _fit(model, dst, cfg)
        panels[f"china_compress_{model}"] = _dense(model, moved)
        for variant, fitted, shared in (
            ("transfer", moved, SHARED),
            ("fresh", fresh, ()),
        ):
            est = _dense(model, fitted)
            storage = _storage(model, cfg, dst.shape, shared)
            rows.append(
                ["compress", model, str(cfg), variant, storage, f"{psnr(dst, est):.3f}"]
            )

        # Prediction: learn on the astronaut's training pixels, reuse two modes on
        # china, fit china's own modes on its training pixels, score its test pixels.
        cfg = config(predict[model])
        # Panels keep the observed pixels and show the model only where it predicts.
        learned = _fit(model, src, cfg, mask=src_train)
        if model == "cp":
            terms["predict"] = _largest_term(learned, src)
        panels[f"astronaut_predict_{model}"] = np.where(
            src_train, src, _dense(model, learned)
        )
        moved = _fit(model, dst, cfg, mask=dst_train, fixed=_shared(model, learned))
        fresh = _fit(model, dst, cfg, mask=dst_train)
        panels[f"china_predict_{model}"] = np.where(
            dst_train, dst, _dense(model, moved)
        )
        for variant, fitted, shared in (
            ("transfer", moved, SHARED),
            ("fresh", fresh, ()),
        ):
            est = _dense(model, fitted)
            storage = _storage(model, cfg, dst.shape, shared)
            score = psnr(dst, est, dst_test)
            rows.append(["predict", model, str(cfg), variant, storage, f"{score:.3f}"])

    panels["astronaut_predict_interpolate"] = fold(
        interpolate(src_img, src_labels == 0)
    )
    base = interpolate(dst_img, dst_labels == 0)
    panels["china_predict_interpolate"] = fold(base)
    rows.append(
        ["predict", "interpolate", "", "", 0, f"{psnr(dst, fold(base), dst_test):.3f}"]
    )

    header = ["job", "model", "config", "variant", "storage", "psnr"]
    write_rows("image_transfer.csv", header, rows)
    (DATA / "image_cp_terms.json").write_text(json.dumps(terms, indent=2) + "\n")
    _save_panels(panels, src_img, dst_img, src_labels, dst_labels)


# Crops shown at full resolution, so block artefacts stay visible.
CROPS = {
    "astronaut": (slice(24, 216), slice(152, 344)),
    "china": (slice(96, 288), slice(224, 416)),
}


def _save_panels(panels, src_img, dst_img, src_labels, dst_labels):
    images = {"astronaut": src_img, "china": dst_img}
    labels = {"astronaut": src_labels, "china": dst_labels}
    out = {}
    for name, img in images.items():
        rows, cols = CROPS[name]
        out[f"{name}_original"] = img[rows, cols]
        out[f"{name}_observed"] = (
            img[rows, cols] * (labels[name][rows, cols] == 0)[..., None]
        )
    for key, tensor in panels.items():
        rows, cols = CROPS[key.split("_")[0]]
        out[key] = np.clip(unfold_image(tensor), 0, 1)[rows, cols]
    np.savez_compressed(
        DATA / "image_panels.npz",
        **{k: (v * 255).round().astype(np.uint8) for k, v in out.items()},
    )


def main() -> None:
    job = sys.argv[1]
    if job == "predict":
        sweep_predict(int(sys.argv[2]))
    else:
        {"compress": sweep_compress, "transfer": transfer}[job]()


if __name__ == "__main__":
    main()
