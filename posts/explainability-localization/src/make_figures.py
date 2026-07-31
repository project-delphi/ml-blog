"""Regenerate every figure in the post, standalone, and run every build gate.

Run from the post directory:

    ../../.venv-explainability/bin/python src/make_figures.py

The post's code cells call the same functions in ``figures.py``, so the PNGs
written here and the images rendered into the page come from one implementation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution as A
import figures as F
import house as H
import iris_stats as I
import mnist_model as M
import numpy as np
import torch

# The backend is selected in __main__, never at import time. The post imports
# this module to reuse mnist_gates() and background_contrast(), and a module-level
# matplotlib.use("Agg") would switch the notebook off the inline backend and
# silently render every figure in the post as a bare "<Figure size ...>" repr.

OUT = Path(__file__).resolve().parent.parent / "figures"

# Relative tolerance on the completeness axiom. Integrated gradients is
# evaluated in float32 over a 512-point Riemann sum, so exact equality is not
# available; 1e-3 is two orders of magnitude tighter than anything the figure
# depends on and still catches a genuine implementation error.
IG_TOLERANCE = 1e-3


def mnist_gates(t: M.Trained, idx: dict[str, int]) -> list[str]:
    """Check every claim Part I makes about its own arithmetic.

    Args:
        t: The trained bundle.
        idx: Running-example indices.

    Returns:
        One line per check.

    Raises:
        AssertionError: If a check fails.
    """
    lines = []
    for key, i in idx.items():
        x, target = t.xte[i], int(t.pred[i])

        ig = A.integrated_gradients(t.model, x, target)
        delta = A.target_logit(t.model, x, target) - A.target_logit(
            t.model, torch.zeros(1, 1, 28, 28), target
        )
        rel = A.completeness_gap(t.model, x, target, ig) / abs(delta)
        assert rel < IG_TOLERANCE, (key, rel)
        lines.append(f"[{key}] integrated gradients sum to f(x) - f(0) within {rel:.1e} relative")

        # Grad-CAM two ways: through the features/head split, and through a
        # forward hook on the ReLU. Bit-identical or the implementation is wrong.
        direct = A.grad_cam(t.model, x, target)
        hooked = A.grad_cam_hooked(t.model, x, target)
        assert np.array_equal(direct, hooked), key
        lines.append(f"[{key}] Grad-CAM via hooks is bit-identical to Grad-CAM via the head split")

        # On a GAP head, Grad-CAM reduces exactly to Zhou et al.'s CAM, scaled
        # by the 1/49 from averaging over the 7x7 grid.
        gap = float(np.abs(A.cam(t.model, x, target) / 49.0 - direct).max())
        assert gap < 1e-5, (key, gap)
        lines.append(f"[{key}] Grad-CAM equals CAM / 49 to {gap:.1e} absolute")

        # Integrated gradients assigns exactly zero wherever the image equals
        # the black baseline. This is definitional, and the post says so.
        black = t.xte[i, 0].numpy() == 0
        assert np.all(ig[black] == 0.0), key
        lines.append(f"[{key}] all {black.sum()} baseline-valued pixels receive exactly zero from IG")

    # The prose says the single largest gradient in the whole saliency map lands
    # on a pixel the image leaves empty. Unlike the checks above that is a fact
    # about this checkpoint rather than a theorem, so it is gated rather than
    # trusted: a re-render that moves the maximum onto the stroke should fail
    # the build and force the sentence to be rewritten, not silently falsify it.
    i = idx["wrong"]
    sal = np.abs(A.saliency(t.model, t.xte[i], int(t.pred[i])))
    black = t.xte[i, 0].numpy() == 0
    assert black.ravel()[sal.argmax()], "saliency maximum is no longer on a baseline-valued pixel"
    lines.append(
        f"[wrong] the largest saliency value in the map ({sal.max():.2f}) is on a "
        "baseline-valued pixel"
    )

    return lines


def background_contrast(t: M.Trained, i: int) -> dict[str, float]:
    """Quantify what each method says about pixels the image leaves empty.

    Args:
        t: The trained bundle.
        i: Test-set index.

    Returns:
        Counts and magnitudes used verbatim in the post's prose.
    """
    x, target = t.xte[i], int(t.pred[i])
    img = t.xte[i, 0].numpy()
    black = img == 0

    # Pixels whose whole 7x7 occlusion patch is also black: occluding there
    # changes nothing at all, so occlusion is structurally blind to them.
    padded = np.pad(img, 3)
    fully_black = np.array(
        [[bool((padded[r : r + 7, c : c + 7] == 0).all()) for c in range(28)] for r in range(28)]
    )

    sal = A.saliency(t.model, x, target)
    ig = A.integrated_gradients(t.model, x, target)
    occ = A.occlusion(t.model, x, target)
    return {
        "black_pixels": int(black.sum()),
        "fully_black_neighbourhoods": int(fully_black.sum()),
        "ig_nonzero_on_black": int((np.abs(ig[black]) > 0).sum()),
        "saliency_nonzero_on_black": int((np.abs(sal[black]) > 0).sum()),
        "saliency_max_on_black": float(np.abs(sal[black]).max()),
        "saliency_max_on_fully_black": float(np.abs(sal[fully_black]).max()),
        "saliency_max_overall": float(np.abs(sal).max()),
        "occlusion_max_on_fully_black": float(np.abs(occ[fully_black]).max()),
        "occlusion_max_overall": float(np.abs(occ).max()),
    }


def main() -> int:
    """Build every figure and run every gate.

    Returns:
        Process exit code.
    """
    H.use_house_style()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # ---- Part I -----------------------------------------------------------
    t = M.get_trained()
    idx = M.pick_examples(t)
    print(f"MNIST test accuracy {t.accuracy:.4f}")
    for line in mnist_gates(t, idx):
        print("  ok:", line)
    stats = background_contrast(t, idx["wrong"])
    for key, value in stats.items():
        print(f"  {key}: {value}")

    H.save(F.fig_examples(t, idx), OUT / "01-examples.png")
    H.save(
        F.fig_method(
            t, idx, lambda m, x, c: np.abs(A.saliency(m, x, c)), "saliency", False,
            "gradient magnitude of the predicted logit with respect to each pixel",
        ),
        OUT / "02-saliency.png",
    )
    H.save(
        F.fig_method(
            t, idx, A.occlusion, "occlusion", True,
            "drop in the predicted logit when a 7x7 patch is blacked out",
        ),
        OUT / "03-occlusion.png",
    )
    H.save(
        F.fig_method(
            t, idx, A.grad_cam, "Grad-CAM", False,
            "channel-weighted activations of the last convolution, upsampled from 7x7",
        ),
        OUT / "04-gradcam.png",
    )
    H.save(
        F.fig_method(
            t, idx, A.integrated_gradients, "integrated gradients", True,
            "path-integrated attribution from a black baseline; red supports, blue opposes",
        ),
        OUT / "05-integrated-gradients.png",
    )
    H.save(F.fig_sanity(t, idx, seed=H.SEED + 1), OUT / "06-sanity-check.png")
    H.save(F.fig_agreement(t, idx), OUT / "07-agreement.png")

    # ---- Part II ----------------------------------------------------------
    for line in I.self_check():
        print("  ok:", line)

    df, target = I.load()
    X = df[I.FEATURES].to_numpy()

    mask = target > 0
    fit = I.fit_logistic(I.standardise(X[mask]), (target[mask] == 2).astype(float), I.FEATURES)
    H.save(F.fig_coefficients(fit), OUT / "08-coefficients.png")

    path = I.separation_path(
        I.standardise(X), (target == 0).astype(float), (1, 2, 5, 10, 25, 50, 100)
    )
    H.save(F.fig_separation(path), OUT / "09-separation.png")

    anova = I.anova_table(df)
    H.save(F.fig_anova(df, anova), OUT / "10-anova.png")
    H.save(F.fig_pca(X, target), OUT / "11-pca.png")

    split = I.fit_multinomial(seed=H.SEED % 2**31)
    perm = I.permutation_importance(
        split.model.predict, split.Xte, split.yte, repeats=200, rng=H.rng_for("permutation")
    )
    print(f"  multinomial held-out accuracy {split.test_accuracy:.4f}")
    H.save(F.fig_permutation(perm, anova, fit.table().iloc[1:]), OUT / "12-permutation.png")

    written = sorted(p.name for p in OUT.glob("*.png"))
    print(f"\nwrote {len(written)} figures to {OUT} in {time.time() - start:.1f}s")
    for name in written:
        print("  ", name)
    return 0


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    raise SystemExit(main())
