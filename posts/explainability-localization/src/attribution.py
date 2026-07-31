"""Four ways to localise a logit onto the parts of an image.

Every function here takes an image in ``[0, 1]`` and a target class, and returns
a 28x28 map. They differ in what they treat as a *part* and in what they mean by
*responsible*:

===================== ============== ==================================
function              part           responsible means
===================== ============== ==================================
``saliency``          pixel          the local gradient is large
``occlusion``         image patch    hiding it costs the model score
``grad_cam``          conv channel   the channel is active and it matters
``integrated_gradients``  pixel      accumulated gradient along a path
                                     from a baseline to the image
===================== ============== ==================================

All four differentiate or perturb in ``[0, 1]`` pixel space, not in the
network's normalised input space, so "set this pixel to the baseline" means an
actually black pixel rather than a pixel at the dataset mean.
"""

from __future__ import annotations

import numpy as np
import torch
from mnist_model import SmallCNN, normalise


def _as_batch(x01: torch.Tensor) -> torch.Tensor:
    """Coerce a 28x28 or 1x28x28 image to a 1x1x28x28 batch.

    Args:
        x01: Image in [0, 1].

    Returns:
        A 4-D tensor of shape (1, 1, 28, 28).
    """
    return x01.reshape(1, 1, 28, 28).clone()


def target_logit(model: SmallCNN, x01: torch.Tensor, target: int) -> float:
    """Return the model's logit for one class on one image.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class index.

    Returns:
        The target logit as a float.
    """
    with torch.no_grad():
        return float(model(normalise(_as_batch(x01)))[0, target])


def saliency(model: SmallCNN, x01: torch.Tensor, target: int) -> np.ndarray:
    """Gradient of the target logit with respect to each input pixel.

    This is the one-term Taylor expansion of the model around this image: the
    logit's sensitivity to an infinitesimal nudge of each pixel, holding the
    rest fixed. Magnitude answers "which pixels could change the answer fastest",
    which is not the same question as "which pixels produced this answer".

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose logit is differentiated.

    Returns:
        Signed 28x28 gradient map.
    """
    x = _as_batch(x01).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    model(normalise(x))[0, target].backward()
    return x.grad[0, 0].detach().numpy()


def occlusion(
    model: SmallCNN,
    x01: torch.Tensor,
    target: int,
    patch: int = 7,
    fill: float = 0.0,
) -> np.ndarray:
    """Drop in the target logit when a patch of the image is blanked out.

    Responsibility here is defined by removal: slide a square of ``fill`` over
    every position, ask what the model loses, and credit the centre pixel with
    the loss. Positive means the patch was supporting the prediction. Unlike a
    gradient this is a finite, visible perturbation — and unlike a gradient it
    depends entirely on what "removed" is taken to mean.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose logit is tracked.
        patch: Side length of the occluding square, in pixels.
        fill: Value written into the occluded square.

    Returns:
        28x28 map of logit drops, one per patch centre.
    """
    base = target_logit(model, x01, target)
    half = patch // 2
    img = _as_batch(x01)

    batch, centres = [], []
    for r in range(28):
        for c in range(28):
            occluded = img.clone()
            occluded[
                :, :, max(0, r - half) : r + half + 1, max(0, c - half) : c + half + 1
            ] = fill
            batch.append(occluded)
            centres.append((r, c))

    with torch.no_grad():
        logits = model(normalise(torch.cat(batch)))[:, target].numpy()

    out = np.zeros((28, 28), dtype=np.float64)
    for (r, c), value in zip(centres, logits):
        out[r, c] = base - value
    return out


def grad_cam(model: SmallCNN, x01: torch.Tensor, target: int) -> np.ndarray:
    """Grad-CAM over the final convolutional layer.

    The parts are the 128 channels of the last convolution, not pixels. Each
    channel gets a weight equal to the average gradient of the target logit over
    its 7x7 map — read as "how much does turning this channel up help?" — and the
    channels are then summed with those weights. The ReLU keeps only evidence
    *for* the class, discarding anything arguing against it. The result is a 7x7
    map, upsampled to pixel resolution for display; its coarseness is intrinsic,
    not an artefact of the upsampling.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose logit is differentiated.

    Returns:
        28x28 map, non-negative, upsampled from the 7x7 conv grid.
    """
    activations = model.features(normalise(_as_batch(x01)))  # (1, 128, 7, 7)
    activations.retain_grad()
    model.zero_grad(set_to_none=True)
    model.head(activations)[0, target].backward()

    weights = activations.grad[0].mean(dim=(1, 2))  # (128,) channel weights
    cam = torch.relu((weights[:, None, None] * activations[0]).sum(0))  # (7, 7)
    return _upsample(cam.detach().numpy())


def _upsample(cam7: np.ndarray) -> np.ndarray:
    """Bilinearly resize a 7x7 Grad-CAM map to 28x28.

    Args:
        cam7: Map on the convolutional grid.

    Returns:
        28x28 map.
    """
    t = torch.tensor(cam7, dtype=torch.float32).reshape(1, 1, *cam7.shape)
    up = torch.nn.functional.interpolate(t, size=(28, 28), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def cam(model: SmallCNN, x01: torch.Tensor, target: int) -> np.ndarray:
    """The original class activation map: the head's own weights, no gradients.

    Zhou et al.'s CAM predates Grad-CAM and works only for a network ending in
    global average pooling and one linear layer — exactly this one. It weights
    each channel by that layer's weight for the target class. Grad-CAM was
    invented to lift the same idea to architectures without that structure, and
    on a network that *does* have it the two agree exactly, up to the 1/49 from
    averaging over the 7x7 grid. The post asserts that identity rather than
    citing it.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose weights are used.

    Returns:
        28x28 map, non-negative, upsampled from the 7x7 conv grid.
    """
    with torch.no_grad():
        activations = model.features(normalise(_as_batch(x01)))[0]  # (128, 7, 7)
        w = model.fc.weight[target]  # (128,)
        raw = torch.relu((w[:, None, None] * activations).sum(0))
    return _upsample(raw.numpy())


def grad_cam_hooked(model: SmallCNN, x01: torch.Tensor, target: int) -> np.ndarray:
    """Grad-CAM computed through forward and backward hooks.

    Functionally identical to :func:`grad_cam`, which reads the activations
    directly through the model's ``features``/``head`` split. This version exists
    because hooks are how Grad-CAM is written for a network you cannot edit, and
    because agreement between the two is a check that neither is wrong.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose logit is differentiated.

    Returns:
        28x28 map, non-negative.
    """
    caught: dict[str, torch.Tensor] = {}

    def grab(module, inputs, output):
        output.retain_grad()
        caught["act"] = output

    handle = model.act3.register_forward_hook(grab)
    try:
        model.zero_grad(set_to_none=True)
        model(normalise(_as_batch(x01)))[0, target].backward()
    finally:
        handle.remove()

    activations = caught["act"]
    weights = activations.grad[0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * activations[0]).sum(0))
    return _upsample(cam.detach().numpy())


def integrated_gradients(
    model: SmallCNN,
    x01: torch.Tensor,
    target: int,
    baseline: torch.Tensor | None = None,
    steps: int = 512,
) -> np.ndarray:
    """Integrated gradients from a baseline image to this image.

    Average the gradient along the straight line from ``baseline`` to ``x01``,
    then multiply by the displacement. The multiplication is what buys
    *completeness*: the attributions sum exactly to the logit difference between
    the image and the baseline, so the explanation is a genuine decomposition of
    a number rather than a heatmap of sensitivities. It is also what makes the
    baseline decisive — any pixel equal to the baseline is assigned exactly zero,
    whatever the model does there.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose logit is attributed.
        baseline: Reference image in [0, 1]; defaults to all black.
        steps: Number of Riemann points along the path.

    Returns:
        Signed 28x28 attribution map summing to f(x) - f(baseline).
    """
    x = _as_batch(x01)
    base = torch.zeros_like(x) if baseline is None else _as_batch(baseline)
    # Midpoint rule: unbiased for the linear part and far more accurate than
    # left endpoints at the same cost, which matters because completeness is
    # checked numerically rather than assumed.
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    path = base + alphas.reshape(-1, 1, 1, 1) * (x - base)
    path.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    model(normalise(path))[:, target].sum().backward()

    avg_grad = path.grad.mean(0, keepdim=True)
    return ((x - base) * avg_grad)[0, 0].detach().numpy()


def completeness_gap(
    model: SmallCNN,
    x01: torch.Tensor,
    target: int,
    attribution: np.ndarray,
    baseline: torch.Tensor | None = None,
) -> float:
    """Absolute error in the completeness axiom for an attribution map.

    Args:
        model: Trained network.
        x01: Image in [0, 1].
        target: Class whose logit was attributed.
        attribution: The 28x28 map to test.
        baseline: Reference image; defaults to all black.

    Returns:
        ``|sum(attribution) - (f(x) - f(baseline))|``.
    """
    base = torch.zeros_like(_as_batch(x01)) if baseline is None else baseline
    delta = target_logit(model, x01, target) - target_logit(model, base, target)
    return abs(float(attribution.sum()) - delta)


def randomised_copy(model: SmallCNN, seed: int) -> SmallCNN:
    """Return a copy of the network with every weight freshly re-initialised.

    Used for the model-randomisation sanity check: an attribution method that
    produces the same picture for a trained and an untrained network is reading
    the image, not the model.

    Args:
        model: Trained network, used only for its architecture.
        seed: Seed for the re-initialisation.

    Returns:
        An untrained network of the same shape, in eval mode.
    """
    torch.manual_seed(seed)
    fresh = SmallCNN()
    fresh.eval()
    return fresh


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two attribution maps.

    Args:
        a: First map.
        b: Second map.

    Returns:
        Spearman's rho over all 784 pixels.
    """
    from scipy.stats import spearmanr

    return float(spearmanr(a.ravel(), b.ravel()).statistic)
