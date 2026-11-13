"""Shared pieces of the post's two CIFAR-10 experiments: data, augmentation, and the
batch regularisers that push a batch of codes toward N(0, I_d).

Every regulariser takes a batch z of shape (n, d) and returns a scalar. All of them
aim at the same target, the standard normal N(0, I_d), whose score is s(z) = -z.

- ``ksd_imq``  kernel Stein discrepancy, IMQ kernel (1 + |x - y|^2)^(-1/2), U-statistic.
- ``ksd_rbf``  kernel Stein discrepancy, Gaussian kernel at l^2 = d / log n, U-statistic.
- ``ksd_mix``  kernel Stein discrepancy under IMQ + 10 * Gaussian, the Gaussian part at the
               batch-median bandwidth: ksd_imq + 10 ksd_rbf.
- ``moments``  the U-statistic of KSD under the linear kernel 1 + x^T y, an unbiased
               estimate of |E z|^2 + |E zz^T - I|_F^2 (the post proves the identity): KSD
               that sees two moments. ``moments_v`` is the familiar biased V-statistic.
- ``mmd``      maximum mean discrepancy against a fresh sample from N(0, I_d) of the same
               size (unbiased U-statistic), with the Wasserstein autoencoder's kernel.
- ``sigreg``   the Epps-Pulley statistic on random one-dimensional projections, against
               N(0, 1) (Balestriero & LeCun 2025): a sliced characteristic-function test.
- ``vicreg``   VICReg's variance hinge and off-diagonal covariance terms (Bardes et al.
               2022), in its published 25 : 1 ratio.
- ``kl``       the VAE's per-sample KL(N(mu, diag sigma^2) || N(0, I)).
"""

import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CIFAR = Path.home() / ".keras" / "datasets" / "cifar-10-batches-py"


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_cifar(root=CIFAR):
    """CIFAR-10 as uint8 tensors (N, 3, 32, 32) and int64 labels."""

    def batch(name):
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])

    tr = [batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.concatenate([b[0] for b in tr])
    ytr = np.concatenate([b[1] for b in tr])
    xte, yte = batch("test_batch")
    t = torch.from_numpy
    return t(xtr), t(ytr).long(), t(xte), t(yte).long()


# --------------------------------------------------------------------------- augmentation
def augment(x, gen):
    """SimCLR-style views on the GPU: random resized crop (scale 0.2-1, ratio 3/4-4/3),
    horizontal flip, colour jitter (p = 0.8) and grayscale (p = 0.2). x is float in [0, 1].
    """
    n, dev = x.shape[0], x.device

    def rand(*s):
        return torch.rand(*s, generator=gen, device=dev)

    area = 0.2 + 0.8 * rand(n)
    logr = torch.log(torch.tensor(3 / 4)) + rand(n) * torch.log(torch.tensor(16 / 9))
    ratio = torch.exp(logr)
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (rand(n) * 2 - 1) * (1 - w)
    cy = (rand(n) * 2 - 1) * (1 - h)
    flip = torch.where(rand(n) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )

    jit = (rand(n) < 0.8).float().view(n, 1, 1, 1)
    b = 1 + (rand(n) * 0.8 - 0.4).view(n, 1, 1, 1)
    c = 1 + (rand(n) * 0.8 - 0.4).view(n, 1, 1, 1)
    s = 1 + (rand(n) * 0.8 - 0.4).view(n, 1, 1, 1)
    y = x * b
    m = y.mean(dim=(1, 2, 3), keepdim=True)
    y = (y - m) * c + m
    g = 0.299 * y[:, :1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:]
    y = (y - g) * s + g
    x = jit * y.clamp(0, 1) + (1 - jit) * x
    gmask = (rand(n) < 0.2).float().view(n, 1, 1, 1)
    gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]).expand_as(x)
    return gmask * gray + (1 - gmask) * x


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def normalise(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# --------------------------------------------------------------------------- regularisers
def _sqdist(x, y):
    return (
        x.pow(2).sum(1)[:, None] + y.pow(2).sum(1)[None, :] - 2 * x @ y.T
    ).clamp_min(0)


def stein_kernel_imq(x, y, c2, beta=-0.5):
    """Stein kernel of N(0, I_d) under the IMQ base kernel k = (c2 + |x - y|^2)^beta.

    k_p = x^T y k + 2 beta u b^{beta-1} - 2 beta d b^{beta-1} - 4 beta (beta-1) u b^{beta-2},
    with u = |x - y|^2 and b = c2 + u. (Derived in the post; checked against autograd in
    src/check_stein_kernel.py.)
    """
    d = x.shape[1]
    u = _sqdist(x, y)
    b = c2 + u
    k = b.pow(beta)
    k1 = k / b
    k2 = k1 / b
    return (
        (x @ y.T) * k
        + 2 * beta * u * k1
        - 2 * beta * d * k1
        - 4 * beta * (beta - 1) * u * k2
    )


def ksd(z, c2=1.0, beta=-0.5, ustat=True):
    n = z.shape[0]
    kp = stein_kernel_imq(z, z, c2, beta)
    if ustat:
        return (kp.sum() - kp.diagonal().sum()) / (n * (n - 1))
    return kp.mean()


def rbf_l2(n, d):
    """The SVGD median heuristic evaluated on the target: the median squared distance
    between two N(0, I_d) draws is about 2d, and l^2 = med^2 / (2 log n) = d / log n.
    """
    return d / math.log(n)


def stein_kernel_rbf(x, y, l2):
    """Stein kernel of N(0, I_d) under k = exp(-|x - y|^2 / (2 l2)):
    k_p = k (x^T y + d / l2 - u (1 / l2 + 1 / l2^2)).
    """
    d = x.shape[1]
    u = _sqdist(x, y)
    return torch.exp(-u / (2 * l2)) * (x @ y.T + d / l2 - u * (1 / l2 + 1 / l2**2))


def ksd_rbf(z, l2=None, ustat=True):
    n, d = z.shape
    kp = stein_kernel_rbf(z, z, l2 if l2 is not None else rbf_l2(n, d))
    if ustat:
        return (kp.sum() - kp.diagonal().sum()) / (n * (n - 1))
    return kp.mean()


MIX_W = 10.0  # weight of the Gaussian part in the mixed kernel IMQ + MIX_W * Gaussian


def median_l2(z, detach=False):
    """l^2 = med^2 / (2 log n), med^2 the median squared distance between batch codes.
    Gradient flows through the median, so the batch cannot lower the loss by spreading
    out until the kernel stops seeing it (a detached median rewards exactly that;
    ``detach=True`` exists for bandwidth_checks.py to show it).
    """
    n = z.shape[0]
    iu = torch.triu_indices(n, n, 1, device=z.device)
    u = _sqdist(z, z)
    if detach:
        u = u.detach()
    return u[iu[0], iu[1]].median() / (2 * math.log(n))


def ksd_mix(z, w=MIX_W):
    """KSD under the mixed kernel IMQ(c = 1) + w * Gaussian(l^2 = median_l2(z)). A Stein
    kernel is linear in its base kernel, so this is ksd_imq + w * ksd_rbf.
    """
    return ksd(z, c2=1.0) + w * ksd_rbf(z, l2=median_l2(z))


def moments_v(z):
    """|mean|^2 + |M2_hat - I|_F^2: the V-statistic of KSD under the linear kernel
    1 + x^T y. Biased: on N(0, I) batches its mean is (d^2 + 2d) / n, not 0.
    """
    n, d = z.shape
    m = z.mean(0)
    m2 = z.T @ z / n
    return m.pow(2).sum() + (m2 - torch.eye(d, device=z.device)).pow(2).sum()


def moments(z):
    """U-statistic of KSD under the linear kernel: (1 / n(n-1)) sum_{i != j} k_p(z_i, z_j)
    with k_p(x, y) = x^T y (1 + x^T y) - |x|^2 - |y|^2 + d. Unbiased for
    |E z|^2 + |E zz^T - I|_F^2, and O(n d^2) through the sums below.
    """
    n, d = z.shape
    sq = z.pow(2).sum(1)
    s = z.sum(0)
    g = z.T @ z
    cross = s.pow(2).sum() - sq.sum()  # sum_{i != j} x_i^T x_j
    cross2 = g.pow(2).sum() - sq.pow(2).sum()  # sum_{i != j} (x_i^T x_j)^2
    norms = 2 * (n - 1) * sq.sum()  # sum_{i != j} |x_i|^2 + |x_j|^2
    return (cross + cross2 - norms) / (n * (n - 1)) + d


WAE_SCALES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def mmd(z, gen=None):
    """Unbiased MMD^2 between the batch and a fresh N(0, I_d) sample of the same size,
    with the Wasserstein autoencoder's kernel: sum over s of C / (C + |x - y|^2),
    C = 2 d s (Tolstikhin et al. 2018, prior variance 1).
    """
    n, d = z.shape
    y = torch.randn(z.shape, generator=gen, device=z.device)

    def k(a, b):
        u = _sqdist(a, b)
        return sum(2 * d * s / (2 * d * s + u) for s in WAE_SCALES)

    kxx, kyy, kxy = k(z, z), k(y, y), k(z, y)
    off = n * (n - 1)
    return (
        (kxx.sum() - kxx.diagonal().sum()) / off
        + (kyy.sum() - kyy.diagonal().sum()) / off
        - 2 * kxy.mean()
    )


_T = torch.linspace(-5, 5, 17)
_W = torch.exp(-_T.pow(2) / 2)
_TRAP = torch.full((17,), 10 / 16)
_TRAP[0] = _TRAP[-1] = 5 / 16


def sigreg(z, n_slices=256, gen=None):
    """Mean over random unit directions a of the Epps-Pulley statistic
    n * int |phi_n(t) - e^{-t^2/2}|^2 e^{-t^2/2} dt, trapezoid rule on 17 knots in [-5, 5].
    """
    n, d = z.shape
    a = torch.randn(d, n_slices, generator=gen, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    p = z @ a  # (n, M)
    t, w, trap = (v.to(z.device) for v in (_T, _W, _TRAP))
    arg = p[:, :, None] * t  # (n, M, T)
    re = torch.cos(arg).mean(0)  # (M, T)
    im = torch.sin(arg).mean(0)
    target = torch.exp(-t.pow(2) / 2)
    err = (re - target).pow(2) + im.pow(2)
    return n * (err * w * trap).sum(-1).mean()


def vicreg(z, gamma=1.0, eps=1e-4):
    """VICReg's variance hinge plus its off-diagonal covariance term, in the paper's
    25 : 1 weighting, scaled so that lam * vicreg(z) with lam = 1 matches VICReg's variance
    and covariance weights relative to its invariance term for one view. jepa.py averages
    the two views' terms, so there lam = 2 matches VICReg.
    """
    n, d = z.shape
    std = torch.sqrt(z.var(0) + eps)
    var = F.relu(gamma - std).mean()
    zc = z - z.mean(0)
    cov = zc.T @ zc / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + off.pow(2).sum() / d / 25.0


def kl_gauss(mu, logvar):
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1).mean()


REGS = ("ksd_imq", "ksd_mix", "ksd_rbf", "moments", "mmd", "sigreg", "vicreg")
# every name batch_reg accepts
NAMES = REGS + ("ksd_imq_v", "ksd_rbf_wide", "moments_v", "none")


def batch_reg(name, z, gen=None):
    if name == "ksd_imq":
        return ksd(z, c2=1.0, ustat=True)
    if name == "ksd_imq_v":
        return ksd(z, c2=1.0, ustat=False)
    if name == "ksd_rbf":
        return ksd_rbf(z)
    if name == "ksd_rbf_wide":  # the MMD-style median heuristic, l^2 = d
        return ksd_rbf(z, l2=float(z.shape[1]))
    if name == "ksd_mix":
        return ksd_mix(z)
    if name == "moments":
        return moments(z)
    if name == "moments_v":
        return moments_v(z)
    if name == "mmd":
        return mmd(z, gen=gen)
    if name == "sigreg":
        return sigreg(z, gen=gen)
    if name == "vicreg":
        return vicreg(z)
    if name == "none":
        return z.new_zeros(())
    raise ValueError(name)


# --------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def gaussian_diagnostics(z, gen=None, n_slices=512):
    """How far a set of codes (N, d) is from N(0, I_d), by yardsticks none of the
    regularisers optimises directly.

    - mean_norm, cov_err: |mean| and |Cov - I|_F / sqrt(d) (second-moment fit).
    - sliced_w2: mean over random unit directions of the squared 2-Wasserstein distance
      between the projected codes and N(0, 1), from sorted quantiles.
    - excess_kurt: mean |excess kurtosis| of the projections (0 for a Gaussian).
    """
    z = z.double()
    N, d = z.shape
    m = z.mean(0)
    cov = torch.cov(z.T)
    a = torch.randn(d, n_slices, generator=gen, device=z.device, dtype=torch.float64)
    a = a / a.norm(dim=0, keepdim=True)
    p = (z @ a).sort(0).values
    q = torch.distributions.Normal(0.0, 1.0).icdf(
        (torch.arange(N, dtype=torch.float64, device=z.device) + 0.5) / N
    )
    sw2 = (p - q[:, None]).pow(2).mean(0).mean()
    pc = p - p.mean(0)
    k = (pc.pow(4).mean(0) / pc.pow(2).mean(0).pow(2) - 3).abs().mean()
    return {
        "mean_norm": float(m.norm()),
        "cov_err": float(
            (cov - torch.eye(d, dtype=cov.dtype, device=cov.device)).norm() / d**0.5
        ),
        "sliced_w2": float(sw2),
        "excess_kurt": float(k),
    }
