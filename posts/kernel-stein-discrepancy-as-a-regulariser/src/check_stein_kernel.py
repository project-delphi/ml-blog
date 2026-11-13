"""Check the post's closed forms against automatic differentiation, and the estimator
facts it quotes. Exits non-zero on any failure.

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/check_stein_kernel.py

1. The IMQ Stein kernel of N(0, I) (common.stein_kernel_imq) and the Gaussian one
   (common.stein_kernel_rbf) against the generic four-term formula
   k_p = s(x)^T s(y) k + s(x)^T grad_y k + s(y)^T grad_x k + tr(grad_x grad_y^T k).
2. The linear kernel 1 + x^T y: its V-statistic KSD^2 equals |mean|^2 + |M2 - I|_F^2
   (common.moments_v), and common.moments equals its U-statistic by brute force.
3. On Gaussian batches (d = 64, n = 256) the U-statistic averages 0 and the
   V-statistic (d^2 + 2d) / n.
"""

import sys

import common
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
failed = 0


def check(ok, msg):
    global failed
    print(("ok   " if ok else "FAIL ") + msg)
    failed += not ok


def generic(kfun, x, y):
    d = x.shape[0]
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    k = kfun(x, y)
    gx, gy = torch.autograd.grad(k, (x, y), create_graph=True)
    tr = sum(torch.autograd.grad(gx[a], y, retain_graph=True)[0][a] for a in range(d))
    return float((x @ y) * k - x @ gy - y @ gx + tr)  # s(x) = -x, s(y) = -y


d = 5
for c2, beta in [(1.0, -0.5), (3.0, -0.3), (0.5, -0.8)]:
    worst = 0.0
    for _ in range(20):
        x, y = torch.randn(d) * 1.5, torch.randn(d)
        g = generic(lambda a, b: (c2 + (a - b).pow(2).sum()) ** beta, x, y)
        worst = max(
            worst,
            abs(g - float(common.stein_kernel_imq(x[None], y[None], c2, beta)[0, 0])),
        )
    check(
        worst < 1e-12, f"IMQ Stein kernel, c2={c2}, beta={beta}: max error {worst:.1e}"
    )
for l2 in (0.5, 1.0, 6.0):
    worst = 0.0
    for _ in range(20):
        x, y = torch.randn(d), torch.randn(d) * 0.8
        g = generic(lambda a, b: torch.exp(-(a - b).pow(2).sum() / (2 * l2)), x, y)
        worst = max(
            worst, abs(g - float(common.stein_kernel_rbf(x[None], y[None], l2)[0, 0]))
        )
    check(worst < 1e-12, f"Gaussian Stein kernel, l2={l2}: max error {worst:.1e}")

z = torch.randn(40, d) * 1.3 + 0.2
kp = torch.tensor(
    [
        [generic(lambda a, b: 1 + a @ b, z[i], z[j]) for j in range(40)]
        for i in range(40)
    ]
)
check(
    abs(float(kp.mean()) - float(common.moments_v(z))) < 1e-10,
    "linear kernel: V-statistic == moments_v",
)
off = (kp.sum() - kp.diagonal().sum()) / (40 * 39)
check(
    abs(float(off) - float(common.moments(z))) < 1e-10,
    "linear kernel: U-statistic == moments",
)

g = torch.Generator().manual_seed(0)
dd, n = 64, 256
u = torch.tensor(
    [float(common.moments(torch.randn(n, dd, generator=g))) for _ in range(400)]
)
v = torch.tensor(
    [float(common.moments_v(torch.randn(n, dd, generator=g))) for _ in range(400)]
)
check(
    abs(float(u.mean())) < 3 * float(u.std()) / 20,
    f"moments U-statistic on N(0, I) batches: mean {u.mean():.3f}",
)
check(
    abs(float(v.mean()) - (dd * dd + 2 * dd) / n) < 0.2,
    f"moments V-statistic: mean {v.mean():.3f} vs (d^2 + 2d)/n = {(dd * dd + 2 * dd) / n:.3f}",
)
sys.exit(1 if failed else 0)
