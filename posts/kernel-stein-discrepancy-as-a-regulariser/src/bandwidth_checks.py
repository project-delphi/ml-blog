"""The two numbers behind the mixed kernel's design, at d = 64 and n = 256.

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/bandwidth_checks.py

1. The weight w = 10: the ratio of the IMQ part's standard deviation to the Gaussian
   part's (l^2 = d / log n), each a KSD U-statistic on Gaussian batches.
2. Why the gradient flows through the median bandwidth: for batches s * N(0, I), the
   share of the descent direction that points outward, -grad . z / (|grad| |z|), for the
   Gaussian part at the batch-median bandwidth with the median detached and with it
   differentiated, and for IMQ. Positive above s = 1 means descent spreads an already
   too-wide batch. Three seeds; the range is printed.
"""

import common
import torch

torch.set_default_dtype(torch.float64)
D, N = 64, 256


def gauss_median(z, detach):
    return common.ksd_rbf(z, l2=common.median_l2(z, detach=detach))


g = torch.Generator().manual_seed(0)
imq = torch.tensor(
    [float(common.ksd(torch.randn(N, D, generator=g), c2=1.0)) for _ in range(150)]
)
rbf = torch.tensor(
    [float(common.ksd_rbf(torch.randn(N, D, generator=g))) for _ in range(150)]
)
print(
    f"null sd: IMQ {imq.std():.2e}, Gaussian {rbf.std():.2e}, ratio {imq.std() / rbf.std():.1f} (w = 10)"
)

LOSSES = {
    "Gaussian, median detached": lambda z: gauss_median(z, True),
    "Gaussian, median differentiated": lambda z: gauss_median(z, False),
    "IMQ": lambda z: common.ksd(z, c2=1.0),
}
for name, f in LOSSES.items():
    row = []
    for s in (0.3, 0.7, 1.5, 3.0, 10.0):
        shares = []
        for seed in range(3):
            gs = torch.Generator().manual_seed(100 + seed)
            z = (s * torch.randn(N, D, generator=gs)).requires_grad_(True)
            f(z).backward()
            shares.append(
                float((-z.grad * z.detach()).sum() / z.detach().norm() / z.grad.norm())
            )
        row.append(f"s={s}: {min(shares):+.2f}..{max(shares):+.2f}")
    print(f"{name:32s} outward share  " + "   ".join(row))
