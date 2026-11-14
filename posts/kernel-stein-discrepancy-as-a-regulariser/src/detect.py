"""What a batch regulariser can see: detection rates against N(0, I_d) for laws that
have mean 0 and covariance I_d, so they agree with the target on two moments.

For each law, dimension d and regulariser, draw ``reps`` batches of n = 256 under the
null N(0, I_d) and under the law; the detection rate is the fraction of the law's
batches whose regulariser value exceeds the null's 95th percentile. A regulariser that
depends on the batch only through its first two moments detects at about the 5% level.

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/detect.py

Writes results/detect.json.
"""

import json
import math
from pathlib import Path

import common
import torch

HERE = Path(__file__).resolve().parents[1]
torch.set_default_dtype(torch.float64)


def whiten_mixture(centres, sigma2):
    """Mixture of N(c_k, sigma2 I) with equal weights, mapped to mean 0, covariance I."""
    d = centres.shape[1]
    centres = centres - centres.mean(0)
    cov = centres.T @ centres / len(centres) + sigma2 * torch.eye(d)
    ev, U = torch.linalg.eigh(cov)
    W = U @ torch.diag(ev.rsqrt()) @ U.T
    return centres, W


def sampler(law, d, g):
    if law == "gaussian":
        return lambda n: torch.randn(n, d, generator=g)
    if law == "rademacher":
        return lambda n: torch.randint(0, 2, (n, d), generator=g).double() * 2 - 1
    if law == "student5":
        # t_5 = N(0, 1) / sqrt(chi^2_5 / 5), the chi-square as a sum of five squared
        # normals so every draw comes from g; scaled to variance 1.
        def draw_t(n):
            chi2 = torch.randn(n, d, 5, generator=g).pow(2).sum(-1)
            return (
                torch.randn(n, d, generator=g) / torch.sqrt(chi2 / 5) * math.sqrt(3 / 5)
            )

        return draw_t
    if law == "shell":
        return lambda n: (
            torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=1)
            * math.sqrt(d)
        )
    if law == "clusters":
        k = 10
        centres = torch.randn(k, d, generator=g) * 1.5
        centres, W = whiten_mixture(centres, 0.25)

        def draw(n):
            lab = torch.randint(0, k, (n,), generator=g)
            return (centres[lab] + 0.5 * torch.randn(n, d, generator=g)) @ W

        return draw
    raise ValueError(law)


LAWS = ("gaussian", "rademacher", "student5", "shell", "clusters")
REGS = ("moments", "ksd_imq", "ksd_rbf_wide", "ksd_rbf", "ksd_mix", "mmd", "sigreg")


def main(n=256, reps=200, dims=(2, 8, 64)):
    torch.manual_seed(0)
    out = {
        "n": n,
        "reps": reps,
        "dims": list(dims),
        "laws": list(LAWS),
        "regs": list(REGS),
        "rate": {},
    }
    for d in dims:
        g = torch.Generator().manual_seed(d)
        null = {r: [] for r in REGS}
        for _ in range(reps):
            z = torch.randn(n, d, generator=g)
            for r in REGS:
                null[r].append(float(common.batch_reg(r, z, gen=g)))
        q95 = {r: torch.tensor(v).quantile(0.95).item() for r, v in null.items()}
        for law in LAWS:
            draw = sampler(law, d, torch.Generator().manual_seed(1000 + d))
            alt = {r: [] for r in REGS}
            for _ in range(reps):
                z = draw(n)
                for r in REGS:
                    alt[r].append(float(common.batch_reg(r, z, gen=g)))
            key = f"{law}|{d}"
            out["rate"][key] = {r: sum(v > q95[r] for v in alt[r]) / reps for r in REGS}
            out.setdefault("snr", {})[key] = {
                r: (torch.tensor(alt[r]).mean() - torch.tensor(null[r]).mean()).item()
                / torch.tensor(null[r]).std().item()
                for r in REGS
            }
            print(d, law, out["rate"][f"{law}|{d}"], flush=True)
    path = HERE / "results" / "detect.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
