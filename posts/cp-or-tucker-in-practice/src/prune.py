r"""Over-parameterised CP with a factor penalty: does the penalty find the rank?

The tensor is synthetic, so the true rank is known and recovery can be graded:
a rank-6 CP tensor of shape 60 x 50 x 40 with Gaussian factors, scaled to unit
Frobenius norm, plus Gaussian noise of norm 0.1. Every fit starts at rank 30.

Three penalties on the factor matrices and an unpenalised baseline, all fitted
by proximal gradient descent with backtracking. Every step ends by equalising
each component's norms across modes: that leaves the fit unchanged, can only
lower the penalty, and conditions the next gradient step, so the baseline gets
it too.

- none:  no penalty
- group: sum over columns of the column's L2 norm (group lasso)
- l1:    sum of absolute entries
- wd:    sum of squared entries (weight decay)

`fit` is imported by the post for one demonstration fit. Run as a script, it
sweeps penalty strengths and seeds and writes data/prune_<kind>.csv, one file
per penalty so the four can run side by side.

Usage:
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/prune.py group
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

DATA = Path(__file__).resolve().parent.parent / "data"
SHAPE = (60, 50, 40)
TRUE_RANK = 6
FIT_RANK = 30
NOISE = 0.1
ITERS = 4000
MAX_STEP = 1e3  # backtracking only ever shrinks the step; this bounds its growth
ALIVE = 1e-6  # a component with weight below this, on a unit-norm tensor, is gone
STRENGTHS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SEEDS = range(5)


def make_problem(seed: int):
    """Noiseless tensor, noisy tensor, and a small random rank-30 start."""
    g = torch.Generator().manual_seed(seed)
    truth = [torch.randn(n, TRUE_RANK, generator=g) for n in SHAPE]
    clean = cp_tensor(truth)
    clean = clean / clean.norm()
    noise = torch.randn(SHAPE, generator=g)
    noisy = clean + NOISE * noise / noise.norm()
    start = [0.3 * torch.randn(n, FIT_RANK, generator=g) for n in SHAPE]
    return clean, noisy, start


def cp_tensor(factors):
    return torch.einsum("ir,jr,kr->ijk", *factors)


def component_weights(factors):
    """lambda_r: the product over modes of column r's L2 norm."""
    return torch.stack([f.norm(dim=0) for f in factors]).prod(0)


def penalty(factors, kind):
    if kind == "group":
        return sum(f.norm(dim=0).sum() for f in factors)
    if kind == "l1":
        return sum(f.abs().sum() for f in factors)
    if kind == "wd":
        return sum((f**2).sum() for f in factors)
    return torch.tensor(0.0)


def prox(factors, kind, step, strength):
    """Proximal map of the non-smooth penalties; weight decay stays in the gradient."""
    if kind == "group":
        return [
            f * (1 - step * strength / f.norm(dim=0).clamp_min(1e-300)).clamp(min=0)
            for f in factors
        ]
    if kind == "l1":
        return [f.sign() * (f.abs() - step * strength).clamp(min=0) for f in factors]
    return factors


def rebalance(factors, kind):
    """Rescale each component's columns to equal norm; the product is unchanged.

    A component with one all-zero column is zero already, so every column of it
    is zeroed. Rescaling it instead would leave its other columns near 1e-100,
    and arithmetic on numbers that small is slow.
    """
    p = 1 if kind == "l1" else 2
    norms = torch.stack([f.norm(p=p, dim=0) for f in factors])
    dead = (norms == 0).any(0)
    norms = norms.clamp_min(1e-300)
    target = norms.log().mean(0).exp().masked_fill(dead, 0.0)
    return [f * (target / n) for f, n in zip(factors, norms)]


def fit(data, start, kind, strength, iters=ITERS):
    factors = [f.clone() for f in start]
    step = 1.0

    def smooth(fs):
        value = 0.5 * ((data - cp_tensor(fs)) ** 2).sum()
        if kind == "wd":
            value = value + strength * penalty(fs, "wd")
        return value

    for _ in range(iters):
        if not any(f.any() for f in factors):
            break  # every component is zero: the gradient is zero from here on
        live = [f.clone().requires_grad_(True) for f in factors]
        value = smooth(live)
        grads = torch.autograd.grad(value, live)
        value = value.detach()
        while True:  # backtracking on the smooth part
            trial = prox(
                [(f - step * g) for f, g in zip(factors, grads)], kind, step, strength
            )
            moves = [t - f for t, f in zip(trial, factors)]
            bound = (
                value
                + sum((g * m).sum() for g, m in zip(grads, moves))
                + sum((m**2).sum() for m in moves) / (2 * step)
            )
            if smooth(trial) <= bound + 1e-15:
                break
            step *= 0.5
            if not torch.isfinite(bound) or step < 1e-12:
                raise FloatingPointError("backtracking failed: the fit diverged")
        factors = trial
        step = min(step * 1.5, MAX_STEP)
        factors = rebalance(factors, kind)
    return factors


def main() -> None:
    kind = sys.argv[1]
    with (DATA / f"prune_{kind}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["kind", "strength", "seed", "alive", "error_to_data", "error_to_truth"]
        )
        for strength in [0.0] if kind == "none" else STRENGTHS:
            for seed in SEEDS:
                clean, noisy, start = make_problem(seed)
                factors = fit(noisy, start, kind, strength)
                fitted = cp_tensor(factors)
                writer.writerow(
                    [
                        kind,
                        strength,
                        seed,
                        int((component_weights(factors) > ALIVE).sum()),
                        f"{(noisy - fitted).norm().item():.5f}",
                        f"{(clean - fitted).norm().item():.5f}",
                    ]
                )
                fh.flush()
                print(kind, strength, seed, flush=True)


if __name__ == "__main__":
    main()
