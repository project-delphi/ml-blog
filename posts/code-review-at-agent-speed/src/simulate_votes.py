r"""Simulate a k-of-n verifier vote on candidate findings, and print every figure
section "Vote thresholds" quotes, apart from the precision and accuracy columns
of its verifier table, which copy Du et al.'s Table 2.

The inputs are measured, not assumed. Du et al. (2026,
arXiv:2601.18844) took 433 alarms from Tencent's in-house static analyzer,
105 of them real bugs and 328 false positives, and asked four LLMs to judge
each one. Table 2 gives every model's recall (its true positive rate) and its
FPR_R = TN / (TN + FP), which is specificity, so the false positive rate is
1 - FPR_R. Each cell is already a majority of three runs of one model.

What the paper does not publish is which alarms each model got wrong, so how
far the four models' mistakes overlap is the one free parameter, rho. A
finding carries one latent score shared by every verifier; a verifier's
judgement is that score plus its own noise, and rho is the share of the
judgement's variance that is shared. Each verifier's threshold is set so its
marginal rates stay at Table 2's values whatever rho is: rho changes which
findings a verifier gets wrong, not how many.

Usage (from repo root):

    uv run --no-project --with numpy python \
        posts/code-review-at-agent-speed/src/simulate_votes.py

`--no-project` matters: a plain `uv run` here deletes and re-syncs the repo venv.
"""

from __future__ import annotations

from math import prod
from statistics import NormalDist

import numpy as np

N_REAL, N_FALSE = 105, 328  # Du et al., Table 1
RUNS = 2000  # simulated review sets of 433 findings each
SEED = 20261017
RHOS = (0.0, 0.5)

# Du et al., Table 2: (true positive rate, 1 - FPR_R).
PANELS = {
    "prompt only (few-shot)": {
        "GPT-4o": (0.64, 1 - 0.43),
        "Claude-Opus-4": (0.74, 1 - 0.72),
        "Qwen-3-Coder": (0.59, 1 - 0.72),
        "DeepSeek-R1": (0.83, 1 - 0.65),
    },
    "agent with path constraints (LLM4PFA)": {
        "GPT-4o": (0.75, 1 - 0.98),
        "Claude-Opus-4": (0.88, 1 - 0.94),
        "Qwen-3-Coder": (0.79, 1 - 0.98),
        "DeepSeek-R1": (0.86, 1 - 0.95),
    },
}

Z = NormalDist()


def votes(rng, rates, n, rho):
    """Votes of every verifier on n findings of one class: shape (n, verifiers)."""
    shared = rng.standard_normal((n, 1))
    own = rng.standard_normal((n, len(rates)))
    score = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own
    cut = np.array([Z.inv_cdf(1 - r) for r in rates])
    return score > cut


def simulate(panel, rho, rng):
    """Precision, recall, bugs caught and false alarms at every threshold k."""
    tpr = [t for t, _ in panel.values()]
    fpr = [f for _, f in panel.values()]
    n = len(panel)
    out = np.empty((RUNS, n, 4))
    for run in range(RUNS):
        real = votes(rng, tpr, N_REAL, rho).sum(axis=1)
        false = votes(rng, fpr, N_FALSE, rho).sum(axis=1)
        for k in range(1, n + 1):
            tp, fp = (real >= k).sum(), (false >= k).sum()
            # A set that posts nothing has no precision, not a precision of 0.
            prec = tp / (tp + fp) if tp + fp else np.nan
            out[run, k - 1] = prec, tp / N_REAL, tp, fp
    return out


def outcomes(tp, fp):
    """Caught, missed, false alarms, dropped non-bugs, and accuracy."""
    acc = (tp + N_FALSE - fp) / (N_REAL + N_FALSE)
    return (
        f"caught {tp:5.1f}  missed {N_REAL - tp:5.1f}  false alarms {fp:5.1f}"
        f"  dropped {N_FALSE - fp:5.1f}  accuracy {pct(acc)}"
    )


def exact(rates, k):
    """P(at least k votes) for independent verifiers with unequal rates."""
    n = len(rates)
    total = 0.0
    for mask in range(1 << n):
        on = [i for i in range(n) if mask >> i & 1]
        if len(on) >= k:
            total += prod(rates[i] if i in on else 1 - rates[i] for i in range(n))
    return total


def pct(x):
    # A rounded "100%" or "0%" would claim the same outcome in every set.
    if 0.995 <= x < 1:
        return ">99%"
    if 0 < x < 0.005:
        return "<1%"
    return f"{100 * x:.0f}%"


def main():
    rng = np.random.default_rng(SEED)
    print(f"base rate {N_REAL}/{N_REAL + N_FALSE} = {pct(N_REAL / (N_REAL + N_FALSE))}")
    print(f"  post everything  {outcomes(N_REAL, N_FALSE)}")
    print(f"  drop everything  {outcomes(0, 0)}")
    for name, panel in PANELS.items():
        tpr = [t for t, _ in panel.values()]
        fpr = [f for _, f in panel.values()]
        print(f"\n== {name} ==")
        for model, (t, f) in panel.items():
            prec = t * N_REAL / (t * N_REAL + f * N_FALSE)
            print(
                f"  alone  {model:14s} recall {pct(t)}  FPR {pct(f)}  precision {pct(prec)}"
            )
            print(f"         {outcomes(t * N_REAL, f * N_FALSE)}")
        for rho in RHOS:
            sims = simulate(panel, rho, rng)
            mean = np.nanmean(sims, axis=0)
            lo, hi = np.nanpercentile(sims, [5, 95], axis=0)
            fooled = votes(rng, fpr, 200_000, rho).all(axis=1).mean()
            print(
                f"  rho={rho}  (non-bugs every verifier calls real: {100 * fooled:.1f}%)"
            )
            for k in range(1, len(panel) + 1):
                (p, r, tp, fp) = mean[k - 1]
                (pl, rl), (ph, rh) = lo[k - 1, :2], hi[k - 1, :2]
                line = (
                    f"    {k} of {len(panel)}: precision {pct(p)} [{pct(pl)}-{pct(ph)}]"
                    f"  recall {pct(r)} [{pct(rl)}-{pct(rh)}]"
                    f"\n           {outcomes(tp, fp)}"
                )
                if rho == 0.0:
                    # Check the simulation against the closed form.
                    et, ef = exact(tpr, k), exact(fpr, k)
                    ep = et * N_REAL / (et * N_REAL + ef * N_FALSE)
                    assert abs(r - et) < 0.01, (k, r, et)
                    # The mean of a ratio sits a little off the ratio of means.
                    assert abs(p - ep) < 0.02, (k, p, ep)
                    line += f"   exact: precision {pct(ep)} recall {pct(et)}"
                print(line)


if __name__ == "__main__":
    main()
