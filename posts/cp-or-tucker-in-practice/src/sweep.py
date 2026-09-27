r"""CP versus Tucker-2 on one real conv kernel, at matched parameter budgets.

Reads data/resnet18_layer3_1_conv2.npy (written by fetch_kernel.py) and writes
one CSV per run into data/, which the post reads at render time. CP-ALS at the
larger ranks takes minutes per fit, too slow to repeat inside a render.

Budgets are compression ratios of the dense 256*256*3*3 = 589,824 weights:

- CP at rank R stores R*(256 + 256 + 3 + 3) = 518 R numbers.
- Tucker-2 at ranks (R_o, R_i) stores 256 R_o + 256 R_i + 9 R_o R_i numbers.

Usage (one CP seed per process, so seeds can run side by side):
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/sweep.py cp --seed 0
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/sweep.py tucker2
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker

DATA = Path(__file__).resolve().parent.parent / "data"
RATIOS = [64, 32, 16, 8, 4, 2]
CP_ITERS = 1000


def load_kernel() -> np.ndarray:
    return np.load(DATA / "resnet18_layer3_1_conv2.npy").astype(np.float64)


def cp_params(rank: int, shape: tuple[int, ...]) -> int:
    return rank * sum(shape)


def tucker2_params(r_out: int, r_in: int, shape: tuple[int, ...]) -> int:
    """Core plus factors; a channel mode kept whole folds its factor into the core."""
    o, i, h, w = shape
    factors = (o * r_out if r_out < o else 0) + (i * r_in if r_in < i else 0)
    return factors + h * w * r_out * r_in


def cp_rank_for(budget: float, shape: tuple[int, ...]) -> int:
    return round(budget / sum(shape))


def tucker2_pairs_for(
    budget: float, shape: tuple[int, ...], r_outs: range
) -> list[tuple[int, int]]:
    """For each R_o, the largest R_i whose parameter count stays within budget."""
    pairs = []
    for r_out in r_outs:
        # The largest R_i within budget; storage rises with R_i, so scan down.
        fits = [
            r
            for r in range(shape[1], 0, -1)
            if tucker2_params(r_out, r, shape) <= budget
        ]
        if fits:
            pairs.append((r_out, fits[0]))
    return pairs


def run_cp(seed: int) -> None:
    kernel = load_kernel()
    norm = np.linalg.norm(kernel)
    out = DATA / f"sweep_cp_seed{seed}.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "seed",
                "ratio",
                "rank",
                "params",
                "rel_error",
                "recorded_errors",
                "tail_improvement",
                "max_component_norm",
                "seconds",
            ]
        )
        for ratio in RATIOS:
            rank = cp_rank_for(kernel.size / ratio, kernel.shape)
            start = time.time()
            cp, errors = parafac(
                kernel,
                rank=rank,
                init="random",
                random_state=seed,
                n_iter_max=CP_ITERS,
                tol=1e-10,
                linesearch=True,
                return_errors=True,
            )
            rel = np.linalg.norm(kernel - tl.cp_to_tensor(cp)) / norm
            # Degeneracy tell: a rank-one term far larger than the kernel it helps
            # approximate is cancelling against another one.
            comp = np.prod([np.linalg.norm(f, axis=0) for f in cp.factors], axis=0)
            comp = comp * np.asarray(cp.weights)
            # With line search, TensorLy records an error only on iterations that
            # are not line-search steps, so len(errors) is about half of CP_ITERS.
            tail = len(errors) // 10
            writer.writerow(
                [
                    seed,
                    ratio,
                    rank,
                    cp_params(rank, kernel.shape),
                    f"{rel:.6f}",
                    len(errors),
                    f"{errors[-tail - 1] - errors[-1]:.3e}",
                    f"{comp.max() / norm:.4f}",
                    f"{time.time() - start:.1f}",
                ]
            )
            fh.flush()
            print(ratio, rank, round(rel, 4), round(time.time() - start, 1), flush=True)


def run_tucker2() -> None:
    """Best Tucker-2 pair within CP's parameter count at each ratio.

    The budget is CP's own count, so Tucker-2 never gets more numbers than CP. A
    coarse pass over R_o in steps of 4 finds the neighbourhood; a second pass
    tries every R_o within 3 of the coarse winner.
    """
    kernel = load_kernel()
    norm = np.linalg.norm(kernel)

    def rel_error(r_out: int, r_in: int) -> float:
        (core, factors), _ = partial_tucker(
            kernel,
            rank=[r_out, r_in],
            modes=[0, 1],
            init="svd",
            n_iter_max=100,
            tol=1e-7,
        )
        rec = tl.tenalg.multi_mode_dot(core, factors, modes=[0, 1])
        return float(np.linalg.norm(kernel - rec) / norm)

    out = DATA / "sweep_tucker2.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ratio", "budget", "r_out", "r_in", "params", "rel_error"])
        for ratio in RATIOS:
            budget = cp_params(
                cp_rank_for(kernel.size / ratio, kernel.shape), kernel.shape
            )
            tried = {}
            for r_out, r_in in tucker2_pairs_for(
                budget, kernel.shape, range(4, 257, 4)
            ):
                tried[(r_out, r_in)] = rel_error(r_out, r_in)
            best = min(tried, key=tried.get)[0]
            near = range(max(1, best - 3), min(256, best + 3) + 1)
            for r_out, r_in in tucker2_pairs_for(budget, kernel.shape, near):
                if (r_out, r_in) not in tried:
                    tried[(r_out, r_in)] = rel_error(r_out, r_in)
            for (r_out, r_in), err in sorted(tried.items()):
                writer.writerow(
                    [
                        ratio,
                        budget,
                        r_out,
                        r_in,
                        tucker2_params(r_out, r_in, kernel.shape),
                        f"{err:.6f}",
                    ]
                )
            fh.flush()
            print(ratio, "done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["cp", "tucker2"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.method == "cp":
        run_cp(args.seed)
    else:
        run_tucker2()


if __name__ == "__main__":
    main()
