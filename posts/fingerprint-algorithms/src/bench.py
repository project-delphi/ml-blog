"""One scoring harness, shared by every rung.

Each rung produces the same object: a score matrix with one row per probe and
one column per enrolled gallery finger, higher meaning more alike. Everything
reported about a rung is read off that matrix, so the rungs are comparable even
though a pixel correlation, a Gabor descriptor, a minutiae point set and a
learned vector have nothing in common internally.

Two questions get asked of it:

* **Identification.** Sort each row. Where does the probe's true mate land? The
  rank-1 rate is how often it lands first, and the CMC curve is how often it
  lands in the top k as k grows. This is the search problem -- one finger against
  a database.
* **Verification.** Compare the score a probe gives its own mate against the
  scores it gives everyone else. The equal error rate is where the two
  distributions cross: the threshold at which as many genuine pairs are rejected
  as impostor pairs are accepted. This is the phone-unlock problem -- one finger
  against one claim.

A rung can be respectable at one and poor at the other, so the post reports both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Result:
    """Everything the post says about one rung, computed from one score matrix."""

    name: str
    ranks: np.ndarray  # (probes,) 0-based rank of the true mate
    genuine: np.ndarray  # (probes,) score against the true mate
    impostor: np.ndarray  # (probes * (gallery-1),) scores against everyone else
    gallery_size: int
    seconds: float

    @property
    def rank1(self) -> float:
        return float((self.ranks == 0).mean())

    @property
    def rank5(self) -> float:
        return float((self.ranks < 5).mean())

    @property
    def median_rank(self) -> float:
        return float(np.median(self.ranks) + 1)

    def cmc(self, k_max: int | None = None) -> np.ndarray:
        """Cumulative match characteristic: P(true mate within the top k), k = 1..k_max."""
        k_max = k_max or self.gallery_size
        return np.array([(self.ranks < k).mean() for k in range(1, k_max + 1)])

    @property
    def eer(self) -> float:
        """Equal error rate, found by sweeping the threshold over observed scores."""
        thresholds = np.unique(np.concatenate([self.genuine, self.impostor]))
        far = np.array([(self.impostor >= t).mean() for t in thresholds])
        frr = np.array([(self.genuine < t).mean() for t in thresholds])
        return float(np.min(np.maximum(far, frr)))

    @property
    def separation(self) -> float:
        """How many impostor standard deviations separate the two means (d-prime)."""
        pooled = np.sqrt(0.5 * (self.genuine.var() + self.impostor.var()))
        return float((self.genuine.mean() - self.impostor.mean()) / (pooled + 1e-12))

    def row(self) -> dict:
        return {
            "method": self.name,
            "rank-1": self.rank1,
            "rank-5": self.rank5,
            "median rank": self.median_rank,
            "EER": self.eer,
            "d'": self.separation,
            "seconds": self.seconds,
        }


def score(
    name: str,
    matrix: np.ndarray,
    truth: np.ndarray,
    seconds: float = 0.0,
) -> Result:
    """Turn a probe-by-gallery score matrix into a `Result`.

    `truth[i]` is the column holding probe `i`'s true mate.
    """
    matrix = np.asarray(matrix, float)
    n_probes, n_gallery = matrix.shape
    rows = np.arange(n_probes)

    genuine = matrix[rows, truth]
    # Rank counts every gallery entry that scores at least as well as the true
    # mate, minus the mate itself. Counting only strictly-better entries would
    # give a matcher that returns the same score for everything a rank of zero on
    # every probe -- a perfect result for saying nothing. Ties go against the
    # matcher instead, so silence scores last.
    ranks = (matrix >= genuine[:, None]).sum(1) - 1

    mask = np.ones_like(matrix, bool)
    mask[rows, truth] = False
    impostor = matrix[mask]

    return Result(name, ranks, genuine, impostor, n_gallery, seconds)


def table(results: list[Result]) -> str:
    """Fixed-width comparison table, one row per rung."""
    head = f"{'method':<34}{'rank-1':>8}{'rank-5':>8}{'med':>6}{'EER':>8}{'d-prime':>9}{'s':>7}"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.name:<34}{r.rank1:>8.3f}{r.rank5:>8.3f}{r.median_rank:>6.0f}"
            f"{r.eer:>8.3f}{r.separation:>9.2f}{r.seconds:>7.0f}",
        )
    return "\n".join(lines)
