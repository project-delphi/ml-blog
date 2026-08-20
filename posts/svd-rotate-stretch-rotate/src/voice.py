"""A synthetic sustained vowel, its trajectory matrix, and SVD denoising.

The signal is synthetic on purpose: subspace denoising is only checkable
against a clean reference, and no recording comes with one. A sum of harmonics
in white noise is the textbook stand-in for a sustained vowel -- glottal pulses
excite a harmonic stack, and the vocal tract shapes their amplitudes -- which is
the case where the trajectory matrix has an exactly known rank to aim at.
"""

from __future__ import annotations

import numpy as np

FS = 8000  # sample rate, Hz -- narrowband telephony
F0 = 120.0  # glottal fundamental, Hz -- a low-pitched voice
N_HARM = 5  # harmonics retained; each one costs the Hankel matrix rank 2
DURATION = 0.125  # seconds
FRAME = 200  # trajectory window, samples == 25 ms, the usual speech frame
SEED = 0


def vowel(
    fs: int = FS,
    f0: float = F0,
    n_harm: int = N_HARM,
    duration: float = DURATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, clean) for a unit-RMS harmonic stack with 1/k amplitudes."""
    t = np.arange(int(fs * duration)) / fs
    clean = sum(
        (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + 0.7 * k)
        for k in range(1, n_harm + 1)
    )
    return t, clean / np.sqrt(np.mean(clean**2))


def corrupt(
    clean: np.ndarray,
    snr_db: float = 0.0,
    seed: int = SEED,
) -> np.ndarray:
    """Add white Gaussian noise at a chosen signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(clean.size)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2)) * 10 ** (-snr_db / 20)
    return clean + noise


def trajectory(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    """Stack every length-`frame` window of `x` as a column: a Hankel matrix.

    Constant along each anti-diagonal, so one sample appears in many entries.
    That redundancy is what gives the matrix a rank far below its dimensions.
    """
    windows = np.lib.stride_tricks.sliding_window_view(x, frame)
    return windows[: x.size - frame + 1].T


def untrajectory(H: np.ndarray) -> np.ndarray:
    """Average a matrix back to a signal along its anti-diagonals.

    A truncated Hankel matrix is no longer exactly Hankel, so the copies of a
    sample disagree; averaging them is the least-squares Hankel matrix nearest
    to it, and turns the approximation back into something playable.
    """
    frame, k = H.shape
    total = np.zeros(frame + k - 1)
    count = np.zeros_like(total)
    for i in range(frame):
        total[i : i + k] += H[i]
        count[i : i + k] += 1
    return total / count


def denoise(noisy: np.ndarray, rank: int, frame: int = FRAME) -> np.ndarray:
    """Truncate the trajectory matrix to `rank` and fold it back to a signal."""
    U, s, Vt = np.linalg.svd(trajectory(noisy, frame), full_matrices=False)
    return untrajectory((U[:, :rank] * s[:rank]) @ Vt[:rank])


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio of `estimate` against `clean`, in decibels."""
    return 10 * np.log10(np.sum(clean**2) / np.sum((estimate - clean) ** 2))
