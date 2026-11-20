"""Known-answer tests for the derived analyses the posts quote."""

import numpy as np
from umaplab.derived import snn_neighbours, valley_ratio


def test_valley_ratio_unimodal_is_near_one():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(2000, 2))
    labels = (Z[:, 0] > 0).astype(int)  # split one Gaussian down the middle
    assert valley_ratio(Z, labels) > 0.9


def test_valley_ratio_separated_blobs_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1000, 2))
    b = rng.normal(size=(1000, 2)) + [12.0, 0.0]
    Z = np.vstack([a, b])
    labels = np.repeat([0, 1], 1000)
    assert valley_ratio(Z, labels) < 0.05


def test_valley_ratio_matches_analytic_dip():
    """Two unit Gaussians 4 sd apart: density at the midpoint is 0.27 of the peaks."""
    rng = np.random.default_rng(1)
    n = 20000
    labels = np.repeat([0, 1], n // 2)
    x = rng.normal(size=n) + np.where(labels == 0, -2.0, 2.0)
    Z = np.column_stack([x, rng.normal(size=n)])
    # KDE smoothing makes the dip shallower than the exact 0.27, never deeper.
    assert 0.25 < valley_ratio(Z, labels) < 0.45


def test_snn_neighbours_shape_distinct_no_self():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    nb = snn_neighbours(X, K=15, k=10)
    assert nb.shape == (300, 10)
    for i, row in enumerate(nb):
        assert i not in row
        assert len(set(row)) == 10


def test_snn_neighbours_on_a_line_are_the_true_neighbours():
    """On evenly spaced points the shared-neighbour ranking recovers the nearest."""
    X = np.arange(200, dtype=float)[:, None]
    nb = snn_neighbours(X, K=10, k=4)
    i = 100
    assert set(nb[i]) == {98, 99, 101, 102}
