"""MNIST: the cheap path and the expensive path.

**Cheap** — train once, cache the per-example correctness vector and predicted
probabilities for the 10,000 test images.  Every evaluation metric is then a
functional of the empirical distribution of test examples, and bootstrapping it
costs nothing.  Accuracy in particular is a mean of indicators, so its Bayesian
posterior is exactly ``Beta(k, n - k)`` with no simulation at all.

**Expensive** — the weighted likelihood bootstrap: draw
``w ~ Dirichlet(1, ..., 1)`` over the training points and minimise
``sum_i w_i * loss(x_i, theta)``.  One line different from weighted training,
and it costs ``B`` times a full fit.

The backend is torch; there is no silent fallback.  If torch is missing the
import fails loudly, because a post that quietly swaps in a different model and
reports the numbers as if nothing happened is worse than one that does not
build.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

Floats = NDArray[np.float64]

_HERE: Final[Path] = Path(__file__).resolve().parent.parent
RAW: Final[Path] = _HERE / "data" / "raw"
CACHE: Final[Path] = _HERE / "data" / "cache"

SEED: Final[int] = 20260726
HIDDEN: Final[int] = 128
EPOCHS: Final[int] = 3
BATCH: Final[int] = 256
LR: Final[float] = 1e-3


def _device() -> torch.device:
    """Pick the fastest available device, preferring Apple's MPS backend."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_net() -> nn.Module:
    """A deliberately small MLP: 784 -> 128 -> 10.

    Small on purpose.  The point of the expensive path is the *cost multiplier*,
    and that argument is clearest when a single fit is cheap enough that
    ``B = 20`` of them is merely annoying rather than impossible.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 10),
    )


def _load_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download MNIST if needed and return train/test images and labels."""
    RAW.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(RAW / "mnist", train=True, download=True, transform=tf)
    test = datasets.MNIST(RAW / "mnist", train=False, download=True, transform=tf)

    def stack(ds: datasets.MNIST) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(ds, batch_size=len(ds))
        images, labels = next(iter(loader))
        return images, labels

    xtr, ytr = stack(train)
    xte, yte = stack(test)
    return xtr, ytr, xte, yte


@dataclass(frozen=True)
class EvalCache:
    """Per-example test-set results from a single trained model.

    Attributes:
        correct: ``(n_test,)`` boolean correctness indicator.
        probs: ``(n_test, 10)`` predicted class probabilities.
        y_true: ``(n_test,)`` true labels.
        y_pred: ``(n_test,)`` predicted labels.
        train_seconds: Wall-clock cost of the single fit.
        backend: Which path actually ran, for the post to state.
    """

    correct: NDArray[np.bool_]
    probs: NDArray[np.float32]
    y_true: NDArray[np.int64]
    y_pred: NDArray[np.int64]
    train_seconds: float
    backend: str

    @property
    def n(self) -> int:
        """Number of test examples."""
        return int(self.correct.size)

    @property
    def k(self) -> int:
        """Number of correctly classified test examples."""
        return int(self.correct.sum())

    @property
    def accuracy(self) -> float:
        """Plug-in test accuracy, ``k / n``."""
        return self.k / self.n

    def confusion(self) -> NDArray[np.int64]:
        """``(10, 10)`` confusion matrix, rows true and columns predicted."""
        m = np.zeros((10, 10), dtype=np.int64)
        np.add.at(m, (self.y_true, self.y_pred), 1)
        return m


def _train_one(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    weights: torch.Tensor | None,
    device: torch.device,
    seed: int,
) -> nn.Module:
    """Fit the small MLP once, optionally under per-example weights.

    With ``weights=None`` this is ordinary training.  With weights it is the
    weighted likelihood bootstrap: the objective becomes ``sum_i w_i * l_i``
    rather than ``mean_i l_i``.  The weights are rescaled by ``n`` so that
    uniform weights reproduce the unweighted objective exactly and the learning
    rate stays comparable between the two paths.

    Args:
        xtr: ``(n, 1, 28, 28)`` training images.
        ytr: ``(n,)`` training labels.
        weights: Optional ``(n,)`` non-negative weights summing to one.
        device: Compute device.
        seed: Torch seed for initialisation and batch order.

    Returns:
        The trained network, in eval mode.
    """
    torch.manual_seed(seed)
    net = _make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    # reduction='none' is the whole trick: it exposes the per-example loss that
    # the weight vector multiplies.
    criterion = nn.CrossEntropyLoss(reduction="none")

    n = xtr.shape[0]
    w = torch.full((n,), 1.0 / n) if weights is None else weights
    dataset = TensorDataset(xtr, ytr, w * n)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    net.train()
    for _ in range(EPOCHS):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad(set_to_none=True)
            per_example = criterion(net(xb), yb)
            # The weighted likelihood bootstrap, in one line.
            loss = (per_example * wb).mean()
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def _predict(net: nn.Module, xte: torch.Tensor, device: torch.device) -> NDArray[np.float32]:
    """Return ``(n_test, 10)`` predicted probabilities."""
    out = []
    for start in range(0, xte.shape[0], 1024):
        batch = xte[start : start + 1024].to(device)
        out.append(torch.softmax(net(batch), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def train_and_cache(refresh: bool = False) -> EvalCache:
    """Train once and cache per-example test results.

    Args:
        refresh: Retrain even if the cache exists.

    Returns:
        The cached per-example results.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "mnist_eval.npz"
    if path.exists() and not refresh:
        z = np.load(path)
        return EvalCache(
            correct=z["correct"].astype(bool),
            probs=z["probs"],
            y_true=z["y_true"],
            y_pred=z["y_pred"],
            train_seconds=float(z["train_seconds"]),
            backend=str(z["backend"]),
        )

    device = _device()
    xtr, ytr, xte, yte = _load_tensors()
    start = time.perf_counter()
    net = _train_one(xtr, ytr, None, device, SEED)
    elapsed = time.perf_counter() - start

    probs = _predict(net, xte, device)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_true = yte.numpy().astype(np.int64)
    correct = y_pred == y_true
    backend = f"torch {torch.__version__} on {device.type}"

    np.savez_compressed(
        path,
        correct=correct,
        probs=probs,
        y_true=y_true,
        y_pred=y_pred,
        train_seconds=elapsed,
        backend=backend,
    )
    return EvalCache(correct, probs, y_true, y_pred, elapsed, backend)


def weighted_likelihood_bootstrap(B: int = 20, refresh: bool = False) -> dict[str, object]:
    """Refit the model ``B`` times under Dirichlet weights over the training set.

    This is the expensive path.  Each replicate is a full training run, so the
    cost is ``B`` times a fit — nothing for a mean, everything for a network.

    Args:
        B: Number of refits.
        refresh: Recompute even if cached.

    Returns:
        A mapping with per-replicate test accuracies, per-fit seconds, the
        single-fit baseline and the backend string.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "mnist_wlb.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    device = _device()
    xtr, ytr, xte, yte = _load_tensors()
    n = xtr.shape[0]
    rng = np.random.default_rng(SEED)
    y_true = yte.numpy().astype(np.int64)

    accuracies: list[float] = []
    seconds: list[float] = []
    for b in range(B):
        e = rng.exponential(1.0, size=n)
        w = torch.from_numpy((e / e.sum()).astype(np.float32))
        start = time.perf_counter()
        net = _train_one(xtr, ytr, w, device, SEED + 1 + b)
        seconds.append(time.perf_counter() - start)
        probs = _predict(net, xte, device)
        accuracies.append(float((probs.argmax(axis=1) == y_true).mean()))

    result: dict[str, object] = {
        "B": B,
        "accuracies": accuracies,
        "seconds": seconds,
        "seconds_per_fit": float(np.mean(seconds)),
        "total_seconds": float(np.sum(seconds)),
        "backend": f"torch {torch.__version__} on {device.type}",
        "n_train": int(n),
        "epochs": EPOCHS,
        "hidden": HIDDEN,
    }
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    cache = train_and_cache()
    print(f"backend        {cache.backend}")
    print(f"single fit     {cache.train_seconds:.1f}s")
    print(f"test accuracy  {cache.accuracy:.4f}  (k={cache.k}, n={cache.n})")
    wlb = weighted_likelihood_bootstrap()
    print(f"WLB B={wlb['B']}  {wlb['seconds_per_fit']:.1f}s/fit  "
          f"total {wlb['total_seconds']:.0f}s")
