"""The black box: a small CNN on MNIST, trained once and cached.

Everything downstream — every attribution map in Part I — is computed against
the single checkpoint this module writes. Training is on CPU with a fixed seed
rather than on MPS: the accelerator is roughly four times faster here, but its
kernels are not bit-reproducible across releases, and the post names a specific
misclassified test digit in its prose. A running example that silently changes
identity between renders is worse than a slower build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from house import SEED
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

# MNIST's standard normalisation constants (mean and sd of the training set).
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _block(cin: int, cout: int) -> nn.Sequential:
    """Two padded 3x3 convolutions with batch norm, then a halving max-pool.

    Args:
        cin: Input channels.
        cout: Output channels.

    Returns:
        The block as a Sequential module.
    """
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A convolutional trunk, global average pooling, and one linear layer.

    The architecture is chosen for Part I rather than for the leaderboard. The
    last convolution keeps a 7x7 spatial grid, which is the grid Grad-CAM
    localises over, and the head is global average pooling followed by a single
    linear layer — the shape Grad-CAM was derived against, and the shape every
    ResNet-style classifier ends in.

    That choice is load-bearing, and it is not free. Global average pooling
    discards *where* a channel fired, so the trunk has to be deep enough that
    the channels themselves encode parts; a three-convolution version of this
    network with the same head tops out near 96%. The alternative — flatten the
    7x7 map and let the head weight each cell separately — trains faster and
    scores higher, and breaks Grad-CAM outright: the logit then reaches each
    cell through its own weight, so the mean gradient Grad-CAM uses as a channel
    weight averages over cells that disagree. Measured on this data with a
    flatten head, all 49 cells of the pre-ReLU map came out negative and
    Grad-CAM returned an empty image.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = _block(1, 32)  # 28 -> 14
        self.block2 = _block(32, 64)  # 14 -> 7
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # A module rather than a functional call, purely so Grad-CAM can attach a
        # hook to the exact tensor it needs. It holds no parameters.
        self.act3 = nn.ReLU()
        self.fc = nn.Linear(128, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last convolutional activations, shape (N, 128, 7, 7).

        Args:
            x: Normalised input batch of shape (N, 1, 28, 28).

        Returns:
            Post-ReLU activations of the final convolution.
        """
        return self.act3(self.bn3(self.conv3(self.block2(self.block1(x)))))

    def head(self, a: torch.Tensor) -> torch.Tensor:
        """Map final convolutional activations to logits by GAP then linear.

        Args:
            a: Activations of shape (N, 128, 7, 7).

        Returns:
            Logits of shape (N, 10).
        """
        return self.fc(a.mean(dim=(2, 3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full network.

        Args:
            x: Normalised input batch of shape (N, 1, 28, 28).

        Returns:
            Logits of shape (N, 10).
        """
        return self.head(self.features(x))


def normalise(x01: torch.Tensor) -> torch.Tensor:
    """Convert images in [0, 1] to the network's normalised input scale.

    Attribution is always computed in [0, 1] pixel space, because that is the
    space the reader sees; the network is fed through this function. Keeping the
    two separate is what makes a black *baseline* mean an actually black image.

    Args:
        x01: Images in [0, 1], any shape.

    Returns:
        Normalised images of the same shape.
    """
    return (x01 - MNIST_MEAN) / MNIST_STD


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST as [0, 1] tensors, downloading on first use.

    Returns:
        Tuple of (train images, train labels, test images, test labels), with
        images of shape (N, 1, 28, 28) in [0, 1].
    """
    RAW.mkdir(parents=True, exist_ok=True)
    tf = transforms.ToTensor()
    train = datasets.MNIST(RAW / "mnist", train=True, download=True, transform=tf)
    test = datasets.MNIST(RAW / "mnist", train=False, download=True, transform=tf)

    def stack(ds) -> tuple[torch.Tensor, torch.Tensor]:
        x = ds.data.unsqueeze(1).float() / 255.0
        y = ds.targets.clone()
        return x, y

    xtr, ytr = stack(train)
    xte, yte = stack(test)
    return xtr, ytr, xte, yte


def train_model(
    xtr: torch.Tensor, ytr: torch.Tensor, epochs: int = 4, batch: int = 128
) -> SmallCNN:
    """Train the CNN from scratch under a fixed seed.

    Args:
        xtr: Training images in [0, 1], shape (N, 1, 28, 28).
        ytr: Training labels, shape (N,).
        epochs: Number of passes over the training set.
        batch: Minibatch size.

    Returns:
        The trained model in eval mode.
    """
    torch.manual_seed(SEED)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xtr.shape[0]
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(normalise(xtr[idx])), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def logits_for(model: SmallCNN, x: torch.Tensor, batch: int = 1000) -> np.ndarray:
    """Compute logits for a large batch of [0, 1] images.

    Args:
        model: Trained network.
        x: Images in [0, 1], shape (N, 1, 28, 28).
        batch: Chunk size for the forward passes.

    Returns:
        Array of logits, shape (N, 10).
    """
    out = [model(normalise(x[i : i + batch])).numpy() for i in range(0, x.shape[0], batch)]
    return np.concatenate(out, axis=0)


@dataclass(frozen=True)
class Trained:
    """The cached artefacts every figure in Part I is computed from."""

    model: SmallCNN
    xte: torch.Tensor
    yte: torch.Tensor
    logits: np.ndarray

    @property
    def pred(self) -> np.ndarray:
        """Predicted class per test example."""
        return self.logits.argmax(1)

    @property
    def prob(self) -> np.ndarray:
        """Softmax probabilities per test example, shape (N, 10)."""
        z = self.logits - self.logits.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    @property
    def accuracy(self) -> float:
        """Test-set accuracy."""
        return float((self.pred == self.yte.numpy()).mean())


def get_trained(retrain: bool = False) -> Trained:
    """Load the cached model and test-set logits, training once if absent.

    Args:
        retrain: Force a fresh training run even if a cache exists.

    Returns:
        The model, the test set, and its logits.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    ckpt = CACHE / "cnn.pt"
    xtr, ytr, xte, yte = load_mnist()
    model = SmallCNN()
    if ckpt.exists() and not retrain:
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        model.eval()
    else:
        model = train_model(xtr, ytr)
        torch.save(model.state_dict(), ckpt)
    logits = logits_for(model, xte)
    return Trained(model=model, xte=xte, yte=yte, logits=logits)


def pick_examples(t: Trained) -> dict[str, int]:
    """Select the two running examples used throughout Part I.

    The headline example is the test digit the model gets *wrong* with the
    highest confidence — an explanation is most interesting where the model is
    most sure and most mistaken. The control is a correctly classified digit of
    the same true class, so the two maps are comparable.

    Args:
        t: The trained bundle.

    Returns:
        Mapping with keys ``"wrong"`` and ``"right"`` giving test-set indices.
    """
    y = t.yte.numpy()
    conf = t.prob.max(1)
    wrong_mask = t.pred != y
    wrong = int(np.flatnonzero(wrong_mask)[np.argmax(conf[wrong_mask])])

    same_class = (y == y[wrong]) & ~wrong_mask
    candidates = np.flatnonzero(same_class)
    right = int(candidates[np.argmax(conf[candidates])])
    return {"wrong": wrong, "right": right}


if __name__ == "__main__":
    import time

    start = time.time()
    trained = get_trained()
    idx = pick_examples(trained)
    y = trained.yte.numpy()
    print(f"test accuracy {trained.accuracy:.4f}  ({time.time() - start:.1f}s)")
    for key, i in idx.items():
        print(
            f"{key:>5}: index {i}, true {y[i]}, predicted {trained.pred[i]}, "
            f"p={trained.prob[i].max():.4f}"
        )
