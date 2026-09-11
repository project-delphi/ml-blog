"""A learned embedding: rung 3.

The first three rungs are all hand-built. Someone decided that ridge orientation
matters, that a Gabor bank is the way to measure it, that ridge endings and
bifurcations are the landmarks worth keeping. A metric-learning network is told
none of that. It is given pairs and a rule about distances -- two impressions of
one finger should land close together, impressions of different fingers far
apart -- and has to work out what to measure for itself.

The loss here is the batch-hard triplet loss (Hermans, Beyer and Leibe, 2017), a
practical form of the triplet loss of Schroff, Kalenichenko and Philbin (2015).
Each batch is built from several fingers with both impressions present. For every
image in the batch, the hardest positive is its own mate and the hardest negative
is the nearest image of a *different* finger; the loss pushes the second further
than the first by a margin. Training on the hardest cases in the batch, rather
than random ones, is what makes a small batch worth anything.

The honest caveat lives in the post, not the code: this network sees a few
hundred images. The systems that beat hand-built features on this problem see
millions of prints, and the gap is the point rather than an embarrassment.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT = 96  # images are resampled to this before the first convolution
EMBED_DIM = 128
MARGIN = 0.3


class Embedder(nn.Module):
    """A small convolutional network mapping a print to a unit vector.

    Four stride-2 blocks take a 96x96 print down to 6x6, and a global average
    pool throws away where things were, keeping only what was there. The output
    is L2-normalised, so comparing two prints is a dot product and the loss can
    talk about distances on a sphere.
    """

    def __init__(self, dim: int = EMBED_DIM, width: int = 32):
        super().__init__()
        channels = [1, width, width * 2, width * 4, width * 4]
        self.features = nn.Sequential(
            *[
                layer
                for i in range(4)
                for layer in (
                    nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels[i + 1], channels[i + 1], 3, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            ],
        )
        self.head = nn.Linear(channels[-1], dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).mean(dim=(2, 3))
        return F.normalize(self.head(h), dim=1)


def prepare(images: np.ndarray, size: int = INPUT) -> torch.Tensor:
    """Images to a normalised (N, 1, size, size) float tensor."""
    x = torch.from_numpy(images.astype(np.float32) / 255.0)[:, None]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (
        x.std(dim=(2, 3), keepdim=True) + 1e-6
    )


def augment(batch: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random rotation, shift, scale and occlusion.

    Every one of these stands for something the sensor does anyway: the finger
    lands at a different angle and position, presses harder or softer, and covers
    a different part of the platen. With two images per finger the network would
    otherwise memorise the pair rather than learn what makes them a pair.
    """
    n = len(batch)
    angle = (torch.rand(n, generator=generator) * 2 - 1) * (25 * np.pi / 180)
    scale = 1.0 + (torch.rand(n, generator=generator) * 2 - 1) * 0.08
    shift = (torch.rand(n, 2, generator=generator) * 2 - 1) * 0.12

    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, shift[:, 0]
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, shift[:, 1]

    grid = F.affine_grid(theta, batch.shape, align_corners=False)
    out = F.grid_sample(batch, grid, align_corners=False, padding_mode="zeros")

    # Blank a random block: a print that only caught part of the finger.
    size = out.shape[-1] // 3
    for i in range(n):
        if torch.rand(1, generator=generator).item() < 0.5:
            y, x = torch.randint(0, out.shape[-1] - size, (2,), generator=generator)
            out[i, :, y : y + size, x : x + size] = 0
    return out


def batch_hard_triplet(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = MARGIN,
):
    """Triplet loss against the hardest positive and negative in the batch."""
    distance = torch.cdist(embeddings, embeddings)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)

    hardest_positive = (distance * (same & ~eye)).max(dim=1).values
    hardest_negative = (distance + same * 1e6).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def batches(finger: np.ndarray, fingers_per_batch: int, rng: np.random.Generator):
    """Yield index arrays holding both impressions of `fingers_per_batch` fingers.

    The batch is capped at the number of fingers there are. Without the cap a
    training set smaller than one batch yields no batches at all, and the epoch
    loop then completes having done nothing -- which reads, from the outside,
    exactly like a network that trained and learned nothing.
    """
    unique = np.unique(finger)
    fingers_per_batch = max(2, min(fingers_per_batch, len(unique)))
    order = rng.permutation(unique)
    for start in range(0, len(order) - fingers_per_batch + 1, fingers_per_batch):
        chosen = order[start : start + fingers_per_batch]
        yield np.concatenate([np.where(finger == f)[0] for f in chosen])


def train(
    images: np.ndarray,
    finger: np.ndarray,
    epochs: int = 60,
    fingers_per_batch: int = 24,
    learning_rate: float = 2e-3,
    seed: int = 7,
):
    """Fit the embedder on the training fingers. Returns (model, loss history)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    x = prepare(images)
    y = torch.from_numpy(finger.astype(np.int64))

    model = Embedder()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    history = []
    model.train()
    for _ in range(epochs):
        losses = []
        for idx in batches(finger, fingers_per_batch, rng):
            batch = augment(x[idx], generator)
            loss = batch_hard_triplet(model(batch), y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        schedule.step()
        history.append(float(np.mean(losses)) if losses else float("nan"))
    return model, history


@torch.no_grad()
def encode(model: nn.Module, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Embed a set of prints. Rows are unit vectors, so similarity is a dot product."""
    model.eval()
    x = prepare(images)
    out = [model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()
