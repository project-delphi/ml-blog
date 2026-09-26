r"""Fetch one real convolution kernel and cache it for the render.

Writes posts/cp-or-tucker-in-practice/data/, which .gitignore carves out of the
repo-wide `data/` rule so the cache is committed and a clone can rebuild the post
without network access or torchvision's weight cache.

The kernel is ResNet-18's layer3.1.conv2.weight from torchvision's IMAGENET1K_V1
weights: shape 256 x 256 x 3 x 3 (out, in, height, width), stride 1, padding 1.

Usage:
    .venv-cp-tucker/bin/python posts/cp-or-tucker-in-practice/src/fetch_kernel.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import torchvision
from torchvision.models import ResNet18_Weights

OUT = Path(__file__).resolve().parent.parent / "data"
LAYER = "layer3.1.conv2"


def main() -> None:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    conv = model.get_submodule(LAYER)
    kernel = conv.weight.detach().numpy().astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "resnet18_layer3_1_conv2.npy"
    np.save(path, kernel)

    manifest = {
        "source": "torchvision ResNet18_Weights.IMAGENET1K_V1",
        "url": weights.url,
        "torchvision": torchvision.__version__,
        "layer": LAYER,
        "shape": list(kernel.shape),
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fetched": date.today().isoformat(),
    }
    (OUT / "resnet18_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
