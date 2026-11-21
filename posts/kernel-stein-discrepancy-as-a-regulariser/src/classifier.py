"""A small CIFAR-10 classifier, trained once and then frozen, whose penultimate features
score decoded images: the Frechet distance between two image sets is computed in its
256-dimensional feature space (a Frechet *classifier* distance; FID does the same with
an ImageNet Inception network, which this toy study does not download).

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/classifier.py

Writes results/classifier.pt and results/classifier.json (test accuracy, and the feature
mean and covariance of the 10,000 test images, the reference set for every distance).
"""

import json
import sys
import time
from pathlib import Path

import common
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
    )  # fmt: skip


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            block(3, 64), nn.MaxPool2d(2),
            block(64, 128), nn.MaxPool2d(2),
            block(128, 256), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )  # fmt: skip
        self.fc = nn.Linear(256, 10)

    def features(self, x01):
        """x01: images in [0, 1]. Returns (256-d features, logits)."""
        f = self.body(common.normalise(x01))
        return f, self.fc(f)

    def forward(self, x01):
        return self.features(x01)[1]


def load_classifier(dev):
    net = Net().to(dev)
    net.load_state_dict(torch.load(RES / "classifier.pt", map_location=dev))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def crop_flip(x, gen):
    n = x.shape[0]
    pad = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), generator=gen, device=x.device)
    j = torch.randint(0, 9, (n,), generator=gen, device=x.device)
    rows = (i[:, None] + torch.arange(32, device=x.device))[:, None, :, None].expand(
        n, 3, 32, 40
    )
    out = pad.gather(2, rows)
    cols = (j[:, None] + torch.arange(32, device=x.device))[:, None, None, :].expand(
        n, 3, 32, 32
    )
    out = out.gather(3, cols)
    flip = torch.rand(n, generator=gen, device=x.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


def main(epochs=30, batch=256):
    dev = common.device()
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)
    xtr, ytr, xte, yte = common.load_cifar()
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
    net = Net().to(dev)
    opt = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    nb = len(xtr) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.1, total_steps=epochs * nb)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), generator=gen, device=dev)
        for b in range(nb):
            idx = perm[b * batch : (b + 1) * batch]
            x = crop_flip(xtr[idx].float() / 255, gen)
            loss = F.cross_entropy(net(x), ytr[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            acc = (
                sum(
                    float(
                        (
                            net(xte[i : i + 1000].float() / 255).argmax(1)
                            == yte[i : i + 1000]
                        ).sum()
                    )
                    for i in range(0, 10000, 1000)
                )
                / 10000
            )
        print(f"ep {ep + 1} test acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    RES.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), RES / "classifier.pt")
    write_reference(net, xte, yte, epochs)


@torch.no_grad()
def write_reference(net, xte, yte, epochs):
    """Test accuracy, and the feature mean and covariance of the 10,000 test images."""
    f, correct = [], 0
    for i in range(0, 10000, 1000):
        feats, logits = net.features(xte[i : i + 1000].float() / 255)
        f.append(feats.cpu())
        correct += int((logits.argmax(1) == yte[i : i + 1000]).sum())
    f = torch.cat(f).double()
    torch.save({"mu": f.mean(0), "cov": torch.cov(f.T)}, RES / "classifier_ref.pt")
    info = {"test_acc": correct / 10000, "epochs": epochs}
    (RES / "classifier.json").write_text(json.dumps(info))
    print(info)


if __name__ == "__main__":
    if "--ref-only" in sys.argv:
        dev = common.device()
        _, _, xte, yte = common.load_cifar()
        write_reference(load_classifier(dev), xte.to(dev), yte.to(dev), 30)
    else:
        main()
