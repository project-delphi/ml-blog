"""A joint-embedding predictive architecture (JEPA) on CIFAR-10, one anti-collapse
regulariser per run.

Two augmented views of each image go through one encoder (a small ResNet and an MLP
projector to a 64-dimensional embedding z). A small predictor maps each view's embedding
to the other's, and

    loss = mean squared prediction error (both directions) + lam * (R(z1) + R(z2)) / 2

with R one of ``common.REGS`` aimed at N(0, I_64), or ``none``. No EMA target and no
stop-gradient: the regulariser is the only thing that stops collapse.

Evaluation, on frozen features of un-augmented images: a linear probe and a 20-nearest-
neighbour classifier on the 256-d backbone features (standard practice), and the
Gaussian diagnostics and effective rank of the test embeddings.

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/jepa.py --reg ksd_imq --lam 0.1 --seed 10
"""

import argparse
import json
import time
from pathlib import Path

import common
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
D = 64


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )
            if stride != 1 or cin != cout
            else nn.Identity()
        )

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, width=32, d=D):
        super().__init__()
        w = width
        self.backbone = nn.Sequential(
            nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
            Block(w, w, 1), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2), Block(4 * w, 8 * w, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )  # fmt: skip
        self.feat_dim = 8 * w
        self.proj = nn.Sequential(
            nn.Linear(8 * w, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, d)
        )

    def forward(self, x01):
        h = self.backbone(common.normalise(x01))
        return h, self.proj(h)


def predictor(d=D):
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d))


@torch.no_grad()
def embed(enc, x_u8, dev):
    hs, zs = [], []
    for i in range(0, len(x_u8), 1000):
        h, z = enc(x_u8[i : i + 1000].to(dev).float() / 255)
        hs.append(h)
        zs.append(z)
    return torch.cat(hs), torch.cat(zs)


def linear_probe(h_tr, ytr, h_te, yte, dev, epochs=60):
    m, s = h_tr.mean(0), h_tr.std(0) + 1e-6
    h_tr, h_te = (h_tr - m) / s, (h_te - m) / s
    clf = nn.Linear(h_tr.shape[1], 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(h_tr), generator=g, device=dev)
        for i in range(0, len(h_tr), 1024):
            idx = perm[i : i + 1024]
            loss = F.cross_entropy(clf(h_tr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return float((clf(h_te).argmax(1) == yte).float().mean())


@torch.no_grad()
def knn(h_tr, ytr, h_te, yte, k=20, temp=0.07):
    a, b = F.normalize(h_tr, dim=1), F.normalize(h_te, dim=1)
    correct = 0
    for i in range(0, len(b), 1000):
        sim = b[i : i + 1000] @ a.T
        v, idx = sim.topk(k, dim=1)
        w = (v / temp).exp()
        votes = torch.zeros(len(idx), 10, device=b.device).scatter_add_(1, ytr[idx], w)
        correct += int((votes.argmax(1) == yte[i : i + 1000]).sum())
    return correct / len(b)


def effective_rank(z):
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True, choices=common.NAMES)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=str(HERE / "results" / "jepa"))
    ap.add_argument(
        "--val",
        action="store_true",
        help="evaluate on the last 5,000 training images (probe fit on the first 45,000)",
    )
    a = ap.parse_args()

    dev = common.device()
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    xtr, ytr, xte, yte = common.load_cifar()
    xtr_d = xtr.to(dev)

    enc = Encoder().to(dev)
    pred = predictor().to(dev)
    params = [*enc.parameters(), *pred.parameters()]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    nb = len(xtr) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, a.lr, total_steps=a.epochs * nb, pct_start=0.1
    )

    t0, log = time.time(), []
    for ep in range(a.epochs):
        enc.train()
        perm = torch.randperm(len(xtr), generator=gen, device=dev)
        s_inv = s_reg = 0.0
        for b in range(nb):
            x = xtr_d[perm[b * a.batch : (b + 1) * a.batch]].float() / 255
            _, z1 = enc(common.augment(x, gen))
            _, z2 = enc(common.augment(x, gen))
            inv = 0.5 * (F.mse_loss(pred(z1), z2) + F.mse_loss(pred(z2), z1))
            r = 0.5 * (
                common.batch_reg(a.reg, z1, gen) + common.batch_reg(a.reg, z2, gen)
            )
            loss = inv + a.lam * r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            s_inv += inv.detach()
            s_reg += r.detach()
        s_inv, s_reg = float(s_inv), float(s_reg)
        log.append(
            {
                "epoch": ep + 1,
                "inv": s_inv / nb,
                "reg": s_reg / nb,
                "t": time.time() - t0,
            }
        )
        print(
            f"[{a.reg} lam={a.lam:g} s={a.seed}] ep {ep + 1} inv {s_inv / nb:.4f} reg {s_reg / nb:.4g} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    enc.eval()
    if a.val:  # lambda selection never sees the test set
        xtr, xte, ytr, yte = xtr[:45000], xtr[45000:], ytr[:45000], ytr[45000:]
    h_tr, ztr = embed(enc, xtr, dev)
    h_te, zte = embed(enc, xte, dev)
    ytr_d, yte_d = ytr.to(dev), yte.to(dev)
    res = {
        "reg": a.reg, "lam": a.lam, "seed": a.seed, "val": a.val, "epochs": a.epochs, "batch": a.batch, "d": D,
        "probe_backbone": linear_probe(h_tr, ytr_d, h_te, yte_d, dev),
        "knn_backbone": knn(h_tr, ytr_d, h_te, yte_d),
        "probe_embedding": linear_probe(ztr, ytr_d, zte, yte_d, dev),
        "erank_embedding": effective_rank(zte.cpu().double()),
        "erank_backbone": effective_rank(h_te.cpu().double()),
        "codes": common.gaussian_diagnostics(zte.cpu(), gen=torch.Generator().manual_seed(123)),
        "log": log, "seconds": time.time() - t0,
    }  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.reg}_lam{a.lam:g}_s{a.seed}" + ("_val" if a.val else "")
    torch.save(
        {"z_test": zte[:2000].cpu().half(), "y_test": yte[:2000]}, out / f"{tag}.pt"
    )
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print(
        "done",
        tag,
        {k: res[k] for k in ("probe_backbone", "knn_backbone", "erank_embedding")},
        res["codes"],
        flush=True,
    )


if __name__ == "__main__":
    main()
