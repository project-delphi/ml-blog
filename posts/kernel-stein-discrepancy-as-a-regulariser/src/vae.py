"""Autoencoders with a N(0, I) prior on CIFAR-10, one regulariser per run.

    loss = sum-of-squares reconstruction error per image + lam * R

- ``--reg kl``: the VAE. The encoder outputs a mean and a log-variance, the code is
  sampled by reparameterisation, and R is the per-image KL to N(0, I).
- any batch regulariser in ``common.REGS``: the encoder is deterministic (it outputs
  the mean only) and R acts on the batch of codes, which is the aggregate posterior's
  sample (Tolstikhin et al. 2018's Wasserstein autoencoder, with R in place of its MMD).
- ``--reg none``: a plain autoencoder.

Writes results/vae/<reg>_lam<lam>_s<seed>.json with test reconstruction error, the
Gaussian diagnostics of the test codes, and 10,000 decoded prior samples' features under
the frozen classifier (for the Frechet distances in report.py), plus a .pt of 64 decoded
prior samples and 16 reconstructions for the figures.

Run from the repo root (CIFAR-10 python batches under ~/.keras/datasets):

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/vae.py --reg ksd_mix --lam 100
"""

import argparse
import json
import time
from pathlib import Path

import common
import torch
import torch.nn as nn
from classifier import load_classifier

HERE = Path(__file__).resolve().parents[1]
D = 64


class Encoder(nn.Module):
    def __init__(self, d=D, stochastic=False):
        super().__init__()
        self.stochastic = stochastic
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.SiLU(),
            nn.Flatten(),
        )  # fmt: skip
        self.head = nn.Linear(256 * 16, 2 * d if stochastic else d)

    def forward(self, x):
        h = self.head(self.net(x))
        if self.stochastic:
            mu, logvar = h.chunk(2, dim=1)
            return mu, logvar.clamp(-12, 6)
        return h, None


class Decoder(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.fc = nn.Linear(d, 256 * 16)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid(),
        )  # fmt: skip

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 4, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", required=True, choices=common.NAMES + ("kl",))
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=str(HERE / "results" / "vae"))
    a = ap.parse_args()

    dev = common.device()
    torch.manual_seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    xtr, _, xte, yte = common.load_cifar()
    xtr = xtr.to(dev)
    xte_f = xte.to(dev).float() / 255

    enc = Encoder(stochastic=a.reg == "kl").to(dev)
    dec = Decoder().to(dev)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=a.lr)
    steps = a.epochs * (len(xtr) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t0, log = time.time(), []
    for ep in range(a.epochs):
        perm = torch.randperm(len(xtr), generator=gen, device=dev)
        tot_rec = tot_reg = 0.0
        nb = len(xtr) // a.batch
        for b in range(nb):
            x = xtr[perm[b * a.batch : (b + 1) * a.batch]].float() / 255
            mu, logvar = enc(x)
            if a.reg == "kl":
                z = (
                    mu
                    + torch.randn(mu.shape, generator=gen, device=dev)
                    * (0.5 * logvar).exp()
                )
                r = common.kl_gauss(mu, logvar)
            else:
                z = mu
                r = common.batch_reg(a.reg, z, gen=gen)
            rec = (dec(z) - x).pow(2).flatten(1).sum(1).mean()
            loss = rec + a.lam * r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_rec += rec.detach()
            tot_reg += r.detach()
        tot_rec, tot_reg = float(tot_rec), float(tot_reg)
        log.append(
            {
                "epoch": ep + 1,
                "rec": tot_rec / nb,
                "reg": tot_reg / nb,
                "t": time.time() - t0,
            }
        )
        print(
            f"[{a.reg} lam={a.lam:g} s={a.seed}] ep {ep + 1} rec {tot_rec / nb:.2f} reg {tot_reg / nb:.4g} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    enc.eval()
    dec.eval()
    with torch.no_grad():
        # codes for the diagnostics: the aggregate posterior. For the VAE that is
        # mu + sigma * eps; reconstructions use mu, as usual.
        zs, mus, mse = [], [], 0.0
        ge = torch.Generator(device=dev).manual_seed(55)
        for i in range(0, len(xte_f), 1000):
            x = xte_f[i : i + 1000]
            mu, logvar = enc(x)
            mus.append(mu)
            if logvar is not None:
                eps = torch.randn(mu.shape, generator=ge, device=dev)
                zs.append(mu + eps * (0.5 * logvar).exp())
            else:
                zs.append(mu)
            mse += float((dec(mu) - x).pow(2).sum())
        zte = torch.cat(zs)
        mute = torch.cat(mus)
        mse /= xte_f.numel()
        diag = common.gaussian_diagnostics(
            zte.cpu(), gen=torch.Generator().manual_seed(123)
        )

        clf = load_classifier(dev)
        g = torch.Generator(device=dev).manual_seed(1000 + a.seed)
        feats_prior, feats_rec, logits_prior = [], [], []
        for i in range(0, 10000, 1000):
            zp = torch.randn(1000, D, generator=g, device=dev)
            xp = dec(zp)
            f, lg = clf.features(xp)
            feats_prior.append(f.cpu())
            logits_prior.append(lg.cpu())
            f, _ = clf.features(dec(mute[i : i + 1000]))
            feats_rec.append(f.cpu())
        feats_prior = torch.cat(feats_prior)
        feats_rec = torch.cat(feats_rec)
        probs = torch.cat(logits_prior).softmax(1)
        zp = torch.randn(
            64, D, generator=torch.Generator(device=dev).manual_seed(7), device=dev
        )
        grid_prior = dec(zp).cpu()
        grid_rec = dec(mute[:16]).cpu()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.reg}_lam{a.lam:g}_s{a.seed}"
    torch.save(
        {"feats_prior": feats_prior.half(), "feats_rec": feats_rec.half(), "probs_prior": probs.half(),
         "grid_prior": grid_prior.half(), "grid_rec": grid_rec.half(), "z_test": zte[:2000].cpu().half()},
        out / f"{tag}.pt",
    )  # fmt: skip
    res = {
        "reg": a.reg, "lam": a.lam, "seed": a.seed, "epochs": a.epochs, "batch": a.batch,
        "d": D, "test_mse": mse, "codes": diag, "log": log, "seconds": time.time() - t0,
    }  # fmt: skip
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    print("done", tag, f"mse={mse:.5f}", diag, flush=True)


if __name__ == "__main__":
    main()
