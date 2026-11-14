"""Frechet distance between two Gaussians fitted to classifier features, and the
summary of one VAE run that report.py and the pilot table both use.
"""

import json
from pathlib import Path

import torch

RES = Path(__file__).resolve().parents[1] / "results"


def sqrtm_psd(a):
    w, v = torch.linalg.eigh(a)
    return (v * w.clamp_min(0).sqrt()) @ v.T


def frechet(mu1, cov1, mu2, cov2):
    """|mu1 - mu2|^2 + tr(cov1 + cov2 - 2 (cov1^1/2 cov2 cov1^1/2)^1/2)."""
    s1 = sqrtm_psd(cov1)
    cross = sqrtm_psd(s1 @ cov2 @ s1)
    return float((mu1 - mu2).pow(2).sum() + torch.trace(cov1 + cov2 - 2 * cross))


def reference():
    ref = torch.load(RES / "classifier_ref.pt")
    return ref["mu"], ref["cov"]


def summarise_vae(json_path, ref=None):
    """One run's numbers: test MSE, Frechet distances of decoded prior samples and of
    reconstructions to the test set, the code diagnostics, and the share of decoded prior
    samples the classifier labels confidently (max probability > 0.9).
    """
    ref = ref or reference()
    res = json.loads(Path(json_path).read_text())
    blob = torch.load(Path(json_path).with_suffix(".pt"))
    fp = blob["feats_prior"].double()
    fr = blob["feats_rec"].double()
    out = {k: res[k] for k in ("reg", "lam", "seed", "epochs", "test_mse")}
    out["fcd_prior"] = frechet(fp.mean(0), torch.cov(fp.T), *ref)
    out["fcd_rec"] = frechet(fr.mean(0), torch.cov(fr.T), *ref)
    out["confident"] = float(
        (blob["probs_prior"].float().max(1).values > 0.9).float().mean()
    )
    # outliers among the first 2,000 test codes: the largest norm, and the share of the
    # total squared norm carried by the ten largest (about 0.8% for N(0, I_64) draws)
    r = blob["z_test"].double().norm(dim=1).sort(descending=True).values
    out["max_norm"] = float(r[0])
    out["top10_share"] = float(r[:10].pow(2).sum() / r.pow(2).sum())
    out.update(res["codes"])
    return out
