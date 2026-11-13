"""Turn the runs under results/ into the post's figures and the numbers it quotes.

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 --with matplotlib==3.11.2 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/report.py

Reads results/vae/*.json|pt and results/jepa/*.json|pt. The detection and scale tables come
from detect.py and scale_test.py.
Writes figures/vae-frontier.png and figures/vae-samples.png (the cover is made from the
first by scripts/make_cover.py), and prints every table and number the post quotes.
"""

import json
from collections import defaultdict
from pathlib import Path

import fcd
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
FIG = HERE / "figures"

# Light-theme tokens of theme/light.scss, in the post's fixed arm order. Figures sit on
# a white card in both themes.
INK, MUTED, RULE = "#1F2430", "#5F6672", "#D8DBE2"
C = ["#4A3AA7", "#1D6E6E", "#9A5B00", "#A8437A", "#0072B2", "#009E73"]
ARMS_VAE = ["ksd_mix", "ksd_imq", "moments", "mmd", "sigreg", "kl"]
ARMS_JEPA = ["ksd_mix", "ksd_imq", "moments", "mmd", "sigreg", "vicreg"]
LABEL = {
    "ksd_mix": "KSD, mixed kernel",
    "ksd_imq": "KSD, IMQ",
    "moments": "Moments",
    "mmd": "MMD (WAE kernel)",
    "sigreg": "SIGReg",
    "kl": "KL (VAE)",
    "vicreg": "VICReg",
    "none": "none",
}
MARK = {
    "ksd_mix": "o",
    "ksd_imq": "s",
    "moments": "D",
    "mmd": "^",
    "sigreg": "v",
    "kl": "P",
    "vicreg": "P",
}


def colour(arm, arms):
    return C[arms.index(arm)]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------------- VAE
def vae_rows():
    ref = fcd.reference()
    runs = [fcd.summarise_vae(p, ref) for p in sorted((RES / "vae").glob("*.json"))]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["reg"], r["lam"])].append(r)
    rows = []
    keys = (
        "test_mse",
        "fcd_prior",
        "fcd_rec",
        "confident",
        "mean_norm",
        "cov_err",
        "sliced_w2",
        "excess_kurt",
    )
    for (reg, lam), rs in sorted(groups.items()):
        row = {"reg": reg, "lam": lam, "seeds": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs])
            row[k] = float(v.mean())
            row[k + "_lo"], row[k + "_hi"] = float(v.min()), float(v.max())
        row["gap"] = row["fcd_prior"] - row["fcd_rec"]
        rows.append(row)
    return rows


def vae_frontier(rows):
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
    style(ax)
    for arm in ARMS_VAE:
        rs = sorted([r for r in rows if r["reg"] == arm], key=lambda r: r["lam"])
        if not rs:
            continue
        x = [r["test_mse"] * 1e3 for r in rs]
        y = [r["fcd_prior"] for r in rs]
        c = colour(arm, ARMS_VAE)
        ax.plot(x, y, color=c, lw=2, zorder=3)
        for r in rs:
            ax.plot(
                [r["test_mse"] * 1e3] * 2,
                [r["fcd_prior_lo"], r["fcd_prior_hi"]],
                color=c,
                lw=1,
                alpha=0.6,
            )
            ax.plot(
                [r["test_mse_lo"] * 1e3, r["test_mse_hi"] * 1e3],
                [r["fcd_prior"]] * 2,
                color=c,
                lw=1,
                alpha=0.6,
            )
        ax.scatter(x, y, s=46, marker=MARK[arm], color=c, edgecolor="white", linewidth=1.5, zorder=4,
                   label=LABEL[arm])  # fmt: skip
        if arm in (
            "ksd_mix",
            "kl",
        ):  # the two ends of the bundle; the rest sit in the legend
            ax.annotate(LABEL[arm], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5,
                        color=INK, va="center")  # fmt: skip
    base = [r for r in rows if r["reg"] == "none"]
    if base:
        b = base[0]
        ax.scatter(
            [b["test_mse"] * 1e3],
            [b["fcd_prior"]],
            s=60,
            marker="x",
            color=MUTED,
            zorder=4,
        )
        ax.annotate("no regulariser", (b["test_mse"] * 1e3, b["fcd_prior"]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=MUTED, va="center")  # fmt: skip
    ax.set_xlabel(
        "test reconstruction error, mean squared error per pixel (× 10⁻³)",
        color=INK,
        fontsize=9.5,
    )
    ax.set_ylabel("Fréchet distance of decoded N(0, I) draws\nto the test images (lower is better)",
                  color=INK, fontsize=9.5)  # fmt: skip
    seeds = min(r["seeds"] for r in rows if r["reg"] != "none")
    words = {1: "one", 2: "two", 3: "three"}
    ax.set_title(f"Each line is one regulariser at four strengths λ; bars span {words.get(seeds, seeds)} seeds",
                 color=MUTED, fontsize=9, loc="left")  # fmt: skip
    leg = ax.legend(
        loc="upper right",
        fontsize=8.5,
        frameon=False,
        labelcolor=INK,
        handletextpad=0.4,
    )
    for h in leg.legend_handles:
        h.set_sizes([40])
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "vae-frontier.png", facecolor="white")
    plt.close(fig)


def vae_samples(rows, target_mse):
    """Decoded prior samples, one row per arm, at the strength whose test MSE is closest
    to target_mse (seed 0), with the reconstructions of the plain autoencoder on top.
    """
    picks = []
    for arm in ARMS_VAE:
        rs = [r for r in rows if r["reg"] == arm]
        if rs:
            picks.append(min(rs, key=lambda r: abs(r["test_mse"] - target_mse)))
    ncol = 12
    fig, axes = plt.subplots(
        len(picks), 1, figsize=(8.2, 0.78 * len(picks) + 0.2), dpi=200
    )
    for ax, r in zip(np.atleast_1d(axes), picks):
        blob = torch.load(RES / "vae" / f"{r['reg']}_lam{r['lam']:g}_s0.pt")
        g = blob["grid_prior"][:ncol].float().clamp(0, 1)
        strip = torch.cat(
            [torch.nn.functional.pad(im, (1, 1, 1, 1), value=1.0) for im in g], dim=2
        )
        ax.imshow(strip.permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylabel(f"{LABEL[r['reg']]}\nλ = {r['lam']:g}", rotation=0, ha="right", va="center",
                      fontsize=8, color=INK, labelpad=6)  # fmt: skip
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "vae-samples.png", facecolor="white")
    plt.close(fig)
    return picks


# --------------------------------------------------------------------------------- JEPA
def class_separation(z, y):
    """Between-class over within-class sum of squares of embeddings z, labels y."""
    z = z.double()
    mu, sb, sw = z.mean(0), 0.0, 0.0
    for c in y.unique():
        zc = z[y == c]
        m = zc.mean(0)
        sb += len(zc) * float((m - mu).pow(2).sum())
        sw += float((zc - m).pow(2).sum())
    return sb / sw


def jepa_rows():
    runs = []
    for p in sorted((RES / "jepa").glob("*.json")):
        r = json.loads(p.read_text())
        blob = torch.load(p.with_suffix(".pt"))
        r["separation"] = class_separation(blob["z_test"], blob["y_test"])
        r["max_norm"] = float(blob["z_test"].double().norm(dim=1).max())
        runs.append(r)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["reg"], r["lam"])].append(r)
    rows = []
    for (reg, lam), rs in sorted(groups.items()):
        row = {"reg": reg, "lam": lam, "seeds": len(rs)}
        for k in (
            "probe_backbone",
            "knn_backbone",
            "probe_embedding",
            "erank_embedding",
            "separation",
            "max_norm",
        ):
            v = np.array([r[k] for r in rs])
            row[k], row[k + "_lo"], row[k + "_hi"] = (
                float(v.mean()),
                float(v.min()),
                float(v.max()),
            )
        for k in ("mean_norm", "cov_err", "sliced_w2", "excess_kurt"):
            v = np.array([r["codes"][k] for r in rs])
            row[k], row[k + "_lo"], row[k + "_hi"] = (
                float(v.mean()),
                float(v.min()),
                float(v.max()),
            )
        rows.append(row)
    return rows


def main():
    if (RES / "vae").exists():
        rows = vae_rows()
        vae_frontier(rows)
        picks = vae_samples(rows, target_mse=0.012)
        print("\nVAE (means over seeds; ranges in brackets)")
        for r in rows:
            print(
                f"{r['reg']:8s} lam {r['lam']:<7g} n={r['seeds']} mse {r['test_mse'] * 1e3:.2f}e-3 "
                f"fcd_prior {r['fcd_prior']:.2f} [{r['fcd_prior_lo']:.2f}, {r['fcd_prior_hi']:.2f}] "
                f"fcd_rec {r['fcd_rec']:.2f} gap {r['gap']:.2f} cov {r['cov_err']:.3f} sw2 {r['sliced_w2']:.4f} "
                f"kurt {r['excess_kurt']:.2f} mean {r['mean_norm']:.2f}"
            )
        print(
            "sample grid picks:",
            [(p["reg"], p["lam"], round(p["test_mse"] * 1e3, 2)) for p in picks],
        )
    if (RES / "jepa-pilot").exists():
        print("\nJEPA pilot (validation linear probe, 4 epochs, seed 0)")
        pilot = defaultdict(dict)
        for p in sorted((RES / "jepa-pilot").glob("*_val.json")):
            r = json.loads(p.read_text())
            pilot[r["reg"]][r["lam"]] = r["probe_backbone"]
        for reg, by_lam in sorted(pilot.items()):
            best = max(by_lam, key=by_lam.get)
            cells = "  ".join(
                f"{lam:g}: {v * 100:.1f}" for lam, v in sorted(by_lam.items())
            )
            print(f"{reg:8s} {cells}   -> {best:g}")
    if (RES / "jepa").exists():
        print("\nJEPA (means over seeds; ranges in brackets)")
        for r in jepa_rows():
            print(
                f"{r['reg']:8s} lam {r['lam']:<6g} n={r['seeds']} probe {r['probe_backbone'] * 100:.1f} "
                f"[{r['probe_backbone_lo'] * 100:.1f}, {r['probe_backbone_hi'] * 100:.1f}] "
                f"knn {r['knn_backbone'] * 100:.1f} [{r['knn_backbone_lo'] * 100:.1f}, {r['knn_backbone_hi'] * 100:.1f}] "
                f"emb-probe {r['probe_embedding'] * 100:.1f} erank {r['erank_embedding']:.1f} "
                f"sep {r['separation']:.3f} [{r['separation_lo']:.3f}, {r['separation_hi']:.3f}] "
                f"maxnorm {r['max_norm']:.1f} cov {r['cov_err']:.3f} sw2 {r['sliced_w2']:.4f} kurt {r['excess_kurt']:.2f}"
            )


if __name__ == "__main__":
    main()
