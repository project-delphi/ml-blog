"""Make the post's static energy-gap and link-function figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

destination = Path(__file__).resolve().parents[1] / "figures" / "energy-gap.png"
destination.parent.mkdir(parents=True, exist_ok=True)
eta = np.linspace(-4, 4, 500)
logit = 1 / (1 + np.exp(-eta))
probit = ndtr(eta / 1.7)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=100)
fig.patch.set_facecolor("#f7f8f8")
for ax in axes:
    ax.set_facecolor("#f7f8f8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#87949b")
    ax.tick_params(colors="#35454d")
    ax.axvline(0, color="#b7c4c7", lw=1)

axes[0].plot(eta, np.zeros_like(eta), color="#d18230", lw=3, label=r"$E_0=0$")
axes[0].plot(eta, -eta, color="#008d86", lw=3, label=r"$E_1=-\eta$")
axes[0].set(
    xlim=(-4, 4),
    ylim=(-4.5, 4.5),
    xlabel=r"Score $\eta$",
    ylabel="State energy",
    title="A score is an energy gap",
)
axes[0].legend(frameon=False, loc="upper right")

axes[1].plot(eta, logit, color="#008d86", lw=3, label="Logit")
axes[1].plot(
    eta, probit, color="#7356a8", lw=2.5, ls="--", label="Probit (scale matched)"
)
axes[1].axhline(0.5, color="#b7c4c7", lw=1)
axes[1].set(
    xlim=(-4, 4),
    ylim=(-0.04, 1.04),
    xlabel=r"Score $\eta$",
    ylabel=r"$P(y=1\mid x)$",
    title="Normalization turns it into a probability",
)
axes[1].legend(frameon=False, loc="lower right")

fig.tight_layout(pad=2.2)
fig.savefig(destination, dpi=100, facecolor=fig.get_facecolor())
