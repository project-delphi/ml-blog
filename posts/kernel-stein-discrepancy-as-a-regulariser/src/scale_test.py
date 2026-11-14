"""Can a regulariser, trained on fresh batches, put the scale of every coordinate back
at 1? An encoder in miniature: z = eps * s + b with eps ~ N(0, I_64) redrawn every step,
and only s (per-coordinate scale) and b (shift) learned, by Adam on the regulariser
alone. The right answer is s = 1, b = 0 in all 64 coordinates. Starting scales are
log-uniform on [0.05, 3].

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/scale_test.py

Writes results/scale.json: for each regulariser and seed, the smallest and largest
learned scale and the mean |log s|.
"""

import json
from pathlib import Path

import common
import torch

HERE = Path(__file__).resolve().parents[1]
D, N, STEPS, SEEDS = 64, 256, 1500, (0, 1, 2)

LOSSES = {
    "moments_v": lambda z, g: common.moments_v(z),
    "moments": lambda z, g: common.moments(z),
    "ksd_imq_v": lambda z, g: common.ksd(z, c2=1.0, ustat=False),
    "ksd_imq": lambda z, g: common.ksd(z, c2=1.0),
    "ksd_rbf": lambda z, g: common.ksd_rbf(z),
    "ksd_mix_fixed": lambda z, g: (
        common.ksd(z, c2=1.0) + common.MIX_W * common.ksd_rbf(z)
    ),
    "ksd_mix": lambda z, g: common.ksd_mix(z),
    "mmd": lambda z, g: common.mmd(z, gen=g),
    "sigreg": lambda z, g: common.sigreg(z, gen=g),
}


def run(name, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(100 + seed)
    lo, hi = torch.log(torch.tensor(0.05)), torch.log(torch.tensor(3.0))
    logs = (lo + (hi - lo) * torch.rand(D, generator=g)).requires_grad_(True)
    b = torch.zeros(D, requires_grad=True)
    opt = torch.optim.Adam([logs, b], lr=0.02)
    for _ in range(STEPS):
        z = torch.randn(N, D, generator=g) * logs.exp() + b
        loss = LOSSES[name](z, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    s = logs.detach().exp()
    return {
        "s_min": float(s.min()),
        "s_max": float(s.max()),
        "mean_abs_log_s": float(s.log().abs().mean()),
        "shift": float(b.detach().norm()),
    }


def main():
    out = {"d": D, "n": N, "steps": STEPS, "seeds": list(SEEDS), "runs": {}}
    for name in LOSSES:
        out["runs"][name] = [run(name, s) for s in SEEDS]
        r = out["runs"][name]
        print(
            f"{name:14s} s in [{min(x['s_min'] for x in r):.2f}, {max(x['s_max'] for x in r):.2f}]"
            f"  mean|log s| {sum(x['mean_abs_log_s'] for x in r) / len(r):.3f}",
            flush=True,
        )
    path = HERE / "results" / "scale.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
