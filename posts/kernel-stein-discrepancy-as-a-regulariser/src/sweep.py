"""Run a grid of vae.py or jepa.py jobs, skipping any whose result JSON exists.

    uv run --no-project --with torch==2.15.0 --with numpy==2.5.4 \
        python posts/kernel-stein-discrepancy-as-a-regulariser/src/sweep.py vae --workers 2

Grids are defined in GRIDS; ``--pilot`` runs the short lambda-calibration grid instead
and writes to results/<exp>-pilot/.
"""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"

PILOT_LAMS = (1.0, 100.0)
PILOT_LAMS_KL = (0.03, 0.3)
PILOT_LAMS_JEPA = (0.01, 0.1, 1.0, 10.0)
GRIDS = {
    # VAE: four strengths per regulariser, chosen from the 3-epoch pilot so that each
    # line runs from barely regularised to saturated (Adam makes very large lambda act
    # alike). JEPA: filled in from its pilot.
    "vae": {
        "none": (0.0,),
        "kl": (0.03, 0.1, 0.3, 1.0),
        "moments": (0.1, 1.0, 10.0, 100.0),
        "sigreg": (1.0, 10.0, 100.0, 1000.0),
        "mmd": (10.0, 100.0, 1000.0, 10000.0),
        "ksd_imq": (10.0, 100.0, 1000.0, 10000.0),
        "ksd_mix": (10.0, 100.0, 1000.0, 10000.0),
    },
    # JEPA: each arm at the strength with the best validation probe in the 4-epoch
    # pilot over lambda in {0.01, 0.1, 1, 10} (seed 0; final seeds are 10-12).
    "jepa": {
        "ksd_imq": (0.1,),
        "ksd_mix": (0.01,),
        "mmd": (1.0,),
        "moments": (0.01,),
        "sigreg": (0.01,),
        "vicreg": (1.0,),
    },
}
PILOT_REGS = {
    "vae": ("kl", "moments", "mmd", "sigreg", "ksd_imq", "ksd_mix"),
    "jepa": ("vicreg", "moments", "mmd", "sigreg", "ksd_imq", "ksd_mix"),
}


def jobs(exp, pilot, seeds, epochs):
    if pilot:
        out = RES / f"{exp}-pilot"
        lams = PILOT_LAMS_JEPA if exp == "jepa" else PILOT_LAMS
        grid = {r: PILOT_LAMS_KL if r == "kl" else lams for r in PILOT_REGS[exp]}
    else:
        out = RES / exp
        grid = GRIDS[exp]
    for (reg, lams), seed in itertools.product(grid.items(), seeds):
        for lam in lams:
            tag = f"{reg}_lam{lam:g}_s{seed}"
            if (out / f"{tag}.json").exists():
                continue
            cmd = [
                sys.executable,
                str(HERE / f"{exp}.py"),
                "--reg",
                reg,
                "--lam",
                str(lam),
            ]
            cmd += ["--seed", str(seed), "--out", str(out)]
            if epochs:
                cmd += ["--epochs", str(epochs)]
            if pilot and exp == "jepa":
                cmd += ["--val"]
            yield tag, cmd, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=("vae", "jepa"))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=0)
    a = ap.parse_args()
    queue = list(jobs(a.exp, a.pilot, a.seeds, a.epochs))
    print(f"{len(queue)} jobs", flush=True)
    running = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < a.workers:
            tag, cmd, out = queue.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / f"{tag}.log", "w")
            running.append(
                (
                    tag,
                    subprocess.Popen(
                        cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE
                    ),
                )
            )
        for tag, p in list(running):
            if p.poll() is not None:
                running.remove((tag, p))
                print(f"{time.time() - t0:7.0f}s {tag} exit {p.returncode}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
