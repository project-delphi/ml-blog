"""Run a grid of vae.py or jepa.py jobs, skipping any whose result JSON exists.

The defaults reproduce the post. From the repo root, with CIFAR-10's python batches
under ~/.keras/datasets and results/classifier.pt from classifier.py:

    RUN="uv run --no-project --with torch==2.15.0 --with numpy==2.5.4"
    $RUN python posts/kernel-stein-discrepancy-as-a-regulariser/src/sweep.py vae --pilot
    $RUN python posts/kernel-stein-discrepancy-as-a-regulariser/src/sweep.py vae
    $RUN python posts/kernel-stein-discrepancy-as-a-regulariser/src/sweep.py jepa --pilot
    $RUN python posts/kernel-stein-discrepancy-as-a-regulariser/src/sweep.py jepa

- ``vae --pilot``: 3 epochs, seed 0, two strengths per regulariser (results/vae-pilot/).
- ``vae``: GRIDS["vae"], 15 epochs, seeds 0 and 1 (results/vae/).
- ``jepa --pilot``: 4 epochs, seed 0, lambda in {0.01, 0.1, 1, 10}, scored on 5,000
  held-out training images (``--val``; results/jepa-pilot/, tagged ``_val``).
- ``jepa``: GRIDS["jepa"], 12 epochs, seeds 10-12; the unregularised run uses seed 10
  only (results/jepa/).

``--workers 2`` runs two jobs at once, which on an Apple M1 Pro was about 1.7 times the
throughput of one. ``--seeds`` and ``--epochs`` override the defaults.
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
    # alike).
    "vae": {
        "none": (0.0,),
        "kl": (0.03, 0.1, 0.3, 1.0),
        "moments": (0.1, 1.0, 10.0, 100.0),
        "sigreg": (1.0, 10.0, 100.0, 1000.0),
        "mmd": (10.0, 100.0, 1000.0, 10000.0),
        "ksd_imq": (10.0, 100.0, 1000.0, 10000.0),
        "ksd_mix": (10.0, 100.0, 1000.0, 10000.0),
    },
    # JEPA: each arm at the strength with the best validation probe in the pilot.
    "jepa": {
        "ksd_imq": (0.1,),
        "ksd_mix": (0.01,),
        "mmd": (1.0,),
        "moments": (0.01,),
        "sigreg": (0.01,),
        "vicreg": (1.0,),
        "none": (0.0,),
    },
}
PILOT_REGS = {
    "vae": ("kl", "moments", "mmd", "sigreg", "ksd_imq", "ksd_mix"),
    "jepa": ("vicreg", "moments", "mmd", "sigreg", "ksd_imq", "ksd_mix"),
}
DEFAULTS = {  # (exp, pilot) -> seeds, epochs
    ("vae", True): ([0], 3),
    ("vae", False): ([0, 1], 15),
    ("jepa", True): ([0], 4),
    ("jepa", False): ([10, 11, 12], 12),
}


def jobs(exp, pilot, seeds, epochs):
    if pilot:
        out = RES / f"{exp}-pilot"
        lams = PILOT_LAMS_JEPA if exp == "jepa" else PILOT_LAMS
        grid = {r: PILOT_LAMS_KL if r == "kl" else lams for r in PILOT_REGS[exp]}
    else:
        out = RES / exp
        grid = GRIDS[exp]
    val = pilot and exp == "jepa"
    for (reg, lams), seed in itertools.product(grid.items(), seeds):
        if exp == "jepa" and reg == "none" and seed != seeds[0]:
            continue  # collapse needs no second seed
        for lam in lams:
            tag = f"{reg}_lam{lam:g}_s{seed}" + ("_val" if val else "")
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
            cmd += ["--seed", str(seed), "--out", str(out), "--epochs", str(epochs)]
            if val:
                cmd += ["--val"]
            yield tag, cmd, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=("vae", "jepa"))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args()
    seeds, epochs = DEFAULTS[(a.exp, a.pilot)]
    queue = list(jobs(a.exp, a.pilot, a.seeds or seeds, a.epochs or epochs))
    print(f"{len(queue)} jobs", flush=True)
    running = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < a.workers:
            tag, cmd, out = queue.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = open(out / f"{tag}.log", "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            running.append((tag, proc, log))
        for job in list(running):
            tag, p, log = job
            if p.poll() is not None:
                running.remove(job)
                log.close()
                print(f"{time.time() - t0:7.0f}s {tag} exit {p.returncode}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
