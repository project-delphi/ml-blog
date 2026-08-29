"""Write the JSON the efficiency plots and the browser widget share.

Usage:
    .venv-tensor-factorizations/bin/python \\
        posts/uses-of-tensor-factorizations/src/export_widget_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensors as T

OUT = Path(__file__).resolve().parent.parent / "widget-data"


def main() -> None:
    rng = np.random.default_rng(T.SEED)
    kernel, _ = T.make_cp_kernel(rng)
    ms = [T.TT_MODE] * T.TT_ORDER
    ns = [T.TT_MODE] * T.TT_ORDER
    matrix, _ = T.make_tt_matrix(rng, ms, ns)
    cube, true_factors = T.make_mixing_cube(rng)

    payload = {
        "cp": T.sweep_cp(kernel),
        "tt": T.sweep_tt(matrix, ms, ns),
        "mixing": T.mixing_fit(cube, true_factors),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "curves.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    cp = payload["cp"]
    tt = payload["tt"]
    mix = payload["mixing"]
    print(f"wrote {path}")
    print(
        "CP R=16 rel_error="
        f"{cp['rel_error'][cp['ranks'].index(16)]:.4f} "
        f"params={cp['params'][cp['ranks'].index(16)]}"
    )
    print(
        "TT r=4 rel_error="
        f"{tt['rel_error'][tt['ranks'].index(4)]:.4f} "
        f"params={tt['params'][tt['ranks'].index(4)]}"
    )
    print(
        f"mixing CP rel_error={mix['cp_rel_error']:.4f} "
        f"mean |corr| CP={mix['mean_cp_corr']:.3f} "
        f"SVD={mix['mean_svd_corr']:.3f}"
    )


if __name__ == "__main__":
    main()
