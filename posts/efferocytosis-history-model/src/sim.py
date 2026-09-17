"""Simulators for the post: a supply-limited toy, and the two-wave experiment.

Both are synthetic. The toy exists to show that a population uptake curve can be
pinned by target supply while the per-cell allocation underneath it is free. The
two-wave simulator stands in for a live-imaging efferocytosis assay with two
labelled target waves, several donors, and cells that can be lost from tracking.

Nothing here reads a file or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- true smooths

#: Uptake first raises the intensity, then depresses it as undigested cargo
#: accumulates. The peak is the priming arm, the decline the satiety arm.
def f1_true(load: np.ndarray) -> np.ndarray:
    return 0.55 * load * np.exp(-load / 2.5) - 0.10 * load


#: A refractory window: the intensity is suppressed right after an uptake and
#: recovers over roughly an hour.
def f2_true(gap_hours: np.ndarray) -> np.ndarray:
    return -1.2 * np.exp(-gap_hours / 0.75)


def f3_true(nbr_load: np.ndarray, mode: str) -> np.ndarray:
    """Neighbour term. Under 'depletion' the neighbours act only through the
    opportunity offset, so the appetite term is flat at zero.
    """
    if mode == "depletion":
        return np.zeros_like(nbr_load, dtype=float)
    if mode == "signalling":
        return -0.25 * np.log1p(nbr_load)
    raise ValueError(f"unknown neighbour mode: {mode!r}")


# ------------------------------------------------------------------- the toy


@dataclass(frozen=True)
class ToyRun:
    """One supply-limited wave under one allocation rule."""

    cumulative: np.ndarray  # population uptake through time
    per_cell: np.ndarray  # final count per cell
    experienced: np.ndarray  # boolean, the pre-designated experienced cells


def toy_mechanisms(
    seed: int = 11,
    n_cells: int = 200,
    n_steps: int = 60,
    pool: int = 900,
    gamma: float = 1.1,
    experienced_frac: float = 0.35,
) -> tuple[ToyRun, ToyRun]:
    """Two allocation rules drawing on one fixed, limited target pool.

    Because the pool is what runs out, the population curve is supply-limited
    and comes out nearly identical either way. Only the allocation differs:
    under priming the experienced cells take a larger share, under satiety they
    take a smaller one and the naive cells absorb what is left.
    """
    rng = np.random.default_rng(seed)
    experienced = np.zeros(n_cells, dtype=bool)
    experienced[rng.choice(n_cells, size=int(experienced_frac * n_cells), replace=False)] = True

    runs = []
    for sign in (+1.0, -1.0):  # priming, then satiety
        step_rng = np.random.default_rng(seed + 1)
        weight = np.exp(sign * gamma * experienced.astype(float))
        counts = np.zeros(n_cells, dtype=int)
        left = pool
        cumulative = np.zeros(n_steps + 1)
        for step in range(n_steps):
            # Demand is capped by what the pool still holds, so both rules eat
            # through the same supply on the same schedule.
            demand = min(left, int(round(pool / n_steps * 1.35)))
            if demand > 0:
                share = weight / weight.sum()
                taken = step_rng.multinomial(demand, share)
                counts += taken
                left -= demand
            cumulative[step + 1] = counts.sum()
        runs.append(ToyRun(cumulative=cumulative, per_cell=counts, experienced=experienced))
    return runs[0], runs[1]


# ------------------------------------------------------- the two-wave assay

DT = 0.25  # hours per simulation bin
WAVE1 = (0, 24)  # bins: 0-6 h
WAVE2 = (36, 72)  # bins: 9-18 h, after a wash at 6 h and a 3 h interval
N_BINS = 72
# 4 h to clear one internalised corpse. This has to be slow relative to the
# interval between waves, or every cell enters the second wave with an empty
# phagolysosomal compartment and the first wave leaves no trace in the load.
DIGEST_BINS = 16
RADIUS = 0.12  # a cell reaches targets within this distance
GAP_CAP = 12.0  # hours, the value a cell that has never eaten carries


@dataclass(frozen=True)
class Experiment:
    """Per-(cell, bin) records plus the per-cell truth behind them."""

    panel: pd.DataFrame
    cells: pd.DataFrame
    neighbour_mode: str


def _well(
    rng: np.random.Generator,
    arm: str,
    n_cells: int,
    n_targets: int,
    focal_frac: float,
    lam0: float,
    offset: float,
    neighbour_mode: str,
    loss_hazard: float,
    avidity_sd: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    pos = rng.random((n_cells, 2))
    adjacency = (
        np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) < RADIUS
    ).astype(float)
    np.fill_diagonal(adjacency, 0.0)

    avidity = rng.normal(0.0, avidity_sd, size=n_cells)

    # Wave 1: identical budget in both treated arms, different placement.
    if arm == "none":
        pos1 = np.zeros((0, 2))
        arrive1 = np.zeros(0, dtype=int)
        designated = np.zeros(n_cells, dtype=bool)
    else:
        arrive1 = rng.integers(WAVE1[0], WAVE1[1], size=n_targets)
        designated = np.zeros(n_cells, dtype=bool)
        designated[
            rng.choice(n_cells, size=int(focal_frac * n_cells), replace=False)
        ] = True
        if arm == "broad":
            pos1 = rng.random((n_targets, 2))
        else:  # focal: the same targets, packed around a pre-defined subset
            hosts = rng.choice(np.flatnonzero(designated), size=n_targets)
            pos1 = np.clip(pos[hosts] + rng.normal(0.0, 0.045, size=(n_targets, 2)), 0, 1)

    # Wave 2 is common to every arm.
    pos2 = rng.random((n_targets, 2))
    arrive2 = rng.integers(WAVE2[0], WAVE2[1], size=n_targets)

    tpos = np.vstack([pos1, pos2])
    arrive = np.concatenate([arrive1, arrive2])
    wave = np.concatenate([np.ones(len(arrive1), dtype=int), np.full(n_targets, 2)])
    reach = np.linalg.norm(pos[:, None, :] - tpos[None, :, :], axis=-1) < RADIUS
    reach_f = reach.astype(float)

    alive = np.ones(len(arrive), dtype=bool)
    load = np.zeros(n_cells)
    queue = np.zeros((n_cells, DIGEST_BINS))
    last_event = np.full(n_cells, -np.inf)
    # Tracking loss as a constant per-bin hazard. A cell whose first failure
    # falls past the end of the run is simply never lost.
    lost_at = rng.geometric(loss_hazard, size=n_cells) - 1
    lost_at = np.where(lost_at >= N_BINS, N_BINS, lost_at).astype(int)

    rec: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("bin", "cell", "y", "load", "gap", "nbr", "opportunity", "at_risk", "wave")
    }
    uptake_by_wave = np.zeros((n_cells, 2), dtype=int)

    for b in range(N_BINS):
        if b == WAVE1[1]:
            # The wash. Uneaten first-wave corpses are removed, which is what
            # makes "naive" mean "ate nothing in the first wave" rather than
            # "has not got round to the leftovers yet". Without this the first
            # wave never ends: leftovers stay edible for the whole run, they
            # inflate the second-wave opportunity differentially by arm, and
            # over half of first-wave uptake lands after the first wave.
            alive[wave == 1] = False

        slot = b % DIGEST_BINS
        load = load - queue[:, slot]
        queue[:, slot] = 0.0

        available = alive & (arrive <= b)
        opportunity = reach_f @ available.astype(float)
        gap = np.where(
            np.isneginf(last_event), GAP_CAP, np.minimum((b - last_event) * DT, GAP_CAP),
        )
        nbr = adjacency @ load
        at_risk = (b < lost_at).astype(float)

        eta = (
            f1_true(load)
            + f2_true(gap)
            + f3_true(nbr, neighbour_mode)
            + offset
            + avidity
        )
        rate = lam0 * np.exp(eta) * opportunity * DT * at_risk
        y = rng.poisson(np.clip(rate, 0.0, 50.0))

        # Allocate against the actual targets in reach, in a random cell order,
        # so two neighbours cannot both eat the same corpse.
        y_done = np.zeros(n_cells, dtype=int)
        for i in rng.permutation(np.flatnonzero(y > 0)):
            pool = np.flatnonzero(reach[i] & alive & (arrive <= b))
            if pool.size == 0:
                continue
            # Choose at random among the corpses in reach. Taking them in index
            # order would quietly prefer wave-1 targets, which are stored first.
            take = rng.permutation(pool)[: int(y[i])]
            alive[take] = False
            y_done[i] = take.size
            uptake_by_wave[i, 0] += int((wave[take] == 1).sum())
            uptake_by_wave[i, 1] += int((wave[take] == 2).sum())

        rec["bin"].append(np.full(n_cells, b))
        rec["cell"].append(np.arange(n_cells))
        rec["y"].append(y_done)
        rec["load"].append(load.copy())
        rec["gap"].append(gap)
        rec["nbr"].append(nbr)
        rec["opportunity"].append(opportunity)
        rec["at_risk"].append(at_risk)
        rec["wave"].append(np.full(n_cells, 1 if b < WAVE1[1] else (2 if b >= WAVE2[0] else 0)))

        queue[:, slot] = y_done
        load = load + y_done
        last_event = np.where(y_done > 0, b, last_event)

    panel = {k: np.concatenate(v) for k, v in rec.items()}
    cells = {
        "cell": np.arange(n_cells),
        "avidity": avidity,
        "designated": designated,
        "wave1_uptake": uptake_by_wave[:, 0],
        "wave2_uptake": uptake_by_wave[:, 1],
        "lost_at": lost_at,
    }
    return panel, cells


def simulate_experiment(
    seed: int = 20260917,
    n_donors: int = 6,
    n_cells: int = 80,
    n_targets: int = 600,
    focal_frac: float = 0.30,
    lam0: float = 0.050,
    donor_sd: float = 0.30,
    well_sd: float = 0.15,
    avidity_sd: float = 0.45,
    loss_hazard: float = 0.002,
    neighbour_mode: str = "depletion",
    arms: tuple[str, ...] = ("broad", "focal", "none"),
) -> Experiment:
    """Run the assay: every donor contributes one well to every arm.

    Arm is assigned, so the broad-versus-focal contrast is randomised by design
    even though each cell's own uptake history is not.
    """
    rng = np.random.default_rng(seed)
    donor_effect = rng.normal(0.0, donor_sd, size=n_donors)

    panels, cell_frames = [], []
    well_id = 0
    for donor in range(n_donors):
        for arm in arms:
            well_effect = rng.normal(0.0, well_sd)
            panel, cells = _well(
                rng=np.random.default_rng(rng.integers(0, 2**32)),
                arm=arm,
                n_cells=n_cells,
                n_targets=n_targets,
                focal_frac=focal_frac,
                lam0=lam0,
                offset=donor_effect[donor] + well_effect,
                neighbour_mode=neighbour_mode,
                loss_hazard=loss_hazard,
                avidity_sd=avidity_sd,
            )
            frame = pd.DataFrame(panel)
            frame["donor"] = donor
            frame["arm"] = arm
            frame["well"] = well_id
            frame["uid"] = well_id * n_cells + frame["cell"]
            panels.append(frame)

            cframe = pd.DataFrame(cells)
            cframe["donor"] = donor
            cframe["arm"] = arm
            cframe["well"] = well_id
            cframe["uid"] = well_id * n_cells + cframe["cell"]
            cell_frames.append(cframe)
            well_id += 1

    return Experiment(
        panel=pd.concat(panels, ignore_index=True),
        cells=pd.concat(cell_frames, ignore_index=True),
        neighbour_mode=neighbour_mode,
    )


def second_wave_panel(exp: Experiment) -> pd.DataFrame:
    """Rows the intensity model is fitted on: second wave, still tracked, and
    with at least one target in reach. A cell with no target in reach carries no
    information about appetite, which is what the offset encodes.
    """
    df = exp.panel
    keep = (df["wave"] == 2) & (df["at_risk"] > 0) & (df["opportunity"] > 0)
    out = df.loc[keep].copy()
    out["log_offset"] = np.log(out["opportunity"] * DT)
    return out.reset_index(drop=True)


def coarsen(panel: pd.DataFrame, factor: int = 6) -> pd.DataFrame:
    """Aggregate the fitting grid to `factor` x DT hours.

    The counting-process likelihood factorises into Poisson-like contributions
    on any partition of the follow-up, so a coarser grid is a cheaper version of
    the same model rather than a different one. Counts sum; the covariates are
    read at the start of each block, where they are the predictors for it.
    """
    p = panel.copy()
    p["block"] = p["bin"] // factor
    # Exposure is summed, not averaged. `second_wave_panel` has already dropped
    # bins where the cell was untracked or had nothing in reach, so a block can
    # hold fewer than `factor` bins. Crediting it the full block width would
    # hand a cell that was at risk for one bin the exposure of six, and it
    # would do so precisely for the censored and target-depleted cells the
    # offset exists to handle.
    p["exposure"] = p["opportunity"] * DT
    out = (
        p.sort_values("bin")
        .groupby(["uid", "block"], as_index=False)
        .agg(
            y=("y", "sum"),
            load=("load", "first"),
            gap=("gap", "first"),
            nbr=("nbr", "first"),
            exposure=("exposure", "sum"),
            opportunity=("opportunity", "mean"),
            bins=("bin", "size"),
            donor=("donor", "first"),
            well=("well", "first"),
            arm=("arm", "first"),
        )
    )
    out["log_offset"] = np.log(out["exposure"])
    return out
