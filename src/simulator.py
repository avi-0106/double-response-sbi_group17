"""
Simulator for the RDM with double responses.

Both accumulators race toward the SAME threshold continuously.
Evidence is never reset - after the first response, the loser
keeps accumulating from wherever it already was. If it crosses
the threshold within a 250ms window after the first response,
that's a double response.
"""

import numpy as np
import yaml
from pathlib import Path

with open(Path(__file__).parent / "config.yml") as f:
    config = yaml.safe_load(f)

sim_cfg = config["simulation"]
MAX_T = sim_cfg["max_t"]
DT = sim_cfg["dt"]
N_TRIALS = sim_cfg["n_trials_per_subject"]
DR_WINDOW = 0.25  # 250ms, from Evans et al. (2020)

rng = np.random.default_rng(config["seed"])


def context():
    return {"n": N_TRIALS}


def simulate_dataset(nu, alpha1, tau, n=N_TRIALS, max_t=MAX_T, dt=DT, dr_window=DR_WINDOW):
    total_t = max_t + dr_window
    n_steps = int(total_t / dt)
    sqrt_dt = np.sqrt(dt)

    evidence0 = np.zeros(n)
    evidence1 = np.zeros(n)
    crossed0 = np.zeros(n, dtype=bool)
    crossed1 = np.zeros(n, dtype=bool)
    cross_time0 = np.full(n, np.inf)
    cross_time1 = np.full(n, np.inf)

    for step in range(1, n_steps + 1):
        t = step * dt
        evidence0 += nu[0] * dt + rng.normal(scale=sqrt_dt, size=n)
        evidence1 += nu[1] * dt + rng.normal(scale=sqrt_dt, size=n)

        newly0 = (~crossed0) & (evidence0 >= alpha1)
        newly1 = (~crossed1) & (evidence1 >= alpha1)
        cross_time0[newly0] = t
        cross_time1[newly1] = t
        crossed0 |= newly0
        crossed1 |= newly1

        if crossed0.all() and crossed1.all():
            break

    win_time = np.minimum(cross_time0, cross_time1)
    lose_time = np.maximum(cross_time0, cross_time1)
    win_time = np.where(np.isinf(win_time), max_t, win_time)

    correct = np.where(cross_time0 < cross_time1, 0.0, 1.0)
    first_response_time = win_time + tau

    gap = lose_time - win_time
    double_flag = ((gap <= dr_window) & (~np.isinf(lose_time))).astype(float)
    second_response_time = np.where(double_flag == 1.0, lose_time + tau, 0.0)

    return {
        "first_response_time": first_response_time,
        "correct": correct,
        "double_flag": double_flag,
        "second_response_time": second_response_time,
    }