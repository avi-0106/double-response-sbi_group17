"""
Simulator for the two-threshold RDM (vectorized version).

Instead of looping one trial at a time, this runs ALL trials for a
person simultaneously using numpy arrays - much faster.
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

rng = np.random.default_rng(config["seed"])


def race_to_threshold(drift, threshold, n, max_t=MAX_T, dt=DT):
    """
    Race n trials at once toward a threshold.
    drift can be one number (same for all trials) or an array (different per trial).
    Returns an array of n crossing times.
    """
    n_steps = int(max_t / dt)
    sqrt_dt = np.sqrt(dt)

    drift_arr = np.full(n, drift) if np.isscalar(drift) else drift
    evidence = np.zeros(n)
    crossed = np.zeros(n, dtype=bool)
    crossing_time = np.full(n, max_t)

    for step in range(1, n_steps + 1):
        t = step * dt
        noise = rng.normal(scale=sqrt_dt, size=n)
        evidence += drift_arr * dt + noise

        newly_crossed = (~crossed) & (evidence >= threshold)
        crossing_time[newly_crossed] = t
        crossed |= newly_crossed

        if crossed.all():
            break

    return crossing_time


def simulate_dataset(nu, alpha1, alpha2, tau, n=N_TRIALS):
    t0 = race_to_threshold(nu[0], alpha1, n)
    t1 = race_to_threshold(nu[1], alpha1, n)

    correct = np.where(t0 < t1, 0.0, 1.0)
    first_response_time = np.minimum(t0, t1) + tau
    loser_drift = np.where(t0 < t1, nu[1], nu[0])

    t2 = race_to_threshold(loser_drift, alpha2, n)
    double_flag = (t2 < MAX_T).astype(float)
    second_response_time = np.where(double_flag == 1.0, t2 + tau, 0.0)

    return {
        "first_response_time": first_response_time,
        "correct": correct,
        "double_flag": double_flag,
        "second_response_time": second_response_time,
    }


def trial(nu, alpha1, alpha2, tau):
    """Single-trial version, kept for compatibility with earlier tests."""
    data = simulate_dataset(nu, alpha1, alpha2, tau, n=1)
    return (
        data["first_response_time"][0],
        data["correct"][0],
        data["double_flag"][0],
        data["second_response_time"][0],
    )