"""
Simulator for the two-threshold RDM.

Given one parameter set from prior(), generates a batch of
simulated trials: first-response time, correctness, whether a
double response occurred, and the second-response time.
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


def first_passage_time(drift, threshold, max_t=MAX_T, dt=DT):
    """
    Simulate one accumulator racing toward a threshold, step by step.
    Returns the time it crosses the threshold, or max_t if it never does.
    """
    evidence = 0.0
    t = 0.0
    n_steps = int(max_t / dt)

    for _ in range(n_steps):
        t += dt
        # forward push from the drift, plus random noise each step
        evidence += drift * dt + rng.normal(scale=np.sqrt(dt))
        if evidence >= threshold:
            return t

    return max_t


def trial(nu, alpha1, alpha2, tau):
    """
    Simulate one full trial: both accumulators race to alpha1,
    then the loser keeps racing to alpha2.
    """
    t0 = first_passage_time(nu[0], alpha1)
    t1 = first_passage_time(nu[1], alpha1)

    if t0 < t1:
        first_response_time = t0 + tau
        correct = 0.0  # accumulator 0 won
        loser_drift = nu[1]
    else:
        first_response_time = t1 + tau
        correct = 1.0  # accumulator 1 won
        loser_drift = nu[0]

    t2 = first_passage_time(loser_drift, alpha2)
    double_flag = 1.0 if t2 < MAX_T else 0.0
    second_response_time = (t2 + tau) if double_flag == 1.0 else 0.0

    return first_response_time, correct, double_flag, second_response_time

def simulate_dataset(nu, alpha1, alpha2, tau, n=N_TRIALS):
    """
    Run n independent trials for one parameter set.
    Returns arrays matching what a real participant's data would look like.
    """
    first_response_time = np.zeros(n)
    correct = np.zeros(n)
    double_flag = np.zeros(n)
    second_response_time = np.zeros(n)

    for i in range(n):
        rt1, corr, dbl, rt2 = trial(nu, alpha1, alpha2, tau)
        first_response_time[i] = rt1
        correct[i] = corr
        double_flag[i] = dbl
        second_response_time[i] = rt2

    return {
        "first_response_time": first_response_time,
        "correct": correct,
        "double_flag": double_flag,
        "second_response_time": second_response_time,
    }