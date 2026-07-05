"""
Prior distributions for the two-threshold RDM.

Each call to prior() draws one full set of parameters for a
simulated participant: two drift rates, the first threshold,
the second threshold, and the non-decision time.
"""

import numpy as np
import yaml
from pathlib import Path

# load config.yml once when this file gets imported
with open(Path(__file__).parent / "config.yml") as f:
    config = yaml.safe_load(f)

priors_cfg = config["priors"]

# one seeded generator so results are reproducible
rng = np.random.default_rng(config["seed"])


def prior():
    nu = rng.gamma(
        shape=priors_cfg["nu"]["shape"],
        scale=priors_cfg["nu"]["scale"],
        size=2,
    )

    alpha1 = rng.gamma(
        shape=priors_cfg["alpha1"]["shape"],
        scale=priors_cfg["alpha1"]["scale"],
    )

    # gap is scaled by drift speed, not a flat distance, so fast and slow
    # participants both get a proportionally sized second threshold
    extra_time = rng.exponential(scale=priors_cfg["alpha2_gap"]["scale"])
    loser_speed = np.mean(nu)
    gap = extra_time * loser_speed
    alpha2 = alpha1 + gap

    tau = rng.exponential(scale=priors_cfg["tau"]["scale"])

    return {"nu": nu, "alpha1": alpha1, "alpha2": alpha2, "tau": tau}