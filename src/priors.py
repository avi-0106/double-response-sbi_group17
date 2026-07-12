"""
Prior distributions for the RDM with double responses.

Each call to prior() draws one full set of parameters for a
simulated participant: two drift rates, the shared threshold,
and the non-decision time.
"""

import numpy as np
import yaml
from pathlib import Path

with open(Path(__file__).parent / "config.yml") as f:
    config = yaml.safe_load(f)

priors_cfg = config["priors"]
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

    tau = rng.exponential(scale=priors_cfg["tau"]["scale"])

    return {"nu": nu, "alpha1": alpha1, "tau": tau}