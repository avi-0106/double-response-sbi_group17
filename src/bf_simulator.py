"""
Combines context, prior, and simulate_dataset into one BayesFlow
simulator object - this is what actually gets used for training.
"""

import bayesflow as bf
from priors import prior
from simulator import context, simulate_dataset

simulator = bf.make_simulator([context, prior, simulate_dataset])