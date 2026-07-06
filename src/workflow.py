"""
Combines the simulator, adapter, and neural networks into one
trainable BayesFlow workflow.
"""

import bayesflow as bf
from bf_simulator import simulator
from adapter import adapter

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=bf.networks.CouplingFlow(),
    summary_network=bf.networks.DeepSet(),
    inference_variables=["nu", "alpha1", "alpha2", "tau"],
    inference_conditions=["n"],
    summary_variables=["first_response_time", "correct", "double_flag", "second_response_time"],
)