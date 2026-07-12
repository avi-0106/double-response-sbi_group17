"""
Adapter: tells BayesFlow which values are parameters to learn,
which are observed trial data, and rescales everything to a
similar range so training works well.
"""

import bayesflow as bf

adapter = (
    bf.Adapter()
    # trial-by-trial data: order doesn't matter, treat as an unordered set
    .as_set(["first_response_time", "correct", "double_flag", "second_response_time"])
    # these must always be positive (rates, times, thresholds)
    .constrain(["nu", "alpha1", "tau"], lower=0)
    # rescale each parameter using its real mean/std, so the network
    # sees roughly similar-sized numbers for everything
    .standardize(include="nu", mean=2.502, std=1.105)
    .standardize(include="alpha1", mean=1.005, std=0.449)
    .standardize(include="tau", mean=0.159, std=0.165)
    # bundle the 3 parameters together - this is what the network predicts
    .concatenate(["nu", "alpha1", "tau"], into="inference_variables")
    # bundle the trial data together - this is what the network reads to make its guess
    .concatenate(
        ["first_response_time", "correct", "double_flag", "second_response_time"],
        into="summary_variables",
    )
)