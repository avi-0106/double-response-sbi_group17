"""
Loads the trained model so teammates can use it without retraining.
"""
import keras
from adapter import adapter

from pathlib import Path
approximator = keras.saving.load_model(Path(__file__).parent.parent / "results" / "trained_model.keras")

def get_posterior(trial_data, n_samples=500):
    return approximator.sample(conditions=trial_data, num_samples=n_samples)