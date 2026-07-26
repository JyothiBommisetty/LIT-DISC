import numpy as np


class UncertaintyPropagator:
    def propagate_features(self, features, confidences):
        return {}

    def propagate_signals(self, signals):
        return signals

    def compute_confidence_interval(self, value, std_error, level=0.95):
        z = 1.96
        margin = z * std_error
        return (max(0.0, value - margin), min(1.0, value + margin))
