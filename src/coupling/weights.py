import numpy as np


class AdaptiveWeightComputer:
    def compute_weights(self, signals, intensity: str = "medium") -> np.ndarray:
        return np.ones((6, 6, 4)) * 0.5
