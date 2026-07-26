import numpy as np


class KernelComponentComputer:
    def compute(self, taus: np.ndarray) -> np.ndarray:
        return np.ones((6, len(taus))) * 0.5

    def compute_component(self, tau: float, params: dict) -> float:
        return 0.5
