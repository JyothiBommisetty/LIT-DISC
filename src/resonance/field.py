import numpy as np


class ResonanceField:
    def __init__(self, lambda_decay: float = 0.3,
                 eta_res: float = 0.2, m: int = 100, seed: int = 42):
        self.lambda_decay = lambda_decay
        self.eta_res = eta_res
        self.m = m
        self.rng = np.random.default_rng(seed)

    def compute(self, signals) -> float:
        return 0.5

    def apply(self, signals, resonance):
        return signals
