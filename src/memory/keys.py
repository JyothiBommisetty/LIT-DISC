import numpy as np


class FourierFeatureKeyGenerator:
    def __init__(self, input_dim: int = 8, key_dim: int = 32,
                 sigma: float = 1.0, seed: int = 42):
        self.key_dim = key_dim
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, sigma, (key_dim, input_dim))
        self.b = self.rng.uniform(0, 2 * np.pi, key_dim)

    def generate_key(self, signal_vector: np.ndarray) -> np.ndarray:
        proj = self.W @ signal_vector + self.b
        return np.sqrt(2.0 / self.key_dim) * np.cos(proj)
