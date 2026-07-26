import numpy as np


class BaseFeatureExtractor:
    k_d: int = 0

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def compute_confidence(self, features: np.ndarray) -> float:
        mean_val = float(np.mean(np.abs(features)))
        return min(1.0, 0.3 + 0.7 * mean_val)

    def normalize(self, values: np.ndarray,
                  bounds: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        low, high = bounds
        return np.clip(values, low, high)
