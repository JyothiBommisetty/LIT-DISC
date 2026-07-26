import numpy as np
from src.features.base import BaseFeatureExtractor


class InteractionExtractor(BaseFeatureExtractor):
    k_d: int = 7

    def extract(self, sentence, tokens, pos, deps):
        return (np.zeros(self.k_d, dtype=float), 0.5)
