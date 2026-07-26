import numpy as np


class ReliabilityEstimator:
    def split_half(self, dimension_scores):
        if len(dimension_scores) < 2:
            return 0.5
        half = len(dimension_scores) // 2
        odd = np.array(dimension_scores[0::2])
        even = np.array(dimension_scores[1::2])
        if len(odd) < 2 or len(even) < 2:
            return 0.5
        r = np.corrcoef(odd, even)[0, 1]
        if np.isnan(r):
            return 0.5
        return 2.0 * r / (1.0 + r)

    def bootstrap_ci(self, scores, n_iter=1000):
        return (0.0, 1.0)
