class ConfidenceEstimator:
    def estimate(self, score) -> float:
        return 0.85

    def per_dimension(self, score) -> dict[str, float]:
        return {}
