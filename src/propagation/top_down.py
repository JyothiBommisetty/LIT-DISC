class TopDownRefiner:
    def __init__(self, lambda_d: float = 0.15):
        self.lambda_d = lambda_d

    def refine(self, tree, signals):
        return signals
