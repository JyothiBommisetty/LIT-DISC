class RefinementController:
    def __init__(self, n_iterations: int = 3, epsilon: float = 1e-4):
        self.n_iterations = n_iterations
        self.epsilon = epsilon

    def run(self, tree, signals, bottom_up, top_down):
        signals = bottom_up.propagate(tree, signals)
        signals = top_down.refine(tree, signals)
        return signals, 1
