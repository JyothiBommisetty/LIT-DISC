class SignalAugmenter:
    def __init__(self, eta_m: float = 0.1):
        self.eta_m = eta_m

    def augment(self, signals, memory_store):
        return signals

    def augment_signal(self, signal, traces):
        return signal
