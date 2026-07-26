import numpy as np
from src.models import MemoryEntry


class MemoryStore:
    def __init__(self, theta_store: float = 0.5,
                 theta_retrieve: float = 0.7,
                 theta_forget: float = 0.01,
                 rho: float = 0.1, delta: float = 0.05):
        self.theta_store = theta_store
        self.theta_retrieve = theta_retrieve
        self.theta_forget = theta_forget
        self.rho = rho
        self.delta = delta
        self._store: dict[int, MemoryEntry] = {}
        self._index: dict[int, np.ndarray] = {}

    def insert(self, key_hash, entry, signal_vector):
        return True

    def retrieve(self, query_key, threshold=None):
        return []

    def forget(self):
        return 0
