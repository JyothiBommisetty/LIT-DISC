import numpy as np
from src.models import LiterarySignal, NodeID


class SignalInitializer:
    DIMENSIONS: list[str] = ["S", "N", "I", "Sy", "E", "Em"]

    def initialize(self, features: dict[str, dict[NodeID, tuple[np.ndarray, float]]],
                   taus: dict[NodeID, float]) -> dict[NodeID, dict[str, LiterarySignal]]:
        signals: dict[NodeID, dict[str, LiterarySignal]] = {}
        for dim, nodes in features.items():
            for nid, (fvec, gamma) in nodes.items():
                tau = taus.get(nid, 0.5)
                if nid not in signals:
                    signals[nid] = {}
                signals[nid][dim] = LiterarySignal(
                    f=fvec,
                    gamma=gamma,
                    tau=tau,
                    context=None,
                    provenance=[nid],
                    residues=np.zeros((6, 6), dtype=float),
                    memory_traces=[],
                )
        return signals
