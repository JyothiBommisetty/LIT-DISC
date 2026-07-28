import numpy as np
from src.models import LiterarySignal, NodeID, NarrativeEmbedding
from src.config import HMPDTConfig
from src.coupling.kernel import CouplingKernel
from src.coupling.memory import MemorySystem
from src.resonance import NarrativeResonance
from src.uncertainty import UncertaintyPropagator


class RefinementController:
    """Refinement controller for K-iteration HMP-DT inference.

    Algorithm §15, Steps 4a–4h:
      4a. Bottom-up with horizontal coupling
      4b. Paragraph → Chapter (salience-weighted)
      4c. Chapter → Narrative Layer (spline)
      4d. Recursive resonance
      4e. Memory storage
      4f. Memory decay
      4g. Top-down refinement
      4h. Convergence check

    Integrates uncertainty propagation (§13) and bootstrap estimation.
    """

    def __init__(self, config: HMPDTConfig | None = None,
                 coupling_kernel: CouplingKernel | None = None,
                 memory: MemorySystem | None = None,
                 resonance: NarrativeResonance | None = None,
                 uncertainty: UncertaintyPropagator | None = None):
        c = config or HMPDTConfig()
        self.n_iterations = c.refinement_iterations
        self.beta_topdown = c.beta_topdown
        self.epsilon = c.epsilon_convergence
        self.n_samples = c.narrative_samples
        self.kernel = coupling_kernel or CouplingKernel()
        self.memory = memory or MemorySystem()
        self.resonance = resonance or NarrativeResonance()
        self.uncertainty = uncertainty or UncertaintyPropagator()

    def run(self, tree, signals: dict[NodeID, dict[str, LiterarySignal]],
            bottom_up, top_down,
            dimensions: list[str] | None = None) -> tuple[dict[NodeID, dict[str, LiterarySignal]], int]:
        dims = dimensions or ["S", "N", "I", "Sy", "E", "Em"]
        # Store coupling kernel/memory/resonance/uncertainty for shared state
        if hasattr(bottom_up, 'kernel'):
            bottom_up.kernel = self.kernel
        if hasattr(bottom_up, 'memory'):
            bottom_up.memory = self.memory
        if hasattr(bottom_up, 'resonance'):
            bottom_up.resonance = self.resonance
        if hasattr(bottom_up, 'uncertainty'):
            bottom_up.uncertainty = self.uncertainty

        # Determine H dimension from feature k_d values (before propagation, max_len = 8)
        k_d_map = {"S": 8, "N": 8, "I": 7, "Sy": 7, "E": 7, "Em": 7}
        max_kd = max(k_d_map.get(d, 7) for d in dims)
        h_dim = len(dims) * max_kd
        self.resonance.reset(h_dim)

        current = signals
        n_iters_done = 0
        for k in range(1, self.n_iterations + 1):
            # 4a. Bottom-up with horizontal coupling
            tau = k / self.n_iterations if self.n_iterations > 0 else 0.5
            current = bottom_up.propagate(tree, current, dimensions=dims,
                                           coupling_tau=tau, refine_iter=k)

            # 4d. Recursive resonance at narrative layer τ_samples
            delta_tau = 1.0 / max(self.n_samples, 1)
            for s in range(self.n_samples):
                tau_s = s * delta_tau
                H_tau = self._interpolate_H(tree, current, dims, tau_s)
                self.resonance.update(H_tau, delta_tau)

            # 4e. Memory storage (for all nodes with strong signals)
            self._store_memories(current, dims)

            # 4f. Memory decay
            self.memory.decay_all()

            # 4g. Top-down refinement
            current = top_down.refine(tree, current, beta_down=self.beta_topdown)

            n_iters_done = k

        return current, n_iters_done

    def _interpolate_H(self, tree, signals: dict[NodeID, dict[str, LiterarySignal]],
                       dims: list[str], tau: float) -> np.ndarray:
        """Interpolate embedding vector at arbitrary τ for resonance."""
        components: list[np.ndarray] = []
        for d in dims:
            level4_nodes = tree.get_level(4)
            taus: list[float] = []
            f_vals: list[np.ndarray] = []
            for node in level4_nodes:
                nid = (node.level, node.index)
                if nid in signals and d in signals[nid]:
                    taus.append(signals[nid][d].tau)
                    f_vals.append(signals[nid][d].f)
            if len(f_vals) >= 2:
                taus_a = np.array(taus)
                dists = np.abs(taus_a - tau)
                inv = 1.0 / (dists + 1e-10)
                w = inv / np.sum(inv)
                interp = np.sum(w[:, np.newaxis] * np.array(f_vals), axis=0)
                components.append(interp)
            elif len(f_vals) == 1:
                components.append(f_vals[0])
            else:
                components.append(np.zeros(7, dtype=float))
        if not components:
            return np.zeros(7, dtype=float)
        max_len = max(len(c) for c in components)
        padded = [np.pad(c, (0, max_len - len(c)), mode='constant') for c in components]
        return np.concatenate(padded)

    def _store_memories(self, signals: dict[NodeID, dict[str, LiterarySignal]],
                        dims: list[str]) -> None:
        """Store strong signals in memory. Eq. (12.1)."""
        for nid, dim_dict in signals.items():
            for d in dims:
                if d in dim_dict:
                    sig = dim_dict[d]
                    if self.memory.should_store(sig.f, sig.gamma):
                        self.memory.store_signal(sig)

    def bootstrap_ldri(self, embedding: NarrativeEmbedding,
                        n_samples: int = 100) -> dict:
        """Bootstrap confidence intervals for LDRI estimate."""
        return self.uncertainty.bootstrap_ldri(embedding, n_samples)

    def get_uncertainty_evolution(self) -> list[dict]:
        """Get per-level uncertainty statistics for explanation trace."""
        return [
            {"level": lev, "stats": self.uncertainty.get_level_stats(lev)}
            for lev in range(1, 6)
        ]

    def reset(self) -> None:
        self.memory.clear()
        self.uncertainty.clear()
        if hasattr(self, 'resonance'):
            self.resonance.reset(7)  # default dim, will be overridden on first run
