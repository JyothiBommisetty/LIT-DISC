"""Literary Signal Memory subsystem. Eqs. (12.1)–(12.7).

Implements:
  - Random Fourier Features key projection (Eq. 12.2)
  - Storage with threshold condition (Eq. 12.1) and capacity enforcement
  - Cosine similarity retrieval (Eq. 12.3)
  - Reinforcement on retrieval (Eq. 12.4)
  - Selective decay (Eq. 12.5) — only non-retrieved entries decay
  - Purging below forget threshold (Eq. 12.6)
  - Signal augmentation (Eq. 12.7)
  - Persistence: save/load to/from .npz
  - Maximum capacity: evicts lowest-beta entry when full

Architecture Reference: §9 Memory Subsystem
"""
import math
import os
import numpy as np
from src.config import MemoryParams
from src.models import LiterarySignal, MemoryEntry


class RandomFourierFeatures:
    """Random Fourier Features key projection. Eq. (12.2).

    k(f) = sqrt(2/k_key) · cos(Wf + b)
    W_ij ~ N(0, sigma^2),  b_i ~ Uniform(0, 2*pi)

    Fixed seeds (0 for W, 1 for b) ensure deterministic keys.
    """

    def __init__(self, params: MemoryParams | None = None):
        p = params or MemoryParams()
        self.k_key = p.key_dim
        self.sigma = p.sigma_kernel
        self._W_dict: dict[int, np.ndarray] = {}
        self._b_dict: dict[int, np.ndarray] = {}
        self._seed_W = p.seed_W
        self._seed_b = p.seed_b

    def _ensure_initialized(self, input_dim: int) -> None:
        if input_dim not in self._W_dict:
            rng_W = np.random.default_rng(self._seed_W + input_dim)
            rng_b = np.random.default_rng(self._seed_b + input_dim)
            self._W_dict[input_dim] = rng_W.normal(0, self.sigma, size=(self.k_key, input_dim))
            self._b_dict[input_dim] = rng_b.uniform(0, 2 * math.pi, size=(self.k_key,))

    def project(self, f: np.ndarray) -> np.ndarray:
        """Compute Fourier feature key. Eq. (12.2)."""
        self._ensure_initialized(len(f))
        W = self._W_dict[len(f)]
        b = self._b_dict[len(f)]
        raw = W @ f + b
        cos_val = np.cos(raw)
        scale = math.sqrt(2.0 / self.k_key)
        return scale * cos_val

    def get_params(self) -> dict:
        """Return current W and b matrices (for serialization)."""
        return {
            "W_dict": {str(k): v for k, v in self._W_dict.items()},
            "b_dict": {str(k): v for k, v in self._b_dict.items()},
        }

    def set_params(self,
                   W_dict: dict[str, np.ndarray],
                   b_dict: dict[str, np.ndarray]) -> None:
        """Restore W and b from serialized state."""
        self._W_dict = {int(k): v for k, v in W_dict.items()}
        self._b_dict = {int(k): v for k, v in b_dict.items()}


class MemorySystem:
    """Memory store with all lifecycle operations. Eqs. (12.1)–(12.7).

    Lifecycle (per Architecture §9.2):
      1. Key computation     — make_key(f)                     (Eq. 12.2)
      2. Store decision      — should_store(f, gamma)          (Eq. 12.1)
      3. Retrieval           — retrieve(f)                     (Eq. 12.3)
      4. Augmentation        — augment(f, gamma, matches)      (Eq. 12.7)
      5. Reinforcement       — reinforce(idx, cos_sim)         (Eq. 12.4)
      6. Decay               — decay_all(iteration_tau)        (Eq. 12.5)
      7. Purging             — purge()                         (Eq. 12.6)
      8. Storage             — store_signal(signal)            (Eq. 12.1)

    Capacity: |M| <= max_size (default 200). Evicts lowest-beta entry when exceeded.
    """

    def __init__(self, params: MemoryParams | None = None):
        p = params or MemoryParams()
        self.params = p
        self.rff = RandomFourierFeatures(p)
        self.store: list[MemoryEntry] = []
        self._key_norms: list[float] = []  # cached L2 norms of stored keys
        self._time: int = 0
        self._retrieved_this_iteration: set[int] = set()
        self.W_mem: np.ndarray | None = None
        self._mem_dim: int | None = None

    # ── Key Computation (Eq. 12.2) ──

    def make_key(self, f: np.ndarray) -> np.ndarray:
        return self.rff.project(f)

    # ── Storage Decision (Eq. 12.1) ──

    def should_store(self, f: np.ndarray, gamma: float) -> bool:
        return float(np.linalg.norm(f)) * gamma > self.params.theta_store

    def store_signal(self, signal: LiterarySignal) -> None:
        """Store a signal. Evicts lowest-beta entry if at capacity."""
        key = self.rff.project(signal.f)
        key_norm = float(np.linalg.norm(key))
        entry = MemoryEntry(
            key=key,
            signal=signal,
            beta=1.0,
            storage_tau=signal.tau,
            retrieval_count=0,
            last_retrieved=-1.0,
        )
        if len(self.store) >= self.params.max_size:
            self._evict_lowest_beta()
        self.store.append(entry)
        self._key_norms.append(key_norm)

    def _evict_lowest_beta(self) -> None:
        if not self.store:
            return
        idx = int(np.argmin([e.beta for e in self.store]))
        self.store.pop(idx)
        self._key_norms.pop(idx)

    # ── Retrieval (Eq. 12.3) ──

    def retrieve(self, f: np.ndarray, current_tau: float = -1.0) -> list[MemoryEntry]:
        """Retrieve similar memories via cosine similarity.

        Returns list of MemoryEntry objects whose cosine similarity
        between query key and stored key exceeds theta_retrieve.

        Each retrieved entry is reinforced (Eq. 12.4) and tracked
        for selective decay.
        """
        if not self.store:
            return []
        key = self.rff.project(f)
        query_norm = np.linalg.norm(key)

        n = len(self.store)
        keys_matrix = np.array([e.key for e in self.store])
        stored_norms = np.array(self._key_norms)
        dots = keys_matrix @ key
        denom = np.maximum(query_norm * stored_norms, 1e-15)
        cos_sims = dots / denom

        matches: list[MemoryEntry] = []
        for i in range(n):
            cos_sim = float(cos_sims[i])
            if cos_sim > self.params.theta_retrieve:
                matches.append(self.store[i])
                self.reinforce(i, cos_sim, current_tau)
                self._retrieved_this_iteration.add(i)
        return matches

    @staticmethod
    def _cosine_similarity(k1: np.ndarray, k2: np.ndarray,
                           norm_k1: float | None = None,
                           norm_k2: float | None = None) -> float:
        n1 = norm_k1 if norm_k1 is not None else np.linalg.norm(k1)
        n2 = norm_k2 if norm_k2 is not None else np.linalg.norm(k2)
        denom = max(n1 * n2, 1e-15)
        return float(np.dot(k1, k2) / denom)

    # ── Reinforcement (Eq. 12.4) ──

    def reinforce(self, idx: int, cos_sim: float,
                  current_tau: float = -1.0) -> None:
        """Reinforce memory activation. Eq. (12.4).

        beta = min(1.0, beta + rho * cos(k, k_j))
        """
        entry = self.store[idx]
        new_beta = min(1.0, entry.beta + self.params.rho_reinforce * max(0.0, cos_sim))
        new_count = entry.retrieval_count + 1
        self.store[idx] = MemoryEntry(
            key=entry.key, signal=entry.signal,
            beta=float(new_beta),
            storage_tau=entry.storage_tau,
            retrieval_count=new_count,
            last_retrieved=current_tau if current_tau >= 0 else entry.last_retrieved,
        )

    # ── Decay (Eq. 12.5) — Selective ──

    def decay_all(self, iteration_tau: float = -1.0) -> None:
        """Apply exponential decay to non-retrieved entries. Eq. (12.5).

        Only entries NOT retrieved this iteration are decayed.
        Retrieved entries maintain their activation.
        """
        self._time += 1
        new_store: list[MemoryEntry] = []
        new_norms: list[float] = []
        for i, entry in enumerate(self.store):
            if i in self._retrieved_this_iteration:
                new_store.append(entry)
                new_norms.append(self._key_norms[i])
            else:
                decayed = entry.beta * (1.0 - self.params.delta_decay)
                new_store.append(MemoryEntry(
                    key=entry.key, signal=entry.signal,
                    beta=float(decayed),
                    storage_tau=entry.storage_tau,
                    retrieval_count=entry.retrieval_count,
                    last_retrieved=entry.last_retrieved,
                ))
                new_norms.append(self._key_norms[i])
        self.store = new_store
        self._key_norms = new_norms
        self._retrieved_this_iteration.clear()

    # ── Forgetting (aliasing for decay, useful as separate step) ──

    def forget(self, iteration_tau: float = -1.0) -> None:
        """Alias for decay_all. Reinforces the forgetting semantics of Eq. (12.5)."""
        self.decay_all(iteration_tau)

    # ── Purging (Eq. 12.6) ──

    def purge(self) -> None:
        """Remove entries with beta below forget threshold. Eq. (12.6)."""
        new_store = []
        new_norms: list[float] = []
        for i, e in enumerate(self.store):
            if e.beta >= self.params.theta_forget:
                new_store.append(e)
                new_norms.append(self._key_norms[i])
        self.store = new_store
        self._key_norms = new_norms

    # ── Signal Augmentation (Eq. 12.7) ──

    def augment(self, f: np.ndarray, gamma: float,
                matches: list[MemoryEntry]) -> np.ndarray:
        """Augment current signal with retrieved memories. Eq. (12.7).

        f_aug = f + sum_j beta_j * W_mem * f_j
        """
        self._ensure_mem_dim(len(f))
        delta = np.zeros_like(f, dtype=float)
        for entry in matches:
            min_len = min(len(entry.signal.f), self.W_mem.shape[0])
            proj = self.W_mem[:min_len, :min_len] @ entry.signal.f[:min_len]
            if len(proj) < len(delta):
                proj = np.pad(proj, (0, len(delta) - len(proj)), mode='constant')
            delta += entry.beta * proj[:len(delta)]
        return f + delta

    def _ensure_mem_dim(self, dim: int) -> None:
        if self._mem_dim != dim:
            self.W_mem = np.eye(dim, dtype=float)
            self._mem_dim = dim

    # ── Diagnostics ──

    def get_stats(self) -> dict[str, float | int]:
        """Return diagnostic statistics about the memory store."""
        if not self.store:
            return {
                "size": 0, "max_size": self.params.max_size,
                "avg_beta": 0.0, "min_beta": 0.0, "max_beta": 0.0,
                "total_retrievals": 0, "capacity_pct": 0.0,
            }
        betas = [e.beta for e in self.store]
        total_ret = sum(e.retrieval_count for e in self.store)
        return {
            "size": len(self.store),
            "max_size": self.params.max_size,
            "avg_beta": float(np.mean(betas)),
            "min_beta": float(np.min(betas)),
            "max_beta": float(np.max(betas)),
            "total_retrievals": total_ret,
            "capacity_pct": round(100.0 * len(self.store) / self.params.max_size, 1),
        }

    def get_activations(self) -> list[float]:
        return [e.beta for e in self.store]

    def get_avg_beta(self) -> float:
        if not self.store:
            return 0.0
        return float(np.mean([e.beta for e in self.store]))

    # ── Persistence ──

    def persist(self, path: str) -> None:
        """Save memory state to .npz file.

        Architecture §6.2: memory.npz with entries and activations.
        """
        if not self.store:
            np.savez_compressed(path, empty=True)
            return
        n = len(self.store)
        keys = np.array([e.key for e in self.store], dtype=float)
        betas = np.array([e.beta for e in self.store], dtype=float)
        storage_taus = np.array([e.storage_tau for e in self.store], dtype=float)
        retrieval_counts = np.array([e.retrieval_count for e in self.store], dtype=int)
        last_retrieved = np.array([e.last_retrieved for e in self.store], dtype=float)
        # Serialize LiterarySignals as tuples
        f_dims = [len(e.signal.f) for e in self.store]
        max_f_dim = max(f_dims) if f_dims else 0
        f_arr = np.zeros((n, max_f_dim), dtype=float)
        gamma_arr = np.zeros(n, dtype=float)
        tau_arr = np.zeros(n, dtype=float)
        for i, e in enumerate(self.store):
            f_arr[i, :len(e.signal.f)] = e.signal.f
            gamma_arr[i] = e.signal.gamma
            tau_arr[i] = e.signal.tau
        np.savez_compressed(
            path,
            keys=keys, betas=betas, storage_taus=storage_taus,
            retrieval_counts=retrieval_counts, last_retrieved=last_retrieved,
            f_arr=f_arr, gamma_arr=gamma_arr, tau_arr=tau_arr,
            f_dims=np.array(f_dims, dtype=int),
            **{f"W_dim_{k}": v for k, v in self.rff._W_dict.items()},
            **{f"b_dim_{k}": v for k, v in self.rff._b_dict.items()},
            time=self._time,
        )

    def load(self, path: str) -> bool:
        """Load memory state from .npz file. Returns True on success."""
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=False)
        if "empty" in data:
            self.clear()
            return True
        W_dict = {}
        b_dict = {}
        for key in data:
            if key.startswith("W_dim_"):
                d = key[len("W_dim_"):]
                W_dict[d] = data[key]
            elif key.startswith("b_dim_"):
                d = key[len("b_dim_"):]
                b_dict[d] = data[key]
        self.rff.set_params(W_dict, b_dict)
        self._time = int(data["time"])
        n = len(data["keys"])
        self.store = []
        self._key_norms = []
        for i in range(n):
            kd = int(data["f_dims"][i])
            f = data["f_arr"][i, :kd].copy()
            sig = LiterarySignal(
                f=f,
                gamma=float(data["gamma_arr"][i]),
                tau=float(data["tau_arr"][i]),
                context=None, provenance=[], residues=np.zeros((6, 6)),
                memory_traces=[],
            )
            entry = MemoryEntry(
                key=data["keys"][i].copy(),
                signal=sig,
                beta=float(data["betas"][i]),
                storage_tau=float(data["storage_taus"][i]),
                retrieval_count=int(data["retrieval_counts"][i]),
                last_retrieved=float(data["last_retrieved"][i]),
            )
            self.store.append(entry)
            self._key_norms.append(float(np.linalg.norm(entry.key)))
        return True

    # ── Lifecycle Management ──

    def clear(self) -> None:
        self.store.clear()
        self._key_norms.clear()
        self._time = 0
        self._retrieved_this_iteration.clear()

    def mark_iteration_end(self) -> None:
        """End-of-iteration: decay then purge."""
        self.decay_all()
        self.purge()

    @property
    def size(self) -> int:
        return len(self.store)

    @property
    def is_full(self) -> bool:
        return len(self.store) >= self.params.max_size
