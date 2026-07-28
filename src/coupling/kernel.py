import math
import numpy as np
from src.models import LiterarySignal
from src.config import CouplingParams
from src.coupling.weights import AdaptiveWeights, _COMPONENT_NAMES
from src.coupling.components import KernelComponents


class CouplingKernel:
    """Complete coupling kernel κ_{d,d'}(τ). Eq. (11.9).

    κ = Σ_i η_i(τ) · C_i
    """

    def __init__(self, params: CouplingParams | None = None,
                 weights: AdaptiveWeights | None = None,
                 components: KernelComponents | None = None):
        self.weights = weights or AdaptiveWeights(params)
        self.components = components or KernelComponents(params)

    def compute(self, tau: float, f_d: np.ndarray, f_dp: np.ndarray,
                gamma_d: float, gamma_dp: float,
                tau_d: float, tau_dp: float,
                p_d: np.ndarray | None = None,
                p_dp: np.ndarray | None = None,
                v_d: np.ndarray | None = None,
                v_dp: np.ndarray | None = None,
                motif_sims: list[float] | None = None,
                rel_step: int = -1) -> float:
        """Compute κ_{d,d'}(τ). Eq. (11.9).

        C_sim (Eq. 11.3): context similarity
        C_prox (Eq. 11.4): narrative proximity
        C_conf (Eq. 11.5): confidence-weighted similarity
        C_comp (Eq. 11.6): semantic compatibility
        C_sym (Eq. 11.7): symbolic coherence
        C_rel (Eq. 11.8): interaction reliability
        """
        eta = self.weights.compute(tau)

        min_len_f = min(len(f_d), len(f_dp))
        norm_f_d = np.linalg.norm(f_d[:min_len_f])
        norm_f_dp = np.linalg.norm(f_dp[:min_len_f])
        c_sim = self.components.context_similarity(
            f_d, f_dp, p_d, p_dp,
            norm_a=norm_f_d, norm_b=norm_f_dp)
        c_prox = self.components.narrative_proximity(tau_d, tau_dp)
        c_conf = self.components.confidence(
            gamma_d, gamma_dp, f_d, f_dp,
            norm_a=norm_f_d, norm_b=norm_f_dp)

        def _unit(v):
            n = np.linalg.norm(v)
            return v / n if n > 1e-12 else np.ones(len(v)) / math.sqrt(len(v))
        c_comp = self.components.compatibility(
            v_d if v_d is not None else _unit(np.ones_like(f_d)),
            v_dp if v_dp is not None else _unit(np.ones_like(f_dp)))
        c_sym = self.components.symbolic_coherence(motif_sims or [])
        c_rel = self.components.interaction_reliability(rel_step)

        comp_vals = {
            "C_sim": c_sim,
            "C_prox": c_prox,
            "C_conf": c_conf,
            "C_comp": c_comp,
            "C_sym": c_sym,
            "C_rel": c_rel,
        }

        kappa = sum(eta[name] * comp_vals[name] for name in _COMPONENT_NAMES)
        self.components.record_kappa(kappa)
        return float(kappa)

    def compute_full(self, tau: float, signals: dict[str, LiterarySignal],
                     dims: list[str],
                     p_vectors: dict[str, np.ndarray] | None = None,
                     v_vectors: dict[str, np.ndarray] | None = None,
                     motif_sims: dict[tuple[str, str], list[float]] | None = None,
                     rel_step: int = -1) -> dict[tuple[str, str], float]:
        """Compute κ for all dimension pairs (d, d') with d != d'."""
        self.components.reset_history()
        kappas: dict[tuple[str, str], float] = {}
        for d in dims:
            for dp in dims:
                if d >= dp:
                    continue
                sd = signals.get(d)
                sdp = signals.get(dp)
                if sd is None or sdp is None:
                    continue
                kappa = self.compute(
                    tau=tau,
                    f_d=sd.f, f_dp=sdp.f,
                    gamma_d=sd.gamma, gamma_dp=sdp.gamma,
                    tau_d=sd.tau, tau_dp=sdp.tau,
                    p_d=p_vectors.get(d) if p_vectors else None,
                    p_dp=p_vectors.get(dp) if p_vectors else None,
                    v_d=v_vectors.get(d) if v_vectors else None,
                    v_dp=v_vectors.get(dp) if v_vectors else None,
                    motif_sims=motif_sims.get((d, dp)) if motif_sims else None,
                    rel_step=rel_step)
                kappas[(d, dp)] = kappa
        return kappas


class CouplingOperator:
    """Coupling operator Λ and signal update. Eqs. (11.10)–(11.11).

    Λ_{d,d'}(a, b) = κ_{d,d'}(τ) · M_{d,d'} · (a ⊙ b)   (Eq. 11.10)
    f̃_d = f_d + Σ_{d'≠d} Λ_{d,d'}(f_d, f_d')           (Eq. 11.11)

    M_{d,d'} is a diagonal coupling matrix (identity for zero-shot).
    When a and b have different lengths, projection uses min(k_d, k_d').
    """

    def __init__(self, M: np.ndarray | None = None):
        """Initialize with fixed coupling matrix M.

        If M is None, uses identity for every pair (zero-shot).
        Otherwise M should be max_k x max_k where max_k = max(k_d).
        """
        self._M = M

    def _ensure_same_length(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        min_len = min(len(a), len(b))
        return a[:min_len], b[:min_len]

    def apply(self, kappa: float, f_d: np.ndarray, f_dp: np.ndarray,
              M_ddp: np.ndarray | None = None) -> np.ndarray:
        """Λ_{d,d'}(f_d, f_d'). Returns vector added to f_d. Eq. (11.10)."""
        a, b = self._ensure_same_length(f_d, f_dp)
        elem_prod = a * b
        if M_ddp is not None:
            min_len = len(elem_prod)
            if M_ddp.shape[0] >= min_len and M_ddp.shape[1] >= min_len:
                projection = M_ddp[:min_len, :min_len] @ elem_prod
            else:
                projection = elem_prod
        else:
            projection = elem_prod
        delta = kappa * projection
        if len(delta) < len(f_d):
            delta = np.pad(delta, (0, len(f_d) - len(delta)), mode='constant')
        return delta

    def update(self, f_d: np.ndarray, deltas: list[np.ndarray]) -> np.ndarray:
        """f̃_d = f_d + Σ_{d'≠d} Λ_{d,d'}(f_d, f_d'). Eq. (11.11)."""
        result = f_d.copy().astype(float)
        for delta in deltas:
            result += delta[:len(result)]
        return np.clip(_clip_to_domain(result), -2.0, 2.0)


def _clip_to_domain(f: np.ndarray) -> np.ndarray:
    """Ensure features remain bounded per Assumption A6."""
    return np.clip(f, -2.0, 2.0)
