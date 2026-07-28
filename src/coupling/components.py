import math
import numpy as np
from src.config import CouplingParams


class KernelComponents:
    """Individual kernel components C_i for literary signal coupling.

    Eqs. (11.3) – (11.8).
    """

    def __init__(self, params: CouplingParams | None = None):
        p = params or CouplingParams()
        self.sigma_tau = p.sigma_tau
        self._history: list[float] = []
        self._history_sum: float = 0.0

    def reset_history(self) -> None:
        """Clear coupling history for C_rel."""
        self._history.clear()
        self._history_sum = 0.0

    def record_kappa(self, kappa: float) -> None:
        """Record a coupling value for future C_rel computation."""
        self._history.append(kappa)
        self._history_sum += kappa

    def context_similarity(self, f_d: np.ndarray, f_dp: np.ndarray,
                           p_d: np.ndarray | None = None,
                           p_dp: np.ndarray | None = None,
                           norm_a: float | None = None,
                           norm_b: float | None = None) -> float:
        """C_sim = cos(f_d ⊕ p_d, f_d' ⊕ p_d'). Eq. (11.3).

        Handles different-length vectors by computing cosine on the
        min-dimensional intersection.
        """
        a = f_d if p_d is None else np.concatenate([f_d, p_d])
        b = f_dp if p_dp is None else np.concatenate([f_dp, p_dp])
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        na = np.linalg.norm(a) if norm_a is None else norm_a
        nb = np.linalg.norm(b) if norm_b is None else norm_b
        if na < 1e-12 or nb < 1e-12:
            return 1.0
        return float(np.dot(a, b) / (na * nb))

    def narrative_proximity(self, tau_d: float, tau_dp: float) -> float:
        """C_prox = exp(-|τ_d - τ_d'| / σ_τ), σ_τ = 0.1. Eq. (11.4)."""
        return float(math.exp(-abs(tau_d - tau_dp) / self.sigma_tau))

    def confidence(self, gamma_d: float, gamma_dp: float,
                   f_d: np.ndarray, f_dp: np.ndarray,
                   norm_a: float | None = None,
                   norm_b: float | None = None) -> float:
        """C_conf = γ_d · γ_d' · sim(f_d, f_d'). Eq. (11.5).

        Handles different-length vectors by computing on min dimension.
        """
        min_len = min(len(f_d), len(f_dp))
        f_d, f_dp = f_d[:min_len], f_dp[:min_len]
        nd = np.linalg.norm(f_d) if norm_a is None else norm_a
        ndp = np.linalg.norm(f_dp) if norm_b is None else norm_b
        if nd < 1e-12 or ndp < 1e-12:
            sim = 0.0
        else:
            sim = float(np.dot(f_d, f_dp) / (nd * ndp))
        sim = max(0.0, min(1.0, sim))
        return float(gamma_d * gamma_dp * sim)

    def compatibility(self, v_d: np.ndarray, v_dp: np.ndarray) -> float:
        """C_comp = (1 + v_d^T v_d') / 2. Eq. (11.6).

        Handles different-length vectors by projecting to min dimension.
        """
        min_len = min(len(v_d), len(v_dp))
        inner = float(np.dot(v_d[:min_len], v_dp[:min_len]))
        return float((1.0 + inner) / 2.0)

    def symbolic_coherence(self, motif_sims: list[float]) -> float:
        """C_sym = 1/(|T|+1) · Σ_{t∈T} sim(f_d^t, f_d'^t). Eq. (11.7).

        Args:
            motif_sims: list of cosine similarities for shared motif types.
                        Empty list if no shared motifs (returns 0).
        """
        if not motif_sims:
            return 0.0
        return float(sum(motif_sims) / (len(motif_sims) + 1.0))

    def interaction_reliability(self, t: int = -1) -> float:
        """C_rel^(t) = (1/t) · Σ_{s=1}^{t-1} κ_{d,d'}^{(s)}. Eq. (11.8).

        Args:
            t: current coupling step (1-indexed). If t <= 1, returns the
                running mean of all recorded kappa values (default behavior).
        """
        n = len(self._history)
        if n == 0:
            return 0.5
        if t <= 1 or t - 1 >= n:
            return float(self._history_sum / n)
        recent = self._history[:t - 1]
        if not recent:
            return 0.5
        return float(sum(recent) / len(recent))
