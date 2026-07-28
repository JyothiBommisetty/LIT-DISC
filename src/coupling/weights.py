import numpy as np
from src.config import CouplingParams, CouplingComponentParams


_COMPONENT_NAMES = ["C_sim", "C_prox", "C_conf", "C_comp", "C_sym", "C_rel"]


class AdaptiveWeights:
    """Adaptive coupling weights η_i(τ). Eqs. (11.1)–(11.2).

    η̃_i(τ) = α_i + β_i · ½[1 + cos(2π/ω_i · (τ - μ_i))] · 𝕀[|τ-μ_i| ≤ ω_i/2]

    η_i(τ) = η̃_i(τ) / Σ_j η̃_j(τ)
    """

    def __init__(self, params: CouplingParams | None = None):
        p = params or CouplingParams()
        self.components: dict[str, CouplingComponentParams] = {
            "C_sim": p.C_sim,
            "C_prox": p.C_prox,
            "C_conf": p.C_conf,
            "C_comp": p.C_comp,
            "C_sym": p.C_sym,
            "C_rel": p.C_rel,
        }

    def compute_raw(self, tau: float) -> dict[str, float]:
        """Compute unnormalized η̃_i(τ) for each component. Eq. (11.2)."""
        result: dict[str, float] = {}
        for name, cp in self.components.items():
            dist = abs(tau - cp.mu)
            half_width = cp.omega / 2.0
            if dist <= half_width:
                arg = 2.0 * np.pi * (tau - cp.mu) / cp.omega
                cos_term = np.cos(arg)
                window = 0.5 * (1.0 + cos_term)
                eta_tilde = cp.alpha + cp.beta_i * window
            else:
                eta_tilde = cp.alpha
            result[name] = float(eta_tilde)
        return result

    def compute(self, tau: float) -> dict[str, float]:
        """Compute normalized η_i(τ). Eq. (11.1)."""
        raw = self.compute_raw(tau)
        total = sum(raw.values())
        if total < 1e-15:
            n = len(raw)
            return {k: 1.0 / n for k in raw}
        return {k: float(v / total) for k, v in raw.items()}

    def compute_array(self, tau: float) -> np.ndarray:
        """Return normalized weights as numpy array in component order."""
        eta = self.compute(tau)
        return np.array([eta[c] for c in _COMPONENT_NAMES], dtype=float)
