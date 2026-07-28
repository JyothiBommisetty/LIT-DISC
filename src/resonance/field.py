import numpy as np


class NarrativeResonance:
    """Narrative resonance field — HMP-DT Eqs. (12.8)–(12.9).

    Recursive formulation:
      R(τ_i) = R(τ_{i-1}) · e^{-Δτ/λ} + H(τ_{i-1}) · Δτ       Eq. (12.8)
      f_res(τ) = f(τ) + η_res · R(τ)                              Eq. (12.9)

    Properties:
      - Recursive: O(1) per-step update (total O(m) vs O(m²) for direct integral)
      - Long-range influence: e^{-|τ-τ'|/λ} decay for past narrative positions
      - Numerical approximation: rectangle rule with O(Δτ²) per-step error
    """

    def __init__(self, lambda_decay: float = 0.3,
                 eta_resonance: float = 0.2):
        self.lambda_decay = lambda_decay
        self.eta_resonance = eta_resonance
        self.R: np.ndarray | None = None
        self._prev_H: np.ndarray | None = None
        self._current_tau: float = 0.0
        self._trajectory: list[tuple[float, np.ndarray]] = []
        self._n_updates: int = 0

    def reset(self, dim: int, initial_tau: float = 0.0) -> None:
        """Reset resonance field to zero vector of given dimension."""
        self.R = np.zeros(dim, dtype=float)
        self._prev_H = None
        self._current_tau = initial_tau
        self._trajectory = [(initial_tau, self.R.copy())]
        self._n_updates = 0

    def update(self, H: np.ndarray, delta_tau: float) -> np.ndarray:
        """Recursive resonance update. Eq. (12.8).

        R(τ_i) = R(τ_{i-1}) · e^{-Δτ/λ} + H(τ_{i-1}) · Δτ

        The first call stores H as H(τ₀) and returns R = 0.
        The second call begins the recursion with H(τ₀).
        """
        if self.R is None:
            self.reset(len(H))
        if self._prev_H is None:
            self._prev_H = H.copy()
            self._current_tau += delta_tau
            self._n_updates += 1
            return self.R.copy()
        lam = max(self.lambda_decay, 1e-15)
        decay = np.exp(-delta_tau / lam)
        self.R = self.R * decay + self._prev_H * delta_tau
        self._prev_H = H.copy()
        self._current_tau += delta_tau
        self._trajectory.append((self._current_tau, self.R.copy()))
        self._n_updates += 1
        return self.R.copy()

    def augment(self, f: np.ndarray) -> np.ndarray:
        """Resonance-augmented signal. Eq. (12.9).

        f_res(τ) = f(τ) + η_res · R(τ)

        If R is shorter than f, pads R with zeros.
        If R is longer, slices R to match f length.
        """
        if self.R is None:
            return f.copy()
        r = self.R[:len(f)]
        if len(r) < len(f):
            r = np.pad(r, (0, len(f) - len(r)), mode='constant')
        return f + self.eta_resonance * r

    def influence_weight(self, tau_prime: float) -> float:
        """Long-range influence weight.

        influence(τ') = exp(-|τ_current - τ'| / λ)

        Returns how much the narrative state at τ' contributes to
        the current resonance field.
        """
        return float(np.exp(
            -abs(self._current_tau - tau_prime) / max(self.lambda_decay, 1e-15)
        ))

    def effective_horizon(self, threshold: float = 0.01) -> float:
        """Distance in τ at which influence falls below threshold.

        τ_horizon = -λ · ln(threshold)

        With λ=0.3, threshold=0.01: horizon ≈ 1.38 (> 1.0, so full
        narrative span is covered).
        """
        return -self.lambda_decay * np.log(max(threshold, 1e-15))

    @staticmethod
    def numerical_error_bound(delta_tau: float,
                               max_curvature: float = 1.0) -> float:
        """Rectangle rule error bound per step.

        |R_numerical - R_exact| ≤ max_curvature · Δτ² / 2  =  O(Δτ²)

        Args:
            delta_tau: Step size.
            max_curvature: Max second derivative of integrand.
                           Default 1.0 for normalized signals in [0,1].
        """
        return max_curvature * delta_tau ** 2 / 2.0

    def get_field(self) -> np.ndarray:
        """Current resonance field vector R(τ_current)."""
        if self.R is None:
            return np.array([], dtype=float)
        return self.R.copy()

    def get_current_tau(self) -> float:
        """Current narrative position τ."""
        return self._current_tau

    def get_trajectory(self) -> list[tuple[float, np.ndarray]]:
        """List of (tau, R) tuples for all updates."""
        return list(self._trajectory)

    def get_n_updates(self) -> int:
        """Number of update steps performed."""
        return self._n_updates

    def summary_statistics(self, d: int = 12) -> np.ndarray:
        """Summary statistics from trajectory. For H_res embedding (Eq. 14.1).

        Returns d-dim array with per-component:
          [mean, max, final, slope, var, fft_low]  repeated
        """
        if len(self._trajectory) < 2:
            return np.zeros(d, dtype=float)
        traj = np.array([r for _, r in self._trajectory])
        stats: list[float] = []
        for col in range(traj.shape[1]):
            vals = traj[:, col]
            stats.extend([
                float(np.mean(vals)),
                float(np.max(vals)),
                float(vals[-1]),
                float(vals[-1] - vals[0]),
                float(np.var(vals)),
                float(np.abs(np.fft.rfft(vals)[1]))
                if len(vals) > 2 else 0.0,
            ])
        if not stats:
            return np.zeros(d, dtype=float)
        arr = np.array(stats, dtype=float)
        if len(arr) >= d:
            return arr[:d]
        return np.pad(arr, (0, d - len(arr)), mode='constant')
