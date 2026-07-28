import numpy as np
from src.config import ResonanceParams


class RecursiveResonance:
    """Recursive resonance field. Eqs. (12.8)–(12.9).

    R(τ_i) = R(τ_{i-1}) · exp(-Δτ/λ) + H(τ_{i-1}) · Δτ      (Eq. 12.8)
    f_res(τ) = f(τ) + η_res · R(τ)                             (Eq. 12.9)

    The update uses the PREVIOUS H from the prior call, per the algorithm §6.5:
      R[i] = R[i-1] · exp(-Δτ/λ) + H(τ_{i-1}) · Δτ
    """

    def __init__(self, params: ResonanceParams | None = None):
        p = params or ResonanceParams()
        self.lambda_decay = p.lambda_decay
        self.eta_resonance = p.eta_resonance
        self.R: np.ndarray | None = None
        self._prev_H: np.ndarray | None = None
        self._field_dim: int | None = None
        self._trajectory: list[np.ndarray] = []

    def reset(self, dim: int) -> None:
        """Reset resonance field to zero vector of given dimension."""
        self.R = np.zeros(dim, dtype=float)
        self._prev_H = None
        self._field_dim = dim
        self._trajectory = [self.R.copy()]

    def update(self, H: np.ndarray, delta_tau: float) -> np.ndarray:
        """Recursive resonance update. Eq. (12.8).

        R_i = R_{i-1} · exp(-Δτ/λ) + H(τ_{i-1}) · Δτ

        The first call uses H as H(τ_0) to compute R(τ_1).
        """
        if self.R is None:
            self.reset(len(H))
        if self._prev_H is None:
            self._prev_H = H.copy()
            return self.R.copy()
        decay = np.exp(-delta_tau / max(self.lambda_decay, 1e-15))
        self.R = self.R * decay + self._prev_H * delta_tau
        self._prev_H = H.copy()
        self._trajectory.append(self.R.copy())
        return self.R.copy()

    def augment(self, f: np.ndarray, tau: float | None = None) -> np.ndarray:
        """Resonance-augmented signal. Eq. (12.9).

        f_res(τ) = f(τ) + η_res · R(τ)
        """
        if self.R is None:
            return f.copy()
        return f + self.eta_resonance * self.R[:len(f)]

    def get_field(self) -> np.ndarray:
        """Current resonance field vector."""
        if self.R is None:
            return np.array([], dtype=float)
        return self.R.copy()

    def get_trajectory(self) -> list[np.ndarray]:
        """Full resonance field trajectory."""
        return list(self._trajectory)

    def summary_statistics(self, d: int) -> np.ndarray:
        """Compute summary statistics of full resonance trajectory. Used for H_res.

        Returns 12-d vector: [mean, max, final, slope, var, fft_low] padded to d.
        """
        if len(self._trajectory) < 2:
            return np.zeros(d, dtype=float)
        traj = np.array(self._trajectory)
        stats = []
        for col in range(traj.shape[1]):
            vals = traj[:, col]
            stats.extend([
                float(np.mean(vals)),
                float(np.max(vals)),
                float(vals[-1]),
                float(vals[-1] - vals[0]),
                float(np.var(vals)),
                float(np.abs(np.fft.rfft(vals)[1]) if len(vals) > 2 else 0.0),
            ])
        if not stats:
            return np.zeros(d, dtype=float)
        arr = np.array(stats, dtype=float)
        if len(arr) >= d:
            return arr[:d]
        return np.pad(arr, (0, d - len(arr)), mode='constant')
