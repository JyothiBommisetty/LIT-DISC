import math
import numpy as np
from src.models import LiterarySignal, NodeID, NarrativeEmbedding
from src.coupling.kernel import CouplingKernel


class UncertaintyPropagator:
    """Uncertainty propagation across the 5-level hierarchy.

    Implements HMP-DT v1.0 §13 — Final Uncertainty Equations:
      - Eq. 13.1:  σ²_total = σ²_e + σ²_s + σ²_i + σ²_t + σ²_m
      - Eq. 13.2:  γ = 1 - min(1.0, σ_total / σ_max)
      - Eq. 13.3:  Sentence → Paragraph variance
      - Eq. 13.4:  Paragraph → Chapter variance
      - Eq. 13.5:  Chapter → Narrative variance
      - Eq. 13.6:  Coupling uncertainty

    Additional capabilities:
      - Covariance tracking between discourse dimensions
      - Confidence refinement from propagated variance
      - Bootstrap confidence intervals for LDRI estimates
    """

    def __init__(self, sigma_max: float = 1.0,
                 sigma_e: float = 0.10,
                 sigma_s: float = 0.10,
                 sigma_i: float = 0.05,
                 sigma_t: float = 0.05,
                 sigma_m: float = 0.05):
        self.sigma_max = sigma_max
        self.sigma_e = sigma_e
        self.sigma_s = sigma_s
        self.sigma_i = sigma_i
        self.sigma_t = sigma_t
        self.sigma_m = sigma_m
        # Variance store: (nid, dim) → σ²
        self._var: dict[tuple[NodeID, str], float] = {}
        # Covariance store: (nid, d1, d2) → Cov[f_d1, f_d2]
        self._cov: dict[tuple[NodeID, str, str], float] = {}
        # Per-level variance lists for downstream queries
        self._level_vars: dict[int, list[float]] = {i: [] for i in range(1, 6)}

    # ── Initialization ──────────────────────────────────────────

    def initialize_from_gamma(self, nid: NodeID, dim: str,
                              gamma: float) -> None:
        """Set initial extraction variance from confidence.

        Inverts Eq. 13.2:  σ_total = σ_max · (1 - γ)
        Then isolates σ²_e by subtracting known uncertainty components.
        """
        sigma_total = self.sigma_max * (1.0 - gamma)
        sigma_sq_total = max(0.0, sigma_total ** 2)
        known = (self.sigma_s ** 2 + self.sigma_i ** 2
                 + self.sigma_t ** 2 + self.sigma_m ** 2)
        sigma_e_sq = max(0.0, sigma_sq_total - known)
        self._var[(nid, dim)] = sigma_e_sq
        self._level_vars[1].append(sigma_sq_total)

    def initialize_from_signals(
            self,
            signals: dict[NodeID, dict[str, LiterarySignal]]
    ) -> None:
        """Initialize all variances from signal gamma values."""
        for nid, dim_dict in signals.items():
            for dim, sig in dim_dict.items():
                self.initialize_from_gamma(nid, dim, sig.gamma)

    # ── Variance access ──────────────────────────────────────────

    def set_variance(self, nid: NodeID, dim: str,
                     sigma_sq: float) -> None:
        self._var[(nid, dim)] = max(0.0, sigma_sq)

    def get_variance(self, nid: NodeID, dim: str) -> float:
        return self._var.get((nid, dim), 0.0)

    def total_variance(self, nid: NodeID, dim: str) -> float:
        """Eq. 13.1:  σ²_total = σ²_e + σ²_s + σ²_i + σ²_t + σ²_m"""
        sigma_e_sq = self._var.get((nid, dim), 0.0)
        return (sigma_e_sq + self.sigma_s ** 2 + self.sigma_i ** 2
                + self.sigma_t ** 2 + self.sigma_m ** 2)

    def total_std(self, nid: NodeID, dim: str) -> float:
        """Standard deviation from total variance."""
        return math.sqrt(max(0.0, self.total_variance(nid, dim)))

    # ── Confidence refinement (Eq. 13.2) ─────────────────────────

    def refine_confidence(self, nid: NodeID, dim: str) -> float:
        """Eq. 13.2: γ = 1 - min(1.0, σ_total / σ_max)"""
        sigma_total = self.total_std(nid, dim)
        return 1.0 - min(1.0, sigma_total / max(self.sigma_max, 1e-15))

    def refine_confidence_from_variance(self, sigma_sq: float) -> float:
        """Refine confidence from a raw variance value. Eq. 13.2."""
        sigma_total = math.sqrt(max(0.0, sigma_sq))
        return 1.0 - min(1.0, sigma_total / max(self.sigma_max, 1e-15))

    # ── Sentence → Paragraph (Eq. 13.3) ─────────────────────────

    def propagate_sentence_to_paragraph(
            self,
            child_ids: list[NodeID],
            nid: NodeID,
            sc: float,
            dims: list[str],
    ) -> dict[str, float]:
        """Eq. 13.3:  σ²(P) = (1/n²) Σ σ²(S_i) + (1 - SC) - 2·SC·σ²_avg

        Returns dict of dimension → refined gamma for the paragraph node.
        """
        gammas: dict[str, float] = {}
        if not child_ids:
            return gammas
        for d in dims:
            sigmas = [self.get_variance(cid, d) for cid in child_ids]
            n = len(sigmas)
            sigma_sq_avg = sum(sigmas) / n
            sigma_sq_P = (sum(sigmas) / n ** 2
                          + (1.0 - sc)
                          - 2.0 * sc * sigma_sq_avg)
            sigma_sq_P = max(0.0, sigma_sq_P)
            self.set_variance(nid, d, sigma_sq_P)
            self._level_vars[2].append(sigma_sq_P)
            gammas[d] = self.refine_confidence_from_variance(sigma_sq_P)
        return gammas

    # ── Paragraph → Chapter (Eq. 13.4) ──────────────────────────

    def propagate_paragraph_to_chapter(
            self,
            child_ids: list[NodeID],
            nid: NodeID,
            alphas: list[float],
            dims: list[str],
    ) -> dict[str, float]:
        """Eq. 13.4:  σ²(C) = Σ α_j² · σ²(P_j) + 0.1 · (1 - max α_j)

        Returns dict of dimension → refined gamma for the chapter node.
        """
        gammas: dict[str, float] = {}
        if not child_ids or not alphas:
            return gammas
        max_alpha = max(alphas)
        for d in dims:
            sigmas = [self.get_variance(cid, d) for cid in child_ids]
            sigma_sq_C = sum(a ** 2 * s
                             for a, s in zip(alphas, sigmas) if s >= 0.0)
            sigma_sq_C += 0.1 * (1.0 - max_alpha)
            sigma_sq_C = max(0.0, sigma_sq_C)
            self.set_variance(nid, d, sigma_sq_C)
            self._level_vars[3].append(sigma_sq_C)
            gammas[d] = self.refine_confidence_from_variance(sigma_sq_C)
        return gammas

    # ── Chapter → Narrative Layer (Eq. 13.5) ────────────────────

    def propagate_chapter_to_narrative(
            self,
            chapters: list[tuple[NodeID, float]],
            nid: NodeID,
            spline_weights: list[float],
            tau_target: float,
            dims: list[str],
    ) -> dict[str, float]:
        """Eq. 13.5:  σ²(τ) = Σ φ_i(τ)² · σ²(C_i) + 0.05 · min_i |τ - τ_i|²

        chapters: list of (chapter_nid, chapter_tau)
        spline_weights: φ_i(τ) interpolation weights, same order as chapters.
        Returns dict of dimension → refined gamma.
        """
        gammas: dict[str, float] = {}
        if not chapters or not spline_weights:
            return gammas
        chapter_taus = [ct for _, ct in chapters]
        min_dist = min(abs(tau_target - t) for t in chapter_taus)

        for d in dims:
            sigmas = [self.get_variance(cid, d) for cid, _ in chapters]
            sigma_sq_N = sum(w ** 2 * s
                             for w, s in zip(spline_weights, sigmas)
                             if s >= 0.0)
            sigma_sq_N += 0.05 * min_dist ** 2
            sigma_sq_N = max(0.0, sigma_sq_N)
            self.set_variance(nid, d, sigma_sq_N)
            self._level_vars[4].append(sigma_sq_N)
            gammas[d] = self.refine_confidence_from_variance(sigma_sq_N)
        return gammas

    # ── Discourse Layer → same as narrative for now ──────────────

    def propagate_narrative_to_discourse(
            self,
            narrative_ids: list[NodeID],
            nid: NodeID,
            dims: list[str],
    ) -> dict[str, float]:
        """Aggregate narrative-layer variances to discourse layer.

        Uses simple mean of child variances (discourse layer pools
        narrative-level signals; the spec defers this to narrative-level
        propagation with top-down refinement).
        """
        gammas: dict[str, float] = {}
        if not narrative_ids:
            return gammas
        for d in dims:
            sigmas = [self.get_variance(cid, d) for cid in narrative_ids]
            sigma_sq_D = sum(sigmas) / max(len(sigmas), 1)
            sigma_sq_D = max(0.0, sigma_sq_D)
            self.set_variance(nid, d, sigma_sq_D)
            self._level_vars[5].append(sigma_sq_D)
            gammas[d] = self.refine_confidence_from_variance(sigma_sq_D)
        return gammas

    # ── Coupling uncertainty (Eq. 13.6) ─────────────────────────

    def coupling_uncertainty(self, kappa: float,
                             var_d: float,
                             var_dp: float) -> float:
        """Eq. 13.6: σ²_i(d,d') = κ² · [Var(f_d) + Var(f_d')]"""
        return kappa ** 2 * (var_d + var_dp)

    # ── Covariance tracking ──────────────────────────────────────

    def update_covariance(self, nid: NodeID,
                          dim_signals: dict[str, np.ndarray]) -> None:
        """Track pairwise covariance between dimension feature vectors.

        Cov[f_d, f_d'] = E[(f_d - μ_d)(f_d' - μ_d')]
        """
        dims = list(dim_signals.keys())
        for i, d1 in enumerate(dims):
            for d2 in dims[i + 1:]:
                f1 = dim_signals[d1]
                f2 = dim_signals[d2]
                min_len = min(len(f1), len(f2))
                a = f1[:min_len]
                b = f2[:min_len]
                cov = float(np.cov(a, b)[0, 1]) if min_len > 1 else 0.0
                self._cov[(nid, d1, d2)] = cov
                self._cov[(nid, d2, d1)] = cov

    def get_covariance(self, nid: NodeID,
                       d1: str, d2: str) -> float:
        """Get covariance between dimensions d1 and d2 at node nid."""
        return self._cov.get((nid, d1, d2), 0.0)

    def get_covariance_matrix(self, nid: NodeID,
                              dims: list[str]) -> np.ndarray:
        """Get covariance matrix for all dimensions at node nid."""
        n = len(dims)
        mat = np.zeros((n, n), dtype=float)
        for i, d1 in enumerate(dims):
            for j, d2 in enumerate(dims):
                if i == j:
                    mat[i, j] = self.get_variance(nid, d1)
                else:
                    mat[i, j] = self.get_covariance(nid, d1, d2)
        return mat

    # ── Full hierarchy propagation ───────────────────────────────

    def propagate_all(self, tree, signals, dims):
        """Run full uncertainty propagation matching the pipeline hierarchy.

        Tracks variances at each level by propagating from the previous
        level's results. The `signals` dict should contain level-1 signals;
        this method creates intermediate tracking keys as it propagates.
        """
        self.initialize_from_signals(signals)
        # Track which nodes have been processed at each level
        processed: dict[int, set[NodeID]] = {1: set(signals.keys())}
        for level in range(2, 6):
            processed[level] = set()
            level_nodes = tree.get_level(level)
            if not level_nodes:
                continue
            for node in level_nodes:
                nid = (node.level, node.index)
                child_ids = node.children
                if not child_ids:
                    continue
                # Only use children that were processed at the previous level
                prev_processed = processed.get(level - 1, set())
                valid_children = [c for c in child_ids if c in prev_processed]
                if not valid_children:
                    continue
                if level == 2:
                    self.propagate_sentence_to_paragraph(
                        valid_children, nid, 0.5, dims)
                    processed[2].add(nid)
                elif level == 3:
                    alphas = [1.0 / len(valid_children)] * len(valid_children)
                    self.propagate_paragraph_to_chapter(
                        valid_children, nid, alphas, dims)
                    processed[3].add(nid)
                elif level == 4:
                    chapters = [(c, 0.5) for c in valid_children]
                    weights = [1.0 / len(valid_children)] * len(valid_children)
                    self.propagate_chapter_to_narrative(
                        chapters, nid, weights, node.tau, dims)
                    processed[4].add(nid)
                elif level == 5:
                    self.propagate_narrative_to_discourse(
                        valid_children, nid, dims)
                    processed[5].add(nid)

    # ── Bootstrap confidence intervals ───────────────────────────

    def bootstrap_ldri(
            self,
            embedding: NarrativeEmbedding,
            n_samples: int = 100,
            noise_scale: float = 0.01,
            rng_seed: int = 42,
    ) -> dict:
        """Bootstrap confidence intervals for LDRI.

        Adds Gaussian noise to each H_d subspace, recomputes LDRI
        components, and returns distribution statistics.

        Returns dict with:
          - ldri_samples: list of raw LDRI values
          - mean: bootstrap mean
          - std_error: bootstrap standard error
          - ci_95: (lower, upper) 95% CI
          - reliability: 1 - min(1.0, std_error / 0.1)
        """
        dim_names = ["S", "N", "I", "Sy", "E", "Em"]
        rng = np.random.default_rng(rng_seed)
        raw = embedding.to_vector()
        d = embedding.d

        samples: list[float] = []
        for _ in range(n_samples):
            noise = rng.normal(0, noise_scale, size=d)
            noisy_vec = raw + noise
            # Recompute signal richness on noisy embedding
            norms = []
            for dn in dim_names:
                Hd = getattr(embedding, f"H_{dn}", None)
                if Hd is None or len(Hd) == 0:
                    norms.append(0.0)
                else:
                    noise_d = rng.normal(0, noise_scale, size=len(Hd))
                    perturbed = Hd + noise_d
                    norm = float(np.linalg.norm(perturbed))
                    a_d = 1.0 - math.exp(-norm ** 2 / (2.0 * 0.3 ** 2))
                    norms.append(a_d)
            R = float(np.linalg.norm(norms) / math.sqrt(6.0))
            samples.append(R)

        mean_val = float(np.mean(samples)) if samples else 0.0
        std_err = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        lo = max(0.0, mean_val - 1.96 * std_err)
        hi = min(1.0, mean_val + 1.96 * std_err)
        rel = 1.0 - min(1.0, std_err / 0.1)

        return {
            "ldri_samples": samples,
            "mean": mean_val,
            "std_error": std_err,
            "ci_95": (lo, hi),
            "reliability": rel,
        }

    def compute_confidence_interval(
            self, value: float, std_error: float,
            level: float = 0.95) -> tuple[float, float]:
        """Confidence interval using z-score.

        Default z=1.96 for 95% CI. Clamps to [0, 1].
        """
        z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_map.get(level, 1.96)
        margin = z * std_error
        return (max(0.0, value - margin), min(1.0, value + margin))

    # ── Diagnostics ──────────────────────────────────────────────

    def get_level_stats(self, level: int) -> dict:
        """Summary statistics for variances at a given hierarchy level."""
        vars_ = self._level_vars.get(level, [])
        if not vars_:
            return {"count": 0, "mean": 0.0, "std": 0.0,
                    "min": 0.0, "max": 0.0}
        arr = np.array(vars_)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def clear(self) -> None:
        """Reset all stored variances and covariances."""
        self._var.clear()
        self._cov.clear()
        self._level_vars = {i: [] for i in range(1, 6)}
