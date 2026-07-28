import json
import math
import numpy as np
from pathlib import Path
from src.models import NarrativeEmbedding, NodeID


_DIMS = ["S", "N", "I", "Sy", "E", "Em"]
_VALID_D = {40, 120, 240, 480}


class NarrativeEmbeddingConstructor:
    """Narrative Embedding constructor — HMP-DT Eqs. (14.1)–(14.2).

    Constructs H ∈ ℝ^d by concatenating:
      - 6 per-dimension subspaces  H_d  ∈ ℝ^{d/10}   (Eq. 14.2)
      - 1 pairwise-interaction     H_pairs ∈ ℝ^{3d/10}
      - 1 resonance subspace       H_res  ∈ ℝ^{d/10}

    Eq. 14.2:  H_d = [Mean(f_d), Max(f_d), Slope(f_d),
                       Var(f_d), FFT_low(f_d)]
    """

    def __init__(self, d: int = 120):
        if d not in _VALID_D:
            raise ValueError(
                f"Embedding dimension d={d} not in allowed set {_VALID_D}")
        if d % 10 != 0:
            raise ValueError(f"d={d} must be divisible by 10")
        self.d = d
        self.d_per_dim = d // 10
        self.d_pairs = 3 * d // 10

    # ── Public API ──────────────────────────────────────────────

    def construct(self,
                  signals: dict[NodeID, dict[str, 'LiterarySignal']],
                  tree,
                  resonance_trajectory: list[np.ndarray] | None = None,
                  ) -> NarrativeEmbedding:
        """Build full Narrative Embedding from hierarchical signals.

        Args:
            signals: dict of NodeID → dimension → LiterarySignal
            tree: hierarchy tree with get_level(level)
            resonance_trajectory: list of R vectors from NarrativeResonance.
                                  If None, extracts from signals.

        Returns: NarrativeEmbedding with validated dimensions.
        """
        dim_vectors: dict[str, np.ndarray] = {}
        for d in _DIMS:
            dim_vectors[d] = self._compute_dim_subspace(d, signals, tree)

        H_pairs = self._compute_pairs(dim_vectors)
        H_res = self._compute_resonance_subspace(
            resonance_trajectory, signals, tree)

        emb = NarrativeEmbedding(
            H_S=dim_vectors["S"], H_N=dim_vectors["N"],
            H_I=dim_vectors["I"], H_Sy=dim_vectors["Sy"],
            H_E=dim_vectors["E"], H_Em=dim_vectors["Em"],
            H_pairs=H_pairs, H_res=H_res, d=self.d,
        )
        self.validate(emb)
        return emb

    # ── Per-dimension subspace (Eq. 14.2) ───────────────────────

    def _compute_dim_subspace(self, dim: str,
                              signals: dict,
                              tree) -> np.ndarray:
        """Eq. 14.2: H_d = [Mean(f_d), Max(f_d), Slope(f_d),
                             Var(f_d), FFT_low(f_d)]

        Computes 5 statistics per feature component, then pads to d/10.
        """
        all_features: list[np.ndarray] = []
        for nid, dim_dict in signals.items():
            if dim in dim_dict:
                sig = dim_dict[dim]
                all_features.append(sig.f)
        if not all_features:
            return np.zeros(self.d_per_dim, dtype=float)

        f_stack = np.array(all_features, dtype=float)
        k_d = f_stack.shape[1]
        stats: list[float] = []
        for col in range(k_d):
            vals = f_stack[:, col]
            mu = float(np.mean(vals))
            mx = float(np.max(vals))
            n = len(vals)
            slope = float((vals[-1] - vals[0]) / max(n - 1, 1))
            var_ = float(np.var(vals, ddof=0))
            fft_low = float(
                np.abs(np.fft.rfft(vals)[1])
            ) if n > 2 else 0.0
            stats.extend([mu, mx, slope, var_, fft_low])

        vec = np.array(stats, dtype=float)
        if len(vec) >= self.d_per_dim:
            return vec[:self.d_per_dim]
        return np.pad(vec, (0, self.d_per_dim - len(vec)), mode='constant')

    # ── Pairwise interaction subspace ───────────────────────────

    def _compute_pairs(self,
                       dim_vectors: dict[str, np.ndarray]) -> np.ndarray:
        """Compute pairwise cosine similarities between dimension subspaces.

        6 dimensions → 15 unique pairs → pad to 3d/10.
        """
        pairs: list[float] = []
        for i in range(len(_DIMS)):
            for j in range(i + 1, len(_DIMS)):
                vi = dim_vectors.get(_DIMS[i], np.zeros(self.d_per_dim))
                vj = dim_vectors.get(_DIMS[j], np.zeros(self.d_per_dim))
                ni = float(np.linalg.norm(vi))
                nj = float(np.linalg.norm(vj))
                if ni > 1e-10 and nj > 1e-10:
                    sim = float(np.dot(vi, vj) / (ni * nj))
                else:
                    sim = 0.0
                pairs.append(abs(sim))
        vec = np.array(pairs, dtype=float)
        if len(vec) >= self.d_pairs:
            return vec[:self.d_pairs]
        return np.pad(vec, (0, self.d_pairs - len(vec)), mode='constant')

    # ── Resonance subspace ──────────────────────────────────────

    def _compute_resonance_subspace(
            self,
            trajectory: list[np.ndarray] | None,
            signals: dict,
            tree,
    ) -> np.ndarray:
        """Compute H_res from resonance trajectory or signals.

        Per-component statistics: [mean, max, final, slope, var, fft_low]
        repeated for each component, padded to d/10.
        """
        if trajectory is not None and len(trajectory) >= 2:
            traj = np.array(trajectory, dtype=float)
        else:
            traj = self._extract_resonance_trajectory(signals, tree)

        if traj.shape[0] < 2:
            return np.zeros(self.d_per_dim, dtype=float)

        stats: list[float] = []
        for col in range(traj.shape[1]):
            vals = traj[:, col]
            stats.extend([
                float(np.mean(vals)),
                float(np.max(vals)),
                float(vals[-1]),
                float(vals[-1] - vals[0]),
                float(np.var(vals, ddof=0)),
                float(np.abs(np.fft.rfft(vals)[1]))
                if len(vals) > 2 else 0.0,
            ])
        vec = np.array(stats, dtype=float)
        if len(vec) >= self.d_per_dim:
            return vec[:self.d_per_dim]
        return np.pad(vec, (0, self.d_per_dim - len(vec)), mode='constant')

    @staticmethod
    def _extract_resonance_trajectory(signals: dict, tree) -> np.ndarray:
        """Build a fallback resonance-like trajectory from level-4 signal norms."""
        level4 = tree.get_level(4) if hasattr(tree, 'get_level') else []
        if not level4:
            return np.zeros((0, 7), dtype=float)
        norms: list[np.ndarray] = []
        for node in level4:
            nid = (node.level, node.index)
            vec: list[float] = []
            for d in _DIMS:
                if nid in signals and d in signals[nid]:
                    vec.append(float(np.linalg.norm(signals[nid][d].f)))
                else:
                    vec.append(0.0)
            norms.append(np.array(vec, dtype=float))
        if not norms:
            return np.zeros((0, 7), dtype=float)
        return np.array(norms, dtype=float)

    # ── Dimension validation ────────────────────────────────────

    def validate(self, embedding: NarrativeEmbedding) -> None:
        """Validate all subspace dimensions match configuration. Raises ValueError."""
        errors: list[str] = []
        for dim_name in _DIMS:
            attr = f"H_{dim_name}"
            vec = getattr(embedding, attr, None)
            if vec is None:
                errors.append(f"Missing {attr}")
            elif len(vec) != self.d_per_dim:
                errors.append(
                    f"{attr} len={len(vec)}, expected {self.d_per_dim}")
        for attr, expected_len in [("H_pairs", self.d_pairs),
                                    ("H_res", self.d_per_dim)]:
            vec = getattr(embedding, attr, None)
            if vec is None:
                errors.append(f"Missing {attr}")
            elif len(vec) != expected_len:
                errors.append(
                    f"{attr} len={len(vec)}, expected {expected_len}")
        if embedding.d != self.d:
            errors.append(f"embedding.d={embedding.d}, constructor d={self.d}")
        total = len(embedding.to_vector())
        if total != self.d:
            errors.append(
                f"to_vector() len={total}, expected d={self.d}")

        if errors:
            raise ValueError(
                "NarrativeEmbedding dimension validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors))

    # ── Serialization ───────────────────────────────────────────

    def save(self, embedding: NarrativeEmbedding,
             path: str | Path) -> None:
        """Serialize NarrativeEmbedding to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "H_S": embedding.H_S,
            "H_N": embedding.H_N,
            "H_I": embedding.H_I,
            "H_Sy": embedding.H_Sy,
            "H_E": embedding.H_E,
            "H_Em": embedding.H_Em,
            "H_pairs": embedding.H_pairs,
            "H_res": embedding.H_res,
            "_d": np.array([embedding.d]),
        }
        np.savez_compressed(str(path), **arrays)

    def load(self, path: str | Path) -> NarrativeEmbedding:
        """Deserialize NarrativeEmbedding from .npz file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")
        data = np.load(str(path))
        d = int(data["_d"][0])
        return NarrativeEmbedding(
            H_S=data["H_S"], H_N=data["H_N"],
            H_I=data["H_I"], H_Sy=data["H_Sy"],
            H_E=data["H_E"], H_Em=data["H_Em"],
            H_pairs=data["H_pairs"], H_res=data["H_res"],
            d=d,
        )

    def to_json(self, embedding: NarrativeEmbedding) -> str:
        """Serialize to JSON string for inspection/debugging."""
        obj = {
            "d": embedding.d,
            "d_per_dim": self.d_per_dim,
            "subspaces": {
                "H_S": embedding.H_S.tolist(),
                "H_N": embedding.H_N.tolist(),
                "H_I": embedding.H_I.tolist(),
                "H_Sy": embedding.H_Sy.tolist(),
                "H_E": embedding.H_E.tolist(),
                "H_Em": embedding.H_Em.tolist(),
                "H_pairs": embedding.H_pairs.tolist(),
                "H_res": embedding.H_res.tolist(),
            },
            "stats": {
                "norm": float(np.linalg.norm(embedding.to_vector())),
            },
        }
        return json.dumps(obj, indent=2)
