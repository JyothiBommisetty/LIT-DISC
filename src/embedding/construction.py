import numpy as np
from src.models import NarrativeEmbedding, NodeID


class NarrativeEmbeddingConstructor:
    DIM_OFFSET = 72
    RES_OFFSET = 108

    def __init__(self, n_components: int = 12, d: int = 120):
        self.n_components = n_components
        self.d = d
        self.d_per_dim = d // 10

    def construct(self, signals: dict[NodeID, dict[str, "LiterarySignal"]],
                  tree) -> NarrativeEmbedding:
        dims = ["S", "N", "I", "Sy", "E", "Em"]
        dim_vectors = {}
        for d in dims:
            dim_vectors[d] = self._compute_dim_vector(d, signals, tree)

        H_S = self._pad_or_truncate(dim_vectors["S"], self.d_per_dim)
        H_N = self._pad_or_truncate(dim_vectors["N"], self.d_per_dim)
        H_I = self._pad_or_truncate(dim_vectors["I"], self.d_per_dim)
        H_Sy = self._pad_or_truncate(dim_vectors["Sy"], self.d_per_dim)
        H_E = self._pad_or_truncate(dim_vectors["E"], self.d_per_dim)
        H_Em = self._pad_or_truncate(dim_vectors["Em"], self.d_per_dim)

        H_pairs = self._compute_pairs(dim_vectors, self.d_per_dim * 3)
        H_res = self._compute_resonance(signals, tree, self.d_per_dim)

        return NarrativeEmbedding(
            H_S=H_S, H_N=H_N, H_I=H_I,
            H_Sy=H_Sy, H_E=H_E, H_Em=H_Em,
            H_pairs=H_pairs, H_res=H_res, d=self.d,
        )

    def _compute_dim_vector(self, dim: str,
                            signals: dict[NodeID, dict[str, "LiterarySignal"]],
                            tree) -> np.ndarray:
        all_f = []
        for nid, dim_dict in signals.items():
            if dim in dim_dict:
                sig = dim_dict[dim]
                all_f.append(sig.f * sig.gamma)
        if not all_f:
            return np.zeros(self.n_components)
        f_stack = np.array(all_f)
        mean_f = np.mean(f_stack, axis=0)
        max_f = np.max(f_stack, axis=0)
        std_f = np.std(f_stack, axis=0)
        n = len(all_f)
        if n > 1:
            slope = (f_stack[-1] - f_stack[0]) / max(1, n - 1)
        else:
            slope = np.zeros_like(mean_f)
        combined = np.concatenate([mean_f, max_f, std_f, slope])
        if len(combined) >= self.n_components:
            return combined[:self.n_components]
        return np.pad(combined, (0, self.n_components - len(combined)))

    def _compute_pairs(self, dim_vectors: dict[str, np.ndarray],
                       target_size: int) -> np.ndarray:
        dims = ["S", "N", "I", "Sy", "E", "Em"]
        pairs = []
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                di = dims[i]
                dj = dims[j]
                vi = dim_vectors.get(di, np.zeros(self.n_components))
                vj = dim_vectors.get(dj, np.zeros(self.n_components))
                ni = np.linalg.norm(vi)
                nj = np.linalg.norm(vj)
                if ni > 1e-10 and nj > 1e-10:
                    sim = float(np.dot(vi, vj) / (ni * nj))
                else:
                    sim = 0.0
                pairs.append(abs(sim))
        vec = np.array(pairs, dtype=float)
        if len(vec) >= target_size:
            return vec[:target_size]
        return np.pad(vec, (0, target_size - len(vec)))

    def _compute_resonance(self, signals, tree, target_size: int) -> np.ndarray:
        all_levels = [tree.get_level(l) for l in (5, 4, 3)]
        nodes = []
        for level_nodes in all_levels:
            for node in level_nodes:
                nodes.append(node)
        features = []
        for nid, dim_dict in signals.items():
            for d, sig in dim_dict.items():
                features.append(np.linalg.norm(sig.f) * sig.gamma)
        if not features:
            return np.zeros(target_size)
        mean_act = float(np.mean(features))
        std_act = float(np.std(features))
        max_act = float(np.max(features))
        n_tau = len(nodes)
        slope = max_act / max(1, n_tau)

        res_vec = np.array([
            mean_act, std_act, max_act, slope,
            mean_act * max_act, std_act * slope,
            np.sqrt(mean_act + 1e-10), np.sqrt(max_act + 1e-10),
            0.0, 0.0, 0.0, 0.0,
        ], dtype=float)
        if len(res_vec) >= target_size:
            return res_vec[:target_size]
        return np.pad(res_vec, (0, target_size - len(res_vec)))

    def _pad_or_truncate(self, v: np.ndarray, size: int) -> np.ndarray:
        if len(v) >= size:
            return v[:size]
        return np.pad(v, (0, size - len(v)))
