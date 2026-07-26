import numpy as np
from src.models import LiterarySignal, NodeID


class BottomUpPropagator:
    def __init__(self, gamma_threshold: float = 0.3):
        self.gamma_threshold = gamma_threshold

    def propagate(self, tree, signals: dict[NodeID, dict[str, LiterarySignal]],
                  dimensions: list[str] | None = None) -> dict[NodeID, dict[str, LiterarySignal]]:
        result: dict[NodeID, dict[str, LiterarySignal]] = {}
        for nid, dim_dict in signals.items():
            result[nid] = dict(dim_dict)

        dims = dimensions or ["S"]

        for level in range(2, 6):
            level_nodes = tree.get_level(level)
            for node in level_nodes:
                nid = (node.level, node.index)
                child_ids = node.children
                if not child_ids:
                    continue
                dim_signals: dict[str, list[tuple[np.ndarray, float, float]]] = {d: [] for d in dims}
                for cid in child_ids:
                    if cid not in result:
                        continue
                    for d in dims:
                        if d in result[cid]:
                            sig = result[cid][d]
                            if sig.gamma >= self.gamma_threshold:
                                dim_signals[d].append((sig.f, sig.gamma, sig.tau))

                if level == 2:
                    for d in dims:
                        items = dim_signals[d]
                        if not items:
                            continue
                        f_agg, gamma_agg = self._aggregate_weighted(items)
                        sc = self._semantic_coherence(items)
                        gamma_agg = min(1.0, gamma_agg + sc / 2.0)
                        if nid not in result:
                            result[nid] = {}
                        result[nid][d] = LiterarySignal(
                            f=f_agg, gamma=gamma_agg, tau=node.tau,
                            context=None, provenance=[nid],
                            residues=np.zeros((6, 6), dtype=float),
                            memory_traces=[],
                        )

                elif level == 3:
                    for d in dims:
                        items = dim_signals[d]
                        if not items:
                            continue
                        alphas = self._salience_weights(items, beta_sal=2.0)
                        f_agg = sum(a * f for a, (f, g, t) in zip(alphas, items))
                        gamma_agg = float(np.mean([g for _, g, _ in items]))
                        if nid not in result:
                            result[nid] = {}
                        result[nid][d] = LiterarySignal(
                            f=f_agg, gamma=gamma_agg, tau=node.tau,
                            context=None, provenance=[nid],
                            residues=np.zeros((6, 6), dtype=float),
                            memory_traces=[],
                        )

                elif level in (4, 5):
                    for d in dims:
                        items = dim_signals[d]
                        if not items:
                            continue
                        taus = np.array([t for _, _, t in items])
                        f_vals = np.array([f for f, _, _ in items])
                        g_vals = np.array([g for _, g, _ in items])
                        if len(items) > 1:
                            weights = self._spline_weights(taus, node.tau)
                            f_agg = np.sum(weights[:, np.newaxis] * f_vals, axis=0)
                            gamma_agg = float(np.average(g_vals, weights=weights))
                        else:
                            f_agg = f_vals[0]
                            gamma_agg = float(g_vals[0])
                        if nid not in result:
                            result[nid] = {}
                        result[nid][d] = LiterarySignal(
                            f=f_agg, gamma=gamma_agg, tau=node.tau,
                            context=None, provenance=[nid],
                            residues=np.zeros((6, 6), dtype=float),
                            memory_traces=[],
                        )

        return result

    def _aggregate_weighted(self, items: list[tuple[np.ndarray, float, float]]) -> tuple[np.ndarray, float]:
        f_vals = np.array([f for f, g, _ in items])
        g_vals = np.array([g for _, g, _ in items])
        total_g = np.sum(g_vals)
        if total_g < 1e-10:
            f_agg = np.mean(f_vals, axis=0)
            g_agg = 0.0
        else:
            f_agg = np.sum(f_vals * g_vals[:, np.newaxis], axis=0) / total_g
            g_agg = float(np.mean(g_vals))
        return f_agg, g_agg

    def _semantic_coherence(self, items: list[tuple[np.ndarray, float, float]]) -> float:
        if len(items) < 2:
            return 0.0
        f_vals = np.array([f for f, _, _ in items])
        sims = []
        for i in range(len(f_vals)):
            for j in range(i + 1, len(f_vals)):
                fi = f_vals[i]
                fj = f_vals[j]
                ni = np.linalg.norm(fi)
                nj = np.linalg.norm(fj)
                if ni < 1e-10 or nj < 1e-10:
                    sims.append(0.0)
                else:
                    sims.append(float(np.dot(fi, fj) / (ni * nj)))
        return float(np.mean(sims)) if sims else 0.0

    def _salience_weights(self, items: list[tuple[np.ndarray, float, float]],
                          beta_sal: float = 2.0) -> np.ndarray:
        scores = np.array([beta_sal * np.linalg.norm(f) * max(g, 0.1) for f, g, _ in items])
        scores = scores - np.max(scores)
        exp_scores = np.exp(np.clip(scores, -500, 500))
        total = np.sum(exp_scores)
        if total < 1e-10:
            return np.ones(len(items)) / len(items)
        return exp_scores / total

    def _spline_weights(self, taus: np.ndarray, target_tau: float) -> np.ndarray:
        dists = np.abs(taus - target_tau)
        min_dist = np.min(dists)
        if min_dist < 1e-10:
            w = np.zeros(len(taus))
            w[np.argmin(dists)] = 1.0
            return w
        inv_dists = 1.0 / (dists + 1e-10)
        return inv_dists / np.sum(inv_dists)
