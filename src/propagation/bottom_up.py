import numpy as np
from src.models import LiterarySignal, NodeID
from src.coupling.kernel import CouplingKernel, CouplingOperator
from src.coupling.memory import MemorySystem
from src.resonance import NarrativeResonance
from src.uncertainty import UncertaintyPropagator


_DIMS = ["S", "N", "I", "Sy", "E", "Em"]


class BottomUpPropagator:
    """Bottom-up signal propagation with horizontal coupling.

    Integrates vertical aggregation (Eqs. 10.2–10.7) with
    horizontal coupling (Eqs. 11.9–11.11), memory (Eqs. 12.1–12.7),
    resonance (Eqs. 12.8–12.9), and uncertainty propagation (§13).
    """

    def __init__(self, gamma_threshold: float = 0.3,
                 coupling_kernel: CouplingKernel | None = None,
                 coupling_operator: CouplingOperator | None = None,
                 memory: MemorySystem | None = None,
                 resonance: NarrativeResonance | None = None,
                 uncertainty: UncertaintyPropagator | None = None):
        self.gamma_threshold = gamma_threshold
        self.kernel = coupling_kernel or CouplingKernel()
        self.operator = coupling_operator or CouplingOperator()
        self.memory = memory or MemorySystem()
        self.resonance = resonance or NarrativeResonance()
        self.uncertainty = uncertainty or UncertaintyPropagator()

    def propagate(self, tree, signals: dict[NodeID, dict[str, LiterarySignal]],
                  dimensions: list[str] | None = None,
                  coupling_tau: float | None = None,
                  refine_iter: int = 1) -> dict[NodeID, dict[str, LiterarySignal]]:
        result: dict[NodeID, dict[str, LiterarySignal]] = {}
        for nid, dim_dict in signals.items():
            result[nid] = {}
            for d, sig in dim_dict.items():
                result[nid][d] = LiterarySignal(
                    f=sig.f.copy(), gamma=sig.gamma, tau=sig.tau,
                    context=sig.context, provenance=list(sig.provenance),
                    residues=sig.residues.copy(),
                    memory_traces=list(sig.memory_traces),
                )

        dims = dimensions or _DIMS
        tau = coupling_tau if coupling_tau is not None else 0.5

        # Initialize uncertainty from input signals
        self.uncertainty.initialize_from_signals(signals)

        for level in range(2, 6):
            level_nodes = tree.get_level(level)
            for node in level_nodes:
                nid = (node.level, node.index)
                child_ids = node.children
                if not child_ids:
                    continue
                dim_signals: dict[str, list[tuple[np.ndarray, float, float]]] = {d: [] for d in dims}

                # Collect child signals, apply memory augmentation and horizontal coupling
                for cid in child_ids:
                    if cid not in result:
                        continue
                    child_dim_signals: dict[str, LiterarySignal] = {}
                    for d in dims:
                        if d in result[cid]:
                            sig = result[cid][d]
                            child_dim_signals[d] = sig

                    if not child_dim_signals:
                        continue

                    # Apply memory retrieval + augmentation (Eqs. 12.3, 12.7)
                    aug_signals: dict[str, np.ndarray] = {}
                    new_gammas: dict[str, float] = {}
                    for d, sig in child_dim_signals.items():
                        matches = self.memory.retrieve(sig.f)
                        if matches:
                            f_aug = self.memory.augment(sig.f, sig.gamma, matches)
                            aug_signals[d] = f_aug
                            new_gammas[d] = sig.gamma
                        else:
                            aug_signals[d] = sig.f.copy()
                            new_gammas[d] = sig.gamma

                    # Apply horizontal coupling (Eqs. 11.9–11.11)
                    coupled: dict[str, np.ndarray] = {}
                    for d in dims:
                        if d not in aug_signals:
                            continue
                        f_d = aug_signals[d]
                        deltas: list[np.ndarray] = []
                        for dp in dims:
                            if dp == d or dp not in aug_signals:
                                continue
                            sd = child_dim_signals[d]
                            sdp = child_dim_signals[dp]
                            kappa = self.kernel.compute(
                                tau=tau,
                                f_d=f_d, f_dp=aug_signals[dp],
                                gamma_d=new_gammas[d],
                                gamma_dp=new_gammas[dp],
                                tau_d=sd.tau, tau_dp=sdp.tau)
                            delta = self.operator.apply(kappa, f_d, aug_signals[dp])
                            deltas.append(delta)
                        coupled[d] = self.operator.update(f_d, deltas)

                    for d in dims:
                        if d not in coupled:
                            continue
                        sig = child_dim_signals[d]
                        f_coupled = coupled[d]
                        gamma_mem = new_gammas[d]

                        # Memory storage check (Eq. 12.1)
                        if self.memory.should_store(sig.f, sig.gamma):
                            self.memory.store_signal(sig)

                        if sig.gamma >= self.gamma_threshold:
                            dim_signals[d].append((f_coupled, gamma_mem, sig.tau))

                # Vertical aggregation per level
                if level == 2:
                    for d in dims:
                        items = dim_signals[d]
                        if not items:
                            continue
                        f_agg, gamma_agg = self._aggregate_weighted(items)
                        sc = self._semantic_coherence(items)
                        gamma_agg = min(1.0, gamma_agg + sc / 2.0)
                        # Uncertainty propagation: Sentence → Paragraph (Eq. 13.3)
                        self.uncertainty.propagate_sentence_to_paragraph(
                            child_ids, nid, sc, dims)
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
                    # Uncertainty propagation: Paragraph → Chapter (Eq. 13.4)
                    alphas_list = self._salience_weights(
                        dim_signals[dims[0]] if dim_signals[dims[0]] else [],
                        beta_sal=2.0)
                    if len(alphas_list) == len(child_ids):
                        self.uncertainty.propagate_paragraph_to_chapter(
                            child_ids, nid, list(alphas_list), dims)

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

                        # Apply resonance augmentation at narrative/discourse level (Eq. 12.9)
                        if self.resonance.R is not None:
                            f_agg = self.resonance.augment(f_agg)

                        if nid not in result:
                            result[nid] = {}
                        result[nid][d] = LiterarySignal(
                            f=f_agg, gamma=gamma_agg, tau=node.tau,
                            context=None, provenance=[nid],
                            residues=np.zeros((6, 6), dtype=float),
                            memory_traces=[],
                        )
                    # Uncertainty propagation: Chapter → Narrative / Narrative → Discourse
                    chapters_with_taus = [
                        (cid, result[cid][dims[0]].tau if cid in result
                         and dims[0] in result[cid] else 0.5)
                        for cid in child_ids if cid in result
                    ]
                    chapter_ids = [c for c, _ in chapters_with_taus]
                    taus_arr = np.array([t for _, t in chapters_with_taus])
                    spline_w = self._spline_weights(taus_arr, node.tau)
                    if len(chapter_ids) > 0 and len(spline_w) == len(chapter_ids):
                        if level == 4:
                            self.uncertainty.propagate_chapter_to_narrative(
                                chapters_with_taus, nid,
                                list(spline_w), node.tau, dims)
                        else:
                            self.uncertainty.propagate_narrative_to_discourse(
                                chapter_ids, nid, dims)

        return result

    def _aggregate_weighted(self, items: list[tuple[np.ndarray, float, float]]) -> tuple[np.ndarray, float]:
        f_vals = np.array([f for f, g, _ in items], dtype=float)
        g_vals = np.array([g for _, g, _ in items], dtype=float)
        total_g = np.sum(g_vals)
        if total_g < 1e-10:
            f_agg = np.mean(f_vals, axis=0)
            g_agg = 0.0
        else:
            f_agg = np.sum(f_vals * g_vals[:, np.newaxis], axis=0) / total_g
            g_agg = float(np.mean(g_vals))
        return f_agg, g_agg

    def _semantic_coherence(self, items: list[tuple[np.ndarray, float, float]]) -> float:
        n = len(items)
        if n < 2:
            return 0.0
        if n > 200:
            import random
            items = random.sample(items, 200)
            n = 200
        f_vals = np.array([f for f, _, _ in items], dtype=float)
        norms = np.linalg.norm(f_vals, axis=1)
        norms[norms < 1e-10] = 1.0
        normalized = f_vals / norms[:, np.newaxis]
        sim_matrix = normalized @ normalized.T
        triu = np.triu_indices(n, k=1)
        sims = sim_matrix[triu]
        return float(np.mean(sims)) if len(sims) > 0 else 0.0

    def _salience_weights(self, items: list[tuple[np.ndarray, float, float]],
                          beta_sal: float = 2.0) -> np.ndarray:
        norms = np.array([np.linalg.norm(f) for f, g, _ in items], dtype=float)
        gammas = np.array([max(g, 0.1) for _, g, _ in items], dtype=float)
        scores = beta_sal * norms * gammas
        scores = scores - np.max(scores)
        exp_scores = np.exp(np.clip(scores, -500, 500))
        total = np.sum(exp_scores)
        if total < 1e-10:
            return np.ones(len(items)) / len(items)
        return exp_scores / total

    def _spline_weights(self, taus: np.ndarray, target_tau: float) -> np.ndarray:
        dists = np.abs(taus - target_tau)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 1e-10:
            w = np.zeros(len(taus))
            w[min_idx] = 1.0
            return w
        inv_dists = 1.0 / (dists + 1e-10)
        return inv_dists / np.sum(inv_dists)
