from __future__ import annotations

import math

import numpy as np

from src.models import ExplanationTrace, LiterarySignal, NodeID

_DIMS = ["S", "N", "I", "Sy", "E", "Em"]
_DIM_LABELS = {
    "S": "Symbolic",
    "N": "Narrative",
    "I": "Interaction",
    "Sy": "Semantic",
    "E": "Evolution",
    "Em": "Emotion",
}


class SignalLineageGenerator:
    def generate(self, signals: dict[NodeID, dict[str, LiterarySignal]],
                 tree) -> list[dict]:
        lineage = []
        for nid, dim_dict in sorted(signals.items()):
            for dim, sig in dim_dict.items():
                entry = {
                    "node_id": list(nid),
                    "level": nid[0],
                    "index": nid[1],
                    "dimension": dim,
                    "dimension_label": _DIM_LABELS.get(dim, dim),
                    "tau": float(sig.tau),
                    "gamma": float(sig.gamma),
                    "feature_norm": float(np.linalg.norm(sig.f)),
                    "provenance": [list(p) for p in sig.provenance],
                    "memory_trace_count": len(sig.memory_traces or []),
                }
                lineage.append(entry)
        return lineage

    def dimension_trace(self, dim: str,
                        signals: dict[NodeID, dict[str, LiterarySignal]],
                        tree) -> list[dict]:
        trace = []
        for nid, dim_dict in sorted(signals.items()):
            if dim in dim_dict:
                sig = dim_dict[dim]
                trace.append({
                    "node_id": list(nid),
                    "level": nid[0],
                    "index": nid[1],
                    "tau": float(sig.tau),
                    "gamma": float(sig.gamma),
                    "feature_norm": float(np.linalg.norm(sig.f)),
                })
        return trace

    def hierarchy_trace(self, node_id: tuple[int, int],
                        signals: dict[NodeID, dict[str, LiterarySignal]],
                        tree) -> list[dict]:
        hierarchy = []
        current: NodeID | None = node_id
        while current is not None:
            children = []
            if current in signals:
                for dim, sig in signals[current].items():
                    children.append({
                        "dimension": dim,
                        "gamma": float(sig.gamma),
                        "norm": float(np.linalg.norm(sig.f)),
                    })
            hierarchy.append({
                "node_id": list(current),
                "level": current[0],
                "index": current[1],
                "signals": children,
            })
            current = self._find_parent(current, tree)
        return hierarchy

    @staticmethod
    def _find_parent(nid: NodeID, tree) -> NodeID | None:
        level = nid[0]
        if level >= 5:
            return None
        parent_level = level + 1
        parent_nodes = tree.get_level(parent_level) if hasattr(tree, 'get_level') else []
        for node in parent_nodes:
            if hasattr(node, 'children') and nid in node.children:
                return (node.level, node.index)
        return None


class PropagationPathGenerator:
    def generate(self, signals: dict[NodeID, dict[str, LiterarySignal]],
                 tree) -> dict:
        levels = {}
        for level in range(1, 6):
            nodes = tree.get_level(level) if hasattr(tree, 'get_level') else []
            level_data = []
            for node in nodes:
                nid = (node.level, node.index)
                dim_info = {}
                if nid in signals:
                    for dim, sig in signals[nid].items():
                        dim_info[dim] = {
                            "gamma": float(sig.gamma),
                            "norm": float(np.linalg.norm(sig.f)),
                            "tau": float(sig.tau),
                        }
                level_data.append({
                    "node_id": list(nid),
                    "tau": float(node.tau) if hasattr(node, 'tau') else 0.5,
                    "parent": list(node.parent) if hasattr(node, 'parent') and node.parent else None,
                    "children": [list(c) for c in node.children] if hasattr(node, 'children') else [],
                    "dimensions": dim_info,
                })
            levels[f"level_{level}"] = {
                "label": self._level_label(level),
                "node_count": len(nodes),
                "nodes": level_data,
            }
        return {
            "hierarchy": levels,
            "description": self._describe_path(signals, tree),
        }

    def path_summary(self, tree) -> list[dict]:
        steps = []
        for level in range(1, 6):
            nodes = tree.get_level(level) if hasattr(tree, 'get_level') else []
            child_count = sum(
                len(node.children) if hasattr(node, 'children') else 0
                for node in nodes
            )
            steps.append({
                "level": level,
                "label": self._level_label(level),
                "node_count": len(nodes),
                "total_children": child_count,
            })
        return steps

    @staticmethod
    def _level_label(level: int) -> str:
        return {1: "Sentence", 2: "Paragraph", 3: "Chapter",
                4: "Narrative Layer", 5: "Discourse"}.get(level, f"Level {level}")

    @staticmethod
    def _describe_path(signals: dict[NodeID, dict[str, LiterarySignal]],
                       tree) -> str:
        level_counts: dict[int, int] = {}
        for nid in signals:
            lev = nid[0]
            level_counts[lev] = level_counts.get(lev, 0) + 1
        parts = []
        for lev in sorted(level_counts):
            parts.append(f"{level_counts[lev]} {PropagationPathGenerator._level_label(lev).lower()}(s)")
        return " → ".join(parts) if parts else "No signals"


class InteractionHistoryGenerator:
    def generate(self, signals: dict[NodeID, dict[str, LiterarySignal]],
                 tree) -> dict:
        interactions = []
        for nid, dim_dict in sorted(signals.items()):
            dims_present = list(dim_dict.keys())
            if len(dims_present) < 2:
                continue
            for i in range(len(dims_present)):
                for j in range(i + 1, len(dims_present)):
                    d1 = dims_present[i]
                    d2 = dims_present[j]
                    sig1 = dim_dict[d1]
                    sig2 = dim_dict[d2]
                    cos_sim = self._cosine_similarity(sig1.f, sig2.f)
                    interactions.append({
                        "node_id": list(nid),
                        "level": nid[0],
                        "index": nid[1],
                        "dimension_pair": f"{d1}-{d2}",
                        "dimension_labels": [_DIM_LABELS.get(d1, d1), _DIM_LABELS.get(d2, d2)],
                        "cosine_similarity": float(cos_sim),
                        "gamma_1": float(sig1.gamma),
                        "gamma_2": float(sig2.gamma),
                        "tau": float(sig1.tau),
                    })
        return {
            "interactions": interactions,
            "summary": self._summarize(interactions),
        }

    def per_level(self, signals: dict[NodeID, dict[str, LiterarySignal]],
                  tree) -> dict[str, list[dict]]:
        by_level: dict[int, list] = {}
        for nid, dim_dict in signals.items():
            lev = nid[0]
            if lev not in by_level:
                by_level[lev] = []
            dims_present = list(dim_dict.keys())
            if len(dims_present) < 2:
                continue
            for i in range(len(dims_present)):
                for j in range(i + 1, len(dims_present)):
                    d1 = dims_present[i]
                    d2 = dims_present[j]
                    sig1 = dim_dict[d1]
                    sig2 = dim_dict[d2]
                    cos_sim = self._cosine_similarity(sig1.f, sig2.f)
                    by_level[lev].append({
                        "node_id": list(nid),
                        "pair": f"{d1}-{d2}",
                        "cosine_similarity": float(cos_sim),
                    })
        return {f"level_{k}": v for k, v in sorted(by_level.items())}

    @staticmethod
    def _cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
        min_len = min(len(f1), len(f2))
        a, b = f1[:min_len], f2[:min_len]
        n1 = float(np.linalg.norm(a))
        n2 = float(np.linalg.norm(b))
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))

    @staticmethod
    def _summarize(interactions: list[dict]) -> dict:
        if not interactions:
            return {"count": 0, "mean_cosine": 0.0, "max_cosine": 0.0, "min_cosine": 0.0}
        cosines = [i["cosine_similarity"] for i in interactions]
        pair_counts: dict[str, int] = {}
        for i in interactions:
            pair_counts[i["dimension_pair"]] = pair_counts.get(i["dimension_pair"], 0) + 1
        return {
            "count": len(interactions),
            "mean_cosine": float(np.mean(cosines)),
            "max_cosine": float(np.max(cosines)),
            "min_cosine": float(np.min(cosines)),
            "pairs": pair_counts,
        }


class MemoryActivationGenerator:
    def generate(self, memory_store, tau: float = 0.5,
                 top_k: int = 10) -> list[dict]:
        entries = self._get_entries(memory_store)
        activations = []
        for entry in entries:
            act = {
                "key_hash": hash(entry.key.tobytes()) if hasattr(entry, 'key') else 0,
                "beta": float(entry.beta) if hasattr(entry, 'beta') else 0.0,
                "storage_tau": float(entry.storage_tau) if hasattr(entry, 'storage_tau') else 0.0,
                "retrieval_count": int(entry.retrieval_count) if hasattr(entry, 'retrieval_count') else 0,
                "last_retrieved": float(entry.last_retrieved) if hasattr(entry, 'last_retrieved') else -1.0,
                "recency": self._recency(entry, tau),
            }
            activations.append(act)
        activations.sort(key=lambda x: x["beta"], reverse=True)
        return activations[:top_k]

    def summary(self, memory_store) -> dict:
        entries = self._get_entries(memory_store)
        if not entries:
            return {"total_entries": 0, "mean_beta": 0.0, "max_beta": 0.0, "total_retrievals": 0}
        betas = [e.beta for e in entries if hasattr(e, 'beta')]
        retrievals = [e.retrieval_count for e in entries if hasattr(e, 'retrieval_count')]
        return {
            "total_entries": len(entries),
            "mean_beta": float(np.mean(betas)) if betas else 0.0,
            "max_beta": float(np.max(betas)) if betas else 0.0,
            "total_retrievals": int(np.sum(retrievals)) if retrievals else 0,
        }

    @staticmethod
    def _get_entries(memory_store):
        if hasattr(memory_store, '_store'):
            return list(memory_store._store.values())
        if hasattr(memory_store, 'entries'):
            return memory_store.entries
        return []

    @staticmethod
    def _recency(entry, current_tau: float) -> float:
        if not hasattr(entry, 'storage_tau'):
            return 0.0
        age = current_tau - entry.storage_tau
        return max(0.0, 1.0 - age)


class ResonanceContributionGenerator:
    def generate(self, resonance_engine,
                 signals: dict[NodeID, dict[str, LiterarySignal]] | None = None,
                 tree=None) -> dict:
        trajectory = self._get_trajectory(resonance_engine)
        R_current = self._get_R(resonance_engine)
        R_norm = float(np.linalg.norm(R_current)) if R_current is not None else 0.0
        contrib = {
            "resonance_current_norm": R_norm,
            "trajectory_length": len(trajectory),
            "trajectory": self._summarize_trajectory(trajectory),
            "signal_augmentation": self._compute_augmentation(R_current, signals, tree),
        }
        return contrib

    def trajectory(self, resonance_engine) -> list[dict]:
        trajectory = self._get_trajectory(resonance_engine)
        result = []
        for i, R in enumerate(trajectory):
            result.append({
                "step": i,
                "norm": float(np.linalg.norm(R)),
                "mean": float(np.mean(R)) if len(R) > 0 else 0.0,
                "max": float(np.max(R)) if len(R) > 0 else 0.0,
            })
        return result

    def contribution_factor(self, H_res: np.ndarray) -> float:
        norm = float(np.linalg.norm(H_res))
        max_norm = math.sqrt(len(H_res))
        return min(1.0, norm / max(1e-10, max_norm))

    @staticmethod
    def _get_trajectory(resonance_engine):
        if resonance_engine is None:
            return []
        if hasattr(resonance_engine, '_trajectory'):
            raw = resonance_engine._trajectory
            # Handle list of (tau, R) tuples
            if raw and isinstance(raw[0], tuple):
                return [r for _, r in raw]
            return raw
        if hasattr(resonance_engine, 'trajectory'):
            return resonance_engine.trajectory
        return []

    @staticmethod
    def _get_R(resonance_engine):
        if resonance_engine is None:
            return None
        if hasattr(resonance_engine, 'R'):
            return resonance_engine.R
        return None

    @staticmethod
    def _summarize_trajectory(trajectory: list) -> list[dict]:
        if not trajectory:
            return []
        norms = [float(np.linalg.norm(R)) for R in trajectory]
        return [
            {"step": 0, "norm": norms[0]},
            {"step": len(trajectory) // 2, "norm": norms[len(trajectory) // 2] if len(trajectory) > 1 else norms[0]},
            {"step": len(trajectory) - 1, "norm": norms[-1]},
        ]

    @staticmethod
    def _compute_augmentation(R, signals, tree) -> dict:
        if R is None:
            return {"applied": False, "amplification": 0.0}
        R_norm = float(np.linalg.norm(R))
        return {
            "applied": R_norm > 0.01,
            "amplification": R_norm,
            "description": f"Resonance field norm = {R_norm:.4f}" if R_norm > 0.01 else "No significant resonance",
        }


class UncertaintyEvolutionGenerator:
    def generate(self, uncertainty_propagator,
                 signals: dict[NodeID, dict[str, LiterarySignal]],
                 tree) -> list[dict]:
        levels = []
        for level in range(1, 6):
            stats = uncertainty_propagator.get_level_stats(level) if hasattr(
                uncertainty_propagator, 'get_level_stats') else {"count": 0}
            levels.append({
                "level": level,
                "label": self._level_label(level),
                "stats": stats,
                "mean_confidence": self._mean_confidence(uncertainty_propagator, signals, level, tree),
            })
        return levels

    def per_dimension(self, uncertainty_propagator,
                      signals: dict[NodeID, dict[str, LiterarySignal]],
                      dims: list[str] | None = None) -> list[dict]:
        dims = dims or _DIMS
        result = []
        for d in dims:
            variances = []
            for nid in signals:
                v = uncertainty_propagator.get_variance(nid, d) if hasattr(
                    uncertainty_propagator, 'get_variance') else 0.0
                variances.append(v)
            result.append({
                "dimension": d,
                "label": _DIM_LABELS.get(d, d),
                "mean_variance": float(np.mean(variances)) if variances else 0.0,
                "max_variance": float(np.max(variances)) if variances else 0.0,
            })
        return result

    def trace(self, uncertainty_propagator) -> list[dict]:
        if hasattr(uncertainty_propagator, '_level_vars'):
            trace = []
            for level in range(1, 6):
                vars_ = uncertainty_propagator._level_vars.get(level, [])
                if vars_:
                    for i, v in enumerate(vars_):
                        trace.append({
                            "level": level,
                            "step": i,
                            "variance": float(v),
                            "confidence": 1.0 - min(1.0, math.sqrt(max(0.0, v)) / max(
                                uncertainty_propagator.sigma_max, 1e-15)),
                        })
            return trace
        return []

    @staticmethod
    def _level_label(level: int) -> str:
        return {1: "Sentence", 2: "Paragraph", 3: "Chapter",
                4: "Narrative Layer", 5: "Discourse"}.get(level, f"Level {level}")

    @staticmethod
    def _mean_confidence(up, signals, level: int, tree) -> float:
        nodes = tree.get_level(level) if hasattr(tree, 'get_level') else []
        confidences = []
        for node in nodes:
            nid = (node.level, node.index)
            for dim in _DIMS:
                gamma = up.refine_confidence(nid, dim) if hasattr(up, 'refine_confidence') else 0.85
                confidences.append(gamma)
        return float(np.mean(confidences)) if confidences else 0.0


class ContributionBreakdownGenerator:
    def generate(self, ldri_score) -> dict:
        return {
            "ldri": float(ldri_score.ldri),
            "raw": float(ldri_score.raw),
            "gamma_global": float(ldri_score.gamma_global),
            "reliability": float(ldri_score.reliability),
            "components": dict(ldri_score.components),
            "fractions": dict(ldri_score.fractions),
        }

    def by_dimension(self, activations: np.ndarray,
                     dims: list[str] | None = None) -> list[dict]:
        dims = dims or _DIMS
        total = float(np.sum(activations)) + 1e-15
        return [
            {
                "dimension": d,
                "label": _DIM_LABELS.get(d, d),
                "activation": float(activations[i]),
                "contribution": float(activations[i] / total),
            }
            for i, d in enumerate(dims)
        ]

    def textual_summary(self, ldri_score) -> str:
        comp = ldri_score.components
        frac = ldri_score.fractions
        lines = [
            f"LDRI = {ldri_score.ldri:.4f}  (raw = {ldri_score.raw:.4f})",
            f"Confidence Γ = {ldri_score.gamma_global:.4f}",
            f"Reliability = {ldri_score.reliability:.4f}",
            "",
            "Decomposition:",
        ]
        for key in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            val = comp.get(key, 0.0)
            fr = frac.get(key, 0.0)
            lines.append(f"  {key}: {val:.4f}  ({fr:.1%} of LDRI)")
        return "\n".join(lines)


class CompositeExplanationGenerator:
    def __init__(self):
        self.signal_lineage = SignalLineageGenerator()
        self.propagation_path = PropagationPathGenerator()
        self.interaction_history = InteractionHistoryGenerator()
        self.memory_activation = MemoryActivationGenerator()
        self.resonance_contribution = ResonanceContributionGenerator()
        self.uncertainty_evolution = UncertaintyEvolutionGenerator()
        self.contribution_breakdown = ContributionBreakdownGenerator()

    def generate(self, signals: dict[NodeID, dict[str, LiterarySignal]],
                 tree,
                 ldri_score,
                 memory_store=None,
                 resonance_engine=None,
                 uncertainty_propagator=None) -> ExplanationTrace:
        trace = ExplanationTrace(
            signal_lineage=self.signal_lineage.generate(signals, tree),
            propagation_path=self.propagation_path.generate(signals, tree),
            interaction_history=self.interaction_history.generate(signals, tree),
            uncertainty_evolution=self.uncertainty_evolution.generate(
                uncertainty_propagator, signals, tree),
            resonance_contribution=self.resonance_contribution.generate(
                resonance_engine, signals, tree),
            memory_activations=self.memory_activation.generate(memory_store),
        )
        return trace

    def as_dict(self, signals, tree, ldri_score,
                memory_store=None, resonance_engine=None,
                uncertainty_propagator=None) -> dict:
        trace = self.generate(signals, tree, ldri_score,
                              memory_store, resonance_engine, uncertainty_propagator)
        breakdown = self.contribution_breakdown.generate(ldri_score)
        return {
            "signal_lineage": trace.signal_lineage,
            "propagation_path": trace.propagation_path,
            "interaction_history": trace.interaction_history,
            "uncertainty_evolution": trace.uncertainty_evolution,
            "resonance_contribution": trace.resonance_contribution,
            "memory_activations": trace.memory_activations,
            "contribution_breakdown": breakdown,
        }
