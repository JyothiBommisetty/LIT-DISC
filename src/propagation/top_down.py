import numpy as np
from src.models import LiterarySignal, NodeID


class TopDownRefiner:
    """Top-down refinement operator. Eq. (10.9).

    f_i^{(ℓ)}[d] ← f_i^{(ℓ)}[d] + β_↓ · Proj_↓(f_parent^{(ℓ+1)}[d])
    """

    def __init__(self, lambda_d: float = 0.15):
        self.lambda_d = lambda_d

    def refine(self, tree, signals: dict[NodeID, dict[str, LiterarySignal]],
               beta_down: float = 0.3) -> dict[NodeID, dict[str, LiterarySignal]]:
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

        # Top-down: level 5 → 4 → 3 → 2 → 1
        for level in range(4, 0, -1):
            level_nodes = tree.get_level(level)
            for node in level_nodes:
                nid = (node.level, node.index)
                parent_id = node.parent
                if parent_id is None:
                    continue
                if nid not in result:
                    continue
                if parent_id not in result:
                    continue
                for d, sig in result[nid].items():
                    if d in result[parent_id]:
                        parent_sig = result[parent_id][d]
                        # Project parent signal down: pad/truncate to child dim
                        child_len = len(sig.f)
                        parent_f = parent_sig.f
                        if len(parent_f) > child_len:
                            projected = parent_f[:child_len]
                        else:
                            projected = np.pad(parent_f, (0, child_len - len(parent_f)),
                                               mode='constant')
                        # Eq. (10.9): f_i ← f_i + β_↓ · Proj_↓(f_parent)
                        refined_f = sig.f + beta_down * projected
                        result[nid][d] = LiterarySignal(
                            f=refined_f, gamma=sig.gamma, tau=sig.tau,
                            context=sig.context, provenance=list(sig.provenance),
                            residues=sig.residues.copy(),
                            memory_traces=list(sig.memory_traces),
                        )
        return result
