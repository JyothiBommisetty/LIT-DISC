import numpy as np
from src.models import HierarchyNode, NodeID


class HierarchyTree:
    def __init__(self):
        self._nodes: dict[NodeID, HierarchyNode] = {}

    def build(self, sentences: list[str], paragraph_groups: list[list[int]],
              chapter_groups: list[list[int]]) -> None:
        self._nodes.clear()
        n_sent = len(sentences)
        n_par = len(paragraph_groups)
        n_ch = len(chapter_groups)
        tau_sents = self._compute_tau(n_sent)

        for i in range(n_sent):
            nid = (1, i)
            self._nodes[nid] = HierarchyNode(
                level=1, index=i, parent=None, children=[], tau=float(tau_sents[i]),
            )

        for p_idx, sents_in_par in enumerate(paragraph_groups):
            nid = (2, p_idx)
            avg_tau = float(np.mean([tau_sents[s] for s in sents_in_par])) if sents_in_par else float(p_idx) / max(1, n_par)
            children = [(1, s) for s in sents_in_par]
            self._nodes[nid] = HierarchyNode(
                level=2, index=p_idx, parent=None, children=children, tau=avg_tau,
            )
            for c in children:
                if c in self._nodes:
                    old = self._nodes[c]
                    self._nodes[c] = HierarchyNode(
                        level=old.level, index=old.index, parent=nid,
                        children=old.children, tau=old.tau,
                    )

        for ch_idx, pars_in_ch in enumerate(chapter_groups):
            nid = (3, ch_idx)
            children = [(2, p) for p in pars_in_ch]
            taus = [self._nodes[(2, p)].tau for p in pars_in_ch]
            avg_tau = float(np.mean(taus)) if taus else float(ch_idx) / max(1, n_ch)
            self._nodes[nid] = HierarchyNode(
                level=3, index=ch_idx, parent=None, children=children, tau=avg_tau,
            )
            for c in children:
                if c in self._nodes:
                    old = self._nodes[c]
                    self._nodes[c] = HierarchyNode(
                        level=old.level, index=old.index, parent=nid,
                        children=old.children, tau=old.tau,
                    )

        nid = (4, 0)
        children = [(3, i) for i in range(n_ch)]
        self._nodes[nid] = HierarchyNode(
            level=4, index=0, parent=None, children=children, tau=0.5,
        )
        for c in children:
            if c in self._nodes:
                old = self._nodes[c]
                self._nodes[c] = HierarchyNode(
                    level=old.level, index=old.index, parent=nid,
                    children=old.children, tau=old.tau,
                )

        nid = (5, 0)
        children = [(4, 0)]
        self._nodes[nid] = HierarchyNode(
            level=5, index=0, parent=None, children=children, tau=0.5,
        )
        old = self._nodes[(4, 0)]
        self._nodes[(4, 0)] = HierarchyNode(
            level=4, index=0, parent=nid, children=old.children, tau=old.tau,
        )

    def _compute_tau(self, n: int) -> np.ndarray:
        if n <= 1:
            return np.array([0.5])
        return np.linspace(0.0, 1.0, n)

    def get_node(self, node_id: NodeID) -> HierarchyNode:
        return self._nodes[node_id]

    def get_level(self, level: int) -> list[HierarchyNode]:
        return [n for nid, n in self._nodes.items() if nid[0] == level]

    def get_children(self, node_id: NodeID) -> list[HierarchyNode]:
        return [self._nodes[cid] for cid in self._nodes[node_id].children]

    def get_parent(self, node_id: NodeID) -> HierarchyNode | None:
        parent_id = self._nodes[node_id].parent
        return self._nodes[parent_id] if parent_id else None

    def traverse_bottom_up(self) -> list[NodeID]:
        order = []
        for level in range(1, 6):
            for nid, node in sorted(self._nodes.items()):
                if nid[0] == level:
                    order.append(nid)
        return order

    def traverse_top_down(self) -> list[NodeID]:
        order = []
        for level in range(5, 0, -1):
            for nid, node in sorted(self._nodes.items()):
                if nid[0] == level:
                    order.append(nid)
        return order
