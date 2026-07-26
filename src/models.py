# LIT-DISC Shared Data Models
# Frozen dataclasses for all HMP-DT and LDRI data structures.
# Reference: Architecture Specification §4

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


NodeID = tuple[int, int]  # (hierarchy_level, node_index)
Dimension = str           # "S", "N", "I", "Sy", "E", "Em"


@dataclass(frozen=True)
class LiterarySignal:
    f: np.ndarray              # feature vector, ℝ^{k_d}
    gamma: float               # confidence, [0, 1]
    tau: float                 # narrative position, [0, 1]
    context: Any               # interpretive context
    provenance: list[NodeID]   # parent chain
    residues: np.ndarray       # interaction residues, [0,1]^{6×6}
    memory_traces: list[Any]   # (key, activation) pairs


@dataclass(frozen=True)
class HierarchyNode:
    level: int                 # 1–5
    index: int                 # position within level
    parent: NodeID | None      # parent node
    children: list[NodeID]     # child node IDs
    tau: float                 # narrative position


@dataclass(frozen=True)
class NarrativeEmbedding:
    H_S: np.ndarray            # ℝ^{d/10}
    H_N: np.ndarray            # ℝ^{d/10}
    H_I: np.ndarray            # ℝ^{d/10}
    H_Sy: np.ndarray           # ℝ^{d/10}
    H_E: np.ndarray            # ℝ^{d/10}
    H_Em: np.ndarray           # ℝ^{d/10}
    H_pairs: np.ndarray        # ℝ^{3d/10}
    H_res: np.ndarray          # ℝ^{d/10}
    d: int = 120               # total dimension

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.H_S, self.H_N, self.H_I,
            self.H_Sy, self.H_E, self.H_Em,
            self.H_pairs, self.H_res,
        ])


@dataclass(frozen=True)
class LDRIScore:
    ldri: float                # [0,1] point estimate
    gamma_global: float        # [0,1] global confidence
    reliability: float         # [0,1] bootstrap stability
    raw: float                 # LDRI_raw (Γ=1)
    components: dict[str, float]   # decomposition components
    fractions: dict[str, float]    # contribution fractions


@dataclass(frozen=True)
class MemoryEntry:
    key: np.ndarray            # Fourier feature key, ℝ^{32}
    signal: LiterarySignal     # stored signal
    beta: float                # activation [0,1]
    storage_tau: float         # τ at storage
    retrieval_count: int       # number of retrievals


@dataclass(frozen=True)
class ExplanationTrace:
    signal_lineage: list[dict]
    propagation_path: dict
    interaction_history: dict
    uncertainty_evolution: list[dict]
    resonance_contribution: float
    memory_activations: list[dict]


@dataclass(frozen=True)
class LITDISCResult:
    text_id: str
    embedding: NarrativeEmbedding
    ldri: LDRIScore
    explanation: ExplanationTrace
    metadata: dict
