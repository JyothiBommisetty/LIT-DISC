# LIT-DISC Module Interfaces (Protocol Classes)
# Reference: Architecture Specification §5

from __future__ import annotations
from typing import Protocol
import numpy as np
from .models import LiterarySignal, MemoryEntry, NarrativeEmbedding, HierarchyNode


class FeatureExtractor(Protocol):
    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        ...


class Propagator(Protocol):
    def aggregate_sentence_to_paragraph(self, signals: list[LiterarySignal]) -> LiterarySignal:
        ...  # HMP-DT Eq. (10.2)–(10.4)

    def aggregate_paragraph_to_chapter(self, signals: list[LiterarySignal],
                                        beta_sal: float) -> tuple[LiterarySignal, np.ndarray]:
        ...  # HMP-DT Eq. (10.5)–(10.6)

    def interpolate_to_narrative(self, chapter_signals: list[tuple[float, LiterarySignal]],
                                  tau: np.ndarray) -> LiterarySignal:
        ...  # HMP-DT Eq. (10.7)

    def project_to_discourse(self, narrative_signals: dict[str, LiterarySignal],
                              W: dict[str, np.ndarray]) -> np.ndarray:
        ...  # HMP-DT Eq. (10.8)

    def refine_top_down(self, child: LiterarySignal, parent: LiterarySignal,
                        beta_down: float) -> LiterarySignal:
        ...  # HMP-DT Eq. (10.9)


class Coupler(Protocol):
    def compute_adaptive_weights(self, tau: float, params: dict) -> np.ndarray:
        ...  # HMP-DT Eq. (11.1)–(11.2)

    def compute_kernel_components(self, signals: dict[str, LiterarySignal],
                                   tau: float) -> dict[str, float]:
        ...  # HMP-DT Eq. (11.3)–(11.8)

    def compute_kernel(self, components: dict[str, float],
                        weights: np.ndarray) -> float:
        ...  # HMP-DT Eq. (11.9)

    def apply_coupling(self, signals: dict[str, LiterarySignal],
                        kappa: np.ndarray, M: np.ndarray) -> dict[str, LiterarySignal]:
        ...  # HMP-DT Eq. (11.10)–(11.11)


class MemorySubsystem(Protocol):
    def make_key(self, f: np.ndarray) -> np.ndarray:
        ...  # HMP-DT Eq. (12.2)

    def should_store(self, signal: LiterarySignal) -> bool:
        ...  # HMP-DT Eq. (12.1)

    def retrieve(self, key: np.ndarray) -> list[MemoryEntry]:
        ...  # HMP-DT Eq. (12.3)

    def reinforce(self, entry: MemoryEntry, similarity: float) -> None:
        ...  # HMP-DT Eq. (12.4)

    def decay_all(self, delta_t: int) -> None:
        ...  # HMP-DT Eq. (12.5)

    def purge(self) -> None:
        ...  # HMP-DT Eq. (12.6)

    def augment(self, signal: LiterarySignal, retrieved: list[MemoryEntry]) -> LiterarySignal:
        ...  # HMP-DT Eq. (12.7)


class ResonanceEngine(Protocol):
    def update(self, R_prev: np.ndarray, H_current: np.ndarray,
               delta_tau: float, lam: float) -> np.ndarray:
        ...  # HMP-DT Eq. (12.8)

    def augment(self, signal: LiterarySignal, resonance: np.ndarray,
                eta_res: float) -> LiterarySignal:
        ...  # HMP-DT Eq. (12.9)


class UncertaintyEngine(Protocol):
    def total_variance(self, sources: dict[str, float]) -> float:
        ...  # HMP-DT Eq. (13.1)

    def confidence(self, sigma_total: float, sigma_max: float) -> float:
        ...  # HMP-DT Eq. (13.2)

    def propagate_to_paragraph(self, children: list[float],
                                coherence: float) -> tuple[float, float]:
        ...  # HMP-DT Eq. (13.3)

    def propagate_to_chapter(self, paragraph_vars: list[float],
                              alphas: np.ndarray) -> float:
        ...  # HMP-DT Eq. (13.4)

    def propagate_to_narrative(self, chapter_vars: list[float],
                                phis: np.ndarray, taus: np.ndarray,
                                tau: float) -> float:
        ...  # HMP-DT Eq. (13.5)

    def coupling_variance(self, kappa: float, var_d: float, var_dp: float) -> float:
        ...  # HMP-DT Eq. (13.6)


class LDRICalculator(Protocol):
    def compute_activations(self, H: dict[str, np.ndarray],
                             sigma: dict[str, float]) -> np.ndarray:
        ...  # LDRI Eq. (2)

    def signal_richness(self, activations: np.ndarray) -> float:
        ...  # LDRI Eq. (3)–(4)

    def coupling_factor(self, H_pairs: np.ndarray) -> float:
        ...  # LDRI Eq. (5), (9)

    def resonance_factor(self, H_res: np.ndarray) -> float:
        ...  # LDRI Eq. (6), (9)

    def memory_factor(self, memory_avg_beta: float) -> float:
        ...  # LDRI Eq. (7), (9)

    def global_confidence(self, gammas: dict[str, float]) -> float:
        ...  # LDRI Eq. (11)–(12)

    def ldri(self, R_signal: float, C_tilde: float, T_tilde: float,
              M_tilde: float, Gamma: float,
              alpha: float, beta: float, gamma: float, delta: float) -> float:
        ...  # LDRI Eq. (8), (10)


class Decomposer(Protocol):
    def intrinsic(self, R_signal: float, Gamma: float, alpha: float, S: float) -> float:
        ...  # LDRI Eq. (16)

    def interaction(self, C: float, Gamma: float, beta: float, S: float) -> float:
        ...  # LDRI Eq. (17)

    def temporal(self, T: float, Gamma: float, gamma: float, S: float) -> float:
        ...  # LDRI Eq. (18)

    def resonance(self, R_tau: np.ndarray, lam: float, Gamma: float, gamma: float, S: float) -> float:
        ...  # LDRI Eq. (19)

    def memory(self, M: float, Gamma: float, delta: float, S: float) -> float:
        ...  # LDRI Eq. (20)

    def uncertainty_adj(self, ldri_raw: float, Gamma: float) -> float:
        ...  # LDRI Eq. (21)–(22)

    def fractions(self, components: dict[str, float], ldri: float) -> dict[str, float]:
        ...  # LDRI Eq. (23)


class ExplanationGenerator(Protocol):
    def generate(self, converged_state, coupling_history, memory_store,
                 resonance_field, uncertainty_state) -> dict:
        ...


class NarrativePositionComputer(Protocol):
    def compute(self, sentences: list[str], tokens: list[list[str]]) -> np.ndarray:
        ...
