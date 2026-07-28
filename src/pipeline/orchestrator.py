import math
import os
import time
import numpy as np
from src.models import LITDISCResult, ExplanationTrace
from src.config import LITDISCConfig
from src.loader import TextLoader
from src.preprocessing import Tokenizer, SentenceSplitter
from src.hierarchy import ParagraphDetector, ChapterDetector, HierarchyTree
from src.features import (
    SemanticExtractor, NarrativeExtractor, InteractionExtractor,
    SymbolicExtractor, EvolutionExtractor, EmotionExtractor,
)
from src.signals import SignalInitializer
from src.propagation import BottomUpPropagator, TopDownRefiner, RefinementController
from src.coupling.memory import MemorySystem
from src.resonance import NarrativeResonance
from src.uncertainty import UncertaintyPropagator
from src.embedding import NarrativeEmbeddingConstructor
from src.ldri import LDRICalculator
from src.explainability import CompositeExplanationGenerator, ContributionBreakdownGenerator
from src.reports import ReportGenerator


class NarrativePositionComputer:
    def compute(self, sentences: list[str], tokens: list[list[str]]) -> np.ndarray:
        n = len(sentences)
        if n <= 1:
            return np.array([0.5])
        vel = np.ones(n)
        for i, tkns in enumerate(tokens):
            verbs = sum(1 for t in tkns if t.isalpha() and len(t) > 2)
            vel[i] = max(0.1, verbs / max(1, len(tkns)))
        vel[0] = max(vel[0], 0.1)
        cum = np.cumsum(vel)
        return cum / cum[-1]


class LITDISCPipeline:
    """Complete LIT-DISC pipeline with full HMP-DT v1.0 integration.

    Pipeline steps:
      1. Load  →  2. Split  →  3. Detect (paragraphs, chapters)
      4. Build hierarchy tree  →  5. Tokenize  →  6. τ computation
      7. Feature extraction (6 dims)  →  8. Signal initialization
      9. K-iteration refinement loop:
         a. Bottom-up propagation with horizontal coupling
         b. Memory storage & decay
         c. Recursive resonance update
         d. Top-down refinement
     10. Embedding construction (with resonance trajectory)
     11. LDRI calculation (with memory avg beta)
     12. Explanation trace generation
     13. Return LITDISCResult
    """

    def __init__(self, config: LITDISCConfig | None = None,
                 refinement_mode: str = "full"):
        self.config = config or LITDISCConfig()
        self.refinement_mode = refinement_mode

        # Core pipeline components
        self.loader = TextLoader()
        self.tokenizer = Tokenizer()
        self.splitter = SentenceSplitter()
        self.para_detector = ParagraphDetector()
        self.chapter_detector = ChapterDetector()
        self.tree = HierarchyTree()
        self.tau_computer = NarrativePositionComputer()

        # Feature extractors (6 dimensions)
        self.extractors = {
            "S": SemanticExtractor(),
            "N": NarrativeExtractor(),
            "I": InteractionExtractor(),
            "Sy": SymbolicExtractor(),
            "E": EvolutionExtractor(),
            "Em": EmotionExtractor(),
        }

        # Signal initialization
        self.signal_init = SignalInitializer()

        # Full HMP-DT algorithm components
        hmp_cfg = self.config.hmp_dt
        self.memory = MemorySystem(hmp_cfg.memory)
        self.resonance = NarrativeResonance(
            hmp_cfg.resonance.lambda_decay,
            hmp_cfg.resonance.eta_resonance)
        self.uncertainty = UncertaintyPropagator()
        self.bottom_up = BottomUpPropagator(
            memory=self.memory,
            resonance=self.resonance,
            uncertainty=self.uncertainty)
        self.top_down = TopDownRefiner()
        self.controller = RefinementController(
            hmp_cfg,
            memory=self.memory,
            resonance=self.resonance,
            uncertainty=self.uncertainty)

        # Embedding & LDRI
        self.embedder = NarrativeEmbeddingConstructor()
        self.ldri_calc = LDRICalculator(self.config.ldri)

        # Explainability
        self.explanation_generator = CompositeExplanationGenerator()
        self.contribution_breakdown = ContributionBreakdownGenerator()

        # Report
        self.reporter = ReportGenerator()

        # Timing
        self._timings: dict[str, float] = {}

    def run(self, text_path: str) -> LITDISCResult:
        t_start = time.perf_counter()
        text_id = os.path.splitext(os.path.basename(text_path))[0]

        # 1. Load
        t0 = time.perf_counter()
        text = self.loader.load(text_path)
        self._timings["load"] = time.perf_counter() - t0

        # 2. Split sentences
        t0 = time.perf_counter()
        sentences = self.splitter.split(text)
        self._timings["split"] = time.perf_counter() - t0

        # 3. Detect paragraphs and chapters
        t0 = time.perf_counter()
        para_groups = self.para_detector.detect(sentences, text)
        ch_groups = self.chapter_detector.detect(para_groups, text)
        self._timings["detect"] = time.perf_counter() - t0

        # 4. Build hierarchy tree
        t0 = time.perf_counter()
        self.tree.build(sentences, para_groups, ch_groups)
        self._timings["tree"] = time.perf_counter() - t0

        # 5. Tokenize
        t0 = time.perf_counter()
        sentence_tokens = [self.tokenizer.tokenize_sentence(s) for s in sentences]
        sent_tokens = [td["tokens"] for td in sentence_tokens]
        self._timings["tokenize"] = time.perf_counter() - t0

        # 6. Compute narrative positions (τ)
        t0 = time.perf_counter()
        taus = self.tau_computer.compute(sentences, sent_tokens)
        tau_map = {(1, i): float(taus[i]) for i in range(len(sentences))}
        self._timings["tau"] = time.perf_counter() - t0

        # 7. Feature extraction
        t0 = time.perf_counter()
        SymbolicExtractor.reset_motif_memory()
        dim_features: dict[str, dict[tuple, tuple[np.ndarray, float]]] = {
            d: {} for d in self.extractors
        }
        for i, sent in enumerate(sentences):
            td = sentence_tokens[i]
            nid = (1, i)
            for dim_key, extractor in self.extractors.items():
                fvec, gamma = extractor.extract(
                    sent, td["tokens"], td["pos"], td["deps"])
                dim_features[dim_key][nid] = (fvec, gamma)
        self._timings["features"] = time.perf_counter() - t0

        # 8. Signal initialization
        t0 = time.perf_counter()
        signals = self.signal_init.initialize(dim_features, tau_map)
        self._timings["signal_init"] = time.perf_counter() - t0

        dims = list(self.extractors.keys())

        # 9. Refinement loop
        t0 = time.perf_counter()
        if self.refinement_mode == "full":
            # Reset shared state
            self.memory.clear()
            # Determine H dimension from feature k_d values (max_kd = 8 across all dims)
            k_d_map = {"S": 8, "N": 8, "I": 7, "Sy": 7, "E": 7, "Em": 7}
            max_kd = max(k_d_map.get(d, 7) for d in dims)
            h_dim = len(dims) * max_kd
            self.resonance.reset(h_dim)
            self.uncertainty.clear()

            propagated, n_iters = self.controller.run(
                self.tree, signals, self.bottom_up, self.top_down,
                dimensions=dims)

            # Build resonance trajectory from controller's resonance field
            resonance_trajectory: list[np.ndarray] | None = None
            if hasattr(self.controller.resonance, '_trajectory'):
                resonance_trajectory = [
                    r for _, r in self.controller.resonance._trajectory
                ]
        else:
            # Simple single-pass mode (original behavior)
            propagated = self.bottom_up.propagate(
                self.tree, signals, dimensions=dims,
                coupling_tau=0.5, refine_iter=1)
            n_iters = 1
            resonance_trajectory = None
        self._timings["refinement"] = time.perf_counter() - t0

        # 10. Embedding construction
        t0 = time.perf_counter()
        embedding = self.embedder.construct(
            propagated, self.tree,
            resonance_trajectory=resonance_trajectory)
        self._timings["embedding"] = time.perf_counter() - t0

        # 11. LDRI calculation
        t0 = time.perf_counter()
        confidences = self._extract_confidences(propagated)
        mem_avg_beta = self.memory.get_avg_beta()
        ldri_score = self.ldri_calc.compute(
            embedding, confidences, mem_avg_beta=mem_avg_beta)
        self._timings["ldri"] = time.perf_counter() - t0

        # 12. Explanation trace
        t0 = time.perf_counter()
        explanation_data = self.explanation_generator.as_dict(
            propagated, self.tree, ldri_score,
            memory_store=self.memory,
            resonance_engine=self.controller.resonance if self.refinement_mode == "full" else None,
            uncertainty_propagator=self.uncertainty,
        )
        breakdown = self.contribution_breakdown.generate(ldri_score)
        explanation_data["contribution_breakdown"] = breakdown

        explanation = ExplanationTrace(
            signal_lineage=explanation_data.get("signal_lineage", []),
            propagation_path=explanation_data.get("propagation_path", {}),
            interaction_history=explanation_data.get("interaction_history", {}),
            uncertainty_evolution=explanation_data.get("uncertainty_evolution", []),
            resonance_contribution=explanation_data.get("resonance_contribution", 0.0),
            memory_activations=explanation_data.get("memory_activations", []),
        )
        self._timings["explain"] = time.perf_counter() - t0

        total_time = time.perf_counter() - t_start
        self._timings["total"] = total_time

        # 13. Build result
        result = LITDISCResult(
            text_id=text_id,
            embedding=embedding,
            ldri=ldri_score,
            explanation=explanation,
            metadata={
                "sentences": len(sentences),
                "paragraphs": len(para_groups),
                "chapters": len(ch_groups),
                "refinement_iterations": n_iters,
                "refinement_mode": self.refinement_mode,
                "memory_entries": self.memory.size,
                "memory_avg_beta": mem_avg_beta,
                "active_dimensions": dims,
                "timings_s": dict(self._timings),
                "total_time_s": total_time,
                "resonance_trajectory_steps": len(resonance_trajectory) if resonance_trajectory else 0,
            },
        )
        return result

    def run_all(self, text_paths: list[str]) -> list[LITDISCResult]:
        return [self.run(p) for p in text_paths]

    def get_timings(self) -> dict[str, float]:
        return dict(self._timings)

    def reset(self) -> None:
        self.memory.clear()
        self.uncertainty.clear()
        self._timings.clear()

    def _extract_confidences(self, signals: dict) -> dict[str, float]:
        confs: dict[str, list[float]] = {}
        for nid, dim_dict in signals.items():
            for dim, sig in dim_dict.items():
                if dim not in confs:
                    confs[dim] = []
                confs[dim].append(sig.gamma)
        return {d: float(np.mean(vs)) if vs else 0.5 for d, vs in confs.items()}
