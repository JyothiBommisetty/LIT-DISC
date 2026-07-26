import os
import numpy as np
from src.models import LITDISCResult, ExplanationTrace
from src.config import LITDISCConfig
from src.loader import TextLoader
from src.preprocessing import Tokenizer, SentenceSplitter
from src.hierarchy import ParagraphDetector, ChapterDetector, HierarchyTree
from src.features import SemanticExtractor
from src.signals import SignalInitializer
from src.propagation import BottomUpPropagator
from src.embedding import NarrativeEmbeddingConstructor
from src.ldri import LDRICalculator
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
    def __init__(self, config: LITDISCConfig | None = None):
        self.config = config or LITDISCConfig()
        self.loader = TextLoader()
        self.tokenizer = Tokenizer()
        self.splitter = SentenceSplitter()
        self.para_detector = ParagraphDetector()
        self.chapter_detector = ChapterDetector()
        self.tree = HierarchyTree()
        self.tau_computer = NarrativePositionComputer()
        self.extractors = {
            "S": SemanticExtractor(),
        }
        self.signal_init = SignalInitializer()
        self.propagator = BottomUpPropagator()
        self.embedder = NarrativeEmbeddingConstructor()
        self.ldri_calc = LDRICalculator(self.config.ldri)
        self.reporter = ReportGenerator()

    def run(self, text_path: str) -> LITDISCResult:
        text_id = os.path.splitext(os.path.basename(text_path))[0]
        text = self.loader.load(text_path)
        sentences = self.splitter.split(text)
        para_groups = self.para_detector.detect(sentences, text)
        ch_groups = self.chapter_detector.detect(para_groups, text)
        self.tree.build(sentences, para_groups, ch_groups)

        sentence_tokens = [self.tokenizer.tokenize_sentence(s) for s in sentences]
        sent_tokens = [td["tokens"] for td in sentence_tokens]
        taus = self.tau_computer.compute(sentences, sent_tokens)
        tau_map = {(1, i): float(taus[i]) for i in range(len(sentences))}

        dim_features: dict[str, dict[tuple, tuple[np.ndarray, float]]] = {"S": {}}
        for i, sent in enumerate(sentences):
            td = sentence_tokens[i]
            nid = (1, i)
            fvec, gamma = self.extractors["S"].extract(sent, td["tokens"], td["pos"], td["deps"])
            dim_features["S"][nid] = (fvec, gamma)

        signals = self.signal_init.initialize(dim_features, tau_map)
        propagated = self.propagator.propagate(self.tree, signals, dimensions=["S"])

        embedding = self.embedder.construct(propagated, self.tree)
        confidences = self._extract_confidences(propagated)
        ldri_score = self.ldri_calc.compute(embedding, confidences)

        explanation = ExplanationTrace(
            signal_lineage=[],
            propagation_path={},
            interaction_history={},
            uncertainty_evolution=[],
            resonance_contribution=0.0,
            memory_activations=[],
        )

        result = LITDISCResult(
            text_id=text_id,
            embedding=embedding,
            ldri=ldri_score,
            explanation=explanation,
            metadata={
                "sentences": len(sentences),
                "paragraphs": len(para_groups),
                "chapters": len(ch_groups),
                "active_dimensions": ["S"],
            },
        )
        return result

    def _extract_confidences(self, signals: dict) -> dict[str, float]:
        confs: dict[str, list[float]] = {}
        for nid, dim_dict in signals.items():
            for dim, sig in dim_dict.items():
                if dim not in confs:
                    confs[dim] = []
                confs[dim].append(sig.gamma)
        return {d: float(np.mean(vs)) if vs else 0.5 for d, vs in confs.items()}
