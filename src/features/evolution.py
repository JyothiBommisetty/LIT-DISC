"""Evolution feature extractor (E dimension, k_d=7).

Extracts features related to character development, relationship arcs,
and temporal trajectory.

HMP-DT Feature Dictionary, Features E1-E7.
"""
import re
import numpy as np
from src.features.base import BaseFeatureExtractor


# Change / development verbs
_CHANGE_VERBS = {"changed", "transformed", "evolved", "grew", "developed",
                 "matured", "shifted", "became", "turned", "emerged",
                 "improved", "declined", "deteriorated", "progressed",
                 "advanced", "regressed", "adapted", "adjusted", "altered",
                 "modified", "weakened", "strengthened", "deepened",
                 "widened", "narrowed", "rose", "fell", "increased",
                 "decreased", "began", "ended", "started", "stopped"}

# Relationship / social proximity words
_RELATIONSHIP_WORDS = {"together", "apart", "close", "distant", "near",
                       "far", "friend", "enemy", "lover", "stranger",
                       "partner", "spouse", "husband", "wife", "mother",
                       "father", "sister", "brother", "daughter", "son",
                       "family", "marriage", "friendship", "love", "hate",
                       "trust", "betray", "loyal", "ally", "oppose"}

# Comparative / contrast markers
_COMPARATIVE_WORDS = {"more", "less", "better", "worse", "stronger",
                      "weaker", "faster", "slower", "greater", "lesser",
                      "earlier", "later", "older", "younger", "before",
                      "after", "than", "unlike", "different", "same",
                      "changed", "still", "yet", "already", "anymore",
                      "no longer", "different from", "compared to"}

# Development / progression markers
_PROGRESSION_WORDS = {"first", "then", "next", "finally", "eventually",
                      "gradually", "slowly", "rapidly", "suddenly",
                      "ultimately", "meanwhile", "later", "earlier",
                      "initially", "subsequently", "previously",
                      "afterwards", "at first", "in the end",
                      "over time", "as time passed", "by now"}

# Temporal progression indicators
_TENSE_CHANGE_WORDS = {"now", "then", "before", "after", "today",
                       "yesterday", "tomorrow", "ago", "later",
                       "currently", "previously", "presently"}

# Closure / resolution markers
_CLOSURE_WORDS = {"finally", "at last", "in the end", "ultimately",
                  "after all", "all in all", "in conclusion",
                  "ended", "finished", "completed", "resolved",
                  "accepted", "understood", "realized", "knew"}


class EvolutionExtractor(BaseFeatureExtractor):
    k_d: int = 7

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        if not tokens:
            return (np.zeros(self.k_d, dtype=float), 0.0)

        n = len(tokens)

        arc_slope = self._compute_arc_slope(tokens, pos)
        relationship_state = self._compute_relationship_state(tokens, pos)
        change_magnitude = self._compute_change_magnitude(tokens, pos)
        developmental_tokens = self._compute_developmental_tokens(tokens, pos)
        temporal_progression = self._compute_temporal_progression(tokens, pos)
        contrast_marking = self._compute_contrast_marking(tokens, pos)
        resolution_index = self._compute_resolution_index(tokens, pos)

        features = np.array([
            arc_slope, relationship_state, change_magnitude,
            developmental_tokens, temporal_progression,
            contrast_marking, resolution_index,
        ], dtype=float)

        gamma = self.compute_confidence(features)
        return self.normalize(features, bounds=(-1.0, 1.0) if self.k_d > 0 else (0.0, 1.0)), gamma

    def _compute_arc_slope(self, tokens: list[str],
                            pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        positive_change = sum(1 for t in tokens if t.lower() in
                              {"improved", "grew", "better", "stronger",
                               "progressed", "advanced", "rose", "increased",
                               "emerged", "developed", "matured", "recovered"})
        negative_change = sum(1 for t in tokens if t.lower() in
                              {"declined", "worse", "weaker", "regressed",
                               "deteriorated", "fell", "decreased", "decayed",
                               "collapsed", "worsened", "suffered", "lost"})
        net = (positive_change - negative_change) / max(1, n)
        return max(-1.0, min(1.0, net * 3.0))

    def _compute_relationship_state(self, tokens: list[str],
                                      pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.5
        positive_rel = sum(1 for t in tokens if t.lower() in
                          {"together", "close", "friend", "lover", "love",
                           "trust", "loyal", "ally", "family", "marriage",
                           "friendship", "partner", "together"})
        negative_rel = sum(1 for t in tokens if t.lower() in
                          {"apart", "distant", "enemy", "stranger", "hate",
                           "betray", "oppose", "alone", "separate"})
        total = positive_rel + negative_rel
        if total == 0:
            return 0.5
        ratio = positive_rel / total
        return float(ratio)

    def _compute_change_magnitude(self, tokens: list[str],
                                   pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        changes = sum(1 for t in tokens if t.lower() in _CHANGE_VERBS)
        progress = sum(1 for t in tokens if t.lower() in _PROGRESSION_WORDS)
        comparative = sum(1 for t in tokens if t.lower() in _COMPARATIVE_WORDS)
        score = changes * 0.3 + progress * 0.2 + comparative * 0.15
        return min(1.0, score)

    def _compute_developmental_tokens(self, tokens: list[str],
                                       pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        dev = sum(1 for t in tokens if t.lower() in
                 {"grew", "learned", "understood", "realized", "discovered",
                  "became", "started", "began", "continued", "kept",
                  "persisted", "struggled", "overcame", "survived",
                  "adapted", "evolved"})
        return min(1.0, dev / n * 2.0)

    def _compute_temporal_progression(self, tokens: list[str],
                                       pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        progression = sum(1 for t in tokens if t.lower() in _PROGRESSION_WORDS)
        tense_markers = sum(1 for t in tokens if t.lower() in _TENSE_CHANGE_WORDS)
        had = sum(1 for t in tokens if t.lower() == "had")
        score = progression * 0.2 + tense_markers * 0.25 + (had * 0.15 if had > 0 else 0.0)
        return min(1.0, score)

    def _compute_contrast_marking(self, tokens: list[str],
                                   pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        contrasts = sum(1 for t in tokens if t.lower() in
                       {"but", "however", "although", "though", "yet",
                        "nevertheless", "nonetheless", "still",
                        "instead", "rather", "unlike", "whereas",
                        "while", "despite", "in spite of", "on the other hand",
                        "in contrast", "conversely"})
        comparatives = sum(1 for t in tokens if t.lower() in
                          {"more", "less", "better", "worse", "rather than"})
        score = contrasts * 0.25 + comparatives * 0.15
        return min(1.0, score)

    def _compute_resolution_index(self, tokens: list[str],
                                   pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        closure = sum(1 for t in tokens if t.lower() in _CLOSURE_WORDS)
        realized = sum(1 for t in tokens if t.lower() in
                      {"realized", "understood", "accepted", "knew",
                       "saw", "recognized", "acknowledged"})
        score = closure * 0.25 + realized * 0.2
        return min(1.0, score)
