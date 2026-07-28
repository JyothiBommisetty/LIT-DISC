"""Narrative feature extractor (N dimension, k_d=8).

Extracts features related to narrative structure: tension, pacing,
scene construction, temporal shifts, and narrative trajectory.

HMP-DT Feature Dictionary, Features B1-B8.
"""
import re
import numpy as np
from collections import Counter
from src.features.base import BaseFeatureExtractor


# Temporal shift markers
_PAST_TENSE_MARKERS = {"was", "were", "had", "did", "would", "could",
                       "should", "might", "used", "began", "started"}
_PAST_PARTICIPLE_ENDINGS = {"ed", "en", "t"}  # simplified
_FUTURE_MARKERS = {"will", "shall", "going to", "would"}
_FLASHBACK_MARKERS = {"remembered", "recalled", "had been", "used to",
                      "thought back", "earlier", "before", "once"}

# Subordination markers
_SUBORDINATORS = {"because", "although", "while", "when", "if", "unless",
                  "since", "after", "before", "though", "whereas", "until",
                  "as", "once", "that", "which", "who", "whom", "whose",
                  "where", "wherever", "whenever"}

# Named entity pattern (capitalized words, excluding sentence start)
_NE_PATTERN = re.compile(r"\b[A-Z][a-z]+\b")

# Change / narrative progression words
_NARRATIVE_VERBS = {"arrived", "departed", "left", "came", "went", "entered",
                    "exited", "returned", "moved", "turned", "changed",
                    "transformed", "emerged", "vanished", "appeared"}

# Scene transition markers
_SCENE_MARKERS = {"later", "meanwhile", "suddenly", "then", "next",
                  "afterwards", "eventually", "finally", "immediately",
                  "at last", "at once", "all at once", "presently"}


class NarrativeExtractor(BaseFeatureExtractor):
    k_d: int = 8

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        if not tokens:
            return (np.zeros(self.k_d, dtype=float), 0.0)

        n = len(tokens)

        tension = self._compute_tension(tokens, pos)
        pace = self._compute_pace(tokens, pos)
        scene_type = self._compute_scene_type(sentence, tokens, pos)
        temporal_shift = self._compute_temporal_shift(tokens, pos)
        scene_entropy = self._compute_scene_entropy(tokens, pos)
        flashback_marker = self._compute_flashback_marker(tokens, pos)
        narrative_distance = self._compute_narrative_distance(tokens, pos)
        structural_complexity = self._compute_structural_complexity(sentence, tokens, pos)

        features = np.array([
            tension, pace, scene_type, temporal_shift,
            scene_entropy, flashback_marker, narrative_distance,
            structural_complexity,
        ], dtype=float)

        gamma = self.compute_confidence(features)
        return self.normalize(features), gamma

    def _compute_tension(self, tokens: list[str], pos: list[str]) -> float:
        n = len(tokens)
        if n < 3:
            return 0.0
        n_verbs = sum(1 for p in pos if p == "VERB")
        n_narrative = sum(1 for t in tokens if t.lower() in _NARRATIVE_VERBS)
        action_density = (n_verbs + n_narrative) / n
        return min(1.0, action_density * 2.0)

    def _compute_pace(self, tokens: list[str], pos: list[str]) -> float:
        n = len(tokens)
        if n < 2:
            return 0.5
        short_sentence = 1.0 if n <= 8 else 0.0
        long_sentence = 0.0 if n <= 8 else min(1.0, (n - 8) / 32.0)
        n_scene = sum(1 for t in tokens if t.lower() in _SCENE_MARKERS)
        scene_density = n_scene / n if n > 0 else 0.0
        pace_val = short_sentence * 0.5 + scene_density * 0.5
        if long_sentence > 0.5:
            pace_val = pace_val * 0.5
        return min(1.0, pace_val)

    def _compute_scene_type(self, sentence: str, tokens: list[str],
                            pos: list[str]) -> float:
        quote_count = sentence.count("\"") + sentence.count("'")
        quote_pairs = quote_count / 2.0
        if quote_pairs >= 1:
            dialogue_indicator = min(1.0, quote_pairs / 4.0)
        else:
            dialogue_indicator = 0.0
        has_speech_verb = any(t.lower() in {"said", "asked", "replied",
                               "whispered", "shouted", "cried", "answered"}
                              for t in tokens)
        if dialogue_indicator > 0.5 or has_speech_verb:
            return dialogue_indicator
        n_content = sum(1 for p in pos if p in {"NOUN", "VERB", "ADJ", "ADV", "PROPN"})
        n_desc = sum(1 for p in pos if p in {"ADJ", "ADV"})
        desc_ratio = n_desc / n_content if n_content > 0 else 0.0
        narrative_indicator = min(1.0, desc_ratio * 2.0)
        return 1.0 - narrative_indicator

    def _compute_temporal_shift(self, tokens: list[str],
                                 pos: list[str]) -> float:
        past = sum(1 for t in tokens if t.lower() in _PAST_TENSE_MARKERS)
        future = sum(1 for t in tokens if t.lower() in _FUTURE_MARKERS)
        shifters = past + future
        if shifters == 0 and pos:
            n_verbs = sum(1 for p in pos if p == "VERB")
            return min(0.3, n_verbs / len(pos) * 0.5)
        return min(1.0, shifters / max(1, len(tokens)) * 3.0)

    def _compute_scene_entropy(self, tokens: list[str],
                                pos: list[str]) -> float:
        named = [t for t, p in zip(tokens, pos)
                 if p == "PROPN" or (t[0].isupper() and p not in {"DET", "ADP", "CCONJ"})]
        if not named:
            return 0.0
        counts = Counter(t.lower() for t in named)
        total = len(named)
        entropy = -sum((c / total) * np.log(c / total + 1e-10) for c in counts.values())
        max_entropy = np.log(len(counts) + 1)
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

    def _compute_flashback_marker(self, tokens: list[str],
                                   pos: list[str]) -> float:
        markers = sum(1 for t in tokens if t.lower() in _FLASHBACK_MARKERS)
        had_pp = False
        for i, t in enumerate(tokens):
            if t.lower() == "had" and i + 1 < len(tokens):
                next_pos = pos[i + 1] if i + 1 < len(pos) else ""
                if next_pos == "VERB":
                    had_pp = True
        score = markers * 0.25 + (0.5 if had_pp else 0.0)
        return min(1.0, score)

    def _compute_narrative_distance(self, tokens: list[str],
                                     pos: list[str]) -> float:
        first_person = sum(1 for t in tokens if t.lower() in {"i", "me", "my",
                            "we", "us", "our", "mine", "ours"})
        third_person = sum(1 for t in tokens if t.lower() in {"he", "she", "it",
                            "they", "him", "her", "them", "his", "their"})
        pronouns = first_person + third_person
        if pronouns == 0:
            return 0.5
        ratio = first_person / pronouns
        distance = 1.0 - ratio
        return min(1.0, max(0.0, distance))

    def _compute_structural_complexity(self, sentence: str,
                                        tokens: list[str],
                                        pos: list[str]) -> float:
        n = len(tokens)
        if n < 5:
            return 0.0
        subordinators = sum(1 for t in tokens if t.lower() in _SUBORDINATORS)
        clauses = subordinators + 1
        clause_density = clauses / max(1, n)
        comma_count = sentence.count(",")
        semicolon_count = sentence.count(";")
        colon_count = sentence.count(":")
        punct_complexity = min(1.0, (comma_count + semicolon_count * 2 + colon_count * 2) / 5.0)
        complexity = clause_density * 3.0 + punct_complexity * 0.4
        return min(1.0, complexity)
