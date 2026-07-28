"""Symbolic feature extractor (Sy dimension, k_d=7).

Extracts features related to symbolism, figurative language,
motif recurrence, and symbolic coherence.

HMP-DT Feature Dictionary, Features D1-D7.
"""
import re
import numpy as np
from collections import Counter
from src.features.base import BaseFeatureExtractor


# Sensory words (visual, auditory, tactile, olfactory, gustatory)
_SENSORY_WORDS = {
    "visual": {"see", "saw", "look", "looked", "watch", "watched", "gaze",
               "gazed", "glow", "glimmer", "shine", "shone", "bright",
               "dark", "shadow", "light", "flash", "sparkle", "gleam",
               "dim", "fade", "faded", "vanish", "appear", "appeared",
               "red", "blue", "green", "black", "white", "grey", "golden",
               "silver", "color", "colour", "pale", "glare", "glimpse"},
    "auditory": {"hear", "heard", "sound", "noise", "voice", "whisper",
                 "shout", "cry", "scream", "silence", "quiet", "loud",
                 "rustle", "crash", "bang", "tap", "knock", "ring",
                 "echo", "melody", "tune", "music", "step", "footstep",
                 "breath", "breathe", "gasp", "sigh"},
    "tactile": {"touch", "touched", "feel", "felt", "grip", "grasp",
                "hold", "cold", "warm", "hot", "soft", "hard", "rough",
                "smooth", "sharp", "dull", "wet", "dry", "heavy", "light",
                "pressure", "pain", "ache", "tingle", "shiver", "tremble"},
    "olfactory": {"smell", "smelled", "scent", "odor", "fragrance",
                  "perfume", "stink", "stench", "musty", "fresh", "rotten",
                  "aroma", "reek", "whiff"},
    "gustatory": {"taste", "tasted", "bitter", "sweet", "sour", "salty",
                  "spicy", "delicious", "flavor", "savor"},
}

_ALL_SENSORY = set()
for cat in _SENSORY_WORDS:
    _ALL_SENSORY.update(_SENSORY_WORDS[cat])

# Abstract / symbolic nouns
_ABSTRACT_NOUNS = {"love", "hate", "fear", "hope", "dream", "time", "life",
                   "death", "soul", "spirit", "heart", "mind", "fate",
                   "destiny", "truth", "beauty", "wisdom", "power", "faith",
                   "justice", "freedom", "memory", "desire", "pain", "joy",
                   "sorrow", "grief", "pride", "shame", "honor", "glory",
                   "nature", "sky", "sea", "earth", "fire", "water", "wind",
                   "moon", "sun", "star", "light", "darkness", "shadow",
                   "mirror", "door", "window", "bridge", "road", "path",
                   "garden", "forest", "river", "mountain", "storm", "rain",
                   "snow", "night", "morning", "spring", "winter"}

# Metaphor-related cognitive verbs
_COGNITIVE_VERBS = {"seem", "appear", "become", "remain", "feel", "sense",
                    "know", "understand", "realize", "imagine", "dream",
                    "wonder", "believe", "suppose", "suspect", "perceive",
                    "discern", "intuit", "conceive", "envision"}

# Euphemism / indirectness markers
_EUPHEMISM_MARKERS = {"passed away", "departed", "no longer with us",
                      "between jobs", "economical with the truth",
                      "adult content", "intimate", "special",
                      "in a better place", "sleep with", "lie with",
                      "gentleman's club", "comfort woman"}

# Simple euphemism words (single token)
_EUPHEMISM_WORDS = {"passed", "departed", "intimate", "special"}

# Motif / repetition tracking state
_MOTIF_MEMORY: dict[str, int] = {}  # word → count across document
_SEEN_SENTENCES: int = 0


class SymbolicExtractor(BaseFeatureExtractor):
    k_d: int = 7

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        if not tokens:
            return (np.zeros(self.k_d, dtype=float), 0.0)

        figurative_density = self._compute_figurative_density(tokens, pos)
        euphemism_strength = self._compute_euphemism_strength(tokens)
        motif_recurrence = self._compute_motif_recurrence(tokens, pos)
        symbolic_density = self._compute_symbolic_density(tokens, pos)
        imagery_intensity = self._compute_imagery_intensity(tokens)
        allegorical_marking = self._compute_allegorical_marking(tokens, pos)
        connotative_depth = self._compute_connotative_depth(tokens, pos)

        features = np.array([
            figurative_density, euphemism_strength, motif_recurrence,
            symbolic_density, imagery_intensity, allegorical_marking,
            connotative_depth,
        ], dtype=float)

        gamma = self.compute_confidence(features)
        return self.normalize(features), gamma

    def _compute_figurative_density(self, tokens: list[str],
                                     pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        cognitive = sum(1 for t in tokens if t.lower() in _COGNITIVE_VERBS)
        sensory = sum(1 for t in tokens if t.lower() in _ALL_SENSORY)
        abstract_nouns = sum(1 for t, p in zip(tokens, pos)
                             if t.lower() in _ABSTRACT_NOUNS and p in {"NOUN", "PROPN"})
        figurative = cognitive + sensory * 0.5 + abstract_nouns * 0.3
        return min(1.0, figurative / n * 2.0)

    def _compute_euphemism_strength(self, tokens: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        euphemisms = sum(1 for t in tokens if t.lower() in _EUPHEMISM_WORDS)
        return min(1.0, euphemisms / max(1, n) * 3.0)

    def _compute_motif_recurrence(self, tokens: list[str],
                                   pos: list[str]) -> float:
        global _SEEN_SENTENCES, _MOTIF_MEMORY
        content = [t.lower() for t, p in zip(tokens, pos)
                   if p in {"NOUN", "VERB", "ADJ", "PROPN"} and len(t) > 3]
        if not content:
            return 0.0
        _SEEN_SENTENCES += 1
        seen = set()
        recurrence_count = 0
        for w in content:
            if w in _MOTIF_MEMORY:
                _MOTIF_MEMORY[w] += 1
                if w not in seen:
                    recurrence_count += 1
                    seen.add(w)
            else:
                _MOTIF_MEMORY[w] = 1
        total = len(content)
        ratio = recurrence_count / total if total > 0 else 0.0
        return min(1.0, ratio * 3.0)

    def _compute_symbolic_density(self, tokens: list[str],
                                   pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        symbolic = sum(1 for t, p in zip(tokens, pos)
                       if t.lower() in _ABSTRACT_NOUNS and p in {"NOUN", "PROPN"})
        return min(1.0, symbolic / n * 2.0)

    def _compute_imagery_intensity(self, tokens: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        sensory = sum(1 for t in tokens if t.lower() in _ALL_SENSORY)
        visual = sum(1 for t in tokens if t.lower() in _SENSORY_WORDS["visual"])
        auditory = sum(1 for t in tokens if t.lower() in _SENSORY_WORDS["auditory"])
        tactile = sum(1 for t in tokens if t.lower() in _SENSORY_WORDS["tactile"])
        multimodal = visual + auditory + tactile
        bonus = min(1.0, multimodal / 3.0) * 0.3  # bonus for mixing modalities
        return min(1.0, sensory / n * 2.0 + bonus)

    def _compute_allegorical_marking(self, tokens: list[str],
                                      pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        modals = sum(1 for t in tokens if t.lower() in
                     {"would", "could", "might", "shall", "should",
                      "may", "must", "perhaps", "maybe", "as if",
                      "as though", "seems like"})
        conditionals = sum(1 for t in tokens if t.lower() in
                          {"if", "unless", "provided", "assuming", "suppose"})
        score = modals * 0.2 + conditionals * 0.15
        return min(1.0, score)

    def _compute_connotative_depth(self, tokens: list[str],
                                    pos: list[str]) -> float:
        content = [t.lower() for t, p in zip(tokens, pos)
                   if p in _ABSTRACT_NOUNS or
                   (p in {"NOUN", "VERB", "ADJ"} and len(t) > 3)]
        if not content:
            return 0.0
        total = len(content)
        abstract = sum(1 for w in content if w in _ABSTRACT_NOUNS)
        sensory_total = sum(1 for w in content if w in _ALL_SENSORY)
        symbolic_count = abstract + sensory_total * 0.5
        ratio = symbolic_count / total if total > 0 else 0.0
        return min(1.0, ratio * 2.0)

    @classmethod
    def reset_motif_memory(cls) -> None:
        global _MOTIF_MEMORY, _SEEN_SENTENCES
        _MOTIF_MEMORY.clear()
        _SEEN_SENTENCES = 0
