import math
import numpy as np
from collections import Counter
from src.features.base import BaseFeatureExtractor


_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
_STOP_WORDS = {
    "the", "a", "an", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "shall", "should", "may", "might", "must", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "in", "on", "at", "to",
    "for", "with", "by", "from", "of", "about", "into", "through", "during",
    "and", "or", "but", "nor", "yet", "so", "if", "then", "else", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "some", "any", "no", "not", "only", "very", "just", "as", "up",
    "down", "out", "off", "over", "under", "again", "further", "once", "here",
    "there", "what", "which", "who", "whom", "whose", "than", "too",
}


class SemanticExtractor(BaseFeatureExtractor):
    k_d: int = 8

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        if not tokens:
            return (np.zeros(self.k_d, dtype=float), 0.0)

        n = len(tokens)

        n_content = sum(1 for p in pos if p in _CONTENT_POS)
        lexical_density = n_content / n if n > 0 else 0.0

        word_lengths = [len(t) for t in tokens if t.isalpha()]
        avg_word_len = np.mean(word_lengths) if word_lengths else 0.0
        polysemy = min(1.0, avg_word_len / 12.0)

        content_words = [t.lower() for t, p in zip(tokens, pos) if p in _CONTENT_POS]
        unique_content = len(set(content_words))
        type_token = unique_content / len(content_words) if content_words else 0.0
        concreteness = 1.0 - type_token

        n_stop = sum(1 for t in tokens if t.lower() in _STOP_WORDS)
        stop_ratio = n_stop / n if n > 0 else 0.0
        coherence = 1.0 - abs(stop_ratio - 0.5) * 2.0
        coherence = min(1.0, max(0.0, coherence))

        freq = Counter(t.lower() for t in tokens if t.isalpha())
        total_alpha = sum(freq.values())
        if total_alpha > 0:
            info_density = -sum((c / total_alpha) * math.log(c / total_alpha + 1e-10)
                                for c in freq.values()) / math.log(len(freq) + 1)
        else:
            info_density = 0.0

        n_nouns = sum(1 for p in pos if p == "NOUN" or p == "PROPN")
        noun_ratio = n_nouns / n if n > 0 else 0.0

        n_adjs = sum(1 for p in pos if p == "ADJ")
        adj_ratio = n_adjs / n if n > 0 else 0.0

        verb_ranks = self._verb_sophistication(tokens, pos)
        verb_soph = 1.0 - verb_ranks

        features = np.array([
            lexical_density,
            polysemy,
            concreteness,
            coherence,
            info_density,
            noun_ratio,
            adj_ratio,
            verb_soph,
        ], dtype=float)

        gamma = self.compute_confidence(features)
        return self.normalize(features), gamma

    def _verb_sophistication(self, tokens: list[str], pos: list[str]) -> float:
        common_verbs = {
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call",
        }
        verb_indices = [i for i, p in enumerate(pos) if p == "VERB"]
        if not verb_indices:
            return 0.0
        sophisticated = sum(1 for i in verb_indices
                           if tokens[i].lower() not in common_verbs
                           and tokens[i].isalpha())
        return sophisticated / len(verb_indices)

    def _sentence_length_penalty(self, tokens: list[str]) -> float:
        n = len(tokens)
        if n < 3:
            return 0.2
        elif n < 5:
            return 0.5
        elif n > 40:
            return 0.6
        return 1.0
