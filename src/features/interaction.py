"""Interaction feature extractor (I dimension, k_d=7).

Extracts features related to dialogue and social interaction:
speaker-hearer encoding, turn-taking dynamics, dialogue density.

HMP-DT Feature Dictionary, Features C1-C7.
"""
import re
import numpy as np
from src.features.base import BaseFeatureExtractor


# Speech verb patterns
_SPEECH_VERBS = {"said", "asked", "replied", "answered", "whispered",
                 "shouted", "cried", "called", "yelled", "murmured",
                 "muttered", "exclaimed", "shrieked", "barked", "snapped",
                 "began", "continued", "added", "responded", "retorted",
                 "suggested", "offered", "agreed", "protested", "insisted"}

# Question/uncertainty markers
_QUESTION_WORDS = {"what", "where", "when", "why", "how", "who", "whose",
                   "whom", "which", "is", "are", "was", "were", "do",
                   "does", "did", "has", "have", "had", "can", "could",
                   "will", "would", "shall", "should", "may", "might"}

# First-person pronouns (in dialogue, indicate speaker self-reference)
_FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}

# Second-person pronouns (indicate addressee)
_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}

# Imperative markers
_IMPERATIVE_VERBS = {"look", "listen", "watch", "come", "go", "give",
                     "take", "tell", "show", "try", "stop", "wait",
                     "stay", "leave", "help", "see", "get", "make"}

# Hedges and politeness markers
_HEDGES = {"perhaps", "maybe", "possibly", "probably", "might", "could",
           "would", "should", "i think", "i guess", "i suppose", "sort of",
           "kind of", "i wonder", "if you don't mind", "please"}


class InteractionExtractor(BaseFeatureExtractor):
    k_d: int = 7

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        if not tokens:
            return (np.zeros(self.k_d, dtype=float), 0.0)

        n = len(tokens)

        speaker, hearer = self._encode_speaker_hearer(sentence, tokens, pos)
        utterance_length = self._utterance_length(sentence, tokens)
        turn_taking = self._compute_turn_taking(sentence)
        dialogue_density = self._compute_dialogue_density(sentence, tokens, pos)
        speaker_asymmetry = self._compute_speaker_asymmetry(sentence, tokens)
        question_density = self._compute_question_density(sentence, tokens, pos)
        politeness = self._compute_politeness(tokens)

        features = np.array([
            speaker, hearer, utterance_length, turn_taking,
            dialogue_density, speaker_asymmetry, politeness,
        ], dtype=float)

        gamma = self.compute_confidence(features)
        return self.normalize(features), gamma

    def _has_quoted_speech(self, sentence: str) -> bool:
        """Detect if sentence contains quoted speech."""
        quotes = sentence.count("\"")
        s_quotes = sentence.count("\u201c") + sentence.count("\u201d")
        quotes += s_quotes
        return quotes >= 2

    def _extract_quote_content(self, sentence: str) -> list[str]:
        """Extract text within quotes."""
        parts = re.findall(r'["\u201c]([^"\u201d]+)["\u201d]', sentence)
        return parts

    def _encode_speaker_hearer(self, sentence: str, tokens: list[str],
                                pos: list[str]) -> tuple[float, float]:
        has_speech = self._has_quoted_speech(sentence)
        has_speech_verb = any(t.lower() in _SPEECH_VERBS for t in tokens)
        if not has_speech and not has_speech_verb:
            return 0.0, 0.0

        first_p = sum(1 for t in tokens if t.lower() in _FIRST_PERSON)
        second_p = sum(1 for t in tokens if t.lower() in _SECOND_PERSON)
        total = first_p + second_p
        if total == 0:
            return 0.5, 0.5

        speaker = first_p / total if total > 0 else 0.0
        hearer = second_p / total if total > 0 else 0.0
        return float(speaker), float(hearer)

    def _utterance_length(self, sentence: str, tokens: list[str]) -> float:
        parts = self._extract_quote_content(sentence)
        if not parts:
            return 0.0
        total_quote_len = sum(len(q.split()) for q in parts)
        story_len = len(tokens)
        if story_len == 0:
            return 0.0
        ratio = total_quote_len / story_len
        return min(1.0, ratio)

    def _compute_turn_taking(self, sentence: str) -> float:
        parts = self._extract_quote_content(sentence)
        if len(parts) >= 2:
            return min(1.0, (len(parts) - 1) * 0.5)
        return 0.0

    def _compute_dialogue_density(self, sentence: str, tokens: list[str],
                                   pos: list[str]) -> float:
        parts = self._extract_quote_content(sentence)
        in_quote = len(parts) > 0
        has_speech_verb = any(t.lower() in _SPEECH_VERBS for t in tokens)
        if in_quote:
            quote_words = sum(len(q.split()) for q in parts)
            total_words = len(tokens)
            return min(1.0, quote_words / max(1, total_words))
        elif has_speech_verb:
            return 0.3
        return 0.0

    def _compute_speaker_asymmetry(self, sentence: str,
                                    tokens: list[str]) -> float:
        parts = self._extract_quote_content(sentence)
        has_speech = len(parts) > 0
        if not has_speech:
            return 0.0
        first_p = sum(1 for t in tokens if t.lower() in _FIRST_PERSON)
        second_p = sum(1 for t in tokens if t.lower() in _SECOND_PERSON)
        total_p = first_p + second_p
        if total_p == 0:
            return 0.5
        imbalance = abs(first_p - second_p) / total_p
        return float(imbalance)

    def _compute_question_density(self, sentence: str, tokens: list[str],
                                   pos: list[str]) -> float:
        ends_with_question = sentence.strip().endswith("?")
        parts = self._extract_quote_content(sentence)
        q_in_quotes = sum(1 for q in parts if "?" in q)
        q_words = sum(1 for t in tokens if t.lower() in _QUESTION_WORDS)
        score = 0.0
        if ends_with_question:
            score += 0.5
        if q_in_quotes > 0:
            score += 0.3 * q_in_quotes
        score += 0.2 * (q_words / max(1, len(tokens)))
        return min(1.0, score)

    def _compute_politeness(self, tokens: list[str]) -> float:
        hedge_count = sum(1 for t in tokens if t.lower() in _HEDGES)
        please = sum(1 for t in tokens if t.lower() == "please")
        thanks = sum(1 for t in tokens if t.lower() in {"thanks", "thank you"})
        score = hedge_count * 0.25 + please * 0.3 + thanks * 0.3
        return min(1.0, score)
