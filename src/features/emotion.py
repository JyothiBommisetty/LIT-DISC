"""Emotion feature extractor (Em dimension, k_d=7).

Extracts features related to emotional content: valence, arousal,
intensity, emotional diversity, and trajectory.

HMP-DT Feature Dictionary, Features F1-F7.
"""
import math
import numpy as np
from collections import Counter
from src.features.base import BaseFeatureExtractor


# Positive valence words (general sentiment)
_POSITIVE_WORDS = {
    "love", "joy", "happy", "beautiful", "wonderful", "great", "excellent",
    "amazing", "fantastic", "delight", "pleasure", "bliss", "ecstasy",
    "elated", "thrilled", "glad", "cheerful", "optimistic", "hopeful",
    "grateful", "thankful", "peaceful", "calm", "serene", "tender",
    "gentle", "warm", "comfort", "ease", "relief", "content", "satisfied",
    "proud", "confident", "brave", "strong", "free", "alive", "bright",
    "sunshine", "smile", "laugh", "laughter", "celebrate", "triumph",
    "victory", "success", "achievement", "blessing", "miracle", "dream",
    "passion", "desire", "longing", "hope", "caring", "kindness",
    "compassion", "mercy", "forgive", "forgiveness", "grace", "beauty",
}

# Negative valence words
_NEGATIVE_WORDS = {
    "hate", "anger", "rage", "fury", "sad", "sadness", "grief", "sorrow",
    "despair", "hopeless", "misery", "suffering", "pain", "agony",
    "anguish", "torment", "dread", "fear", "terror", "horror", "panic",
    "anxiety", "worry", "doubt", "shame", "guilt", "regret", "remorse",
    "bitter", "resent", "envy", "jealous", "greed", "selfish", "cruel",
    "violent", "brutal", "harsh", "dark", "gloom", "desolate", "lonely",
    "isolated", "abandoned", "rejected", "betrayed", "hurt", "wounded",
    "broken", "destroyed", "lost", "empty", "hollow", "numb", "tired",
    "exhausted", "weary", "disgust", "contempt", "scorn", "disappointed",
    "frustrated", "defeated", "helpless", "worthless", "inferior",
}

# High-arousal words
_HIGH_AROUSAL = {
    "ecstasy", "rage", "terror", "panic", "excitement", "fury", "passion",
    "thrill", "desperation", "hysteria", "frantic", "wild", "intense",
    "explosive", "volatile", "urgent", "violent", "fierce", "furious",
    "frantic", "desperate", "manic", "frantic", "erupt", "explode",
    "scream", "shout", "run", "race", "fight", "attack", "crash",
    "shatter", "blast", "surge", "rush", "burst", "blaze", "flame",
}

# Low-arousal words
_LOW_AROUSAL = {
    "calm", "peaceful", "serene", "tranquil", "gentle", "quiet", "still",
    "relaxed", "sleepy", "drowsy", "lazy", "lethargic", "sluggish",
    "bored", "tired", "weary", "exhausted", "numb", "dull", "passive",
    "placid", "restful", "mellow", "soft", "subtle", "faint", "mild",
    "moderate", "steady", "stable", "composed", "collected",
}

# Basic emotion categories (for diversity)
_EMOTION_CATEGORIES = {
    "joy": {"joy", "happy", "delight", "ecstasy", "bliss", "elated",
            "thrilled", "cheerful", "glad", "content", "pleased"},
    "sadness": {"sad", "sorrow", "grief", "despair", "misery", "melancholy",
                "gloom", "gloomy", "mournful", "somber", "bleak"},
    "anger": {"anger", "rage", "fury", "resent", "envy", "jealous",
              "bitter", "frustrated", "irritated", "annoyed", "hostile"},
    "fear": {"fear", "terror", "horror", "dread", "panic", "anxiety",
             "worry", "alarm", "fright", "scared", "afraid", "nervous"},
    "surprise": {"surprise", "astonish", "amaze", "shock", "stun",
                 "startle", "wonder", "awe", "bewilder", "dumbfound"},
    "disgust": {"disgust", "revulsion", "contempt", "scorn", "loathing",
                "aversion", "repulsion", "distaste"},
    "trust": {"trust", "confidence", "faith", "belief", "reliance",
              "dependence", "loyalty", "acceptance"},
    "anticipation": {"anticipation", "expectation", "hope", "longing",
                     "desire", "yearning", "eager", "excited", "curious"},
}

# Intensity amplifiers
_INTENSIFIERS = {"very", "extremely", "incredibly", "absolutely",
                 "completely", "totally", "utterly", "deeply", "profoundly",
                 "intensely", "powerfully", "strongly", "terribly",
                 "horribly", "dreadfully", "awfully", "so", "too", "such",
                 "quite", "rather", "remarkably", "extraordinarily"}

# First and second person pronouns (emotional involvement)
_INVOLVEMENT_WORDS = {"i", "me", "my", "mine", "we", "us", "our", "ours",
                      "you", "your", "yours"}


class EmotionExtractor(BaseFeatureExtractor):
    k_d: int = 7

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[np.ndarray, float]:
        if not tokens:
            return (np.zeros(self.k_d, dtype=float), 0.0)

        n = len(tokens)

        valence = self._compute_valence(tokens, pos)
        arousal = self._compute_arousal(tokens, pos)
        intensity = self._compute_intensity(tokens, pos)
        emotional_diversity = self._compute_emotional_diversity(tokens)
        sentiment_extremity = self._compute_sentiment_extremity(tokens, pos)
        emotional_contagion = self._compute_emotional_contagion(tokens, pos)
        valence_variance = self._compute_valence_variance(sentence, tokens, pos)

        features = np.array([
            valence, arousal, intensity, emotional_diversity,
            sentiment_extremity, emotional_contagion, valence_variance,
        ], dtype=float)

        gamma = self.compute_confidence(features)
        return self.normalize(features, bounds=(-1.0, 1.0)), gamma

    def _compute_valence(self, tokens: list[str],
                          pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        pos_count = sum(1 for t in tokens if t.lower() in _POSITIVE_WORDS)
        neg_count = sum(1 for t in tokens if t.lower() in _NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / max(1, total)

    def _compute_arousal(self, tokens: list[str],
                          pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        high = sum(1 for t in tokens if t.lower() in _HIGH_AROUSAL)
        low = sum(1 for t in tokens if t.lower() in _LOW_AROUSAL)
        total_affect = high + low
        if total_affect == 0:
            return 0.0
        return high / total_affect

    def _compute_intensity(self, tokens: list[str],
                            pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        all_emotion = (set(_POSITIVE_WORDS) | set(_NEGATIVE_WORDS) |
                       set(_HIGH_AROUSAL) | set(_LOW_AROUSAL))
        emotion_tokens = sum(1 for t in tokens if t.lower() in all_emotion)
        intensifiers = sum(1 for t in tokens if t.lower() in _INTENSIFIERS)
        exclamation = sum(1 for t in tokens if "!" in t)
        raw_intensity = (emotion_tokens / n * 0.6 +
                        intensifiers / n * 0.3 +
                        exclamation * 0.1)
        return min(1.0, raw_intensity)

    def _compute_emotional_diversity(self, tokens: list[str]) -> float:
        activated = set()
        for t in tokens:
            tl = t.lower()
            for cat, words in _EMOTION_CATEGORIES.items():
                if tl in words:
                    activated.add(cat)
        n_categories = len(activated)
        return n_categories / len(_EMOTION_CATEGORIES)

    def _compute_sentiment_extremity(self, tokens: list[str],
                                      pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        neg_count = sum(1 for t in tokens if t.lower() in _NEGATIVE_WORDS)
        pos_count = sum(1 for t in tokens if t.lower() in _POSITIVE_WORDS)
        total_affect = neg_count + pos_count
        if total_affect == 0:
            return 0.0
        majority = max(pos_count, neg_count)
        extremeness = majority / total_affect
        return float(extremeness)

    def _compute_emotional_contagion(self, tokens: list[str],
                                      pos: list[str]) -> float:
        n = len(tokens)
        if n == 0:
            return 0.0
        involvement = sum(1 for t in tokens if t.lower() in _INVOLVEMENT_WORDS)
        return min(1.0, involvement / n * 2.0)

    def _compute_valence_variance(self, sentence: str, tokens: list[str],
                                    pos: list[str]) -> float:
        all_tokens = [t.lower() for t in tokens]
        if not all_tokens:
            return 0.0
        window_size = max(2, len(all_tokens) // 3)
        if window_size < 2:
            return 0.0
        valences = []
        for i in range(0, len(all_tokens), window_size):
            window = all_tokens[i:i + window_size]
            pos_count = sum(1 for w in window if w in _POSITIVE_WORDS)
            neg_count = sum(1 for w in window if w in _NEGATIVE_WORDS)
            total = pos_count + neg_count
            if total > 0:
                valences.append((pos_count - neg_count) / total)
        if len(valences) < 2:
            return 0.0
        mean_v = np.mean(valences)
        variance = float(np.var(valences))
        return min(1.0, variance * 5.0)
