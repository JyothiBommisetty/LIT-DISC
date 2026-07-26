import re


_TOKEN_RE = re.compile(
    r"(?:[A-Za-z0-9]+(?:'[A-Za-z]+)?)"           # words with apostrophe
    r"|(?:[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)"       # hyphenated words
    r'|(?:["\u201c\u201d])'                       # opening quotes
    r'|(?:["\u201c\u201d])'                       # closing quotes
    r"|(?:--|\u2014)"                              # dashes
    r"|(?:[.?!]+)"                                 # sentence punctuation
    r"|(?:[,;:()\[\]{}])"                          # other punctuation
    r"|(?:\S)"                                     # any other non-space
)


_NOUN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ist", "er", "or", "ing", "ship", "dom", "hood", "al", "ry", "age", "ure", "ty", "cy", "ny", "th", "ce")
_ADJ_SUFFIXES = ("ous", "ive", "ful", "less", "able", "ible", "al", "ic", "ical", "ly", "ish", "like", "ate", "ent", "ant", "some", "oid", "ular", "ine", "ar", "ory", "ary", "ial", "ual")
_VERB_SUFFIXES = ("ate", "ify", "ize", "ise", "en", "ed", "ing", "er", "ish", "ure")


class Tokenizer:
    def __init__(self, model: str = "en_core_web_sm"):
        self.model_name = model

    def tokenize(self, text: str) -> list[dict]:
        sentences = re.split(r"(?<=[.?!])\s+", text.strip())
        return [self.tokenize_sentence(s) for s in sentences if s.strip()]

    def tokenize_sentence(self, sentence: str) -> dict:
        tokens = _TOKEN_RE.findall(sentence)
        pos = self._guess_pos(tokens)
        lemmas = self._guess_lemmas(tokens)
        return {
            "tokens": tokens,
            "pos": pos,
            "lemmas": lemmas,
            "deps": self._guess_deps(tokens, pos),
        }

    def _guess_pos(self, tokens: list[str]) -> list[str]:
        pos_tags = []
        for token in tokens:
            lower = token.lower()
            if not token.isalpha():
                if any(c in token for c in ".?!"):
                    pos_tags.append("PUNCT")
                elif token in ('"', "'", '"', "'"):
                    pos_tags.append("PUNCT")
                elif token in (",", ";", ":", "(", ")", "[", "]", "{", "}"):
                    pos_tags.append("PUNCT")
                elif token in ("--", "—"):
                    pos_tags.append("PUNCT")
                else:
                    pos_tags.append("SYM")
            elif token[0].isupper() and token.lower() not in ("i", "a"):
                pos_tags.append("PROPN")
            elif token.lower() in ("the", "a", "an"):
                pos_tags.append("DET")
            elif token.lower() in ("and", "or", "but", "nor", "yet", "so", "for"):
                pos_tags.append("CCONJ")
            elif token.lower() in ("in", "on", "at", "to", "for", "with", "by", "from", "of", "about", "into", "through", "during", "before", "after", "above", "below", "between", "under", "over", "without"):
                pos_tags.append("ADP")
            elif token.lower() in ("i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"):
                pos_tags.append("PRON")
            elif token.lower() in ("is", "am", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could", "shall", "should", "may", "might", "must", "need", "dare"):
                pos_tags.append("AUX")
            elif lower.endswith("ly"):
                if lower in ("only", "early", "daily", "weekly", "monthly", "yearly", "silly", "holy"):
                    pos_tags.append("ADJ")
                else:
                    pos_tags.append("ADV")
            elif lower in ("not", "never", "no", "nor", "neither", "hardly", "barely", "scarcely", "rarely", "seldom"):
                pos_tags.append("PART")
            elif lower in ("very", "quite", "rather", "somewhat", "pretty", "almost", "nearly", "really", "just", "too", "so", "more", "most", "less", "least"):
                pos_tags.append("ADV")
            elif lower.endswith(_NOUN_SUFFIXES):
                pos_tags.append("NOUN")
            elif lower.endswith(_ADJ_SUFFIXES):
                pos_tags.append("ADJ")
            elif lower.endswith(_VERB_SUFFIXES):
                pos_tags.append("VERB")
            elif lower.endswith("ed"):
                pos_tags.append("VERB")
            elif lower.endswith("ing"):
                pos_tags.append("VERB")
            elif lower.endswith("s") and len(lower) > 2 and lower[-2] not in "'s":
                pos_tags.append("NOUN")
            elif lower.endswith("n't") or lower.endswith("nt"):
                pos_tags.append("PART")
            else:
                pos_tags.append("NOUN")
        return pos_tags

    def _guess_lemmas(self, tokens: list[str]) -> list[str]:
        lemmas = []
        for token in tokens:
            lower = token.lower()
            if lower.endswith('s') and not lower.endswith('ss') and len(lower) > 2:
                lemmas.append(lower[:-1])
            elif lower.endswith('ing'):
                base = lower[:-3]
                if len(base) >= 2:
                    lemmas.append(base + 'e' if base[-1] not in 'aeiou' else base)
                else:
                    lemmas.append(lower)
            elif lower.endswith('ed'):
                base = lower[:-2]
                if base.endswith('i'):
                    lemmas.append(base[:-1] + 'y')
                else:
                    lemmas.append(base)
            else:
                lemmas.append(lower)
        return lemmas

    def _guess_deps(self, tokens: list[str], pos: list[str]) -> list[tuple]:
        return []
