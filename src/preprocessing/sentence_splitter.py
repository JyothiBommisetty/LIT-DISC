import re


ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "dept", "est", "govt", "inc", "ltd", "co", "corp", "vs", "etc",
    "e.g", "i.e", "al", "ch", "vol", "no", "pp", "p", "fig",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "gen", "col", "maj", "capt", "lt", "sgt", "cpl", "pvt",
    "repr", "sen", "gov", "pres", "dept",
}


class SentenceSplitter:
    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"([.?!])(\n)", r"\1 \2", text)
        text = re.sub(r"(\.)([A-Z])", r"\1 \2", text)
        candidates = re.split(r"(?<=[.?!])\s+", text)
        result = []
        for i, cand in enumerate(candidates):
            stripped = cand.strip()
            if not stripped:
                continue
            if i < len(candidates) - 1 and result:
                words = stripped.split()
                if words and words[-1].rstrip(".").lower() in ABBREVIATIONS:
                    result[-1] = result[-1] + " " + stripped
                    continue
                if stripped.endswith('"') or stripped.endswith("'"):
                    stripped = stripped[:-1]
                words = stripped.split()
                if len(words) == 1 and words[0].endswith("."):
                    result[-1] = result[-1] + " " + stripped
                    continue
            result.append(stripped)
        return result
