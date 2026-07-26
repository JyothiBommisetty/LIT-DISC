import re


class ParagraphDetector:
    def detect(self, sentences: list[str], text: str) -> list[list[int]]:
        blocks = re.split(r"\n\s*\n", text.strip())
        para_ranges = []
        char_idx = 0
        text_clean = text.strip()
        para_boundaries = []
        for block in blocks:
            block_start = text_clean.find(block, char_idx)
            if block_start == -1:
                continue
            block_end = block_start + len(block)
            para_boundaries.append((block_start, block_end))
            char_idx = block_end
        sent_boundaries = []
        pos = 0
        for s in sentences:
            start = text_clean.find(s.strip(), pos)
            if start == -1:
                return self._detect_by_count(sentences)
            end = start + len(s.strip())
            sent_boundaries.append((start, end))
            pos = end
        groups = []
        for s_start, s_end in sent_boundaries:
            found = False
            for i, (p_start, p_end) in enumerate(para_boundaries):
                if s_start >= p_start and s_end <= p_end:
                    found = True
                    if len(groups) <= i:
                        groups.append([])
                    break
            if not found:
                if groups:
                    groups[-1].append(len(groups[-1]) + (len(groups[-1]) if groups[-1] else 0))
                else:
                    pass
        sent_idx = 0
        groups.clear()
        for p_start, p_end in para_boundaries:
            para_sents = []
            while sent_idx < len(sentences):
                s_start, s_end = sent_boundaries[sent_idx]
                if s_start >= p_start and s_end <= p_end:
                    para_sents.append(sent_idx)
                    sent_idx += 1
                else:
                    break
            if para_sents:
                groups.append(para_sents)
        while sent_idx < len(sentences):
            if not groups:
                groups.append([])
            groups[-1].append(sent_idx)
            sent_idx += 1
        return groups

    def _detect_by_count(self, sentences: list[str]) -> list[list[int]]:
        n = len(sentences)
        if n <= 3:
            return [list(range(n))]
        chunk = max(1, n // 3)
        groups = []
        for i in range(0, n, chunk):
            groups.append(list(range(i, min(i + chunk, n))))
        return groups


class ChapterDetector:
    def detect(self, paragraphs: list[list[int]], text: str) -> list[list[int]]:
        return [list(range(len(paragraphs)))]
