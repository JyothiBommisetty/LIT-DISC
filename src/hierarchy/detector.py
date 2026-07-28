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
    """Detects chapter boundaries from text patterns.

    Recognizes:
      - 'Chapter N', 'CHAPTER N', 'Chapitre N'
      - 'Part X', 'Part One'
      - Roman numeral headings at line start
      - '---', '***', '···' line separators
      - Numbered headings like '1.', 'I.'
    """

    _CHAPTER_PATTERNS = [
        re.compile(r'^(?:Chapter|CHAPTER|Chapitre|CHAPITRE)\s+[\dIVXLCDMivxlcdm]+', re.UNICODE),
        re.compile(r'^(?:Part|PART|Book|BOOK|Section|SECTION)\s+[\dIVXLCDMivxlcdm]+', re.UNICODE),
        re.compile(r'^\s*(?:I{1,3}|IV|V|VI{1,3}|IX|X{1,3})\s*\.?\s*$'),
        re.compile(r'^\s*\d+\s*\.\s*$'),
        re.compile(r'^[*-]{3,}$'),
        re.compile(r'^\.{3,}$'),
    ]

    def detect(self, paragraphs: list[list[int]], text: str) -> list[list[int]]:
        lines = text.split('\n')
        chapter_break_indices: set[int] = set()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            for pat in self._CHAPTER_PATTERNS:
                if pat.match(stripped):
                    # This line signals a chapter start; map to paragraph index
                    para_idx = self._line_to_paragraph(lines, i, paragraphs)
                    if para_idx is not None and para_idx > 0:
                        chapter_break_indices.add(para_idx)
                    break

        if not chapter_break_indices:
            if len(paragraphs) >= 4:
                auto_n = max(1, len(paragraphs) // 3)
                chapter_break_indices = {auto_n, 2 * auto_n}
            else:
                return [list(range(len(paragraphs)))]

        sorted_breaks = sorted(chapter_break_indices)
        groups = []
        prev = 0
        for brk in sorted_breaks:
            if brk > prev:
                groups.append(list(range(prev, brk)))
            prev = brk
        if prev < len(paragraphs):
            groups.append(list(range(prev, len(paragraphs))))
        return groups or [list(range(len(paragraphs)))]

    @staticmethod
    def _line_to_paragraph(lines: list[str], line_idx: int,
                           paragraph_sentence_indices: list[list[int]]) -> int | None:
        """Estimate which paragraph index a given line belongs to."""
        # Count non-empty lines up to this point as a rough proxy
        non_empty_before = sum(1 for j in range(line_idx) if lines[j].strip())
        para_count = len(paragraph_sentence_indices)
        total_non_empty = sum(1 for l in lines if l.strip())
        if total_non_empty == 0:
            return None
        ratio = non_empty_before / total_non_empty
        estimated = int(ratio * para_count)
        return max(0, min(para_count - 1, estimated))
