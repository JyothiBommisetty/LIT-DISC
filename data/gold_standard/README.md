# LIT-DISC Gold Standard Evaluation Corpus v1.0

**25 literary passages** across 5 categories for expert annotation.

## Structure

```
gold_standard/
  passages/              # 25 text files (SNC-01 through NWC-05)
  ratings/               # Expert rating files (one JSON per expert)
  metadata.json          # Passage metadata (author, title, year, genre, etc.)
  ratings_schema.json    # JSON Schema for ratings
  rubric.md              # Scoring rubric for expert annotators
```

## Categories

| Code | Name | Works | Time Period |
|------|------|-------|-------------|
| SNC | Standardized Novel Corpus | 5 classic novels | 1700-1927 |
| MNC | Modern Novel Corpus | 5 contemporary novels | 2000-2017 |
| SSC | Short Story Corpus | 5 short stories | 1899-1999 |
| MLC | Multilingual Parallel Corpus | 5 works in translation | 1857-1967 |
| NWC | Non-Western Narrative Corpus | 5 non-Western works | 1008-1997 |

## Annotation Protocol

Each passage is rated on a **1-7 Likert scale** across 7 axes:

1. **Semantic (S)** — lexical density, wordplay, figurative language
2. **Narrative (N)** — tension curve, pacing, scene structure
3. **Interaction (I)** — dialogue complexity, social interplay
4. **Symbolic (Sy)** — motifs, symbols, allegorical resonance
5. **Evolution (E)** — character change, developmental trajectory
6. **Emotion (Em)** — affective range, intensity dynamics
7. **Overall LDRI** — holistic discourse richness

Plus confidence (1-5) and free-text notes.

## Usage

```bash
# Run LDRI pipeline across all passages
python -c "
from src.pipeline import LITDISCPipeline
import json, glob

p = LITDISCPipeline()
results = {}
for f in sorted(glob.glob('data/gold_standard/passages/*.txt')):
    pid = f.split('\\\\')[-1].split('_')[0]
    r = p.run(f)
    results[pid] = {'ldri': r.ldri.ldri, 'raw': r.ldri.raw,
                    'components': r.ldri.components}
json.dump(results, open('data/gold_standard/ldri_scores.json','w'), indent=2)
"
```
