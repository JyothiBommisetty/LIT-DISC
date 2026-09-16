# Gold Standard Summary — LIT-DISC v1.1

## Dataset

The LIT-DISC Gold Standard Evaluation Corpus v1.0 consists of 25 curated literary passages from 25 authors across 5 categories, annotated for expert validation.

## Passage Statistics

| Statistic | Value |
|-----------|-------|
| Total passages | 25 |
| Total authors | 25 |
| Total categories | 5 |
| Average words/passage | 200.7 |
| Average runtime | 0.21s |
| Total processing time | 5.2s |
| Runtime range | 0.12s–0.33s |

## Categories

| Category | Description | N | Mean LDRI | Mean Gamma |
|----------|-------------|---|-----------|------------|
| SNC | Social/Narrative Classic | 5 | 0.0403 | 0.0428 |
| MLC | Modernist Literary Classic | 5 | 0.0330 | 0.0365 |
| NWC | Non-Western Classic | 5 | 0.0266 | 0.0291 |
| SSC | Short Story Classic | 5 | 0.0197 | 0.0214 |
| MNC | Modern Narrative Contemporary | 5 | 0.0188 | 0.0213 |

**Ranking:** SNC > MLC > NWC > SSC > MNC

## Authors (Top 5 by LDRI)

| Rank | Author | LDRI | Category |
|------|--------|------|----------|
| 1 | Virginia Woolf | 0.0551 | SNC |
| 2 | Marcel Proust | 0.0522 | MLC |
| 3 | Charles Dickens | 0.0511 | SNC |
| 4 | Franz Kafka | 0.0423 | MLC |
| 5 | Fyodor Dostoevsky | 0.0404 | SNC |

## LDRI Statistics

| Statistic | LDRI | Raw LDRI | Gamma |
|-----------|------|----------|-------|
| Mean | 0.0277 | 0.9128 | 0.0302 |
| Median | 0.0275 | 0.9112 | 0.0295 |
| Std | 0.0133 | 0.0240 | 0.0142 |
| Min | 0.0088 | 0.8645 | 0.0096 |
| Max | 0.0551 | 0.9530 | 0.0590 |
| CV | 47.9% | 2.6% | 47.0% |

## Component Statistics (mean ± std)

| Component | Mean | Std | Contribution |
|-----------|------|-----|-------------|
| Uncertainty | 0.8851 | 0.0225 | Dominant |
| Intrinsic | 0.0394 | 0.0243 | Second |
| Temporal | 0.0042 | 0.0020 | Small |
| Memory | 0.0041 | 0.0020 | Small |
| Interaction | 0.0028 | 0.0013 | Small |
| Resonance | 0.0002 | 0.0001 | Negligible |

## Feature Statistics (ranked by mean)

| Feature | Mean | Std |
|---------|------|-----|
| Semantic | 3.499 | 0.497 |
| Narrative | 3.110 | 0.463 |
| Evolution | 2.973 | 0.510 |
| Interaction | 2.645 | 0.560 |
| Symbolic | 2.586 | 0.387 |
| Emotion | 2.551 | 0.507 |
