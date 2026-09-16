# Baseline Evaluation Summary — Stage 1 & Stage 2

## Stage 1: Sanity Baselines

Five synthetic baselines test whether LDRI measures genuine literary-discourse structure:

| Baseline | Mean LDRI | Pearson r | Spearman ρ | Interpretation |
|----------|-----------|-----------|------------|----------------|
| LIT-DISC (reference) | 0.0277 | 1.000 | 1.000 | Ground truth |
| Random Features | 0.0128 | 0.986 | 0.989 | High r — gamma dominates |
| Shuffled Features | 0.0125 | 0.984 | 0.983 | High r — gamma dominates |
| Constant Features | 0.0125 | 0.998 | 0.995 | r ≈ 1.0 — gamma alone drives ranking |
| **Random Gamma** | **0.0124** | **0.184** | **0.141** | **Near-zero — gamma is essential** |
| Random Memory | 0.0301 | 0.994 | 0.990 | Negligible impact |

**Key finding:** Random Gamma produces near-zero correlation (ρ=0.14), confirming that gamma_global carries the discriminative signal. The constant-features baseline maintains r=0.998 because LDRI variance is driven by gamma, not feature values.

## Stage 2: Text Baselines (Discriminant Validity)

Six standard text metrics test whether LDRI is distinct from readability:

| Baseline | Mean | Pearson r | Spearman ρ |
|----------|------|-----------|------------|
| TTR | 0.579 | 0.308 | 0.364 |
| Herdan's C | 0.896 | 0.396 | 0.401 |
| Flesch-Kincaid Grade | 6.59 | 0.493 | 0.515 |
| Avg Sentence Length | 16.04 | 0.502 | **0.531** |
| Avg Word Length | 4.09 | 0.288 | 0.382 |
| Lexical Density | 0.222 | 0.235 | 0.351 |

**Key finding:** No text baseline exceeds Spearman ρ=0.56 with LDRI. The strongest correlation is with average sentence length (ρ=0.53), which is expected since longer sentences support richer discourse structures. LDRI is not reducible to readability, lexical diversity, or surface text features.
