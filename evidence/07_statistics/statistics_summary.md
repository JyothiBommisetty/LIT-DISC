# Statistics Summary — LIT-DISC v1.1

## Gold Standard (25 passages, sigma=5.0)

### Means and Ranges

| Variable | Mean | Std | CV | Min | Max |
|----------|------|-----|----|-----|------|
| LDRI | 0.0277 | 0.0133 | 47.9% | 0.0088 | 0.0551 |
| Raw LDRI | 0.9128 | 0.0240 | 2.6% | 0.8645 | 0.9530 |
| Gamma | 0.0302 | 0.0142 | 47.0% | 0.0096 | 0.0590 |
| Runtime (s) | 0.21 | 0.06 | — | 0.12 | 0.33 |
| Word count | 200.7 | — | — | 166 | 323 |

### Component Means

| Component | Mean | Std | % of Total |
|-----------|------|-----|-----------|
| Uncertainty | 0.8851 | 0.0225 | 88.5% |
| Intrinsic | 0.0394 | 0.0243 | 3.9% |
| Temporal | 0.0042 | 0.0020 | 0.4% |
| Memory | 0.0041 | 0.0020 | 0.4% |
| Interaction | 0.0028 | 0.0013 | 0.3% |
| Resonance | 0.0002 | 0.0001 | 0.02% |

### Feature Means (ranked)

| Feature | Mean | Median | Std |
|---------|------|--------|-----|
| Semantic | 3.50 | 3.49 | 0.50 |
| Narrative | 3.11 | 3.18 | 0.46 |
| Evolution | 2.97 | 2.92 | 0.51 |
| Interaction | 2.65 | 2.55 | 0.56 |
| Symbolic | 2.59 | 2.58 | 0.39 |
| Emotion | 2.55 | 2.66 | 0.51 |

### Variance Decomposition (log-space)

| Source | Full (25 passages) | Breakdown |
|--------|-------------------|-----------|
| var(log(LDRI)) | 0.2609 | — |
| Gamma contribution | **96.5%** | var(log(gamma)) / var(log(LDRI)) |
| Raw contribution | 0.3% | var(log(raw)) / var(log(LDRI)) |
| Covariance | 3.2% | 2·cov(log(gamma), log(raw)) / var(log(LDRI)) |

### Distribution Shape

| Variable | Shape | Skewness | Kurtosis |
|----------|-------|----------|----------|
| LDRI | Positively skewed | +0.60 | −0.61 |
| Raw LDRI | Approximately normal | −0.32 | −0.81 |
| Gamma | Positively skewed | +0.60 | −0.55 |
| Runtime | Positively skewed | +0.65 | −0.58 |

## Full Novel (105 windows, sigma=5.0)

| Variable | Mean | Std | CV | Min | Max |
|----------|------|-----|----|-----|------|
| LDRI | 0.0100 | 0.0022 | 21.8% | 0.0057 | 0.0155 |
| Raw LDRI | 0.9503 | 0.0093 | 1.0% | 0.9202 | 0.9609 |
| Gamma | 0.0105 | 0.0022 | 21.3% | 0.0061 | 0.0162 |
| Intrinsic | 0.0308 | 0.0230 | — | — | — |
| Runtime (s) | 0.81 | 0.16 | — | 0.29 | 1.46 |

### Variance Decomposition (full novel)

| Source | Value |
|--------|-------|
| Gamma contribution | 95.3% |
| Raw contribution | 0.2% |
| Covariance | 4.5% |

## Comparison: Gold Standard vs Full Novel

| Metric | Gold Standard | Full Novel | Ratio |
|--------|-------------|------------|-------|
| LDRI mean | 0.0277 | 0.0100 | 2.8× higher in GS |
| LDRI CV | 47.9% | 21.8% | 2.2× more variation in GS |
| Raw LDRI mean | 0.9128 | 0.9503 | 1.04× higher in novel |
| Raw LDRI CV | 2.6% | 1.0% | 2.6× more variation in GS |
| Gamma mean | 0.0302 | 0.0105 | 2.9× higher in GS |
| Gamma CV | 47.0% | 21.3% | 2.2× more variation in GS |
| Gamma dominance | 96.5% | 95.3% | Essentially unchanged |

## Correlation Summary

| Pair | r (GS) | r (Novel) |
|------|--------|-----------|
| LDRI × Gamma | 0.999+ | 0.999+ |
| Raw LDRI × Gamma | 0.10 | 0.50 |
| LDRI × Runtime | 0.34 | 0.23 |
| LDRI × Word Count | 0.17 | 0.06 |

## Outlier Analysis

| Metric | Lower fence | Upper fence | Outliers |
|--------|------------|------------|----------|
| LDRI | −0.0125 | 0.0649 | 0 |
| Runtime | 0.05s | 0.37s | 0 |
| Gamma | −0.0125 | 0.0709 | 0 |
