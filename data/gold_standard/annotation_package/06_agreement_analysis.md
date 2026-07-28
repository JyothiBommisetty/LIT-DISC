# LIT-DISC Agreement Analysis

## Inter-Rater Reliability Computation

---

### 1. Overview

This document specifies how inter-annotator agreement will be computed for the LIT-DISC gold-standard corpus. Agreement analysis serves two purposes:

1. **Quality assurance**: Establishing that expert ratings are reliable enough to serve as ground truth
2. **Upper bound estimation**: Determining the maximum achievable correlation between any computational metric and expert judgment

---

### 2. Krippendorff's Alpha

#### 2.1 Definition

Krippendorff's alpha (alpha) is the primary agreement metric. It measures observed agreement relative to chance agreement, on a scale where 1 = perfect agreement, 0 = chance agreement, and negative values indicate systematic disagreement.

The formula is:

```
alpha = 1 - (D_o / D_e)
```

Where:
- D_o = observed disagreement (based on the chosen distance metric)
- D_e = expected disagreement (based on chance)

For ordinal data (7-point Likert scale), we use the ordinal distance metric:

```
distance(x, y) = (x - y)^2
```

#### 2.2 Interpretation

| alpha Value | Interpretation |
|-------------|----------------|
| >= 0.80 | Excellent agreement |
| 0.67 - 0.79 | Good agreement (acceptable for tentative conclusions) |
| 0.60 - 0.66 | Moderate agreement (minimum acceptable threshold) |
| 0.40 - 0.59 | Weak agreement |
| < 0.40 | Poor agreement |

**Target threshold:** alpha >= 0.60 (overall LDRI dimension). This aligns with standard thresholds in content analysis (Krippendorff, 2004) and is realistic for literary constructs, where inter-annotator agreement typically ranges from 0.40-0.70.

#### 2.3 Computation

Krippendorff's alpha is computed using the `krippendorff` Python package:

```python
import krippendorff
import numpy as np

# reliability_data: shape (num_annotators, num_passages)
# for each dimension separately
alpha = krippendorff.alpha(
    reliability_data=reliability_data,
    level_of_measurement='ordinal'
)
```

**Separate alphas are computed for each of the 7 axes** (S, N, I, Sy, E, Em, O), producing 7 agreement values.

A **multi-dimensional alpha** is also computed using the sum of squared distances across all 7 dimensions as the distance metric.

#### 2.4 Minimum Annotator Requirement

Krippendorff's alpha is reliable with as few as 2 annotators, but 3-5 are recommended. With 2 annotators, alpha is equivalent to a chance-corrected correlation. With 3+, alpha captures agreement among the group.

| Annotators | Alpha reliability | Notes |
|------------|-------------------|-------|
| 2 | Moderate | Use pairwise ICC as secondary metric |
| 3 | Good | Minimum recommended for this study |
| 4-5 | Excellent | Enables outlier detection |

---

### 3. Intraclass Correlation Coefficient (ICC)

#### 3.1 Definition

ICC measures the proportion of total variance attributable to between-passage differences (signal) rather than between-annotator differences (noise). Higher ICC means annotators agree on the relative ordering of passages.

#### 3.2 Model Selection

We use **ICC(2, k)** (two-way random effects, absolute agreement, average of k raters):

- **Two-way**: Each passage is rated by all annotators (crossed design)
- **Random effects**: Annotators are considered a random sample from the population of possible annotators
- **Absolute agreement**: Agreement on exact scores, not just relative ordering
- **Average of k raters**: Reliability of the mean score across all annotators

```python
from pingouin import intraclass_corr

# data format: long-form with columns: 'passage', 'annotator', 'score'
icc = intraclass_corr(
    data=data,
    targets='passage',
    raters='annotator',
    ratings='score'
)

# Extract ICC(2, k)
icc_2k = icc[icc['Type'] == 'ICC2k']['ICC'].values[0]
```

#### 3.3 Interpretation

| ICC Value | Interpretation |
|-----------|----------------|
| >= 0.90 | Excellent for research purposes |
| 0.75 - 0.89 | Good reliability |
| 0.50 - 0.74 | Moderate reliability |
| < 0.50 | Poor reliability |

**Target threshold:** ICC(2,k) >= 0.75 for the Overall LDRI dimension.

---

### 4. Pairwise Agreement

#### 4.1 Pairwise Spearman Correlation

For each pair of annotators, compute Spearman's rank correlation (rho) across all 25 passages, for each dimension separately. This measures how well annotators agree on the **relative ranking** of passages, independent of absolute score differences.

```python
from scipy.stats import spearmanr

rho, p_value = spearmanr(annotator_a_scores, annotator_b_scores)
```

**Target threshold:** Mean pairwise rho >= 0.50.

#### 4.2 Exact Agreement Rate

The proportion of scores on which two annotators give exactly the same value:

```
exact_agreement = count(exact_matches) / total_comparisons
```

**Target threshold:** Exact agreement > 30% (expected to be lower than correlation-based metrics due to the 7-point scale).

#### 4.3 Adjacent Agreement Rate

The proportion of scores on which two annotators differ by at most 1 point:

```
adjacent_agreement = count(diff <= 1) / total_comparisons
```

**Target threshold:** Adjacent agreement > 70%.

---

### 5. Confidence-Weighted Agreement

#### 5.1 Weighting Scheme

Agreement metrics can be weighted by annotator confidence to downweight scores where annotators are uncertain. This is particularly useful for identifying whether disagreements are concentrated in low-confidence passages.

```python
def confidence_weighted_alpha(reliability_data, confidence_data):
    # confidence_data: shape (num_annotators, num_passages)
    # Each entry is 1-5 (confidence)
    weights = confidence_data / 5.0  # normalize to [0.2, 1.0]
    # Apply weights to disagreement calculation
    ...
```

#### 5.2 Comparison

| Metric | Unweighted | Confidence-Weighted | Interpretation |
|--------|------------|---------------------|----------------|
| Krippendorff's alpha | alpha_u | alpha_w | If alpha_w > alpha_u, disagreement concentrates in low-confidence passages |
| Mean score | mean_u | mean_w | Weighted mean gives more influence to confident ratings |

If confidence-weighted alpha is substantially higher (>0.05) than unweighted alpha, this suggests that low-confidence scores are a source of disagreement and the rubric may need clarification for future studies.

---

### 6. Agreement Matrix

A complete agreement analysis produces the following matrix:

| Dimension | Krippendorff's alpha | ICC(2,k) | Mean Pairwise rho | Exact Agree | Adjacent Agree |
|-----------|---------------------|----------|-------------------|-------------|----------------|
| Semantic (S) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Narrative (N) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Interaction (I) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Symbolic (Sy) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Evolution (E) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Emotion (Em) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Overall (O) | 0.00 | 0.00 | 0.00 | 0.0% | 0.0% |
| Multi-dimensional | 0.00 | — | — | — | — |

(Values filled after annotation collection.)

---

### 7. Reporting

Agreement statistics will be reported as follows in publications:

> "Three expert annotators rated 25 literary passages on seven dimensions using a 1-7 Likert scale. Inter-annotator reliability was assessed using Krippendorff's alpha (ordinal distance) and ICC(2,k). Overall discourse richness achieved alpha = [VALUE], indicating [INTERPRETATION]. Per-dimension alphas ranged from [MIN] (for [DIMENSION]) to [MAX] (for [DIMENSION]). The mean pairwise Spearman correlation was [VALUE]. All reported agreement statistics meet or exceed the pre-registered threshold of alpha >= 0.60."

---

### 8. Software

| Metric | Software | Version |
|--------|----------|---------|
| Krippendorff's alpha | `krippendorff` (Python) | >= 0.6 |
| ICC | `pingouin` (Python) | >= 0.5 |
| Spearman correlation | `scipy.stats` (Python) | >= 1.10 |
| Confidence-weighted | Custom implementation | — |

All analysis scripts will be published alongside the dataset.
