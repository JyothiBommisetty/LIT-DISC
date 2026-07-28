# LIT-DISC Quality Control Procedures

## Ensuring Data Quality in Expert Annotations

---

### 1. Overview

Quality control (QC) operates at three levels:

| Level | Stage | Responsibility |
|-------|-------|----------------|
| Level 1 | Pre-submission | Self-check by annotator |
| Level 2 | Automated | Validation script on submission |
| Level 3 | Manual | Review by research team |

This document defines all three levels.

---

### 2. Pre-Submission Self-Check (Level 1)

Before submitting, annotators should verify:

**Completeness:**
- [ ] All 25 passages have been scored
- [ ] All 7 axes have been scored for each passage
- [ ] Confidence has been assigned for each passage

**Range:**
- [ ] All scores are integers between 1 and 7
- [ ] All confidence values are integers between 1 and 5

**Consistency:**
- [ ] No passage has the same score for all 7 axes (unlikely in a diverse corpus)
- [ ] Notes are free of identifying information (names, institutions)

**File integrity:**
- [ ] File is valid JSON
- [ ] File is named correctly: `[EXPERT-ID]_ratings.json`
- [ ] Expert ID in filename matches expert ID in the JSON content

---

### 3. Automated QC Checks (Level 2)

Upon submission, the validation script performs the following checks.

#### 3.1 Schema Compliance

| Check | Criteria | Rejection |
|-------|----------|-----------|
| Top-level structure | Must contain `expert` object and `ratings` array | Hard reject |
| Expert object | Must contain `id`, `background`, `date` | Hard reject |
| Ratings array | Must be a list of 25 rating objects | Hard reject |
| Rating object keys | Must contain `passage_id`, `scores`, `confidence` | Hard reject |
| Scores object | Must contain exactly 7 keys: `semantic`, `narrative`, `interaction`, `symbolic`, `evolution`, `emotion`, `overall` | Hard reject |
| Notes | Optional; if present must be string | Soft warning |

#### 3.2 Range Compliance

| Check | Criteria | Rejection |
|-------|----------|-----------|
| Score range | Each score is integer in [1, 7] | Hard reject |
| Confidence range | Each confidence is integer in [1, 5] | Hard reject |

#### 3.3 Completeness

| Check | Criteria | Rejection |
|-------|----------|-----------|
| Passage coverage | All 25 passage IDs present exactly once | Hard reject |
| Score completeness | Each passage has all 7 scores | Hard reject |
| No null values | No null, undefined, or missing values | Hard reject |

#### 3.4 Consistency Checks

| Check | Criteria | Warning |
|-------|----------|---------|
| Uniform scoring | Same score for all 7 axes on any passage | Soft warning |
| Uniform confidence | Same confidence for all 25 passages | Soft warning |
| Extreme patterns | All scores 6-7 or all 1-2 across all passages | Soft warning |
| Score-creep | Scores monotonically increase or decrease across passages | Soft warning |

#### 3.5 Duplicate Detection

| Check | Criteria | Rejection |
|-------|----------|-----------|
| Duplicate scores | Two passages with identical 7-score vectors | Soft warning |
| Duplicate passages | Same passage_id twice | Hard reject |

#### 3.6 Rejection Criteria

A submission receives a **hard reject** if any hard-reject check fails. The annotator receives a detailed error report and must resubmit within 7 days.

A submission receives a **soft warning** if any warning flag is raised. The annotator is asked to confirm the flagged scores but may proceed without resubmission.

---

### 4. Manual QC Review (Level 3)

After automated validation, the research team reviews:

#### 4.1 Annotation Notes Review

- Notes are checked for: clarity, specificity, relevance
- Vague or generic notes (e.g., "seems fine") may trigger a follow-up
- Notes revealing misunderstanding of axes trigger a clarification conversation

#### 4.2 Pattern Analysis

The research team examines:

- **Annotator bias**: consistently high or low relative to group mean
- **Central tendency**: clustering scores around 3-4 with no extremes
- **Range restriction**: using fewer than 4 of the 7 scale points
- **Dimension confusion**: high correlation between dimensions that should be distinct (e.g., Semantic and Symbolic correlated at r > 0.90)

#### 4.3 Attention Check Passages

Two passages in the corpus serve as attention checks:

- **Passage A** (SNC-05): Hemingway's minimalist dialogue. Low Semantic, moderate Interaction, low Symbolic. An annotator scoring this as 6 across all dimensions may not be attending carefully.
- **Passage B** (SSC-02): Joyce's *The Dead* closing. Very high Semantic, Symbolic, Emotion. An annotator scoring this as 2 across all dimensions may not be attending carefully.

If both attention checks fail (both scored opposite to expectation by >3 points), the submission is flagged for follow-up.

---

### 5. Response to QC Findings

| Outcome | Action | Timeline |
|---------|--------|----------|
| Pass all checks | Rating accepted; expert compensated | Within 5 working days |
| Soft warnings only | Rating accepted; annotator notified of flags | Within 5 working days |
| Hard reject (correctable) | Annotator receives error report; resubmission requested | Within 48 hours of submission |
| Hard reject (systematic) | Annotator invited to discuss with research team | Within 1 week |
| Repeat violations | Annotator may be replaced; partial compensation for valid passages | Case-by-case |

---

### 6. Quality Metrics

After all annotations are collected, the following quality metrics are computed:

| Metric | Target | Action if below target |
|--------|--------|------------------------|
| Krippendorff's alpha (overall) | >= 0.60 | Review calibration; consider dropping outlier annotator |
| Pairwise Spearman rho | >= 0.50 | Flag annotator pairs for review |
| Complete submission rate | >= 80% | Review protocol burden; consider simplifying |
| Hard reject rate | <= 10% | Review instructions clarity; improve template |
| Mean confidence | >= 3.0 | If low, review rubric clarity |

---

### 7. Data Exclusion Criteria

An annotator's data may be excluded from analysis if:

1. **Systematic bias**: Annotator's scores deviate from group mean by >2 standard deviations on >50% of passages (checked after at least 3 annotators have submitted)
2. **Range restriction**: Annotator uses <= 3 distinct score values across all 25 passages (e.g., only scores 3 and 4)
3. **Inverted scoring**: Annotator consistently scores known high-richness passages low and low-richness passages high
4. **Incomplete**: Annotator does not resubmit after hard reject within 14 days

Excluded data is flagged but retained for transparency. The exclusion decision is documented with rationale.
