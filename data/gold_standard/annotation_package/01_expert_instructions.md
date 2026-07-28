# LIT-DISC Expert Annotation Study

## Expert Instructions

### 1. Study Purpose

The Literary Intelligence through Text-based Discourse Analysis (LIT-DISC) project aims to develop and validate a computational framework for measuring the richness of literary discourse. The core metric, the Literary Discourse Richness Index (LDRI), decomposes discourse richness into six theoretically motivated dimensions: Semantic, Narrative, Interaction, Symbolic, Evolution, and Emotion.

This study collects expert literary judgments to serve as ground truth for validating the LDRI metric. Expert ratings will be compared against computational scores produced by the Hierarchical Message Passing on Discourse Trees (HMP-DT) model. The research question is: **does LDRI correlate with expert literary judgments of discourse richness?**

Your ratings will directly shape the development and calibration of the computational framework. This is not a test of your knowledge but an elicitation of your expert judgment.

### 2. Estimated Completion Time

| Activity | Duration |
|----------|----------|
| Reading these instructions | 15 minutes |
| Calibration session (in-person or video call) | 60 minutes |
| Annotating 25 passages | 3-4 hours |
| Total | 4-5 hours |

Annotation may be completed over multiple sessions. The calibration session must be completed before independent annotation begins.

### 3. Eligibility Requirements

Participants must meet at least two of the following criteria:

- Graduate-level training in literary studies, narratology, or a related field (MA, PhD, or current PhD candidate)
- Published research in literary analysis, stylistics, or digital humanities
- Teaching experience in literary analysis at the undergraduate level or above
- Five or more years of professional engagement with literary criticism

No prior knowledge of computational linguistics or machine learning is required.

### 4. Ethical Considerations

This study has been designed in accordance with ethical guidelines for human-subjects research in the humanities.

- **Voluntary participation**: You may withdraw at any time without penalty
- **Data use**: Anonymized ratings will be published as part of an open research dataset
- **Anonymity**: Your identity will be anonymised in all publications (expert ID only)
- **No sensitive content**: The passages are drawn from published literary works. If any passage causes discomfort, you may skip it and note the reason.

### 5. Confidentiality

Your individual ratings will be kept confidential within the research team. Only aggregate inter-annotator agreement statistics and anonymised rating data will be published. The mapping between expert IDs and identities is stored separately and will not be disclosed.

### 6. Annotation Workflow

```
Step 1: Calibration Session (60 min, group)
  ├── Read these instructions
  ├── Review Annotation Manual
  ├── Annotate 3 practice passages as group
  └── Discuss scores and resolve disagreements

Step 2: Independent Annotation (3-4 hours, solo)
  ├── Read passage (2-3 min)
  ├── Assign 7 scores (3-5 min)
  ├── Assign confidence (1 min)
  └── Write optional notes (2 min)
  Repeat for all 25 passages

Step 3: Submit ratings as structured JSON file
```

### 7. How Scores Will Be Used

Your ratings serve three purposes:

1. **Ground truth for validation**: Mean expert ratings per passage will be compared against computational LDRI scores to measure correlation (Spearman's rho). The target is rho >= 0.60.

2. **Dimensional decomposition validation**: Per-dimension expert ratings (Semantic, Narrative, etc.) will be compared against LDRI decomposition components to verify that the computational dimensions correspond to human-judged dimensions.

3. **Inter-annotator reliability benchmarking**: Agreement between experts establishes the upper bound of achievable correlation. Krippendorff's alpha >= 0.60 is the target threshold.

### 8. Compensation

Expert annotators will be compensated at a rate of [AMOUNT] per completed annotation. Payment is issued upon submission of a valid, complete rating file.

---

*This study is conducted as part of the LIT-DISC research project. For questions, contact [PRINCIPAL INVESTIGATOR NAME] at [EMAIL ADDRESS].*
