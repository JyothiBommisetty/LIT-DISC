# LIT-DISC Expert Annotation Package

## Ready-to-Use Materials for a Formal Expert Annotation Study

This directory contains all materials required to conduct a formal expert annotation study for the LIT-DISC project's gold-standard evaluation corpus.

---

### Package Contents

| File | Description |
|------|-------------|
| `01_expert_instructions.md` | Professional study overview for expert participants |
| `02_annotation_manual.md` | Detailed scoring guide for all 7 axes with examples |
| `03_calibration_guide.md` | Training session plan with 3 practice passages |
| `04_annotation_protocol.md` | Complete per-passage workflow specification |
| `05_quality_control.md` | Automated and manual QC procedures with rejection criteria |
| `06_agreement_analysis.md` | Krippendorff's alpha, ICC, pairwise agreement computation |
| `07_data_validation.md` | JSON validation specification |
| `08_faq.md` | Frequently asked questions for expert annotators |
| `09_checklist.md` | Pre-submission checklist for annotators |
| `validate_ratings.py` | Automated JSON validator (run on submitted files) |

---

### Required Companion Files

- **Passages**: `data/gold_standard/passages/` (25 text files)
- **Metadata**: `data/gold_standard/metadata.json`
- **Rubric**: `data/gold_standard/rubric.md` (condensed reference)
- **Schema**: `data/gold_standard/ratings_schema.json`

---

### Workflow for Research Team

```
1. Recruit 3-5 expert annotators
2. Distribute this package + passage files
3. Schedule and conduct calibration session (60 min)
4. Annotators complete independent annotation (3-4 hours)
5. Collect JSON files → run validate_ratings.py
6. Resolve validation issues with annotators
7. Compute inter-annotator agreement (Part 6)
8. Compute mean expert scores per passage per dimension
9. Compare against computational LDRI scores
```

---

### File Naming Convention

| Item | Pattern | Example |
|------|---------|---------|
| Expert rating | `[EXPERT-ID]_ratings.json` | `EXPERT-01_ratings.json` |
| Validation output | `[EXPERT-ID]_validation.txt` | `EXPERT-01_validation.txt` |

---

### Validation Command

```bash
python validate_ratings.py EXPERT-01_ratings.json [more files...]
```

Output: pass/fail per check with detailed error messages for hard rejects.

---

### Inter-Annotator Agreement Script (TBD)

A separate script for computing Krippendorff's alpha, ICC, and pairwise agreement will be provided after all annotations are collected. See `06_agreement_analysis.md` for the specification.
