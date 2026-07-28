# LIT-DISC Data Validation

## Automated JSON Validation Specification

---

### 1. Validation Script

The following Python script validates expert rating files automatically. It performs all checks described in Part 5 (Quality Control) and produces a detailed validation report.

The file is at: `data/gold_standard/annotation_package/validate_ratings.py`

---

### 2. Validation Schema

The expected JSON structure is formally defined in `data/gold_standard/ratings_schema.json`. The validator enforces this schema.

**Required structure:**

```json
{
  "expert": {
    "id": "string (required, e.g., 'EXPERT-01')",
    "background": "string (required)",
    "date": "string (required, ISO date format 'YYYY-MM-DD')"
  },
  "ratings": [
    {
      "passage_id": "string (required, must match corpus IDs)",
      "scores": {
        "semantic": "integer (1-7, required)",
        "narrative": "integer (1-7, required)",
        "interaction": "integer (1-7, required)",
        "symbolic": "integer (1-7, required)",
        "evolution": "integer (1-7, required)",
        "emotion": "integer (1-7, required)",
        "overall": "integer (1-7, required)"
      },
      "confidence": "integer (1-5, required)",
      "notes": "string (optional)"
    }
  ]
}
```

---

### 3. Valid Passage IDs

The corpus contains exactly 25 passages:

```
SNC-01, SNC-02, SNC-03, SNC-04, SNC-05
MNC-01, MNC-02, MNC-03, MNC-04, MNC-05
SSC-01, SSC-02, SSC-03, SSC-04, SSC-05
MLC-01, MLC-02, MLC-03, MLC-04, MLC-05
NWC-01, NWC-02, NWC-03, NWC-04, NWC-05
```

All 25 must be present exactly once.

---

### 4. Validation Output

The validator produces a report with three sections:

```
VALIDATION REPORT
=================
File: EXPERT-01_ratings.json
Status: PASS / PASS WITH WARNINGS / HARD REJECT

SCHEMA:    PASS
RANGES:    PASS
COMPLETE:  PASS
CONSISTENT: WARNING (passage SSC-03: all scores identical)
DUPLICATES: PASS

WARNINGS:
- SSC-03: Uniform scoring (all 7 axes = 4)

ERRORS:
  (none)
```
