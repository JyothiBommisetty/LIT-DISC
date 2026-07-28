# LIT-DISC Annotation Protocol

## Complete Workflow Specification

---

### 1. Preparation

Before beginning annotation, ensure you have:

- [ ] Read the Expert Instructions (01_expert_instructions.md)
- [ ] Read the Annotation Manual (02_annotation_manual.md)
- [ ] Completed the Calibration Session (03_calibration_guide.md)
- [ ] Downloaded the 25 passage files from the shared repository
- [ ] Downloaded the submission template (JSON format)
- [ ] Confirmed your expert ID with the research team

---

### 2. Annotation Workflow

For each passage, follow the sequence below without deviation. Estimated time per passage: 8-12 minutes.

```
╔════════════════════════════════════════════╗
║          PER-PASSAGE WORKFLOW              ║
╠════════════════════════════════════════════╣
║                                            ║
║  Step 1: Read Passage (2-3 min)            ║
║    ↓                                       ║
║  Step 2: Score Semantic (S) (1 min)        ║
║    ↓                                       ║
║  Step 3: Score Narrative (N) (1 min)       ║
║    ↓                                       ║
║  Step 4: Score Interaction (I) (1 min)     ║
║    ↓                                       ║
║  Step 5: Score Symbolic (Sy) (1 min)       ║
║    ↓                                       ║
║  Step 6: Score Evolution (E) (1 min)       ║
║    ↓                                       ║
║  Step 7: Score Emotion (Em) (1 min)        ║
║    ↓                                       ║
║  Step 8: Score Overall (O) (1 min)         ║
║    ↓                                       ║
║  Step 9: Assign Confidence (30 sec)        ║
║    ↓                                       ║
║  Step 10: Write Notes (optional, 2 min)    ║
║                                            ║
╚════════════════════════════════════════════╝
```

#### Step 1: Read Passage

Read the passage through once at a natural reading pace. Do not take notes during the first reading. Allow the passage to make its initial impression.

Read the passage a second time, more slowly. This time, attend to the literary features described in the Annotation Manual. You may underline or mark the text.

**Recommendation:** Read aloud for passages with prominent dialogue or rhythmic prose.

#### Step 2-8: Assign Scores

Score in the listed order (S, N, I, Sy, E, Em, O) without revisiting previous scores. This prevents the overall score from being a simple average of dimensions.

For each score:
- Refer to the Annotation Manual axis description
- Consider the full range (1-7); avoid central tendency bias (scoring everything 3-5)
- Score based on what is present in the passage, not on what you imagine the full work might contain
- If torn between two scores, choose the higher one and note the uncertainty

#### Step 9: Assign Confidence

Rate your overall confidence in the scoring for this passage on a 1-5 scale:

| Score | Meaning |
|-------|---------|
| 1 | Guessing; passage is unusually difficult to judge |
| 2 | Several scores could be plausible |
| 3 | Reasonably sure of most scores |
| 4 | Confident in scoring pattern |
| 5 | Certain; passage clearly displays its richness profile |

Confidence is per-passage, not per-dimension. It reflects your overall certainty.

#### Step 10: Write Notes (Optional)

Notes are encouraged but not required. Useful notes include:
- Which specific features drove high or low scores
- Sources of uncertainty
- Comparisons to other passages
- Any contextual knowledge that influenced scoring (e.g., "This is the climax of the novel")

---

### 3. Timing Recommendations

| Session | Recommended Duration | Break |
|---------|---------------------|-------|
| Session 1: Calibration | 60 min continuous | — |
| Session 2: Passages 1-8 | 90 min | 10 min after passage 4 |
| Session 3: Passages 9-16 | 90 min | 10 min after passage 12 |
| Session 4: Passages 17-25 | 90 min | 10 min after passage 21 |

**Do not annotate more than 8 passages in a single session.** Fatigue degrades judgment quality.

**Do not annotate more than 25 passages in a single day.** Spread across at least 3 sessions.

**Take a 10-minute break between every 4 passages** even if you feel you do not need it.

---

### 4. Submission Format

Submit ratings as a single JSON file named `[EXPERT_ID]_ratings.json`.

**File structure:**

```json
{
  "expert": {
    "id": "EXPERT-01",
    "background": "PhD candidate, Comparative Literature",
    "date": "2026-08-01"
  },
  "ratings": [
    {
      "passage_id": "SNC-01",
      "scores": {
        "semantic": 6,
        "narrative": 4,
        "interaction": 7,
        "symbolic": 3,
        "evolution": 3,
        "emotion": 5,
        "overall": 5
      },
      "confidence": 4,
      "notes": "Austen's dialogue carries the passage; subtext is exceptionally rich."
    },
    {
      "passage_id": "SNC-02",
      "scores": { "...": "..." },
      "confidence": 3,
      "notes": "..."
    }
  ]
}
```

**Validation rules:**
- All 25 passages must be included
- Each passage must have all 7 scores (each 1-7)
- Confidence must be 1-5
- Notes are optional (empty string or omit)
- Passage IDs must match the corpus exactly: `SNC-01` through `NWC-05`

See Part 7 (Data Validation) for automatic validation details.

---

### 5. File Naming Convention

| Item | Naming Convention | Example |
|------|-------------------|---------|
| Rating file | `[EXPERT-ID]_ratings.json` | `EXPERT-01_ratings.json` |
| Notes (optional supplementary) | `[EXPERT-ID]_notes.txt` | `EXPERT-01_notes.txt` |

---

### 6. Resubmission Policy

If validation detects errors, you will receive a validation report within 48 hours of submission. You may resubmit a corrected file. Resubmissions must be complete files (not partial updates).

| Error Type | Resolution | Deadline |
|------------|------------|----------|
| Schema violation | Correct structure and resubmit | 7 days |
| Missing passage | Add scores and resubmit | 7 days |
| Out-of-range score | Adjust and resubmit | 7 days |
| Score correction (annotator-initiated) | Submit corrected file with change log | 14 days |

---

### 7. Contact

For questions during annotation:

- **Protocol questions**: [RESEARCHER NAME] — [EMAIL]
- **Technical issues (file format, submission)**: [TECHNICAL CONTACT] — [EMAIL]
- **Content questions (passage text)**: [RESEARCHER NAME] — [EMAIL]

*Expected response time: within 24 hours during working days.*
