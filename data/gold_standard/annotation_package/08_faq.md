# LIT-DISC Expert Annotation: Frequently Asked Questions

---

### 1. General Questions

**Q: I am not a computational linguist. Can I still participate?**
A: Absolutely. This study requires literary expertise, not computational skills. The scoring system is designed for literature scholars. You do not need to understand the LIT-DISC framework to score passages.

**Q: How long will this take?**
A: Approximately 4-5 hours total: 1 hour for the calibration session and 3-4 hours for annotating 25 passages. We recommend spreading this over at least 3 sessions on separate days.

**Q: Can I annotate on my own schedule?**
A: Yes. The calibration session is the only synchronous activity. After that, you may annotate at your own pace, as long as the completed file is submitted by the deadline.

**Q: Can I skip passages I find difficult?**
A: All 25 passages must be scored for the submission to be valid. However, you may use the confidence scale to indicate uncertainty, and you may add notes explaining what made the passage difficult.

---

### 2. Scoring Questions

**Q: What does a score of 4 mean?**
A: A score of 4 (Moderate) means the dimension is "clearly present with some variation or nuance; competent." It is the midpoint of the scale. Not all passages will score 4 — rich passages will score higher, simpler passages lower.

**Q: Should I use the full range of the scale?**
A: Yes. The 25 passages span from functional prose (SNC-05 Hemingway) to maximal richness (SSC-02 Joyce). Use the full 1-7 range to distinguish them. Central tendency bias (scoring everything 3-5) reduces the informativeness of the data.

**Q: What if a passage has no dialogue?**
A: Score Interaction as 1 (Absent). This is expected for solitary passages. The overall LDRI score can still be high if other dimensions are rich.

**Q: What if I am torn between two scores?**
A: Choose the higher score and note the uncertainty in your comments. Alternatively, note it during calibration and the group can discuss.

**Q: Can I revise a score after moving to the next passage?**
A: No. The protocol asks you to score in sequence (S, N, I, Sy, E, Em, O) without revising previous scores. If you later realize an error, note it and explain in the submission notes.

---

### 3. Technical Questions

**Q: How do I submit my ratings?**
A: Fill in the JSON template and save it as `[EXPERT-ID]_ratings.json`. Submit via the shared repository or email attachment as instructed by the research team.

**Q: What if I make a mistake in the JSON format?**
A: The research team will run your file through an automated validator. If errors are found, you will receive a detailed report and may resubmit within 7 days.

**Q: What if I lose my work?**
A: Save your progress frequently. We recommend saving after every 5 passages. If using a cloud drive, enable offline access.

**Q: Can I use a spreadsheet instead of JSON?**
A: No. The submission must be in JSON format for automated processing. If you prefer to draft in a spreadsheet, you may use this CSV format and we can convert it:

```
passage_id,semantic,narrative,interaction,symbolic,evolution,emotion,overall,confidence,notes
SNC-01,6,4,7,3,3,5,5,4,"Austen's dialogue..."
```

Contact the research team for CSV-to-JSON conversion assistance.

---

### 4. Rubric Questions

**Q: How is Symbolic richness different from Semantic richness?**
A: Semantic richness concerns **language** (vocabulary, metaphor, wordplay). Symbolic richness concerns **meaning** carried by objects, images, and patterns across the text. A passage can be semantically rich (dense, sophisticated vocabulary) without being symbolic (no recurring motifs), and vice versa.

**Q: How is Emotion richness different from Evolution richness?**
A: Emotion richness concerns **affective states** at a given moment. Evolution richness concerns **change** over time. A static passage with intense feeling is high Emotion, low Evolution. A passage showing character change without explicit emotional expression would be the reverse.

**Q: What is the Overall LDRI score supposed to capture that the individual dimensions miss?**
A: The overall score captures **integration and interaction** between dimensions. A passage where high Semantic, high Symbolic, and high Emotion work together to produce a unified effect may deserve a higher overall score than the average of its dimensions suggests.

---

### 5. Calibration Questions

**Q: What if I disagree with the group during calibration?**
A: Disagreement is expected and productive. The calibration session is designed to surface differences in interpretation, not to enforce conformity. You are free to score according to your own judgment during independent annotation.

**Q: Do I have to attend the calibration session?**
A: Yes. The calibration session is mandatory. It ensures all annotators share a baseline understanding of the rubric, which is necessary for reliable data collection.

**Q: Can I annotate before the calibration session?**
A: No. Please complete all annotation after the calibration session. Doing annotation beforehand defeats the purpose of calibration.

---

### 6. Data and Privacy

**Q: Will my individual scores be published?**
A: Anonymised scores (identified only by expert ID, e.g., "EXPERT-01") will be published as part of an open research dataset. Your name will not appear in any publication.

**Q: Can I withdraw after submitting?**
A: Yes. You may withdraw your data within 30 days of submission by contacting the research team. After 30 days, data may be incorporated into analyses and cannot be easily removed.

**Q: Will I be named in the acknowledgements?**
A: Yes, unless you prefer to remain anonymous. Please indicate your preference to the research team.

---

### 7. Compensation

**Q: When will I be paid?**
A: Payment is issued within 30 days of a valid, complete submission passing automated validation.

**Q: What if my submission fails validation?**
A: You will receive a detailed error report and may resubmit a corrected file within 7 days. Payment is tied to a valid submission.

**Q: Do I get paid for the calibration session?**
A: Yes. The calibration session is compensated at the same hourly rate as annotation.
