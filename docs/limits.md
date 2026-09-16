# Limitations — LIT-DISC v1.1

All known limitations organized by severity and source.

---

## HIGH Severity — Architectural

### L1. Gamma Dominance (LDRI = Gamma × Constant)

**The problem:** LDRI is dominated by the uncertainty weight gamma (96.5% of variance). Raw LDRI varies by only 0.3% (gold standard) to 0.2% (novel). This means LDRI measures *confidence-weighted* richness, not *intrinsic* richness.

**Why it matters:** The metric name ("Literary Discourse Richness Index") claims to measure richness but actually measures uncertainty. A reader expecting a richness metric will be misled.

**Evidence:** `evidence/07_statistics/statistics_summary.md` (variance decomposition), `evidence/03_baselines/baseline_summary.md` (random gamma ρ=0.14), `evidence/04_ablations/ablation_summary.md` (without uncertainty Δ=+0.885)

**Mitigation:** Reformulate LDRI to distribute signal across components rather than concentrating it in uncertainty. Or rename the metric to reflect what it actually measures.

### L2. Zero Cross-Window Memory Retrieval

**The problem:** The RFF-based memory system fills to capacity (200 entries) in the first ~800 words and never retrieves across 105 windows. Cross-window retrieval = 0.

**Why it matters:** Memory is a core architectural component. If it never retrieves, it is not functioning as a memory. It is a per-window bag.

**Evidence:** `evidence/05_scalability/scalability_summary.md` (memory analysis), `evidence/05_scalability/memory_analysis.csv`

**Mitigation:** Redesign memory with: (a) semantic chunking, (b) sliding-window overlap, (c) lower retrieval threshold, (d) different key generation (e.g., content-based hashing).

### L3. Vestigial Resonance Component

**The problem:** Mean resonance = 0.0002 (gold standard), 0.00008 (novel). Removing resonance changes LDRI by |Δ| = 0.0016 (SMALL effect). Resonance contributes ~0.02% of total component value.

**Why it matters:** Resonance is a "dead" component — it adds complexity without measurable contribution. Including it in the framework inflates the theoretical apparatus without empirical support.

**Evidence:** `evidence/07_statistics/statistics_summary.md` (resonance mean), `evidence/04_ablations/ablation_summary.md`

**Mitigation:** Redesign the resonance mechanism or remove it. A reimplementation should produce values comparable in magnitude to other components.

### L4. Memory, Resonance, Coupling Ablations Show Negligible Effects

**The problem:** All three ablations produce |Δ| < 0.002. The framework's non-uncertainty components collectively contribute < 1% of LDRI variation.

**Why it matters:** The framework has 6 dimensions and 47 equations, but most of this machinery is functionally irrelevant to the output. This is a parsimony problem.

**Evidence:** `evidence/04_ablations/ablation_summary.md`

**Mitigation:** Either: (a) redesign components to contribute meaningfully, (b) simplify the framework to the components that matter (uncertainty + narrative), or (c) acknowledge that the framework is over-parameterized for the current corpus.

---

## HIGH Severity — Validation

### L5. No Expert Annotation Validation

**The problem:** All gold-standard passages have expert scores of 4 (placeholder). There is zero external validation of LDRI against human judgments.

**Why it matters:** Without external validation, LDRI is an internal metric. No claim of literary relevance can be supported.

**Evidence:** `evidence/02_validation/gold_standard_summary.md`, `evidence/10_claims/unsupported_claims.md` (U3)

**Mitigation:** Conduct an expert annotation study with ≥ 3 annotators, ≥ 25 passages, measured inter-annotator agreement, and LDRI-expert correlation.

### L6. Single Full-Length Novel Tested

**The problem:** Novel-scale evaluation was performed on exactly one novel (The Mistress of Spices). All scalability, memory, and longitudinal claims rest on a single data point.

**Why it matters:** Generalizability cannot be established from N=1. The LDRI behavior at novel length may be specific to this novel's style, genre, or length.

**Evidence:** `evidence/05_scalability/scalability_summary.md`, `evidence/06_case_study/mistress_case_study.md`

**Mitigation:** Replicate on 5–10 novels spanning different genres, periods, styles, and languages.

### L7. 25-Passage Gold Standard Is Small

**The problem:** 25 passages is adequate for exploratory analysis but underpowered for correlation studies (e.g., ρ between LDRI and sentence length = 0.53).

**Why it matters:** Correlation estimates with N=25 have wide confidence intervals. Some reported correlations may not replicate.

**Evidence:** `evidence/07_statistics/statistics_summary.md`

**Mitigation:** Expand gold standard to ≥ 100 passages. This would require more diverse authorship and category representation.

---

## MEDIUM Severity — Design

### L8. Fixed 800-Word Windows

**The problem:** The novel is processed as 800-word windows. Window size affects feature computation, memory capacity, and LDRI values. No sensitivity analysis was performed.

**Why it matters:** Claims about LDRI's narrative arc are window-size dependent. Wider windows might produce different (or flatter) trajectories.

**Evidence:** `evidence/05_scalability/scalability_summary.md`

**Mitigation:** Test window sizes from 200 to 2,000 words and report effect on LDRI distribution and event detection.

### L9. Paragraph-Boundary Window Splitting

**The problem:** Windows respect paragraph boundaries (greedy addition). This means window sizes vary and some windows may end mid-scene.

**Why it matters:** Scene breaks do not align with paragraph breaks. A critical narrative event could be split across two windows, attenuating its LDRI signal.

**Evidence:** `evidence/05_scalability/scalability_summary.md`

**Mitigation:** Test chapter-boundary and scene-boundary windows. Compare LDRI profiles across splitting strategies.

### L10. Placeholder Expert Scores in Gold Standard

**The problem:** The gold standard dataset includes "expert_annotations" and "expert_scores" with all values set to 4 (on a 0–5 scale). These are placeholders from the original generation script.

**Why it matters:** Having placeholder data in the released dataset is misleading. A user might assume real annotations exist.

**Evidence:** `evidence/02_validation/gold_standard_summary.md`

**Mitigation:** Either (a) replace with real annotations, or (b) remove from the public dataset with clear documentation.

---

## MEDIUM Severity — Corpus

### L11. English-Only Corpus

**The problem:** All 25 gold-standard passages and the single novel are in English. The framework's language-dependence is unknown.

**Why it matters:** The framework may be English-specific (spaCy, NRC lexicon, etc.). Applying it to non-English text would require significant adaptation.

**Evidence:** All evidence files

**Mitigation:** Evaluate on at least one non-English novel (e.g., Spanish, French, Hindi) with appropriate NLP pipeline adaptation.

### L12. 20th-Century Western Canon Bias

**The problem:** The gold standard heavily features 20th-century authors (Woolf, Rushdie, Roy, Ferrante) with some 19th-century (Dickens, Dostoevsky, Tolstoy) and contemporary (Adichie, Hamid) authors. No pre-1800 or non-Western tradition authors except those writing in English as a second language.

**Why it matters:** LDRI's behavior on medieval, early modern, or non-Western literary traditions is unknown.

**Evidence:** `evidence/02_validation/gold_standard_summary.md`

**Mitigation:** Expand the corpus to include pre-1800 authors and more non-Western literary traditions.

### L13. Only Narrative Fiction

**The problem:** The corpus is exclusively narrative fiction. Poetry, drama, creative non-fiction, and genre fiction (romance, thriller, sci-fi) are absent.

**Why it matters:** The framework is marketed as measuring "literary discourse" but has only been tested on a single genre.

**Evidence:** `evidence/02_validation/gold_standard_summary.md`

**Mitigation:** Test on poetry (shorter forms), drama (dialogue-heavy), non-fiction (argumentative), and genre fiction.

---

## MEDIUM Severity — Methodology

### L14. Post-Hoc Narrative Event Identification

**The problem:** The 16 narrative events were identified by examining LDRI extremes and then reading the corresponding text. This is circular — events are defined by the metric and then used to validate the metric.

**Why it matters:** The analysis confirms the authors' expectations rather than independently testing a hypothesis.

**Evidence:** `evidence/06_case_study/narrative_event_log.md`

**Mitigation:** Pre-register predicted events based on narrative theory before computing LDRI.

### L15. No Null Model

**The problem:** No comparison against a null model (random LDRI trajectory, shuffled text, or permuted window order). The apparent narrative structure in LDRI may be an artifact of any time series with 21.8% CV.

**Why it matters:** Without a null model, the claim that LDRI "traces the narrative arc" is speculative.

**Evidence:** `evidence/06_case_study/mistress_case_study.md`

**Mitigation:** Generate a null LDRI distribution by shuffling text or randomizing features, and compare the observed trajectory's structure.

### L16. No Significance Testing

**The problem:** Differences between categories, authors, and narrative events are reported without confidence intervals or p-values.

**Why it matters:** The reported differences (e.g., SNC=0.040 vs MNC=0.019) may or may not be statistically significant.

**Evidence:** `evidence/07_statistics/statistics_summary.md`

**Mitigation:** Bootstrap confidence intervals for all reported means and differences. Report effect sizes.

### L17. Single-Rater Literary Analysis

**The problem:** The narrative event identification and interpretation were performed by a single rater (the developer). No inter-rater reliability.

**Why it matters:** Literary interpretation is subjective. Another reader might identify different events or interpret them differently.

**Evidence:** `evidence/06_case_study/narrative_event_log.md`

**Mitigation:** Have ≥ 2 independent raters identify narrative events. Report inter-rater agreement (Cohen's κ or percentage agreement).

### L18. No Retrieval Threshold Sensitivity Analysis

**The problem:** The retrieval threshold θ=0.7 was chosen without justification. Zero cross-window retrievals may be an artifact of this threshold.

**Why it matters:** A lower threshold (θ=0.3 or 0.5) might produce meaningful retrievals and change the memory analysis.

**Evidence:** `evidence/05_scalability/memory_analysis.csv`

**Mitigation:** Sweep retrieval thresholds from 0.1 to 0.9 in 0.1 increments. Report retrieval counts and LDRI impact for each.

---

## LOW Severity — Presentation

### L19. Metric Name Does Not Match Behavior

**The problem:** "Literary Discourse Richness Index" implies intrinsic richness, but LDRI measures confidence-weighted uncertainty.

**Why it matters:** Misleading name will cause confusion and may attract criticism from reviewers.

**Evidence:** All evidence files

**Mitigation:** Rename to "Confidence-Weighted Discourse Index" or similar, or reformulate the metric to match the name.

### L20. Incomplete Code Quality

**The problem:** 87 ruff violations (unused imports, import order) and 14 mypy errors (missing stubs, untyped bodies).

**Why it matters:** Code quality issues, while not affecting correctness, may reduce reviewer confidence in reproducibility.

**Evidence:** `evidence/09_reproducibility/reproducibility.md`

**Mitigation:** Run `ruff --fix` and add missing type stubs. Targeted cleanup before paper submission.

### L21. Corpus Copyright Restrictions

**The problem:** The gold-standard passages cannot be shared due to copyright. Only computed scores are public.

**Why it matters:** Results cannot be fully reproduced by independent researchers who do not have access to the same editions.

**Evidence:** `evidence/09_reproducibility/reproducibility.md`

**Mitigation:** Use public-domain texts wherever possible. Document editions and page numbers for copyrighted works. Consider releasing a parallel public-domain corpus.

---

## Summary Table

| ID | Limitation | Severity | Source | Can Mitigate? |
|----|-----------|----------|--------|---------------|
| L1 | Gamma dominance | HIGH | Architecture | Reformulate or rename |
| L2 | Zero cross-window retrieval | HIGH | Memory design | Redesign memory |
| L3 | Vestigial resonance | HIGH | Resonance design | Redesign or remove |
| L4 | Negligible non-uncertainty components | HIGH | Architecture | Redesign or simplify |
| L5 | No expert validation | HIGH | Missing data | New study |
| L6 | Single novel tested | HIGH | Novel eval design | Replicate on 5+ novels |
| L7 | Small gold standard (N=25) | HIGH | Corpus design | Expand to 100+ passages |
| L8 | Fixed window size | MEDIUM | Methodology | Sensitivity analysis |
| L9 | Paragraph-boundary splitting | MEDIUM | Methodology | Test other strategies |
| L10 | Placeholder expert scores | MEDIUM | Data quality | Remove or replace |
| L11 | English-only corpus | MEDIUM | Corpus design | Cross-language eval |
| L12 | 20th-century canon bias | MEDIUM | Corpus design | Expand corpus |
| L13 | Only narrative fiction | MEDIUM | Corpus design | Test other genres |
| L14 | Post-hoc event identification | MEDIUM | Methodology | Pre-register predictions |
| L15 | No null model | MEDIUM | Methodology | Generate null distribution |
| L16 | No significance testing | MEDIUM | Methodology | Bootstrap + CI |
| L17 | Single-rater analysis | MEDIUM | Methodology | Multi-rater study |
| L18 | No retrieval threshold sweep | MEDIUM | Methodology | Threshold sensitivity |
| L19 | Misleading metric name | LOW | Presentation | Rename |
| L20 | Incomplete code quality | LOW | Engineering | Cleanup |
| L21 | Copyright restrictions | LOW | Legal/Data | Public-domain corpus |
