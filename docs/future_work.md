# Future Work — LIT-DISC Roadmap

## LIT-DISC v1.2 (Immediate — Next 3 Months)

Addresses gaps that can be filled with existing infrastructure — no architectural changes.

### 1. Expert Annotation Study

| Task | Description | Evidence Gap |
|------|-------------|-------------|
| Recruit 3+ annotators | Literary scholars familiar with discourse analysis | `evidence/10_claims/unsupported_claims.md:U3` |
| Annotate 25 gold-standard passages | Rate literary-discourse richness on 1–5 scale | `evidence/02_validation/gold_standard_summary.md:L5` |
| Measure inter-annotator agreement | Cohen's κ or ICC | — |
| Compute LDRI–expert correlation | Spearman ρ, Pearson r | — |
| Publish anonymized annotations | Alongside LDRI scores | — |

**Success criterion:** ρ ≥ 0.7 between LDRI and mean expert rating.

### 2. Multi-Novel Replication

| Task | Description | Evidence Gap |
|------|-------------|-------------|
| Select 5 novels | Different genres, periods, styles (e.g., *Pride and Prejudice*, *Beloved*, *The Great Gatsby*, *Neuromancer*, *Things Fall Apart*) | `evidence/05_scalability/scalability_summary.md:L6` |
| Process all 5 novels | Same 800-word window pipeline | — |
| Report LDRI means, ranges, CVs | Compare to Mistress baseline | — |
| Report memory retrieval counts | Cross-novel retrieval statistics | `evidence/05_scalability/scalability_summary.md:L2` |
| Report gamma dominance | Check if 95%+ holds across novels | — |

**Success criterion:** LDRI dynamics are consistent across ≥ 4/5 novels.

### 3. Cross-Language Evaluation

| Task | Description | Evidence Gap |
|------|-------------|-------------|
| Select 1 non-English novel | French, Spanish, or Hindi (language with available NLP pipeline) | `evidence/02_validation/gold_standard_summary.md:L11` |
| Adapt pipeline | Replace spaCy English with spaCy models for target language; replace NRC lexicon with target-language emotion lexicon | — |
| Process and compare | LDRI distribution comparison with English novels | — |

**Success criterion:** Non-English LDRI distribution and gamma dominance are qualitatively similar to English.

### 4. Null Model and Significance Testing

| Task | Description | Evidence Gap |
|------|-------------|-------------|
| Bootstrap confidence intervals | For all LDRI means, differences, correlations | `evidence/07_statistics/statistics_summary.md:L16` |
| Shuffled-text baseline | Compare novel LDRI to same-words-randomized | `evidence/06_case_study/mistress_case_study.md:L15` |
| Random-walk comparison | Test if LDRI trajectory differs from AR(1) process | — |

**Success criterion:** LDRI trajectory has significantly more structure than null model (p < 0.05).

### 5. Retrieval Threshold Sweep

| Task | Description | Evidence Gap |
|------|-------------|-------------|
| Sweep θ from 0.1 to 0.9 | On Mistress of Spices full-novel data | `evidence/05_scalability/scalability_summary.md:L18` |
| Report retrieval counts per threshold | — | — |
| Report LDRI effect per threshold | Does retrieval change LDRI? | — |
| Identify optimal threshold | Threshold that maximizes retrieval without false positives | — |

**Success criterion:** At least one threshold produces > 0 cross-window retrievals.

### 6. Window Size Sensitivity Analysis

| Task | Description | Evidence Gap |
|------|-------------|-------------|
| Test 200, 400, 600, 800, 1000, 1200, 1600, 2000 words | On the Mistress of Spices | `evidence/05_scalability/scalability_summary.md:L8` |
| Report LDRI distribution per size | Mean, CV, range | — |
| Report event detection consistency | Do the 16 events appear at all window sizes? | — |

**Success criterion:** Narrative event detection is stable across window sizes 400–1200.

---

## LIT-DISC v2.0 (Long-Term — 6–12 Months)

Requires architectural changes to the HMP-DT pipeline.

### 7. Redesigned Memory Architecture

**Problem:** Current RFF-based memory never retrieves across windows. Keys drift because they are computed from sentence features that change across narrative contexts.

**Proposed redesigns:**

| Approach | Description | Expected improvement |
|----------|-------------|---------------------|
| Semantic chunking | Group sentences into narrative chunks (rather than individual sentences) before storing | Fewer, more distinctive entries; higher retrieval probability |
| Sliding-window overlap | Adjacent windows share sentences (50% overlap) | Guarantees cross-window connections |
| Content-based hashing | Use sentence embedding (e.g., SBERT) as key instead of RFF features | Semantic keys invariant to surface form changes |
| Hierarchical memory | Short-term (window) + long-term (chapter/section) with distinct retrieval thresholds | Enables both local and global retrieval |

**Success criterion:** ≥ 10% of windows produce at least one cross-window retrieval at novel length.

### 8. New Resonance Formulation

**Problem:** Current resonance is the product of element-wise min between sentence embeddings — produces near-zero values.

**Proposed redesigns:**

| Approach | Description | Expected improvement |
|----------|-------------|---------------------|
| Cross-window echo | Measure how thematic elements from earlier windows reappear in later windows | Resonance ≈ 0 for novel content, > 0 for developments |
| Lexical chain resonance | Compute lexical chain persistence across windows (based on WordNet or embedding similarity) | Quantifiable narrative development |
| Attention-based resonance | Use a cross-attention mechanism between current and past window representations | Learned resonance weights |

**Success criterion:** Mean resonance ≥ 0.01 (comparable to other components) and removal changes LDRI by > 0.01.

### 9. Alternative Confidence Modulation

**Problem:** Gamma dominates 96.5% of LDRI variance. Features contribute < 1%.

**Proposed alternatives:**

| Approach | Description | Expected gamma contribution |
|----------|-------------|---------------------------|
| Arithmetic mean (no gamma) | LDRI = (sum of components) / 6 | 0% — no gamma |
| Learned weighted sum | Components weighted by learned attention | < 25% (attention distributed) |
| Hierarchical gamma | Separate gamma per HMP-DT level, combined in a tree | < 50% (gamma at multiple levels) |
| Gamma as feature only | Gamma is one of 6 components, not a multiplier | 16.7% (equal if uniform) |

**Success criterion:** Gamma contribution drops to ≤ 50% while preserving discriminative power (ρ ≥ 0.8 with full v1.1 LDRI ranking).

### 10. Adaptive Long-Context Retrieval

**Problem:** Memory has a fixed capacity (200 entries) that fills immediately, preventing long-range access.

**Proposed design:**

| Component | Description |
|-----------|-------------|
| Capacity scaling | Memory capacity scales with text length (not fixed at 200) |
| Eviction policy | LRF (Least Recently Fetched) — not LRU — since narrative content may need to be revisited after long gaps |
| Retrieval scoring | Combined similarity + recency + novelty scoring |
| Attention-based retrieval | Learn to attend to past windows based on current content |

**Success criterion:** Long-range retrieval (> 10 windows apart) occurs for ≥ 5% of windows.

### 11. Expanded Gold Standard

| Dimension | Current | Target |
|-----------|---------|--------|
| Passages | 25 | 100+ |
| Authors | 25 | 50+ |
| Categories | 5 narrative fiction | 10 (add poetry, drama, non-fiction, genre fiction) |
| Languages | 1 (English) | 3+ (English, French, Spanish, Hindi) |
| Periods | 19th–21st century | 16th–21st century |
| Annotations | None | 3+ annotators per passage |

### 12. Pre-Registered Literary Analysis

**Design:** Before computing LDRI for a new novel, pre-register:
- Predicted events (with narrative justification)
- Expected LDRI direction (up/down/neutral)
- Expected magnitude (small/medium/large)

Compare predictions to observed LDRI. Compute prediction accuracy as a formal test of LDRI's literary relevance.

---

## Summary: v1.2 vs v2.0

| Feature | v1.2 (3 months) | v2.0 (6–12 months) |
|---------|-----------------|-------------------|
| Expert validation | ✅ Yes | — |
| Multi-novel replication | ✅ Yes | — |
| Cross-language evaluation | ✅ Yes | — |
| Null model + significance | ✅ Yes | — |
| Threshold sensitivity | ✅ Yes | — |
| Window size sensitivity | ✅ Yes | — |
| Memory redesign | — | ✅ Yes |
| Resonance redesign | — | ✅ Yes |
| Gamma redistribution | — | ✅ Yes |
| Long-context retrieval | — | ✅ Yes |
| Expanded corpus | — | ✅ Yes |
| Pre-registered analysis | — | ✅ Yes |

**End state:** LIT-DISC v2.0 is an architecture where all 6 components contribute meaningfully to LDRI, memory retrieves across the full novel, resonance tracks narrative development, and the metric is validated against human judgment across multiple languages and genres.
