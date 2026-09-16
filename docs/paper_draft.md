# LIT-DISC v1.1: A Confidence-Weighted Discourse Metric — Lessons from an Empirical Post-Mortem

**Target:** Springer CCIS (Communications in Computer and Information Science)
**Length target:** 12–15 pages (~8,000 words)
**Evidence root:** `evidence/`

---

## Abstract

Computational literary studies lacks standardized metrics for measuring discourse-level properties of narrative text. We present LIT-DISC v1.1, a computational pipeline that decomposes literary discourse into six dimensions — semantic, narrative, interaction, symbolic, evolution, and emotion — through a hierarchical multi-scale discourse processing tree (HMP-DT). Through comprehensive empirical validation on a 25-passage gold standard (25 authors, 5 narrative categories) and a full-length novel (The Mistress of Spices, 80,728 words), we identify a critical structural property: 96.5% of the metric's discriminative signal comes from a single component — the confidence weight gamma — while the remaining discourse richness factors contribute less than 3% of variance. This gamma dominance is confirmed as architectural rather than a scaling artifact: it persists at 95.3% at 400× novel scale, and null-model experiments show that even random embeddings preserve the author ranking (Spearman ρ = 0.995). Additional architectural findings include a memory system that fills to capacity in the first window and achieves zero cross-window retrievals across 105 windows, a resonance module contributing 0.02% of component value, and an activation function that required a single-parameter correction (sigma_max: 0.1 → 5.0) to avoid 99% saturation. The central contribution of this paper is not LIT-DISC as a mature metric, but the diagnostic methodology — a reproducible, evidence-based framework for validating whether a computational metric measures what it claims to measure. All code, data, and 42 evidence documents are publicly available.

**Keywords:** computational literary studies, discourse analysis, metric validation, narrative analysis, reproducibility

---

## 1. Introduction

### 1.1 Background

Literary discourse — the way narrative text organizes meaning across sentences, paragraphs, and chapters — is a central object of study in the humanities. Close reading, the traditional method for analyzing discourse, produces rich qualitative insights but does not scale to corpus-level analysis. Computational literary studies (CLS) has emerged as a field that applies natural language processing and statistical methods to literary texts, enabling analysis at scales impossible for human readers alone.

Despite progress in authorship attribution, genre classification, and stylometry, a fundamental gap remains: there is no widely accepted computational metric for literary-discourse richness that jointly captures semantic density, narrative structure, emotional arc, symbolic patterning, character evolution, and interactional dynamics. Existing NLP tools measure individual properties — readability, lexical diversity, sentiment — but do not integrate these into a coherent discourse-level representation.

### 1.2 Problem Statement

Computational approaches to text analysis have achieved success in specific domains:

- **Readability metrics** (Flesch-Kincaid, Dale-Chall) measure surface-level syntactic complexity but capture nothing about narrative content or discourse structure.
- **Lexical diversity measures** (type-token ratio, Moving Average TTR) quantify vocabulary richness but are insensitive to how words create meaning across sentences.
- **Semantic similarity methods** (topic modeling, word embeddings, SBERT) capture distributional semantics but lack mechanisms for tracking narrative progression, emotional arcs, or symbolic patterning.

None of these approaches can jointly model the six dimensions that literary scholars consider essential to discourse: semantics, narrative structure, interactional dynamics, emotional content, symbolic meaning, and character evolution. Furthermore, existing methods typically provide a single numerical score with no decomposition, making interpretation difficult.

### 1.3 Research Gap

We identify five specific gaps in current computational literary analysis:

1. **Single-property analysis.** Existing metrics measure one aspect of text (e.g., readability) in isolation. No integrated framework captures multiple discourse dimensions simultaneously.
2. **Limited interpretability.** Black-box neural methods produce accurate but uninterpretable scores. Literary scholars require decomposable metrics that explain why a text scores as it does.
3. **Absence of hierarchical structure.** Most metrics operate at the sentence or document level, ignoring the intermediate levels (paragraph, chapter, narrative arc) that structure literary discourse.
4. **Reproducibility deficit.** The majority of CLS papers do not release code, data, or detailed methodology, making independent verification impossible.
5. **Validation gap.** Metrics are often proposed without systematic validation against baselines, ablations, or null models, making it unclear what they actually measure.

### 1.4 Proposed Solution

We present LIT-DISC (Literary Discourse Index via Structured Computation), a computational framework that addresses these gaps through a hierarchical multi-scale discourse processing tree (HMP-DT). The framework decomposes literary text into six discourse dimensions, each computed from hand-crafted features, propagated through a 5-level hierarchical refinement loop, and aggregated into the Literary Discourse Richness Index (LDRI).

Rather than presenting LIT-DISC as a mature, ready-to-deploy metric, this paper takes an unusual approach: we conduct a thorough empirical post-mortem that reveals what the metric actually measures, identify three critical architectural failures, and provide a reproducible methodology for metric validation that the field can adopt. The value of this paper lies not in the metric itself, but in the diagnostic findings and the validation framework.

### 1.5 Contributions

1. **An open, reproducible metric pipeline.** LIT-DISC v1.1 provides a fully specified computational framework with 55 documented equations, 20 configurable parameters, and deterministic execution — all publicly available with 42 evidence documents.
2. **A diagnostic finding of gamma dominance.** Through baselines, ablations, variance decomposition, null models, and novel-scale replication, we demonstrate that 96.5% of LDRI's signal comes from confidence-weighting, not feature richness — a structural property of the geometric-mean aggregation design.
3. **Empirical identification of three dormant components.** The memory system (zero cross-window retrieval), resonance module (0.02% contribution), and coupling kernel (near-constant output) contribute negligibly to the final metric in their current form.
4. **A reproducible validation methodology.** Our baseline-ablation-null-model protocol provides a template for validating computational metrics, applicable beyond LIT-DISC.
5. **Honest accounting of limitations.** Twenty-one documented limitations, five unsupported claims, and a clear path to v2.0 — establishing that negative results and diagnostic findings are publishable contributions.

---

## 2. Related Work

### 2.1 Literary Analysis

Traditional literary analysis (close reading, New Criticism, structuralism) provides rich qualitative frameworks for understanding discourse but relies on human expertise that cannot scale beyond individual texts. Computational literary studies (Moretti, 2005; Underwood, 2019) applies quantitative methods to literary corpora, but existing work focuses on macro-level phenomena — genre evolution, character networks, thematic trends — rather than discourse-level richness at the passage level.

### 2.2 Discourse Modeling

Rhetorical Structure Theory (RST; Mann & Thompson, 1988) and Segmented Discourse Representation Theory (SDRT; Asher & Lascarides, 2003) provide formal models of discourse structure but require manual annotation. The Penn Discourse TreeBank (PDTB; Prasad et al., 2008) provides corpus-level annotations but covers news text, not literary narrative. None of these frameworks produce a continuous metric of discourse richness.

### 2.3 Narrative Modeling

Narrative analysis in NLP has focused on event detection (Chambers & Jurafsky, 2008), plot structure identification (Elson et al., 2010), and character arcs (Bamman et al., 2014). These approaches identify specific narrative elements but do not aggregate them into an overall discourse richness score. The Narrative Four的特征 Framework (Narrativity; Abbott, 2008) provides a qualitative foundation but has no computational instantiation.

### 2.4 Hierarchical NLP

Hierarchical approaches to text have been applied in document classification (Yang et al., 2016), summarization (Cohan et al., 2018), and discourse parsing (Joty et al., 2013). These models learn hierarchical representations from data but are task-specific and do not produce interpretable discourse dimensions. The 5-level hierarchy in LIT-DISC (sentence → paragraph → chapter → narrative → discourse) is inspired by these approaches but uses hand-crafted features rather than learned representations.

### 2.5 Research Gap

No existing framework provides: (a) a multi-dimensional discourse richness metric, (b) with hierarchical propagation through document structure, (c) full decomposition into interpretable components, (d) deterministic reproducibility, and (e) systematic validation against null models. LIT-DISC attempts to fill this gap, and — critically — our validation reveals why this is difficult and what must be fixed.

---

## 3. Methodology

### 3.1 Framework Overview

LIT-DISC processes natural language text through a 10-stage pipeline:

```
Text → Load → Sentence Split → Paragraph/Chapter Detect
→ Build Hierarchy Tree → Tokenize → Narrative Position (τ)
→ Feature Extraction (6 dimensions) → Signal Initialization
→ K-iteration Refinement Loop (propagate, couple, memory, resonance, uncertainty)
→ Embedding Construction → LDRI Calculation
```

The pipeline takes a text file as input and produces a `LITDISCResult` containing the final LDRI score, global confidence (Gamma), raw (unweighted) LDRI, bootstrap reliability, and six component decomposition values.

**Figure 1:** Overall framework architecture — block diagram of the 10-stage pipeline with color-coded stages.

### 3.2 HMP-DT: The Hierarchical Discourse Tree

The Hierarchical Multi-Phase Discourse Processing Tree decomposes text into five structural levels:

| Level | Name | Granularity | Node Count |
|-------|------|-------------|------------|
| 1 | Surface | Sentence | n (variable) |
| 2 | Structure | Paragraph | p (≪ n) |
| 3 | Abstraction | Chapter | c (≪ p) |
| 4 | Conceptual | Narrative layer | 1 |
| 5 | Relational | Discourse layer | 1 |

Features are extracted at the sentence level and propagated upward through the hierarchy using confidence-weighted aggregation, with horizontal coupling between dimensions at each level. A K-iteration refinement loop (K=5) performs bottom-up propagation, memory-augmented coupling, recursive resonance updating, and top-down refinement.

**Figure 2:** The 5-level hierarchy tree with example node counts and propagation paths.

### 3.3 Six Discourse Dimensions

Each dimension captures a distinct aspect of literary discourse:

| Dimension | Symbol | Features | What It Captures |
|-----------|--------|----------|------------------|
| Semantic | S | 8 | Lexical density, polysemy, information density, coherence |
| Narrative | N | 8 | Tension, pacing, scene type, temporal shifts, structural complexity |
| Interaction | I | 7 | Dialogue density, turn-taking, speaker/hearer encoding, question density |
| Symbolic | Sy | 7 | Figurative density, motif recurrence, imagery intensity |
| Evolution | E | 7 | Arc slope, relationship state, change magnitude |
| Emotion | Em | 7 | Valence, arousal, emotional diversity, intensity |

Each dimension produces a feature vector f_d ∈ ℝ^{k_d} and a per-signal confidence γ_d ∈ [0,1] for each sentence node.

### 3.4 Mathematical Formulation

We present the core equations for LDRI computation. See Appendix A for the complete set of 55 equations.

**Embedding construction.** After signal propagation, per-dimension embedding subspaces are constructed from propagated feature statistics:

```
H_d = [Mean(f_d), Max(f_d), Slope(f_d), Var(f_d), FFT_low(f_d)]   (Eq. 1)
```

where Slope(f_d) = (f_{n,d} - f_{1,d}) / (n-1) captures temporal trend.

**Activation.** Each dimension's embedding norm is mapped through the Rayleigh cumulative distribution function:

```
a_d = 1 - exp(-||H_d||² / 2σ²_max)                                 (Eq. 2)
```

where σ_max = 5.0 (v1.1). This is NOT a sigmoid — see Section 5.1 for the activation redesign discovery.

**Signal richness.** The per-dimension activations are aggregated:

```
R_signal = ||a|| / √6 = sqrt((1/6) Σ a_d²)                        (Eq. 3)
```

**Global confidence.** Per-dimension confidences are aggregated into a single weight:

```
Γ = Π min(1.0, max(0.0, γ̄_d))                                     (Eq. 4)
```

where γ̄_d is the mean per-dimension confidence across all nodes.

**Raw LDRI (unweighted).** The four discourse factors are combined via weighted geometric mean:

```
LDRI_raw = (R_signal^α · C̃^β · T̃^γ · M̃^δ)^{1/S}                (Eq. 5)
```

where C̃ is the normalized coupling factor, T̃ is the normalized resonance factor, M̃ is the normalized memory factor, S = α+β+γ+δ, and the exponents control each factor's importance (default: α=2.0, β=γ=δ=1.0).

**Final LDRI.** The raw score is modulated by global confidence:

```
LDRI = Γ · LDRI_raw                                                (Eq. 6)
```

**Key insight:** Since Γ varies by 21–47% (CV) while LDRI_raw varies by less than 3%, the final metric is dominated by Γ. LDRI is effectively a confidence-weighted constant — a finding we confirm empirically in Section 5.

### 3.5 Algorithm

The full pipeline executes as Algorithm 1, performing K=5 iterations of the refinement loop. Each iteration: (a) propagates signals upward through the hierarchy with horizontal coupling, (b) stores and retrieves from memory, (c) updates the resonance field, and (d) refines signals top-down.

**Algorithm 1:** LIT-DISC Pipeline (simplified)
```
Input: text file path P
Output: LITDISCResult R

 1: text ← Load(P)
 2: sentences ← SentenceSplit(text)
 3: paragraphs, chapters ← DetectStructure(sentences, text)
 4: tree ← BuildHierarchy(sentences, paragraphs, chapters)
 5: tokens ← Tokenize(sentences)
 6: τ ← ComputeNarrativePosition(sentences, tokens)
 7: features ← ExtractFeatures(sentences, tokens, τ)  (6 dims)
 8: signals ← InitializeSignals(features, τ)
 9: for k ← 1..K do
10:     signals ← BottomUpPropagation(signals, tree)
11:     signals ← HorizontalCoupling(signals)
12:     memory ← StoreAndRetrieve(signals, memory)
13:     resonance ← UpdateResonance(signals, resonance)
14:     signals ← TopDownRefinement(signals, tree)
15: embedding ← ConstructEmbedding(signals, resonance)
16: LDRI ← ComputeLDRI(embedding, memory, resonance)
17: return LITDISCResult(ldri=LDRI, ...)
```

### 3.6 Complexity

**Runtime:** O(K · n · M · k_key) worst-case, where n = sentences, K = iterations (5), M = memory capacity (200), k_key = key dimension (32). Empirically: 1.05 seconds per 1,000 words on standard hardware.

**Memory:** O(n · k_d + M · k_key), empirically 200 stored entries at capacity.

**Space:** O(n) for I/O, O(n · k_d) for features, O(M · k_key) for memory. Practical peak memory below 100 MB for novel-length texts.

---

## 4. Experimental Setup

### 4.1 Dataset

**Gold standard.** 25 passages from 25 authors across 5 narrative categories: Social Comedy (Austen, Dickens, Dostoevsky, Woolf, Hemingway), Modern/Contemporary (Mantel, Saunders, Ishiguro, Atwood, Ferrante), Short Stories (Chekhov, Joyce, Borges, Munro, Carver), Literary Classics (Flaubert, Proust, Kafka, Tolstoy, Garcia Marquez), and Non-Western Classics (Murasaki, Lu Xun, Mahfouz, Roy, Rushdie). Passages are approximately 200 words each (range: 166–323), all in English (original or translation).

**Full novel.** The Mistress of Spices (Chitra Banerjee Divakaruni, 1997), 80,728 words, processed as 105 windows of ~800 words each with paragraph-boundary alignment. Selected for public availability, manageable length, and discourse-rich content (postcolonial magical realism).

### 4.2 Environment

Python 3.14, Windows 11, Intel Core i7. Dependencies: numpy, standard library (re, math, json). No GPU, no deep learning frameworks. Deterministic execution with explicit seeds (RFF: seed 0 for W, seed 1 for b; bootstrap: seed 42).

### 4.3 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| σ_max | 5.0 | Activation scale (v1.1) |
| K | 5 | Refinement iterations |
| Max memory | 200 | Storage capacity |
| k_key | 32 | RFF key dimension |
| θ_store | 0.5 | Storage threshold |
| θ_retrieve | 0.7 | Retrieval threshold |
| θ_forget | 0.01 | Forgetting threshold |
| d | 120 | Embedding dimension |
| α | 2.0 | Signal richness weight |
| β | 1.0 | Coupling weight |
| γ | 1.0 | Resonance weight |
| δ | 1.0 | Memory weight |
| λ | 0.3 | Resonance decay |
| B | 100 | Bootstrap samples |

### 4.4 Evaluation Metrics

- **LDRI:** Final richness score (Γ-weighted), range [0,1]
- **Gamma (Γ):** Global confidence, computed as product of per-dimension confidences
- **Raw LDRI:** Unweighted richness (Γ=1), range [0,1]
- **Bootstrap reliability:** ρ = 1 - min(1.0, σ_LDRI / 0.1), with B=100 Gaussian-noise samples
- **Variance decomposition:** log-space variance partitioning into gamma, raw, and covariance components
- **Component fractions:** Six decomposition values (intrinsic, interaction, temporal, resonance, memory, uncertainty) normalized to sum ≈ 1

### 4.5 Experimental Workflow

All experiments follow a standardized protocol: load text → run pipeline → extract LITDISCResult → compute evaluation metrics → record in structured JSON/CSV. Each experiment is independently reproducible with documented seeds. The full workflow is automated via the verification suite (116 tests) and experiment scripts.

**Figure 3:** Experimental workflow — from text to evidence documents.

---

## 5. Results

### 5.1 Activation Validation

**Activation function analysis (v1.0 vs v1.1).**

In v1.0, σ_max was set to 0.1. Feature norms ||H_d|| were in the range 10–14 (mean ≈ 11.09), producing normalized arguments z = ||H_d||/σ_max ≈ 110 — far beyond the 5σ saturation threshold of the Rayleigh CDF. Result: 149 of 150 activations saturated at > 0.999, and LDRI had CV = 0.31% (effectively constant).

A single parameter change — σ_max: 0.1 → 5.0 — placed z ≈ 2.2 in the sensitive region of the CDF.

**Table 1:** Activation redesign results.

| Metric | v1.0 (σ=0.1) | v1.1 (σ=5.0) | Change |
|--------|--------------|--------------|--------|
| Saturated activations | 149/150 (99%) | 6/150 (4%) | −95% |
| Mean activation | 0.996 | 0.575 | −42% |
| Activation CV | 0.2% | 22.1% | 111× |
| LDRI mean | 0.9008 | 0.0277 | −97% |
| LDRI CV | 0.31% | 47.9% | 155× |
| Raw LDRI mean | 0.9131 | 0.9128 | Unchanged |
| Author Spearman ρ | — | 0.98 | Ranking preserved |

**Key finding:** The raw LDRI is unchanged (0.913 vs 0.913) — the metric's "richness" estimate is identical. Only the confidence weighting changes, producing a 155× improvement in CV. This is the first indication that gamma, not feature richness, drives LDRI.

**Figure 4:** Activation distribution — v1.0 (all > 0.999) vs v1.1 (uniform across [0.2, 1.0]).

### 5.2 Baseline Comparison

We evaluated four baseline conditions on all 25 gold-standard passages:

| Condition | LDRI mean | LDRI std | ρ (vs full) | Interpretation |
|-----------|-----------|----------|-------------|----------------|
| Full pipeline | 0.0277 | 0.0133 | 1.000 | Reference |
| Constant features | 0.0280 | 0.0130 | 0.998 | Gamma alone replicates LDRI |
| Random gamma | 0.0272 | 0.0030 | **0.14** | Gamma is essential |
| Dummy | 0.0 | 0.0 | — | Baseline |

**Key finding:** Random Gamma produces ρ = 0.14 with the full pipeline — proving gamma is the sole carrier of discriminative signal. Constant features reproduce LDRI almost exactly (ρ = 0.998), confirming that feature variation contributes negligibly.

**Null model.** Two additional null conditions:

| Condition | LDRI mean | LDRI CV | ρ (vs baseline) |
|-----------|-----------|---------|-----------------|
| Baseline | 0.0277 | 46.9% | 1.000 |
| Shuffled text | 0.0278 | 48.8% | **0.960** |
| Random embeddings | 0.0302 | 46.0% | **0.995** |

**Shuffled text (ρ=0.96):** Paragraph order contributes negligibly to LDRI. The metric captures per-passage feature statistics, not narrative structure.

**Random embeddings (ρ=0.995):** Even replacing all embedding subspaces with matched Gaussian noise, the ranking is nearly identical. Raw LDRI saturates at 0.9986, confirming LDRI ≈ Gamma.

### 5.3 Ablation Study

Five ablation conditions:

| Condition | LDRI | Δ | Effect |
|-----------|------|---|--------|
| Full | 0.0277 | — | Reference |
| No memory | 0.0264 | −0.0013 | SMALL |
| No resonance | 0.0261 | −0.0016 | SMALL |
| No coupling | 0.0292 | +0.0015 | SMALL |
| **No uncertainty** | **0.9126** | **+0.8850** | **LARGE** |
| No narrative | 0.0393 | +0.0116 | MODERATE |

**Without uncertainty (gamma=1.0):** LDRI collapses to raw LDRI (0.913) — a near-constant. Removing gamma increases the score by 0.885, the largest ablation effect by two orders of magnitude. Without narrative, LDRI increases because the narrative dimension has the lowest per-dimension gamma.

### 5.4 Statistical Analysis

**Descriptive statistics (gold standard, N=25):**
- LDRI: mean=0.0277, std=0.0133, CV=47.9%, range=[0.0088, 0.0551]
- Gamma: mean=0.0302, std=0.0139, CV=46.0%
- Raw LDRI: mean=0.9128, std=0.0235, CV=2.6%

**Category ranking (by LDRI):** SNC (0.0403) > MLC (0.0330) > NWC (0.0266) > SSC (0.0197) > MNC (0.0188)

**Author extremes:** Woolf (0.055) > Proust (0.052) > Dickens (0.051) > Kafka (0.042) > ... > Mantel (0.009)

**Variance decomposition:**
- var(log(LDRI)) = 0.2609
- Gamma contribution: **96.5%**
- Raw contribution: 0.3%
- Covariance: 3.2%

**Correlation with readability:** ρ ≤ 0.53 with all readability/lexical metrics — LDRI captures something orthogonal to standard text metrics.

### 5.5 Scalability

**Runtime:** 85.0 seconds for 80,728 words (105 windows). Linear scaling: 1.05 seconds per 1,000 words (r² = 0.999). Second half / first half runtime ratio = 1.08× (negligible degradation).

**Figure 5:** Runtime scaling — cumulative words vs execution time.

**Longitudinal LDRI (novel):** Mean=0.0100 (~1/3 of gold standard mean), CV=21.8%, gamma CV=21.3%.

**Variance decomposition (novel):** Gamma = 95.3%, raw = 0.2%, covariance = 4.5%.

**Gamma dominance is structurally unchanged at 400× scale:** 96.5% (gold standard) → 95.3% (novel). The asymmetry is not a small-sample artifact.

### 5.6 Memory Analysis

**Memory system behavior:** The memory fills to capacity (200/200 entries) in the first window. Zero cross-window retrievals across all 105 windows (at θ_retrieve = 0.7).

**Threshold sweep (θ = 0.1–0.9):**

| θ | SNC-04 LDRI | SNC-04 avg_β | Effect |
|---|-------------|-------------|--------|
| 0.1 | 0.05593 | 1.0000 | Within-window retrieval universal |
| 0.5 | 0.05587 | 0.9968 | Near-identical |
| 0.7 | 0.05507 | 0.9920 | Baseline |
| 0.9 | 0.05323 | 0.9827 | LDRI varies < 5% |

**Key finding:** Within-window retrieval is universal at all thresholds (avg_β ≥ 0.98). The zero cross-window retrieval problem is architectural — RFF keys computed from sentence-level features drift too far across narrative context for even θ=0.1 to match. Lowering the threshold would not solve the problem.

**Figure 6:** Memory growth — entries over cumulative words (fills to 200 in first window, stays at capacity).

**Figure 7:** Retrieval frequency — zero retrievals across all 105 windows.

### 5.7 Case Study

**Narrative arc.** The Mistress of Spices was processed in 105 windows across 8 narrative phases:

| Phase | Windows | LDRI Pattern | Key Content |
|-------|---------|-------------|-------------|
| Opening | 1–10 | Oscillating, falling | Youth → store exposition |
| Early crisis | 11–20 | Rise then plunge | Channeling the Old One |
| Middle | 21–35 | Lower with spike | Makaradwaj lore (peak) |
| Spiritual crisis | 36–50 | Two spikes | Self-punishment (novel peak: 0.0155) |
| Action | 51–60 | Violence spike | Racist assault (0.015) |
| Memory/trauma | 61–75 | Low, introspective | First-person reflection |
| Climax | 76–85 | Peak→valley | Reunion (0.014) → Renunciation (0.0057) |
| Resolution | 86–105 | Transformation | Liberation → denouement |

**16 narrative events** were identified via LDRI/component z-score thresholds. The novel's climax (renunciation) scores the lowest LDRI (0.0057), while a spiritual crisis scene scores highest (0.0155). This illustrates that LDRI captures linguistic complexity, not narrative importance — a critical distinction.

**Figure 8:** LDRI over 105 windows with event annotations.
**Figure 9:** Gamma over 105 windows (drives all variation).
**Figure 10:** Raw LDRI over 105 windows (near-constant, CV=1.0%).
**Figure 11:** Variance decomposition: gold standard vs novel (side-by-side).

---

## 6. Discussion

### 6.1 Why Gamma Dominates

The evidence for gamma dominance is conclusive across five independent analyses:

1. **Variance decomposition:** 96.5% of var(log(LDRI)) comes from gamma.
2. **Baseline (constant features):** ρ = 0.998 — features contribute nothing beyond gamma.
3. **Baseline (random gamma):** ρ = 0.14 — without gamma, LDRI is random.
4. **Null model (random embeddings):** ρ = 0.995 — even destroying embeddings preserves ranking.
5. **Novel-scale replication:** 95.3% at 400× scale — not a small-sample artifact.

The root cause is structural: LDRI = Gamma × LDRI_raw, where LDRI_raw is a weighted geometric mean of factors that are all near-constant. The geometric mean of near-1 values (0.913) produces near-constant output. Gamma, computed from per-dimension feature confidences (Eq. 21), provides all variation.

This is not a bug that can be fixed by parameter tuning — it is a consequence of the metric design. The six discourse dimensions, extracted from hand-crafted lexica, produce feature vectors whose norms are stable across texts. The embedding construction aggregates these into summary statistics (mean, max, slope, variance) that compress away the variation.

### 6.2 Memory Behavior

The memory system's failure to retrieve across windows is traced to the RFF key representation. Keys are computed from sentence-level feature vectors via random Fourier projection (Eq. 22). As narrative context changes, these feature vectors drift, and their RFF projections drift correspondingly.

The threshold sweep (Section 5.6) demonstrates that even θ=0.1 does not produce cross-window retrieval, and within-window retrieval is universal at all thresholds. This rules out a threshold-calibration explanation. The problem is that the RFF representation lacks temporal or semantic invariance — it encodes the exact feature distribution of each sentence, and across windows these distributions differ sufficiently to fall below any cosine similarity threshold.

Three potential solutions: (a) semantic chunking — store paragraph-level rather than sentence-level representations; (b) content-based hashing — use topic-level or event-level keys rather than feature-level; (c) overlapping windows — ensure contextual continuity between consecutive windows.

### 6.3 Resonance Behavior

The resonance module contributes a mean of 0.0002 (gold standard) and 0.00008 (novel) — approximately 0.02% of total component value. Ablation confirms removal changes LDRI by |Δ| = 0.0016.

The resonance field update (Eq. 40) is a discretized exponential moving average of the concatenated signal vector. With λ=0.3, the effective time constant is approximately 3-4 narrative time steps, meaning the field decays rapidly. For a 200-word passage, the resonance trajectory contains insufficient steps for meaningful dynamics. For the novel, the narrative sampling rate (100 samples per window) produces near-zero accumulated field values.

The mechanism is either theoretically unnecessary for this task or requires a fundamentally different formulation — e.g., cross-window echo, lexical chain persistence, or long-range recurrence.

### 6.4 Interpretation

The evidence supports a reinterpretation of what LIT-DISC measures. The current framework behaves as a **confidence-weighted discourse estimator** rather than a **pure richness estimator**. The LDRI score is primarily determined by the per-dimension confidence gamma, which reflects the average magnitude of feature values — higher feature activations produce higher confidence, which produces higher LDRI.

This is not meaningless: passages with stronger emotional language, denser narrative features, and richer lexical patterns do produce higher gamma values. The ranking Woolf > Ferrante > Rushdie > Mantel is interpretable. But the claim that LDRI measures "literary discourse richness" in a multi-dimensional sense is not supported by the evidence. The six dimensions, the hierarchical propagation, the coupling kernel, and the embedding construction — the majority of the pipeline's complexity — contribute negligibly to the metric's output.

### 6.5 Lessons Learned

1. **Engineering lesson:** Complex multi-component pipelines can appear to work while only one component drives the output. Variance decomposition and null models should be standard validation steps for any proposed metric.

2. **Interpretability lesson:** Geometric mean aggregation over near-constant factors produces near-constant output. Metric design must ensure that the aggregation mechanism preserves the signal from each component, rather than compressing it away.

3. **Reproducibility lesson:** Open-sourcing the full methodology — including negative results — enables the community to build on honest foundations. This paper's value lies in what went wrong, not what went right.

4. **Validation lesson:** A metric that passes unit tests and produces interpretable rankings can still fail to measure what it claims. Ablation studies, null models, baseline comparisons, and architectural diagnosis are essential before deployment.

---

## 7. Limitations

### 7.1 Theoretical

**L1 — No external human validation.** LDRI has not been compared against expert annotations of discourse richness. The interpretable rankings (Woolf > Mantel) could reflect any number of latent properties (sentence length, vocabulary size, emotional intensity). A 25-passage expert annotation study is the highest-priority future work.

**L2 — Single novel tested.** Full-novel evaluation (The Mistress of Spices) is a single data point. Replication across 5+ novels (different genres, periods, languages) is needed before generalization claims.

**L3 — English-only lexica.** Feature extraction lexicons are English-only (WordNet, LIWC, NRC VAD, custom word lists). The framework's applicability to other languages is untested.

### 7.2 Experimental

**L4 — Small gold standard (N=25).** With 5 passages per category, category-level statistics are unreliable. Confidence intervals on category means are wide.

**L5 — No significance testing.** All reported differences — category rankings, baseline conditions, ablation effects — lack confidence intervals or p-values. Effect sizes appear large (e.g., no-uncertainty Δ=+0.885) but statistical significance is unknown.

**L6 — No null model for narrative time series.** The LDRI trajectory over 105 windows (CV=21.8%) may be indistinguishable from a random walk or autoregressive process with matched variance.

### 7.3 Computational

**L7 — Parameter sensitivity.** σ_max=5.0 was chosen post-hoc to match observed feature norms (10–14). If feature norms change (different lexica, languages, or dimensions), σ_max would need recalibration.

**L8 — Sliding-window artifacts.** The 800-word window with paragraph boundaries introduces edge effects. Events near window boundaries may be split or duplicated.

**L9 — Hand-crafted features.** The 44 features across 6 dimensions are rule-based and lexicon-dependent. Learned feature representations (e.g., fine-tuned embeddings) could improve sensitivity.

### 7.4 Validation

**L10 — Circular event analysis.** Narrative events (Section 5.7) were identified from LDRI extremes rather than independent annotation. The relationship between LDRI and narrative events is correlational, not predictive.

**L11 — Single-rater interpretation.** Literary analysis of narrative events (Section 5.7) represents one reader's judgment. Inter-rater reliability is not established.

**L12 — Zero ablated component independence.** The ablation study removes components one at a time, but components interact. The "no uncertainty" ablation (gamma=1.0) changes LDRI by +0.885, but the effect of removing both uncertainty and another component jointly is unknown.

---

## 8. Conclusion

We presented LIT-DISC v1.1, a computational framework for literary-discourse analysis, and conducted a thorough empirical post-mortem that revealed three critical architectural findings: gamma dominance (96.5% of signal from confidence-weighting), dormant memory (zero cross-window retrieval), and vestigial resonance (0.02% contribution). The activation redesign (σ_max: 0.1 → 5.0) corrected a saturation pathology and improved CV by 155×, but confirmed that raw LDRI is near-constant regardless of scaling.

The framework's strength is its reproducibility — 55 documented equations, 20 configurable parameters, deterministic execution, and 42 publicly available evidence documents. Its weakness is that the metric does not measure what it claims to measure. The six discourse dimensions, hierarchical propagation, and embedding construction contribute negligibly to LDRI's output. The metric is a confidence-weighted discourse estimator, not a multi-dimensional richness index.

**Future work (v1.2, 3 months):** Expert annotation study (25 gold-standard passages, 6 dimensions, 3 annotators), multi-novel replication (5+ novels), cross-language evaluation, null model testing.

**Future work (v2.0, 6–12 months):** Redesigned memory (semantic chunking, sliding-window overlap, content-based hashing), new resonance formulation (cross-window echo, lexical chain persistence), alternative confidence modulation (arithmetic mean, learned weights, hierarchical gamma), and adaptive long-context retrieval.

The central contribution of this paper is methodological: a template for validating whether a computational metric measures what it claims to measure, with honest accounting of failures. We believe this template is applicable beyond LIT-DISC and can serve as a standard for metric validation in computational literary studies.

---

## References

[1] Moretti, F.: Graphs, Maps, Trees: Abstract Models for a Literary History. Verso (2005)
[2] Underwood, T.: Distant Horizons: Digital Evidence and Literary Change. University of Chicago Press (2019)
[3] Mann, W.C., Thompson, S.A.: Rhetorical Structure Theory: Toward a functional theory of text organization. Text 8(3), 243–281 (1988)
[4] Asher, N., Lascarides, A.: Logics of Conversation. Cambridge University Press (2003)
[5] Prasad, R., et al.: The Penn Discourse TreeBank 2.0. LREC (2008)
[6] Chambers, N., Jurafsky, D.: Unsupervised learning of narrative event chains. ACL (2008)
[7] Elson, D.K., et al.: Extracting social networks from literary fiction. ACL (2010)
[8] Bamman, D., et al.: A Bayesian mixed effects model of literary character. ACL (2014)
[9] Abbott, H.P.: The Cambridge Introduction to Narrative. Cambridge University Press (2008)
[10] Yang, Z., et al.: Hierarchical attention networks for document classification. NAACL (2016)
[11] Cohan, A., et al.: A discourse-aware attention model for abstractive summarization of long documents. NAACL (2018)
[12] Joty, S., et al.: A novel discourse parser based on support vector machine. ACL (2013)
[13] Rahimi, A., Recht, B.: Random features for large-scale kernel machines. NeurIPS (2007)
[14] Pennebaker, J.W., et al.: The development and psychometric properties of LIWC. (2015)
[15] Mohammad, S.M., Turney, P.D.: Crowdsourcing a word-emotion association lexicon. Computational Intelligence 29(3) (2013)

---

## Appendix A: Full Equation Set

Complete 55-equation mathematical formulation from `docs/methodology_equations.md`. Includes activation (Eq. 1), signal richness (Eq. 2), coupling factor (Eq. 3), resonance factor (Eq. 4), memory factor (Eq. 5), normalized factors (Eq. 6), global confidence (Eq. 7), raw LDRI (Eq. 8), final LDRI (Eq. 9), decomposition (Eq. 10–17), reliability (Eq. 18), narrative position (Eq. 19–20), confidence computation (Eq. 21), RFF keys (Eq. 22), memory operations (Eq. 23–28), coupling kernel (Eq. 29–39), resonance update (Eq. 40–41), top-down refinement (Eq. 42), vertical aggregation (Eq. 43–46), uncertainty propagation (Eq. 47–52), and embedding construction (Eq. 53–55).

---

## Appendix B: Parameter Table

Complete 20-parameter table with meanings, defaults, sensitivity ratings, and source file locations.

---

## Appendix C: Variable Dictionary

Complete 44-entry variable dictionary with symbols, meanings, ranges, defaults, and source code references.

---

## Appendix D: Implementation Mapping

55-row mapping from each equation to its Python function, file, and line numbers.

---

## Appendix E: Additional Figures

Supplementary figures: activation function sweep (σ=0.05–10.0), component evolution across novel, category-level box plots, pairwise correlation matrix.

---

## Appendix F: Evidence Index

Complete listing of all 42 evidence documents with descriptions.
