# Paper Outline — LIT-DISC v1.1

A publication-ready structure with section references to evidence files, claims, and figures.

---

## Title

**LIT-DISC v1.1: A Confidence-Weighted Discourse Metric — Lessons from an Empirical Post-Mortem**

*Target venues:* COLING, ACL Workshop on Computational Literary Studies, Journal of Computational Literary Studies (JCLS), Digital Scholarship in the Humanities (DSH)  
*Recommended repositioning:* Per peer review, the manuscript is stronger as a lessons-learned / cautionary tale about metric validation than as a mature metric deployment. The central contribution is not LDRI but the diagnostic finding that 96.5% of signal comes from confidence-weighting, not feature richness.

---

## Abstract (250 words)

LIT-DISC is a computational pipeline for measuring literary-discourse properties through a hierarchical discourse processing tree. We present v1.1 alongside a critical post-mortem of what it actually measures. The core finding is diagnostic: 96.5% of the metric's discriminative signal comes from a single component — the confidence weight gamma — while the remaining discourse richness factors contribute <3% of variance. This gamma dominance is structural (95.3% at 400× novel scale) and not reducible to readability or lexical diversity (max ρ = 0.53). Scaling to a full-length novel (The Mistress of Spices, 80,728 words) reveals additional architectural failures: a memory system that fills to capacity in the first window and never retrieves across 105 windows, a resonance module contributing 0.02% of component value, and an activation function that required a single-parameter correction (sigma_max: 0.1 → 5.0) to achieve 155× CV improvement from 0.31% to 47.9%. The paper provides a reproducible methodology (55 equations, 116/117 tests passing), honest accounting of limitations (21 documented), and concrete paths to v2.0. The central lesson is that complex multi-component metrics can appear to measure what they claim while actually measuring confidence-weighting — a finding with implications for any computational metric built from hand-crafted features. All artifacts are publicly available.

**Keywords:** literary discourse, computational literary studies, metric validation, narrative analysis, discourse processing

---

## 1. Introduction (~1,500 words)

### 1.1 The Problem

Computational literary studies lacks a standard metric for literary-discourse richness. Existing approaches fall into two camps: (a) qualitative close reading (not scalable, not computable), and (b) text-as-data methods (topic modeling, sentiment, readability) that capture distributional properties but not discourse complexity.

**Evidence:** `evidence/01_framework/framework_summary.md`

### 1.2 Our Approach

LIT-DISC addresses this gap through a hierarchical multi-scale discourse processing tree (HMP-DT) that decomposes text into six discourse dimensions, each computed from learned features and weighted by activation-based confidence, then aggregated into the Literary Discourse Richness Index (LDRI).

**Claim:** C1–C3 | **Evidence:** `evidence/01_framework/framework_summary.md`

### 1.3 Key Challenges

- **Activation saturation:** At initial sigma_max=0.1, 99% of activations saturated — the metric was effectively constant (CV=0.31%).
- **Gamma dominance:** LDRI is 96.5% determined by uncertainty weighting, not feature richness.
- **Validation gap:** No external human annotation study exists.

**Evidence:** `evidence/02_validation/activation_design_review.md`

### 1.4 Contributions

1. LIT-DISC v1.1: principled activation redesign (sigma_max=0.1 → 5.0) eliminating saturation
2. Comprehensive validation: 116/117 tests, 4 baselines, 5 ablations, 25-passage gold standard
3. First full-novel evaluation: 80,728 words, 105 windows, persistent memory, narrative event mapping
4. Open-science artifacts: all code, evidence, and analysis publicly available

**Related claims:** C50–C54 (v1.0 → v1.1 improvements)

### 1.5 Paper Structure

Section 2 reviews related work. Section 3 presents the HMP-DT framework. Section 4 describes experimental setup. Section 5 reports validation results. Section 6 presents the Mistress of Spices case study. Section 7 discusses implications and limitations. Section 8 concludes with future work.

---

## 2. Related Work (~2,000 words)

### 2.1 Discourse Analysis Frameworks

Rhetorical Structure Theory (RST), Segmented Discourse Representation Theory (SDRT), Penn Discourse TreeBank (PDTB). All are discrete (tree relations, manual annotations) — continuous automatic discourse metrics are rare.

### 2.2 Text-as-Data Metrics

Readability (Flesch-Kincaid, Dale-Chall), lexical diversity (TTR, MTTR), perplexity (language models), sentiment, LIWC categories. Each captures a single dimension; none measures discourse richness.

**Evidence:** `evidence/03_baselines/baseline_summary.md` (LDRI correlations ρ = −0.16 to 0.53)

### 2.3 Computational Literary Studies

Genre classification, authorship attribution, stylometry, narrative generation. These use text metrics as features but do not propose a unified richness metric.

### 2.4 Metric Validation Standards

Need for: reproducibility, numerical stability, interpretability, human validation. LIT-DISC addresses the first three but not the fourth (future work).

**Evidence:** `evidence/09_reproducibility/reproducibility.md`, `evidence/10_claims/unsupported_claims.md:U3`

---

## 3. Framework — HMP-DT and LDRI (~3,000 words)

### 3.1 Architecture Overview

5-level hierarchy: Surface → Structure → Abstraction → Conceptual → Relational. 6 discourse dimensions (emotion, narrative, symbolic, semantic, evolution, interaction). Pipeline: Load → Preprocess → Hierarchy → Features → Signals → HMP-DT (propagate, couple, memory, resonance, uncertainty) → Embed → LDRI.

**Figure 1:** Architecture diagram (Mermaid)
**Evidence:** `evidence/01_framework/framework_summary.md`

### 3.2 Feature Computation

47 equations across 6 dimensions. Each dimension d produces feature norms ||H_d|| from text tokens.

**Claim:** C1 | **Evidence:** `evidence/01_framework/framework_summary.md`

### 3.3 Embedding and Activation

After feature propagation, embedding subspaces $\mathbf{H}_d \in \mathbb{R}^{d/10}$ are constructed per dimension (Eq. 53–55) from propagated signal statistics. The pairwise interaction subspace $\mathbf{H}_{\text{pairs}}$ and resonance trajectory subspace $\mathbf{H}_{\text{res}}$ are computed alongside.

Each dimension's activation is then computed from its embedding norm via the Rayleigh CDF:
$$a_d = 1 - \exp\left(-\|\mathbf{H}_d\|^2 / 2\sigma_{\max}^2\right)$$
This is NOT a sigmoid — see Results Section 5.2 for the activation redesign discovery.

**Figure 2:** Activation function at sigma=0.1 (saturated) vs sigma=5.0 (sensitive region)
**Evidence:** `evidence/02_validation/activation_design_review.md`, `docs/methodology_equations.md§3.1,§3.29`

### 3.4 LDRI Computation

LDRI = gamma × (R_signal² × C_tilde × T_tilde × M_tilde)^(1/5)

where each component: coupling (C), temporal coherence (T), memory modulation (M), resonance (R), and gamma (uncertainty weighting).

**Claim:** C2 | **Evidence:** `evidence/01_framework/framework_summary.md`

### 3.5 Memory Architecture

RFF-based key generation, Gaussian similarity threshold retrieval, max_size=200, FIFO eviction, sliding-window decay (half-life 20 windows).

**Evidence:** `evidence/01_framework/framework_summary.md`, `evidence/05_scalability/scalability_summary.md`

*(The activation redesign story (v1.0 sigma_max=0.1 → v1.1 sigma_max=5.0) is presented in Results Section 5.2 as an empirical finding, not in Framework. A brief summary: a single parameter change from 0.1 to 5.0 eliminated 95% of activation saturation, increased LDRI CV from 0.31% to 47.9% (155×), and preserved author ranking ρ=0.98.)*

---

## 4. Experimental Setup (~2,000 words)

### 4.1 Gold-Standard Corpus

25 passages from 25 authors, 5 narrative categories (SNC, MLC, NWC, SSC, MNC), ~200 words each. Passage selection criteria.

**Figure 3:** Gold-standard composition (table + category distribution)
**Evidence:** `evidence/02_validation/gold_standard_summary.md`

### 4.2 Full-Novel Selection

The Mistress of Spices (Chitra Banerjee Divakaruni, 1997). Rationale: public availability, manageable length (80,728 words), literary recognition (Booker shortlist), discourse-rich content (magical realism, postcolonial identity).

**Evidence:** `evidence/06_case_study/mistress_case_study.md`

### 4.3 Processing Pipeline

spaCy tokenization, 800-word windows with paragraph boundaries, persistent memory across windows (monkey-patched to prevent memory.clear between windows).

**Evidence:** `evidence/09_reproducibility/reproducibility.md`

### 4.4 Baseline Conditions

| Condition | Change | Purpose |
|-----------|--------|---------|
| Full pipeline | None | Reference |
| Constant features | Fix all H_d to mean | Test feature contribution |
| Random gamma | Replace gamma with uniform random | Test gamma essentiality |
| Dummy | Zero all inputs | Test base behavior |

**Evidence:** `evidence/03_baselines/baseline_summary.md`

### 4.5 Ablation Conditions

| Condition | Change | Purpose |
|-----------|--------|---------|
| Without memory | Disable memory component | Test memory contribution |
| Without resonance | Disable resonance | Test resonance contribution |
| Without coupling | Disable adaptive coupling | Test coupling contribution |
| Without uncertainty | Set gamma=1.0 | Test gamma essentiality |
| Without narrative | Zero narrative features | Test narrative dimension |

**Evidence:** `evidence/04_ablations/ablation_summary.md`

### 4.6 Evaluation Metrics

LDRI (mean, std, CV), raw LDRI, gamma, runtime, memory usage, narrative event detection, variance decomposition, Spearman ranking correlations.

**Evidence:** `evidence/07_statistics/statistics_summary.md`

### 4.7 Reproducibility

Environment: Windows, Python 3.14. Determinism verified (< 1e-12 difference on re-run). Full execution order documented.

**Evidence:** `evidence/09_reproducibility/reproducibility.md`

---

## 5. Results (~4,000 words)

### 5.1 Verification Suite

116/117 tests pass (39 equation, 10 numerical, 3 determinism, 57 memory/temporal, 6 edge cases, 1 performance failure — pre-existing, machine-dependent).

**Evidence:** `evidence/09_reproducibility/reproducibility.md`

### 5.2 Activation Redesign: The Central Methodological Finding

**Discovery:** In v1.0, sigma_max was set to 0.1. Feature norms $\|\mathbf{H}_d\|$ were in the range 10–14 (mean ≈ 11.09), producing normalized arguments $z = 110$ — far beyond the $5\sigma$ saturation threshold of the Rayleigh CDF. Result: **149 of 150 activations saturated at $> 0.999$**, and LDRI had CV = 0.31% (effectively constant).

**Fix:** A single parameter change — sigma_max: 0.1 → 5.0 (config.py:59). At sigma=5.0, $z = 11/5 = 2.2$, operating in the sensitive region of the CDF.

**Results:**

| Metric | v1.0 (sigma=0.1) | v1.1 (sigma=5.0) | Change |
|--------|------------------|------------------|--------|
| Saturated activations | 149/150 (99%) | 6/150 (4%) | −95% |
| Mean activation | 0.996 | 0.575 | −42% |
| Activation CV | 0.2% | 22.1% | 111× |
| LDRI CV | 0.31% | 47.9% | 155× |
| Author Spearman ρ | — | 0.98 (vs v1.0) | preserved |

**Figure 4:** Activation distribution: v1.0 (all 0.999+) vs v1.1 (uniform across [0.2, 1.0])
**Evidence:** `evidence/02_validation/activation_design_review.md`, `evidence/02_validation/regression_report.md`

### 5.3 Gold-Standard LDRI

**Mean LDRI = 0.0277** (Std=0.0133, CV=47.9%, Range=[0.0088, 0.0551])

**Category ranking:** SNC (0.0403) > MLC (0.0329) > NWC (0.0272) > SSC (0.0198) > MNC (0.0188)

**Author extremes:** Woolf (0.055), Ferrante (0.041), Rushdie (0.035), Mantel (0.009)

**Figure 5:** LDRI by author (horizontal bar chart)
**Figure 6:** LDRI by category (box plot)
**Evidence:** `evidence/07_statistics/statistics_summary.md`

### 5.4 Variance Decomposition

**log-space analysis:**

| Source | Full | Breakdown |
|--------|------|-----------|
| var(log(LDRI)) | 0.2609 | — |
| Gamma contribution | **96.5%** | var(log(gamma)) / var(log(LDRI)) |
| Raw contribution | 0.3% | var(log(raw)) / var(log(LDRI)) |
| Covariance | 3.2% | 2·cov / var(log(LDRI)) |

**Figure 7:** Variance decomposition (pie chart)
**Evidence:** `evidence/07_statistics/statistics_summary.md`

**This is the paper's central empirical finding.** Gamma dominance is not a scaling artifact — it is structural, as confirmed at novel scale (Section 5.8). See Discussion Section 7.2 for architectural interpretation.

### 5.5 Baseline Results

| Condition | LDRI mean | LDRI std | ρ (vs full) | Interpretation |
|-----------|-----------|----------|-------------|----------------|
| Full pipeline | 0.0277 | 0.0133 | 1.000 | Reference |
| Constant features | 0.0280 | 0.0130 | 0.998 | Gamma alone almost replicates LDRI |
| Random gamma | 0.0272 | 0.0030 | **0.14** | Gamma is essential |
| Dummy | 0.0 | 0.0 | — | Base behavior |

**Key finding:** Random gamma ρ = 0.14 proves gamma is the sole carrier of discriminative signal.

**Comparison metrics:**

| Metric | ρ with LDRI |
|--------|-------------|
| Log(Flesch-Kincaid grade) | −0.16 |
| Type-token ratio | 0.36 |
| Log(avg sentence length) | 0.53 |

**LDRI is not reducible to readability or lexical diversity.**

**Evidence:** `evidence/03_baselines/baseline_summary.md`

### 5.6 Null Model Baseline

**Design:** Two null conditions on all 25 gold-standard passages:
- **(a) Shuffled text:** paragraphs randomly reordered (preserves tokens, destroys narrative sequence)
- **(b) Random embeddings:** all 8 embedding subspaces replaced with Gaussian noise (mean/variance matched to empirical distribution, preserving scale but destroying signal)

**Results:**

| Metric | Baseline | Shuffled | Random Embeddings |
|--------|----------|----------|-------------------|
| LDRI mean | 0.0277 | 0.0278 | 0.0302 |
| LDRI CV | 46.9% | 48.8% | 46.0% |
| Raw mean | 0.9128 | 0.9135 | **0.9986** |
| Spearman ρ (vs baseline) | — | **0.960** | **0.995** |

**Key findings:**
1. **Shuffling barely matters (ρ=0.96).** Paragraph order contributes negligibly to LDRI — the metric captures per-passage feature statistics, not narrative structure.
2. **Random embeddings preserve ranking (ρ=0.995).** With noise replacing all embeddings, raw LDRI saturates (0.9986) and LDRI ≈ Gamma. The ranking is unchanged.
3. **Gamma is the sole carrier.** The null model proves experimentally that the embedding pipeline contributes nothing to LDRI's discriminative signal — gamma from feature extraction drives everything.

**Evidence:** `evidence/03_baselines/null_model_summary.md`, `data/experiments/null_model/null_model_results.json`

### 5.7 Ablation Results

| Condition | LDRI | Δ | Effect |
|-----------|------|---|--------|
| Full | 0.0277 | — | Reference |
| No memory | 0.0264 | −0.0013 | SMALL |
| No resonance | 0.0261 | −0.0016 | SMALL |
| No coupling | 0.0292 | +0.0015 | SMALL |
| **No uncertainty** | **0.9126** | **+0.8850** | **LARGE** |
| No narrative | 0.0393 | +0.0116 | MODERATE |

**Without uncertainty = raw LDRI (0.913).** Removing gamma collapses LDRI to a constant. Without narrative, LDRI increases because narrative has the lowest per-dimension gamma.

**Evidence:** `evidence/04_ablations/ablation_summary.md`

### 5.8 Full-Novel Scalability

**Runtime:** 85.0s for 80,728 words (105 windows). Linear scaling: 1.05s/1k words, r=0.23, second/first half ratio = 1.08×.

**Figure 8:** Runtime scaling (cumulative words vs time)
**Evidence:** `evidence/05_scalability/scalability_summary.md`

**Memory:** 200/200 entries filled in first window. Zero cross-window retrievals across all 105 windows.

**Figure 9:** Memory growth (windows vs entries)
**Evidence:** `evidence/05_scalability/memory_analysis.csv`

**Longitudinal LDRI:** Mean=0.0100 (∼1/3 of gold standard), CV=21.8%, gamma CV=21.3%.

**Variance decomposition (novel):** Gamma = 95.3%, raw = 0.2%, covariance = 4.5%.

**Gamma dominance is structurally unchanged at novel scale.** The 400× increase in text length does not reduce the asymmetry (96.5% → 95.3%).

**Evidence:** `evidence/05_scalability/scalability_summary.md`

---

## 6. Case Study — The Mistress of Spices (~3,000 words)

### 6.1 Novel Synopsis

Chitra Banerjee Divakaruni's The Mistress of Spices (1997). Narrative follows Tilo, a spice mistress in Oakland, torn between magical duty and human connection. Postcolonial magical realism, 80,728 words.

**Evidence:** `evidence/06_case_study/mistress_case_study.md`

### 6.2 LDRI Trajectory

105 windows, 8 narrative phases:

| Phase | Windows | LDRI Pattern | Content |
|-------|---------|-------------|---------|
| Opening | 1–10 | Oscillating, falling | Youth → store exposition |
| Early crisis | 11–20 | Rise then plunge | Channeling the Old One |
| Middle | 21–35 | Lower with spike | Makaradwaj lore (peak: +0.0078) |
| Spiritual crisis | 36–50 | Two spikes | Self-punishment (novel peak: 0.0155) |
| Action | 51–60 | Violence spike | Racist assault (0.015) |
| Memory/trauma | 61–75 | Low, introspective | First-person reflection |
| Climax | 76–85 | Peak→valley | Reunion (0.014) → Renunciation (0.0057) |
| Resolution | 86–105 | Transformation | Liberation → denouement |

**Figure 10:** LDRI over 105 windows (with event annotations)
**Figure 11:** Raw LDRI over 105 windows (near-constant)
**Figure 12:** Gamma over 105 windows (drives all variation)
**Evidence:** `evidence/06_case_study/narrative_event_log.md`

### 6.3 Narrative Events

16 events identified via LDRI/component z-score thresholds. For each event: window, cumulative words, LDRI, component values, text excerpt, narrative interpretation.

**Table 3:** Major narrative events with LDRI values

**Key events:**
- **E4 (W25→W26):** Largest jump (+0.0078). Makaradwaj spice lore.
- **E6 (W40):** Novel peak (0.0155). Spiritual crisis: "Spices is this. The way. You have chosen. To. Punish me."
- **E13 (W81):** Reunion joy (0.014). Hugging, laughing, tears.
- **E14 (W82):** Deepest valley (0.0057). "EVERYTHING MUST GO." — renunciation.
- **E15 (W89):** Transformation (0.015). "O you who are young."

**Evidence:** `evidence/06_case_study/narrative_event_log.md`

### 6.4 Interpretation

LDRI peaks correspond to high literary-discourse intensity: crisis, violence, reunion, transformation. Valleys correspond to exposition, introspection, transaction.

**However:** The novel's most important moment (renunciation, W82) scores lowest. LDRI captures linguistic complexity, not narrative importance. This distinction is critical.

**Evidence:** `evidence/06_case_study/mistress_case_study.md`

### 6.5 Comparison with Gold Standard

| Metric | Gold Standard (25) | Full Novel (105) | Ratio |
|--------|-------------------|------------------|-------|
| LDRI mean | 0.0277 | 0.0100 | 2.8× |
| LDRI CV | 47.9% | 21.8% | 2.2× |
| Gamma dominance | 96.5% | 95.3% | 1.01× |
| Raw LDRI | 0.913 | 0.950 | 1.04× |

**Figure 13:** Variance decomposition: gold standard vs novel (side-by-side bar)
**Evidence:** `evidence/05_scalability/variance_decomposition_novel.csv`

---

## 7. Discussion (~3,000 words)

### 7.1 What LDRI Actually Measures

**LDRI = gamma × near-constant.** The metric measures confidence-weighted discourse richness, where confidence (gamma) carries 96.5% of the signal. Raw LDRI varies by only 0.3–2.6%.

**Implication:** The metric name ("Literary Discourse Richness Index") is misleading. It should be understood as a confidence-weighted index, not an intrinsic richness index.

**Evidence:** `evidence/07_statistics/statistics_summary.md`, `evidence/03_baselines/baseline_summary.md`

### 7.2 Gamma Dominance: Architectural or Fundamental?

Three experiments confirm it is architectural:
1. **Novel evaluation:** Gamma = 95.3% — unchanged from 96.5% at 400× scale
2. **Constant Features baseline:** r > 0.99 — features contribute nothing beyond gamma's mediation
3. **Random Gamma baseline:** ρ = 0.14 — without gamma, LDRI is random

**Conclusion:** Gamma dominance is a structural property of LDRI = gamma × (geometric mean of components). The geometric mean of near-1 values produces near-constant output; gamma provides all variation.

**New null-model evidence (Section 5.6):** Random embeddings produce ρ=0.995 with baseline — even destroying the entire embedding pipeline leaves ranking unchanged. Gamma from feature extraction alone explains 99.5% of the metric's behavior.

**Evidence:** `evidence/02_validation/regression_report.md`, `evidence/03_baselines/baseline_summary.md`, `evidence/05_scalability/scalability_summary.md`

### 7.3 Memory: Dormant at Novel Length

The RFF-based memory fills to capacity (200 entries) in window 1 and never retrieves across 105 windows. This is the single most surprising architectural finding.

**Why it happens:** RFF keys are computed from sentence-level features. As the narrative context changes, keys drift below the retrieval threshold (θ=0.7). Memory becomes a per-window slot-filling operation without cross-window function.

**Threshold sweep diagnosis (θ=0.1–0.9):** Even at θ=0.1, within-window retrieval is universal (avg_beta=1.0 across all passages). But across 105 novel windows, retrieval is zero at θ=0.7. The problem is not the threshold — lowering it would not help. The RFF keys drift so far across narrative context that even the most permissive threshold would not match. **The problem is architectural:** the RFF representation of sentence-level features lacks temporal or semantic invariance needed for cross-window retrieval.

**Evidence:** `evidence/05_scalability/threshold_sweep_summary.md`

**Implication:** The memory system as designed does not achieve its stated goal. A structural redesign (e.g., semantic chunking, content-based hashing, overlapping windows) is required for v2.0.

**Evidence:** `evidence/05_scalability/scalability_summary.md`

### 7.4 Resonance: Vestigial

Mean resonance at novel length = 0.00008 (0.0002 in gold standard). Removal changes LDRI by |Δ| = 0.0016. Resonance contributes ~0.02% of total component value.

**Implication:** The resonance mechanism is either theoretically unnecessary or incorrectly implemented. Redesign or removal required.

**Evidence:** `evidence/04_ablations/ablation_summary.md`, `evidence/07_statistics/statistics_summary.md`

### 7.5 Narrative Dimension Highlights

The narrative dimension has the lowest per-dimension gamma. When removed, LDRI *increases* by +0.012 (+42%). This indicates that narrative features are noisier (lower signal-to-noise) than other dimensions.

**Possible causes:** (a) Narrative features (temporal markers, character references, causal connectors) are sparser than semantic or emotional features. (b) The narrative dimension's feature design may be suboptimal compared to established emotion/semantic lexica.

**Evidence:** `evidence/04_ablations/ablation_summary.md`

### 7.6 Limitations

| ID | Limitation | Severity | Reference |
|----|-----------|----------|-----------|
| L1 | Gamma dominance | HIGH | Section 7.2 |
| L2 | Zero cross-window retrieval | HIGH | Section 7.3 |
| L3 | Vestigial resonance | HIGH | Section 7.4 |
| L5 | No expert validation | HIGH | Section 7.7 |
| L6 | Single novel tested | HIGH | Section 6 |
| L7 | Small gold standard (N=25) | HIGH | Section 5.3 |
| L8–L21 | Additional limitations | MED/LOW | `evidence/limits.md` |

**Full limitation catalog:** `evidence/limits.md`

### 7.7 Comparison with Prior Work

- **vs readability (Flesch-Kincaid, ρ=−0.16):** LDRI captures something orthogonal
- **vs lexical diversity (TTR, ρ=0.36):** Weak positive — richer vocabulary correlates weakly
- **vs sentence length (ρ=0.53):** Moderate, expected — discourse complexity requires syntactic space

**Evidence:** `evidence/03_baselines/baseline_summary.md`

### 7.8 Implications for Computational Literary Studies

1. **Uncertainty is informative:** Gamma dominance is not necessarily a bug. Confidence-weighted metrics are appropriate when measurement uncertainty varies across text segments.
2. **Scalability is achievable:** LIT-DISC processes 80k words in 85 seconds — practical for corpus-level analysis.
3. **Memory needs rethinking:** Current RFF-based memory is inadequate for novel-length cross-window retrieval.

### 7.9 Threats to Validity

- **Circular event analysis:** Events identified from LDRI extremes (Section 6.3) — independent annotation needed
- **Null model confirms gamma dominance:** Shuffled text (ρ=0.96) and random embeddings (ρ=0.995) preserve ranking — LDRI measures gamma, not discourse richness (Section 5.6)
- **Single-rater interpretation:** Literary analysis (Section 6.4) represents one reader's judgment
- **No significance testing:** All reported differences — category rankings, baseline conditions, ablation effects — lack confidence intervals or p-values. Effect sizes appear large (e.g., no-uncertainty Δ=+0.885) but statistical significance is unknown
- **Activation sensitivity:** The sigma_max=5.0 setting was chosen post-hoc to match observed feature norms (10–14). If feature norms change (different lexica, languages, or dimensions), sigma_max may need recalibration

---

## 8. Conclusion (~1,000 words)

### 8.1 Summary of Contributions

1. **LIT-DISC v1.1:** A validated, reproducible metric for discourse confidence with linear scaling to novel length
2. **Empirical findings:** Gamma dominance (96.5%), dormant memory (zero cross-window retrieval), vestigial resonance
3. **Methodological contributions:** Baseline framework, ablation protocol, narrative event mapping methodology
4. **Open science:** All evidence publicly available with 42 documentation files

### 8.2 Key Takeaways

- LDRI measures **confidence-weighted** discourse richness, not intrinsic richness
- The metric scales linearly, making corpus-level analysis practical
- Three architectural components (memory, resonance, coupling) contribute negligibly in their current form
- Narrative event mapping is feasible but requires post-hoc interpretation, not prediction

### 8.3 Future Work

**v1.2 (3 months):** Expert annotation study, multi-novel replication (5+ novels), cross-language evaluation (1+ non-English novel), null model testing, retrieval threshold sweep, window size sensitivity

**v2.0 (6–12 months):** Redesigned memory (semantic chunking, sliding-window overlap, content-based hashing), new resonance formulation (cross-window echo, lexical chain persistence), alternative confidence modulation (arithmetic mean, learned weights, hierarchical gamma), adaptive long-context retrieval

**Full roadmap:** `evidence/future_work.md`

### 8.4 Closing

LIT-DISC v1.1 provides a foundation for computational literary-discourse analysis — a reproducible metric, a validated protocol, and an honest accounting of what it can and cannot measure. The path to v2.0 is clear, and the open-science artifacts enable the community to build on this work.

---

## Figures (13 total)

| # | Description | Location |
|---|-------------|----------|
| 1 | Architecture diagram | `evidence/01_framework/framework_summary.md` |
| 2 | Activation function: sigma=0.1 vs 5.0 | `evidence/08_figures` (new) |
| 3 | Gold-standard composition | `evidence/02_validation/gold_standard_summary.md` |
| 4 | Activation distribution (v1.0 vs v1.1) | `evidence/08_figures` (new) |
| 5 | LDRI by author | `evidence/08_figures` (new) |
| 6 | LDRI by category | `evidence/08_figures` (new) |
| 7 | Variance decomposition (pie) | `evidence/08_figures/events/variance_comparison.png` |
| 8 | Runtime scaling | `evidence/08_figures/runtime/runtime_scaling.png` |
| 9 | Memory growth | `evidence/08_figures/memory/memory_growth.png` |
| 10 | LDRI over 105 windows | `evidence/08_figures/ldri/ldri_progression.png` |
| 11 | Raw LDRI over 105 windows | `evidence/08_figures/ldri/raw_ldri_progression.png` |
| 12 | Gamma over 105 windows | `evidence/08_figures/gamma/gamma_progression.png` |
| 13 | Variance decomposition: GS vs novel | `evidence/08_figures/events/variance_comparison.png` |

## Tables (critical three)

| # | Description | Evidence |
|---|-------------|----------|
| 1 | Baseline results (4 conditions) | `evidence/03_baselines/baseline_summary.md` |
| 2 | Ablation results (5 conditions) | `evidence/04_ablations/ablation_summary.md` |
| 3 | Major narrative events (16 events) | `evidence/06_case_study/narrative_event_log.md` |

---

## Appendix

**A. Full Feature List** — 47 equations (reference)
**B. Passages Source List** — 25 passages, author, novel, year, edition
**C. Full Novel Results** — 105-window data (partial)
**D. Evidence Index** — All files with descriptions (`evidence/evidence_index.md`)
**E. Limitations Catalog** — 21 limitations (`evidence/limits.md`)
**F. Unsupported Claims** — 5 claims needing future work (`evidence/10_claims/unsupported_claims.md`)
