# Results Story — LIT-DISC v1.1

The chronological narrative of experimental discovery, from problem through final conclusions.

---

## 1. The Problem

**Context:** LIT-DISC v1.0 (sigma_max = 0.1) was a working framework that computed Literary Discourse Richness Index (LDRI) across 25 gold-standard passages. The pipeline processed text through the Hierarchical Multi-scale Discourse Processing Tree (HMP-DT), computing features across 6 dimensions, and reducing them via uncertainty-weighted geometric mean.

**The symptom:** The output LDRI values "looked right" — they ranked Virginia Woolf above Salman Rushdie above Elena Ferrante. But the mechanism producing this ranking was suspect.

**The question that started everything:** *What is LDRI actually measuring?*

## 2. Initial Evaluation (v1.0)

**Evidence:** `evidence/02_validation/activation_design_review.md`

The gold-standard evaluation at sigma=0.1 showed:
- LDRI mean = 0.9008, CV = 0.31%
- Raw LDRI mean = 0.9131, CV = 0.31%
- Gamma mean = 0.9865, CV = 0.23%

These numbers tell a story: **everything is 0.9**. LDRI, raw LDRI, and gamma all hover near 0.9 with almost no variation. The system could rank passages, but the ranking was driven by differences in the fourth decimal place.

**This is not a metric for literary scholarship. It's a system shouting "everything is the same."**

## 3. Saturation Discovery

**Evidence:** `evidence/02_validation/activation_design_review.md`

The activation function was a sigmoid with sigma_max = 0.1. The feature norms ||H_d|| were 10–14. A norm of 11 divided by sigma=0.1 gives 110. Sigmoid(110) ≈ 1.0 for all practical purposes.

**The diagnosis:** 149 out of 150 activations were saturated at >0.999. The entire system was operating in the flat tail of the sigmoid where nothing discriminates. The "ranking" was numerical noise amplified by floating-point precision.

**Mathematical root cause:** The sigmoid's scale parameter sigma_max was 100–140× smaller than the feature norms it received. Sigmoid(z) ≈ 1 for z > 5. With z = ||H_d|| / sigma, every input was z > 100. Every activation was functionally identical.

## 4. Diagnostic Investigation

**Evidence:** `evidence/02_validation/activation_design_review.md`

Three converging lines of evidence confirmed the diagnosis:

1. **Saturation count:** 149/150 activations at >0.999
2. **Feature norm analysis:** All ||H_d|| ≈ 10–14, sigma=0.1 → z=100–140
3. **CV collapse:** LDRI CV = 0.31% means essentially no variation

**The insight:** The activation function (phi = sigmoid with learned steepness) was designed to prune dimensions. But with sigma_max=0.1, it pruned *everything*, making all dimensions equally important and the metric indifferent to textual variation.

**Why wasn't this caught earlier?** Because the ranking was *reproducible*. The same saturated system gave the same answers on every run. The ranking was deterministic but meaningless — the numerical equivalent of a constant function.

## 5. Sigma Redesign (v1.1)

**Evidence:** `evidence/02_validation/activation_design_review.md`, `src/config.py:59`

The fix was a single parameter change: sigma_max = 5.0. This aligns the activation scale with observed feature norms (||H_d|| ≈ 10–14, so z = 2–3, operating in the sigmoid's linear region).

**The verification:** At sigma_max = 5.0:
- Saturation drops from 99% to 4% (6/150 activations > 0.999)
- Mean activation drops from 0.996 to 0.575
- Activation CV increases from 0.2% to 22.1%
- Activation range expands from [0.973, 0.999] to [0.229, 0.999]

**Ranking preservation:** Despite the radical change in absolute values, author Spearman ρ = 0.98. The system ranks authors almost identically before and after. The signal was always there in the features; the activation was just masking it.

**The engineering lesson:** A parameter can be "wrong" by two orders of magnitude and still produce seemingly reasonable rankings, because relative structure propagates through a monotonic transformation. Only CV analysis reveals the pathology.

## 6. Regression Study

**Evidence:** `evidence/02_validation/regression_report.md`

A point-by-point comparison of v1.0 vs v1.1 across all metrics:

| Metric | v1.0 | v1.1 | Change |
|--------|------|------|--------|
| LDRI mean | 0.9008 | 0.0277 | −97% |
| LDRI CV | 0.31% | 47.9% | **155× increase** |
| Raw LDRI mean | 0.9131 | 0.9128 | −0.03% (identical) |
| Gamma mean | 0.9865 | 0.0302 | −97% |
| Activation mean | 0.996 | 0.575 | −42% |
| Author Spearman ρ | 1.0 | 0.98 | preserved |

**The critical finding:** Raw LDRI is unchanged. LDRI = gamma × raw_ldri. The raw LDRI captures feature geometry; gamma carries all discriminative signal. v1.0's gamma was uniformly near 1.0 (no uncertainty); v1.1's gamma operates across [0.01, 0.06], producing LDRI in a meaningful range.

**No regressions in any component.** The intrinsic component changes (as expected — it contains the activation), but temporal, memory, interaction, and resonance are stable.

## 7. Gold-Standard Validation

**Evidence:** `evidence/02_validation/gold_standard_summary.md`, `evidence/03_baselines/baseline_summary.md`, `evidence/04_ablations/ablation_summary.md`, `evidence/07_statistics/statistics_summary.md`

With v1.1 validated, three experimental stages established the metric's behavior:

### Stage 1: Baselines

**Evidence:** `evidence/03_baselines/baseline_summary.md`

| Baseline | LDRI mean | LDRI std | Correlation with full |
|----------|-----------|----------|---------------------|
| Full pipeline | 0.0277 | 0.0133 | 1.000 |
| Constant features | 0.0280 | 0.0130 | 0.998 |
| Random gamma | 0.0272 | 0.0030 | 0.18 |
| Dummy | 0.0 | 0.0 | — |

**Random Gamma ρ = 0.14.** This is the single most important baseline result. If you replace the computed gamma with random values, LDRI ranking collapses. This proves gamma is not decorative — it's the sole carrier of discriminative signal.

**Constant Features r > 0.99.** If you hold features constant and let gamma vary, LDRI is almost unchanged. This confirms the dominance from the opposite direction: features contribute almost nothing to LDRI variation.

### Stage 2: Comparisons

| Condition | ρ with full LDRI | Interpretation |
|-----------|-----------------|----------------|
| Full LDRI | 1.00 | Reference |
| Log readability | 0.41 | Readability ≠ richness |
| Avg sentence length | 0.53 | Moderate correlation |
| Type-token ratio | 0.36 | Weak correlation |
| Flesch-Kincaid grade | −0.16 | No relationship |

LDRI is not reducible to readability or lexical diversity. The moderate correlation with sentence length (ρ=0.53) is expected — longer sentences carry more discourse complexity.

### Stage 3: Ablations

**Evidence:** `evidence/04_ablations/ablation_summary.md`

| Ablation | LDRI | Δ | Effect |
|----------|------|---|--------|
| Full pipeline | 0.0277 | — | Reference |
| Without memory | 0.0264 | −0.0013 | SMALL |
| Without resonance | 0.0261 | −0.0016 | SMALL |
| Without coupling | 0.0292 | +0.0015 | SMALL |
| Without uncertainty | 0.9126 | +0.8850 | **LARGE** |
| Without narrative | 0.0393 | +0.0116 | MODERATE |

**Without uncertainty → LDRI = 0.913 = raw LDRI.** This is the second critical result. Setting gamma = 1.0 (no uncertainty weighting) reproduces raw LDRI exactly. LDRI ≈ gamma × constant, where the "constant" varies by only 2.6% across passages.

**Without narrative → LDRI increases by 42%.** Removing the narrative dimension (which has the lowest per-dimension gamma) increases LDRI because the remaining dimensions have higher confidence.

### Summary of Gold-Standard Findings

**Evidence:** `evidence/07_statistics/statistics_summary.md`

- **Mean LDRI = 0.0277** (range: 0.0088–0.0551)
- **Category ranking:** SNC (0.0403) > MLC (0.0329) > NWC (0.0272) > SSC (0.0198) > MNC (0.0188)
- **Author extremes:** Woolf (0.055) > Ferrante (0.041) > Rushdie (0.035) > ... > Mantel (0.009)
- **Variance decomposition:** gamma = 96.5%, raw LDRI = 0.3%, covariance = 3.2%
- **Feature means:** semantic (3.50) > narrative (3.11) > evolution (2.97) > interaction (2.65) > symbolic (2.59) > emotion (2.55)

## 8. Full-Novel Evaluation

**Evidence:** `evidence/05_scalability/scalability_summary.md`

With the gold-standard established, the question shifted: *Does LDRI work at novel length?*

**Design:** Process *The Mistress of Spices* (80,728 words) as 105 windows of ~800 words with paragraph-boundary respecting split. Memory persists across all windows.

### Runtime Scaling

- Total: 85.0 seconds for 80,728 words
- Average: 0.81s per window, 1.05s per 1,000 words
- Runtime–word correlation: r = 0.23
- Second/first half ratio: 1.08×
- **Conclusion: Linear scaling, no bottlenecks**

### Memory Analysis

- Peak memory entries: 200/200 (capacity)
- Windows at capacity: 105/105
- Cross-window retrievals: **0** across all windows
- First window fill rate: 200 entries in ~800 words

**The memory system fills to capacity in the first window and never retrieves.** RFF keys drift across narrative windows, preventing cosine-similarity matches above the retrieval threshold (θ = 0.7). Memory is functionally a per-window bag, not a cross-window narrative memory.

### Longitudinal LDRI

- Mean LDRI: 0.0100 (vs 0.0277 in gold standard)
- Std: 0.0022 | CV: 21.8% | Range: [0.0057, 0.0155]
- Gamma mean: 0.0105 | Gamma CV: 21.3%
- Raw LDRI mean: 0.9503 | Raw LDRI CV: 1.0%

**Novel LDRI is ~1/3 of gold-standard LDRI.** The reduction is driven by lower gamma at novel scale, not feature depletion. Raw LDRI is actually higher at novel length (0.950 vs 0.913).

### Variance Decomposition (Novel)

| Source | Gold Standard | Full Novel |
|--------|-------------|------------|
| Gamma contribution | 96.5% | 95.3% |
| Raw contribution | 0.3% | 0.2% |
| Covariance | 3.2% | 4.5% |

**Gamma dominance is structurally unchanged at novel scale.** A 400× increase in text length does not reduce the asymmetry. The covariance term rises slightly (3.2% → 4.5%), suggesting a modest positive correlation between gamma and raw LDRI (r = 0.50).

### Gamma–Raw LDRI Correlation

Gold standard: r = 0.10 (negligible). Full novel: r = 0.50 (moderate). The increase suggests that at novel length, passages that are more feature-rich also receive higher confidence weighting. This is the first hint of a meaningful signal in the gamma-raw interaction — but even at r = 0.50, the interaction remains small.

## 9. Literary Case Study

**Evidence:** `evidence/06_case_study/mistress_case_study.md`, `evidence/06_case_study/narrative_event_log.md`

### The Narrative Arc in LDRI

Processing Llosa: "The Mistress of Spices" (Chitra Banerjee Divakaruni) revealed 16 narrative events across 105 windows, each mapped to LDRI changes:

| Phase | Windows | LDRI Pattern | Narrative Content |
|-------|---------|-------------|-------------------|
| Opening | 1–10 | Oscillating, falling | Youth narrative → store exposition |
| Early crisis | 11–20 | Rise then plunge | Channeling the Old One |
| Middle | 21–35 | Lower with spike | Makaradwaj lore (largest jump: +0.0078) |
| Spiritual crisis | 36–50 | Two spikes | Novel peak at W40 (LDRI=0.0155): "Spices is this. The way. You have chosen. To. Punish me." |
| Action | 51–60 | Violence spike | Racist assault on Raven (W55: LDRI=0.015) |
| Memory/trauma | 61–75 | Low, introspective | First-person reflection |
| Climax | 76–85 | Tight peak→valley | Reunion (W81: 0.014) → Renunciation (W82: 0.0057, deepest valley) |
| Resolution | 86–105 | Transformation then denouement | Physical liberation (W89: 0.015) |

### Key Interpretations

1. **LDRI peaks correspond to scenes of highest literary-discourse intensity:** spiritual crisis, violence, emotional reunion, bodily transformation.

2. **LDRI valleys correspond to exposition, introspection, commercial transaction:** pragmatic dialogue, first-person memory, customer interactions.

3. **The tightest peak→valley is W81→W82 (reunion→renunciation):** Δ = −0.0082 in a single window transition. This captures the novel's central dramatic pivot.

4. **Raw LDRI is lower during introspective passages:** 5 windows with z < −2.3 are all first-person reflection, suggesting (weakly) that raw LDRI captures some narrative mode variation.

### A Critical Observation

The LDRI trajectory is **interpretable but coarse**. The deviation between W81 (reunion, LDRI=0.014) and W40 (spiritual crisis, peak LDRI=0.0155) is only 10%. The system can identify *that* something changed, but cannot answer *how much* or *why* at a fine-grained level.

## 10. Final Conclusions

### What LIT-DISC v1.1 Achieved

1. **A mathematically grounded, reproducible metric for literary-discourse richness.** The HMP-DT framework provides a principled decomposition of text into 6 discourse dimensions, each computed from features and weighted by learned activation confidences.

2. **Validated activation design.** sigma_max = 5.0 eliminates the v1.0 saturation pathology while preserving ranking structure. The activation function now operates in its sensitive region.

3. **Comprehensive experimental foundation.** 66/67 verification tests, 25 gold-standard passages, 4 baselines, 5 ablations, 105-window novel evaluation. Every claim is traceable to evidence.

4. **First literary case study at novel length.** 80,728 words processed with persistent memory. The LDRI trajectory traces the narrative arc of a complete novel.

### What LIT-DISC v1.1 Reveals About Its Own Architecture

1. **LDRI is gamma.** The metric is overwhelmingly determined by uncertainty weighting. Raw LDRI contributes < 1% of variation. LDRI ≈ gamma × constant.

2. **Memory is dormant.** The RFF-based memory fills to capacity in one window and never retrieves across windows. Memory's contribution to LDRI is |Δ| = 0.0013 when removed entirely.

3. **Resonance is vestigial.** Mean resonance = 0.00008 in novel evaluation. |Δ| = 0.0016 when removed. It contributes ~0.02% of total component value.

4. **Gamma dominance is structural, not a scaling artifact.** It persists at 95–96.5% regardless of text length (400× increase tested). Addressing it requires reformulating LDRI, not scaling.

### The Central Unsolved Problem

**LDRI is a gamma-weighted constant.** The system produces a per-window confidence estimate (gamma) and multiplies it by a near-constant feature richness (raw LDRI). The feature richness varies by only 1–3% (CV), while gamma varies by 21–47%.

This means:
- LDRI tells us which passages the model is **confident** about
- It does NOT tell us which passages are **most discourse-rich**
- These are different questions, and LIT-DISC currently answers only the first

The paper must clearly distinguish between "confidence-weighted richness" and "intrinsic richness" — and acknowledge that LIT-DISC v1.1 measures only the former.

### Path Forward

**Evidence:** `future_work.md`

The immediate roadmap addresses three gaps:
1. **Expert validation:** Real human annotations to establish what LDRI corresponds to
2. **Cross-novel replication:** At least 5 full novels to test generalizability
3. **Cross-language evaluation:** At least one non-English novel

The long-term roadmap addresses the architectural limitations:
1. **Redesigned memory** with semantic chunking and sliding-window overlap
2. **New resonance formulation** that produces meaningful cross-window values
3. **Alternative confidence modulation** that distributes signal across components
4. **Adaptive long-context retrieval** for genuine narrative memory
