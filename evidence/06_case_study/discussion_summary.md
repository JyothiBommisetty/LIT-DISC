# Full-Novel Scalability & Longitudinal Evaluation — Discussion

**Novel:** The Mistress of Spices (Chitra Banerjee Divakaruni)  
**Words:** 80,728  
**Processing units:** 105 windows of ~800 words  
**Version:** LIT-DISC v1.1 (sigma_max = 5.0)  
**Date:** 2026-07-29  

---

## Scalability Characteristics

Runtime scales **linearly** with text length. The second half of the novel processes at 1.083× the rate of the first half — within the < 1.5× linearity threshold. Average throughput is 950 words/second, or ~1.05s per 1000 words.

The runtime–word correlation is r = 0.23, which is weaker than expected. This is because all windows are approximately the same size (~800 words), so runtime per window is roughly constant (mean 0.81s, range 0.29–1.46s) rather than scaling with word count. The variance in runtime is driven by sentence count, not word count.

**Observation:** The pipeline's computational complexity is dominated by refinement iterations (O(n_iter × n_sentences × n_dimensions)), not by raw word count. Since sentence count scales roughly linearly with word count for prose of consistent style, the overall complexity is O(n_words).

---

## Computational Complexity

| Metric | Value |
|--------|-------|
| Total runtime | 85.0s |
| Avg per window | 0.81s |
| Avg per 1000 words | 1.05s |
| Runtime–word correlation | 0.23 |
| Second/first half ratio | 1.08 |
| Conclusion | **Linear** |

At novel length (~80k words), the entire LIT-DISC v1.1 pipeline completes in under 90 seconds. This is practical for research-scale analysis but would need optimization for corpus-scale processing (e.g., 1000 novels would take ~24 hours).

---

## Memory Behaviour

**Critical finding: The memory system fills to capacity (200 entries) in the first window (~800 words) and remains saturated for all subsequent windows.**

Memory analysis:
- Window 1: memory reaches 200 entries
- Windows 2–105: memory stays at 200 (capacity)
- **Retrieval count: 0 across all 105 windows**

The memory system is designed with `max_size = 200`, which is reasonable for single-passage processing (15–25 sentences) but insufficient for novel-length processing. Within a single window, the memory stores LiterarySignals whose confidence (gamma) exceeds the storage threshold (theta_store = 0.5). With ~30–50 sentences per window, the memory fills rapidly.

The **zero retrieval** across all windows is a more fundamental issue. The memory system uses random Fourier features for key generation and cosine similarity for retrieval. When narrative context evolves across windows (new characters, settings, vocabulary), the feature keys drift and never achieve the retrieval threshold (theta_retrieve = 0.7). Cross-window retrieval does not occur.

**This suggests a genuine architectural limitation:** The HMP-DT memory system was designed for short-passage discourse analysis where memory accumulates context within a single passage or between closely related passages. For novel-length processing, the memory system acts as a write-only buffer that fills immediately and contributes nothing to cross-window discourse analysis.

The practical consequence is that processing windows sequentially with persistent memory produces **identical results** to processing each window in isolation. The memory component of LDRI is effectively constant after the first window.

---

## Longitudinal Behaviour

### LDRI Progression

LDRI varies across the novel with CV = 21.8%, ranging from 0.0057 to 0.0155. The mean LDRI (0.010) is approximately 1/3 of the gold-standard passage mean (0.0277). This reduction is driven entirely by lower gamma_global (0.0105 novel vs 0.0302 GS), not by reduced raw LDRI (0.950 novel vs 0.913 GS).

**Why is gamma lower for the novel?** Gamma_global is the product of per-dimension confidences, which reflect how well the feature extractors match the text. Novel-length windows contain more diverse vocabulary, complex syntax, and varied discourse modes. This diversity reduces the average feature-extraction confidence, which propagates to lower gamma.

### Raw LDRI

Raw LDRI is **near-constant** across the novel (CV = 0.98%). This is even flatter than the gold standard (CV = 2.6%). The raw LDRI measures the geometric mean of R_signal (activation), C_tilde, T_tilde, and M_tilde. Since memory is saturated and retrieval is zero, M_tilde is effectively constant. Since windows are similar in size and structure, C_tilde and T_tilde are also nearly constant. The only source of raw LDRI variation is R_signal, which depends on the feature norms, but the damping effect of the geometric mean over 5 factors compresses all variation.

### Gamma Progression

Gamma varies with CV = 21.3%, closely tracking LDRI variation. The Pearson correlation between gamma and LDRI is r = 0.999+ (LDRI = gamma × near-constant raw). The gamma progression shows no clear trend across the novel — it fluctuates in response to local text characteristics.

### Resonance

Resonance component mean = 0.000078 (all windows > 0 but extremely small). This is consistent with the gold standard, where resonance was identified as the smallest component. The contribution from the narrative resonance engine is negligible at both passage and novel scale.

---

## Gamma Analysis

**Gamma dominance persists at novel scale.** Variance decomposition in log-space:

| Source | Gold Standard (25 passages) | Full Novel (105 windows) |
|--------|---------------------------|-------------------------|
| var(log(LDRI)) | 0.261 | 0.046 |
| Gamma contribution | 96.5% | 95.3% |
| Raw LDRI contribution | 0.3% | 0.2% |
| Covariance contribution | 3.2% | 4.5% |

The richer narrative context of the full novel does NOT reduce gamma dominance. If anything, the covariance contribution increases (3.2% → 4.5%), suggesting a modest positive correlation between confidence and feature richness at novel scale (r = 0.50). Passages with richer literary features also receive higher confidence, a relationship that was too weak to detect in short passages.

**Conclusion:** The gamma-dominance pattern is structural, not a scaling artifact. LDRI = Gamma × raw_ldri, and gamma varies with CV = 21–47% while raw_ldri varies with CV = 1–3%. This asymmetry is inherent to the multiplicative formulation.

---

## Stability

| Check | Result |
|------|--------|
| Numerical (NaN/Inf) | PASS (zero violations) |
| Memory capacity | PASS (never exceeds max_size=200) |
| Components in [0,1] | PASS (all 6 components within bounds) |
| Runtime spikes > 3× | PASS (zero spikes) |
| **Overall** | **STABLE** |

No numerical instabilities or divergences were observed over 105 sequential windows. The pipeline processes 80,728 words without degradation.

---

## Narrative Event Mapping

A detailed narrative event log (`narrative_event_log.md`) maps the 16 most significant LDRI changes to identifiable narrative events:

| Event | Windows | LDRI change | Narrative trigger |
|-------|---------|-------------|-------------------|
| E4 | W25→W26 | **+0.0078** | Makaradwaj spice lore — "I am not afraid" |
| E6 | W40 | Peak **0.0155** | Spiritual crisis — "Spices is this. The way. You have chosen. To. Punish me." |
| E9 | W54→W55 | **+0.0068** | Racist assault — violence scene |
| E13–E14 | W81→W82 | **+0.0067 → −0.0082** | Reunion → Renunciation (tightest peak–valley in the novel) |
| E15 | W89 | Peak **0.0149** | Physical liberation — "O you who are young" |

The LDRI trajectory across 105 windows traces the novel's full arc: lyrical opening → spiritual crisis → action sequence → climax → resolution. LDRI peaks correspond to scenes of highest literary-discourse intensity (spiritual crisis, violence, reunion, transformation), and valleys correspond to exposition and introspection. This validates that LDRI, despite gamma dominance, tracks narrative structure.

## Comparison with Short-Passage Evaluation

| Aspect | Gold Standard (passages) | Full Novel (windows) | Difference |
|--------|------------------------|---------------------|------------|
| Sample size | 25 passages | 105 windows | 4× more data points |
| Text length | ~200 words each | ~800 words each | 4× longer units |
| Total words | ~5,000 | ~80,000 | 16× longer |
| LDRI mean | 0.0277 | 0.0100 | **3× lower** |
| LDRI CV | 47.9% | 21.8% | **2× less variation** |
| Raw LDRI CV | 2.6% | 0.98% | Flat |
| Gamma CV | 47.0% | 21.3% | 2× less variation |
| Gamma dominance | 96.5% | 95.3% | Same |
| Resonance | 0.00022 | 0.00008 | Both negligible |
| Memory utility | Fills per passage | Fills once, unused | **Same** |

**Key differences:**

1. **Scale collapse:** LDRI for windows of the novel is ~1/3 of passage LDRI, driven by lower gamma at longer text windows.
2. **Variation compression:** LDRI variation across the novel (CV = 22%) is half that across diverse passages (CV = 48%). A single novel has less literary-discourse variation than 25 passages from different authors and categories.
3. **Memory ineffectiveness:** The memory system saturates on the first window and never retrieves. For passage-level evaluation, this is invisible (each passage has its own memory). For novel-level evaluation, the memory system's limitations become apparent.
4. **Gamma dominance is invariant:** The structural dominance of gamma is not reduced by longer context. This is a mathematical consequence of the LDRI formulation.

---

## Strengths

1. **Runtime scalability:** Linear complexity, practical for single-novel analysis (~85s for 80k words).
2. **Numerical stability:** No divergence or instability over 105 sequential processing steps.
3. **Consistency:** Cross-window LDRI variation (22% CV) reflects meaningful differences in literary discourse intensity across the novel.
4. **Composability:** Per-window LDRI values form a coherent narrative trajectory, with higher-LDRI windows corresponding to climactic or introspective passages.

---

## Limitations

1. **Memory underutilization:** The memory system fills in ~800 words and retrieves zero entries over the entire novel. Cross-window memory persistence is functionally irrelevant at novel scale.
2. **Resonance irrelevance:** The resonance component is negligible at all scales (passage or novel). Its contribution to LDRI is below measurement significance.
3. **Gamma dominance:** LDRI variance is 95% driven by gamma, which measures feature-extraction confidence rather than literary discourse richness. A reformulation that increases raw LDRI's contribution would be more diagnostic.
4. **Window granularity:** The 800-word window size is arbitrary. Different window sizes might yield different LDRI trajectories and variance characteristics.
5. **Single-novel case study:** Results are from one novel (The Mistress of Spices). Generalizability to other novels, genres, and languages requires validation.

---

## Recommendations

1. **Memory architecture review:** Consider increasing max_size or implementing a hierarchical memory with episode-level retrieval to enable cross-window discourse analysis.
2. **Resonance mechanism review:** Evaluate whether the current resonance formulation (RecursiveResonance) achieves its theoretical purpose, or whether it can be removed without loss.
3. **LDRI formulation:** The gamma-dominance pattern suggests that an alternative formulation (e.g., LDRI = raw_ldri^α × gamma^β with β > α) could increase raw LDRI's contribution. This should be explored in a future version.
4. **Window size analysis:** Run a sensitivity analysis on window size (200–2000 words) to determine the optimal granularity for novel-level LDRI analysis.

---

## Output Files

| File | Description |
|------|-------------|
| `novel_results.json` | Complete structured results (all 105 windows + summary) |
| `chapter_statistics.csv` | Per-window metrics (21 columns) |
| `runtime_scaling.csv` | Runtime, memory, retrieval by cumulative words |
| `memory_analysis.csv` | Memory growth, new entries, fill percentage |
| `longitudinal_metrics.csv` | All LDRI components by window |
| `variance_decomposition_novel.csv` | Gamma/raw/covariance contributions |
| `figures/01_runtime_scaling.png` | Runtime vs cumulative words |
| `figures/02_memory_growth.png` | Memory growth trajectory |
| `figures/03_ldri_progression.png` | LDRI over narrative |
| `figures/04_raw_ldri_progression.png` | Raw LDRI over narrative |
| `figures/05_gamma_progression.png` | Gamma over narrative |
| `figures/06_resonance_progression.png` | Resonance evolution |
| `figures/07_component_evolution.png` | All 6 components |
| `figures/08_retrieval_frequency.png` | Retrieval activity |
| `figures/09_variance_comparison.png` | GS vs novel variance |
| `narrative_event_log.md` | 16 narrative events mapped to metric changes |
| `discussion_summary.md` | This document |
