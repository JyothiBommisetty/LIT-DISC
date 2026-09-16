# Claims and Evidence — LIT-DISC v1.1

This document maps every major claim in the paper to its supporting evidence files, figures, and statistics.

---

## Framework Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C1 | HMP-DT models literary text through 6 discourse dimensions | `01_framework/framework_summary.md` | Mermaid diagram | 6 dimensions, 47 equations | YES | High |
| C2 | LDRI is computed as Gamma × (R_signal² × C_tilde × T_tilde × M_tilde)^(1/5) | `01_framework/framework_summary.md` | — | Final LDRI = 0.0088–0.0551 | YES | High |
| C3 | The pipeline processes text through 6 sequential stages | `01_framework/framework_summary.md` | Architecture diagram | 25 passages in 5.2s | YES | High |
| C4 | The only v1.1 change is sigma_max: 0.1 → 5.0 | `01_framework/framework_summary.md`, `09_reproducibility/reproducibility.md` | — | 1 line changed in src/config.py:59 | YES | High |
| C5 | All parameters except sigma_max are frozen | `01_framework/framework_summary.md` | Parameter table | 20 parameters listed | YES | High |

## Activation Design Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C6 | At sigma=0.1, activations are 99% saturated | `02_validation/activation_design_review.md`, `02_validation/regression_report.md` | — | 149/150 activations > 0.999 | YES | High |
| C7 | At sigma=5.0, saturation drops to 4% | `02_validation/sigma5_validation.json`, `02_validation/regression_report.md` | — | 6/150 activations > 0.999 | YES | High |
| C8 | Feature norms ||H_d|| ≈ 10–14 are 100–140× sigma=0.1 | `02_validation/activation_design_review.md` | — | Mean ||H_d|| = 11.09 | YES | High |
| C9 | sigma=5.0 preserves ranking structure | `02_validation/regression_report.md` | — | Author Spearman ρ = 0.98 | YES | High |
| C10 | sigma=5.0 was selected from a 10-value sweep | `02_validation/activation_design_review.md` | — | Sweep range: 0.05–10.0 | YES | High |

## Validation Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C11 | All 39 core equations are verified | `09_reproducibility/reproducibility.md` | — | 39/39 pass | YES | High |
| C12 | The pipeline is numerically stable | `09_reproducibility/reproducibility.md` | — | 10/10 numerical tests pass | YES | High |
| C13 | Results are deterministic | `09_reproducibility/reproducibility.md` | — | 3/3 determinism tests pass | YES | High |
| C14 | Edge cases are handled | `09_reproducibility/reproducibility.md` | — | 6/6 edge-case tests pass | YES | High |
| C15 | All 66 verification tests pass (116/117 including sub-tests) | `09_reproducibility/reproducibility.md` | — | 116/117 pass (1 perf failure pre-existing) | YES | High |

## Gold Standard Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C16 | The gold standard comprises 25 passages from 25 authors across 5 categories | `02_validation/gold_standard_summary.md` | — | 25 passages, 25 authors, 5 categories | YES | High |
| C17 | Mean LDRI = 0.0277 (sigma=5.0) | `02_validation/analysis_summary.json`, `07_statistics/statistics_summary.md` | — | Mean=0.0277, Std=0.0133, CV=47.9% | YES | High |
| C18 | Raw LDRI is near-constant (CV=2.6%) | `07_statistics/statistics_summary.md` | — | Mean=0.9128, Std=0.0240 | YES | High |
| C19 | Uncertainty is the dominant component | `02_validation/descriptive_statistics.csv` | — | Mean=0.885, 88.5% of total | YES | High |
| C20 | Resonance is negligible | `02_validation/descriptive_statistics.csv` | — | Mean=0.0002, 0.02% of total | YES | High |
| C21 | Category ranking: SNC > MLC > NWC > SSC > MNC | `02_validation/category_statistics.csv` | — | SNC=0.0403, MNC=0.0188 | YES | High |
| C22 | Virginia Woolf has highest LDRI (0.055) | `02_validation/author_statistics.csv` | — | Woolf=0.055, Mantel=0.009 (lowest) | YES | High |

## Baseline Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C23 | Random Gamma destroys LDRI correlation (ρ=0.14) | `03_baselines/baseline_summary.md` | — | ρ=0.14, r=0.18 | YES | High |
| C24 | LDRI is not reducible to readability | `03_baselines/baseline_summary.md` | — | Max ρ=0.53 (avg sentence length) | YES | High |
| C25 | TTR correlation with LDRI is weak | `03_baselines/baseline_summary.md` | — | ρ=0.36 | YES | High |
| C26 | Constant Features baseline r>0.99 confirms gamma dominance | `03_baselines/baseline_summary.md` | — | r=0.998, ρ=0.995 | YES | High |

## Ablation Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C27 | Without Uncertainty: LDRI collapses to raw_ldri (Δ=+0.885) | `04_ablations/ablation_summary.md` | — | LDRI=0.913 vs baseline 0.028 | YES | High |
| C28 | Without Memory: |Δ| < 0.002 | `04_ablations/ablation_summary.md` | — | Mean |Δ|=0.0013 | YES | High |
| C29 | Without Resonance: |Δ| < 0.002 | `04_ablations/ablation_summary.md` | — | Mean |Δ|=0.0016 | YES | High |
| C30 | Without Coupling: |Δ| < 0.002 | `04_ablations/ablation_summary.md` | — | Mean |Δ|=0.0015 | YES | High |
| C31 | Without Narrative: LDRI increases (Δ=+0.012) | `04_ablations/ablation_summary.md` | — | LDRI=0.039 vs baseline 0.028 | YES | High |

## Scalability Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C32 | LIT-DISC runtime scales linearly | `05_scalability/scalability_summary.md` | `runtime/runtime_scaling.png` | 85s for 80k words, 1.05s/1k words, second/first half ratio=1.08 | YES | High |
| C33 | 105 windows of ~800 words process in 85s | `05_scalability/novel_results.json` | — | 84.8s total, 0.81s avg per window | YES | High |
| C34 | No numerical instability at novel scale | `05_scalability/scalability_summary.md` | — | All 105 windows stable, no NaNs | YES | High |

## Memory Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C35 | Memory fills to capacity in the first window | `05_scalability/memory_analysis.csv` | `memory/memory_growth.png` | 200/200 entries in first ~800 words | YES | High |
| C36 | Memory remains saturated for all 105 windows | `05_scalability/memory_analysis.csv` | — | 105/105 windows at capacity | YES | High |
| C37 | Cross-window retrieval is zero | `05_scalability/memory_analysis.csv` | `memory/retrieval_frequency.png` | 0 retrievals across 105 windows | YES | High |
| C38 | Memory contributes minimally to LDRI | `04_ablations/ablation_summary.md` | — | |Δ| = 0.0013 when memory removed | YES | High |

## Gamma Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C39 | Gamma dominates 96.5% of LDRI variance (gold standard) | `07_statistics/statistics_summary.md` | `events/variance_comparison.png` | 96.5% of var(log(LDRI)) | YES | High |
| C40 | Gamma dominates 95.3% at novel scale | `05_scalability/variance_decomposition_novel.csv` | — | 95.3% of var(log(LDRI)) | YES | High |
| C41 | Longer context does NOT reduce gamma dominance | `05_scalability/scalability_summary.md` | — | 96.5% → 95.3% (Δ = −1.2pp) | YES | High |
| C42 | Gamma carries the discriminative signal | `03_baselines/baseline_summary.md` | — | Random Gamma ρ=0.14 | YES | High |

## Longitudinal / Narrative Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C43 | LDRI traces the narrative arc | `06_case_study/narrative_event_log.md` | `ldri/ldri_progression.png` | 16 events mapped to LDRI changes | YES | Medium-High |
| C44 | LDRI peaks correspond to high-intensity scenes | `06_case_study/mistress_case_study.md` | — | Peak at spiritual crisis (W40, LDRI=0.0155) | YES | Medium-High |
| C45 | LDRI valleys correspond to exposition/introspection | `06_case_study/narrative_event_log.md` | — | Valley at renunciation (W82, LDRI=0.0057) | YES | Medium-High |
| C46 | The tightest peak→valley is W81→W82 (reunion→renunciation) | `06_case_study/narrative_event_log.md` | — | Δ = −0.0082 | YES | Medium-High |
| C47 | Raw LDRI is lower during introspective passages | `06_case_study/narrative_event_log.md` | — | 5 windows with z < −2.3 are all first-person reflection | YES | Medium |

## Resonance Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C48 | Resonance is negligible at all scales | `07_statistics/statistics_summary.md` | `components/resonance_progression.png` | Mean=0.0002 (GS), 0.00008 (novel) | YES | High |
| C49 | Removing resonance does not affect LDRI | `04_ablations/ablation_summary.md` | — | |Δ| = 0.0016 | YES | High |

## v1.0 → v1.1 Improvement Claims

| # | Claim | Evidence File(s) | Figure | Statistics | Supported | Confidence |
|---|-------|-----------------|--------|------------|-----------|------------|
| C50 | Raw LDRI CV increased 8.5× (0.3% → 2.6%) | `02_validation/regression_report.md` | — | CV 0.0031 → 0.0263 | YES | High |
| C51 | Saturation reduced from 99% to 4% | `02_validation/regression_report.md` | — | 149/150 → 6/150 activations saturated | YES | High |
| C52 | Author ranking preserved (ρ=0.98) | `02_validation/regression_report.md` | — | Spearman ρ=0.982 | YES | High |
| C53 | No regression in any component | `02_validation/regression_report.md` | — | Only intrinsic changes (intended) | YES | High |
| C54 | Runtime is unaffected by sigma change | `02_validation/validation_findings.md` | — | sigma only in LDRI calc (≪1% of runtime) | YES | High |

---

## Unsupported Claims

| # | Claim | Reason | Action Needed |
|---|-------|--------|---------------|
| U1 | Memory enables long-range narrative coherence | Not supported — zero cross-window retrieval observed | Reformulate memory system for novel-length retrieval |
| U2 | Resonance captures narrative development | Not supported — resonance is ~0 at all scales | Re-evaluate resonance mechanism or remove |
| U3 | LDRI correlates with expert annotations | Not supported — all expert scores are 4 (placeholder) | Real expert annotations needed |
| U4 | Raw LDRI becomes discriminative at novel scale | Not supported — novel CV=1.0% is even lower than GS CV=2.6% | Structural LDRI reformulation needed |
| U5 | Gamma dominance reduces with longer context | Not supported — 96.5% → 95.3% (unchanged) | LDRI = Gamma × raw formulation is inherently asymmetric |
