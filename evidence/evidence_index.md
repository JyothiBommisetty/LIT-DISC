# Evidence Index — LIT-DISC v1.1

## 01_framework — Architectural and Design Documentation

| File | Description |
|------|-------------|
| `framework_summary.md` | Overview of the HMP-DT framework: 5-level hierarchy, 6 dimensions, full pipeline architecture |

## 02_validation — Gold Standard and Validation Results

| File | Description |
|------|-------------|
| `sigma5_validation.json` | Raw per-passage LDRI and component values for all 25 gold-standard passages (sigma=5.0) |
| `analysis_summary.json` | Structured JSON summary of all gold-standard analysis metrics |
| `activation_design_review.md` | Activation function analysis: sigma=0.1 saturation (99%), sigma=5.0 saturation (4%), sweep from 0.05–10.0 |
| `gold_standard_summary.md` | Summary of gold-standard composition: 25 passages, 25 authors, 5 categories |
| `validation_findings.md` | Key findings from the gold standard evaluation and cross-comparisons |
| `regression_report.md` | v1.0 → v1.1 regression analysis: ranking preserved (ρ=0.98), CV improves 8.5× |
| `descriptive_statistics.csv` | Per-passage LDRI, raw LDRI, gamma, runtime, word count, component means |
| `category_statistics.csv` | LDRI means by narrative category (SNC=0.040, MNC=0.019) |
| `author_statistics.csv` | Per-author LDRI, raw, gamma (Woolf=0.055 highest, Mantel=0.009 lowest) |
| `feature_statistics.csv` | Per-feature statistics: semantic (3.50), narrative (3.11), evolution (2.97), interaction (2.65), symbolic (2.59), emotion (2.55) |

## 03_baselines — Baseline Comparisons

| File | Description |
|------|-------------|
| `baseline_summary.md` | Analysis of all baseline conditions: Random Gamma (ρ=0.14), Constant Features (r>0.99), Dummy (std=0) |
| `baseline_summary.json` | Structured JSON summary of baseline comparisons |
| `baseline_comparison.csv` | Per-passage LDRI for all 4 conditions (full, constant features, no gamma, dummy) |
| `null_model_summary.md` | Null model baseline: shuffled text (ρ=0.96 vs baseline), random embeddings (ρ=0.995) — gamma is sole carrier |

## 04_ablations — Ablation Studies

| File | Description |
|------|-------------|
| `ablation_summary.md` | Stage 3 ablation results: Without Uncertainty (Δ=+0.885, LARGE), others Δ<0.002 |
| `ablation_summary.json` | Structured JSON summary of all ablation results |
| `ablation_results.csv` | Per-passage LDRI for 5 ablation conditions (no memory, resonance, coupling, uncertainty, narrative) |

## 05_scalability — Full-Novel Scalability

| File | Description |
|------|-------------|
| `scalability_summary.md` | Scalability analysis: linear runtime (85s/80k words), gamma dominance unchanged (95.3%→96.5%) |
| `novel_results.json` | Complete per-window LDRI, components, gamma, runtime, word count for all 105 windows |
| `threshold_sweep_summary.md` | Memory retrieval threshold sweep (θ=0.1–0.9): within-window retrieval universal, problem is architectural |
| `runtime_scaling.csv` | Cumulative words vs runtime for every window |
| `memory_analysis.csv` | Per-window memory entries, peak, retrievals, capacity status |
| `longitudinal_metrics.csv` | Per-window LDRI, gamma, raw_ldri, and smoothed versions |
| `chapter_statistics.csv` | Aggregated LDRI statistics per chapter (12 chapters) |
| `variance_decomposition_novel.csv` | Variance decomposition: gamma (95.3%), raw (0.2%), covariance (4.5%) |

## 06_case_study — Literary Case Study

| File | Description |
|------|-------------|
| `mistress_case_study.md` | Literary analysis of The Mistress of Spices: narrative summary and LDRI/memory findings |
| `narrative_event_log.md` | 16 narrative events mapped to LDRI changes with window text excerpts |
| `discussion_summary.md` | Evaluation discussion: gamma dominance, memory limitations, feature contributions |

## 07_statistics — Descriptive and Inferential Statistics

| File | Description |
|------|-------------|
| `statistics_summary.md` | Comprehensive statistics: means, CVs, variance decomposition, correlations, outlier analysis |

## 08_figures — Generated Figures

| File | Description |
|------|-------------|
| `figure_catalog.md` | Catalog of all 9 figures with suggested captions for paper sections |
| `runtime/runtime_scaling.png` | Runtime vs cumulative words — linear scaling |
| `memory/memory_growth.png` | Memory entries over cumulative words — fills to 200 in first window |
| `memory/retrieval_frequency.png` | Retrievals per window — zero across all 105 windows |
| `ldri/ldri_progression.png` | LDRI over narrative progression — traces narrative arc |
| `ldri/raw_ldri_progression.png` | Raw LDRI — near-constant (CV=1.0%) |
| `gamma/gamma_progression.png` | Gamma over narrative — drives all LDRI variation |
| `components/component_evolution.png` | All 6 LDRI components over narrative |
| `components/resonance_progression.png` | Resonance alone — near-zero throughout |
| `events/variance_comparison.png` | Variance contribution comparison: GS vs novel |

## 09_reproducibility — Reproducibility

| File | Description |
|------|-------------|
| `reproducibility.md` | Full environment, dependencies, execution order, test results, and determinism confirmation |

## 10_claims — Claims Traceability

| File | Description |
|------|-------------|
| `claims_and_evidence.md` | 54 claims (C1–C54) mapped to evidence files and assessed for support level |
| `unsupported_claims.md` | 5 unsupported claims: memory retrieval (U1), resonance (U2), expert correlation (U3), raw LDRI scaling (U4), gamma dominance reduction (U5) |

---

## Summary

| Subdirectory | Summary Documents | Raw Data Files | Figures |
|-------------|-----------------|---------------|---------|
| 01_framework | 1 | 0 | 0 |
| 02_validation | 5 | 6 | 0 |
| 03_baselines | 2 | 2 | 0 |
| 04_ablations | 1 | 2 | 0 |
| 05_scalability | 2 | 6 | 0 |
| 06_case_study | 3 | 0 | 0 |
| 07_statistics | 1 | 0 | 0 |
| 08_figures | 1 | 0 | 9 |
| 09_reproducibility | 1 | 0 | 0 |
| 10_claims | 2 | 0 | 0 |
| **Total** | **19** | **16** | **9** |
