# Validation Findings — LIT-DISC v1.1

## Activation Redesign

**Problem:** At sigma_max = 0.1 (v1.0), Gaussian RBF activations saturated at 1.0 for 99% of 150 activations (25 passages × 6 dimensions). Feature norms ||H_d|| ≈ 10–14 were 100–140× the kernel width.

**Fix:** sigma_max = 5.0 (v1.1). Single-line config change in `src/config.py:59`.

**Result:** Saturation drops from 99% to 4%. Activations span 0.54–0.99. All 6 dimensions show non-trivial CV (0.019–0.260). Only dimensions with ||H_d|| > 13 (6/150) remain saturated.

**Evidence:** `activation_design_review.md`, `sigma5_validation.json`, `regression_report.md`

## Descriptive Statistics (Gold Standard, 25 passages)

| Metric | Mean | Std | CV |
|--------|------|-----|----|
| LDRI | 0.0277 | 0.0133 | 47.9% |
| Raw LDRI | 0.9128 | 0.0240 | 2.6% |
| Gamma | 0.0302 | 0.0142 | 47.0% |
| Runtime | 0.21s | 0.06s | — |

Dominant component: Uncertainty (mean=0.885). LDRI variation is driven by gamma_global (CV=47%) not raw_ldri (CV=2.6%).

## Ranking Stability

- **Category ranking:** SNC > MLC > NWC > SSC > MNC (bottom two swapped vs v1.0)
- **Author ranking:** Spearman ρ = 0.98 vs v1.0
- Top 5 authors stable: Virginia Woolf #1, Hilary Mantel #25

## Key Observations

1. Raw LDRI CV improved from 0.3% to 2.6% (8.5x) after sigma correction
2. Gamma dominance reduced from 100.3% to 96.5% of log(LDRI) variance
3. All 66 equation/stability/determinism/edge-case tests pass
4. Runtime unchanged by sigma change (sigma is in LDRI calc only, not HMP-DT propagation)
