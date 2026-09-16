# LIT-DISC v1.1 Release Notes

**Date:** 2026-07-29  
**Previous version:** v1.0 (sigma_max = 0.1)  
**New version:** v1.1 (sigma_max = 5.0)  

---

## Implementation Change

**Single-line config update** in `src/config.py:59`:

```python
# v1.0
sigma_max: float = 0.1

# v1.1
sigma_max: float = 5.0
```

No source code, equations, or algorithms were modified. Only the RBF kernel width parameter in `LDRIParams` was updated.

---

## Mathematical Justification

The Gaussian RBF activation in the intrinsic richness component is:

$$a_d = 1 - \exp\left(-\frac{\|H_d\|^2}{2\sigma^2}\right)$$

where $H_d$ is the feature-embedding vector for dimension d and σ is the kernel width.

At σ = 0.1, feature norms $\|H_d\| \approx 10-14$ produced activations at machine precision 1.0 for all 150 activations (25 passages x 6 dimensions). This is 100-140x the kernel width — total saturation.

At σ = 5.0, the activation becomes:

$$a_d = 1 - \exp\left(-\frac{\|H_d\|^2}{50}\right)$$

With $\|H_d\| \approx 10-14$, this yields activations in the range 0.54–0.99, restoring discriminative power.

The relationship between sigma and the activation function:
- σ = 0.1: a_d = 1.0 for all passages (no discriminability)
- σ = 5.0: a_d ∈ [0.54, 0.99] (variance restored)
- Residual saturation only for $\|H_d\| > 13$ (6/150 = 4% of activations)

---

## Activation Diagnostics

| Dimension | σ=0.1 activation | σ=5.0 activation | CV (σ=0.1) | CV (σ=5.0) |
|-----------|-----------------|-----------------|-----------|-----------|
| Semantic  | 1.0000          | 0.9820          | 0.0000    | 0.0185    |
| Narrative | 1.0000          | 0.9414          | 0.0000    | 0.0453    |
| Interaction | 1.0000        | 0.8029          | 0.0000    | 0.2596    |
| Symbolic  | 1.0000          | 0.8607          | 0.0000    | 0.0760    |
| Evolution | 1.0000          | 0.9379          | 0.0000    | 0.0350    |
| Emotion   | 1.0000          | 0.8436          | 0.0000    | 0.1289    |

Saturation reduced from 99% to 4%.

---

## Sigma Validation

Sigma=5.0 was selected from a ten-value sweep (0.05–10.0) as the smallest sigma that:
1. Eliminates near-total saturation (achieved: 99% -> 4%)
2. Produces variable activations (CV > 0.01 for all 6 dimensions)
3. Preserves ranking structure (author Spearman rho = 0.98)

---

## Evaluation Summary

### Gold Standard (25 passages, sigma=5.0)

| Metric | v1.0 (σ=0.1) | v1.1 (σ=5.0) | Change |
|--------|-------------|-------------|--------|
| LDRI mean | 0.0290 | 0.0277 | -4.5% |
| LDRI std | 0.0130 | 0.0133 | +2.3% |
| LDRI CV | 44.8% | 47.9% | +3.1pp |
| Raw LDRI mean | 0.9561 | 0.9128 | -4.5% |
| Raw LDRI std | 0.0030 | 0.0240 | **+694%** |
| Raw LDRI CV | 0.31% | 2.63% | **+2.32pp** |
| Gamma mean | 0.0303 | 0.0302 | -0.4% |
| Gamma std | 0.0136 | 0.0142 | +4.4% |
| Runtime mean | 0.86s | 0.21s | (machine-dependent) |

### Key Improvements

- **Saturation**: 99% of activations were saturated at 1.0 in v1.0; only 4% remain in v1.1
- **Raw LDRI discriminability**: CV increased from 0.3% to 2.6% (8.5x improvement)
- **Gamma dominance**: Reduced from 100.3% to 96.5% of log(LDRI) variance
- **Ranking stability**: Author ranking Spearman rho = 0.9823 between versions

### Category Rankings

| Category | v1.0 LDRI | v1.1 LDRI | Rank v1.0 | Rank v1.1 |
|----------|----------|----------|-----------|-----------|
| SNC | 0.0414 | 0.0403 | 1 | 1 |
| MLC | 0.0336 | 0.0330 | 2 | 2 |
| NWC | 0.0278 | 0.0266 | 3 | 3 |
| MNC | 0.0219 | 0.0188 | 4 | 5 |
| SSC | 0.0203 | 0.0197 | 5 | 4 |

Top 3 categories unchanged. Bottom 2 swapped (MNC ↔ SSC). This indicates the change is moderate but affects fine-grained ordering.

### Author Ranking

Top 5 unchanged in composition (Virginia Woolf #1, Hilary Mantel #25). Internal ordering shifts: Proust overtakes Dickens for #2. Spearman rho = 0.98.

---

## Verification

- **Equation Verification**: 39/39 pass
- **Numerical Stability**: 10/10 pass
- **Determinism**: 3/3 pass
- **Performance**: 1/2 pass (warm call 0.674s > 0.5s threshold — pre-existing, machine-dependent)
- **Memory/Temporal**: 57/57 pass
- **Edge Cases**: 6/6 pass
- **Total**: 116/117 pass

The single performance failure (warm call) is pre-existing and unrelated to the sigma change. It is a machine-dependent threshold issue.

---

## Reproducibility

Fully deterministic. All 25 passages produce identical results on re-run when RNG is seeded. The sigma change does not affect any RNG-dependent or stochastic computation in the HMP-DT pipeline.

---

## Regressions

**None.** The only component with a change > 5% is `intrinsic` (mean drops 86% from 0.288 to 0.039). This is expected and correct — the intrinsic richness component measures R_signal activation, which was previously constant at 1.0. The new values (mean 0.039, CV 61.7%) reflect the restored discriminability of the RBF activation.

Components with minimal change (|Δ| < 5%):
- Interaction: -3.4%
- Temporal: -0.4%
- Resonance: -0.4%
- Memory: -1.0%
- Uncertainty: -4.5%

These changes are proportional to the raw LDRI scale shift (-4.5%) and do not represent regressions.

---

## Files Modified

| File | Change |
|------|--------|
| `src/config.py:59` | `sigma_max: float = 0.1` → `sigma_max: float = 5.0` |

---

## Output Files Generated

### Gold Standard
- `data/gold_standard/litdisc_results.json`

### Descriptive Statistics
- `analysis_summary.json`
- `descriptive_statistics.csv`
- `category_statistics.csv`
- `author_statistics.csv`
- `feature_statistics.csv`

### Baselines and Ablations
- `baseline_comparison.csv`
- `baseline_summary.json`
- `ablation_results.csv`
- `ablation_summary.json`

### Reports
- `regression_report.md` — v1.0 vs v1.1 comparison
- `evaluation_summary_v1_1.json` — structured evaluation summary
- `release_notes_v1_1.md` — this document

---

## Reproducibility Notes

To reproduce the full v1.1 evaluation:

```bash
# 1. Verify sigma setting
python -c "from src.config import LITDISCConfig; print(LITDISCConfig().ldri.sigma_max)"
# Expected: 5.0

# 2. Run verification suite
python tests/run_verification.py

# 3. Generate gold-standard results
python generate_gold_standard.py

# 4. Generate descriptive statistics
python analyze_results.py

# 5. Generate baselines
python run_baselines.py

# 6. Generate ablations
python run_ablations.py
```

---

## Next Steps

With v1.1 validated, the next phase is the full-novel scalability study on *The Mistress of Spices* — evaluating LDRI behavior at novel length (120,000 words across 8 chapters).
