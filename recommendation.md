# Shadow Evaluation Recommendation — sigma = 5.0

**Date:** 2026-07-29  
**Baseline:** sigma_max = 0.1 (v1.0.0)  
**Candidate:** sigma_max = 5.0  
**Data:** 25 gold-standard passages  

---

## How the shadow evaluation was performed

A `LITDISCPipeline` was instantiated with `LITDISCConfig(ldri=replace(LITDISCConfig().ldri, sigma_max=5.0))`. No source files were modified — the override is purely in the config object passed at construction time. All 25 passages were re-processed identically to the gold-standard evaluation.

---

## Answers to the five diagnostic questions

### Q1: Does saturation disappear?

| Regime | Saturated activations (> 0.999) |
|--------|-------------------------------|
| sigma = 0.1 | **99%** (149/150) |
| sigma = 5.0 | **4%** (6/150) |

**Partially.** Saturation drops from 99% to 4%. The 6 remaining saturated activations occur in passages with the largest feature norms (||H_d|| > ~12.5). With sigma=5.0, the RBF reaches a_d = 0.999 at ||H|| = 5.0 × √(2·ln(1000)) ≈ 13.1. Norms exceeding this threshold still saturate.

**Recommendation:** sigma=5.0 is a significant improvement (99%→4%) but not a complete cure. The 4% residual saturation affects only the strongest-signal dimensions in the strongest-signal passages.

### Q2: Does raw LDRI become discriminative?

| Metric | sigma = 0.1 | sigma = 5.0 | Change |
|--------|-------------|-------------|--------|
| raw_ldri CV | 0.0031 (0.3%) | **0.0263 (2.6%)** | **8.5x increase** |
| raw_ldri mean | 0.956 | 0.913 | -4.5% |

**Yes.** The coefficient of variation of raw_ldri increases from 0.3% to 2.6% — an 8.5-fold improvement. While 2.6% is still modest compared to gamma's 47%, raw_ldri now carries measurable discriminative information that was previously absent.

### Q3: Does gamma cease to dominate?

| Contribution to var(log(LDRI)) | sigma = 0.1 | sigma = 5.0 |
|-------------------------------|-------------|--------------|
| log(gamma_global) | **100.3%** | **96.5%** |
| log(raw_ldri) | 0.0% | 0.3% |
| covariance | -0.3% | +3.2% |

**No.** Gamma still dominates 96.5% of LDRI variance. The raw_ldri contribution increased from ~0% to 0.3% — an improvement but still two orders of magnitude below gamma's contribution. The structural asymmetry (LDRI = Gamma × near-constant) persists because gamma has 47% CV while raw_ldri has only 2.6% CV.

### Q4: Are category rankings more meaningful?

| Category | Old LDRI (σ=0.1) | New LDRI (σ=5.0) | Rank (old) | Rank (new) |
|----------|-----------------|-----------------|-----------|-----------|
| SNC | 0.0414 | 0.0395 | 1 | 1 |
| MLC | 0.0336 | 0.0326 | 2 | 2 |
| NWC | 0.0278 | 0.0270 | 3 | 3 |
| MNC | 0.0219 | 0.0204 | 4 | 5 |
| SSC | 0.0203 | 0.0207 | 5 | 4 |

The top three categories (SNC > MLC > NWC) are stable. The bottom two (MNC and SSC) swap. The spread between highest and lowest is nearly identical (0.0133 vs 0.0134). The change is moderate and not clearly more meaningful — the ranking perturbation is within the noise floor of a 5-passage-per-category sample.

### Q5: Are there any regressions?

| Aspect | sigma = 0.1 | sigma = 5.0 | Assessment |
|--------|-------------|-------------|------------|
| Avg runtime | 0.86s | 0.38s | Faster (sigma changes LDRI calc only, not propagation) |
| Component: intrinsic | 0.288 (mean) | **0.039 (mean)** | **Large change** — intrinsic drops 86% because it depends directly on R_signal |
| Component: interaction | 0.00286 | 0.00276 | Minimal change |
| Component: temporal | 0.00421 | 0.00419 | Minimal change |
| Component: resonance | 0.000225 | 0.000223 | Minimal change |
| Component: memory | 0.00419 | 0.00414 | Minimal change |
| Component: uncertainty | 0.927 | 0.885 | Moderate — drops because raw_ldri decreased |
| Author ranking ρ | — | ρ = 0.982 | Near-perfect preservation |

**No regressions.** The only large component change is in intrinsic (which should change — it measures activation richness, and sigma=5.0 makes activations non-trivial). All other components, runtimes, and rankings are preserved. The author ranking correlation is ρ = 0.98 between the two conditions.

---

## Overall Assessment

sigma = 5.0 is a **safe and beneficial change**:

| Criterion | Verdict |
|-----------|---------|
| Mathematical consistency | Preserved (same RBF kernel, different σ) |
| Saturation reduction | 99% → 4% **(large improvement)** |
| Raw LDRI discriminability | CV 0.3% → 2.6% **(8.5x improvement)** |
| Ranking stability | Author ρ = 0.98, categories mostly stable |
| Runtime impact | Neutral (sigma is in LDRI calc only) |
| Regression risk | Low (only intrinsic component changes materially) |

**The improvement is real but partial.** Sigma=5.0 eliminates near-total saturation (99%→4%) and gives raw_ldri measurable variance (CV 2.6%). However, raw_ldri still contributes only 0.3% of LDRI log-variance versus gamma's 96.5%. The gamma-dominance problem is structural — it comes from the multiplicative LDRI = Gamma × raw form, not from sigma alone.

### Comparison with sigma=10.0

From the design review sweep, sigma=10.0 would achieve 0% saturation and a raw_ldri CV of ~5-8%, roughly halving gamma's dominance. The trade-off is that activations would be more compressed toward zero for most passages (mean a ≈ 0.46 at σ=10 vs 0.90 at σ=5).

### Final recommendation

**Proceed with sigma = 5.0.** It is a safe, one-line config change that provides material improvement without regressions. If further discriminability is needed, consider sigma = 10.0 or an adaptive sigma strategy after additional validation on longer texts (novels, where the signal-to-noise ratio is higher).

---

## Files generated

- `comparison_before_after.csv` — 40-row comparison table of all metrics
- `sigma5_validation.json` — per-passage results for all 25 passages
- `recommendation.md` — this document
