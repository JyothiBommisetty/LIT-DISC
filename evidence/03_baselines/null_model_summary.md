# Null Model Baseline Summary

**Date:** 2026-07-30
**Experiment:** `data/experiments/null_model/run_null_model.py`
**Data:** `data/experiments/null_model/null_model_results.json`

## Design

Two null conditions applied to all 25 gold-standard passages:

| Condition | Change | What It Tests |
|-----------|--------|---------------|
| Shuffled text | Paragraphs randomly reordered | Does LDRI measure narrative structure? |
| Random embeddings | All 8 embedding subspaces replaced with Gaussian noise (mean/variance matched to empirical distribution) | Does LDRI measure anything beyond gamma? |

## Results

### Summary Statistics

| Metric | Baseline | Shuffled | Random Embeddings |
|--------|----------|----------|-------------------|
| LDRI mean | 0.0277 | 0.0278 | 0.0302 |
| LDRI std | 0.0130 | 0.0136 | 0.0139 |
| LDRI CV | 46.9% | 48.8% | 46.0% |
| Gamma mean | 0.0302 | 0.0303 | 0.0302 |
| Raw mean | 0.9128 | 0.9135 | **0.9986** |
| Raw std | 0.0235 | 0.0245 | 0.0006 |

### Ranking Preservation

| Comparison | LDRI Spearman ρ | Category ρ |
|-----------|-----------------|------------|
| Baseline vs Shuffled | **0.960** | 1.000 |
| Baseline vs Random Embeddings | **0.995** | 1.000 |

## Interpretation

### Finding 1: Shuffling barely matters (ρ=0.96)

LDRI is nearly insensitive to paragraph order. The ranking of 25 passages by LDRI is almost identical whether paragraphs are in original or random order. This means LDRI measures **per-passage feature statistics**, not narrative structure or discourse flow.

The slight CV increase (46.9% → 48.8%) suggests that some passages are more affected by shuffling than others, but the effect is marginal. Category ordering is perfectly preserved (ρ_cat = 1.00).

### Finding 2: Random embeddings produce the same ranking (ρ=0.995)

When all embedding subspaces are replaced with matched Gaussian noise, the LDRI ranking is nearly identical to the baseline (ρ=0.995). This is because:

1. Raw LDRI saturates near 1.0 (0.9986) with random embeddings (large random norms → high activations)
2. LDRI = Gamma × Raw ≈ Gamma × 1.0 ≈ Gamma
3. Gamma is computed from feature extraction (Eq. 21), not from embeddings

**Result: The embeddings contribute nothing to LDRI's discriminative signal.** The 0.3% raw LDRI variation is the only information from the entire embedding construction pipeline.

### Finding 3: Gamma is the sole carrier

Both null models confirm what variance decomposition already showed: gamma carries 96.5% of LDRI's signal. The null model extends this by showing that even destroying the embeddings entirely leaves the ranking unchanged.

## Implications for Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| LDRI measures discourse richness | **UNSUPPORTED** | LDRI unchanged by paragraph shuffling (ρ=0.96) |
| Embedding construction adds value | **REFUTED** | Random embeddings preserve ranking (ρ=0.995) |
| Gamma dominance is structural | **CONFIRMED** | Null model proves gamma is the sole carrier |
| LDRI is not reducible to readability | CONFIRMED (from baselines) | ρ ≤ 0.53 with readability metrics |
