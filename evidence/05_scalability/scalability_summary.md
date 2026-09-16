# Scalability Summary — The Mistress of Spices

**Novel:** The Mistress of Spices (Chitra Banerjee Divakaruni)  
**Windows:** 105 of ~800 words  
**Total words:** 80,728  
**Total runtime:** 85.0s  
**Frozen config:** LIT-DISC v1.1 (sigma_max = 5.0)

---

## Runtime Scaling

| Metric | Value |
|--------|-------|
| Total runtime | 85.0s |
| Avg per window | 0.81s |
| Avg per 1k words | 1.05s |
| Runtime–word correlation | r = 0.23 |
| Second/first half ratio | 1.08× |
| **Conclusion** | **LINEAR** |

Runtime scales linearly because: (a) each window's computational cost is proportional to sentence count, (b) sentence count scales with word count, (c) no super-linear components. The weak correlation (r=0.23) is due to near-constant window size (~800 words).

## Memory Scaling

| Metric | Value |
|--------|-------|
| Peak memory entries | 200/200 (capacity) |
| Windows at capacity | 105/105 |
| Cross-window retrieval | **0** across all windows |
| First window fill rate | 200 entries in ~800 words |

Memory fills to capacity in the first window and never retrieves a single entry from previous windows. The RFF key-generation scheme produces keys that drift across narrative windows, preventing cosine-similarity matches above the retrieval threshold (θ=0.7).

## Longitudinal LDRI

| Metric | Value |
|--------|-------|
| LDRI mean | 0.0100 |
| LDRI std | 0.0022 |
| LDRI CV | 21.8% |
| LDRI range | [0.0057, 0.0155] |
| Raw LDRI mean | 0.9503 |
| Raw LDRI CV | 0.98% |
| Gamma mean | 0.0105 |
| Gamma CV | 21.3% |

## Variance Decomposition (log-space)

| Source | Gold Standard (25 passages) | Full Novel (105 windows) |
|--------|---------------------------|-------------------------|
| Gamma contribution | 96.5% | 95.3% |
| Raw LDRI contribution | 0.3% | 0.2% |
| Covariance contribution | 3.2% | 4.5% |

Gamma dominance is essentially unchanged at novel scale. The covariance term increases (3.2% → 4.5%), suggesting a modest positive correlation between gamma and raw LDRI at novel length (r=0.50).

## Stability

| Check | Result |
|-------|--------|
| Numerical (NaN/Inf) | PASS |
| Memory capacity | PASS (never exceeds 200) |
| Components in [0,1] | PASS |
| Runtime spikes | PASS (max 3× mean) |
| **Overall** | **STABLE** |
