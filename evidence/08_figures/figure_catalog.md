# Figure Catalog — LIT-DISC v1.1

## Runtime

| File | Purpose | Paper Section | Caption Suggestion |
|------|---------|---------------|-------------------|
| `runtime/runtime_scaling.png` | Runtime vs cumulative words for 105-window novel processing | Scalability (§5) | "Runtime scales linearly with text length at ~1.05s per 1,000 words. The LIT-DISC v1.1 pipeline processed 80,728 words in 85 seconds across 105 windows." |

## Memory

| File | Purpose | Paper Section | Caption Suggestion |
|------|---------|---------------|-------------------|
| `memory/memory_growth.png` | Memory entries over cumulative words | Memory Analysis (§6) | "Memory fills to capacity (200 entries) in the first ~800 words and remains saturated for the entire novel. No cross-window retrieval was observed." |
| `memory/retrieval_frequency.png` | Retrieval count per window | Memory Analysis (§6) | "Retrieval count across 105 windows is zero. The RFF-based memory system does not retrieve across narratively distinct windows." |

## LDRI

| File | Purpose | Paper Section | Caption Suggestion |
|------|---------|---------------|-------------------|
| `ldri/ldri_progression.png` | LDRI over narrative progression | Longitudinal Behaviour (§7) | "LDRI across the novel traces the narrative arc: lyrical opening, spiritual crisis, violence, reunion, renunciation, and resolution." |
| `ldri/raw_ldri_progression.png` | Raw LDRI over narrative progression | Longitudinal Behaviour (§7) | "Raw LDRI is near-constant (CV=1.0%) across the entire novel, confirming that gamma drives all LDRI variation." |

## Gamma

| File | Purpose | Paper Section | Caption Suggestion |
|------|---------|---------------|-------------------|
| `gamma/gamma_progression.png` | Gamma (confidence) over narrative | Gamma Analysis (§8) | "Gamma_global varies with CV=21.3% and closely tracks LDRI (r>0.999). Gamma remains the dominant factor in LDRI at novel scale." |

## Components

| File | Purpose | Paper Section | Caption Suggestion |
|------|---------|---------------|-------------------|
| `components/component_evolution.png` | All 6 LDRI components over narrative | Component Evolution (§7) | "Uncertainty dominates (mean=0.89); intrinsic, temporal, memory, and interaction are small; resonance is negligible." |
| `components/resonance_progression.png` | Resonance component over narrative | Resonance Analysis | "Resonance contribution is near-zero throughout the novel, consistent with gold-standard findings." |

## Events

| File | Purpose | Paper Section | Caption Suggestion |
|------|---------|---------------|-------------------|
| `events/variance_comparison.png` | Variance contribution: Gold Standard vs Full Novel | Gamma Analysis (§8) | "Gamma dominance is nearly identical between the gold standard (96.5%) and the full novel (95.3%). Longer context does not reduce gamma's dominance." |
