# Unsupported Claims — LIT-DISC v1.1

These are claims that the paper makes (or implies) but that LIT-DISC v1.1 evidence does not yet support.

---

## U1: Memory enables long-range narrative coherence

**Claim:** The memory system tracks narrative development across a long text.

**Evidence:** Zero cross-window retrieval across 105 windows. Memory fills to capacity in the first window and never retrieves. The RFF key scheme produces keys that drift across narratively distinct windows, preventing cosine-similarity matches above the θ=0.7 threshold.

**Severity:** HIGH. The memory architecture does not achieve its stated goal at novel length.

**Recommendation:** Reformulate the memory system for cross-window retrieval — e.g., chunk-level keys, semantic hashing, or sliding-window overlap.

---

## U2: Resonance captures narrative development

**Claim:** Resonance measures how narrative elements echo or develop across the text.

**Evidence:** Mean resonance = 0.0002 (gold standard) and 0.00008 (novel). Removing resonance changes LDRI by |Δ| < 0.002. Resonance contributes ~0.02% of total component value.

**Severity:** HIGH. Resonance is a dead component.

**Recommendation:** Either redesign the resonance mechanism to produce meaningful values or remove it from the framework.

---

## U3: LDRI correlates with expert annotations

**Claim:** LDRI correlates with expert human judgments of literary-discourse richness.

**Evidence:** All gold standard passages have expert scores of 4 (placeholder from generation). No real expert annotation study has been conducted.

**Severity:** MEDIUM. Without human validation, LDRI remains an internal metric.

**Recommendation:** Conduct an expert annotation study with at least 3 annotators and 25 passages. Compute inter-annotator agreement and LDRI–expert correlation.

---

## U4: Raw LDRI becomes discriminative at novel scale

**Claim (implicit):** At novel length, accumulated context produces discriminative raw LDRI variation.

**Evidence:** Raw LDRI CV = 1.0% at novel scale (vs 2.6% at gold standard). Raw LDRI is even less discriminative, not more.

**Severity:** MEDIUM. The implicit claim is false — but the paper does not explicitly make this claim.

**Recommendation:** Remove any implication that raw LDRI becomes useful at scale. Acknowledge that structural LDRI change requires a different formulation.

---

## U5: Gamma dominance reduces with longer context

**Claim (implicit):** As context accumulates, raw LDRI contributes more to variation.

**Evidence:** Gamma dominance: 96.5% (gold standard) vs 95.3% (novel). A difference of −1.2 percentage points. Effectively unchanged.

**Severity:** LOW. The paper does not strongly assert this, but it is a natural expectation that readers may have.

**Recommendation:** Acknowledge explicitly that LDRI = Gamma × raw_ldri is inherently asymmetric, and that longer context does not reduce gamma dominance.
