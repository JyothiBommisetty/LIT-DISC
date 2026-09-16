# LIT-DISC Springer CCIS Peer Review Report

**Reviewer:** Senior NLP Researcher / Computational Literary Studies  
**Date:** 2026-07-29  
**Manuscript:** LIT-DISC: A Computable Metric for Literary Discourse Richness  
**Material reviewed:** Methodology specification (`methodology_equations.md`) plus evidence of experimental design.

---

## 1. Novelty — Score: 4/10

### Genuinely Novel Elements

1. **The activation redesign story (v1.0 → v1.1).** The discovery that 99% of activations were saturated at sigma=0.1, and that a single parameter change to sigma=5.0 produces 155× CV improvement while preserving ranking (ρ=0.98), is a genuine empirical contribution. It serves as a cautionary tale for validation of computational metrics.

2. **Traceability from code to equations.** The 55-equation mapping with source-file citations is methodologically rigorous. This is rare in CLS papers.

### Elements That Resemble Existing Work

1. **The pipeline is a standard NLP feature extraction pipeline.** Loading, sentence splitting, tokenization, POS tagging, feature extraction, aggregation — this is the architecture of every text analysis system since 2010.

2. **The Rayleigh CDF activation** is simply $1 - \exp(-x^2/2\sigma^2)$, which is a well-known function (chi-square survival function, Gaussian tail probability). Calling it "Rayleigh CDF" is correct but does not make it novel.

3. **The 6 discourse dimensions** are hand-crafted lexical feature sets. Each dimension is a bag of lexicon-based features: count emotion words, count narrative verbs, count sensory words, etc. This is 1990s computational linguistics.

4. **RFF for memory keys** is a standard technique from kernel methods (Rahimi & Recht, 2007). The application to narrative memory is moderately novel but poorly validated (zero retrieval).

5. **Weighted geometric mean aggregation** is standard in multi-metric indexing.

6. **The coupling kernel** is a linear combination of 6 similarity functions — essentially a learned kernel over hand-crafted features. Nothing architecturally novel.

### Overclaimed Contributions

1. **"Hierarchical Multi-Phase Discourse Theory"** is not a theory. It is a processing pipeline with 5 hardcoded levels (sentence, paragraph, chapter, narrative, discourse). There is no theoretical justification for exactly 5 levels, no proof of convergence, no axiomatic foundation, and no empirical validation that the hierarchy corresponds to any cognitive or linguistic reality. The term "theory" is inappropriate for a pipeline architecture.

2. **"Literary Discourse Richness Index"** is misleading. As the methodology document itself states (line 288): *"The metric is effectively a confidence-weighted constant."* A metric that varies by <3% on its "richness" component and 96.5% on its "uncertainty" component does not measure richness. It measures confidence.

3. **"47 equations"** — counts should be reduced. Many "equations" are trivial: Eq. 5 is `M = min(1.0, max(0.0, mean_beta))`, Eq. 6 is `(1+C)/2`, Eq. 10 is the inverse of Eq. 6. Padding the equation count inflates the claimed contribution.

4. **"K-iteration refinement loop"** with K=5 — the implementation runs 5 iterations of bottom-up propagation, but no convergence check is ever evaluated. The code has an `epsilon_convergence` parameter (config.py:68) that is **never used** in the refinement loop (controller.py:63-87). The refinement loop always runs exactly K iterations regardless of convergence.

### Verdict on Novelty

The methodological novelty is LOW. The paper describes a standard feature extraction pipeline with an activation corrected from a broken setting, a dormant memory system, a vestigial resonance module, and a metric dominated by uncertainty weighting. The novelty lies not in the mathematical framework but in the **honest empirical diagnosis** of these issues — which is valuable but not what the paper claims.

---

## 2. Methodology — Score: 3/10

### Mathematical Consistency

**Critical issues:**

1. **Equation 1 vs Equation 53: Circularity.** Eq. 1 computes activation $a_d$ from $\|\mathbf{H}_d\|$. But $\mathbf{H}_d$ is the embedding subspace built in Eq. 53 from the propagated signals. The activation $a_d$ is used in Eq. 2 to compute $R_{\text{signal}}$, which goes into Eq. 8 (raw LDRI), which goes into Eq. 9 (LDRI). But the embedding $\mathbf{H}_d$ itself was computed from propagated signals that never used LDRI. This is not circular in execution — but the flow is: features → embedding → activation → LDRI. The activation does NOT feed back into the embedding or signal propagation. This means Eq. 1–9 are a **post-hoc scoring** of the embedding, NOT an integral part of the HMP-DT inference. This should be stated explicitly.

2. **Gamma is defined twice.** The symbol $\gamma$ is used for:
   - The resonance exponent in LDRI (default 1.0, config.py:57)
   - The per-dimension confidence from feature extraction (Eq. 21)
   - The global confidence $\Gamma$ (Eq. 7)
   This is confusing. The variable table (Section 4) lists $\gamma$ for resonance exponent and $\gamma_d$ for per-dimension confidence, but Eq. 7 uses $\Gamma$ (uppercase gamma) while referring to per-dimension values as $\bar{\gamma}_d$. The notation needs systematic cleaning.

3. **Eq. 14 (resonance component) uses $\lambda$ without definition.** The constant $\lambda = 0.3$ appears in $e^{-1/\lambda} / (1 - e^{-1/\lambda})$ but $\lambda$ is not defined in or near Eq. 14. It's defined later in Section 3.25 (Eq. 40) as the resonance decay constant. A forward reference to an undefined variable is a mathematical error.

4. **Eq. 49 (sentence to paragraph variance) can produce negative values.** The code handles this with `max(0.0, sigma_sq_P)` (propagation.py:123), but the mathematical equation as written allows $\sigma^2(P) < 0$, which is impossible for a variance. The equation should include the clamping in its definition.

5. **Eq. 50 adds $0.1 \times (1 - \max \alpha_j)$.** The constant 0.1 is an arbitrary magic number. Why 0.1? What does it represent? It appears nowhere in the parameter table.

6. **Eq. 51 adds $0.05 \times \min_i |\tau - \tau_i|^2$.** Another arbitrary magic number. Why 0.05? What is the justification?

7. **The "Spline weights" in Eq. 46 are NOT spline weights.** They are inverse-distance weights (1/|distance| normalized to sum to 1). A spline weight would involve polynomial basis functions, knot points, and smoothness constraints. Calling inverse-distance weighting "spline weights" is mathematically incorrect.

### Equation Ordering

The ordering is generally logical but has two problems:

1. **Resonance factor T (Eq. 4) comes before recurrence (Eq. 40).** The reader encounters $T$ long before they learn what resonance actually is. The final LDRI (Eq. 9) uses $\tilde{T}$, but the resonance mechanism (Eq. 40–41) comes 36 equations later.

2. **Embedding construction (Eq. 53–55) comes at the end** but many earlier equations ($a_d$, $C$, $T$) depend on it. The ordering should be: embedding first, then activation, then factors, then LDRI.

### Symbol Definitions

1. **$\Gamma$ is defined as global confidence (Eq. 7) but the lowercase $\gamma$ appears in 5 different roles.**
2. **$C$ is used for both coupling factor (Eq. 3) and the coupling kernel components ($C_{\text{sim}}$, etc.).** The variable table distinguishes them, but the proximity in the document is confusing.
3. **$\phi_i(\tau)$ in Eq. 51** is defined as a spline interpolation weight in the variable table but is actually the inverse-distance weight from Eq. 46.
4. **$\delta$ appears twice** in the variable table: memory decay rate (0.05) and memory exponent in LDRI (1.0). These are completely different parameters using the same Greek letter.

### Weak Mathematical Justification

1. **Why the geometric mean?** The paper uses $\text{LDRI} = \Gamma \cdot (\prod \text{factor}^{w_i})^{1/\sum w_i}$. Why not an arithmetic mean? Why not a learned weighted sum? Why not a product? The geometric mean is presented without justification.

2. **Why exponents $\alpha=2.0, \beta=1.0, \gamma=1.0, \delta=1.0$?** This makes $R_{\text{signal}}$ twice as important as the other three factors. Why? There is no empirical or theoretical justification. And since the parameter table itself rates them "LOW: all factors near-constant," the exponent choice is irrelevant.

3. **Why $\sigma_{\text{max}} = 5.0$?** The document says it "aligns with feature norms (10-14)." But feature norms depend on the embedding construction (Eq. 53), which depends on signal propagation, which depends on features, which depends on lexicon design. If any lexicon changes, the optimal sigma changes. There is no principled parameter selection — only post-hoc fitting to the observed feature norms.

4. **The decomposition (Eq. 11–17) lacks derivation.** These equations appear to be ad-hoc log-scale approximations. Eq. 14 includes a mysterious constant $e^{-1/\lambda}/(1 - e^{-1/\lambda})$ with no explanation of where it comes from. The fraction normalization (Eq. 17) adds $c_{\text{unc}}$ to the denominator, making fractions not sum to 1.0 from the raw decomposition alone.

### Unnecessary Equations

- **Eq. 6** $\tilde{C} = (1+C)/2$: Since $C \in [0,1]$, this maps [0,1] → [0.5, 1.0]. This is a linear shift that could be absorbed into the exponent in Eq. 8.
- **Eq. 10** is the inverse of Eq. 6. Including both is redundant — they cancel out mathematically.
- **Eq. 5** $M = \min(1.0, \max(0.0, \bar{\beta}))$ is a clamp on a value already in [0,1].
- **Eq. 18** reliability: $\rho = 1 - \min(1.0, \sigma/0.1)$ is an arbitrary rescaling. Why 0.1 as the denominator?

### Redundant Equations

- **Eq. 3 and Eq. 4** are structurally identical formulas for different inputs. They could be presented as a single "norm-based factor" equation applied to different subspaces.
- **Eq. 29–34** (six coupling components) could be collapsed — most are variations of cosine similarity with different inputs.
- **Eq. 12, 13, 15** (interaction, temporal, memory components) are structurally identical: $\frac{w}{S} \cdot \Gamma \cdot \log(\max(1, 1 + X))$. This should be presented once.

### Algorithm Issues

1. **The algorithm calls `Memory.Retrieve(sig.f)` at line 29** but the retrieval threshold θ=0.7 never produces matches (empirically zero across 105 windows). This means lines 29–31 and 37–39 are dead code in practice.

2. **The refinement loop (lines 22–55) runs K=5 iterations** but the convergence parameter `epsilon` is never checked. Every iteration is identical in structure.

3. **The algorithm shows resonance update (lines 46–49) INSIDE the K-iteration loop**, but the resonance field accumulates across ALL iterations. This means resonance is updated 5 × 100 = 500 times per passage, which is far more than needed for narrative $\tau$ spanning [0,1].

### Complexity Analysis Issues

1. **The dominant term is listed as $O(K \cdot n \cdot M \cdot k_{\text{key}})$** but empirically, memory retrieval produces zero matches. The actual complexity is $O(K \cdot n \cdot M)$ just for computing cosine similarities that never succeed. This should be noted.

2. **The complexity table includes $L$ (lexicon lookup cost)** without specifying its value or that it is essentially O(1) for set/dict lookups.

3. **The embedding construction is listed as $O(n \cdot k_d + d)$** but Eq. 53 requires computing 5 statistics per feature component per dimension — this includes an FFT per component, which is $O(k_d \cdot n \log n)$ worst case.

### Summary

The methodology has serious mathematical issues: redundant equations, unclearly defined variables, unmotivated constants, and at least one mathematically impossible equation (Eq. 49 can produce negative variance). The framework is over-parameterized (20 configurable parameters, most rated "LOW sensitivity").

---

## 3. Scientific Writing — Score: 5/10

### Logical Flow

The document flows reasonably: overview → theory → equations → variables → algorithm → complexity → parameters → activation design → mapping → reproducibility → figures. This is a good structure for a methodology section.

However, the document serves dual purposes (documentation AND methodology) poorly. It alternates between:
- Implementation documentation ("*Source: `src/ldri/calculator.py:46-48`*")
- Mathematical formulation ("**Equation 1** (Activation from feature norm)")
- Empirical results ("149 out of 150 activations saturated at > 0.999")

A methodology section should not contain implementation source paths, empirical findings, or code snippets. These belong in reproducibility appendices and results sections.

### Section Ordering

The current ordering has a structural problem: **the embedding is defined last (Eq. 53–55) but used throughout (Eq. 1, 3, 4).** The reader encounters $\mathbf{H}_d$ in Eq. 1 but doesn't learn how it's constructed until Eq. 53. The methodology should define the embedding first.

### Transitions

The transitions between subsections are mechanical (horizontal rules with no bridging text). Each equation section reads as a standalone formula card rather than part of a coherent narrative.

### Paragraph Quality

Good: The parameter table and variable dictionary are excellent reference material. The implementation mapping table is thorough and would satisfy a reproducibility reviewer.

Weak: The "Why Each Dimension Exists" section (2.3) is purely descriptive. Each paragraph says "[Dimension X] captures [property]." There is no theoretical grounding, no citation, no argument for why these six and not others.

### Springer Writing Style Issues

1. **Code paths in methodology.** A Springer CCIS paper should never include `src/ldri/calculator.py:46-48` in the methodology section. This is fine for an appendix but distracting in the main text.

2. **"Implementation note" in Eq. 1.** The note says "It is NOT the sigmoid described in earlier architecture docs." This internal reference to different versions of the same project is inappropriate for a paper. Either the activation is a Rayleigh CDF or a sigmoid — the paper must present a single, consistent formulation.

3. **Overly detailed equation tables.** Each equation is followed by a 3–5 row table. For 55 equations, this is excessive. Springer CCIS limits papers to 12–15 pages. With 55 equations × average 8 lines each = 440 lines just for equation tables. This is roughly 8 pages of equations alone.

4. **The document states "The metric is effectively a confidence-weighted constant"** (line 288). This is a self-diagnosed fatal flaw. Publishing a paper that explicitly states its own metric doesn't work as named would be unusual. This sentence belongs in a limitations section, not in the methodology.

### Repetition

1. **Section 8 (Activation Design)** is largely redundant with information already in Sections 3.1 and 7. The sigma_max parameter is defined in Eq. 1, listed in the parameter table, and discussed again in Section 8. Three treatments of the same material.

2. **The embedding construction (Section 3.29)** repeats the 6-statistic descriptions for three subspaces (per-dimension, pairs, resonance) when they're structurally identical.

3. **The coupling kernel components (Eq. 29–34)** all have similar structure but are presented as 6 separate equations.

---

## 4. Experimental Design — Score: 4/10

### What Is Planned (Based on Evidence Files)

- 25-passage gold standard (25 authors, 5 categories)
- 4 baselines (full, constant features, random gamma, dummy)
- 5 ablations (no memory, no resonance, no coupling, no uncertainty, no narrative)
- 1 full novel (The Mistress of Spices, 105 windows)
- Comparison metrics: readability, TTR, sentence length

### Missing Baselines — CRITICAL

1. **Random baseline.** Compute LDRI on randomized text (word permutation, sentence shuffling). Without this, the claim that LDRI "traces the narrative arc" is unsupported. The metric may simply reflect vocabulary statistics.

2. **Null LDRI model.** Generate LDRI from random embeddings with the same marginal distribution. Compare the 21.8% CV to what a null model produces. If a random walk produces similar CV, the narrative arc claim collapses.

3. **Sentence-order ablation.** Process the novel with sentences randomly permuted. If LDRI trajectory changes, LDRI captures narrative ordering. If not, LDRI captures lexical content only.

4. **Cross-encoder baseline.** Compare LDRI against a modern sentence embedding (e.g., SBERT) similarity between consecutive windows. Any discourse metric should outperform a generic semantic similarity baseline.

5. **Simple feature baselines.** Compare LDRI against individual feature means (e.g., just semantic features, just emotion features) as richness proxies.

6. **Perplexity baseline.** Language model perplexity is a standard text quality metric. If LDRI is essentially a confidence-weighted constant, perplexity may perform equally well or better.

### Missing Evaluations

1. **No error analysis.** What passages does LDRI misrank? Where are the largest errors relative to any ground truth? (But there is no ground truth.)

2. **No confidence intervals.** The decomposition fractions (Eq. 17) are reported as point estimates. Bootstrap confidence intervals around the decomposition would reveal whether gamma dominance is statistically significant.

3. **No sensitivity analysis.** The parameter table itself rates most parameters "LOW sensitivity." This should be demonstrated experimentally, not asserted.

4. **No cross-validation.** The 25 passages are used for evaluation but not split into development/test. There is no out-of-sample validation.

5. **No author-style baseline.** Two passages from the same author should be more similar in LDRI than passages from different authors. This would validate cross-author discriminability. But the gold standard uses only one passage per author, making this impossible.

### Missing Ablations

1. **Feature dimension ablations.** Remove each of the 6 feature extractors individually and measure LDRI change. The current "without narrative" ablation is a start, but the other 5 dimensions are not individually tested.

2. **Window size sensitivity.** The novel uses 800-word windows. Results may change at 200, 400, 1600 words.

3. **K-iteration sensitivity.** Does K=1 produce the same LDRI as K=5? If the convergence parameter is never checked, lower K may suffice.

4. **Hierarchy depth.** Does a 3-level hierarchy (no narrative, no discourse) produce similar results? The top two levels are hardcoded with one node each — they add structural complexity but may contribute nothing empirically.

### Missing Statistical Analysis

1. **No significance tests.** Category differences (SNC=0.040 vs MNC=0.019) are reported without p-values or confidence intervals.

2. **No effect sizes.** Cohen's d or similar for all pairwise comparisons.

3. **No multiple comparison correction.** Six categories × five ablation conditions = 30 comparisons. The reported "LARGE" effect for uncertainty ablation would survive correction, but the "MODERATE" narrative ablation effect may not.

4. **No inter-annotator agreement.** The expert scores are placeholders (all 4). This is stated as a limitation, but the paper should not include them at all if they're fictional.

### Missing Robustness Tests

1. **Preprocessing variation.** The tokenizer is a regex (not spaCy, despite claiming a spaCy model). Tests with an actual spaCy tokenizer would reveal sensitivity.

2. **Encoding robustness.** Text loader supports multiple encodings. Does LDRI vary with encoding?

3. **Memory threshold sweep.** θ_retrieve = 0.7 produces zero retrievals. A sweep from 0.1 to 0.9 would determine whether the threshold or the architecture is the problem.

### Verdict on Experiments

The experimental design is **insufficient for a Springer CCIS paper**. The core claim — that LDRI measures literary discourse richness — has no external validation. There is no human annotation, no comparison to established metrics, no statistical rigor. The experimental story is: "We built a metric, it has a flaw (gamma dominance), we know the flaw, and we cannot fix it yet." This is not a publishable finding without addressing the flaw.

---

## 5. Figures — Score: N/A (Not Yet Drawn)

### Essential Figures (Must Include)

1. **Activation function comparison.** The v1.0 (sigma=0.1, saturated) vs v1.1 (sigma=5.0, sensitive) Rayleigh CDF plot. This is the paper's most compelling empirical finding and MUST be Figure 1 or 2. Without it, the sigma redesign story is invisible.

2. **Variance decomposition bar chart.** Gamma (96.5%) vs raw LDRI (0.3%) vs covariance (3.2%). This figure tells the paper's central (unintended) story: the metric doesn't measure what it claims to measure.

3. **LDRI trajectory for the novel.** The 105-window LDRI progression with narrative events annotated. This is required for the case study.

4. **Pipeline architecture diagram.** Standard block diagram showing the 13 pipeline stages.

### Recommended Figures

5. **Hierarchy tree visualization.** 5-level tree showing how sentence → paragraph → chapter → narrative → discourse nodes relate.

6. **Memory analysis plots.** Memory fill rate, beta distribution, retrieval count per window.

7. **Baseline comparison bar chart.** Full LDRI vs constant features vs random gamma vs dummy — the ρ values.

8. **Ablation effect size plot.** |Δ| for each ablation on a log scale, with the uncertainty ablation clearly dominating.

9. **Runtime scaling plot.** Linear scaling confirmation (optional — supportive but not essential).

### Unnecessary Figures

- **Equation dependency graph (Figure 3).** 55 nodes is too complex for a readable figure. A reduced version showing only the critical path (Eq. 1→2→8→9) is fine, but the full DAG belongs in supplementary material.
- **Dimension interaction diagram (Figure 5).** Unless a clear pattern emerges, this will be a indecipherable graph of pairwise interactions.

---

## 6. Related Work — Score: 3/10

The methodology document does not include a related work section. Based on the planned paper outline, the following gaps need addressing:

### Missing Citations

1. **Rahimi & Recht (2007)** — Random Fourier Features. The RFF method is used for memory keys but uncited.

2. **Mikolov et al. (2013)** / **Pennington et al. (2014)** — Word embeddings. The semantic dimension uses word-level features but no pretrained embeddings.

3. **Devlin et al. (2019)** — BERT and contextual embeddings. A 2026 paper cannot ignore contextual embeddings entirely, even if it doesn't use them.

4. **Coh-Metrix (Graesser et al., 2004; McNamara et al., 2014)** — The most directly relevant prior work. Coh-Metrix computes 200+ indices of textual cohesion and readability, including some discourse-level features. The paper must compare against Coh-Metrix explicitly.

5. **LIWC (Pennebaker et al., 2001, 2015)** — The emotion and interaction dimensions are essentially LIWC categories. The paper must cite LIWC and explain how LIT-DISC differs from extending LIWC with geometric mean aggregation.

6. **Narrative theory frameworks** — Genette (1980), Bal (1997), etc. If the paper claims literary-theoretic grounding, it must cite narrative theory. Currently there are zero literary theory citations.

7. **Computational narrative understanding** — Elson et al. (2010) on narrative schemas, Finlayson (2012) on plot units, Gervás (2013) on computational narratology.

8. **Readability metrics** — Flesch (1948), Kincaid et al. (1975), Dale-Chall (1948). The baselines include Flesch-Kincaid but don't cite it.

9. **Text quality assessment** — Louis & Nenkova (2013), systems for assessing text quality without reference to gold standards.

### Weak Comparisons

The planned comparison against readability (ρ=0.53 with sentence length, ρ=−0.16 with Flesch-Kincaid) is inadequate. These are not competing metrics for literary discourse richness — they are designed to measure reading difficulty. A stronger comparison would be against:
- Coh-Metrix dimensions (narrativity, syntactic simplicity, word concreteness, referential cohesion, deep cohesion)
- Perplexity from a causal language model
- Supervised literary quality classifiers

### Missing Research Gap

The paper needs to clearly articulate a research gap that LIT-DISC fills. Current candidates:
- No existing metric specifically measures "literary discourse richness" (but why is this a gap? Why does the world need this metric?)
- Existing metrics are discrete/manual (RST) while LIT-DISC is continuous/automatic (but so are all embedding-based methods)
- Existing metrics don't incorporate uncertainty (but LIT-DISC's uncertainty dominates to the point of making the metric a constant)

None of these are compelling. The paper needs a clearer value proposition.

---

## 7. Paper Structure — Score: 6/10

### Current Ordering

The planned ordering (Introduction → Related Work → Methodology → Experimental Setup → Results → Case Study → Discussion → Conclusion) is standard and appropriate for CCIS.

### Proposed Changes

1. **Merge "Experimental Setup" into "Methodology."** For a short CCIS paper (12-15 pages), having separate methodology and experimental setup sections is wasteful. The experimental setup (corpus, baselines, ablations, evaluation metrics) should be a subsection of Methodology.

2. **Move activation design story to Results, not Methodology.** Section 8 of the methodology document (activation design with v1.0→v1.1 comparison) contains empirical findings (149/150 activations saturated, CV improvement 155×). This is a **result**, not a methodology design choice. The methodology should state "we use sigma=5.0" and the results should show "and here's why this matters."

3. **Move gamma dominance diagnosis from Methodology to Discussion.** The note that "the metric is effectively a confidence-weighted constant" (methodology line 288) is a critical finding that should headline the Discussion, not be buried in the methodology.

4. **The implementation mapping table (Section 9) belongs in an appendix or supplementary material.** A 55-row table in the main text would consume 3-4 pages.

5. **The computational complexity analysis should be in a short subsection, not a standalone major section.**

### Recommended Section Order for CCIS

1. Introduction (1.5 pages)
2. Related Work (2 pages)
3. The LIT-DISC Framework (4-5 pages)
   3.1 HMP-DT Architecture
   3.2 Discourse Dimensions
   3.3 Embedding Construction (Eq. 53-55 FIRST)
   3.4 Activation Function (Eq. 1)
   3.5 LDRI Computation (Eq. 2-9)
   3.6 Memory System (Eq. 22-28)
   3.7 Coupling Mechanism (Eq. 29-39)
   3.8 Resonance (Eq. 40-41)
   3.9 Uncertainty Propagation (Eq. 47-52)
   3.10 Parameters and Complexity
4. Experimental Setup (1 page)
   4.1 Gold Standard Corpus
   4.2 Full-Novel Selection
   4.3 Baselines and Ablations
   4.4 Evaluation Protocol
5. Results (3 pages)
   5.1 Activation Redesign (MOVE FROM METHODOLOGY)
   5.2 Gold-Standard Analysis
   5.3 Baselines and Ablations
   5.4 Full-Novel Scalability
6. Case Study: The Mistress of Spices (2 pages)
7. Discussion (1.5 pages)
   7.1 Gamma Dominance (MOVE FROM METHODOLOGY)
   7.2 Architectural Limitations
   7.3 Implications
8. Conclusion (0.5 pages)

---

## 8. Publication Readiness — Score: BORDERLINE

### Acceptance Probability: 30-40% at CCIS

### Why It Might Be Accepted

- **Thorough implementation-to-equation traceability** — rare and valued by CL venues
- **Honest presentation of limitations** — the paper identifies its own flaws (gamma dominance, zero memory retrieval, vestigial resonance)
- **The activation redesign story** is a clear methodological contribution
- **Full-novel evaluation** at 80k words is a genuine scalability demonstration
- **Open-science artifacts** — the evidence folder with 42 documents is exceptional

### Why It Will Likely Be Rejected

- **The metric doesn't measure what it claims to measure.** A reviewer will read "Literary Discourse Richness Index" and then "the metric is effectively a confidence-weighted constant" and ask: why is this called a richness index? The name is false advertising.

- **No external validation.** Without expert annotations or correlation with any established measure, LDRI is a self-referential metric. A CCIS reviewer will not accept "we haven't done expert validation yet" as an excuse.

- **The framework is overbuilt but underperforming.** 47 equations, 20 parameters, 6 coupling components, 5-level hierarchy, RFF memory, recursive resonance — but the final metric is 96.5% determined by a single factor (gamma), memory never retrieves, resonance is near-zero, and all but one ablation produce |Δ| < 0.002. A reviewer will ask: why does this framework have so much machinery that demonstrably does nothing?

- **Low novelty.** The activation correction is valuable but not sufficient for a full paper. The rest is standard NLP pipeline engineering.

- **Weak statistical rigor.** No significance tests, no confidence intervals, no effect sizes, no multiple comparison correction.

- **The "theory" is not a theory.** Calling HMP-DT a "theory" will not survive peer review. A reviewer with a linguistics background will ask for falsifiable predictions, which the framework cannot provide.

### Mitigating Factors

- CCIS has lower acceptance standards than ACL/COLING/EMNLP
- The case study and evidence organization may compensate for methodological weaknesses
- If positioned as a "work-in-progress report" rather than a mature framework, acceptance chances improve

---

## 9. Missing Content

### Results Section (CRITICAL — Entirely Missing)

The most important missing section. The paper cannot be submitted without:
- Gold-standard LDRI values (25 passages)
- Baseline comparisons (4 conditions)
- Ablation results (5 conditions)
- Statistical analysis (significance, effect sizes)
- Full-novel results (runtime, memory, LDRI trajectory)
- Variance decomposition (gold standard + novel)
- Activation comparison (v1.0 vs v1.1)

### Discussion Section (CRITICAL — Entirely Missing)

- Interpretation of gamma dominance
- Why memory never retrieves
- Why resonance is near-zero
- Comparison with prior work
- Implications and limitations
- Threats to validity

### Conclusion Section (CRITICAL — Entirely Missing)

- Summary of contributions
- Key takeaways
- Limitations (summary)
- Future work

### Threats to Validity Section (IMPORTANT — Missing)

A dedicated threats-to-validity section addressing:
- Internal validity: circular event identification, no null model, single-rater analysis
- External validity: single novel, 25-passage corpus, English only, 20th-century canon
- Construct validity: metric name vs behavior, gamma dominance
- Statistical validity: small N, no significance tests, no power analysis

### Limitations Section (IMPORTANT — Missing as Standalone)

The limitations are documented in `limits.md` but not in the paper. A condensed version (1 paragraph per limitation, 7-10 limitations) should be in the Discussion.

### Reproducibility Appendix (IMPORTANT — Missing as Appendix)

The reproducibility notes from Section 10 of the methodology document should be expanded into a dedicated appendix covering:
- Full execution environment
- Seed values and their locations
- Full parameter listing
- Data provenance
- Expected runtime on reference hardware

### Results Tables (CRITICAL — Entirely Missing)

At minimum:
- Table 1: LDRI, raw LDRI, gamma for all 25 passages (mean, std, CV, min, max)
- Table 2: Category-level LDRI (5 categories × mean, std)
- Table 3: Baseline conditions (4 rows × LDRI mean, std, ρ)
- Table 4: Ablation conditions (6 rows × LDRI mean, Δ)
- Table 5: Novel evaluation (105-window summary: mean, std, CV, min, max for LDRI, raw, gamma)

### Figures (IMPORTANT — Entirely Missing)

See Section 5 of this review. At minimum: activation comparison, variance decomposition, LDRI trajectory, pipeline diagram.

---

## 10. Action Plan — Prioritized Checklist

### CRITICAL (Will Cause Rejection If Missing)

- [ ] **C1: Rename the metric or reformulate it.** LDRI is not a richness index. Options: (a) rename to "Confidence-Weighted Discourse Index" (CWDI), (b) reformulate so features contribute more than 0.3% of variance, or (c) explicitly position the paper as "a cautionary tale about metric validation" rather than "a new literary discourse metric." (c) is the most honest and likely most publishable.

- [ ] **C2: Remove placeholder expert scores.** Having all-4 annotations in the gold standard is misleading. Either obtain real annotations or remove the field entirely.

- [ ] **C3: Add statistical significance testing.** Bootstrap confidence intervals for LDRI means, category differences, and ablation effects. At minimum, report 95% CIs.

- [ ] **C4: Write the Results section.** This is the most substantial missing piece. Include all 5 required tables.

- [ ] **C5: Write the Discussion section.** Address gamma dominance, zero retrieval, vestigial resonance explicitly.

- [ ] **C6: Write the Conclusion.** 1 page max.

- [ ] **C7: Remove "theory" from "HMP-DT."** Refer to it as a "framework," "pipeline," or "architecture." Do not call it a theory unless you can provide falsifiable predictions.

- [ ] **C8: Fix the equation ordering.** Place embedding construction (Eq. 53–55) before activation (Eq. 1). The reader needs to know what $\mathbf{H}_d$ is before seeing it in an equation.

- [ ] **C9: Add null model baseline.** At minimum, shuffled-text baseline and random-embedding LDRI comparison.

### IMPORTANT (Will Significantly Weaken the Paper If Missing)

- [ ] **I1: Add memory threshold sweep.** Test θ_retrieve from 0.1 to 0.9. If retrievals occur at lower thresholds, the memory architecture can be partially salvaged. If not, the memory section must acknowledge the architectural failure more directly.

- [ ] **I2: Add coherence / semantic similarity baseline.** Compare LDRI against SBERT cosine similarity between windows. This tests whether LDRI adds anything beyond generic semantic similarity.

- [ ] **I3: Add perplexity baseline.** Language model perplexity per window. If LDRI correlates with perplexity, LDRI is measuring predictability, not richness.

- [ ] **I4: Clean up notation.** Remove duplicate $\delta$, clarify $\gamma$ vs $\Gamma$, define $\lambda$ before Eq. 14.

- [ ] **I5: Add confidence intervals for decomposition.** Bootstrap the variance decomposition (gamma: 96.5%) to show it's statistically significant.

- [ ] **I6: Add related work citations.** Coh-Metrix, LIWC, RFF (Rahimi & Recht 2007), narrative theory (Genette, Bal), contextual embeddings (Devlin et al.).

- [ ] **I7: Fix Eq. 49 to reflect the clamping.** The equation as written can produce negative variance.

- [ ] **I8: Remove or justify magic numbers.** Explain 0.1 (Eq. 50) and 0.05 (Eq. 51).

- [ ] **I9: Add window size sensitivity.** Test at least 200, 400, 800, 1600 words on one novel.

- [ ] **I10: Write Limitations section** (7-10 limitations, 1 paragraph each).

- [ ] **I11: Write Threats to Validity section** (4 types × 2-3 threats each).

### OPTIONAL (Improves Quality but Not Strictly Required)

- [ ] **O1: Add error analysis.** Which passages are most/least separable by LDRI?

- [ ] **O2: Add cross-validation.** Split 25 passages into train/test for any learned components (but there are none).

- [ ] **O3: Add K-iteration sensitivity.** Compare K=1 vs K=5 LDRI values.

- [ ] **O4: Add hierarchy depth ablation.** Remove levels 4–5 and compare.

- [ ] **O5: Add per-feature-dimension ablations.** Remove each of the 6 feature extractors individually.

- [ ] **O6: Move implementation mapping to appendix.** 55 rows × 5 columns is too much for main text.

- [ ] **O7: Condense Equation 6 and 10.** Present the normalization as a single operation, not two equations.

- [ ] **O8: Rename "spline weights" (Eq. 46).** Call them "inverse-distance weights."

- [ ] **O9: Relabel the resonance and coupling factors.** Since both use the same formula (||H||/sqrt(|H|)), present them as instances of a general norm-based factor.

- [ ] **O10: Add a single literary theory citation** to ground the 6 dimensions. Without any, the dimension design appears entirely ad-hoc.

---

## Final Summary

| Criterion | Score | Verdict |
|-----------|-------|---------|
| Novelty | 4/10 | Overclaimed; the activation redesign story is the only genuinely novel element |
| Methodology | 3/10 | Significant mathematical issues: circular notation, impossible equations, redundant presentation, undefined variables, unmotivated constants |
| Scientific Writing | 5/10 | Good structure but mixing implementation docs with mathematical formulation; excessive equation count for CCIS |
| Experimental Design | 4/10 | No external validation, no significance testing, missing critical baselines (null model, perplexity, coherence) |
| Publication Readiness | Borderline | 30-40% acceptance probability. Would be rejected at ACL/COLING/EMNLP. Possible at CCIS if restructured as a work-in-progress/position paper |

**Recommendation to authors:** The paper's greatest strength is its honesty about its own limitations. Consider reframing the paper as: "What We Learned From Building a Literary Discourse Metric" — a lessons-learned / cautionary-tale paper that documents the activation redesign, gamma dominance diagnosis, and architectural limitations. Positioned honestly, this could be a valuable contribution to computational literary studies methodology. Positioned as a mature metric ready for deployment, it will face fatal criticism.
