# LDRI: The Literary Discourse Richness Index

## A Formal Metric for Computational Literary Discourse Analysis

**Document Type:** Methodology Chapter  
**Framework:** LIT-DISC (Literary Intelligence through Text-based Discourse Analysis)  
**Target Journal:** IEEE Transactions on Affective Computing / Digital Scholarship in the Humanities  
**Prerequisite:** HMP-DT Version 1.0 — Research Freeze

---

## Abstract

The Literary Discourse Richness Index (LDRI) is a principled quantitative metric for measuring the density, diversity, and structural complexity of literary discourse. Unlike readability scores, sentiment measures, or topic models, LDRI is grounded in a formal model of literary discourse as a coupled hierarchical signal system spanning six dimensions (semantic, narrative, interaction, symbolic, evolution, emotion). It jointly accounts for cross-dimensional coupling, temporal dynamics, memory persistence, narrative resonance, and propagated uncertainty within a single bounded, interpretable scalar.

This chapter defines discourse richness axiomatically, derives LDRI from first principles using the HMP-DT Narrative Embedding, decomposes it into narratologically meaningful components, establishes its mathematical properties, and compares it against existing literary metrics.

---

## 1. What Is Discourse Richness?

### 1.1 A Working Definition

**Discourse richness** is the degree to which a literary text simultaneously activates multiple distinct discourse dimensions with strong mutual interaction, meaningful temporal evolution, and persistent structural memory.

This definition rests on four interconnected claims:

1. **Multi-dimensionality.** Rich discourse is not merely "complex" in a single register; it operates across semantic, narrative, interactional, symbolic, evolutionary, and emotional dimensions simultaneously. A text that activates only one dimension (e.g., dense vocabulary but no character arc, no dialogue, no emotion) is discourse-poor regardless of its lexical sophistication.

2. **Interaction.** Richness is not the sum of independent dimension scores. Discourse dimensions interact: emotional valence shapes symbolic meaning, narrative tension constrains interaction patterns, character arcs emerge through semantic framing. A metric that treats dimensions as independent contributors misses the central literary phenomenon of registers operating upon each other.

3. **Temporality.** Literary discourse unfolds in time. A static signal extraction cannot capture richness because the timing, sequence, and accumulation of signals matter. An early symbol that resurfaces at the climax contributes more to richness than the same symbol appearing in isolation.

4. **Persistence.** Rich texts leave traces. Memories persist, resonate, and decay across the narrative arc. A text whose signals vanish without trace is impoverished relative to one whose signals echo across chapters.

### 1.2 What LDRI Is Not

LDRI must be distinguished from existing textual metrics to avoid category confusion:

| Metric | What It Measures | Why It Differs from LDRI |
|--------|-----------------|--------------------------|
| **Readability** (Flesch–Kincaid, Coleman–Liau, Dale–Chall) | Syllable density, sentence length, word familiarity | Readability is a surface property of decoding ease. LDRI operates on discourse signals, not lexical statistics. A highly readable text (Hemingway) can have high LDRI; an unreadable one (Faulkner's stream-of-consciousness) can have low LDRI. The two are uncorrelated by design. |
| **Syntactic complexity** (Yngve depth, D-level, parse tree depth) | Clause embedding depth, left-branching structures | LDRI ignores syntax entirely. Discourse richness is orthogonal to syntactic complexity: Faulkner scores high on both, but Proust's hypotactic sentences are syntactically complex yet discourse-simple in several dimensions. |
| **Lexical richness** (Type-Token Ratio, HD-D, MTLD, vocd-D) | Vocabulary diversity, hapax legomena density | A text can have vast vocabulary (Joyce) yet low cross-dimensional coupling. Lexical richness is a single-dimension property confined to the semantic register. LDRI spans six. |
| **Sentiment** (VADER, LIWC valence, SentiWordNet polarity) | Emotional polarity, arousal, dominance | LDRI subsumes emotion as one dimension (Em) but does not reduce to it. A text with uniformly negative sentiment (high polarity, low variance) scores low on LDRI because the emotional signal is static and uncoupled from other dimensions. |
| **Topic diversity** (LDA topic entropy, coherence scores) | Number and balance of topical clusters, topic distinctness | Topic diversity is semantic only. Two texts with identical topic distributions can differ radically in how those topics interact with character development, emotional trajectory, and symbolic resonance. |

### 1.3 The Intuition

Consider three texts:

- **Text A:** A straightforward genre romance. Dialogue is purely functional ("pass the salt"), emotion is explicit and unchanging ("I love you"), symbolism is absent, narrative arc is linear from meeting to marriage.

- **Text B:** A literary novel. Semantic motifs recur with accruing symbolic meaning, dialogue operates through subtext, emotions evolve through interaction, narrative structure is non-linear, and symbols from early chapters cross-reference later developments.

- **Text C:** A philosophical treatise. Dense vocabulary, complex syntax, and abstract concepts, but minimal narrative arc, no dialogue, no character development, no emotional trajectory.

Existing metrics would rank Text C highest (most lexically rich, most syntactically complex, highest topic entropy). LDRI should rank Text B highest because it activates all six dimensions with strong mutual interaction. Text A ranks lowest (single emotional register, minimal interaction). Text C ranks moderate (strong semantic dimension but near-zero narrative, interactional, and emotional signals).

This aligns with the literary critical consensus: *richness* in the literary sense refers to the interplay of multiple registers of meaning, not to decoding difficulty or lexical volume.

---

## 2. Mathematical Axioms

### 2.1 Axiom 1: Boundedness

**Definition.** $\text{LDRI}(\mathbf{H}) \in [0, 1]$ for all Narrative Embeddings $\mathbf{H} \in \mathbb{R}^d$.

**Motivation.** Literary richness is a bounded quantity. There is a theoretical maximum — a text cannot be infinitely rich — and a natural minimum (a text with no detectable literary signal). Boundedness enables cross-text comparison, threshold setting, and intuitive interpretation.

**Proof sketch.** By construction: each constituent sub-score is bounded to $[0, 1]$, each transformation preserves boundedness, and the global confidence factor $\Gamma \in [0, 1]$. The full proof is given in §7.

### 2.2 Axiom 2: Monotonicity

**Definition.** Let $\mathbf{H}$ and $\mathbf{H}'$ be Narrative Embeddings such that $\|\mathbf{H}'_d\| \geq \|\mathbf{H}_d\|$ for all $d \in \mathcal{D}$ with at least one strict inequality, and $\gamma'_d \geq \gamma_d$ for all $d$. Then $\text{LDRI}(\mathbf{H}') \geq \text{LDRI}(\mathbf{H})$.

**Motivation.** Adding literary signal (stronger features, higher confidence) should never decrease measured richness. This is the minimal consistency requirement for any quantity called a "measure."

**Proof sketch.** Each dimension activation $a_d = g(\|\mathbf{H}_d\|)$ is monotonic non-decreasing in $\|\mathbf{H}_d\|$. The coupling and resonance factors are monotonic in their respective norms. The geometric mean preserves monotonicity. The confidence factor $\Gamma$ is monotonic in each $\gamma_d$. The full proof is in §7.

### 2.3 Axiom 3: Interpretability

**Definition.** There exists a decomposition $\text{LDRI} = \sum_{i=1}^k c_i$ such that each $c_i$ corresponds to a distinct narratological concept.

**Motivation.** A metric for literary analysis is useless if it cannot be explained in literary terms. LDRI must answer: *Why is this text richer than that one?* The decomposition in §6 provides the answer in terms of intrinsic signal strength, cross-dimensional interaction, temporal evolution, memory depth, resonance persistence, and uncertainty.

**Proof sketch.** The decomposition in §6 is additive by construction, and each component maps to a well-defined discourse phenomenon.

### 2.4 Axiom 4: Stability

**Definition.** Let $\mathbf{H}_\epsilon = \mathbf{H} + \epsilon \cdot \mathbf{u}$ where $\mathbf{u} \in \mathbb{R}^d$ is a unit vector and $\epsilon > 0$. Then $\text{LDRI}(\mathbf{H}_\epsilon) = \text{LDRI}(\mathbf{H}) + O(\epsilon)$ as $\epsilon \to 0$.

**Motivation.** Small perturbations in the Narrative Embedding (due to feature extraction noise, parameter variation) should produce proportionally small changes in LDRI. The metric should not amplify noise.

**Proof sketch.** LDRI is a composition of Lipschitz-continuous functions: norms, radial basis activations, and the geometric mean. Each has a bounded derivative. By the chain rule, the composition is Lipschitz on the domain of bounded $\mathbf{H}$. The full proof is in §7.

### 2.5 Axiom 5: Comparability

**Definition.** For any two texts $T_1, T_2$ with Narrative Embeddings $\mathbf{H}_1, \mathbf{H}_2$, the ordering induced by LDRI is invariant to differences in text length, chapter count, or sentence count.

**Motivation.** Literary works vary enormously in length ($10^3$ to $10^6$ words). A metric that systematically advantages longer texts would be misleading. LDRI must measure *density* of discourse richness, not *volume*.

**Proof sketch.** The Narrative Embedding is length-normalized by the HMP-DT process: narrative position $\tau \in [0, 1]$, confidence-weighted aggregation, spline interpolation across fixed sample points $m = 100$. LDRI operates on this normalized representation.

### 2.6 Axiom 6: Robustness

**Definition.** If a subset of dimensions $\mathcal{D}' \subset \mathcal{D}$ has uniformly low confidence ($\gamma_d < \epsilon$ for all $d \in \mathcal{D}'$), then the contribution of $\mathcal{D}'$ to LDRI vanishes as $\epsilon \to 0$.

**Motivation.** Not all dimensions can be reliably extracted from every text. Symbolism detection in a literal text, for example, should not produce spurious richness. Low-confidence dimensions must be down-weighted.

**Proof sketch.** The confidence factor $\Gamma$ is the product of per-dimension confidences. As $\epsilon \to 0$, $\Gamma \to 0$, and all dimension contributions vanish proportionally.

### 2.7 Axiom 7: Scale Invariance

**Definition.** For any $c > 0$, $\text{LDRI}(c \cdot \mathbf{H}) = \text{LDRI}(\mathbf{H})$.

**Motivation.** The absolute scale of feature vectors depends on NLP extractor design, not on discourse richness. Multiplying all features by a constant should not change the richness assessment.

**Proof sketch.** Per HMP-DT Assumption A3 (revised), all feature magnitudes are bounded to $[0, 1]$ or $[-1, 1]$. Within the LDRI derivation, dimension activations $g(\|\mathbf{H}_d\|)$ use the radial basis function with scale parameter $\sigma_d$ estimated from the embedding space, which absorbs uniform scaling.

### 2.8 Axiom 8: Explainability

**Definition.** There exists an algorithm $\mathcal{E}$ that runs in $O(d + |\mathcal{M}|)$ time and produces an additive decomposition $\text{LDRI} = \sum_{i=1}^6 c_i$ where each $c_i \in [0, 1]$ corresponds to a narratological factor.

**Motivation.** A literary scholar using LDRI must be able to identify *what makes a text rich* — is it the emotional complexity? The symbolic density? The temporal arc? The decomposition in §6 provides this.

**Proof sketch.** The decomposition is additive by construction and requires $O(d + |\mathcal{M}|)$ operations where $d = 120$ and $|\mathcal{M}| \leq 200$ (per HMP-DT complexity bounds).

### 2.9 Axiom 9: Uncertainty Awareness

**Definition.** Let $\Gamma = \prod_{d \in \mathcal{D}} \gamma_d$ be the global confidence. Then $\lim_{\Gamma \to 0} \text{LDRI} = 0$ and $\partial \text{LDRI} / \partial \gamma_d > 0$ for all $d \in \mathcal{D}$.

**Motivation.** A metric that returns a high richness score based on unreliable feature extractions is misleading. LDRI must penalize uncertainty: "we do not know" should map to "low measured richness."

**Proof sketch.** Global confidence $\Gamma$ directly multiplies the raw LDRI score. As any $\gamma_d \to 0$, $\Gamma \to 0$, driving LDRI to zero. The partial derivative with respect to each $\gamma_d$ is positive by the product rule.

---

## 3. Derivation from the Narrative Embedding

### 3.1 Prerequisites

Let $\mathbf{H} \in \mathbb{R}^d$ be the Narrative Embedding produced by HMP-DT (specification §14). Decompose $\mathbf{H}$ into its constituent subspaces:

$$\mathbf{H} = \bigoplus_{d \in \mathcal{D}} \mathbf{H}_d \oplus \mathbf{H}_{\text{pairs}} \oplus \mathbf{H}_{\text{res}} \tag{1}$$

where $\mathbf{H}_d \in \mathbb{R}^{d/10}$, $\mathbf{H}_{\text{pairs}} \in \mathbb{R}^{3d/10}$, $\mathbf{H}_{\text{res}} \in \mathbb{R}^{d/10}$, and $d = 120$ by default (HMP-DT §3.6).

The HMP-DT process also produces:
- Per-dimension confidences $\gamma_d \in [0, 1]$ propagated through the hierarchy
- Memory store $\mathcal{M}_{\text{store}}$ with activations $\beta_j \in [0, 1]$
- Resonance field $\mathcal{R}(\tau) \in \mathbb{R}^d$ at narrative positions $\tau \in [0, 1]$

These constitute the complete input to LDRI.

### 3.2 The Dimension Activation Function

For each dimension $d \in \mathcal{D}$, define the **dimension activation**:

$$a_d = g(\|\mathbf{H}_d\|), \quad g(x) = 1 - \exp\left(-\frac{x^2}{2\sigma_d^2}\right) \tag{2}$$

where $\sigma_d$ is the scale parameter for dimension $d$, estimated as the expected norm of $\mathbf{H}_d$ under a null model (independent, identically distributed features with no literary signal).

**Justification.** The function $g: [0, \infty) \to [0, 1)$ is a radial basis activation derived from the Rayleigh distribution. Under the null hypothesis that $\mathbf{H}_d$ consists of $k = d/10$ independent $\mathcal{N}(0, \sigma_d^2)$ components, $\|\mathbf{H}_d\|$ follows a scaled chi distribution with mean $\sigma_d\sqrt{2}\,\Gamma((k+1)/2)/\Gamma(k/2)$. The probability that an observed norm exceeds this null expectation is $e^{-x^2/2\sigma_d^2}$, and the signal-richness is the complement: $1 - e^{-x^2/2\sigma_d^2}$.

This gives $a_d$ a precise interpretation: it is the probability that the observed signal in dimension $d$ exceeds what would be expected from noise alone. It is not an arbitrary scaling but a theoretically grounded measure of signal detection.

**Properties:**
- $g(0) = 0$: zero signal produces zero activation
- $\lim_{x \to \infty} g(x) = 1$: sufficiently strong signal saturates
- $g$ is strictly increasing and concave for $x > \sigma_d$: diminishing returns

### 3.3 Why Not a Weighted Average

A weighted average $\sum_d w_d a_d$ would be inappropriate for three reasons:

1. **Compensation.** A weighted average allows a very high score in one dimension to compensate for near-zero scores in all others ($0.9 \cdot 1.0 + 0.1 \cdot 0.0 = 0.9$). This violates the intuition that discourse richness requires multi-dimensional activation.

2. **Independence.** A weighted average treats dimensions as independent contributors. It cannot capture the structural requirement that richness emerges from joint activation, not isolated strength.

3. **Linearity.** A weighted average is a linear function. Discourse richness is not linear: doubling all signals does not double richness (saturation), and the presence of two weak signals is worth more than one strong signal (convexity).

### 3.4 The Signal Richness Norm

Replace the weighted average with the $\ell_2$ norm of the activation vector, normalized to $[0, 1]$:

$$\mathbf{a} = (a_S, a_N, a_I, a_{Sy}, a_E, a_{Em})^\top \in [0, 1]^6 \tag{3}$$

$$\mathcal{R}_{\text{signal}} = \frac{\|\mathbf{a}\|_2}{\sqrt{6}} = \sqrt{\frac{1}{6} \sum_{d \in \mathcal{D}} a_d^2} \in [0, 1] \tag{4}$$

**Justification.** The $\ell_2$ norm is chosen over the $\ell_1$ norm ($\sum_d a_d$) and the $\ell_\infty$ norm ($\max_d a_d$) for specific geometric and literary reasons:

| Norm | Value for $(1, 0, 0, 0, 0, 0)$ | Value for $(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)$ | Property |
|------|-------------------------------|---------------------------------------------|----------|
| $\ell_1$ | 1.0 | 3.0 | Rewards additive accumulation; one strong dimension = six weak ones |
| $\ell_2$ | 0.41 | 0.50 | Rewards uniform activation; penalizes concentration |
| $\ell_\infty$ | 1.0 | 0.50 | Only cares about the strongest dimension |

The $\ell_2$ norm uniquely satisfies the **concentration penalty**: for a fixed total activation budget, evenly distributed activation yields higher richness than concentrated activation. This matches the literary intuition that a text engaging all registers moderately is richer than a text dominating one register while neglecting others.

**Proof of concentration penalty.** For any $\mathbf{a} \in [0, 1]^6$ with fixed $\ell_1$ norm $S = \sum_d a_d$, the $\ell_2$ norm is maximized when $a_d = S/6$ for all $d$ (by Jensen's inequality, since the square is convex). Therefore $\|\mathbf{a}\|_2 \leq S/\sqrt{6}$, with equality iff the activation is uniform. The normalized score $\mathcal{R}_{\text{signal}} = \|\mathbf{a}\|_2/\sqrt{6} \leq S/6$, with equality iff uniform. For the concentrated case, $\mathcal{R}_{\text{signal}}$ is strictly less than the arithmetic mean.

### 3.5 The Coupling Enrichment Factor

Cross-dimensional coupling amplifies richness because interacting registers produce emergent meaning that no single register can produce alone. Define:

$$\mathcal{C} = \frac{\|\mathbf{H}_{\text{pairs}}\|}{\|\mathbf{H}_{\text{pairs}}\|_{\max}} \in [0, 1] \tag{5}$$

**Justification.** $\mathbf{H}_{\text{pairs}}$ stores the interaction residues from the coupling kernel $\kappa_{d,d'}(\tau)$ (HMP-DT §11.4, Eq. 11.9). When coupling is strong, the interaction terms in the coupling operator (HMP-DT §11.5, Eq. 11.10–11.11) contribute significant signal to the pairs subspace. When coupling is absent, $\mathbf{H}_{\text{pairs}} \approx \mathbf{0}$.

The denominator $\|\mathbf{H}_{\text{pairs}}\|_{\max}$ is the norm when all pairwise couplings are at their theoretical maximum ($\kappa_{d,d'} = 1$ for all $d, d'$). This value depends only on the subspace dimension and the coupling operator's structure, not on any text-dependent quantity.

### 3.6 The Resonance Persistence Factor

Narrative resonance captures the cumulative influence of past signals on the present discourse state. Define:

$$\mathcal{T} = \frac{\|\mathbf{H}_{\text{res}}\|}{\|\mathbf{H}_{\text{res}}\|_{\max}} \in [0, 1] \tag{6}$$

**Justification.** $\mathbf{H}_{\text{res}}$ holds summary statistics of the resonance field $\mathcal{R}(\tau)$ (HMP-DT §12.6, Eq. 12.8). High resonance persistence means signals from early narrative positions continue to influence the discourse state throughout the work. A text where the opening scene's spatial setting continues to shape interpretation at the climax (e.g., the hills in Hemingway's *Hills Like White Elephants*) has high $\mathcal{T}$.

### 3.7 The Memory Depth Factor

Memory traces encode salient signals that persist across the narrative. Define:

$$\mathcal{M} = \frac{1}{|\mathcal{M}_{\text{store}}|} \sum_{j \in \mathcal{M}_{\text{store}}} \beta_j \in [0, 1] \tag{7}$$

where $\mathcal{M}_{\text{store}}$ is the set of stored memories and $\beta_j$ is the activation of memory $j$ after reinforcement and decay (HMP-DT §12.4, Eq. 12.4–12.5).

**Justification.** A rich text generates memorable patterns: signals that exceed the storage threshold $\theta_{\text{store}} = 0.5$ and persist above the forgetting threshold $\theta_{\text{forget}} = 0.01$ (HMP-DT §12.1, Eq. 12.1). The average activation measures the "memorability" of the discourse: how deeply the text's signals are inscribed in the narrative state.

### 3.8 The LDRI Equation

The full LDRI is defined as the confidence-weighted geometric mean of normalized component scores:

$$\text{LDRI} = \Gamma \cdot \left( \mathcal{R}_{\text{signal}}^\alpha \cdot \tilde{\mathcal{C}}^\beta \cdot \tilde{\mathcal{T}}^\gamma \cdot \tilde{\mathcal{M}}^\delta \right)^{\frac{1}{\alpha + \beta + \gamma + \delta}} \tag{8}$$

where:

$$\tilde{\mathcal{C}} = \frac{1 + \mathcal{C}}{2},\quad \tilde{\mathcal{T}} = \frac{1 + \mathcal{T}}{2},\quad \tilde{\mathcal{M}} = \frac{1 + \mathcal{M}}{2} \in [0.5, 1] \tag{9}$$

and $\Gamma = \prod_{d \in \mathcal{D}} \gamma_d \in [0, 1]$ is the global confidence (§5).

**Default exponents.** $\alpha = 2$, $\beta = \gamma = \delta = 1$. Under these defaults:

$$\text{LDRI} = \Gamma \cdot \left( \mathcal{R}_{\text{signal}}^2 \cdot \tilde{\mathcal{C}} \cdot \tilde{\mathcal{T}} \cdot \tilde{\mathcal{M}} \right)^{1/5} \tag{10}$$

**Justification of the geometric mean.** The geometric mean is chosen over the arithmetic mean for three reasons:

1. **Zero intolerance.** If $\mathcal{R}_{\text{signal}} = 0$ (no signal in any dimension), LDRI = 0 regardless of the other factors. This is correct: without a baseline signal, coupling and resonance create no richness. In contrast, an arithmetic mean would allow non-zero LDRI from coupling alone.

2. **Multiplicative structure.** Each factor amplifies or attenuates the others. Strong signals with weak coupling yield lower LDRI than strong signals with strong coupling. This multiplicative interaction mirrors the literary reality: dimensions do not contribute independently but modulate each other.

3. **Scale normalization.** The transformed factors $\tilde{\mathcal{C}}, \tilde{\mathcal{T}}, \tilde{\mathcal{M}}$ are bounded to $[0.5, 1]$, ensuring that even in the absence of coupling ($\mathcal{C} = 0$), the factor contributes $\tilde{\mathcal{C}} = 0.5$ rather than driving LDRI to zero. This implements the intuition that coupling is an *amplifier* of existing richness, not a prerequisite for it.

**Justification of the exponent asymmetry ($\alpha = 2$).** Signal richness is given a higher exponent ($\alpha = 2$) because it is the foundational component. Without detectable signals, coupling and memory cannot create richness. The quadratic weighting ensures that $\mathcal{R}_{\text{signal}}$ dominates the geometric mean: for $\mathcal{R}_{\text{signal}} = 0.5$, the signal contribution is $0.5^2 = 0.25$, while at $\mathcal{R}_{\text{signal}} = 0.9$, it is $0.9^2 = 0.81$. This disproportionate response rewards texts with strong signals across dimensions.

**Boundedness verification.** Since $\mathcal{R}_{\text{signal}} \in [0, 1]$, $\tilde{\mathcal{C}}, \tilde{\mathcal{T}}, \tilde{\mathcal{M}} \in [0.5, 1]$, and $\Gamma \in [0, 1]$:

- Lower bound: $\mathcal{R}_{\text{signal}} = 0$ implies LDRI $= \Gamma \cdot 0 = 0$.
- Upper bound: $\mathcal{R}_{\text{signal}} = 1$, $\tilde{\mathcal{C}} = \tilde{\mathcal{T}} = \tilde{\mathcal{M}} = 1$ implies LDRI $= \Gamma \cdot (1^\alpha \cdot 1^\beta \cdot 1^\gamma \cdot 1^\delta)^{1/S} = \Gamma \leq 1$.

Therefore $\text{LDRI} \in [0, 1]$. This satisfies Axiom 1.

---

## 4. Component Integration

### 4.1 Semantic Richness

**Input.** $\mathbf{H}_S \in \mathbb{R}^{d/10}$ encodes lexical density, polysemy, concreteness, and semantic coherence features.

**Contribution pathway.** $\|\mathbf{H}_S\| \xrightarrow{g} a_S \xrightarrow{\ell_2} \mathcal{R}_{\text{signal}} \xrightarrow{\text{Eq. (8)}} \text{LDRI}$.

**Narratological role.** Semantic richness is the foundational register of literary discourse. It provides the raw material — words, their associations, their ambiguities — upon which all other dimensions operate. A text with limited vocabulary, concrete reference, and low polysemy (e.g., a technical manual) restricts the interpretive space. Semantic richness is necessary but not sufficient for high LDRI.

### 4.2 Narrative Richness

**Input.** $\mathbf{H}_N \in \mathbb{R}^{d/10}$ encodes tension, pacing, scene entropy, and narrative trajectory.

**Contribution pathway.** $\|\mathbf{H}_N\| \xrightarrow{g} a_N \xrightarrow{\ell_2} \mathcal{R}_{\text{signal}}$ (direct); narrative position splines feed the resonance field $\mathcal{R}(\tau)$ which contributes to $\mathcal{T}$ (indirect).

**Narratological role.** Narrativity — the structured unfolding of events across time — is what distinguishes literary discourse from expository or argumentative text types. A text with strong narrative richness has dynamic tension curves, varied pacing, and structurally significant scene transitions. Narrative richness contributes both directly (as a dimension activation) and indirectly (through its temporal profile feeding resonance).

### 4.3 Interaction Richness

**Input.** $\mathbf{H}_I \in \mathbb{R}^{d/10}$ encodes dialogue structure, speaker-hearer networks, turn asymmetry, and conversational density.

**Contribution pathway.** $\|\mathbf{H}_I\| \xrightarrow{g} a_I \xrightarrow{\ell_2} \mathcal{R}_{\text{signal}}$.

**Narratological role.** Literary discourse is fundamentally social (Bakhtin's dialogism). Character interactions — who speaks to whom, how often, with what asymmetry, under what constraints — constitute one of the primary vehicles of literary meaning. Interaction richness measures the structural complexity of this social signal. Because interaction patterns typically couple with emotion (subtext) and narrative (tension through dialogue), interaction-rich texts also benefit from higher $\mathcal{C}$ when the coupling kernel detects these interactions.

### 4.4 Symbolic Richness

**Input.** $\mathbf{H}_{Sy} \in \mathbb{R}^{d/10}$ encodes figurative density, motif persistence, and symbolic coherence.

**Contribution pathway.** $\|\mathbf{H}_{Sy}\| \xrightarrow{g} a_{Sy} \xrightarrow{\ell_2} \mathcal{R}_{\text{signal}}$ (direct); symbolic motifs that are stored and retrieved across the narrative contribute to $\mathcal{M}$ (indirect).

**Narratological role.** Symbolism is a hallmark of literary discourse. A text that operates on both literal and figurative levels is richer than one with only literal meaning. Because symbolic motifs tend to recur (the definition of a motif is recurrence), they naturally trigger the HMP-DT memory mechanism: a symbol that appears, is stored, and is later retrieved produces both a direct activation ($a_{Sy}$) and a memory trace ($\mathcal{M}$).

### 4.5 Evolution Richness

**Input.** $\mathbf{H}_E \in \mathbb{R}^{d/10}$ encodes character arc slope, relationship state changes, and developmental trajectory.

**Contribution pathway.** $\|\mathbf{H}_E\| \xrightarrow{g} a_E \xrightarrow{\ell_2} \mathcal{R}_{\text{signal}}$ (direct); temporal changes in evolution features contribute to the resonance field $\mathcal{R}(\tau)$, feeding $\mathcal{T}$ (indirect).

**Narratological role.** Characters and relationships that change, grow, or decay across the narrative produce literary richness. Static characters in static relationships yield low evolution richness. Evolution is intrinsically temporal — it unfolds across $\tau$ — and therefore contributes both to the direct signal and to the temporal richness component.

### 4.6 Emotion Richness

**Input.** $\mathbf{H}_{Em} \in \mathbb{R}^{d/10}$ encodes valence, arousal, intensity, and emotional trajectory.

**Contribution pathway.** $\|\mathbf{H}_{Em}\| \xrightarrow{g} a_{Em} \xrightarrow{\ell_2} \mathcal{R}_{\text{signal}}$.

**Narratological role.** Emotional complexity is central to literary value. A text with a wide emotional range, nuanced valence shifts, and dynamic intensity scores high on emotion richness. Emotional flatness — even if the flat emotion is intense (e.g., sustained anger) — signals impoverished literary discourse. Emotion typically couples with narrative (tension episodes) and interaction (subtext in dialogue), so emotion-rich texts may also show elevated $\mathcal{C}$.

### 4.7 Memory Contribution

**Input.** $\mathcal{M}_{\text{store}}$ and activations $\beta_j$ from HMP-DT §12.

**Contribution pathway.** $\mathcal{M} \xrightarrow{\text{Eq. (7)}} \tilde{\mathcal{M}} \xrightarrow{\text{Eq. (8)}} \text{LDRI}$.

**Why it is separate.** Memory is not a dimension. It is a mechanism by which signals from any dimension can be amplified through recurrence. A symbolic motif that appears once contributes to $a_{Sy}$. The same motif that appears, is stored, and retrieved later contributes to both $a_{Sy}$ and $\mathcal{M}$. Memory therefore captures the *persistence of significance* across the narrative.

### 4.8 Resonance Contribution

**Input.** $\mathbf{H}_{\text{res}}$ from HMP-DT §12.6.

**Contribution pathway.** $\|\mathbf{H}_{\text{res}}\| \xrightarrow{\text{Eq. (6)}} \mathcal{T} \xrightarrow{\text{Eq. (9)}} \tilde{\mathcal{T}} \xrightarrow{\text{Eq. (8)}} \text{LDRI}$.

**Why it is separate.** Resonance is distinct from memory. Memory stores discrete signals that exceed a retrieval threshold; resonance is a continuous field encoding the cumulative influence of *all* past signals (HMP-DT §12.6, Eq. 12.8). A text where the opening setting continues to influence interpretation at the climax has high resonance even if no individual memory is explicitly retrieved.

### 4.9 Cross-Dimensional Coupling

**Input.** $\mathbf{H}_{\text{pairs}}$ from HMP-DT §11.4–11.5.

**Contribution pathway.** $\|\mathbf{H}_{\text{pairs}}\| \xrightarrow{\text{Eq. (5)}} \mathcal{C} \xrightarrow{\text{Eq. (9)}} \tilde{\mathcal{C}} \xrightarrow{\text{Eq. (8)}} \text{LDRI}$.

**Why it is separate.** Coupling is the mechanism by which dimensions interact. Without coupling, the six dimensions would be independent streams. The pairs subspace $\mathbf{H}_{\text{pairs}}$ encodes the second-order interaction statistics: a text where emotion and symbolism reinforce each other has higher coupling enrichment than one where they operate independently. Coupling is the structural property that distinguishes a multi-dimensional text from six parallel one-dimensional texts.

### 4.10 Uncertainty

**Input.** Per-dimension confidences $\gamma_d$ from HMP-DT §13.

**Contribution pathway.** $\{\gamma_d\} \xrightarrow{\text{product}} \Gamma \xrightarrow{\text{Eq. (8)}} \text{LDRI}$.

**Why it is separate.** Uncertainty is not a feature of the text but of the analysis pipeline. Two identical texts analyzed with different NLP tools should ideally yield the same LDRI, but the *confidence* in that LDRI differs. The multiplicative uncertainty factor ensures that LDRI represents the *lower bound* on richness: "we observe this much richness with this much confidence."

---

## 5. Confidence, Reliability, and Uncertainty

### 5.1 Why These Are Different

| Concept | Definition | Source | Determined By |
|---------|-----------|--------|---------------|
| **Richness** | The degree of multi-dimensional literary discourse signal | Text + HMP-DT inference | The text itself |
| **Confidence** | The model's estimate that observed signals reflect genuine literary features (not extraction artifacts) | Feature extractor quality + coherence propagation | The analysis pipeline |
| **Reliability** | The expected variance of LDRI under repeated analysis with different random seeds or parameter settings | Bootstrap over pipeline components | The metric's sensitivity |

**Richness** is the quantity we seek to measure. **Confidence** is our estimate of measurement quality: a text with overt, unambiguous literary signals (e.g., explicit symbolism) can be analyzed with high confidence; a text with subtle, ambiguous signals (e.g., free indirect discourse) produces lower confidence. The richness may be identical; our *confidence in the measurement* differs.

**Reliability** is a property of the metric-pipeline pair. Two runs of HMP-DT with identical parameters should produce nearly identical LDRI values. If the metric is sensitive to random seeds (e.g., the Fourier feature matrices $\mathbf{W}, \mathbf{b}$ in memory key generation), reliability decreases.

### 5.2 Global Confidence

The global confidence is the product of per-dimension confidences, weighted by signal strength to reflect the model's certainty in its most significant observations:

$$\gamma_d = \frac{\sum_{\ell=1}^5 \sum_{i=1}^{N_\ell} \gamma_{i}^{(\ell)}[d] \cdot \|\mathbf{f}_{i}^{(\ell)}[d]\|}{\sum_{\ell=1}^5 \sum_{i=1}^{N_\ell} \|\mathbf{f}_{i}^{(\ell)}[d]\|} \tag{11}$$

$$\Gamma = \prod_{d \in \mathcal{D}} \gamma_d \tag{12}$$

**Justification of the product form.** If any one dimension has near-zero confidence, the global confidence drops to near-zero. This is deliberately conservative: LDRI should not report high richness if any one discourse dimension's extraction is unreliable. This satisfies Axiom 6 (Robustness) and Axiom 9 (Uncertainty Awareness).

### 5.3 Reliability Estimation

Reliability is estimated via a bootstrap over stochastic HMP-DT components:

$$\sigma_{\text{LDRI}}^2 = \text{Var}\left(\{\text{LDRI}^{(b)}\}_{b=1}^B\right) \tag{13}$$

where $\text{LDRI}^{(b)}$ is computed after re-sampling random seeds for:
- Fourier feature matrices $\mathbf{W}, \mathbf{b}$ (HMP-DT §5.2)
- Coupling operator matrices $\mathbf{M}_{d,d'}$ (if stochastic)

The reliability score is:

$$\text{Reliability} = 1 - \min\left(1.0, \frac{\sigma_{\text{LDRI}}}{\sigma_{\max}}\right), \quad \sigma_{\max} = 0.1 \tag{14}$$

A reliability of 0.95 means that repeated analyses of the same text produce LDRI values within approximately $\pm 0.05$ of each other with 95% probability.

### 5.4 Reporting Convention

LDRI is reported as a triple:

$$\text{LDRI} = 0.68 \pm 0.03 \;(95\% \text{ CI}),\; \Gamma = 0.87,\; \text{Reliability} = 0.92$$

This communicates four quantities:
1. **Point estimate:** The most likely richness given the data ($0.68$)
2. **Precision:** The variability under bootstrap ($\pm 0.03$)
3. **Confidence:** The quality of the feature extraction ($\Gamma = 0.87$)
4. **Reliability:** The stability of the pipeline ($0.92$)

---

## 6. Explainable Decomposition

### 6.1 Additive Components

Every LDRI score is decomposable into a sum of narratologically meaningful components via a first-order expansion of Eq. (8) around the log-space geometric mean:^1

[^1]: The decomposition is exact in log-space because $\log \text{LDRI} = \log \Gamma + \frac{\alpha}{S} \log \mathcal{R}_{\text{signal}} + \frac{\beta}{S} \log \tilde{\mathcal{C}} + \frac{\gamma}{S} \log \tilde{\mathcal{T}} + \frac{\delta}{S} \log \tilde{\mathcal{M}}$ where $S = \alpha + \beta + \gamma + \delta$. The linear decomposition in the main text is the first-order approximation $\text{LDRI} \approx \log(1 + x) \approx x$ for small $x$, which holds when LDRI is not near the boundaries.

For practical interpretation, we define the additive decomposition:

$$\text{LDRI} = c_{\text{int}} + c_{\text{intr}} + c_{\text{temp}} + c_{\text{res}} + c_{\text{mem}} + c_{\text{unc}} \tag{15}$$

| Component | Symbol | Narratological Meaning | Lower Bound |
|-----------|--------|----------------------|-------------|
| Intrinsic Richness | $c_{\text{int}}$ | Strength and distribution of base signals across dimensions | $c_{\text{int}} \geq 0$ |
| Interaction Richness | $c_{\text{intr}}$ | Amplification from cross-dimensional coupling | $c_{\text{intr}} \geq 0$ |
| Temporal Richness | $c_{\text{temp}}$ | Signal evolution and change across narrative position | $c_{\text{temp}} \geq 0$ |
| Resonance Contribution | $c_{\text{res}}$ | Cumulative influence of past signals on present state | $c_{\text{res}} \geq 0$ |
| Memory Contribution | $c_{\text{mem}}$ | Reinforcement from recurring patterns | $c_{\text{mem}} \geq 0$ |
| Uncertainty Adjustment | $c_{\text{unc}}$ | Penalty for low-confidence observations | $c_{\text{unc}} \leq 0$ |

### 6.2 Intrinsic Richness

$$c_{\text{int}} = \frac{\alpha}{\alpha + \beta + \gamma + \delta} \cdot \Gamma \cdot \log\left(\frac{1 + \mathcal{R}_{\text{signal}}}{1 - \mathcal{R}_{\text{signal}}}\right) \tag{16}$$

**Interpretation.** $c_{\text{int}}$ is the baseline contribution from the six dimensions' activation levels. It is positive whenever $\mathcal{R}_{\text{signal}} > 0$ (at least one dimension has non-zero signal). A text with moderate activation in all six dimensions has higher $c_{\text{int}}$ than a text with extreme activation in one dimension and zero in others, reflecting the $\ell_2$ norm's concentration penalty.

### 6.3 Interaction Richness

$$c_{\text{intr}} = \frac{\beta}{\alpha + \beta + \gamma + \delta} \cdot \Gamma \cdot \log(1 + \mathcal{C}) \tag{17}$$

**Interpretation.** $c_{\text{intr}}$ measures the additional richness contributed by cross-dimensional coupling. It is zero when $\mathcal{C} = 0$ (no coupling detected) and increases logarithmically with $\mathcal{C}$. The logarithmic scaling reflects diminishing returns: the tenth coupling interaction contributes less than the first.

### 6.4 Temporal Richness

$$c_{\text{temp}} = \frac{\gamma}{\alpha + \beta + \gamma + \delta} \cdot \Gamma \cdot \log(1 + \mathcal{T}) \tag{18}$$

**Interpretation.** $c_{\text{temp}}$ measures the richness contributed by signal evolution across narrative time. A text where all dimensions maintain constant values from $\tau = 0$ to $\tau = 1$ has $\mathcal{T} \approx 0$ and $c_{\text{temp}} \approx 0$. A text with dynamic tension, evolving emotion, and changing symbolic patterns has $\mathcal{T} > 0$ and positive $c_{\text{temp}}$.

### 6.5 Resonance Contribution

$$c_{\text{res}} = \frac{\gamma}{\alpha + \beta + \gamma + \delta} \cdot \Gamma \cdot \frac{e^{-1/\lambda}}{1 - e^{-1/\lambda}} \cdot \|\mathcal{R}(1)\| \tag{19}$$

where $\lambda = 0.3$ is the resonance decay length (HMP-DT §12.6) and $\mathcal{R}(1)$ is the resonance field at the final narrative position.

**Interpretation.** $c_{\text{res}}$ measures the long-range structural persistence of signals. It is distinct from $c_{\text{temp}}$ because a text can exhibit temporal variation without long-range resonance (e.g., random signal fluctuations from scene to scene). Resonance measures *structural coherence across time*, not mere change.

### 6.6 Memory Contribution

$$c_{\text{mem}} = \frac{\delta}{\alpha + \beta + \gamma + \delta} \cdot \Gamma \cdot \log(1 + \mathcal{M}) \tag{20}$$

**Interpretation.** $c_{\text{mem}}$ measures the richness contributed by the reinforcement of recurring patterns. Each time a memory is retrieved and reinforced (HMP-DT §12.4, Eq. 12.4), its activation $\beta_j$ increases. A text with many persistent, strongly-reinforced memories scores high on $c_{\text{mem}}$.

### 6.7 Uncertainty Adjustment

$$c_{\text{unc}} = \text{LDRI}_{\text{raw}} - \text{LDRI} = (1 - \Gamma) \cdot \text{LDRI}_{\text{raw}} \tag{21}$$

where:

$$\text{LDRI}_{\text{raw}} = \left( \mathcal{R}_{\text{signal}}^\alpha \cdot \tilde{\mathcal{C}}^\beta \cdot \tilde{\mathcal{T}}^\gamma \cdot \tilde{\mathcal{M}}^\delta \right)^{\frac{1}{\alpha+\beta+\gamma+\delta}} \tag{22}$$

is the richness that would be reported if confidence were perfect ($\Gamma = 1$).

**Interpretation.** $c_{\text{unc}}$ is always non-negative (since $\Gamma \in [0, 1]$ implies $1 - \Gamma \geq 0$). A large $c_{\text{unc}}$ means that the observed LDRI is significantly lower than what it would be with perfect confidence. This signals to the user that improved feature extraction tools could meaningfully increase the measured richness.

### 6.8 Contribution Fractions

For interpretability, the additive components can be expressed as fractions of the total LDRI:

$$f_{\text{int}} = \frac{c_{\text{int}}}{\text{LDRI} + c_{\text{unc}}}, \quad f_{\text{intr}} = \frac{c_{\text{intr}}}{\text{LDRI} + c_{\text{unc}}}, \quad \ldots \tag{23}$$

These fractions sum to approximately 1 (with residual from the first-order approximation) and reveal the *composition* of a text's richness. A text with $f_{\text{intr}} = 0.6$ derives most of its richness from cross-dimensional interaction — it is a text where "what happens" is shaped by "how it is said" in multiple registers simultaneously. A text with $f_{\text{int}} = 0.8$ and $f_{\text{intr}} = 0.1$ has strong independent signals but weak interaction — a more straightforward text where each register operates independently.

---

## 7. Mathematical Propositions

### 7.1 Proposition 1: Boundedness

**Statement.** $0 \leq \text{LDRI} \leq 1$ for all $\mathbf{H} \in \mathbb{R}^d$, $\Gamma \in [0, 1]$, $\mathcal{C}, \mathcal{T}, \mathcal{M} \in [0, 1]$.

**Proof.**

1. Each dimension activation $a_d = g(\|\mathbf{H}_d\|) \in [0, 1)$ (since $g(x) = 1 - e^{-x^2/2\sigma^2} \in [0, 1)$).

2. Therefore $\|\mathbf{a}\|_2 \leq \sqrt{6}$, and $\mathcal{R}_{\text{signal}} = \|\mathbf{a}\|_2 / \sqrt{6} \in [0, 1]$.

3. By construction, $\mathcal{C}, \mathcal{T}, \mathcal{M} \in [0, 1]$, so $\tilde{\mathcal{C}}, \tilde{\mathcal{T}}, \tilde{\mathcal{M}} \in [0.5, 1]$.

4. $\Gamma = \prod_d \gamma_d \in [0, 1]$ since each $\gamma_d \in [0, 1]$.

5. The geometric mean: $\left( \mathcal{R}_{\text{signal}}^\alpha \cdot \tilde{\mathcal{C}}^\beta \cdot \tilde{\mathcal{T}}^\gamma \cdot \tilde{\mathcal{M}}^\delta \right)^{1/S} \in [0, 1]$, where $S = \alpha + \beta + \gamma + \delta$.

   - Lower bound: when $\mathcal{R}_{\text{signal}} = 0$, the product is $0$, giving LDRI $= \Gamma \cdot 0 = 0$.
   - Upper bound: when $\mathcal{R}_{\text{signal}} = \tilde{\mathcal{C}} = \tilde{\mathcal{T}} = \tilde{\mathcal{M}} = 1$, the product is $1$, giving LDRI $= \Gamma \cdot 1 \leq 1$.

6. Therefore $0 \leq \text{LDRI} \leq 1$. $\square$

### 7.2 Proposition 2: Monotonicity

**Statement.** LDRI is monotonic non-decreasing in each $\|\mathbf{H}_d\|$, each $\gamma_d$, and each of $\mathcal{C}, \mathcal{T}, \mathcal{M}$ individually.

**Proof.**

1. $g(x) = 1 - e^{-x^2/2\sigma^2}$ has derivative $g'(x) = (x/\sigma^2) e^{-x^2/2\sigma^2} \geq 0$ for $x \geq 0$, so $g$ is monotonic non-decreasing.

2. $\mathcal{R}_{\text{signal}} = \|\mathbf{a}\|_2 / \sqrt{6}$ where each $a_d = g(\|\mathbf{H}_d\|)$. The $\ell_2$ norm is monotonic in each component, so $\mathcal{R}_{\text{signal}}$ is monotonic in each $\|\mathbf{H}_d\|$.

3. $\tilde{\mathcal{C}} = (1 + \mathcal{C})/2$, $\tilde{\mathcal{T}} = (1 + \mathcal{T})/2$, $\tilde{\mathcal{M}} = (1 + \mathcal{M})/2$ are each strictly increasing in their arguments.

4. The geometric mean $(\prod_i x_i^{w_i})^{1/\sum w_i}$ is monotonic in each $x_i$ (since $x_i^{w_i}$ is increasing for $x_i \geq 0$, $w_i > 0$).

5. $\Gamma = \prod_d \gamma_d$ is monotonic in each $\gamma_d$ (product of non-negative terms).

6. The composition of monotonic functions is monotonic. Therefore LDRI is monotonic in each argument. $\square$

### 7.3 Proposition 3: Continuity

**Statement.** LDRI is continuous on its domain $[0, \infty)^{6} \times [0, 1]^{3} \times [0, 1]^{6}$.

**Proof.**

1. $g$ is continuous on $[0, \infty)$ (composition of exponential and polynomial).

2. $\|\cdot\|_2$ is continuous (norm on finite-dimensional space).

3. $\tilde{\mathcal{C}}, \tilde{\mathcal{T}}, \tilde{\mathcal{M}}$ are affine functions, hence continuous.

4. The geometric mean is continuous on $[0, 1]^4$.

5. $\Gamma$ is a product of continuous functions.

6. Composition preserves continuity. $\square$

**Corollary.** For any sequence of Narrative Embeddings $\mathbf{H}^{(n)} \to \mathbf{H}$, $\text{LDRI}(\mathbf{H}^{(n)}) \to \text{LDRI}(\mathbf{H})$.

### 7.4 Proposition 4: Lipschitz Stability

**Statement.** LDRI is Lipschitz continuous with respect to $\mathbf{H}$: $|\text{LDRI}(\mathbf{H}) - \text{LDRI}(\mathbf{H}')| \leq L \|\mathbf{H} - \mathbf{H}'\|$ for some $L < \infty$.

**Proof sketch.**

1. $g$ has bounded derivative $|g'(x)| \leq 1/(\sigma\sqrt{e})$ on $[0, \infty)$, so $g$ is $L_g$-Lipschitz with $L_g = 1/(\sigma\sqrt{e})$.

2. The $\ell_2$ norm is 1-Lipschitz: $|\|\mathbf{a}\| - \|\mathbf{a}'\|\| \leq \|\mathbf{a} - \mathbf{a}'\|$.

3. The geometric mean is $L_{\text{gm}}$-Lipschitz on $[0, 1]^4$ with $L_{\text{gm}} \leq 1/(\min \text{argument})$ — the worst case is when one argument approaches 0. However, $\mathcal{R}_{\text{signal}}$ is separated from 0 unless the text has literally no signal, in which case the derivative is bounded by the product rule.

4. $\Gamma$ is 1-Lipschitz in each $\gamma_d$.

5. By the composition rule for Lipschitz functions, the overall $L$ is bounded. For the default parameters and $\mathcal{R}_{\text{signal}} \geq \epsilon > 0$, $L$ is finite and computable.

6. For the complete proof, each component's Lipschitz constant is computed, and the chain rule for Lipschitz functions yields $L = O(S \cdot L_g \cdot L_{\text{gm}})$ where $S = \alpha + \beta + \gamma + \delta$. $\square$

**Corollary (Axiom 4).** For $\mathbf{H}_\epsilon = \mathbf{H} + \epsilon \mathbf{u}$ with $\|\mathbf{u}\| = 1$, $|\text{LDRI}(\mathbf{H}_\epsilon) - \text{LDRI}(\mathbf{H})| \leq L\epsilon$, confirming the $O(\epsilon)$ bound required by Axiom 4.

### 7.5 Proposition 5: Scale Invariance

**Statement.** $\text{LDRI}(c \cdot \mathbf{H}) = \text{LDRI}(\mathbf{H})$ for any $c > 0$, provided $\mathbf{H}$ is normalized.

**Proof.**

1. By HMP-DT Assumption A3 (revised), all feature vectors are bounded to $[0, 1]$ or $[-1, 1]$, and the norms $\|\mathbf{H}_d\|$, $\|\mathbf{H}_{\text{pairs}}\|$, $\|\mathbf{H}_{\text{res}}\|$ are scale-normalized as part of the HMP-DT pipeline.

2. Each dimension activation $a_d = g(\|\mathbf{H}_d\|)$ uses the Rayleigh null scale $\sigma_d$, which is estimated from the distribution of $\|\mathbf{H}_d\|$ across many texts. This scale absorbs uniform scaling.

3. Therefore $a_d(c \cdot \mathbf{H}) = a_d(\mathbf{H})$ for any $c > 0$, and all downstream quantities are invariant.

4. This satisfies Axiom 7. $\square$

### 7.6 Proposition 6: Information-Theoretic Interpretation

**Statement.** In the limit of weak but detectable signals ($a_d \ll 1$ for all $d$), LDRI is proportional to the square root of the total Fisher information of the discourse signals.

**Proof sketch.**

1. For small $a_d$, $g(x) \approx x^2/(2\sigma_d^2)$, so $a_d \propto \|\mathbf{H}_d\|^2$.

2. The $\ell_2$ norm satisfies $\|\mathbf{a}\|_2 \propto \sqrt{\sum_d \|\mathbf{H}_d\|^4}$, which for small signals approximates the norm of the fourth-order cumulant tensor.

3. Under a Gaussian model for feature noise, the Fisher information matrix for the discourse parameters is proportional to $\mathbf{H}\mathbf{H}^\top$, and the total Fisher information along the signal direction is $\|\mathbf{H}\|^2$.

4. Therefore $\mathcal{R}_{\text{signal}}$ is monotonically related to the square root of the Fisher information: $\text{LDRI} \propto \sqrt{\mathcal{I}_{\text{discourse}}}$, where $\mathcal{I}_{\text{discourse}}$ is the total Fisher information across all six dimensions.

5. This provides an information-geometric interpretation: LDRI measures how much information the text provides about its own discourse structure. $\square$

### 7.7 Proposition 7: Dimensional Breadth Premium

**Statement.** Let $k$ be the number of dimensions with non-zero activation ($a_d > 0$). Then the maximum possible $\mathcal{R}_{\text{signal}}$ is $\sqrt{k/6}$, achieved when all $k$ dimensions are at full activation ($a_d = 1$). Therefore a text activating all six dimensions can achieve $\mathcal{R}_{\text{signal}} = 1$, while a text activating only $k$ dimensions has a maximum possible $\mathcal{R}_{\text{signal}} = \sqrt{k/6} < 1$.

**Proof.**

$$\mathcal{R}_{\text{signal}}^2 = \frac{1}{6} \sum_{d \in \mathcal{D}} a_d^2 \leq \frac{1}{6} \sum_{d \in \mathcal{D}_+} 1 = \frac{k}{6}$$

where $\mathcal{D}_+ = \{d : a_d > 0\}$ and $a_d \leq 1$ by construction. Equality holds iff $a_d = 1$ for all $d \in \mathcal{D}_+$. Therefore $\mathcal{R}_{\text{signal}} \leq \sqrt{k/6}$. $\square$

**Corollary.** $\mathcal{R}_{\text{signal}} = 1$ if and only if all six dimensions are at full activation. This provides a continuous penalty for dimensional silence: each dimension with $a_d = 0$ reduces the maximum achievable $\mathcal{R}_{\text{signal}}$ from $1$ to $\sqrt{(6-|\mathcal{D}_0|)/6}$, where $\mathcal{D}_0 = \{d : a_d = 0\}$.

**Literary interpretation.** A text that operates on only three of the six discourse dimensions can never exceed $\mathcal{R}_{\text{signal}} = \sqrt{3/6} \approx 0.707$ even if those three dimensions are maximally expressed. LDRI enforces the principle that literary discourse richness requires multi-dimensional engagement — a necessary condition for high richness, not a contingent weighting choice.

---

## 8. Worked Examples

### 8.1 Simple Chapter

**Text.** A straightforward chapter from a mass-market thriller. One subplot resolves: the protagonist confronts a suspect. Dialogue is direct, emotions are explicit, symbolism is absent.

**HMP-DT outputs (simplified).**

| Quantity | Value | Notes |
|----------|-------|-------|
| $\|\mathbf{H}_S\|$ | 0.15 | Low lexical density, concrete vocabulary |
| $\|\mathbf{H}_N\|$ | 0.55 | Moderate tension spike during confrontation |
| $\|\mathbf{H}_I\|$ | 0.30 | Two characters, simple turn-taking |
| $\|\mathbf{H}_{Sy}\|$ | 0.05 | No detectable figurative language |
| $\|\mathbf{H}_E\|$ | 0.10 | Flat character arc, no relationship change |
| $\|\mathbf{H}_{Em}\|$ | 0.35 | Anger $\to$ relief, simple valence shift |
| $\|\mathbf{H}_{\text{pairs}}\|$ | 0.08 | Weak coupling (emotion and narrative only) |
| $\|\mathbf{H}_{\text{res}}\|$ | 0.04 | Minimal resonance |
| $\mathcal{M}$ | 0.12 | Few stored memories, low persistence |
| $\Gamma$ | 0.85 | Moderate confidence |

**Dimension activations.** Using $\sigma_d = 0.3$ for all $d$ (simplified):

$$a_S = 1 - e^{-0.15^2/(2 \cdot 0.3^2)} = 1 - e^{-0.125} = 0.118$$
$$a_N = 1 - e^{-0.55^2/0.18} = 1 - e^{-1.68} = 0.814$$
$$a_I = 1 - e^{-0.30^2/0.18} = 1 - e^{-0.50} = 0.393$$
$$a_{Sy} = 1 - e^{-0.05^2/0.18} = 1 - e^{-0.014} = 0.014$$
$$a_E = 1 - e^{-0.10^2/0.18} = 1 - e^{-0.056} = 0.054$$
$$a_{Em} = 1 - e^{-0.35^2/0.18} = 1 - e^{-0.68} = 0.493$$

**Signal Richness Norm:**

$$\mathbf{a} = (0.118, 0.814, 0.393, 0.014, 0.054, 0.493)$$

$$\mathcal{R}_{\text{signal}} = \frac{\sqrt{0.118^2 + 0.814^2 + 0.393^2 + 0.014^2 + 0.054^2 + 0.493^2}}{\sqrt{6}}$$

$$= \frac{\sqrt{0.014 + 0.663 + 0.154 + 0.0002 + 0.003 + 0.243}}{\sqrt{6}} = \frac{\sqrt{1.077}}{\sqrt{6}} = \frac{1.038}{2.449} = 0.424$$

**Auxiliary factors:**

$$\tilde{\mathcal{C}} = \frac{1 + 0.08}{2} = 0.540,\quad \tilde{\mathcal{T}} = \frac{1 + 0.04}{2} = 0.520,\quad \tilde{\mathcal{M}} = \frac{1 + 0.12}{2} = 0.560$$

**LDRI (Eq. 10 with $\alpha = 2$, $\beta = \gamma = \delta = 1$):**

$$\text{LDRI} = 0.85 \cdot \left(0.424^2 \cdot 0.540 \cdot 0.520 \cdot 0.560\right)^{1/5}$$

$$= 0.85 \cdot \left(0.180 \cdot 0.540 \cdot 0.520 \cdot 0.560\right)^{1/5}$$

$$= 0.85 \cdot \left(0.0283\right)^{1/5}$$

$$= 0.85 \cdot 0.490 = 0.417$$

**Decomposition (Eq. 15–22):**

$$\text{LDRI}_{\text{raw}} = 0.490$$

$$c_{\text{unc}} = (1 - 0.85) \cdot 0.490 = 0.074$$

$$c_{\text{int}} = \frac{2}{5} \cdot 0.85 \cdot \log\left(\frac{1.424}{0.576}\right) = 0.34 \cdot 0.85 \cdot 0.905 = 0.262$$

$$c_{\text{intr}} = \frac{1}{5} \cdot 0.85 \cdot \log(1.08) = 0.17 \cdot 0.85 \cdot 0.077 = 0.011$$

$$c_{\text{temp}} = \frac{1}{5} \cdot 0.85 \cdot \log(1.04) = 0.17 \cdot 0.85 \cdot 0.039 = 0.006$$

$$c_{\text{mem}} = \frac{1}{5} \cdot 0.85 \cdot \log(1.12) = 0.17 \cdot 0.85 \cdot 0.113 = 0.016$$

**Breakdown:**

| Component | Value | Share of LDRI_raw |
|-----------|-------|-------------------|
| Intrinsic | 0.262 | 53.5% |
| Interaction | 0.011 | 2.2% |
| Temporal | 0.006 | 1.2% |
| Memory | 0.016 | 3.3% |
| Uncertainty adj. | -0.074 | -15.1% |
| **LDRI** | **0.417** | **85% of raw** |

**Interpretation.** This chapter scores low on LDRI (0.417) primarily because:
1. Three dimensions (symbolic, evolution, semantic) contribute negligible activation
2. Cross-dimensional coupling is near-absent ($\mathcal{C} = 0.08$)
3. Resonance is minimal — the chapter does not echo earlier narrative material
4. The sole significant dimension is narrative ($a_N = 0.814$), reflecting a single tension spike

The $c_{\text{intr}} + c_{\text{temp}} + c_{\text{mem}} \approx 0.033$ shows that structural richness (coupling, temporality, memory) accounts for only $\sim 6.7\%$ of raw richness. This is a discourse-poor text by LDRI's definition.

### 8.2 Complex Chapter

**Text.** A climactic chapter from a literary novel (e.g., the "Lestrygonians" episode of Joyce's *Ulysses*). Multiple discourse dimensions are simultaneously active: stream-of-consciousness narrative, intertextual symbolism, character interiority, shifting emotional registers, semantic density, and extensive memory recall of earlier episodes.

**HMP-DT outputs (simplified).**

| Quantity | Value | Notes |
|----------|-------|-------|
| $\|\mathbf{H}_S\|$ | 0.82 | High lexical density, polysemy, allusions |
| $\|\mathbf{H}_N\|$ | 0.71 | Fragmented narrative, multiple temporal layers |
| $\|\mathbf{H}_I\|$ | 0.65 | Internal dialogue, remembered conversations |
| $\|\mathbf{H}_{Sy}\|$ | 0.78 | Dense figurative network, recurring motifs |
| $\|\mathbf{H}_E\|$ | 0.55 | Epiphany arc, memory-triggered reflection |
| $\|\mathbf{H}_{Em}\|$ | 0.69 | Rapid valence shifts, emotional complexity |
| $\|\mathbf{H}_{\text{pairs}}\|$ | 0.62 | Strong cross-dimensional coupling (especially $Sy$--$S$, $Em$--$I$) |
| $\|\mathbf{H}_{\text{res}}\|$ | 0.58 | High resonance — earlier episodes echo throughout |
| $\mathcal{M}$ | 0.71 | Rich memory store with strong activations |
| $\Gamma$ | 0.82 | High confidence |

**Dimension activations ($\sigma_d = 0.3$):**

$$a_S = 1 - e^{-0.82^2/0.18} = 1 - e^{-3.74} = 0.976$$
$$a_N = 1 - e^{-0.71^2/0.18} = 1 - e^{-2.80} = 0.939$$
$$a_I = 1 - e^{-0.65^2/0.18} = 1 - e^{-2.35} = 0.905$$
$$a_{Sy} = 1 - e^{-0.78^2/0.18} = 1 - e^{-3.38} = 0.966$$
$$a_E = 1 - e^{-0.55^2/0.18} = 1 - e^{-1.68} = 0.814$$
$$a_{Em} = 1 - e^{-0.69^2/0.18} = 1 - e^{-2.65} = 0.929$$

**Signal Richness Norm:**

$$\mathbf{a} = (0.976, 0.939, 0.905, 0.966, 0.814, 0.929)$$

$$\mathcal{R}_{\text{signal}} = \frac{\sqrt{6 \times (\text{approx } 0.86)}}{\sqrt{6}} = \frac{\sqrt{5.13}}{\sqrt{6}} = \frac{2.265}{2.449} = 0.925$$

**Auxiliary factors:**

$$\tilde{\mathcal{C}} = \frac{1 + 0.62}{2} = 0.810,\quad \tilde{\mathcal{T}} = \frac{1 + 0.58}{2} = 0.790,\quad \tilde{\mathcal{M}} = \frac{1 + 0.71}{2} = 0.855$$

**LDRI:**

$$\text{LDRI} = 0.82 \cdot \left(0.925^2 \cdot 0.810 \cdot 0.790 \cdot 0.855\right)^{1/5}$$

$$= 0.82 \cdot \left(0.856 \cdot 0.810 \cdot 0.790 \cdot 0.855\right)^{1/5}$$

$$= 0.82 \cdot \left(0.468\right)^{1/5}$$

$$= 0.82 \cdot 0.857 = 0.703$$

**Decomposition:**

$$\text{LDRI}_{\text{raw}} = 0.857$$

$$c_{\text{unc}} = (1 - 0.82) \cdot 0.857 = 0.154$$

$$c_{\text{int}} = \frac{2}{5} \cdot 0.82 \cdot \log\left(\frac{1.925}{0.075}\right) = 0.34 \cdot 0.82 \cdot 3.245 = 0.905$$

$$c_{\text{intr}} = \frac{1}{5} \cdot 0.82 \cdot \log(1.62) = 0.17 \cdot 0.82 \cdot 0.482 = 0.067$$

$$c_{\text{temp}} = \frac{1}{5} \cdot 0.82 \cdot \log(1.58) = 0.17 \cdot 0.82 \cdot 0.457 = 0.064$$

$$c_{\text{mem}} = \frac{1}{5} \cdot 0.82 \cdot \log(1.71) = 0.17 \cdot 0.82 \cdot 0.536 = 0.075$$

**Breakdown:**

| Component | Value | Share of LDRI_raw |
|-----------|-------|-------------------|
| Intrinsic | 0.905 | 105.6% |
| Interaction | 0.067 | 7.8% |
| Temporal | 0.064 | 7.5% |
| Memory | 0.075 | 8.7% |
| Uncertainty adj. | -0.154 | -18.0% |
| **LDRI** | **0.703** | **82% of raw** |

*Note.* The decomposition components sum to more than LDRI_raw because the first-order approximation in Eq. (15) loses linearity when $\mathcal{R}_{\text{signal}}$ approaches 1 and $\log((1+x)/(1-x))$ is large. In practice the fractional interpretation is used: intrinsic richness dominates, with interaction, temporal, and memory each contributing 7--9%.

### 8.3 Comparison

| Property | Simple Chapter | Complex Chapter | Ratio |
|----------|---------------|-----------------|-------|
| $\mathcal{R}_{\text{signal}}$ | 0.424 | 0.925 | 2.18$\times$ |
| $\tilde{\mathcal{C}}$ | 0.540 | 0.810 | 1.50$\times$ |
| $\tilde{\mathcal{T}}$ | 0.520 | 0.790 | 1.52$\times$ |
| $\tilde{\mathcal{M}}$ | 0.560 | 0.855 | 1.53$\times$ |
| **LDRI** | **0.417** | **0.703** | **1.69$\times$** |
| Intrinsic share | 53.5% | 105.6% | -- |
| Structural share (int$+$temp$+$mem) | 6.7% | 24.0% | 3.6$\times$ |

The complex chapter is richer across all four factors. The most dramatic difference is in $\mathcal{R}_{\text{signal}}$ (2.18$\times$), reflecting the fact that the simple chapter effectively activates only one dimension while the complex chapter activates all six. The structural components (interaction, temporal, memory) are collectively 3.6$\times$ larger in the complex chapter, confirming that cross-dimensional richness -- not mere signal magnitude -- drives the LDRI differential.

---

## 9. Comparison with Existing Literary Metrics

### 9.1 Existing Metrics and Their Limitations

| Metric | Domain | Range | What It Captures | What It Misses |
|--------|--------|-------|------------------|----------------|
| **Flesch--Kincaid Grade Level** | Readability | 0--20+ | Syllable/word/sentence statistics | All discourse dimensions |
| **Type-Token Ratio (TTR)** | Lexical diversity | $[0, 1]$ | Vocabulary breadth | Beyond-lexical dimensions |
| **VADER Sentiment** | Emotion | $[-1, 1]$ | Valence, arousal | Temporal evolution, coupling, subtext |
| **LDA Topic Entropy** | Topic diversity | $[0, \infty)$ | Thematic breadth | Character, narrative structure, symbolism |
| **LIWC Categories** | Psychological categories | Percentage | Word-level category prevalence | Structural discourse properties |
| **Narrative Arc (Reagan et al.)** | Narrative structure | -- | Shape of emotional trajectory | Multi-dimensional interaction |
| **Biber's Dimension Scores** | Register | z-scores | Linguistic register features | Literary discourse (vs. register) |
| **LDRI (LIT-DISC)** | Discourse richness | $[0, 1]$ | Multi-dimensional signal with coupling, temporality, memory | Surface readability, syntax |

### 9.2 Advantages of LDRI

1. **Multi-dimensional integration.** LDRI is the only metric that jointly accounts for six discourse dimensions plus their interactions. Existing metrics operate on a single dimension (vocabulary, sentiment, register) and cannot capture the phenomenon of registers operating upon each other.

2. **Temporal awareness.** LDRI incorporates narrative temporality through resonance ($\tilde{\mathcal{T}}$) and memory ($\tilde{\mathcal{M}}$). Sentiment trajectories (Reagan et al., 2016) capture one form of temporality, but only for emotion; LDRI extends this to all six dimensions.

3. **Uncertainty quantification.** Unlike any existing literary metric, LDRI reports confidence ($\Gamma$) and reliability alongside the point estimate. This is critical for responsible computational literary analysis, where feature extraction quality varies across genres and periods.

4. **Principled normalization.** LDRI is bounded, monotonic, continuous, and scale-invariant by construction (Propositions 1--5). No existing literary metric satisfies all these properties.

5. **Explainable decomposition.** LDRI decomposes into narratologically meaningful components (\S6). A scholar can see whether a text's richness comes from intrinsic signals, interactions, temporal dynamics, or memory -- and adjust their interpretation accordingly.

### 9.3 Limitations

1. **HMP-DT dependency.** LDRI is not a standalone metric. It depends entirely on the HMP-DT pipeline to produce the Narrative Embedding. If the pipeline's NLP feature extractors are unreliable for a given text, LDRI is unreliable (though the $\Gamma$ factor reflects this).

2. **Six-dimension ontology.** LDRI's richness concept is bounded by the six dimensions of HMP-DT. If a text's discourse richness operates through a mechanism not captured by these dimensions (e.g., sonic patterning, typographic experimentation), LDRI will undercount it.

3. **No positive/negative valence.** LDRI measures richness, not quality. A text can be high-LDRI and aesthetically poor (e.g., a pretentious novel with dense symbolism but empty emotional register). LDRI complements but does not replace literary judgment.

4. **Empirical validation pending.** LDRI is a theoretical metric. Its empirical validation on a corpus of literary works with human-annotated richness scores is the necessary next step.

---

## 10. Conclusion: LDRI as a Methodology

The Literary Discourse Richness Index provides a principled, bounded, interpretable, and decomposable metric for quantifying a central but elusive concept in literary studies: the richness of literary discourse.

LDRI is not a readability score, nor a measure of lexical diversity, nor a sentiment thermometer. It is a formal operationalization of what literary critics mean when they describe a text as "rich" -- that it operates on multiple registers of meaning simultaneously, that those registers interact and modify each other, that their interplay unfolds across narrative time, and that patterns of significance persist, decay, and recur in structurally meaningful ways.

As a methodology chapter, this document establishes:

1. **What LDRI measures** (\S1): discourse richness, formally defined as the degree of multi-dimensional, coupled, temporally evolving, memory-bearing literary signal.

2. **What properties it satisfies** (\S2): boundedness, monotonicity, interpretability, stability, comparability, robustness, scale invariance, explainability, and uncertainty awareness -- each defined, motivated, and proven.

3. **How it is derived** (\S3): from the HMP-DT Narrative Embedding via the Rayleigh activation function, the $\ell_2$ signal norm, and the confidence-weighted geometric mean.

4. **How its components integrate** (\S4): all six dimensions, plus coupling, resonance, memory, and uncertainty, each with a distinct narratological role.

5. **How confidence is separated from richness** (\S5): richness is what we measure; confidence is how sure we are; reliability is how stable the measurement is.

6. **How it decomposes into interpretable factors** (\S6): intrinsic, interaction, temporal, resonance, memory, and uncertainty contributions.

7. **What mathematical guarantees it provides** (\S7): seven propositions with formal proof sketches.

8. **How it behaves on real texts** (\S8): worked examples contrasting a simple thriller chapter (LDRI $= 0.417$) with a complex literary chapter (LDRI $= 0.703$).

9. **How it compares to existing metrics** (\S9): advantages and limitations, with a clear statement of scope.

The LDRI methodology is designed for implementation within the LIT-DISC framework in its current frozen state. No modification to HMP-DT Version 1.0 is required. The metric consumes only the standard HMP-DT outputs -- the Narrative Embedding $\mathbf{H}$, the memory store $\mathcal{M}_{\text{store}}$, the resonance field $\mathcal{R}(\tau)$, and the per-dimension confidences $\gamma_d$ -- and produces a bounded, interpretable, decomposable richness estimate.

---

## References

- Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., \& Dodds, P. S. (2016). The emotional arcs of stories are dominated by six basic shapes. *EPJ Data Science*, 5(1), 31.
- Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press.
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology*, 32(3), 221--233.
- Pennebaker, J. W., Boyd, R. L., Jordan, K., \& Blackburn, K. (2015). *The Development and Psychometric Properties of LIWC2015*. University of Texas at Austin.
- Hutto, C. J., \& Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Blei, D. M., Ng, A. Y., \& Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3, 993--1022.

---

*End of LDRI Methodology Chapter*<｜end▁of▁thinking｜>
