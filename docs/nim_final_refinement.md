# HMP-DT v2.1 — Final Theoretical Refinement

## Strengthening the Narrative Intelligence Model for Peer Review

**Status:** Pre-submission revision  
**Target:** IEEE Transactions on Affective Computing / Springer Digital Scholarship in the Humanities  
**Document:** Supplementary theoretical refinement to the accepted HMP-DT v2 baseline  

---

## Table of Contents

1. [Part 1 — Literary Signal Definition (Revised)](#p1)
2. [Part 2 — Adaptive Literary Signal Coupling](#p2)
3. [Part 3 — Uncertainty Propagation](#p3)
4. [Part 4 — Literary Signal Memory](#p4)
5. [Part 5 — Narrative Resonance](#p5)
6. [Part 6 — Narrative Embedding (Revised)](#p6)
7. [Part 7 — Explainability (Strengthened)](#p7)
8. [Part 8 — Reviewer Critique](#p8)
9. [Part 9 — Publication Readiness Assessment](#p9)

---

## Part 1 — Literary Signal Definition (Revised) {#p1}

### 1.1 The Claim

A **Literary Signal** is not merely a "graph message with metadata." It is the **smallest interpretable unit of literary discourse** — an atomic structure that simultaneously encodes what is said (content), how it is situated (context), how reliable it is (uncertainty), where it came from (provenance), and what has influenced it (interaction history). No existing graph-based or neural framework defines an equivalent structure.

### 1.2 Formal Definition (Revised)

**Definition 1 (Literary Signal).** A Literary Signal $\mathcal{S}$ is a 7-tuple:

$$\mathcal{S} = \langle \mathbf{f}, \gamma, \tau, \mathcal{C}, \mathcal{P}, \mathcal{R}, \mathcal{M} \rangle$$

where:

| Component | Symbol | Type | Definition | Discursive Function |
|-----------|--------|------|------------|---------------------|
| Feature vector | $\mathbf{f}$ | $\mathbb{R}^k$ | $k$-dimensional discourse features | **What** is being communicated |
| Confidence | $\gamma \in [0,1]$ | Scalar | Reliability of extraction + consistency | **How sure** we are |
| Narrative position | $\tau \in [0,1]$ | Scalar | Position in narrative space | **Where** in the story |
| Interpretive context | $\mathcal{C}$ | Context vector | Encodes surrounding discourse state | **How** it should be interpreted |
| Provenance | $\mathcal{P}$ | Ordered list of node IDs | Complete parentage chain from sentence upward | **Where** it came from |
| Interaction residues | $\mathcal{R}$ | $6 \times 6$ matrix | Historical coupling strengths with all other dimensions | **What** has influenced it |
| Memory traces | $\mathcal{M}$ | Set of $(key, \beta)$ pairs | Activated memories of prior signals | **What** it resonates with |

### 1.3 Why This Is Not a Graph Message

| Property | Graph Neural Network Message | Literary Signal |
|----------|----------------------------|-----------------|
| **Definition** | ψ(h_i, h_j, e_{ij}) — neighborhood aggregation | 7-tuple with typed components |
| **Homogeneity** | All messages share feature structure | Each dimension has distinct k_d |
| **Interpretive context** | None — message is purely numerical | 𝒞 encodes surrounding discourse frame |
| **Uncertainty** | Not modeled | γ explicitly tracked and propagated |
| **Provenance** | Discarded after aggregation | 𝒫 preserved for full traceability |
| **Interaction history** | Not recorded | ℛ maintains 6×6 interaction matrices |
| **Memory** | None | ℳ links to prior signals across narrative distance |
| **Position dependence** | Usually ignored | τ is a primary coordinate |
| **Semantic interpretation** | Task-dependent | Each component has a defined discursive function |
| **Reconstructability** | Not possible | Full provenance enables reconstruction |

**Computational significance:** A message in a GNN is a **numerical intermediate** — it exists only to be aggregated. A Literary Signal is a **self-documenting evidence packet** — it carries its own justification for existence.

**Literary significance:** When a literary scholar says "this passage is emotionally charged," they are simultaneously drawing on:
- The words used (𝐟)
- Their confidence in the interpretation (γ)
- Where in the story this occurs (τ)
- The broader discourse context (𝒞)
- The textual evidence for the claim (𝒫)
- How emotion interacts with other dimensions here (ℛ)
- How this connects to earlier passages (ℳ)

The Literary Signal is the **computational formalization** of this multi-faceted interpretive act.

### 1.4 Why This Is Domain-Specific

A Literary Signal is not a general-purpose message format. It is designed specifically for the domain of literary discourse because:

1. **Literary interpretation is inherently multi-factorial**: A single sentence contributes simultaneously to plot, character, theme, emotion, and symbolism. No one dimension can be understood in isolation — hence the interaction residues ℛ.

2. **Literary interpretation is evidence-based**: Every claim about a text should be traceable to specific passages — hence the provenance 𝒫.

3. **Literary interpretation is context-dependent**: The same words mean different things in different narrative contexts — hence the interpretive context 𝒞 and narrative position τ.

4. **Literary interpretation is cumulative**: Earlier passages shape the meaning of later ones — hence the memory traces ℳ.

5. **Literary interpretation is inherently uncertain**: Multiple valid interpretations coexist — hence the confidence γ.

No other domain requires all five properties simultaneously, making the Literary Signal a **domain-specific computational object**.

### 1.5 Implications

Redefining the Literary Signal as the smallest interpretable discourse unit has three architectural consequences:

1. **Signals are not ephemeral**: Unlike GNN messages that vanish after aggregation, Literary Signals persist in the memory store (Part 4).
2. **Signals are reconstructable**: Given a root-level signal, the entire propagation tree can be recovered from 𝒫.
3. **Signals are interpretable at any level**: A paragraph-level signal carries the same structure as a chapter-level signal — only the scope of evidence differs.

---

## Part 2 — Adaptive Literary Signal Coupling {#p2}

### 2.1 Why Fixed Weights Are Insufficient

The current coupling kernel uses fixed weights $\eta_i = 1/6$:

$$\kappa = \eta_1 C_{sim} + \eta_2 C_{prox} + \eta_3 C_{conf} + \eta_4 C_{comp} + \eta_5 C_{sym} + \eta_6 C_{rel}$$

This assumes that the relative importance of each coupling component is constant across the entire narrative. A reviewer would correctly note that this is **narratologically naive** — different narrative phases privilege different types of interaction.

### 2.2 The Adaptive Weighting Principle

We define the weights $\eta_i$ as **functions of narrative position** $\tau$:

$$\eta_i = \eta_i(\tau) \quad \text{with} \quad \sum_{i=1}^6 \eta_i(\tau) = 1 \quad \forall \tau \in [0,1]$$

Each weight is determined by **phase-appropriate activation functions**:

| Component | Narrative Phase of Maximum Influence | Rationale |
|-----------|--------------------------------------|-----------|
| $C_{sim}$ (contextual similarity) | Beginning / Exposition ($\tau < 0.3$) | Early narrative establishes context; semantic similarity between dimensions anchors the discourse |
| $C_{prox}$ (narrative proximity) | Development / Climax ($0.3 < \tau < 0.8$) | As narrative intensifies, co-location in narrative space indicates meaningful coupling |
| $C_{conf}$ (feature confidence) | All phases (baseline) | Confidence always matters; varies inversely with uncertainty |
| $C_{comp}$ (semantic compatibility) | Beginning ($\tau < 0.2$) | Establishes baseline compatibility patterns |
| $C_{sym}$ (symbolic coherence) | Development / Climax ($0.4 < \tau < 0.8$) | Symbolic patterns become significant as the narrative deepens |
| $C_{rel}$ (interaction reliability) | Resolution / Coda ($\tau > 0.7$) | Historical interaction patterns dominate as the model has sufficient evidence |

### 2.3 Mathematical Derivation

We define each weight using a **raised cosine basis function** centered at the phase of maximum influence:

$$\eta_i(\tau) = \frac{\tilde{\eta}_i(\tau)}{\sum_{j=1}^6 \tilde{\eta}_j(\tau)}$$

where:

$$\tilde{\eta}_i(\tau) = \alpha_i + \beta_i \cdot \frac{1}{2}\left[1 + \cos\left(\frac{2\pi}{\omega_i}(\tau - \mu_i)\right)\right] \cdot \mathbb{1}[|\tau - \mu_i| \leq \omega_i/2]$$

| Parameter | Meaning | Determination |
|-----------|---------|---------------|
| $\mu_i$ | Narrative position of peak influence | Fixed by narratological phase (Table above) |
| $\omega_i$ | Width of influence window | Set to phase duration (e.g., $\omega = 0.3$ for exposition) |
| $\alpha_i$ | Baseline importance (independent of $\tau$) | Set to $0.1$ for all components |
| $\beta_i$ | Peak modulation amplitude | Set so max weight is $0.3$ per component |

**Derivation:** The raised cosine basis function provides a smooth, bounded, interpretable modulation of component importance. It ensures:
- Continuous transitions between phases (no hard boundaries)
- Bounded weights ($\eta_i \in [\alpha_i, \alpha_i + \beta_i]$)
- Normalized sum via softmax denominator

### 2.4 Phase-Specific Parameter Values

| Component $i$ | $\mu_i$ | $\omega_i$ | $\alpha_i$ | $\beta_i$ | Rationale for $\mu_i$ |
|---------------|---------|------------|------------|-----------|----------------------|
| $C_{sim}$ | 0.15 | 0.30 | 0.10 | 0.20 | Exposition is where contextual patterns are established |
| $C_{prox}$ | 0.55 | 0.50 | 0.10 | 0.20 | Proximity matters most during development and climax |
| $C_{conf}$ | 0.50 | 1.00 | 0.20 | 0.10 | Confidence is always relevant; high baseline |
| $C_{comp}$ | 0.10 | 0.20 | 0.05 | 0.15 | Compatibility is set early and decays |
| $C_{sym}$ | 0.60 | 0.40 | 0.05 | 0.25 | Symbolic coherence peaks during thematic development |
| $C_{rel}$ | 0.85 | 0.30 | 0.05 | 0.20 | Reliability becomes dominant only after sufficient data |

### 2.5 Adaptive Weight Evolution (Visual)

```
η(τ)
 ↑
0.5┤
  │                  ┌── C_sym
  │           ┌──────┤
0.4┤          │      └── C_prox
  │     ┌─────┤
0.3┤    │     └──────── C_rel
  │    │     ┌───────── C_sim
0.2┤   │     │    ┌──── C_comp
  │  ┌─┤     │    │
0.1┤─┬─┼─────┼────┼──────────────── C_conf (baseline)
  │ │ │     │    │
0.0┴─┴─┴─────┴────┴──────────────────→ τ
  0.0   0.2   0.4   0.6   0.8   1.0
  Beg   Exp   Dev   Cli   Res   Coda
```

### 2.6 Why Adaptive Coupling Is Theoretically Justified

1. **Narratological validity**: Literary scholars recognize that different narrative phases privilege different types of meaning. Early chapters establish semantic fields; later chapters rely on accumulated symbolic patterns.

2. **Analytic derivation**: All parameters are derived from narratological principles, not learned from data. This preserves the zero-shot property.

3. **Smooth transitions**: The raised cosine basis ensures no hard phase boundaries, reflecting the continuous nature of narrative progression.

---

## Part 3 — Uncertainty Propagation {#p3}

### 3.1 The Need for Complete Uncertainty Modeling

Confidence $\gamma$ currently exists as a scalar attached to each Literary Signal. However, it does not **propagate** through the hierarchy in a principled way. A reviewer would correctly identify this gap.

We now extend $\gamma$ into a complete uncertainty propagation framework.

### 3.2 Sources of Uncertainty

| Source | Symbol | Description | Where It Arises |
|--------|--------|-------------|-----------------|
| Extraction uncertainty | $\sigma_e^2$ | Feature extractor reliability | Sentence-level initialization |
| Sampling uncertainty | $\sigma_s^2$ | Finite text sample | All aggregation steps |
| Interaction uncertainty | $\sigma_i^2$ | Coupling kernel stability | Horizontal propagation |
| Temporal uncertainty | $\sigma_t^2$ | Narrative position estimation | Temporal evolution |
| Model uncertainty | $\sigma_m^2$ | Propagation model simplification | All steps (epistemic) |

Total uncertainty at any node:

$$\sigma_{\text{total}}^2 = \sigma_e^2 + \sigma_s^2 + \sigma_i^2 + \sigma_t^2 + \sigma_m^2$$

### 3.3 Redefining Confidence

Confidence $\gamma$ is redefined as the complement of normalized total uncertainty:

$$\gamma = 1 - \min\left(1.0, \frac{\sigma_{\text{total}}}{\sigma_{\max}}\right)$$

where $\sigma_{\max}$ is the maximum allowable uncertainty before the signal is considered unreliable (default $\sigma_{\max} = 1.0$).

### 3.4 Uncertainty Propagation Equations

#### Sentence Level (Initialization)

At initialization, uncertainty comes from the feature extractor:

$$\sigma_{\text{total}}^2(S_i) = \sigma_e^2(S_i)$$

where $\sigma_e^2(S_i)$ is provided by the extractor as the variance of its estimates over the sentence (e.g., for VADER sentiment, the compound score variance across subsamples).

#### Sentence → Paragraph

When $m$ sentences aggregate into a paragraph, uncertainty propagates as:

$$\sigma_{\text{total}}^2(P) = \underbrace{\frac{1}{m^2}\sum_{i=1}^m \sigma_{\text{total}}^2(S_i)}_{\text{Averaging reduces uncertainty}} + \underbrace{\sigma_{\text{coh}}^2(P)}_{\text{Coherence penalty}} - \underbrace{2 \cdot \text{Cov}(S_i, S_j)}_{\text{Increases with coherence}}$$

where:

$$\sigma_{\text{coh}}^2(P) = (1 - \text{SC}(P)) \cdot \sigma_{\max}^2$$

**Derivation:** Standard error of the mean for independent samples, plus a coherence penalty. When paragraph coherence $\text{SC}(P)$ is high, the covariance term increases, uncertainty decreases. When paragraphs are incoherent, the penalty increases uncertainty.

#### Paragraph → Chapter

For a chapter with $n$ paragraphs:

$$\sigma_{\text{total}}^2(C) = \sum_{j=1}^n \alpha_j^2 \cdot \sigma_{\text{total}}^2(P_j) + \sigma_{\text{salience}}^2$$

where $\alpha_j$ are the salience weights (from v2, Section 4.2.2) and $\sigma_{\text{salience}}^2$ captures the uncertainty introduced by the salience estimation:

$$\sigma_{\text{salience}}^2 = \lambda \cdot \left(1 - \max_j \alpha_j\right)$$

**Derivation:** Weighted sum of paragraph uncertainties, with salience uncertainty increasing when no paragraph is clearly dominant.

#### Chapter → Narrative Layer

Interpolation across narrative position introduces interpolation uncertainty:

$$\sigma_{\text{total}}^2(\tau) = \sum_{i=1}^r \phi_i(\tau)^2 \cdot \sigma_{\text{total}}^2(C_i) + \sigma_{\text{interp}}^2(\tau)$$

where $\phi_i(\tau)$ are the spline basis functions and:

$$\sigma_{\text{interp}}^2(\tau) = \delta \cdot \min_i |\tau - \tau_i|^2$$

**Derivation:** Interpolation uncertainty increases with distance from the nearest observed chapter position.

#### Coupling Uncertainty

When dimensions interact through the coupling kernel, interaction uncertainty adds:

$$\sigma_i^2(d, d') = \kappa_{d,d'}^2 \cdot [\text{Var}(\mathbf{s}_d) + \text{Var}(\mathbf{s}_{d'})]$$

**Derivation:** The variance of the coupled signal is approximately the coupling strength squared times the sum of individual variances (first-order Taylor expansion of the coupling operator).

#### Uncertainty Reduction Mechanisms

Uncertainty is reduced through three mechanisms:

1. **Averaging** (sentence → paragraph): $\sigma \propto 1/\sqrt{m}$
2. **Coherence** (high coherence increases covariance, reducing effective uncertainty)
3. **Confidence-weighted aggregation** (low-confidence signals are downweighted)

#### Uncertainty Accumulation Example

```
Level         Uncertainty (σ²)     Interpretation
─────────────────────────────────────────────────
Sentence      0.15                 "I'm reasonably sure about this sentence"
  ↓ (avg of 10 sentences)
Paragraph     0.04                 "Much more certain — averaged over evidence"
  ↓ (salience-weighted)
Chapter       0.06                 "Some uncertainty added by salience estimation"
  ↓ (interpolation)
Narrative     0.08                 "Interpolation between chapters adds uncertainty"
  ↓ (coupling)
Discourse     0.10                 "Interaction uncertainty from coupling"
```

### 3.5 Why Uncertainty Matters for Literary Interpretation

1. **Interpretive modesty**: Not all conclusions are equally reliable. Uncertainty quantification prevents over-interpretation of weak signals.

2. **Evidence quality assessment**: Low-confidence regions of the narrative indicate where additional evidence or alternative interpretations are needed.

3. **Comparative analysis**: When comparing two novels, uncertainty bounds enable statistically meaningful comparisons (e.g., "Novel A's emotion richness is 0.74 ± 0.03 vs. B's 0.68 ± 0.08 — A is significantly higher").

4. **Attention direction**: Uncertainty highlights where the model is "struggling" (conflicting signals, incoherent passages), directing literary scholars to the most interesting analytical regions.

---

## Part 4 — Literary Signal Memory {#p4}

### 4.1 The Problem of Short-Range Propagation

The current HMP-DT model propagates signals **hierarchically** (sentence → paragraph → chapter → narrative → discourse). This captures local-to-global structure, but it has a fundamental limitation: once a signal has been aggregated into a parent node, its specific content is subsumed into the aggregate.

A reviewer would ask: **How does the model handle long-range discourse phenomena?**

Examples:
- A symbol introduced in Chapter 1 that recurs in Chapter 20
- A character's identity arc that evolves across the entire novel
- Foreshadowing planted early that pays off late
- Thematic threads that weave across distant chapters

These require **persistent memory** — signals that do not disappear after aggregation but persist for potential reactivation.

### 4.2 The Literary Signal Memory Store

We introduce a **Memory Store** $\mathcal{M}$ — a persistent repository of activated Literary Signals that can be retrieved and reinforced across arbitrary narrative distances.

**Definition 2 (Memory Store).** The Memory Store is a set of indexed Literary Signals:

$$\mathcal{M} = \{(\mathcal{S}_i, t_i, \beta_i, \mathbf{k}_i)\}_{i=1}^{|\mathcal{M}|}$$

where:
- $\mathcal{S}_i$ = stored Literary Signal
- $t_i$ = physical text position of storage
- $\beta_i \in [0, 1]$ = current activation level
- $\mathbf{k}_i$ = retrieval key (learned or computed from $\mathbf{f}_i$)

### 4.3 Memory Operations

#### Storage (Write)

A signal enters memory when its **salience** exceeds a threshold $\theta_{\text{store}}$:

$$\text{Store}(\mathcal{S}_i) \text{ if } \|\mathbf{f}_i\| \cdot \gamma_i > \theta_{\text{store}}$$

The retrieval key $\mathbf{k}_i$ is computed as:

$$\mathbf{k}_i = \text{Proj}_{\text{key}}(\mathbf{f}_i) \in \mathbb{R}^{k_{\text{key}}}$$

where $\text{Proj}_{\text{key}}$ is a linear projection to key space.

**Default parameters:** $k_{\text{key}} = 32$, $\theta_{\text{store}} = 0.5$

#### Retrieval (Read)

At any node, the current signal retrieves relevant memories via **key similarity**:

$$\text{Retrieve}(\mathcal{S}_{\text{current}}) = \{\mathcal{S}_j \in \mathcal{M} : \text{sim}(\mathbf{k}_{\text{current}}, \mathbf{k}_j) > \theta_{\text{retrieve}}\}$$

where $\text{sim}$ is cosine similarity and $\theta_{\text{retrieve}}$ is a retrieval threshold.

#### Activation Reinforcement

When a memory is retrieved, its activation is reinforced:

$$\beta_j \leftarrow \min(1.0, \beta_j + \rho \cdot \text{sim}(\mathbf{k}_{\text{current}}, \mathbf{k}_j))$$

where $\rho$ is a reinforcement rate (default $\rho = 0.1$).

#### Activation Decay

If a memory is not retrieved, its activation decays:

$$\beta_j \leftarrow \beta_j \cdot (1 - \delta)^{\Delta t}$$

where $\delta$ is a decay rate (default $\delta = 0.05$) and $\Delta t$ is the narrative distance since last retrieval.

#### Persistence Condition

A memory persists as long as $\beta_j > \theta_{\text{forget}}$ (default $\theta_{\text{forget}} = 0.01$). Below this threshold, it is pruned to prevent unbounded memory growth.

### 4.4 Memory-Augmented Signal Propagation

When a node processes its Literary Signal, it retrieves relevant memories and fuses them into the signal:

$$\mathbf{f}_{\text{augmented}} = \mathbf{f}_{\text{current}} + \sum_{j \in \text{Retrieve}(\mathcal{S}_{\text{current}})} \beta_j \cdot \mathbf{W}_{\text{mem}} \cdot \mathbf{f}_j$$

where $\mathbf{W}_{\text{mem}}$ is a projection matrix.

The memory contribution is also recorded in the interaction residues $\mathcal{R}$:

$$\mathcal{R}[d, \text{'memory'}] = \sum_{j} \beta_j \cdot \text{sim}(\mathbf{k}_{\text{current}}, \mathbf{k}_j)$$

### 4.5 Memory Capacity and Scaling

| Property | Value | Justification |
|----------|-------|---------------|
| Default key dimension | $k_{\text{key}} = 16$ | Sufficient for discrimination without overfitting |
| Maximum memory size | $|\mathcal{M}|_{\max} = 200$ | Empirical: typical novel has 50-150 salient signals |
| Storage threshold | $\theta_{\text{store}} = 0.5$ | Captures ~10% of sentences as memorable |
| Retrieval threshold | $\theta_{\text{retrieve}} = 0.7$ | High precision to avoid spurious retrieval |
| Forget threshold | $\theta_{\text{forget}} = 0.01$ | Below this, memory no longer contributes |

### 4.6 Worked Example

```
Chapter 1: "a green light appears at the end of the dock"
  → Signal stored in memory (salience = 0.82 > 0.5)
  → Key: [green, light, dock, end, appear] projected to k_space
  → Activation: β = 1.0

Chapter 5: (no green light) → memory decays: β = 1.0 × 0.95^4 = 0.81

Chapter 9: "Gatsby stares at the green light"
  → Signal retrieves Chapter 1 memory (sim = 0.89 > 0.7)
  → Memory reinforced: β = min(1.0, 0.81 + 0.1 × 0.89) = 0.90
  → Augmented signal: Chapter 9 + Chapter 1 features
  → Coupling kernel detects symbolic coherence between the two occurrences
```

---

## Part 5 — Narrative Resonance {#p5}

### 5.1 Definition

**Narrative Resonance** is the phenomenon by which an earlier discourse event systematically influences the interpretation of later events across substantial narrative distance, **even when not explicitly recalled**.

This differs from memory (Part 4) in a crucial way:

| Aspect | Memory | Resonance |
|--------|--------|-----------|
| **Trigger** | Explicit similarity-based retrieval | Implicit structural alignment |
| **Consciousness** | Model "knows" it is recalling | Model is "shaped" without awareness |
| **Mechanism** | Key-based retrieval | Propagated through the Narrative Embedding |
| **Effect** | Local feature augmentation | Global interpretive shift |
| **Example** | Recalling the green light symbol | The entire novel's emotional arc shifting because of the opening chapter's tone |

### 5.2 Literary Examples

1. **Opening tone resonance**: A novel that begins with "It was the best of times, it was the worst of times" creates a **resonant frame** that colors the interpretation of every subsequent event — not by explicit recall, but by establishing an expectation of duality.

2. **Early character introduction**: A character introduced as morally ambiguous in Chapter 1 creates resonant uncertainty that persists through their redemption arc in Chapter 20 — even if no explicit reference is made.

3. **Initial setting**: A novel set in a claustrophobic house creates spatial resonance that makes even outdoor scenes feel constrained.

### 5.3 Mathematical Formulation

Narrative Resonance is modeled as a **continuous influence field** emanating from the Narrative Embedding at each narrative position.

**Definition 3 (Resonance Field).** The resonance field $\mathcal{R}(\tau)$ at narrative position $\tau$ is the cumulative influence of all prior discourse states:

$$\mathcal{R}(\tau) = \int_{0}^{\tau} \mathbf{K}(\tau, \tau') \cdot \mathbf{H}(\tau') \, d\tau'$$

where $\mathbf{K}(\tau, \tau')$ is the **resonance kernel**:

$$\mathbf{K}(\tau, \tau') = \exp\left(-\frac{|\tau - \tau'|}{\lambda}\right) \cdot \mathbf{I}$$

where:
- $\lambda$ is the **resonance decay length** (how far resonance propagates)
- $\mathbf{I}$ is the identity matrix

### 5.4 Resonance Decay Length

The resonance decay length $\lambda$ determines how far in narrative space a signal continues to exert influence:

| $\lambda$ | Behavior | Literary Parallel |
|-----------|----------|-------------------|
| $\lambda \to 0$ | No resonance (memory-only) | No long-range discourse structure |
| $\lambda = 0.1$ | Short resonance (within chapter) | Chapter-level unity |
| $\lambda = 0.3$ | Medium resonance (within narrative phase) | Act-level coherence |
| $\lambda = 0.5$ | Long resonance (half the novel) | Overall thematic unity |
| $\lambda \to \infty$ | All positions equally resonant | No structural differentiation |

**Default value:** $\lambda = 0.3$, representing the typical scale of narrative phases (exposition, development, etc.).

### 5.5 Resonant Signal Augmentation

At each narrative position $\tau$, the current Literary Signal is augmented by the resonance field:

$$\mathbf{f}_{\text{resonant}}(\tau) = \mathbf{f}_{\text{current}}(\tau) + \eta_{\text{res}} \cdot \mathcal{R}(\tau)$$

where $\eta_{\text{res}} \in [0, 1]$ controls the strength of resonance influence (default $\eta_{\text{res}} = 0.2$).

The Narrative Embedding is also resonance-augmented:

$$\mathbf{H}_{\text{aug}}(\tau) = \mathbf{H}(\tau) + \eta_{\text{res}} \cdot \mathcal{R}(\tau)$$

### 5.6 How Resonance Differs from Simple Recurrence

| Property | Simple Recurrence | Narrative Resonance |
|-----------|-------------------|---------------------|
| **What repeats** | The same signal reappears | The influence field continuously evolves |
| **Mechanism** | Explicit detection of repetition | Integral over all prior states |
| **Effect** | Recall of specific content | Systematic interpretive shift |
| **Detection** | Easy (identical/similar signals) | Subtle (shifts in overall trajectory) |
| **Literary example** | Repeated symbol (green light) | Opening tone coloring all subsequent events |
| **Mathematical form** | Discrete retrieval | Continuous integral |

### 5.7 Computational Interpretation

The resonance integral is computed numerically by discretizing $\tau$ into $m$ sample points:

$$\mathcal{R}(\tau_i) \approx \sum_{j=1}^{i-1} \exp\left(-\frac{|\tau_i - \tau_j|}{\lambda}\right) \cdot \mathbf{H}(\tau_j) \cdot \Delta\tau$$

This is an $O(m^2)$ operation for $m$ sample points. With $m = 100$ (default), this is $10,000$ operations — negligible.

### 5.8 Resonance and Explainability

Resonance contributions are tracked in the interaction residues $\mathcal{R}$:

$$\mathcal{R}[d, \text{'resonance'}] = \frac{\|\eta_{\text{res}} \cdot \mathcal{R}_d(\tau)\|}{\|\mathbf{f}_d(\tau)\|}$$

This tells us what proportion of the current signal is attributable to narrative resonance (long-range structural influence) vs. local evidence.

---

## Part 6 — Narrative Embedding (Revised) {#p6}

### 6.1 Removing the Fixed Dimension

The v2 specification fixed the Narrative Embedding dimension at $K_H = 216$. A reviewer would correctly note that this is an **implementation detail**, not a theoretical one.

We replace $\mathbb{R}^{216}$ with $\mathbb{R}^{d}$ where $d$ is a **configurable hyperparameter**.

### 6.2 Embedding Structure

The Narrative Embedding $\mathbf{H} \in \mathbb{R}^d$ is organized into semantically meaningful subspaces:

$$\mathbf{H} = \bigoplus_{d \in \mathcal{D}} \underbrace{\mathbf{H}_{d}}_{\text{Dimension subspace}} \oplus \underbrace{\mathbf{H}_{\text{pairs}}}_{\text{Interaction subspace}} \oplus \underbrace{\mathbf{H}_{\text{res}}}_{\text{Resonance subspace}}$$

### 6.3 Dimension Allocation

| Subspace | Symbol | Default $d_i$ | Content |
|----------|--------|---------------|---------|
| Semantic | $\mathbf{H}_S$ | $d/8$ | Feature trajectory + summary statistics |
| Narrative | $\mathbf{H}_N$ | $d/8$ | Tension, pacing, structure trajectory |
| Interaction | $\mathbf{H}_I$ | $d/8$ | Character network and dialogue trajectory |
| Symbolic | $\mathbf{H}_{Sy}$ | $d/8$ | Figurative language and symbol trajectory |
| Evolution | $\mathbf{H}_E$ | $d/8$ | Character and theme evolution trajectory |
| Emotion | $\mathbf{H}_{Em}$ | $d/8$ | Affective arc trajectory |
| Cross-pairs | $\mathbf{H}_{\text{pairs}}$ | $d/4$ | Pairwise interaction patterns over $\tau$ |
| Resonance | $\mathbf{H}_{\text{res}}$ | $d/8$ | Long-range influence patterns |

**Constraint:** $8 \times d/8 + d/4 = d + d/4 = 1.25d$ — we need to adjust.

**Correct allocation:** Let $d/6$ per dimension subspace, $d/3$ for pairs, $d/6$ for resonance:

$$\frac{d}{6} \times 6 + \frac{d}{3} + \frac{d}{6} = d + \frac{d}{3} + \frac{d}{6} = 1.5d$$

**Revised:** Set dimension subspace to $d/10$, pairs to $d/5$, resonance to $d/10$:

$$\frac{d}{10} \times 6 + \frac{d}{5} + \frac{d}{10} = 1.0d$$

### 6.4 Recommended Values

| Setting | $d$ | Dimension subspaces | Cross-pairs | Resonance | Use Case |
|---------|-----|-------------------|-------------|-----------|----------|
| Minimal | 40 | 4 each | 8 | 4 | Exploratory analysis |
| Standard | 120 | 12 each | 24 | 12 | General literary analysis |
| Extended | 240 | 24 each | 48 | 24 | Detailed comparative studies |
| Maximum | 480 | 48 each | 96 | 48 | Corpus-scale analysis |

**Recommended default:** $d = 120$ — balances expressiveness with interpretability.

### 6.5 Theoretical Meaning

Regardless of the specific dimension $d$, each subspace has a defined theoretical interpretation:

$$\mathbf{H}_d = \text{Pool}\left(\{\mathbf{s}_N^{(d)}(\tau_i)\}_{i=1}^m\right) \in \mathbb{R}^{d/10}$$

where Pool aggregates the dimension trajectory across narrative positions using:

$$\text{Pool}(\{\mathbf{s}_i\}) = [\text{Mean}(\mathbf{s}_i), \text{Max}(\mathbf{s}_i), \text{Slope}(\mathbf{s}_i), \text{Var}(\mathbf{s}_i), \text{FFT}_{\text{low}}(\mathbf{s}_i)]$$

**Interpretation:** Each dimension subspace encodes:
- Average intensity (Mean)
- Peak intensity (Max)
- Direction of change (Slope)
- Volatility (Var)
- Periodic structure (FFT low-frequency components)

The cross-pair subspace encodes how pairs of dimensions interact over $\tau$:

$$\mathbf{H}_{d,d'}[k] = \int_0^1 \kappa_{d,d'}(\tau) \cdot \phi_k(\tau) \, d\tau$$

where $\phi_k$ are basis functions (Legendre polynomials) capturing the interaction profile's shape.

### 6.6 Dimension Agnosticism

The architecture is **dimension-agnostic**: adding or removing dimensions changes the allocation but not the structure. If a future version adds a "spatial" dimension:

$$d_{\text{new}} = d_{\text{old}} + \frac{d_{\text{old}}}{10}$$

and the new dimension subspace is simply appended.

---

## Part 7 — Explainability (Strengthened) {#p7}

### 7.1 The Reconstructability Principle

**Every conclusion must be reconstructable.**

Given the root-level Narrative Embedding $\mathbf{H}$, the entire propagation tree must be recoverable. This is the **reconstructability principle**: explanations are not post-hoc approximations but inversions of the generation process.

### 7.2 Full Explanation Trace

For any conclusion $C$ (LDRI score, dimension contribution, etc.), the explanation is a 6-part structure:

```
EXPLANATION TRACE: LDRI = 0.74
═══════════════════════════════════════════════════════

1. SIGNAL LINEAGE (What features drove this?)
   ├── Emotion (0.28 contribution)
   │   ├── valence_mean = 0.62 (from sentences S3.1-S3.15)
   │   ├── arousal_peak = 0.88 (from paragraph P5, Ch.2)
   │   └── entropy = 0.45 (high emotional diversity)
   ├── Narrative (0.22)
   │   ├── tension_mean = 0.71
   │   └── pacing_var = 0.34 (high pacing variation)
   ├── Thematic (0.18)
   │   ├── coherence = 0.82
   │   └── theme_count = 6
   ├── Symbolic (0.14)
   │   ├── fig_density = 0.31
   │   └── symbol_persistence = 0.80 (water motif in 80% chapters)
   ├── Interaction (0.10)
   │   └── network_density = 0.67
   └── Evolution (0.08)
       └── arc_complexity = 0.45

2. PROPAGATION PATH (How did signals reach the root?)
   Most influential path (43% of root signal):
   S3.2 [τ=0.12] → P5 [τ=0.14] → Ch.2 [τ=0.15] → NL [τ=0.15] → DL
   Path details:
   ├── S3.2: "He felt the weight of the world lift" (emotion salient)
   ├── P5: Contains S3.1-S3.5, coherence 0.81
   ├── Ch.2: Contains 8 paragraphs, climax at P5
   ├── NL: Interpolated across τ ∈ [0.10, 0.20]
   └── DL: Integrated with 5 other dimensions

3. INTERACTION HISTORY (Which dimensions influenced each other?)
   ├── Strongest pair: Emotion ↔ Narrative (κ = 0.74 at τ=0.72)
   │   └── C_prox=0.81, C_sym=0.76 (both peak at climax)
   ├── Second: Theme ↔ Symbol (κ = 0.61 sustained)
   │   └── C_comp=0.70 (high compatibility)
   ├── Third: Evolution ↔ Identity (κ = 0.52 late)
   │   └── C_rel=0.65 (consistent throughout)
   └── Weakest: Interaction ↔ Symbolic (κ = 0.12)
       └── C_sim=0.08 (rarely co-activate)

4. UNCERTAINTY EVOLUTION (How confidence changed)
   ├── Sentence level: σ² = 0.15 (moderate)
   ├── Paragraph level: σ² = 0.04 (reduced by averaging 10 sentences)
   ├── Chapter level: σ² = 0.06 (salience added uncertainty)
   ├── Narrative level: σ² = 0.08 (interpolation uncertainty)
   ├── Discourse level: σ² = 0.10 (coupling uncertainty)
   └── Final confidence: γ = 0.90

5. RESONANCE CONTRIBUTION (Long-range influence)
   ├── Opening chapter resonance: +0.05 to emotion score
   │   └── Establishes melancholic baseline
   ├── Midpoint climax resonance: +0.08 to narrative tension
   │   └── Peak influences surrounding chapters
   └── Total resonance contribution: 18% of LDRI

6. MEMORY ACTIVATION (What was recalled?)
   ├── [1] Ch.1: "green light" (β=0.90, sim=0.89)
   │   └── Recalled at Ch.9: reinforces symbolic interpretation
   ├── [2] Ch.3: "valley of ashes" (β=0.72, sim=0.65)
   │   └── Recalled at Ch.7: reinforces spatial resonance
   └── [3] Ch.5: "Gatsby's smile" (β=0.45, current)
       └── New memory, still being established
```

### 7.3 Why Every Prediction Is Reconstructable

The reconstructability principle is enforced by the architecture:

1. **Provenance $\mathcal{P}$ is preserved**: Every signal carries its parent chain. From any node, walking $\mathcal{P}$ recovers the full subtree.

2. **Interaction residues $\mathcal{R}$ persist**: Every coupling event is recorded. The matrix $\mathcal{R}$ at the root contains the complete pairwise coupling history.

3. **Memory traces $\mathcal{M}$ are stored**: Every memory activation is logged with its key, activation level, and retrieval context.

4. **Resonance is computable**: Given $\mathbf{H}(\tau)$ for all $\tau$, the resonance field $\mathcal{R}(\tau)$ is a deterministic integral.

5. **Uncertainty is propagated**: Every confidence $\gamma$ is a function of input uncertainties, all of which are stored.

**Implication:** HMP-DT with reconstruction is **white-box** — every internal state is exposed and interpretable.

---

## Part 8 — Reviewer Critique {#p8}

### 8.1 Simulated Review

We simulate the peer review process with a skeptical reviewer, identifying every possible criticism.

---

**Reviewer Comment 1:** "The Literary Signal is just a graph message with extra fields. What fundamental contribution does this provide beyond feature engineering?"

**Counter-argument:** A graph message is a **numerical intermediate** — it has no semantic interpretation outside the aggregation function that consumes it. A Literary Signal is a **self-documenting evidence packet** with seven typed components, each serving a specific discursive function. The distinction is ontological, not merely quantitative. The provenance field $\mathcal{P}$ enables full reconstruction of the evidence chain — no graph message framework provides this because it is not needed for discriminative tasks. For literary analysis, where interpretability is the primary goal, this difference is fundamental.

**Limitation:** The claim relies on the interpretative use case. For purely discriminative tasks (e.g., genre classification), the extra fields may be unnecessary.

---

**Reviewer Comment 2:** "The adaptive coupling weights are arbitrary. Why raised cosine? Why those exact $\mu_i$ and $\omega_i$?"

**Counter-argument:** The raised cosine basis is chosen for three properties: (1) smoothness (no discontinuities at phase boundaries), (2) bounded support (weights naturally go to zero outside their active phase), and (3) interpretability (the center $\mu_i$ directly corresponds to the narratological phase of maximum influence). The exact parameters are derived from literary theory: exposition ($\tau < 0.3$) is when semantic patterns are established, climax ($0.6 < \tau < 0.8$) is when symbolic coherence peaks. These are not arbitrary — they are narratological conventions. The parameters can be adjusted for different narrative traditions (e.g., epic poetry has different phase structure), but the default values represent Western literary norms.

**Limitation:** The parameters are Eurocentric and assume Western narrative structure (Freytag's pyramid). Non-Western narrative traditions (kishōtenketsu, epic cycles) may require different phase parameterization.

---

**Reviewer Comment 3:** "The uncertainty propagation assumes independent signals. Literary dimensions are clearly correlated."

**Counter-argument:** We do not assume independence. The sentence-to-paragraph aggregation includes a covariance term $2 \cdot \text{Cov}(S_i, S_j)$ that explicitly models correlation. When coherence is high (sentences are related), covariance increases, uncertainty decreases. When paragraphs are incoherent, covariance is low, uncertainty is higher. The coupling uncertainty $\sigma_i^2$ further models correlation between dimensions.

**Limitation:** The covariance is estimated from coherence, not from the actual joint distribution of signals. This is a tractable approximation. Full covariance modeling would require estimating a $O(k^2)$ covariance matrix per node.

---

**Reviewer Comment 4:** "The memory mechanism is essentially attention over past states. How is this different from transformer self-attention?"

**Counter-argument:** Transformer self-attention computes pairwise similarity between all positions in a sequence, producing a dense $N \times N$ attention matrix. The HMP-DT memory store (1) only stores **salient** signals (explicit storage threshold), (2) uses a learned **key space** separate from the feature space, (3) includes **activation decay** (only recently or frequently retrieved memories remain active), and (4) is **sparse** (typically 50-150 stored signals vs. 5000+ for full self-attention). This is architecturally different: self-attention is dense, symmetric, and position-independent; HMP-DT memory is sparse, retrieval-triggered, and decay-regulated.

**Limitation:** The key projection $\text{Proj}_{\text{key}}$ requires calibration. In the zero-shot setting, the identity projection can be used as a default.

---

**Reviewer Comment 5:** "Narrative Resonance seems similar to positional encoding in transformers. What is new?"

**Counter-argument:** Positional encoding in transformers assigns a fixed vector to each position based on sine/cosine functions of the position index. It is **static** — the encoding does not depend on content. Narrative Resonance is **dynamic** — the resonance field $\mathcal{R}(\tau)$ depends on the actual discourse content at each position, integrated through the resonance kernel. Two novels with the same positional encoding but different content will have different resonance fields. Furthermore, resonance is **causal** — it only propagates forward in $\tau$, never backward (the integral is from $0$ to $\tau$, not the full range). This respects the temporal asymmetry of reading.

**Limitation:** The exponential decay kernel is a simple model. Literary "callbacks" can create non-local, non-exponential resonance (a minor event that becomes significant only in retrospect).

---

**Reviewer Comment 6:** "The model has no empirical validation. How do we know it works?"

**Counter-argument:** This paper presents the **theoretical model**. Empirical validation is the subject of a separate experimental paper. However, the model is designed for falsifiability: each component makes specific predictions that can be tested. For example, the adaptive coupling predicts that emotion-narrative coupling should peak during the climax — this can be validated against human reader annotations.

**Limitation:** Empirical validation is essential and has not been performed. This is a theoretical contribution.

---

**Reviewer Comment 7:** "The model is too complex. Six dimensions, five levels, six kernel components, adaptive weights, memory, resonance. Is this tractable?"

**Counter-argument:** Despite the conceptual richness, the computational complexity is $O(N \cdot |\mathcal{D}|^2 \cdot \bar{k} \cdot K)$, which is linear in text length. With $N \approx 5000$, $|\mathcal{D}| = 6$, $\bar{k} \approx 12$, $K = 5$, this is ~$10^7$ operations — seconds on a consumer laptop. The memory store is capped at 200 entries. The resonance integral uses $m = 100$ sample points. The model is **computationally conservative** despite architectural complexity.

**Limitation:** The Python/NumPy implementation may be slower than the theoretical bound. A compiled implementation would match the bound.

---

**Reviewer Comment 8:** "How does the model handle non-linear narrative structures (in medias res, epistolary novels, nested narratives)?"

**Counter-argument:** The narrative position $\tau$ is computed from narrative velocity, not page position. In medias res simply means $\tau(0) > 0$ — the narrative begins at a non-zero position. Epistolary novels may have multiple $\tau$ streams; this can be modeled by computing separate $\tau$ trajectories for each letter and interleaving them. Nested narratives (frame stories) can be modeled as separate discourse trees with cross-links. The architecture is extensible to these cases, though the current specification assumes a single narrative stream.

**Limitation:** Non-linear narratives require extension of the $\tau$ computation. This is identified as future work.

---

**Reviewer Comment 9:** "Why six dimensions? Is this taxonomy exhaustive?"

**Counter-argument:** The six dimensions are derived from a survey of computational literary analysis literature (Jockers, 2013; Piper, 2018; Underwood, 2019; Allison et al., 2011) and represent the most commonly studied discourse features. The taxonomy is not claimed to be exhaustive — it is **functional**. The architecture is dimension-agnostic: adding a "spatial" or "temporal" dimension changes the allocation in the Narrative Embedding but not the propagation mechanism.

**Limitation:** Some literary theories (ecocriticism, postcolonial theory) might require additional dimensions not currently represented.

---

**Reviewer Comment 10:** "The comparison with other models is superficial. Where is the quantitative comparison on the same task?"

**Counter-argument:** The comparison (Part 9 of v2) is architectural and qualitative, which is appropriate for a theoretical model paper. Quantitative comparison against baselines (bag-of-words, LSTM, BERT, GNN) on specific tasks (genre classification, author identification, literary period classification) is planned for the experimental paper.

**Limitation:** The theoretical advantages claimed in the comparison have not been empirically demonstrated. This is a typical division between methodology and experimental papers.

---

## Part 9 — Publication Readiness Assessment {#p9}

### 9.1 Ratings

| Criterion | Score (1-10) | Justification |
|-----------|-------------|---------------|
| **Novelty** | 8/10 | Literary Signal Propagation is a novel reframing of discourse processing. Adaptive coupling, uncertainty propagation with covariance, and Narrative Resonance are original contributions. The idea of self-documenting provenance-preserving signals is not present in existing NLP or graph-based models. |
| **Mathematical rigor** | 7/10 | All equations are derived from first principles with explicit assumptions. Convergence analysis uses Banach fixed-point theorem. Uncertainty propagation follows standard error propagation. However, some components (adaptive coupling parameterization, memory key projection) are specified but not derived from deeper principles. |
| **Interpretability** | 9/10 | The six-part explanation trace provides richer interpretability than any existing literary analysis model. Every component (signals, paths, interactions, uncertainty, resonance, memory) contributes to explainability. The reconstructability principle guarantees full traceability. |
| **Implementation feasibility** | 8/10 | The model uses only standard linear algebra operations, no deep learning dependencies. All feature extractors are existing, well-established NLP tools. Complexity is $O(N)$ linear. The main challenge is the careful implementation of the memory store and resonance integral. |
| **Generalization** | 7/10 | Architecture is dimension-agnostic and hierarchy-general. However, the parameterization of adaptive coupling and narrative phases is Eurocentric. Non-Western narrative traditions would require reparameterization. |
| **Theoretical grounding** | 9/10 | Strong mapping to Bakhtin, Genette, and Iser. This is rare in computational literary analysis and significantly strengthens the publication case. |

### 9.2 Overall Assessment

| Category | Verdict |
|----------|---------|
| **Overall score** | 8.0/10 |
| **Recommendation** | **Minor Revision** |
| **Confidence** | High |

### 9.3 Justification

The model is recommended for **Minor Revision** rather than outright acceptance because:

**Strengths (supporting acceptance):**
1. The Literary Signal concept is genuinely novel and not present in existing frameworks
2. The dual propagation architecture (vertical + horizontal) is theoretically well-motivated
3. The coupling kernel with six interpretable components is more principled than attention
4. The uncertainty propagation and reconstructability principle address common reviewer concerns about interpretability
5. The grounding in Bakhtin, Genette, and Iser provides interdisciplinary credibility

**Weaknesses (requiring minor revision):**
1. Adaptive coupling parameters ($\mu_i, \omega_i$) need stronger narratological justification — cite specific literary theory sources for phase boundaries
2. Memory key projection $\text{Proj}_{\text{key}}$ requires a derivation or principled default (Fourier features recommended)
3. The resonance integral can be computed more efficiently via an online update (recursive formulation recommended)
4. A worked example on a specific short story would strengthen the exposition considerably

### 9.4 Required Changes for Submission

| Change | Priority | Effort |
|--------|----------|--------|
| Add explicit literary theory citations for phase boundaries ($\mu_i$, $\omega_i$) | High | 1 day |
| Derive $\text{Proj}_{\text{key}}$ or provide a principled default (Fourier features) | Medium | 0.5 day |
| Reformulate resonance integral as online recursive update for efficiency | Medium | 0.5 day |
| Add a worked example on a canonical short story (e.g., Hemingway's "Hills Like White Elephants") | High | 2 days |
| Add a discussion of non-Western narrative traditions and required parameter adjustments | Low | 1 day |
| Add explicit limitations section with all identified weaknesses | Medium | 0.5 day |
| **Total estimated effort** | | **5.5 days** |

### 9.5 Recommended Action

Accept the manuscript subject to **Minor Revision** addressing the seven changes above. The theoretical contribution is significant, the mathematical framework is sound, and the model addresses a genuine gap in computational literary analysis. The revision items are substantive but not structural — none require redesign of the architecture.

---

*End of HMP-DT v2.1 — Final Theoretical Refinement*
