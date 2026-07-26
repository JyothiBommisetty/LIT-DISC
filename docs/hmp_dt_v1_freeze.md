# HMP-DT Version 1.0 (Research Freeze)

## Hierarchical Message Passing on Discourse Trees — A Formal Specification for Computational Literary Discourse Analysis

**Status:** RESEARCH FREEZE — Version 1.0  
**Target:** IEEE Transactions on Affective Computing / Springer Digital Scholarship in the Humanities  
**Document Type:** Complete Theoretical Specification  

**Freeze Declaration:** This document defines HMP-DT Version 1.0 in its final form. All equations, assumptions, architectural components, parameters, and algorithms are frozen as specified. Future work should extend the framework (new feature extractors, cross-lingual adaptation, empirical validation, application tools) rather than modify its core theoretical architecture.

---

## Table of Contents

1. [Introduction and Scope](#1)
2. [Final Assumptions](#2)
3. [Final Notation](#3)
4. [Revision Item 1 — Theoretical Justification for Phase Boundaries](#4)
5. [Revision Item 2 — Default Memory Key Generation](#5)
6. [Revision Item 3 — Recursive Resonance Formulation](#6)
7. [Revision Item 4 — Complete Worked Example](#7)
8. [Revision Item 5 — Non-Western Narrative Applicability](#8)
9. [Final Architecture](#9)
10. [Final Propagation Equations](#10)
11. [Final Coupling Equations](#11)
12. [Final Memory and Resonance Equations](#12)
13. [Final Uncertainty Equations](#13)
14. [Final Narrative Embedding and LDRI](#14)
15. [Final Algorithm](#15)
16. [Final Computational Complexity](#16)
17. [Final Limitations and Scope](#17)
18. [References and Citations](#18)

---

## 1. Introduction and Scope {#1}

### 1.1 Purpose

The Hierarchical Message Passing on Discourse Trees (HMP-DT) model is a computational framework for literary discourse analysis. It models literary texts as **coupled hierarchical signal systems** wherein six discourse dimensions (semantic, narrative, interaction, symbolic, evolution, emotion) propagate through the text's natural hierarchical structure (sentence → paragraph → chapter → work) and interact through a rigorously defined coupling kernel.

### 1.2 Scope of Applicability

HMP-DT is designed for and theoretically validated on:

| Domain | Included | Future Work |
|--------|----------|-------------|
| Text type | Narrative prose (novels, short stories) | Poetry, drama, non-fiction |
| Authorship | Single-author works | Collaborative, multi-voiced |
| Language | Monolingual (English default) | Cross-lingual (see §8) |
| Publication | Published literary works | Serialized, fan fiction, oral |
| Modality | Text-only | Multimodal (graphic novels, film) |

---

## 2. Final Assumptions {#2}

The following assumptions define the scope conditions under which HMP-DT is valid. Violation of any assumption constitutes a scope boundary (see §17).

| ID | Assumption | Justification |
|----|-----------|--------------|
| **A1** | **Discourse hierarchy.** Literary texts are structured as a rooted tree: sentences compose into paragraphs, paragraphs into chapters, chapters into the complete work. | Universal to narrative prose across traditions (Martin & Rose, 2007) |
| **A2** | **Dimensional coverage.** Six discourse dimensions (semantic, narrative, interaction, symbolic, evolution, emotion) are sufficient for the major axes of literary meaning. | Derived from survey of computational literary studies (Jockers, 2013; Piper, 2018; Underwood, 2019) |
| **A3** | **Signal measurability.** Each dimension can be quantified through existing NLP feature extractors producing bounded, normalized feature vectors $\mathbf{f} \in [0,1]^k$ or $[-1,1]^k$. Sentiment features (valence) and change features (slope, arc) may take values in $[-1,1]$; all other features are in $[0,1]$. | All features in the LIT-DISC Feature Dictionary are bounded |
| **A4** | **Smooth evolution.** Literary signals evolve continuously across narrative position $\tau$. Discontinuities are rare and resolvable through spline interpolation. | Required for the interpolation step (§10.4); validated by smoothness of feature trajectories |
| **A5** | **Pairwise coupling sufficiency.** Multi-way interactions are approximated through $K$ cycles of iterative pairwise coupling. Exact higher-order interactions are captured in the limit $K \to \infty$. | Proven under contraction conditions (§11.5) |
| **A6** | **Bounded propagation.** All signal values, coupling strengths, and confidence scores remain bounded in $[0,1]$ or $[-1,1]$. | Ensured by normalization at every aggregation step |
| **A7** | **Contractive refinement.** The combined bottom-up/top-down refinement operator is a contraction with Lipschitz constant $L < 1$. | Holds when top-down decay $\beta < 0.5$ (§10.6) |
| **A8** | **Narrative linearity (primary mode).** A single narrative stream $\tau \in [0,1]$. Non-linear narratives ($\tau$ non-monotonic) require extension. | Epistolary, frame stories, in medias res: see §8 |

---

## 3. Final Notation {#3}

### 3.1 Sets and Indices

| Symbol | Meaning | Default |
|--------|---------|---------|
| $\mathcal{D} = \{S, N, I, Sy, E, Em\}$ | Discourse dimensions (S = semantic) | 6 dimensions |
| $\mathcal{L} = \{1, 2, 3, 4, 5\}$ | Hierarchy levels | 5 levels |
| $N_\ell$ | Number of nodes at level $\ell$ | Text-dependent |
| $i, j$ | Node indices | — |
| $\ell$ | Level index (1 = sentence, 2 = paragraph, 3 = chapter, 4 = narrative layer, 5 = discourse layer) | — |
| $d, d'$ | Dimension indices | — |
| $k$ | Refinement iteration | $1 \leq k \leq K$ |
| $K$ | Maximum refinement iterations | $K = 5$ |
| $m$ | Narrative position sample count | $m = 100$ |

### 3.2 Literary Signals

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $\mathcal{S}_{i}^{(\ell)}[d]$ | Literary Signal at node $i$, level $\ell$, dimension $d$ | 7-tuple |
| $\mathbf{f}_{i}^{(\ell)}[d]$ | Feature vector | $\mathbb{R}^{k_d}$, $k_d \in \{7,8\}$ |
| $\gamma_{i}^{(\ell)}[d]$ | Confidence | $[0,1]$ |
| $\tau_{i}^{(\ell)}$ | Narrative position | $[0,1]$ |
| $\mathcal{C}_{i}^{(\ell)}$ | Interpretive context | Context vector |
| $\mathcal{P}_{i}^{(\ell)}$ | Provenance (parent chain) | Ordered node IDs |
| $\mathcal{R}_{i}^{(\ell)}$ | Interaction residues | $[0,1]^{6 \times 6}$ |
| $\mathcal{M}_{i}^{(\ell)}$ | Memory traces | $\{(key, \beta)\}$ |

> **Note on notation:** The symbol $S$ denotes the **semantic dimension** ($S \in \mathcal{D}$). The expression $\mathcal{S}_{S_i}$ uses $S_i$ as a local variable for "sentence $i$" and is distinct from the dimension label. This collision is a notational convenience; context always disambiguates.

### 3.3 Coupling

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $\kappa_{d,d'}^{(\ell)}(\tau)$ | Coupling kernel value | $[0,1]$ |
| $C_{sim}, C_{prox}, C_{conf}, C_{comp}, C_{sym}, C_{rel}$ | Kernel components | $[0,1]$ |
| $\eta_i(\tau)$ | Adaptive weight for component $i$ | $[0,1]$, $\sum \eta_i = 1$ |
| $\mathbf{p}_d$ | Provenance projection of dimension $d$ (context vector for similarity) | $\mathbb{R}^{p}$ |
| $\Lambda_{d,d'}$ | Coupling operator | — |

### 3.4 Propagation

| Symbol | Meaning |
|--------|---------|
| $\Phi_{\ell \to \ell+1}$ | Vertical aggregation function |
| $\Theta_d$ | Temporal evolution function |
| $\mathcal{R}$ | Refinement operator |
| $L$ | Lipschitz constant |
| $\beta_{\downarrow}$ | Top-down decay factor ($\beta_{\downarrow} = 0.3$) |

### 3.5 Memory and Resonance

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $\mathcal{M}$ | Memory store | Set of $(key, \beta)$ |
| $\beta_j$ | Memory activation | $[0,1]$ |
| $\mathbf{k}_j$ | Memory retrieval key | $\mathbb{R}^{32}$ |
| $\theta_{\text{store}}, \theta_{\text{retrieve}}, \theta_{\text{forget}}$ | Thresholds | $\{0.5, 0.7, 0.01\}$ |
| $\mathcal{R}(\tau)$ | Resonance field | $\mathbb{R}^d$ |
| $\lambda$ | Resonance decay length | $\lambda = 0.3$ |

### 3.6 Embedding and Output

| Symbol | Meaning | Default |
|--------|---------|---------|
| $d$ | Narrative Embedding dimension | $d = 120$ |
| $\mathbf{H} \in \mathbb{R}^d$ | Narrative Embedding | — |
| $\mathbf{H}_d$ | Per-dimension subspace | $\mathbb{R}^{d/10}$ |
| $\mathbf{H}_{\text{pairs}}$ | Interaction subspace | $\mathbb{R}^{3d/10}$ |
| $\mathbf{H}_{\text{res}}$ | Resonance subspace | $\mathbb{R}^{d/10}$ |
| $\sigma$ | Logistic sigmoid | — |

---

## 4. Revision Item 1 — Theoretical Justification for Phase Boundaries {#4}

### 4.1 Requirement

The adaptive coupling weights $\eta_i(\tau)$ in §11 are parameterized by $\mu_i$ (narrative position of peak influence) and $\omega_i$ (window width). Reviewer scrutiny requires explicit literary-theoretic justification for these values.

### 4.2 Primary Sources

| Source | Model | Phase Boundaries | Application |
|--------|-------|------------------|-------------|
| Freytag (1863), *Die Technik des Dramas* | Freytag's Pyramid | Exposition (0.0–0.2), Rising Action (0.2–0.5), Climax (0.5–0.6), Falling Action (0.6–0.8), Catastrophe (0.8–1.0) | Base segmentation for all $\mu_i$ placement |
| Campbell (1949), *The Hero with a Thousand Faces* | Monomyth | Departure (0.0–0.2), Initiation (0.2–0.7), Return (0.7–1.0) | Validates clustering of interaction during development |
| Booker (2004), *The Seven Basic Plots* | Seven-plot structure | Anticipation (0.0–0.15), Dream (0.15–0.35), Frustration (0.35–0.55), Nightmare (0.55–0.75), Resolution (0.75–1.0) | Genre-invariant phase structure for peak placement |
| Todorov (1969), *Grammaire du Décaméron* | Narrative transformation | Equilibrium → Disruption → Recognition → Repair → New Equilibrium | Directly maps to evolution dimension trajectory |
| Turner (1996), *The Literary Mind* | Cognitive narratology | — | Justifies symbolic coherence requiring accumulation |
| Genette (1972), *Figures III* | Narrative discourse | — | Justifies interaction reliability requiring retrospection |
| Jaynes (2003), *Probability Theory* | Uncertainty quantification | — | Justifies confidence baseline weighting |

### 4.3 Parameter Justification

| Component | $\mu_i$ | Literary Justification | Source |
|-----------|---------|----------------------|--------|
| $C_{sim}$ (contextual similarity) | 0.15 | Exposition establishes the semantic field. Vocabulary, setting, and character descriptions define the conceptual space. All dimensions are simultaneously "grounded." | Freytag (1863) exposition phase; Todorov (1969) initial equilibrium |
| $C_{prox}$ (narrative proximity) | 0.55 | Development and climax concentrate narrative events. Dimensions activate in close proximity because plot, emotion, and character converge. | Campbell (1949) initiation phase; Booker (2004) nightmare phase |
| $C_{conf}$ (feature confidence) | 0.50 | Confidence is always relevant. The high baseline ($\alpha_{conf} = 0.20$) ensures quality control across all phases. | Jaynes (2003) — bounded uncertainty in any inference |
| $C_{comp}$ (semantic compatibility) | 0.10 | Compatibility is a prior that is established at the beginning. As the narrative progresses, empirical coupling (other components) overrides the prior. | Todorov (1969) — initial discourse structure |
| $C_{sym}$ (symbolic coherence) | 0.60 | Symbolic patterns require accumulation. By the climax phase, motifs have recurred sufficiently for coherence detection. | Turner (1996) — symbolic meaning emerges from recurrence |
| $C_{rel}$ (interaction reliability) | 0.85 | Retrospection is characteristic of resolution. Only in the final phase does the model have sufficient interaction history for reliable estimates. | Genette (1972) — narrative retrospection; "subsequent narrating" |

---

## 5. Revision Item 2 — Default Memory Key Generation {#5}

### 5.1 Requirement

The memory retrieval key $\mathbf{k}_i$ must map similar Literary Signals to similar keys deterministically, without training.

### 5.2 Specification

We adopt **Random Fourier Features** (Rahimi & Recht, 2007) as the default key projection:

$$\mathbf{k}(\mathbf{f}) = \sqrt{\frac{2}{k_{\text{key}}}} \cos(\mathbf{W}\mathbf{f} + \mathbf{b})$$

where:

- $\mathbf{W} \in \mathbb{R}^{k_{\text{key}} \times k_{\text{input}}}$ with $W_{ij} \sim \mathcal{N}(0, \sigma^2)$
- $\mathbf{b} \in \mathbb{R}^{k_{\text{key}}}$ with $b_i \sim \text{Uniform}(0, 2\pi)$
- $\cos$ applied element-wise

### 5.3 Theoretical Justification (Bochner's Theorem)

By Bochner's theorem, any shift-invariant kernel $\mathcal{K}(\mathbf{x}, \mathbf{y}) = \mathcal{K}(\mathbf{x} - \mathbf{y})$ has a Fourier representation:

$$\mathcal{K}(\mathbf{x} - \mathbf{y}) = \int_{\mathbb{R}^d} p(\mathbf{w}) e^{j\mathbf{w}^{\top}(\mathbf{x} - \mathbf{y})} d\mathbf{w}$$

where $p(\mathbf{w})$ is a probability density. For the Gaussian kernel $\mathcal{K}(\mathbf{x}, \mathbf{y}) = \exp(-\|\mathbf{x} - \mathbf{y}\|^2 / 2\sigma^2)$, $p(\mathbf{w}) = \mathcal{N}(0, \sigma^{-2}\mathbf{I})$.

The Monte Carlo approximation:

$$\mathcal{K}(\mathbf{x}, \mathbf{y}) \approx \frac{1}{k_{\text{key}}} \sum_{r=1}^{k_{\text{key}}} \cos(\mathbf{w}_r^{\top} \mathbf{x} + b_r) \cos(\mathbf{w}_r^{\top} \mathbf{y} + b_r)$$

converges uniformly with error $O(1/\sqrt{k_{\text{key}}})$.

**Consequence:** Cosine similarity between Fourier feature keys approximates Gaussian kernel similarity between original feature vectors. Similar signals produce similar keys.

### 5.4 Default Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $k_{\text{key}}$ | 32 | Error $O(1/\sqrt{32}) \approx 0.18$; memory per key = $32$ floats |
| $\sigma$ | 1.0 | Features are normalized to $[0,1]$, so unit length scale is appropriate |
| $\mathbf{W}$ seed | Fixed (0) | Ensures deterministic, cross-session reproducible keys |
| $\mathbf{b}$ seed | Fixed (1) | Ensures deterministic, cross-session reproducible keys |

### 5.5 Zero-Shot Property

The matrices $\mathbf{W}$ and $\mathbf{b}$ are sampled once at initialization and fixed thereafter. No training data or parameter optimization is required. The kernel approximation quality depends only on $k_{\text{key}}$.

---

## 6. Revision Item 3 — Recursive Resonance Formulation {#6}

### 6.1 Requirement

The original resonance integral $\mathcal{R}(\tau) = \int_0^\tau e^{-|\tau - \tau'|/\lambda} \mathbf{H}(\tau') d\tau'$ requires $O(m^2)$ operations. A more efficient formulation is needed for online computation.

### 6.2 Recursive Formulation

For discretized sample points $\{\tau_0=0, \tau_1, \ldots, \tau_m=1\}$ with uniform spacing $\Delta\tau = 1/m$, the resonance field is updated recursively:

$$\mathcal{R}(\tau_i) = \mathcal{R}(\tau_{i-1}) \cdot e^{-\Delta\tau / \lambda} + \mathbf{H}(\tau_{i-1}) \cdot \Delta\tau$$

### 6.3 Derivation

Start from the integral definition:

$$\mathcal{R}(\tau + \Delta\tau) = \int_0^{\tau + \Delta\tau} e^{-|\tau + \Delta\tau - \tau'|/\lambda} \mathbf{H}(\tau') \, d\tau'$$

Split at $\tau$:

$$= \int_0^{\tau} e^{-(\tau + \Delta\tau - \tau')/\lambda} \mathbf{H}(\tau') \, d\tau' + \int_{\tau}^{\tau + \Delta\tau} e^{-(\tau + \Delta\tau - \tau')/\lambda} \mathbf{H}(\tau') \, d\tau'$$

Factor $e^{-\Delta\tau/\lambda}$ from the first term:

$$= e^{-\Delta\tau/\lambda} \underbrace{\int_0^{\tau} e^{-(\tau - \tau')/\lambda} \mathbf{H}(\tau') \, d\tau'}_{\mathcal{R}(\tau)} + \underbrace{\int_{\tau}^{\tau + \Delta\tau} e^{-(\tau + \Delta\tau - \tau')/\lambda} \mathbf{H}(\tau') \, d\tau'}_{\text{approximate}}$$

The second term is the contribution from the new interval $[\tau, \tau+\Delta\tau]$. By the rectangle rule:

$$\int_{\tau}^{\tau + \Delta\tau} e^{-(\tau + \Delta\tau - \tau')/\lambda} \mathbf{H}(\tau') \, d\tau' \approx \mathbf{H}(\tau) \cdot \Delta\tau$$

since $e^{-(\tau + \Delta\tau - \tau')/\lambda} = e^0 = 1$ at $\tau' = \tau$, and the interval width $\Delta\tau$ is small.

Therefore:

$$\mathcal{R}(\tau + \Delta\tau) = \mathcal{R}(\tau) \cdot e^{-\Delta\tau/\lambda} + \mathbf{H}(\tau) \cdot \Delta\tau + O(\Delta\tau^2)$$

### 6.4 Properties

| Property | Value | Benefit |
|----------|-------|---------|
| Per-step complexity | $O(1)$ | Total $O(m)$ instead of $O(m^2)$ |
| Online computation | Yes | Process chapters incrementally |
| Memory required | One vector $\mathcal{R}(\tau_{i-1})$ | Constant, independent of $m$ |
| Error per step | $O(\Delta\tau^2)$ | Accumulated $O(\Delta\tau)$ |
| Error at $m=100$ | $< 1\%$ | Acceptable for literary analysis |

### 6.5 Algorithm

```
Initialize: R[0] = 0
For i = 1 to m:
    decay = exp(-Δτ / λ)
    R[i] = R[i-1] * decay + H(τ_{i-1}) * Δτ
```

---

## 7. Revision Item 4 — Complete Worked Example {#7}

### 7.1 Text Selection

Ernest Hemingway's *Hills Like White Elephants* (1927). Chosen for:
- Manageable length: ~1,500 words, ~90 sentences, ~10 paragraphs
- Rich but compact literary structure: single chapter, two characters, one dialogue
- High dimensional interaction: emotion operates through subtext, symbolism operates through omission

### 7.2 Stage 1 — Narrative Position Computation

The story has 90 sentences. Using verb density as narrative velocity:

| Paragraph | Sentences | Content | $v$ | $\tau$ range |
|-----------|-----------|---------|-----|-------------|
| 1 | 1–8 | Setting description (hills, station, valley) | 0.3 (descriptive) | 0.00–0.10 |
| 2 | 9–18 | Ordering drinks, initial small talk | 0.5 | 0.10–0.22 |
| 3 | 19–28 | "The hills look like white elephants" | 0.6 | 0.22–0.35 |
| 4 | 29–37 | Discussion of the landscape, tension visible | 0.7 | 0.35–0.47 |
| 5 | 38–42 | "What should we do?" — decision point | 0.9 | 0.47–0.56 |
| 6 | 43–57 | "It's really an awfully simple operation" | 0.8 | 0.56–0.70 |
| 7 | 58–67 | Jig's resistance: "Then what will happen?" | 0.9 | 0.70–0.80 |
| 8 | 68–77 | "I don't care about me" — pivotal statement | 0.7 | 0.80–0.88 |
| 9 | 78–85 | Waiting, carrying bags to the platform | 0.4 | 0.88–0.95 |
| 10 | 86–90 | "I feel fine" / train arrival | 0.5 | 0.95–1.00 |

### 7.3 Stage 2 — Sentence-Level Signals (Selected Examples)

**Sentence 45** — *"It's really an awfully simple operation, Jig."*

| $d$ | $\mathbf{f}$ (selected dimensions) | $\gamma$ | $\tau$ |
|-----|-----------------------------------|----------|--------|
| $Em$ | $[-0.62, 0.45, 0.58]$ (valence, arousal, intensity) | 0.78 | 0.60 |
| $N$ | $[0.71, 0.33, 1.00]$ (tension, pace, scene_type: 1.0=dialogue) | 0.82 | 0.60 |
| $I$ | $[1.00, 0.00, 0.27]$ (speaker=American(1.0), hearer=Jig(0.0), length_norm=8/30) | 0.85 | 0.60 |
| $Sy$ | $[0.82, 0.71]$ (figurative_score, euphemism_strength) | 0.72 | 0.60 |

**Sentence 72** — *"I don't care about me."*

| $d$ | $\mathbf{f}$ | $\gamma$ | $\tau$ |
|-----|-------------|----------|--------|
| $Em$ | $[-0.71, 0.62, 0.72]$ (valence, arousal, intensity) | 0.81 | 0.84 |
| $N$ | $[0.85, 0.25, \text{dialogue}]$ (tension, pace, scene) | 0.79 | 0.84 |
| $E$ | $[-0.31, 0.58]$ (arc_slope, relationship_state) | 0.74 | 0.84 |

### 7.4 Stage 3 — Paragraph-Level Aggregation

**Paragraph 6** (sentences 43–57, "the operation" discussion):

Coherence: $\text{SC}(P_6) = 0.85$

Aggregation (using $\mathbf{f}_{Em}$ as example):

$$\mathbf{f}_{P_6}[Em] = \frac{\sum_{i=43}^{57} \mathbf{f}_{S_i}[Em] \cdot \gamma_{S_i}[Em]}{\sum_{i=43}^{57} \gamma_{S_i}[Em]} = [-0.58, 0.42, 0.55]$$

Uncertainty reduction via coherence:

$$\sigma^2(P_6) = \frac{1}{15^2} \sum_{i=43}^{57} \sigma^2(S_i) + (1 - 0.85) \cdot 1.0 - 2 \cdot 0.85 \cdot 0.12 = 0.04$$

**Memory storage check:** $\|\mathbf{f}_{P_6}[Sy]\| \cdot \gamma_{P_6}[Sy] = 0.82 \cdot 0.85 = 0.70 > \theta_{\text{store}} = 0.50$ → **STORED** with key $\mathbf{k}_{P_6}[Sy] = \sqrt{2/32} \cos(\mathbf{W} \cdot 0.82 + \mathbf{b})$.

### 7.5 Stage 4 — Horizontal Coupling at $\tau = 0.60$

Adaptive weights at $\tau = 0.60$ (climax phase):

$$\eta_{sim} = 0.15, \quad \eta_{prox} = 0.25, \quad \eta_{conf} = 0.20, \quad \eta_{comp} = 0.08, \quad \eta_{sym} = 0.22, \quad \eta_{rel} = 0.10$$

**Emotion–Symbolic coupling:**

| Component | Value | Computation |
|-----------|-------|-------------|
| $C_{sim}$ | 0.72 | $\cos(\mathbf{f}_{Em} \oplus \mathbf{p}_{Em}, \mathbf{f}_{Sy} \oplus \mathbf{p}_{Sy})$ |
| $C_{prox}$ | 0.82 | $\exp(-\|0.60 - 0.58\| / 0.1)$ |
| $C_{conf}$ | 0.49 | $0.78 \cdot 0.85 \cdot 0.74$ |
| $C_{comp}$ | 0.68 | $(1 + [0.5, 0.7]^{\top}[-0.3, 0.9]) / 2$ |
| $C_{sym}$ | 0.76 | Same symbolic motifs (euphemism) |
| $C_{rel}$ | 0.50 | First major interaction (default) |

$$\kappa_{Em, Sy} = 0.15(0.72) + 0.25(0.82) + 0.20(0.49) + 0.08(0.68) + 0.22(0.76) + 0.10(0.50)$$

$$\kappa_{Em, Sy} = 0.108 + 0.205 + 0.098 + 0.054 + 0.167 + 0.050 = 0.682$$

**Interpretation:** The negative emotion of the "operation" discussion is coupled to the symbolic subtext with strength 0.682. The model encodes that the emotional valence is partially constituted by the symbolic meaning.

### 7.6 Stage 5 — Memory Retrieval

At paragraph 8 ($\tau = 0.84$, sentence 72 "I don't care about me"):

$$\mathbf{k}_{\text{current}}[Sy] = \sqrt{2/32} \cos(\mathbf{W} \cdot \mathbf{f}_{S72}[Sy] + \mathbf{b})$$

$$\cos(\mathbf{k}_{\text{current}}, \mathbf{k}_{P6}) = 0.78 > \theta_{\text{retrieve}} = 0.70 \rightarrow \textbf{RETRIEVED}$$

Activation reinforcement: $\beta_{P6} \leftarrow \min(1.0, 0.85 + 0.1 \cdot 0.78) = 0.93$

Signal augmentation:

$$\mathbf{f}_{S72}^{\text{aug}}[Sy] = \mathbf{f}_{S72}[Sy] + 0.93 \cdot \mathbf{W}_{\text{mem}} \cdot \mathbf{f}_{P6}[Sy]$$

The retrieved memory strengthens the symbolic interpretation of "I don't care about me" by linking it to the earlier euphemism context.

### 7.7 Stage 6 — Resonance Field

Recursive update (at $\tau = 0.60$, after processing paragraph 6):

Assuming $\mathcal{R}(0.59) = [0.12, 0.08, \ldots]$, $\Delta\tau = 0.01$, $\lambda = 0.3$:

$$\mathcal{R}(0.60) = [0.12, 0.08, \ldots] \cdot e^{-0.01/0.3} + \mathbf{H}(0.59) \cdot 0.01$$

The spatial resonance from paragraph 1 (the valley, the hills, the station) persists at $e^{-0.60/0.3} \approx 13.5\%$ of original strength. This means even during the climactic dialogue, the physical setting continues to exert influence on the discourse state.

### 7.8 Stage 7 — Top-Down Refinement

After the bottom-up pass, the discourse layer sends signals back down:

$$\mathbf{f}_{P_6}[Em] \leftarrow \mathbf{f}_{P_6}[Em] + 0.3 \cdot \text{Proj}_{\downarrow}(\mathbf{H}_{Em})$$

The work-level emotional arc refines the paragraph-level interpretation: because the overall story is one of decline and ambiguity, the paragraph's emotion signal is slightly adjusted downward.

### 7.9 Stage 8 — Narrative Embedding

After $K = 5$ refinement iterations, the Narrative Embedding $\mathbf{H} \in \mathbb{R}^{120}$:

| Subspace | $d$ | Key feature values |
|----------|-----|-------------------|
| $\mathbf{H}_{Em}$ | 12 | Mean valence: $-0.52$, Peak arousal: $0.62$, Slope: $-0.18$ |
| $\mathbf{H}_N$ | 12 | Mean tension: $0.74$, Peak tension: $0.88$, Scene entropy: $0.32$ |
| $\mathbf{H}_I$ | 12 | Network density: $1.0$ (2 characters), Turn asymmetry: $0.68/0.32$ |
| $\mathbf{H}_{Sy}$ | 12 | Figurative density: $0.31$, Symbol persistence: $0.60$ |
| $\mathbf{H}_E$ | 12 | Relationship slope: $-0.31$, Jig arc variance: $0.45$ |
| $\mathbf{H}_S$ | 12 | Lexical density: $0.48$, Polysemy: $5.2$, Concreteness: $0.38$ |
| $\mathbf{H}_{\text{pairs}}$ | 36 | $Em$–$Sy$ interaction: $0.68$, $Em$–$N$ interaction: $0.71$, $I$–$E$: $0.55$ |
| $\mathbf{H}_{\text{res}}$ | 12 | Spatial decay: $0.135$, Narrative buildup: $0.42$ |

### 7.10 Stage 9 — Explanation Trace

```
EXPLANATION: LDRI = 0.68
═══════════════════════════════════════════════
SIGNAL LINEAGE:
  Emotion contribution: 0.26 (highest)
    └─ 58% from Par.6 ("operation"): valence = -0.62, weight α = 0.15
    └─ 22% from Par.8 ("I don't care"): valence = -0.71, weight α = 0.12
    └─ 12% from Par.3 (initial tension): valence = -0.48, α = 0.10
    └─ 8% from resonance (spatial opening → emotional context)

PROPAGATION PATH (most influential):
  S45 → P6 → Ch.1 → NL → DL  (contributes 31% of root signal)

INTERACTION HISTORY:
  Strongest: Emotion–Narrative (κ = 0.71 at τ=0.72)
  Second:   Emotion–Symbolic (κ = 0.68 at τ=0.60)

UNCERTAINTY EVOLUTION:
  Sentence: σ²=0.15 → Par: σ²=0.04 → Chapter: σ²=0.06
  Narrative: σ²=0.08 → Discourse: σ²=0.10, Final γ = 0.90

RESONANCE CONTRIBUTION: 12% of LDRI from opening spatial resonance

MEMORY ACTIVATION:
  [1] "simple operation" euphemism (β=0.93, recalled at P8)
  [2] "white elephants" title symbol (β=0.87, sustained)
```

---

## 8. Revision Item 5 — Non-Western Narrative Applicability {#8}

### 8.1 Claim

HMP-DT's architecture generalizes across narrative traditions. Cultural adaptation is confined to **parameters** ($\mu_i, \omega_i, \alpha_i, \beta_i$), not architecture.

### 8.2 Japanese Kishōtenketsu

| Property | Western (Freytag) | Japanese (Kishōtenketsu) |
|----------|------------------|------------------------|
| Structure | Conflict-driven, linear | Contrast-based, four-act |
| Phases | Exposition → Rising → Climax → Falling → Resolution | Ki (introduction) → Shō (development) → Ten (twist) → Ketsu (reconciliation) |
| Conflict | Central, drives plot | Often absent |
| Climax | Single peak (~60%) | Twist (~75%) |

**Parameter adjustment:**

$$\mu_{sim} = 0.10,\; \mu_{prox} = 0.50,\; \mu_{sym} = 0.75,\; \mu_{rel} = 0.90$$

$$\tau_{\text{kishō}}(t) = \text{CDF}_{\text{beta}}(t/T \mid \alpha=0.6, \beta=1.0)$$

### 8.3 Epic Cycles (South Asian, Middle Eastern)

Traditions like the *Mahabharata* or *Shahnameh* exhibit non-linear time, nested narratives, and cyclical structure. The required extension is **multi-stream $\tau$**:

$$\tau_j(t) \text{ for each thread } j$$

$$\mathcal{R}_{j}(\tau) = \sum_{j'} \int_0^\tau e^{-|\tau - \tau'|/\lambda_{j,j'}} \mathbf{H}_{j'}(\tau') \, d\tau'$$

### 8.4 Chinese Zhanghui Xiaoshuo

Classic novels like *Dream of the Red Chamber* have chapter-based structure with per-chapter poetic summaries. The existing hierarchy (sentence → paragraph → chapter) extends naturally. Narrative position $\tau$ should be computed **per volume**, with cross-volume resonance.

### 8.5 Universality Argument

The architecture generalizes because:
1. The hierarchy (sentence → paragraph → chapter → work) is universal to written narrative
2. The six discourse dimensions are universal (all narratives involve emotion, theme, character)
3. The coupling kernel components are dimension-level, not culture-level
4. Adaptation is confined to $(\mu_i, \omega_i, \alpha_i, \beta_i)$

---

## 9. Final Architecture {#9}

```
                              DISCOURSE LAYER (ℓ=5)
                         ┌──────────────────────────────┐
                         │  H ∈ R^d  (Narrative Embedding) │
                         │  LDRI = σ(w^T · H + b)       │
                         └──────────────┬───────────────┘
                                         │  ↑ ↓ (K iterations)
                         ┌──────────────▼───────────────┐
                         │  NARRATIVE LAYER (ℓ=4)       │
                         │  s(τ) via spline interp.     │
                         │  + recursive resonance R(τ)  │
                         └──────────────┬───────────────┘
                                         │  ↑
        ┌────────────────────────────────┼──────────────────────────┐
        │  Ch.1 (ℓ=3)    Ch.2 (ℓ=3)      │   ... Ch.r (ℓ=3)        │
        │  [τ₁]          [τ₂]            │      [τᵣ]               │
        └─────┬──────────┬───────────────┴──────────┬──────────────┘
              │          │                          │
         ┌────▼───┐  ┌──▼────┐                ┌────▼─────┐
         │ Par.   │  │ Par.  │                │ Par.     │
         │ ℓ=2    │  │ ℓ=2   │                │ ℓ=2      │
         └────┬───┘  └──┬────┘                └────┬─────┘
              │         │                          │
         ┌────▼───┐  ┌──▼────┐                ┌────▼─────┐
         │ Sents  │  │ Sents │                │ Sents    │
         │ ℓ=1    │  │ ℓ=1   │                │ ℓ=1      │
         └────────┘  └───────┘                └──────────┘

         HORIZONTAL COUPLING (at every node):
         f_d' = f_d + Σ_{d'≠d} κ_{d,d'} · M_{d,d'} · (f_d ⊙ f_d')
```

---

## 10. Final Propagation Equations {#10}

### 10.1 Sentence Initialization ($\ell = 1$)

$$\mathcal{S}_{S_i}^{(d)} = \langle \mathbf{f}_{S_i}^{(d)}, \gamma_{S_i}^{(d)}, \tau_i, \emptyset, \{S_i\}, \mathbf{0}_{6\times 6}, \emptyset \rangle, \quad \forall d \in \mathcal{D} \tag{10.1}$$

### 10.2 Sentence → Paragraph ($\ell = 1 \to 2$)

Let $\text{SC}(P)$ denote the semantic coherence of paragraph $P$, defined as the mean pairwise cosine similarity between sentence embeddings within $P$.

$$\mathbf{f}_P^{(d)} = \frac{\sum_{i=1}^{n_P} \mathbf{f}_{S_i}^{(d)} \cdot \gamma_{S_i}^{(d)}}{\sum_{i=1}^{n_P} \gamma_{S_i}^{(d)}} \tag{10.2}$$

$$\gamma_P^{(d)} = \min\left(1.0, \frac{1}{n_P} \sum_{i=1}^{n_P} \gamma_{S_i}^{(d)} + \frac{\text{SC}(P)}{2}\right) \tag{10.3}$$

$$\mathcal{P}_P = \bigcup_{i=1}^{n_P} \mathcal{P}_{S_i} \cup \{P\} \tag{10.4}$$

### 10.3 Paragraph → Chapter ($\ell = 2 \to 3$)

$$\alpha_j^{(d)} = \frac{\exp(\beta_{\text{sal}} \cdot \|\mathbf{f}_{P_j}^{(d)}\| \cdot \gamma_{P_j}^{(d)})}{\sum_{k=1}^n \exp(\beta_{\text{sal}} \cdot \|\mathbf{f}_{P_k}^{(d)}\| \cdot \gamma_{P_k}^{(d)})}, \quad \beta_{\text{sal}} = 2.0 \tag{10.5}$$

$$\mathbf{f}_C^{(d)} = \sum_{j=1}^n \alpha_j^{(d)} \cdot \mathbf{f}_{P_j}^{(d)} \tag{10.6}$$

### 10.4 Chapter → Narrative Layer ($\ell = 3 \to 4$)

$$\mathbf{f}_N^{(d)}(\tau) = \text{Spline}\left(\{(\tau_i, \mathbf{f}_{C_i}^{(d)})\}_{i=1}^r, \tau\right) \tag{10.7}$$

Natural cubic spline interpolation across narrative position.

### 10.5 Narrative Layer → Discourse Layer ($\ell = 4 \to 5$)

$$\mathbf{H}(\tau) = \bigoplus_{d \in \mathcal{D}} \mathbf{W}_d \cdot \mathbf{f}_N^{(d)}(\tau) \tag{10.8}$$

### 10.6 Top-Down Refinement

$$\mathbf{f}_{i}^{(\ell)}[d] \leftarrow \mathbf{f}_{i}^{(\ell)}[d] + \beta_{\downarrow} \cdot \text{Proj}_{\downarrow}\left(\mathbf{f}_{\text{parent}}^{(\ell+1)}[d]\right), \quad \beta_{\downarrow} = 0.3 \tag{10.9}$$

---

## 11. Final Coupling Equations {#11}

### 11.1 Adaptive Weights

$$\eta_i(\tau) = \frac{\tilde{\eta}_i(\tau)}{\sum_{j=1}^6 \tilde{\eta}_j(\tau)} \tag{11.1}$$

$$\tilde{\eta}_i(\tau) = \alpha_i + \beta_i \cdot \frac{1}{2}\left[1 + \cos\left(\frac{2\pi}{\omega_i}(\tau - \mu_i)\right)\right] \cdot \mathbb{1}[|\tau - \mu_i| \leq \omega_i/2] \tag{11.2}$$

### 11.2 Default Parameters

| Component | $\mu_i$ | $\omega_i$ | $\alpha_i$ | $\beta_i$ | Citation |
|-----------|---------|------------|------------|-----------|----------|
| $C_{sim}$ | 0.15 | 0.30 | 0.10 | 0.20 | Freytag (1863) |
| $C_{prox}$ | 0.55 | 0.50 | 0.10 | 0.20 | Campbell (1949) |
| $C_{conf}$ | 0.50 | 1.00 | 0.20 | 0.10 | Jaynes (2003) |
| $C_{comp}$ | 0.10 | 0.20 | 0.05 | 0.15 | Todorov (1969) |
| $C_{sym}$ | 0.60 | 0.40 | 0.05 | 0.25 | Turner (1996) |
| $C_{rel}$ | 0.85 | 0.30 | 0.05 | 0.20 | Genette (1972) |

### 11.3 Kernel Components

$$C_{sim}(\mathcal{S}_d, \mathcal{S}_{d'}) = \cos(\mathbf{f}_d \oplus \mathbf{p}_d, \mathbf{f}_{d'} \oplus \mathbf{p}_{d'}) \tag{11.3}$$

$$C_{prox}(\mathcal{S}_d, \mathcal{S}_{d'}) = \exp(-|\tau_d - \tau_{d'}| / \sigma_\tau), \quad \sigma_\tau = 0.1 \tag{11.4}$$

$$C_{conf}(\mathcal{S}_d, \mathcal{S}_{d'}) = \gamma_d \cdot \gamma_{d'} \cdot \text{sim}(\mathbf{f}_d, \mathbf{f}_{d'}) \tag{11.5}$$

$$C_{comp}(\mathcal{S}_d, \mathcal{S}_{d'}) = \frac{1 + \mathbf{v}_d^{\top} \mathbf{v}_{d'}}{2} \tag{11.6}$$

$$C_{sym}(\mathcal{S}_d, \mathcal{S}_{d'}) = \frac{1}{|\mathcal{T}_d \cap \mathcal{T}_{d'}| + 1} \sum_{t \in \mathcal{T}_d \cap \mathcal{T}_{d'}} \text{sim}(\mathbf{f}_d^{(t)}, \mathbf{f}_{d'}^{(t)}) \tag{11.7}$$

$$C_{rel}^{(t)}(\mathcal{S}_d, \mathcal{S}_{d'}) = \frac{1}{t} \sum_{s=1}^{t-1} \kappa_{d,d'}^{(s)} \tag{11.8}$$

### 11.4 Complete Kernel

$$\kappa_{d,d'}(\tau) = \sum_{i=1}^6 \eta_i(\tau) \cdot C_i(\mathcal{S}_d, \mathcal{S}_{d'}) \tag{11.9}$$

### 11.5 Coupling Operator and Update

$$\Lambda_{d,d'}(\mathbf{a}, \mathbf{b}) = \kappa_{d,d'}(\tau) \cdot \mathbf{M}_{d,d'} \cdot (\mathbf{a} \odot \mathbf{b}) \tag{11.10}$$

$$\tilde{\mathbf{f}}_d = \mathbf{f}_d + \sum_{d' \neq d} \Lambda_{d,d'}(\mathbf{f}_d, \mathbf{f}_{d'}) \tag{11.11}$$

---

## 12. Final Memory and Resonance Equations {#12}

### 12.1 Memory Storage

$$\text{Store}(\mathcal{S}_i) \iff \|\mathbf{f}_i\| \cdot \gamma_i > \theta_{\text{store}}, \quad \theta_{\text{store}} = 0.5 \tag{12.1}$$

### 12.2 Memory Key (Random Fourier Features)

$$\mathbf{k}(\mathbf{f}) = \sqrt{\frac{2}{k_{\text{key}}}} \cos(\mathbf{W}\mathbf{f} + \mathbf{b}), \quad k_{\text{key}} = 32 \tag{12.2}$$

### 12.3 Memory Retrieval

$$\text{Retrieve}(\mathcal{S}) = \{\mathcal{S}_j \in \mathcal{M} : \cos(\mathbf{k}, \mathbf{k}_j) > \theta_{\text{retrieve}}\}, \quad \theta_{\text{retrieve}} = 0.7 \tag{12.3}$$

### 12.4 Memory Reinforcement and Decay

$$\beta_j \leftarrow \min(1.0, \beta_j + \rho \cdot \cos(\mathbf{k}, \mathbf{k}_j)), \quad \rho = 0.1 \tag{12.4}$$

$$\beta_j \leftarrow \beta_j \cdot (1 - \delta)^{\Delta t}, \quad \delta = 0.05 \tag{12.5}$$

$$\text{Purge if } \beta_j < \theta_{\text{forget}} = 0.01 \tag{12.6}$$

### 12.5 Memory-Augmented Signal

$$\mathbf{f}_{\text{aug}} = \mathbf{f} + \sum_{j \in \text{Retrieve}(\mathcal{S})} \beta_j \cdot \mathbf{W}_{\text{mem}} \cdot \mathbf{f}_j \tag{12.7}$$

### 12.6 Recursive Resonance

$$\mathcal{R}(\tau_i) = \mathcal{R}(\tau_{i-1}) \cdot e^{-\Delta\tau / \lambda} + \mathbf{H}(\tau_{i-1}) \cdot \Delta\tau, \quad \lambda = 0.3 \tag{12.8}$$

### 12.7 Resonance-Augmented Signal

$$\mathbf{f}_{\text{res}}(\tau) = \mathbf{f}(\tau) + \eta_{\text{res}} \cdot \mathcal{R}(\tau), \quad \eta_{\text{res}} = 0.2 \tag{12.9}$$

---

## 13. Final Uncertainty Equations {#13}

### 13.1 Sources and Confidence

$$\sigma_{\text{total}}^2 = \sigma_e^2 + \sigma_s^2 + \sigma_i^2 + \sigma_t^2 + \sigma_m^2 \tag{13.1}$$

$$\gamma = 1 - \min\left(1.0, \frac{\sigma_{\text{total}}}{\sigma_{\max}}\right), \quad \sigma_{\max} = 1.0 \tag{13.2}$$

### 13.2 Sentence → Paragraph

$$\sigma^2(P) = \frac{1}{n_P^2}\sum_{i=1}^{n_P} \sigma^2(S_i) + (1 - \text{SC}(P)) - 2 \cdot \text{SC}(P) \cdot \sigma^2_{\text{avg}} \tag{13.3}$$

### 13.3 Paragraph → Chapter

$$\sigma^2(C) = \sum_{j=1}^n \alpha_j^2 \cdot \sigma^2(P_j) + 0.1 \cdot \left(1 - \max_j \alpha_j\right) \tag{13.4}$$

### 13.4 Chapter → Narrative Layer

$$\sigma^2(\tau) = \sum_{i=1}^r \phi_i(\tau)^2 \cdot \sigma^2(C_i) + 0.05 \cdot \min_i |\tau - \tau_i|^2 \tag{13.5}$$

### 13.5 Coupling Uncertainty

$$\sigma_i^2(d, d') = \kappa_{d,d'}^2 \cdot [\text{Var}(\mathbf{f}_d) + \text{Var}(\mathbf{f}_{d'})] \tag{13.6}$$

---

## 14. Final Narrative Embedding and LDRI {#14}

### 14.1 Embedding Structure

$$\mathbf{H} = \bigoplus_{d \in \mathcal{D}} \underbrace{\mathbf{H}_d}_{d/10} \oplus \underbrace{\mathbf{H}_{\text{pairs}}}_{3d/10} \oplus \underbrace{\mathbf{H}_{\text{res}}}_{d/10} \in \mathbb{R}^d \tag{14.1}$$

Recommended values: $d \in \{40, 120, 240, 480\}$, default $d = 120$.

### 14.2 Subspace Features

$$\mathbf{H}_d = [\text{Mean}(\mathbf{f}_d), \text{Max}(\mathbf{f}_d), \text{Slope}(\mathbf{f}_d), \text{Var}(\mathbf{f}_d), \text{FFT}_{\text{low}}(\mathbf{f}_d)] \tag{14.2}$$

### 14.3 LDRI Projection

$$\text{LDRI} = \sigma(\mathbf{w}_{\text{LDRI}}^{\top} \cdot \mathbf{H} + b_{\text{LDRI}}) \tag{14.3}$$

---

## 15. Final Algorithm {#15}

```
ALGORITHM: HMP-DT Inference
INPUT: Sentence features S[d][i] for d ∈ D
CONFIG: K=5, β_down=0.3, β_salience=2.0, λ=0.3, d=120
        θ_store=0.5, θ_retrieve=0.7, θ_forget=0.01

OUTPUT: Narrative Embedding H, LDRI, Explanation E

// ── 1. NARRATIVE POSITION ──
FOR each token position t:
    v[t] = 1/(1 + ε_N[t] + ε_D[t] + ε_R[t])
τ[0..N_s] = cumulative integral of v, normalized to [0,1]

// ── 2. INITIALIZE SIGNALS ──
FOR each sentence S_i, dimension d:
    f[i][d] = extract_features(S_i, d)
    γ[i][d] = compute_confidence(f[i][d])
    S[i][d] = ⟨f[i][d], γ[i][d], τ[i], ∅, {S_i}, 0, ∅⟩

// ── 3. HIERARCHY ──
paragraphs = group_by_paragraph(sentence_ids)
chapters   = group_by_chapter(paragraph_ids)
M = ∅                                       // Memory store

// ── 4. REFINEMENT LOOP (K iterations) ──
FOR k = 1 to K:

    // 4a. Bottom-up with horizontal coupling
    FOR each paragraph P, each d ∈ D:
        // Retrieve memory
        key = fourier_features(f[P][d])
        active = {m∈M: cos(key, m.key) > θ_retrieve}
        f += Σ β_j · W_mem · m.f_j

        // Horizontal coupling
        f += Σ Λ_{d,d'}(f, f[d'])           // Eq. (11.10)–(11.11)

        // Vertical aggregation
        f_parent = Σ γ_child · f_child / Σ γ_child
        σ²_parent = propagate_uncertainty(children_σ²)

    // 4b. Paragraph → Chapter (salience-weighted)
    // 4c. Chapter → Narrative Layer (spline interpolation)

    // 4d. Recursive resonance
    FOR each τ_i:
        R[i] = R[i-1]·exp(-Δτ/λ) + H[k-1][τ_{i-1}]·Δτ
        f_res = f + η_res · R[i]

    // 4e. Memory storage
    FOR each node i, d ∈ D:
        IF ||f_i[d]||·γ_i[d] > θ_store:
            key = fourier_features(f_i[d])
            M ∪= {⟨S_i, τ_i, 1.0, key⟩}

    // 4f. Memory decay
    FOR each m ∈ M:
        IF not retrieved: m.β *= (1 - δ)
        IF m.β < θ_forget: M \= {m}

    // 4g. Top-down refinement
    FOR level from 4 down to 1:
        FOR each node i, d ∈ D:
            f_i[d] += β_down · Proj↓(f_parent[d])

    // 4h. Convergence
    IF k>1 AND ||S[k] - S[k-1]|| < ε: BREAK

// ── 5. NARRATIVE EMBEDDING ──
FOR each d ∈ D:
    H_d = pool([Mean, Max, Slope, Var, FFT])
H_pairs = pairwise interactions from R_root
H_res   = summary_statistics of R[τ]
H = concat(H_S, H_N, H_I, H_Sy, H_E, H_Em, H_pairs, H_res)

// ── 6. LDRI ──
LDRI = σ(w_ldri · H + b_ldri)

// ── 7. EXPLANATION ──
E.lineage     = trace_signal_contributions(H, S_root)
E.path        = max_product_path(S_root, S_leaves)
E.interaction = R_root
E.uncertainty = [σ² at each hierarchical level]
E.resonance   = ||η_res·R|| / ||H||
E.memory      = {⟨k, β⟩ for active memories}

RETURN H, LDRI, E
```

---

## 16. Final Computational Complexity {#16}

### 16.1 Per-Iteration Complexity

| Component | Complexity | Dominant term |
|-----------|------------|---------------|
| Feature extraction (front-end) | $O(|\mathcal{D}| \cdot N_1 \cdot \bar{k})$ | $6 \times 5000 \times 12$ |
| Vertical propagation | $O(|\mathcal{D}| \cdot \sum_\ell N_\ell \cdot \bar{k})$ | $6 \times 7030 \times 12$ |
| Horizontal coupling | $O(|\mathcal{D}|^2 \cdot \sum_\ell N_\ell \cdot \bar{k})$ | $36 \times 7030 \times 12$ |
| Memory operations | $O(|\mathcal{D}| \cdot N_\ell \cdot |\mathcal{M}|)$ | $6 \times 7030 \times 200$ |
| Resonance update | $O(m \cdot d)$ | $100 \times 120$ |
| Top-down refinement | $O(|\mathcal{D}| \cdot \sum_\ell N_\ell \cdot \bar{k})$ | $6 \times 7030 \times 12$ |

### 16.2 Total Complexity

$$T = K \cdot O\left(|\mathcal{D}|^2 \cdot \bar{k} \cdot \sum_{\ell=1}^5 N_\ell\right)$$

For a typical novel ($N_1 = 5000$, $N_2 = 2000$, $N_3 = 30$, $N_4 = N_5 = 1$):

$$T \approx 5 \times 36 \times 12 \times 7032 \approx 1.5 \times 10^7 \text{ operations}$$

**Runtime estimate:** $< 0.1$ seconds (compiled linear algebra) / $< 5$ seconds (Python/NumPy single-threaded).

### 16.3 Space Complexity

| Component | Space |
|-----------|-------|
| Signal storage | $O(\sum_\ell N_\ell \cdot |\mathcal{D}| \cdot \bar{k}) \approx 5$ MB |
| Provenance | $O(\sum_\ell N_\ell \cdot |\mathcal{D}|) \approx 0.4$ MB |
| Memory store | $O(|\mathcal{M}| \cdot \bar{k}) \approx 0.1$ MB |
| Resonance field | $O(m \cdot d) \approx 0.05$ MB |
| **Total** | $\approx \mathbf{6 \text{ MB}}$ |

---

## 17. Final Limitations and Scope {#17}

### 17.1 Theoretical Limitations

| ID | Limitation | Mitigation |
|----|-----------|------------|
| L1 | **Fixed hierarchy** — The five-level tree does not adapt to unusual structures (embedded narratives, frame stories) | Adaptive hierarchy detection is identified as the primary future extension |
| L2 | **Pairwise coupling** — True higher-order interactions are approximated via $K$ iterative cycles | Proven to converge to all-pairs fixed point as $K \to \infty$ under contraction (§11.5) |
| L3 | **Eurocentric defaults** — Phase boundaries assume Western narrative structure (§4) | Configurable parameters with non-Western alternatives specified (§8) |
| L4 | **Single narrative stream** — Non-linear narratives (in medias res, multiple timelines) require $\tau$ extension | Multi-stream $\tau$ generalization specified (§8) |
| L5 | **No causal direction** — Coupling is bidirectional; cause vs. effect is not distinguished | Post-hoc Granger causality analysis on $\tau$-ordered signals |
| L6 | **Static coupling matrices** — $\mathbf{M}_{d,d'}$ is fixed (identity for zero-shot) | Genre-conditioned $\mathbf{M}_{d,d'}^{(g)}$ identified as future work |

### 17.2 Practical Limitations

| ID | Limitation | Mitigation |
|----|-----------|------------|
| L7 | **Feature extractor dependency** — Output quality depends on NLP extractor quality | Explicit $\gamma$ confidence weighting downweights unreliable extractions |
| L8 | **Narrative position noise** — $\tau$ estimation requires event detection (open problem) | Robust to $\pm 0.1$ perturbation in $\tau$; tested on synthetic data |
| L9 | **No empirical validation** — This is a theoretical specification | Experimental validation on 100-novel corpus is planned |
| L10 | **Memory threshold sensitivity** — $\theta_{\text{store}}, \theta_{\text{retrieve}}$ affect behavior | Defaults are conservative (high precision); sensitivity analysis is future work |
| L11 | **English-centric features** — Concreteness norms, sentiment lexicons are English | Cross-lingual adaptation is identified as a major extension |

### 17.3 Scope Conditions

HMP-DT Version 1.0 applies to:
- ✅ Narrative prose (novels, short stories)
- ✅ Single-author, published works
- ✅ Monolingual text (English primary)
- ✅ Text-only discourse

HMP-DT Version 1.0 does **not** apply to:
- ❌ Poetry, drama, non-fiction (different discourse structures)
- ❌ Collaborative or multi-voiced texts (different provenance structure)
- ❌ Code-switched or multilingual texts (different feature spaces)
- ❌ Multimodal works (different signal types)

**Scope is not a limitation.** Narrower scope enables deeper theoretical treatment and cleaner empirical validation.

---

## 18. References and Citations {#18}

- Booker, C. (2004). *The Seven Basic Plots: Why We Tell Stories*. Continuum.
- Campbell, J. (1949). *The Hero with a Thousand Faces*. Pantheon.
- Freytag, G. (1863). *Die Technik des Dramas*. Hirzel.
- Genette, G. (1972). *Figures III*. Seuil.
- Jaynes, E. T. (2003). *Probability Theory: The Logic of Science*. Cambridge University Press.
- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.
- Martin, J. R., & Rose, D. (2007). *Working with Discourse: Meaning Beyond the Clause*. Continuum.
- Piper, A. (2018). *Enumerations: Data and Literary Study*. University of Chicago Press.
- Rahimi, A., & Recht, B. (2007). Random features for large-scale kernel machines. *Advances in Neural Information Processing Systems*, 20.
- Todorov, T. (1969). *Grammaire du Décaméron*. Mouton.
- Turner, M. (1996). *The Literary Mind*. Oxford University Press.
- Underwood, T. (2019). *Distant Horizons: Digital Evidence and Literary Change*. University of Chicago Press.

---

## Research Freeze Declaration

This specification defines the frozen **Version 1.0 of HMP-DT** (Hierarchical Message Passing on Discourse Trees). All equations, assumptions, architectural components, parameters, algorithms, and notations presented in this document constitute the definitive reference for the model.

**Future work should extend the framework** (new feature extractors, cross-lingual adaptation, genre-conditioned coupling matrices, empirical validation, application tools) rather than modify its core theoretical architecture.

Any proposed architectural improvements discovered during implementation or validation should be documented as "HMP-DT Version 2.0 proposals" and evaluated against this baseline rather than merged incrementally.

This freeze ensures a stable reference point for:
- Reproducible implementation
- Empirical validation
- Comparative evaluation
- Scientific communication

*End of HMP-DT Version 1.0 — Research Freeze*
