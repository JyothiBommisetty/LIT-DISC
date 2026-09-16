# LIT-DISC Methodology: Complete Mathematical Formulation

> Reverse-engineered from implementation v1.1.
> Corresponds to Springer CCIS Section 3 (Methodology).

---

## 1. Framework Overview

### 1.1 Overall Architecture

LIT-DISC (Literary Discourse Richness Index) is a computational framework that processes natural language text through a hierarchical multi-scale discourse processing tree (HMP-DT) and produces a scalar richness score.

### 1.2 Pipeline

```
Text File → Load → Sentence Split → Paragraph/Chapter Detect
→ Build Hierarchy Tree → Tokenize → Compute Narrative Positions (τ)
→ Feature Extraction (6 dimensions) → Signal Initialization
→ K-iteration Refinement Loop:
    a. Bottom-up propagation with horizontal coupling
    b. Memory storage & decay
    c. Recursive resonance update
    d. Top-down refinement
→ Embedding Construction → LDRI Calculation → Result
```

### 1.3 Input

- A text file (`.txt`, `.epub`, or `.xml`) containing natural language prose.

### 1.4 Output

- `LITDISCResult` containing:
  - `ldri`: scalar richness score in ℝ([0,1])
  - `gamma_global`: global confidence in ℝ([0,1])
  - `raw`: unweighted LDRI (with Γ = 1) in ℝ([0,1])
  - `reliability`: bootstrap stability in ℝ([0,1])
  - `components`: dict with 6 decomposition values
  - `fractions`: dict with 6 relative contribution values

### 1.5 Processing Stages

| Stage | Module | Purpose |
|-------|--------|---------|
| Load | `loader.loader.TextLoader` | Read text from file |
| Split | `preprocessing.sentence_splitter.SentenceSplitter` | Split into sentences |
| Detect | `hierarchy.detector.ParagraphDetector`, `ChapterDetector` | Detect paragraphs and chapters |
| Build | `hierarchy.tree.HierarchyTree` | Build 5-level hierarchy |
| Tokenize | `preprocessing.tokenizer.Tokenizer` | Tokenize sentences, guess POS |
| τ-compute | `pipeline.orchestrator.NarrativePositionComputer` | Compute narrative positions |
| Features | 6 extractors in `features/` | Extract discourse features |
| Initialize | `signals.initializer.SignalInitializer` | Create LiterarySignal objects |
| Refine | `propagation.controller.RefinementController` | K-iteration HMP-DT inference |
| Embed | `embedding.construction.NarrativeEmbeddingConstructor` | Build embedding vector |
| LDRI | `ldri.calculator.LDRICalculator` | Compute final richness score |

### 1.6 Data Flow

```
Text → [sentences] → [paragraph groups] → [chapter groups]
→ HierarchyTree(5 levels: sentence→paragraph→chapter→narrative→discourse)
→ [tokenized sentences with POS]
→ τ ∈ [0,1] per sentence
→ [6 dim features: H_S, H_N, H_I, H_Sy, H_E, H_Em]  each ℝ^{k_d}
→ [LiterarySignal objects: f, gamma, tau]
→ [propagated signals after refinement]
→ NarrativeEmbedding: H ∈ ℝ^{120} concatenated from subspaces
→ LDRIScore
```

---

## 2. Hierarchical Multi-Phase Discourse Theory (HMP-DT)

### 2.1 Theoretical Basis

HMP-DT decomposes literary text into a 5-level hierarchy:

| Level | Name | Granularity | Node Type |
|-------|------|-------------|-----------|
| 1 | Surface | Sentence | S_i |
| 2 | Structure | Paragraph | P_j |
| 3 | Abstraction | Chapter | C_k |
| 4 | Conceptual | Narrative layer | N_l |
| 5 | Relational | Discourse layer | D_m |

At each level, six discourse dimensions are computed from features extracted from the text.

### 2.2 The Six Discourse Dimensions

| Symbol | Name | Feature Count (k_d) | What It Captures |
|--------|------|---------------------|------------------|
| S | Semantic | 8 | Lexical density, polysemy, coherence, information density |
| N | Narrative | 8 | Tension, pacing, scene type, temporal shifts, structural complexity |
| I | Interaction | 7 | Speaker/hearer encoding, dialogue density, turn-taking, question density |
| Sy | Symbolic | 7 | Figurative density, motif recurrence, symbolic density, imagery intensity |
| E | Evolution | 7 | Arc slope, relationship state, change magnitude, temporal progression |
| Em | Emotion | 7 | Valence, arousal, intensity, emotional diversity, sentiment extremity |

### 2.3 Why Each Dimension Exists

Each dimension corresponds to a distinct aspect of literary-discourse processing as supported by the implementation:

- **Semantic (S):** Lexical properties of the text — how dense, varied, and information-rich the language is.
- **Narrative (N):** Structural properties of storytelling — pacing, tension, scene construction, temporal organization.
- **Interaction (I):** Dialogue and social dynamics — how characters communicate through speech.
- **Symbolic (Sy):** Figurative language — metaphor, symbolism, sensory imagery, motif recurrence.
- **Evolution (E):** Character and relationship development — arcs, change, progression across the narrative.
- **Emotion (Em):** Affective content — valence, arousal, emotional diversity, intensity.

### 2.4 How Dimensions Interact

Dimensions interact through three mechanisms:

1. **Horizontal coupling** (Eq. 11.9–11.11): Feature vectors from different dimensions at the same node are combined via a learned kernel that computes similarity, proximity, confidence, compatibility, symbolic coherence, and reliability components.

2. **Uncertainty propagation** (Eq. 13.1–13.6): Each dimension has its own variance estimate; uncertainties propagate upward through the hierarchy.

3. **Geometric mean in LDRI** (Eq. 10): The final LDRI aggregates all six dimensions through a weighted geometric mean, where the weights (alpha, beta, gamma, delta) control the contribution of each transformed factor.

---

## 3. Complete Mathematical Formulation

### Notation

Let:
- $d \in \{\text{S}, \text{N}, \text{I}, \text{Sy}, \text{E}, \text{Em}\}$ index a discourse dimension
- $\mathbf{f}_d \in \mathbb{R}^{k_d}$ be the feature vector for dimension $d$
- $\gamma_d \in [0,1]$ be the confidence for dimension $d$
- $\tau \in [0,1]$ be the normalized narrative position
- $\mathcal{T}$ be the hierarchy tree with levels $\ell \in \{1,2,3,4,5\}$

---

### 3.1 Activation Function (Rayleigh CDF)

> **Ordering note:** Eqs. 1–9 (LDRI calculation) are presented before the embedding construction (Eqs. 53–55) even though $a_d$ (Eq. 1) depends on $\|\mathbf{H}_d\|$, which is produced by Eq. 53. In execution order, embedding construction happens *after* signal propagation (Algorithm 1, lines 57–58) but *before* LDRI calculation (lines 60–71). The presentation order in this document follows the LDRI equation hierarchy (activation → factors → aggregation → decomposition) rather than pipeline execution order. For pipeline order, see Algorithm 1.

**Equation 1** (Activation from feature norm):
$$a_d = g(\|\mathbf{H}_d\|) = 1 - \exp\left(-\frac{\|\mathbf{H}_d\|^2}{2\sigma_{\max}^2}\right)$$
*Source: `src/ldri/calculator.py:46-48`*

| Variable | Meaning | Range | Default |
|----------|---------|-------|---------|
| $a_d$ | Activation for dimension $d$ | $[0, 1]$ | — |
| $\mathbf{H}_d$ | Embedding subspace for dimension $d$ | $\mathbb{R}^{d/10}$ | — |
| $\|\mathbf{H}_d\|$ | Euclidean norm of subspace | $[0, \infty)$ | — |
| $\sigma_{\max}$ | Scale parameter (sigma_max) | $(0, \infty)$ | $5.0$ |

**Purpose:** Maps feature norms to bounded activations. The Rayleigh cumulative distribution function (CDF) provides a monotonic, smooth activation with principled probabilistic interpretation. At $\sigma_{\max} = 5.0$, typical feature norms $\|\mathbf{H}_d\| \approx 10-14$ produce $z = 2-3$, operating in the sensitive region of the CDF.

**Implementation note:** This is referred to in test code as the "Rayleigh CDF." It is NOT the sigmoid described in earlier architecture docs. The implementation uses $g(x) = 1 - \exp(-x^2 / 2\sigma^2)$.

**Constraints:** $a_d$ is monotonically increasing in $\|\mathbf{H}_d\|$, $a_d \to 1$ as $\|\mathbf{H}_d\| \to \infty$, $a_d = 0$ when $\|\mathbf{H}_d\| = 0$.

---

### 3.2 Signal Richness

**Equation 2** (Signal richness from activations):
$$R_{\text{signal}} = \frac{\|\mathbf{a}\|}{\sqrt{6}} = \sqrt{\frac{1}{6}\sum_{d} a_d^2}$$
*Source: `src/ldri/calculator.py:51-52`*

| Variable | Meaning | Range |
|----------|---------|-------|
| $R_{\text{signal}}$ | Signal richness | $[0, 1]$ |
| $\mathbf{a} \in \mathbb{R}^6$ | Vector of activations $[a_S, a_N, a_I, a_{Sy}, a_E, a_{Em}]$ | $[0, 1]^6$ |
| $6$ | Number of dimensions | constant |

**Purpose:** Aggregates per-dimension activations into a single richness measure. The $\ell_2$ norm divided by $\sqrt{6}$ ensures $R_{\text{signal}} \in [0, 1]$, with $R_{\text{signal}} = 1$ iff all $a_d = 1$, and $R_{\text{signal}} = 0$ iff all $a_d = 0$.

**Interpretation:** Uniform activation across all six dimensions produces a higher $R_{\text{signal}}$ than a concentrated activation in a single dimension (concentration penalty).

---

### 3.3 Coupling Factor

**Equation 3** (Coupling factor from pairwise subspace):
$$C = \frac{\|\mathbf{H}_{\text{pairs}}\|}{\|\mathbf{H}_{\text{pairs}}\|_{\max}} = \min\left(1.0,\ \frac{\|\mathbf{H}_{\text{pairs}}\|}{\sqrt{|\mathbf{H}_{\text{pairs}}|}}\right)$$
*Source: `src/ldri/calculator.py:54-57`*

| Variable | Meaning | Range |
|----------|---------|-------|
| $C$ | Coupling factor | $[0, 1]$ |
| $\mathbf{H}_{\text{pairs}} \in \mathbb{R}^{3d/10}$ | Pairwise interaction subspace | $\mathbb{R}^{36}$ when $d=120$ |
| $\|\mathbf{H}_{\text{pairs}}\|$ | Euclidean norm | $[0, 6]$ |
| $\sqrt{|\mathbf{H}_{\text{pairs}}|}$ | Max possible norm ($\sqrt{36}=6$) | constant |

**Purpose:** Measures pairwise interaction between the six dimension subspaces via cosine similarities of the 15 unique pairs, then normalizes to $[0,1]$.

---

### 3.4 Resonance Factor

**Equation 4** (Resonance factor from resonance subspace):
$$T = \frac{\|\mathbf{H}_{\text{res}}\|}{\|\mathbf{H}_{\text{res}}\|_{\max}} = \min\left(1.0,\ \frac{\|\mathbf{H}_{\text{res}}\|}{\sqrt{|\mathbf{H}_{\text{res}}|}}\right)$$
*Source: `src/ldri/calculator.py:59-62`*

| Variable | Meaning | Range |
|----------|---------|-------|
| $T$ | Resonance factor | $[0, 1]$ |
| $\mathbf{H}_{\text{res}} \in \mathbb{R}^{d/10}$ | Resonance trajectory statistics | $\mathbb{R}^{12}$ when $d=120$ |

**Purpose:** Captures temporal narrative dynamics via statistics of the resonance field trajectory. Structurally identical to the coupling factor.

---

### 3.5 Memory Factor

**Equation 5** (Memory factor):
$$M = \min(1.0, \max(0.0, \bar{\beta}))$$
*Source: `src/ldri/calculator.py:64-65`*

| Variable | Meaning | Range |
|----------|---------|-------|
| $M$ | Memory factor | $[0, 1]$ |
| $\bar{\beta}$ | Average memory activation (beta) across all stored entries | $[0, 1]$ |
| $\beta$ per entry | Memory activation (strength) | $[0, 1]$ |

**Purpose:** Clamped average of memory entry activations, representing how "active" the memory store is. When no entries exist, defaults to 0.

---

### 3.6 Normalized Factors

**Equation 6** (Normalization from $[0,1]$ to $[0.5, 1.0]$):
$$\tilde{C} = \frac{1 + C}{2}, \quad \tilde{T} = \frac{1 + T}{2}, \quad \tilde{M} = \frac{1 + M}{2}$$
*Source: `src/ldri/calculator.py:19-21`*

| Variable | Range |
|----------|-------|
| $\tilde{C}, \tilde{T}, \tilde{M}$ | $[0.5, 1.0]$ |
| $C, T, M$ | $[0, 1]$ before normalization |

**Purpose:** Since $C, T, M \in [0,1]$ (from clamping in Eqs. 3-5), this maps $0 \to 0.5$ and $1 \to 1.0$, ensuring factors always contribute at least 0.5 to the geometric mean.  
**Note:** Eq. 10 inverts this transformation. Together, Eqs. 6 and 10 cancel out mathematically in the decomposition — the shift exists solely to prevent zero factors from zeroing the geometric mean in Eq. 8. The same effect could be achieved by adding a small $\epsilon$ to each factor inside the geometric mean.

---

### 3.7 Global Confidence (Gamma)

**Equation 7** (Global confidence from per-dimension confidences):
$$\Gamma = \prod_{d} \min(1.0, \max(0.0, \bar{\gamma}_d))$$
*Source: `src/ldri/calculator.py:67-73`*

| Variable | Meaning | Range |
|----------|---------|-------|
| $\Gamma$ | Global confidence | $[0, 1]$ |
| $\bar{\gamma}_d$ | Mean per-dimension confidence (mean of $\gamma_d$ across all nodes) | $[0, 1]$ |
| $\gamma_d$ | Per-signal confidence from feature extraction | $[0, 1]$ |

**Purpose:** Aggregates per-dimension confidence into a single global weight. The product form means that any dimension with low confidence drives $\Gamma$ down. When no confidences are provided, $\Gamma = 0.85$ (default).

---

### 3.8 Raw LDRI (Unweighted)

**Equation 8** (Raw LDRI — geometric mean of weighted factors):
$$\text{LDRI}_{\text{raw}} = \left(R_{\text{signal}}^\alpha \cdot \tilde{C}^\beta \cdot \tilde{T}^\gamma \cdot \tilde{M}^{\delta_{\text{exp}}}\right)^{1/S}, \quad S = \alpha + \beta + \gamma + \delta_{\text{exp}}$$
*Source: `src/ldri/calculator.py:75-82`*

| Variable | Meaning | Range | Default |
|----------|---------|-------|---------|
| $\alpha$ | Signal richness exponent | $(0, \infty)$ | $2.0$ |
| $\beta$ | Coupling exponent | $(0, \infty)$ | $1.0$ |
| $\gamma$ | Resonance exponent | $(0, \infty)$ | $1.0$ |
| $\delta_{\text{exp}}$ | Memory exponent | $(0, \infty)$ | $1.0$ |
| $S$ | Sum of exponents | $(0, \infty)$ | $5.0$ |
| $\text{LDRI}_{\text{raw}}$ | Unweighted richness | $[0, 1]$ | — |

**Purpose:** Weighted geometric mean of the four discourse factors. The exponents control each factor's importance. The $1/S$ normalization ensures $\text{LDRI}_{\text{raw}} \in [0,1]$. A lower bound of $10^{-10}$ prevents numerical issues.

**Interpretation:** The weighted geometric mean penalizes low scores across any factor. A passage must score well on ALL factors to achieve a high raw LDRI.

---

### 3.9 Final LDRI

**Equation 9** (Final LDRI):
$$\text{LDRI} = \Gamma \cdot \text{LDRI}_{\text{raw}}$$
*Source: `src/ldri/calculator.py:24`*

| Variable | Range |
|----------|-------|
| $\text{LDRI}$ | $[0, 1]$ |

**Purpose:** The global confidence $\Gamma$ modulates the raw LDRI. This is the final metric value.

**Important:** Since $\Gamma$ varies by 21–47% (CV) while $\text{LDRI}_{\text{raw}}$ varies by < 3%, $\text{LDRI}$ is dominated by $\Gamma$. The metric is effectively a confidence-weighted constant.

---

### 3.10 Component Decomposition

The decomposition expresses each component's contribution to LDRI on a log scale.

**Equation 10** (Inverse normalization — reverses Eq. 6):
$$C = 2\tilde{C} - 1, \quad T = 2\tilde{T} - 1, \quad M = 2\tilde{M} - 1$$
*Source: `src/ldri/calculator.py:122-124`*  
**Note:** This reverses Eq. 6 exactly. Together they are mathematically redundant — the decomposition could operate on $\tilde{C}, \tilde{T}, \tilde{M}$ directly without the inversion.

**Equation 11** (Intrinsic component):
$$c_{\text{int}} = \frac{\alpha}{S} \cdot \Gamma \cdot \log\left(\frac{1 + R_{\text{signal}}}{\max(\epsilon, 1 - R_{\text{signal}})}\right)$$
*Source: `src/ldri/calculator.py:126`*

Where $\epsilon = 10^{-10}$.

**Equation 12** (Interaction component):
$$c_{\text{intr}} = \frac{\beta}{S} \cdot \Gamma \cdot \log\left(\max(1.0, 1 + C)\right)$$
*Source: `src/ldri/calculator.py:127`*

**Equation 13** (Temporal component):
$$c_{\text{temp}} = \frac{\gamma}{S} \cdot \Gamma \cdot \log\left(\max(1.0, 1 + T)\right)$$
*Source: `src/ldri/calculator.py:128`*

**Equation 14** (Resonance component):
$$c_{\text{res}} = \frac{\gamma}{S} \cdot \Gamma \cdot \frac{e^{-1/\lambda}}{1 - e^{-1/\lambda}} \cdot \min(1.0, T)$$
*Source: `src/ldri/calculator.py:129`*

Where $\lambda = 0.3$ (hardcoded decay constant from ResonanceParams; see Eq. 40 for the resonance field update that defines $\lambda$).  
**Note:** The constant $\frac{e^{-1/\lambda}}{1 - e^{-1/\lambda}}$ transforms $\min(1.0, T)$ into a log-scale contribution comparable to the other log-scale components. It is the ratio of the discretized decay term to its complement, arising from the Euler discretization of Eq. 40. There is no closed-form derivation linking this constant to any linguistic or cognitive property.

**Equation 15** (Memory component):
$$c_{\text{mem}} = \frac{\delta_{\text{exp}}}{S} \cdot \Gamma \cdot \log\left(\max(1.0, 1 + M)\right)$$
*Source: `src/ldri/calculator.py:130`*

**Equation 16** (Uncertainty component):
$$c_{\text{unc}} = (1 - \Gamma) \cdot \text{LDRI}_{\text{raw}}$$
*Source: `src/ldri/calculator.py:131`*

**Equation 17** (Fraction normalization):
$$f_k = \frac{c_k}{\text{LDRI} + c_{\text{unc}} + \epsilon}, \quad k \in \{\text{int}, \text{intr}, \text{temp}, \text{res}, \text{mem}, \text{unc}\}$$
*Source: `src/ldri/calculator.py:141-143`*

Where $\epsilon = 10^{-10}$.

---

### 3.11 Reliability (Bootstrap)

**Equation 18** (Reliability from bootstrap):
$$\rho = 1 - \min\left(1.0, \frac{\sigma_{\text{LDRI}}}{0.1}\right)$$
*Source: `src/ldri/calculator.py:84-115`*

Where $\sigma_{\text{LDRI}}$ is the standard deviation of LDRI computed over $B = 100$ bootstrap samples with Gaussian noise $\mathcal{N}(0, 0.01)$ added to each embedding subspace.

---

### 3.12 Narrative Position (τ)

**Equation 19** (Narrative velocity for sentence $i$):
$$v_i = \max\left(0.1, \frac{\text{\# verbs in sentence}_i}{\text{\# tokens in sentence}_i}\right)$$
*Source: `src/pipeline/orchestrator.py:30-36`*

**Equation 20** (Cumulative narrative position):
$$\tau_i = \frac{\sum_{j=1}^{i} v_j}{\sum_{j=1}^{n} v_j}, \quad \tau \in [0, 1]$$
*Source: `src/pipeline/orchestrator.py:35-36`*

**Purpose:** Maps sentences to positions in normalized narrative time $[0, 1]$, where velocity is proportional to verb density.

---

### 3.13 Feature Extraction — Confidence Computation

**Equation 21** (Per-dimension confidence from features):
$$\gamma_d = \min\left(1.0,\ 0.3 + 0.7 \cdot \mathbb{E}[|\mathbf{f}_d|]\right)$$
*Source: `src/features/base.py:14-16`*

| Variable | Range |
|----------|-------|
| $\gamma_d$ | $[0, 1]$ |
| $\mathbb{E}[|\mathbf{f}_d|]$ | $[0, 1]$ after normalization |

**Purpose:** Confidence increases with the mean absolute feature value. A bias of 0.3 ensures minimum confidence even for zero features.

---

### 3.14 Random Fourier Features (Memory Key Generation)

**Equation 22** (RFF key projection):
$$\mathbf{k}(\mathbf{f}) = \sqrt{\frac{2}{k_{\text{key}}}} \cos\left(\mathbf{W}\mathbf{f} + \mathbf{b}\right)$$
*Source: `src/memory/keys.py:12-14` and `src/coupling/memory.py:48-56`*

| Variable | Meaning | Range | Default |
|----------|---------|-------|---------|
| $\mathbf{k}(\mathbf{f})$ | Memory key | $\mathbb{R}^{k_{\text{key}}}$ | $k_{\text{key}} = 32$ |
| $\mathbf{W}$ | Random weight matrix | $\mathbb{R}^{k_{\text{key}} \times d_{\text{in}}}$ | $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ |
| $\mathbf{b}$ | Random bias vector | $\mathbb{R}^{k_{\text{key}}}$ | $b_i \sim \text{Uniform}(0, 2\pi)$ |
| $\sigma$ | Kernel bandwidth | $(0, \infty)$ | $1.0$ |
| $\mathbf{f}$ | Input feature vector | $\mathbb{R}^{d_{\text{in}}}$ | — |

**Seeds:** W uses seed 0, b uses seed 1 (per input dimension). Deterministic.

---

### 3.15 Memory Storage Decision

**Equation 23** (Storage condition):
$$\text{store} = \|\mathbf{f}\| \cdot \gamma > \theta_{\text{store}}$$
*Source: `src/coupling/memory.py:107-108`*

| Variable | Meaning | Default |
|----------|---------|---------|
| $\theta_{\text{store}}$ | Storage threshold | $0.5$ |

---

### 3.16 Memory Retrieval

**Equation 24** (Cosine similarity retrieval):
$$\text{sim}(\mathbf{k}_q, \mathbf{k}_j) = \frac{\mathbf{k}_q \cdot \mathbf{k}_j}{\|\mathbf{k}_q\| \cdot \|\mathbf{k}_j\|} > \theta_{\text{retrieve}}$$
*Source: `src/coupling/memory.py:136-164`*

| Variable | Meaning | Default |
|----------|---------|---------|
| $\theta_{\text{retrieve}}$ | Retrieval threshold | $0.7$ |
| $\mathbf{k}_q$ | Query key (RFF of current feature) | — |
| $\mathbf{k}_j$ | Stored key | — |

---

### 3.17 Memory Reinforcement

**Equation 25** (Reinforcement on retrieval):
$$\beta_j \leftarrow \min\left(1.0, \beta_j + \rho_{\text{reinforce}} \cdot \max(0.0, \cos(\mathbf{k}_q, \mathbf{k}_j))\right)$$
*Source: `src/coupling/memory.py:177-192`*

| Variable | Meaning | Default |
|----------|---------|---------|
| $\rho_{\text{reinforce}}$ | Reinforcement rate | $0.1$ |

---

### 3.18 Memory Decay

**Equation 26** (Selective exponential decay):
$$\beta_j \leftarrow \beta_j \cdot (1 - \delta) \quad \text{(if not retrieved this iteration)}$$
*Source: `src/coupling/memory.py:196-221`*

| Variable | Meaning | Default |
|----------|---------|---------|
| $\delta$ | Decay rate | $0.05$ |

---

### 3.19 Memory Purging

**Equation 27** (Forgetting):
$$\text{purge if } \beta_j < \theta_{\text{forget}}$$
*Source: `src/coupling/memory.py:231-240`*

| Variable | Meaning | Default |
|----------|---------|---------|
| $\theta_{\text{forget}}$ | Forgetting threshold | $0.01$ |

---

### 3.20 Memory Augmentation

**Equation 28** (Signal augmentation from retrieved memories):
$$\mathbf{f}_{\text{aug}} = \mathbf{f} + \sum_{j} \beta_j \cdot \mathbf{W}_{\text{mem}} \cdot \mathbf{f}_j$$
*Source: `src/coupling/memory.py:244-258`*

Where $\mathbf{W}_{\text{mem}} = \mathbf{I}$ (identity matrix, from `_ensure_mem_dim`).

---

### 3.21 Coupling Kernel Components

**Equation 29** (Context similarity):
$$C_{\text{sim}} = \cos(\mathbf{f}_d \oplus \mathbf{p}_d,\ \mathbf{f}_{d'} \oplus \mathbf{p}_{d'})$$
*Source: `src/coupling/components.py:28-46`*

Where $\oplus$ denotes concatenation and $\mathbf{p}_d$ is an optional position vector.

**Equation 30** (Narrative proximity):
$$C_{\text{prox}} = \exp\left(-\frac{|\tau_d - \tau_{d'}|}{\sigma_\tau}\right)$$
*Source: `src/coupling/components.py:48-50`*

Where $\sigma_\tau = 0.1$.

**Equation 31** (Confidence-weighted similarity):
$$C_{\text{conf}} = \gamma_d \cdot \gamma_{d'} \cdot \max(0, \cos(\mathbf{f}_d, \mathbf{f}_{d'}))$$
*Source: `src/coupling/components.py:52-69`*

**Equation 32** (Semantic compatibility):
$$C_{\text{comp}} = \frac{1 + \mathbf{v}_d \cdot \mathbf{v}_{d'}}{2}$$
*Source: `src/coupling/components.py:71-78`*

Where $\mathbf{v}_d$ is an optional value vector (defaults to unit vector).

**Equation 33** (Symbolic coherence):
$$C_{\text{sym}} = \frac{1}{|T| + 1} \sum_{t \in T} \text{sim}(\mathbf{f}_d^t, \mathbf{f}_{d'}^t)$$
*Source: `src/coupling/components.py:80-89`*

Where $T$ is the set of shared motif types (returns 0 if empty).

**Equation 34** (Interaction reliability):
$$C_{\text{rel}}^{(t)} = \frac{1}{t-1} \sum_{s=1}^{t-1} \kappa_{d,d'}^{(s)}$$
*Source: `src/coupling/components.py:91-106`*

Where $\kappa_{d,d'}^{(s)}$ is the coupling value at step $s$.

---

### 3.22 Adaptive Coupling Weights

**Equation 35** (Unnormalized weight for component $i$):
$$\tilde{\eta}_i(\tau) = \alpha_i + \beta_i \cdot \frac{1}{2}\left[1 + \cos\left(\frac{2\pi}{\omega_i}(\tau - \mu_i)\right)\right] \cdot \mathbb{I}\left[|\tau - \mu_i| \leq \frac{\omega_i}{2}\right]$$
*Source: `src/coupling/weights.py:27-41`*

**Equation 36** (Normalized weight):
$$\eta_i(\tau) = \frac{\tilde{\eta}_i(\tau)}{\sum_j \tilde{\eta}_j(\tau)}$$
*Source: `src/coupling/weights.py:43-50`*

| Component | $\mu_i$ | $\omega_i$ | $\alpha_i$ | $\beta_i$ |
|-----------|---------|------------|------------|-----------|
| $C_{\text{sim}}$ | 0.15 | 0.30 | 0.10 | 0.20 |
| $C_{\text{prox}}$ | 0.55 | 0.50 | 0.10 | 0.20 |
| $C_{\text{conf}}$ | 0.50 | 1.00 | 0.20 | 0.10 |
| $C_{\text{comp}}$ | 0.10 | 0.20 | 0.05 | 0.15 |
| $C_{\text{sym}}$ | 0.60 | 0.40 | 0.05 | 0.25 |
| $C_{\text{rel}}$ | 0.85 | 0.30 | 0.05 | 0.20 |

---

### 3.23 Coupling Kernel

**Equation 37** (Complete coupling kernel):
$$\kappa_{d,d'}(\tau) = \sum_{i \in \mathcal{C}} \eta_i(\tau) \cdot C_i$$
*Source: `src/coupling/kernel.py:21-72`*

Where $\mathcal{C} = \{C_{\text{sim}}, C_{\text{prox}}, C_{\text{conf}}, C_{\text{comp}}, C_{\text{sym}}, C_{\text{rel}}\}$.

---

### 3.24 Coupling Operator

**Equation 38** (Coupling operator $\Lambda$):
$$\Lambda_{d,d'}(\mathbf{f}_d, \mathbf{f}_{d'}) = \kappa_{d,d'}(\tau) \cdot \mathbf{M}_{d,d'} \cdot (\mathbf{f}_d \odot \mathbf{f}_{d'})$$
*Source: `src/coupling/kernel.py:128-144`*

Where $\odot$ is element-wise multiplication and $\mathbf{M}_{d,d'}$ is the coupling matrix (identity by default).

**Equation 39** (Signal update):
$$\tilde{\mathbf{f}}_d = \mathbf{f}_d + \sum_{d' \neq d} \Lambda_{d,d'}(\mathbf{f}_d, \mathbf{f}_{d'})$$
*Source: `src/coupling/kernel.py:146-151`*

With clipping to $[-2.0, 2.0]$ per Assumption A6.

---

### 3.25 Recursive Resonance

**Equation 40** (Resonance field update):
$$\mathbf{R}(\tau_i) = \mathbf{R}(\tau_{i-1}) \cdot e^{-\Delta\tau/\lambda} + \mathbf{H}(\tau_{i-1}) \cdot \Delta\tau$$
*Source: `src/resonance/field.py:35-57`*

| Variable | Meaning | Default |
|----------|---------|---------|
| $\lambda$ | Decay constant | $0.3$ |
| $\Delta\tau$ | Time step | $1/n_{\text{samples}}$ |
| $\mathbf{R}$ | Resonance field vector | — |
| $\mathbf{H}$ | Concatenated dimension signal vector | — |

**Equation 41** (Resonance augmentation):
$$\mathbf{f}_{\text{res}}(\tau) = \mathbf{f}(\tau) + \eta_{\text{res}} \cdot \mathbf{R}(\tau)$$
*Source: `src/resonance/field.py:59-72`*

Where $\eta_{\text{res}} = 0.2$.

---

### 3.26 Top-Down Refinement

**Equation 42** (Top-down refinement):
$$\mathbf{f}_i^{(\ell)}[d] \leftarrow \mathbf{f}_i^{(\ell)}[d] + \beta_\downarrow \cdot \text{Proj}_\downarrow\left(\mathbf{f}_{\text{parent}}^{(\ell+1)}[d]\right)$$
*Source: `src/propagation/top_down.py:50-51`*

Where $\beta_\downarrow = 0.3$ and $\text{Proj}_\downarrow$ pads/truncates the parent vector to match the child dimension.

---

### 3.27 Vertical Aggregation

**Equation 43** (Weighted aggregation — sentence to paragraph):
$$\mathbf{f}_{\text{agg}} = \frac{\sum_i \gamma_i \cdot \mathbf{f}_i}{\sum_i \gamma_i}, \quad \gamma_{\text{agg}} = \frac{1}{n}\sum_i \gamma_i$$
*Source: `src/propagation/bottom_up.py:218-228`*

**Equation 44** (Semantic coherence):
$$\text{SC} = \frac{1}{|\text{triu}|} \sum_{i<j} \cos(\mathbf{f}_i, \mathbf{f}_j)$$
*Source: `src/propagation/bottom_up.py:230-245`*

**Equation 45** (Salience weights for paragraph→chapter):
$$\hat{\alpha}_i = \frac{\exp(\beta_{\text{sal}} \cdot \|\mathbf{f}_i\| \cdot \max(\gamma_i, 0.1))}{\sum_j \exp(\beta_{\text{sal}} \cdot \|\mathbf{f}_j\| \cdot \max(\gamma_j, 0.1))}$$
*Source: `src/propagation/bottom_up.py:247-257`*

Where $\beta_{\text{sal}} = 2.0$.

**Equation 46** (Inverse-distance weights for chapter→narrative):
$$w_i = \frac{1/|\tau_i - \tau_{\text{target}}|}{\sum_j 1/|\tau_j - \tau_{\text{target}}|}$$
*Source: `src/propagation/bottom_up.py:259-267`*  
**Note:** The original code names this `_spline_weights` but it implements inverse-distance weighting ($1/|\Delta\tau|$), not spline interpolation. A true spline would involve polynomial basis functions, knot points, and smoothness constraints.

---

### 3.28 Uncertainty Propagation

**Equation 47** (Total variance — Eq. 13.1):
$$\sigma_{\text{total}}^2 = \sigma_e^2 + \sigma_s^2 + \sigma_i^2 + \sigma_t^2 + \sigma_m^2$$
*Source: `src/uncertainty/propagation.py:78-82`*

**Equation 48** (Confidence from variance — Eq. 13.2):
$$\gamma = 1 - \min\left(1.0, \frac{\sigma_{\text{total}}}{\sigma_{\max}}\right)$$
*Source: `src/uncertainty/propagation.py:90-93`*

**Equation 49** (Sentence to paragraph — Eq. 13.3):
$$\sigma^2(P) = \max\left(0,\ \frac{1}{n^2}\sum_i \sigma^2(S_i) + (1 - \text{SC}) - 2 \cdot \text{SC} \cdot \bar{\sigma}^2\right)$$
*Source: `src/uncertainty/propagation.py:102-127`*  
**Note:** The $\max(0,\cdot)$ clamp is required because the equation can produce negative values (combination of subtraction term and low semantic coherence). The implementation applies this clamp at line 123.

**Equation 50** (Paragraph to chapter — Eq. 13.4):
$$\sigma^2(C) = \sum_j \alpha_j^2 \cdot \sigma^2(P_j) + 0.1 \cdot (1 - \max_j \alpha_j)$$
*Source: `src/uncertainty/propagation.py:131-155`*  
**Note:** The constant $0.1$ is an arbitrary magic number representing a baseline uncertainty floor when salience is uneven. It appears nowhere in the parameter table and has no theoretical justification — it ensures non-zero variance even when $\max \alpha_j \approx 1$.

**Equation 51** (Chapter to narrative — Eq. 13.5):
$$\sigma^2(\tau) = \sum_i \phi_i(\tau)^2 \cdot \sigma^2(C_i) + 0.05 \cdot \min_i |\tau - \tau_i|^2$$
*Source: `src/uncertainty/propagation.py:159-189`*  
**Note:** The constant $0.05$ is an arbitrary magic number — it penalizes extrapolation far from known chapter positions. Like the $0.1$ in Eq. 50, it lacks theoretical justification and is not in the parameter table.

**Equation 52** (Coupling uncertainty — Eq. 13.6):
$$\sigma_i^2(d, d') = \kappa^2 \cdot (\text{Var}(f_d) + \text{Var}(f_{d'}))$$
*Source: `src/uncertainty/propagation.py:219-223`*

---

### 3.29 Embedding Construction

**Equation 53** (Per-dimension embedding subspace — Eq. 14.2):
$$\mathbf{H}_d = [\text{Mean}(\mathbf{f}_d),\ \text{Max}(\mathbf{f}_d),\ \text{Slope}(\mathbf{f}_d),\ \text{Var}(\mathbf{f}_d),\ \text{FFT}_{\text{low}}(\mathbf{f}_d)]$$
*Source: `src/embedding/construction.py:70-104`*

For each feature component $j = 1, \ldots, k_d$:
- $\mu_j = \frac{1}{n}\sum_{i=1}^n f_{ij}$
- $m_j = \max_i f_{ij}$
- $s_j = (f_{nj} - f_{1j}) / \max(n-1, 1)$
- $v_j = \frac{1}{n}\sum_i (f_{ij} - \mu_j)^2$
- $\hat{f}_j = |\text{FFT}(f_{\cdot j})[1]|$ (if $n > 2$, else 0)

Concatenates 5 statistics per component, then pads/truncates to $d/10$.

**Equation 54** (Pairwise interaction subspace):
$$\mathbf{H}_{\text{pairs}} = [|\cos(\mathbf{H}_i, \mathbf{H}_j)|]_{i < j} \quad \text{padded to } 3d/10$$
*Source: `src/embedding/construction.py:108-129`*

15 unique pairs from 6 dimensions, padded to $3d/10 = 36$ when $d=120$.

**Equation 55** (Resonance subspace):
$$\mathbf{H}_{\text{res}} = [\text{Mean}(\mathbf{R}),\ \text{Max}(\mathbf{R}),\ \text{Final}(\mathbf{R}),\ \text{Slope}(\mathbf{R}),\ \text{Var}(\mathbf{R}),\ \text{FFT}_{\text{low}}(\mathbf{R})]$$
*Source: `src/embedding/construction.py:133-167`*

Per-component statistics from the resonance trajectory, padded to $d/10$.

---

## 4. Variable Dictionary

| Symbol | Meaning | Range | Default | Source in Code |
|--------|---------|-------|---------|---------------|
| $a_d$ | Activation for dim $d$ | $[0, 1]$ | — | `calculator.py:46-48` |
| $\alpha$ | Signal richness exponent | $(0,\infty)$ | $2.0$ | `config.py:55` |
| $\beta$ | Coupling exponent | $(0,\infty)$ | $1.0$ | `config.py:56` |
| $\beta_\downarrow$ | Top-down refinement rate | $(0,1)$ | $0.3$ | `config.py:67` |
| $\beta_{\text{sal}}$ | Salience exponent | $(0,\infty)$ | $2.0$ | `config.py:66` |
| $\beta_j$ | Memory entry activation | $[0,1]$ | $1.0$ | `memory.py:117` |
| $C$ | Coupling factor | $[0,1]$ | — | `calculator.py:54-57` |
| $\tilde{C}$ | Normalized coupling | $[0,1]$ | — | `calculator.py:19` |
| $C_{\text{sim}}$ | Context similarity | $[-1,1]$ | — | `components.py:28-46` |
| $C_{\text{prox}}$ | Narrative proximity | $[0,1]$ | — | `components.py:48-50` |
| $C_{\text{conf}}$ | Confidence similarity | $[0,1]$ | — | `components.py:52-69` |
| $C_{\text{comp}}$ | Semantic compatibility | $[0,1]$ | — | `components.py:71-78` |
| $C_{\text{sym}}$ | Symbolic coherence | $[0,1]$ | — | `components.py:80-89` |
| $C_{\text{rel}}$ | Interaction reliability | $[0,1]$ | — | `components.py:91-106` |
| $d$ | Total embedding dimension | $\{40,120,240,480\}$ | $120$ | `config.py:50` |
| $\delta$ | Memory decay rate | $(0,1)$ | $0.05$ | `config.py:38` |
| $\delta_{\text{exp}}$ | Memory exponent (LDRI) | $(0,\infty)$ | $1.0$ | `config.py:58` |
| $\mathbf{f}_d$ | Feature vector | $\mathbb{R}^{k_d}$ | — | `features/base.py` |
| $\gamma$ | Resonance exponent | $(0,\infty)$ | $1.0$ | `config.py:57` |
| $\Gamma$ | Global confidence | $[0,1]$ | $0.85$ | `calculator.py:67-73` |
| $\gamma_d$ | Per-dimension confidence | $[0,1]$ | — | `features/base.py:14-16` |
| $\mathbf{H}_d$ | Embedding subspace | $\mathbb{R}^{d/10}$ | — | `construction.py:70-104` |
| $\mathbf{H}_{\text{pairs}}$ | Pairwise subspace | $\mathbb{R}^{3d/10}$ | — | `construction.py:108-129` |
| $\mathbf{H}_{\text{res}}$ | Resonance subspace | $\mathbb{R}^{d/10}$ | — | `construction.py:133-167` |
| $k_d$ | Feature count per dim | $\{7,8\}$ | — | `features/*.py` |
| $k_{\text{key}}$ | Memory key dimension | $[1,\infty)$ | $32$ | `config.py:30` |
| $\kappa_{d,d'}$ | Coupling value | $[0,1]$ | — | `kernel.py:21-72` |
| $\lambda$ | Resonance decay | $(0,\infty)$ | $0.3$ | `config.py:44` |
| $\text{LDRI}$ | Final richness score | $[0,1]$ | — | `calculator.py:24` |
| $\text{LDRI}_{\text{raw}}$ | Unweighted richness | $[0,1]$ | — | `calculator.py:75-82` |
| $M$ | Memory factor | $[0,1]$ | — | `calculator.py:64-65` |
| $\tilde{M}$ | Normalized memory | $[0,1]$ | — | `calculator.py:21` |
| $\mu_i$ | Coupling weight center | $[0,1]$ | (per comp) | `config.py:11` |
| $\omega_i$ | Coupling weight width | $(0,\infty)$ | (per comp) | `config.py:12` |
| $\phi_i(\tau)$ | Inverse-distance weight (Eq. 46) | $[0,1]$ | — | `bottom_up.py:259-267` |
| $\rho_{\text{reinforce}}$ | Memory reinforcement rate | $(0,1)$ | $0.1$ | `config.py:37` |
| $\rho$ | Bootstrap reliability | $[0,1]$ | — | `calculator.py:114` |
| $\eta_{\text{res}}$ | Resonance strength | $(0,\infty)$ | $0.2$ | `config.py:45` |
| $\eta_i(\tau)$ | Normalized coupling weight | $[0,1]$ | — | `weights.py:43-50` |
| $\mathbf{R}$ | Resonance field | $\mathbb{R}^{h_{\text{dim}}}$ | — | `field.py` |
| $R_{\text{signal}}$ | Signal richness | $[0,1]$ | — | `calculator.py:51-52` |
| $\sigma_{\max}$ | Activation scale (sigma_max) | $(0,\infty)$ | $5.0$ | `config.py:59` |
| $\sigma_\tau$ | Narrative proximity scale | $(0,\infty)$ | $0.1$ | `config.py:19` |
| $\sigma_{\text{kernel}}$ | RFF kernel bandwidth | $(0,\infty)$ | $1.0$ | `config.py:31` |
| $T$ | Resonance factor | $[0,1]$ | — | `calculator.py:59-62` |
| $\tilde{T}$ | Normalized resonance | $[0,1]$ | — | `calculator.py:20` |
| $\tau$ | Narrative position | $[0,1]$ | — | `orchestrator.py:25-36` |
| $\theta_{\text{store}}$ | Memory storage threshold | $(0,1)$ | $0.5$ | `config.py:34` |
| $\theta_{\text{retrieve}}$ | Memory retrieval threshold | $(0,1)$ | $0.7$ | `config.py:35` |
| $\theta_{\text{forget}}$ | Memory forgetting threshold | $(0,1)$ | $0.01$ | `config.py:36` |

---

## 5. Algorithm

```
Algorithm 1: LIT-DISC Pipeline
Input:  text file path P
Output: LITDISCResult R

 1:  text ← Load(P)
 2:  sentences ← SentenceSplit(text)
 3:  paragraph_groups ← DetectParagraphs(sentences, text)
 4:  chapter_groups ← DetectChapters(paragraph_groups, text)
 5:  tree ← BuildHierarchyTree(sentences, paragraph_groups, chapter_groups)
 6:  sentence_tokens ← [Tokenize(s) for s in sentences]
 7:  τ ← NarrativePosition(sentences, sentence_tokens)       ▷ Eq. 19-20
 8:  tau_map ← {(1, i): τ[i] for i ← 0..len(sentences)-1}
 9:  
10:  ▷ Feature extraction for 6 dimensions
11:  for i ← 0..len(sentences)-1 do
12:      nid ← (1, i)
13:      for dim ∈ {S, N, I, Sy, E, Em} do
14:          f, γ ← FeatureExtractor[dim].Extract(
15:              sentences[i], tokens[i], pos[i], deps[i])    ▷ Eq. 21
16:          features[dim][nid] ← (f, γ)
17:  
18:  ▷ Signal initialization
19:  signals ← SignalInit(features, tau_map)
20:  
21:  ▷ K-iteration refinement loop
22:  for k ← 1..K do
23:      ▷ a. Bottom-up propagation with horizontal coupling
24:      tau_coupling ← k / K
25:      for level ← 2..5 do
26:          for each node at level do
27:              for each child do
28:                  ▷ Memory retrieval & augmentation (Eq. 24, 28)
29:                  matches ← Memory.Retrieve(sig.f)           ▷ Eq. 24
30:                  if matches then
31:                      f_aug ← Memory.Augment(sig.f, matches) ▷ Eq. 28
32:                  ▷ Horizontal coupling (Eq. 37-39)
33:                  κ ← CouplingKernel(tau, f_d, f_d', γ_d, γ_d',
34:                                     τ_d, τ_d')              ▷ Eq. 37
35:                  Λ ← CouplingOperator(κ, f_d, f_d')          ▷ Eq. 38
36:                  f_coupled ← CouplingUpdate(f_d, Λ)           ▷ Eq. 39
37:                  ▷ Memory storage (Eq. 23)
38:                  if Memory.ShouldStore(sig.f, sig.γ) then     ▷ Eq. 23
39:                      Memory.Store(sig)
40:              ▷ Vertical aggregation (Eq. 43-46)
41:              f_agg, γ_agg ← Aggregate(f_coupled, γ, τ)       ▷ Eq. 43
42:              ▷ Uncertainty propagation (Eq. 47-52)
43:              PropagateUncertainty(level, child_ids, nid)
44:  
45:      ▷ b-c. Recursive resonance (Eq. 40-41)
46:      for s ← 0..n_samples-1 do
47:          τ_s ← s * Δτ
48:          H ← InterpolateH(tree, signals, dims, τ_s)
49:          Resonance.Update(H, Δτ)                              ▷ Eq. 40
50:  
51:      ▷ d. Top-down refinement (Eq. 42)
52:      for level ← 4..1 do
53:          for each node at level do
54:              signals[node] ← TopDownRefine(
55:                  signals[node], signals[parent])             ▷ Eq. 42
56:  
57:  ▷ Embedding construction (Eq. 53-55)
58:  embedding ← ConstructEmbedding(signals, tree, resonance_trajectory)
59:  
60:  ▷ LDRI calculation
61:  confidences ← ExtractConfidences(signals)
62:  mem_avg_beta ← Memory.GetAvgBeta()
63:  activations ← ComputeActivations(embedding)                  ▷ Eq. 1
64:  R_signal ← SignalRichness(activations)                       ▷ Eq. 2
65:  C ← CouplingFactor(embedding.H_pairs)                        ▷ Eq. 3
66:  T ← ResonanceFactor(embedding.H_res)                         ▷ Eq. 4
67:  M ← MemoryFactor(mem_avg_beta)                               ▷ Eq. 5
68:  C̃, T̃, M̃ ← NormalizeFactors(C, T, M)                        ▷ Eq. 6
69:  Γ ← GlobalConfidence(confidences)                            ▷ Eq. 7
70:  LDRI_raw ← GeometricMean(R_signal^α, C̃^β, T̃^γ, M̃^δ)      ▷ Eq. 8
71:  LDRI ← Γ × LDRI_raw                                          ▷ Eq. 9
72:  reliability ← BootstrapReliability(embedding)                ▷ Eq. 18
73:  
74:  return LITDISCResult(ldri=LDRI, gamma=Γ, raw=LDRI_raw, ...)
```

---

## 6. Computational Complexity

Assumptions:
- $n$ = number of sentences in the text
- $p$ = number of paragraphs ($p \ll n$)
- $c$ = number of chapters ($c \ll p$)
- $K$ = refinement iterations (default 5)
- $m$ = narrative samples for resonance (default 100)
- $d$ = embedding dimension (default 120)
- $k_d$ = feature vector length per dimension (max 8)
- $M$ = max memory size (default 200)

| Stage | Time Complexity | Space Complexity | Notes |
|-------|---------------|-----------------|-------|
| Load | $O(n)$ | $O(n)$ | File I/O bound |
| Sentence split | $O(n)$ | $O(n)$ | Regex-based |
| Hierarchy build | $O(n + p + c)$ | $O(n + p + c)$ | Tree construction |
| Tokenization | $O(n \cdot \bar{w})$ | $O(n \cdot \bar{w})$ | $\bar{w} \approx 15$ avg words/sentence |
| Feature extraction (6 dims) | $O(n \cdot k_d \cdot L)$ | $O(n \cdot k_d)$ | $L$ = lexicon lookup cost |
| Refinement loop: $K$ iterations | | | |
| — Bottom-up propagation | $K \cdot O(n + p + c)$ | $O(n \cdot k_d)$ | Vertical + horizontal |
| — Coupling kernel (6 dims → 15 pairs) | $K \cdot O(15 \cdot n)$ | $O(k_d)$ | Pairwise operations |
| — Memory retrieval | $K \cdot O(n \cdot M \cdot k_{\text{key}})$ | $O(M \cdot k_{\text{key}})$ | Cosine similarity search |
| — Resonance update | $K \cdot O(m \cdot d)$ | $O(d)$ | $m = 100$ |
| — Top-down refinement | $K \cdot O(n + p + c)$ | $O(n \cdot k_d)$ | Projection |
| Embedding construction | $O(n \cdot k_d + d)$ | $O(d)$ | Statistics computation |
| LDRI calculation | $O(d)$ | $O(1)$ | Bootstrap: $O(B \cdot d)$ |
| **Worst-case total** | $O(K \cdot n \cdot M \cdot k_{\text{key}})$ | $O(n \cdot k_d + M \cdot k_{\text{key}})$ | |
| **Average-case** | $O(K \cdot n \cdot M \cdot k_{\text{key}})$ | $O(M \cdot k_{\text{key}})$ | Memory retrieval dominates |
| **Empirical** | 1.05s per 1k words on standard hardware | — | Maximum 200 entries |

**Dominant term:** Memory retrieval ($n \cdot M$ pairwise comparisons per iteration) with $K=5$ and $M=200$.

---

## 7. Parameter Table

| Parameter | Meaning | Default | Reason | Sensitivity | Source File |
|-----------|---------|---------|--------|-------------|-------------|
| `sigma_max` | Activation scale | 5.0 | Aligns with feature norms (10-14), prevents saturation | HIGH: controls activation spread | `config.py:59` |
| `alpha` | Signal richness weight | 2.0 | Doubles signal contribution vs other factors | LOW: all factors near-constant | `config.py:55` |
| `beta` | Coupling weight | 1.0 | Baseline coupling importance | LOW: coupling is near-constant | `config.py:56` |
| `gamma` | Resonance weight | 1.0 | Baseline resonance importance | LOW: resonance is near-zero | `config.py:57` |
| `delta` | Memory weight | 1.0 | Baseline memory importance | LOW: memory is near-constant | `config.py:58` |
| `key_dim` | RFF key dimension | 32 | Sufficient for cosine similarity | MEDIUM: affects retrieval | `config.py:30` |
| `sigma_kernel` | RFF kernel bandwidth | 1.0 | Controls RFF output distribution | LOW: deterministic anyway | `config.py:31` |
| `theta_store` | Memory storage threshold | 0.5 | Balances storage vs capacity | MEDIUM: affects fill rate | `config.py:34` |
| `theta_retrieve` | Memory retrieval threshold | 0.7 | Balances precision vs recall | HIGH: zero retrievals at 0.7 | `config.py:35` |
| `theta_forget` | Memory forgetting threshold | 0.01 | Prevents near-zero entries | LOW: decay handles most | `config.py:36` |
| `rho_reinforce` | Memory reinforcement rate | 0.1 | Gradual strength increase | LOW: rarely triggered | `config.py:37` |
| `delta_decay` | Memory decay rate | 0.05 | Moderate per-iteration decay | LOW: affects entry lifespan | `config.py:38` |
| `max_size` | Memory capacity | 200 | Memory constraint | MEDIUM: reached immediately | `config.py:39` |
| `lambda_decay` | Resonance decay constant | 0.3 | Long-range narrative influence | LOW: field stays near-zero | `config.py:44` |
| `eta_resonance` | Resonance strength | 0.2 | Amplitude multiplier | LOW: product with near-zero field | `config.py:45` |
| `dimension` | Embedding dimension | 120 | Standard setting | MEDIUM: affects subspace sizes | `config.py:50` |
| `refinement_iterations` | K-iteration count | 5 | Convergence balance | LOW: signals stable after 1-2 | `config.py:65` |
| `beta_salience` | Salience exponent | 2.0 | Sharpens selection | LOW: small effect on aggregation | `config.py:66` |
| `beta_topdown` | Top-down refinement rate | 0.3 | Modest parent influence | LOW: small update magnitude | `config.py:67` |
| `sigma_tau` | Proximity scale | 0.1 | Tight narrative window | LOW: proximity always near-1 | `config.py:19` |

---

## 8. Activation Design

### 8.1 Original Implementation (v1.0)

In v1.0, `sigma_max = 0.1`. The activation function was:

$$a_d = 1 - \exp\left(-\frac{\|\mathbf{H}_d\|^2}{2 \cdot (0.1)^2}\right)$$

### 8.2 Diagnostic Findings

Feature norms $\|\mathbf{H}_d\|$ were in the range 10-14 (mean ≈ 11.09). At sigma=0.1:

$$z = \frac{\|\mathbf{H}_d\|}{\sigma_{\max}} = \frac{11}{0.1} = 110$$

$$a_d = 1 - \exp(-110^2 / 2) \approx 1.0$$

### 8.3 Why Saturation Occurred

The Rayleigh CDF reaches $g(x) \approx 1$ for $x > 5\sigma$. With $\|\mathbf{H}_d\| \approx 11$ and $\sigma_{\max} = 0.1$, the argument $\frac{\|\mathbf{H}_d\|}{\sigma_{\max}} \approx 110$ is 22 times larger than the $5\sigma$ threshold. Result: **149 out of 150 activations saturated at $> 0.999$**.

The LDRI CV was $0.31\%$ — the metric was effectively constant.

### 8.4 Why Sigma Changed

`src/config.py:59`: `sigma_max: float = 0.1` was changed to `sigma_max: float = 5.0`.

At $\sigma_{\max} = 5.0$:

$$z = \frac{11}{5.0} = 2.2$$

$$a_d = 1 - \exp(-2.2^2 / 2) = 1 - \exp(-2.42) \approx 0.911$$

This operates in the sensitive region of the Rayleigh CDF, where a change in $\|\mathbf{H}_d\|$ produces a proportional change in $a_d$.

### 8.5 Mathematical Effect

| Metric | v1.0 ($\sigma=0.1$) | v1.1 ($\sigma=5.0$) | Change |
|--------|--------------------|--------------------|--------|
| Saturated activations | 149/150 (99%) | 6/150 (4%) | −95% |
| Mean activation | 0.996 | 0.575 | −42% |
| Activation CV | 0.2% | 22.1% | 111× |
| LDRI mean | 0.9008 | 0.0277 | −97% |
| LDRI CV | 0.31% | 47.9% | 155× |
| Raw LDRI mean | 0.9131 | 0.9128 | Unchanged |
| Author Spearman $\rho$ | — | 0.98 (vs v1.0) | Ranking preserved |

### 8.6 Final Implementation

`src/ldri/calculator.py:46-48`:

```python
norm = float(np.linalg.norm(Hd))
a_d = 1.0 - math.exp(-norm ** 2 / (2.0 * sigma ** 2))
```

With `sigma = self.params.sigma_max = 5.0`.

---

## 9. Implementation Mapping

| Paper Eq. | Description | Python Function | File | Lines |
|-----------|-------------|-----------------|------|-------|
| Eq. 1 | Activation (Rayleigh CDF) | `LDRICalculator.compute_activations` | `src/ldri/calculator.py` | 37-49 |
| Eq. 2 | Signal richness | `LDRICalculator.signal_richness` | `src/ldri/calculator.py` | 51-52 |
| Eq. 3 | Coupling factor | `LDRICalculator.coupling_factor` | `src/ldri/calculator.py` | 54-57 |
| Eq. 4 | Resonance factor | `LDRICalculator.resonance_factor` | `src/ldri/calculator.py` | 59-62 |
| Eq. 5 | Memory factor | `LDRICalculator.memory_factor` | `src/ldri/calculator.py` | 64-65 |
| Eq. 6 | Normalized factors | Inline in `compute` | `src/ldri/calculator.py` | 19-21 |
| Eq. 7 | Global confidence | `LDRICalculator.global_confidence` | `src/ldri/calculator.py` | 67-73 |
| Eq. 8 | Raw LDRI | `LDRICalculator._ldri_raw` | `src/ldri/calculator.py` | 75-82 |
| Eq. 9 | Final LDRI | Inline in `compute` | `src/ldri/calculator.py` | 24 |
| Eq. 10 | Inverse normalization | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 122-124 |
| Eq. 11 | Intrinsic component | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 126 |
| Eq. 12 | Interaction component | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 127 |
| Eq. 13 | Temporal component | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 128 |
| Eq. 14 | Resonance component | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 129 |
| Eq. 15 | Memory component | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 130 |
| Eq. 16 | Uncertainty component | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 131 |
| Eq. 17 | Fraction normalization | `LDRICalculator._decompose` | `src/ldri/calculator.py` | 141-143 |
| Eq. 18 | Bootstrap reliability | `LDRICalculator._estimate_reliability` | `src/ldri/calculator.py` | 84-115 |
| Eq. 19 | Narrative velocity | `NarrativePositionComputer.compute` | `src/pipeline/orchestrator.py` | 30-36 |
| Eq. 20 | Cumulative narrative pos | `NarrativePositionComputer.compute` | `src/pipeline/orchestrator.py` | 35-36 |
| Eq. 21 | Per-dimension confidence | `BaseFeatureExtractor.compute_confidence` | `src/features/base.py` | 14-16 |
| Eq. 22 | RFF key projection | `RandomFourierFeatures.project` | `src/coupling/memory.py` | 48-56 |
| Eq. 23 | Storage decision | `MemorySystem.should_store` | `src/coupling/memory.py` | 107-108 |
| Eq. 24 | Retrieval | `MemorySystem.retrieve` | `src/coupling/memory.py` | 136-164 |
| Eq. 25 | Reinforcement | `MemorySystem.reinforce` | `src/coupling/memory.py` | 177-192 |
| Eq. 26 | Decay | `MemorySystem.decay_all` | `src/coupling/memory.py` | 196-221 |
| Eq. 27 | Forgetting | `MemorySystem.purge` | `src/coupling/memory.py` | 231-240 |
| Eq. 28 | Augmentation | `MemorySystem.augment` | `src/coupling/memory.py` | 244-258 |
| Eq. 29 | Context similarity | `KernelComponents.context_similarity` | `src/coupling/components.py` | 28-46 |
| Eq. 30 | Narrative proximity | `KernelComponents.narrative_proximity` | `src/coupling/components.py` | 48-50 |
| Eq. 31 | Confidence similarity | `KernelComponents.confidence` | `src/coupling/components.py` | 52-69 |
| Eq. 32 | Compatibility | `KernelComponents.compatibility` | `src/coupling/components.py` | 71-78 |
| Eq. 33 | Symbolic coherence | `KernelComponents.symbolic_coherence` | `src/coupling/components.py` | 80-89 |
| Eq. 34 | Interaction reliability | `KernelComponents.interaction_reliability` | `src/coupling/components.py` | 91-106 |
| Eq. 35 | Unnormalized weight | `AdaptiveWeights.compute_raw` | `src/coupling/weights.py` | 27-41 |
| Eq. 36 | Normalized weight | `AdaptiveWeights.compute` | `src/coupling/weights.py` | 43-50 |
| Eq. 37 | Coupling kernel | `CouplingKernel.compute` | `src/coupling/kernel.py` | 21-72 |
| Eq. 38 | Coupling operator Λ | `CouplingOperator.apply` | `src/coupling/kernel.py` | 128-144 |
| Eq. 39 | Signal update | `CouplingOperator.update` | `src/coupling/kernel.py` | 146-151 |
| Eq. 40 | Resonance update | `NarrativeResonance.update` | `src/resonance/field.py` | 35-57 |
| Eq. 41 | Resonance augmentation | `NarrativeResonance.augment` | `src/resonance/field.py` | 59-72 |
| Eq. 42 | Top-down refinement | `TopDownRefiner.refine` | `src/propagation/top_down.py` | 14-57 |
| Eq. 43 | Weighted aggregation | `BottomUpPropagator._aggregate_weighted` | `src/propagation/bottom_up.py` | 218-228 |
| Eq. 44 | Semantic coherence | `BottomUpPropagator._semantic_coherence` | `src/propagation/bottom_up.py` | 230-245 |
| Eq. 45 | Salience weights | `BottomUpPropagator._salience_weights` | `src/propagation/bottom_up.py` | 247-257 |
| Eq. 46 | Spline weights | `BottomUpPropagator._spline_weights` | `src/propagation/bottom_up.py` | 259-267 |
| Eq. 47 | Total variance (13.1) | `UncertaintyPropagator.total_variance` | `src/uncertainty/propagation.py` | 78-82 |
| Eq. 48 | Confidence from variance (13.2) | `UncertaintyPropagator.refine_confidence` | `src/uncertainty/propagation.py` | 90-93 |
| Eq. 49 | Sentence→Paragraph (13.3) | `UncertaintyPropagator.propagate_sentence_to_paragraph` | `src/uncertainty/propagation.py` | 102-127 |
| Eq. 50 | Paragraph→Chapter (13.4) | `UncertaintyPropagator.propagate_paragraph_to_chapter` | `src/uncertainty/propagation.py` | 131-155 |
| Eq. 51 | Chapter→Narrative (13.5) | `UncertaintyPropagator.propagate_chapter_to_narrative` | `src/uncertainty/propagation.py` | 159-189 |
| Eq. 52 | Coupling uncertainty (13.6) | `UncertaintyPropagator.coupling_uncertainty` | `src/uncertainty/propagation.py` | 219-223 |
| Eq. 53 | Per-dim subspace (14.2) | `NarrativeEmbeddingConstructor._compute_dim_subspace` | `src/embedding/construction.py` | 70-104 |
| Eq. 54 | Pairwise subspace | `NarrativeEmbeddingConstructor._compute_pairs` | `src/embedding/construction.py` | 108-129 |
| Eq. 55 | Resonance subspace | `NarrativeEmbeddingConstructor._compute_resonance_subspace` | `src/embedding/construction.py` | 133-167 |

---

## 10. Reproducibility Notes

### 10.1 Preprocessing

- **Sentence splitting:** Regex-based (`re.split(r"(?<=[.?!])\s+", text)`) with abbreviation handling (40+ abbreviations).
- **Tokenization:** Regex-based rule tokenizer (not spaCy; the `model` parameter is declared but never used).
- **POS tagging:** Rule-based heuristic (suffix analysis, capitalization, keyword matching). Not a trained model.
- **Dependency parsing:** `_guess_deps()` returns `[]` (empty). Dependencies are not computed.

### 10.2 Feature Extraction

All 6 feature extractors use lexicon-based keyword matching and simple arithmetic. No trained models. Features are normalized to $[0,1]$ or $[-1,1]$ per extractor.

### 10.3 Assumptions

1. Text is prose narrative fiction (not poetry, drama, or expository).
2. English language (lexicons are English-only).
3. Windows of ~200 words (gold standard) or ~800 words (novel evaluation).
4. The 5-level hierarchy is fixed (sentence → paragraph → chapter → narrative → discourse).
5. There is exactly one narrative layer (level 4) and one discourse layer (level 5).

### 10.4 Dependencies

- `numpy`: All vector operations
- `math` (stdlib): Elementary functions
- `re` (stdlib): Regex tokenization
- `chardet` (optional): Encoding detection
- `ebooklib` (optional): EPUB loading
- No deep learning or external API dependencies

### 10.5 Deterministic Behavior

The pipeline is fully deterministic given the same input text and configuration. Randomness is controlled by explicit seeds:

| Component | Seeds | Location |
|-----------|-------|----------|
| RFF W matrix | seed_W = 0 (+ input dim) | `src/config.py:32`, `src/coupling/memory.py:43` |
| RFF b vector | seed_b = 1 (+ input dim) | `src/config.py:33`, `src/coupling/memory.py:44` |
| Bootstrap noise | rng_seed = 42 | `src/ldri/calculator.py:88` |

### 10.6 Configuration

All parameters are in `src/config.py` as frozen dataclasses. The only v1.0 → v1.1 change is `sigma_max: 0.1 → 5.0` on line 59.

---

## 11. Figures to Draw for the Paper

### Figure 1: Framework Architecture
Block diagram showing: Text → Load → Split → Detect → Build Tree → Tokenize → τ-compute → Feature Extraction (6 boxes) → Signal Init → Refinement Loop (K iterations, 4 sub-steps) → Embedding → LDRI. Color-coded by stage.

### Figure 2: Processing Pipeline
Detailed dataflow diagram showing how text becomes sentences, then features, then signals, then propagated signals, then embedding, then LDRI score. Include intermediate data types.

### Figure 3: Equation Dependency Graph
DAG showing the 55 equations as nodes and their dependencies as edges. Root: text. Leaves: LDRI, components, fractions. Highlight the critical path through Eq. 1 → Eq. 2 → Eq. 8 → Eq. 9.

### Figure 4: Data Flow with Dimensions
Show how each of the 6 dimensions (S, N, I, Sy, E, Em) flows through: feature extraction → signal → coupling → propagation → embedding → LDRI.

### Figure 5: Dimension Interaction Diagram
The 6 dimensions arranged in a hexagon with edges showing coupling paths (Eq. 37-39), resonance feedback (Eq. 40-41), and uncertainty propagation (Eq. 47-52).

### Figure 6: Activation Function Comparison
Plot of $g(x) = 1 - \exp(-x^2 / 2\sigma^2)$ for $\sigma = 0.1$ (v1.0, saturated) and $\sigma = 5.0$ (v1.1, sensitive). Mark typical $\|\mathbf{H}_d\|$ range (10-14) to show operating region.

### Figure 7: Hierarchy Tree
Visual representation of the 5-level hierarchy tree with example node counts: $n$ sentences → $p$ paragraphs → $c$ chapters → 1 narrative → 1 discourse.

### Figure 8: Memory Lifecycle
Flowchart showing: Feature vector $\mathbf{f}$ → RFF projection (Eq. 22) → Storage check (Eq. 23) → Store/Reject → Retrieval (Eq. 24) → Reinforcement (Eq. 25) → Decay (Eq. 26) → Purge (Eq. 27).

### Figure 9: Coupling Kernel Composition
Pie or bar chart showing the 6 coupling components ($C_{\text{sim}}$ through $C_{\text{rel}}$) and their adaptive weights $\eta_i(\tau)$ as a function of $\tau$.

### Figure 10: LDRI Score Composition
Stacked bar showing LDRI decomposed into 6 components (Eq. 11-16) for a representative passage, demonstrating gamma dominance.
