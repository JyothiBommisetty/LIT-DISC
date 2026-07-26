# LIT-DISC Software Architecture Specification (v1.0)

## Implementation Architecture for HMP-DT and LDRI

**Document Type:** Software Architecture Specification  
**Framework:** LIT-DISC (Literary Intelligence through Text-based Discourse Analysis)  
**Prerequisites:** HMP-DT Version 1.0 (Research Freeze), LDRI Methodology, Evaluation Methodology  

---

## Table of Contents

1. [High-Level Pipeline](#1)
2. [Module Decomposition](#2)
3. [Data Flow](#3)
4. [Internal Data Structures](#4)
5. [Interfaces](#5)
6. [Storage Formats](#6)
7. [Configuration System](#7)
8. [Explainability Storage](#8)
9. [Memory Subsystem](#9)
10. [Computational Graph](#10)
11. [Parallelization Opportunities](#11)
12. [Error Handling](#12)
13. [Logging](#13)
14. [Reproducibility](#14)
15. [Testing Strategy](#15)

---

## 1. High-Level Pipeline {#1}

### 1.1 Pipeline Overview

The LIT-DISC pipeline transforms raw literary text into a structured discourse analysis through six sequential processing stages:

```
[Text Input]
     |
     v
[Stage 1: Loading & Preprocessing] ─── Tokenization, sentence/paragraph/chapter detection
     |
     v
[Stage 2: Feature Extraction (×6)] ─── Per-dimension feature computation
     |
     v
[Stage 3: HMP-DT Core (K iterations)] ─── Propagation, coupling, memory, resonance,
     |                                      refinement
     v
[Stage 4: Narrative Embedding] ─── Subspace construction and aggregation
     |
     v
[Stage 5: LDRI Calculation] ─── Signal norm, coupling factor, geometric mean,
     |                           confidence, decomposition
     v
[Stage 6: Output Generation] ─── Explanation traces, reports, visualization data
```

### 1.2 Stage Ownership by Module

| Stage | Primary Module | Secondary Modules |
|-------|---------------|-------------------|
| 1 | Loader, Preprocessor | — |
| 2 | FeatureExtractor (×6) | NarrativePosition |
| 3 | HMPDTEngine | Propagator, Coupler, Memory, Resonance, Uncertainty, Refiner |
| 4 | NarrativeEmbedding | — |
| 5 | LDRICalculator | ConfidenceEstimator, Decomposer |
| 6 | ExplanationGenerator, ReportGenerator, Visualizer | — |

---

## 2. Module Decomposition {#2}

### 2.1 Module Map

Every module is keyed to its governing equations in the HMP-DT freeze (`HMP-DT §E`) and LDRI methodology (`LDRI §E`).

#### 2.1.1 Loader (`modules/loader.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Ingest plain text, EPUB, TEI-XML | — |
| Detect encoding (UTF-8, Latin-1) | — |
| Output raw text string | — |

#### 2.1.2 Preprocessor (`modules/preprocessor.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Sentence segmentation | HMP-DT §10.1 (ℓ = 1) |
| Paragraph detection (blank-line heuristic) | HMP-DT §10.2 (ℓ = 2) |
| Chapter detection (heading patterns, numbering) | HMP-DT §10.3 (ℓ = 3) |
| Tokenization, POS tagging, dependency parsing | — |
| Output: hierarchical node structure | — |

#### 2.1.3 Narrative Position Calculator (`modules/narrative_position.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Compute narrative velocity $v[t] = 1/(1 + \epsilon_N[t] + \epsilon_D[t] + \epsilon_R[t])$ | HMP-DT §15, algorithm step 1 |
| Cumulative integration and normalization $\tau[0..N_s] = \text{cumsum}(v) / \text{max}(\text{cumsum}(v))$ | HMP-DT §15, algorithm step 1 |
| Assign $\tau_i$ to every sentence node | HMP-DT Eq. (10.1) |
| Spline interpolation to $m = 100$ sample points | HMP-DT Eq. (10.7) |

#### 2.1.4 Feature Extractors (six modules)

| Module | Dimension | Key Features | Equation Reference |
|--------|-----------|--------------|-------------------|
| `extractors/semantic.py` | $S$ | Lexical density, polysemy, concreteness, semantic coherence | HMP-DT Feature Dictionary, Features A1–A8 |
| `extractors/narrative.py` | $N$ | Tension, pacing, scene entropy, narrative trajectory | HMP-DT Feature Dictionary, Features B1–B8 |
| `extractors/interaction.py` | $I$ | Speaker-hearer network, turn asymmetry, dialogue density | HMP-DT Feature Dictionary, Features C1–C7 |
| `extractors/symbolic.py` | $Sy$ | Figurative density, motif persistence, symbolic coherence | HMP-DT Feature Dictionary, Features D1–D7 |
| `extractors/evolution.py` | $E$ | Arc slope, relationship state, developmental trajectory | HMP-DT Feature Dictionary, Features E1–E7 |
| `extractors/emotion.py` | $Em$ | Valence, arousal, intensity, emotional trajectory | HMP-DT Feature Dictionary, Features F1–F7 |

Each extractor produces $\mathbf{f}_{i}^{(\ell)}[d] \in \mathbb{R}^{k_d}$ and $\gamma_{i}^{(\ell)}[d] \in [0,1]$ for every node at every level.

#### 2.1.5 Signal Initializer (`modules/signal_initializer.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Construct $\mathcal{S}_{S_i}^{(d)} = \langle \mathbf{f}, \gamma, \tau, \emptyset, \{S_i\}, \mathbf{0}_{6\times6}, \emptyset \rangle$ | HMP-DT Eq. (10.1) |
| Assign narrative position $\tau_i$ from NarrativePosition | — |
| Initialize empty interpretive context $\mathcal{C}$, provenance $\mathcal{P}$, interaction residues $\mathcal{R}$, memory traces $\mathcal{M}$ | HMP-DT Eq. (10.1) |

#### 2.1.6 Vertical Propagator (`modules/propagator.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| **Sentence → Paragraph:** confidence-weighted aggregation | HMP-DT Eq. (10.2)–(10.4) |
| **Paragraph → Chapter:** salience-weighted aggregation with $\alpha_j^{(d)}$ | HMP-DT Eq. (10.5)–(10.6) |
| **Chapter → Narrative Layer:** natural cubic spline interpolation | HMP-DT Eq. (10.7) |
| **Narrative Layer → Discourse Layer:** linear projection via $\mathbf{W}_d$ | HMP-DT Eq. (10.8) |
| **Top-down refinement:** project parent signal downward with $\beta_{\downarrow} = 0.3$ | HMP-DT Eq. (10.9) |

#### 2.1.7 Horizontal Coupler (`modules/coupler.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Compute adaptive weights $\eta_i(\tau)$ using raised cosine basis | HMP-DT Eq. (11.1)–(11.2) |
| Compute kernel components $C_{sim}, C_{prox}, C_{conf}, C_{comp}, C_{sym}, C_{rel}$ | HMP-DT Eq. (11.3)–(11.8) |
| Combine into $\kappa_{d,d'}(\tau) = \sum_i \eta_i(\tau) C_i$ | HMP-DT Eq. (11.9) |
| Apply coupling operator $\Lambda_{d,d'}(\mathbf{a}, \mathbf{b})$ | HMP-DT Eq. (11.10) |
| Update feature vectors $\tilde{\mathbf{f}}_d = \mathbf{f}_d + \sum_{d'\neq d} \Lambda_{d,d'}$ | HMP-DT Eq. (11.11) |

#### 2.1.8 Memory Subsystem (`modules/memory.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Compute Fourier feature key $\mathbf{k}(\mathbf{f}) = \sqrt{2/k_{\text{key}}} \cos(\mathbf{Wf} + \mathbf{b})$ | HMP-DT Eq. (12.2) |
| Store signals exceeding $\|\mathbf{f}_i\| \cdot \gamma_i > \theta_{\text{store}}$ | HMP-DT Eq. (12.1) |
| Retrieve signals with $\cos(\mathbf{k}, \mathbf{k}_j) > \theta_{\text{retrieve}}$ | HMP-DT Eq. (12.3) |
| Reinforce and decay memory activations | HMP-DT Eq. (12.4)–(12.5) |
| Purge memories below $\theta_{\text{forget}}$ | HMP-DT Eq. (12.6) |
| Augment current signal with retrieved memories | HMP-DT Eq. (12.7) |

#### 2.1.9 Resonance Calculator (`modules/resonance.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Recursive resonance update $\mathcal{R}(\tau_i) = \mathcal{R}(\tau_{i-1})e^{-\Delta\tau/\lambda} + \mathbf{H}(\tau_{i-1})\Delta\tau$ | HMP-DT Eq. (12.8) |
| Augment current signal $\mathbf{f}_{\text{res}}(\tau) = \mathbf{f}(\tau) + \eta_{\text{res}} \mathcal{R}(\tau)$ | HMP-DT Eq. (12.9) |

#### 2.1.10 Uncertainty Propagator (`modules/uncertainty.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Compute total uncertainty $\sigma_{\text{total}}^2 = \sigma_e^2 + \sigma_s^2 + \sigma_i^2 + \sigma_t^2 + \sigma_m^2$ | HMP-DT Eq. (13.1) |
| Compute confidence $\gamma = 1 - \min(1.0, \sigma_{\text{total}} / \sigma_{\max})$ | HMP-DT Eq. (13.2) |
| Propagate uncertainty: Sentence → Paragraph | HMP-DT Eq. (13.3) |
| Propagate uncertainty: Paragraph → Chapter | HMP-DT Eq. (13.4) |
| Propagate uncertainty: Chapter → Narrative Layer | HMP-DT Eq. (13.5) |
| Coupling uncertainty $\sigma_i^2(d, d')$ | HMP-DT Eq. (13.6) |

#### 2.1.11 Narrative Embedding Constructor (`modules/embedding.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Construct $\mathbf{H}_d = [\text{Mean}, \text{Max}, \text{Slope}, \text{Var}, \text{FFT}_{\text{low}}]$ for each $d$ | HMP-DT Eq. (14.2); LDRI Eq. (1) |
| Construct $\mathbf{H}_{\text{pairs}}$ from interaction residues | HMP-DT Eq. (11.10)–(11.11) |
| Construct $\mathbf{H}_{\text{res}}$ from resonance field statistics | HMP-DT Eq. (12.8) |
| Concatenate: $\mathbf{H} = \bigoplus_d \mathbf{H}_d \oplus \mathbf{H}_{\text{pairs}} \oplus \mathbf{H}_{\text{res}}$ | HMP-DT Eq. (14.1); LDRI Eq. (1) |

#### 2.1.12 LDRI Calculator (`modules/ldri_calculator.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Compute dimension activation $a_d = 1 - \exp(-\|\mathbf{H}_d\|^2 / 2\sigma_d^2)$ | LDRI Eq. (2) |
| Compute signal richness norm $\mathcal{R}_{\text{signal}} = \|\mathbf{a}\|_2 / \sqrt{6}$ | LDRI Eq. (3)–(4) |
| Compute coupling factor $\tilde{\mathcal{C}} = (1 + \mathcal{C}) / 2$ | LDRI Eq. (5), (9) |
| Compute resonance factor $\tilde{\mathcal{T}} = (1 + \mathcal{T}) / 2$ | LDRI Eq. (6), (9) |
| Compute memory factor $\tilde{\mathcal{M}} = (1 + \mathcal{M}) / 2$ | LDRI Eq. (7), (9) |
| Global confidence $\Gamma = \prod_d \gamma_d$ | LDRI Eq. (11)–(12) |
| LDRI = $\Gamma \cdot (\mathcal{R}_{\text{signal}}^\alpha \cdot \tilde{\mathcal{C}}^\beta \cdot \tilde{\mathcal{T}}^\gamma \cdot \tilde{\mathcal{M}}^\delta)^{1/S}$ | LDRI Eq. (8), (10) |

Default exponents: $\alpha = 2,\ \beta = \gamma = \delta = 1$.

#### 2.1.13 Decomposer (`modules/decomposer.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Intrinsic richness $c_{\text{int}}$ | LDRI Eq. (16) |
| Interaction richness $c_{\text{intr}}$ | LDRI Eq. (17) |
| Temporal richness $c_{\text{temp}}$ | LDRI Eq. (18) |
| Resonance contribution $c_{\text{res}}$ | LDRI Eq. (19) |
| Memory contribution $c_{\text{mem}}$ | LDRI Eq. (20) |
| Uncertainty adjustment $c_{\text{unc}}$ | LDRI Eq. (21)–(22) |
| Contribution fractions $f_{\text{int}}, f_{\text{intr}}, \ldots$ | LDRI Eq. (23) |

#### 2.1.14 Explanation Generator (`modules/explanation_generator.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Signal lineage: trace most influential provenance path | HMP-DT §15, algorithm step 7 |
| Propagation path: $\prod$ max product path from root to leaf | HMP-DT §15, algorithm step 7 |
| Interaction history: strongest $\kappa_{d,d'}$ pairs | HMP-DT §15, algorithm step 7 |
| Uncertainty evolution: $\sigma^2$ at each hierarchical level | HMP-DT §15, algorithm step 7 |
| Resonance contribution: $\|\eta_{\text{res}} \mathcal{R}\| / \|\mathbf{H}\|$ | HMP-DT §15, algorithm step 7 |
| Memory activation: active memories with keys and strengths | HMP-DT §15, algorithm step 7 |

#### 2.1.15 Refinement Controller (`modules/refinement_controller.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Orchestrate $K = 5$ refinement iterations | HMP-DT §15, algorithm step 4 |
| Check convergence: $\|\mathbf{S}^{[k]} - \mathbf{S}^{[k-1]}\| < \epsilon$ | HMP-DT §15, algorithm step 4h |
| Early termination if converged | HMP-DT §15, algorithm step 4h |

#### 2.1.16 Configuration Manager (`modules/config.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Load YAML config file | — |
| Validate all parameters against schema | — |
| Provide immutable Config dataclass | — |
| Supply defaults matching HMP-DT freeze | HMP-DT §15, algorithm config |

#### 2.1.17 Report Generator (`modules/report_generator.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Format LDRI triple $(0.68 \pm 0.03, \Gamma = 0.87, \text{Reliability} = 0.92)$ | LDRI §5.4 |
| Render decomposition table | LDRI §6 |
| Render explanation trace | HMP-DT §7.10 |
| Export as LaTeX, Markdown, JSON | — |

#### 2.1.18 Visualization Generator (`modules/visualizer.py`)

| Responsibility | Equation Reference |
|---------------|-------------------|
| Plot narrative position $\tau$ vs. dimension activations | — |
| Plot coupling heatmap $\kappa_{d,d'}$ | — |
| Plot uncertainty evolution $\sigma^2(\ell)$ | — |
| Plot LDRI decomposition as stacked bar | — |
| Plot narrative embedding subspace projections (PCA) | — |

### 2.2 Module Dependency Graph

```
Loader
  |
  v
Preprocessor ──→ NarrativePosition
  |                    |
  v                    v
FeatureExtractor_S  ←─┘
FeatureExtractor_N
FeatureExtractor_I     SignalInitializer
FeatureExtractor_Sy        |
FeatureExtractor_E         v
FeatureExtractor_Em    HMPDTEngine
                         |    |
                    ┌────┘    └──────────┐
                    v                   v
              RefinementController    Propagator
                    |                 Coupler
                    |                 Memory
                    |                 Resonance
                    |                 Uncertainty
                    v                    |
              (K iterations)             v
                                   NarrativeEmbedding
                                         |
                                         v
                                   LDRICalculator
                                         |
                                    ┌────┴────┐
                                    v         v
                              Decomposer   ConfidenceEstimator
                                    |         |
                                    v         v
                              ExplanationGenerator
                                    |
                                    v
                              ReportGenerator  Visualizer
```

---

## 3. Data Flow {#3}

### 3.1 Flow Diagram

```
Raw Text (str)
    |
    v
[Preprocessor]
    |
    ├── Sentences: List[str]
    ├── Paragraphs: List[List[int]]   (sentence index ranges)
    ├── Chapters: List[List[int]]     (paragraph index ranges)
    └── τ: Array[float, N_sentences]
    |
    v
[Feature Extractors × 6]
    |
    ├── f_S: Dict[NodeID → Array[float, k_S=8]], γ_S: Dict[NodeID → float]
    ├── f_N: Dict[NodeID → Array[float, k_N=8]], γ_N: Dict[NodeID → float]
    ├── f_I: Dict[NodeID → Array[float, k_I=7]], γ_I: Dict[NodeID → float]
    ├── f_Sy: Dict[NodeID → Array[float, k_Sy=7]], γ_Sy: Dict[NodeID → float]
    ├── f_E: Dict[NodeID → Array[float, k_E=7]], γ_E: Dict[NodeID → float]
    └── f_Em: Dict[NodeID → Array[float, k_Em=7]], γ_Em: Dict[NodeID → float]
    |
    v
[Signal Initializer]
    |
    └── S: Dict[NodeID → LiterarySignal]  (per dimension)
    |
    v
[HMP-DT Engine — K=5 iterations]
    |
    Each iteration:
    │
    ├── [Memory]     ── retrieve → augment → store → decay
    ├── [Coupler]    ── compute κ → apply Λ → update f
    ├── [Propagator] ── bottom-up aggregation
    │                    (S→P→C→NL→DL)
    ├── [Resonance]  ── recursive R(τ) update
    ├── [Uncertainty]── propagate σ² up the hierarchy
    └── [Refiner]    ── top-down refinement
    |
    v
[Converged State: S_root, S_leaves, M_store, R(τ)]
    |
    v
[Narrative Embedding]
    |
    ├── H_S, H_N, H_I, H_Sy, H_E, H_Em: Array[float, 12]
    ├── H_pairs: Array[float, 36]
    ├── H_res: Array[float, 12]
    └── H: Array[float, 120]  (concatenated)
    |
    v
[LDRI Calculator]
    |
    ├── LDRI: float ∈ [0,1]
    ├── Γ: float ∈ [0,1]
    ├── Reliability: float ∈ [0,1]
    └── Decomposition: {c_int, c_intr, c_temp, c_res, c_mem, c_unc}
    |
    v
[Explanation Generator]
    |
    └── E = {lineage, path, interaction, uncertainty, resonance, memory}
```

### 3.2 Data Volume by Stage

| Stage | Data Type | Per-Sentence Volume | Per-Text Volume (N_S=5000) |
|-------|-----------|---------------------|---------------------------|
| Preprocessing | Tokens | ~20 tokens | ~100K tokens |
| Feature extraction | f ∈ ℝ⁷⁻⁸, γ ∈ ℝ | ~45 floats | ~225K floats |
| Signal initialization | S (7-tuple) | ~2 KB | ~10 MB |
| HMP-DT engine | Intermediate states | ~10 KB | ~50 MB |
| Narrative embedding | H ∈ ℝ¹²⁰ | — | 120 floats |
| LDRI | Scalar + 6 components | — | 7 floats |

### 3.3 Type Annotations (Conceptual)

```
NodeID = tuple[int, int]  # (hierarchy_level, node_index)
Dimension = Literal["S", "N", "I", "Sy", "E", "Em"]

@dataclass(frozen=True)
class LiterarySignal:
    f: NDArray[float]         # feature vector, ℝ^k_d
    gamma: float              # confidence, [0,1]
    tau: float                # narrative position, [0,1]
    context: ContextVector    # interpretive context
    provenance: list[NodeID]  # parent chain
    residues: NDArray[float]  # interaction residues, [0,1]^(6×6)
    memory_traces: list[MemoryTrace]  # (key, activation) pairs

@dataclass(frozen=True)
class NarrativeEmbedding:
    H_S: NDArray[float]       # ℝ^{d/10}
    H_N: NDArray[float]       # ℝ^{d/10}
    H_I: NDArray[float]       # ℝ^{d/10}
    H_Sy: NDArray[float]      # ℝ^{d/10}
    H_E: NDArray[float]       # ℝ^{d/10}
    H_Em: NDArray[float]      # ℝ^{d/10}
    H_pairs: NDArray[float]   # ℝ^{3d/10}
    H_res: NDArray[float]     # ℝ^{d/10}
    d: int = 120              # total dimension

@dataclass(frozen=True)
class LDRIScore:
    ldri: float               # [0,1], point estimate
    gamma_global: float       # [0,1], global confidence
    reliability: float        # [0,1], bootstrap stability
    decomposition: dict[str, float]  # component → value
    fractions: dict[str, float]      # component → fraction of raw
    raw: float                # LDRI_raw (Γ=1)
```

---

## 4. Internal Data Structures {#4}

### 4.1 Node Hierarchy

The text hierarchy is represented as a rooted tree with typed nodes:

```python
@dataclass(frozen=True)
class HierarchyNode:
    level: int                # 1=sentence, 2=paragraph, 3=chapter,
                              # 4=narrative_layer, 5=discourse_layer
    index: int                # position within level
    parent: Optional[NodeID]  # parent node ID
    children: list[NodeID]    # child node IDs
    tau: float                # narrative position (sentences: τ_i; others: mean of children)
```

Tree invariants:
- Level 1 nodes (sentences) are leaves
- Level 5 (discourse layer) has exactly one node (root)
- Every node at level ℓ has a parent at level ℓ+1
- Child τ ranges are nested within parent τ ranges

### 4.2 Literary Signal Container

```python
@dataclass
class SignalStore:
    """Maps (NodeID, Dimension) → LiterarySignal for all nodes at all levels."""
    _data: dict[tuple[NodeID, str], LiterarySignal]

    def get(self, node: NodeID, dim: str) -> LiterarySignal: ...
    def set(self, node: NodeID, dim: str, signal: LiterarySignal) -> None: ...
    def get_level(self, level: int, dim: str) -> dict[NodeID, LiterarySignal]: ...
```

### 4.3 Coupling State

```python
@dataclass
class CouplingState:
    """Captures horizontal coupling at a single hierarchical level."""
    kappa: NDArray[float]      # κ_{d,d'}, shape (6, 6, N_nodes)
    eta: NDArray[float]        # η_i(τ), shape (6, N_nodes)
    components: dict[str, NDArray[float]]  # C_{sim}, C_{prox}, ..., shape (6,6,N_nodes)
```

### 4.4 Memory Store

```python
@dataclass
class MemoryEntry:
    key: NDArray[float]        # Fourier feature key, ℝ^{32}
    signal: LiterarySignal     # The stored signal
    beta: float                # Activation strength, [0,1]
    storage_time: float        # τ at storage
    retrieval_count: int       # Number of retrievals

class MemoryStore:
    _entries: list[MemoryEntry]

    def store(self, signal: LiterarySignal, key: NDArray[float]) -> None: ...
    def retrieve(self, query_key: NDArray[float]) -> list[MemoryEntry]: ...
    def decay(self, delta_t: int) -> None: ...
    def purge(self) -> None: ...
```

### 4.5 Resonance Field

```python
@dataclass
class ResonanceField:
    """Recursive resonance computed at m=100 sample points."""
    tau_points: NDArray[float]      # τ values, shape (m,)
    R_values: list[NDArray[float]]  # R(τ_i) at each point, shape (m,) × ℝ^d
    gamma_res: float = 0.2          # η_res coupling strength
    lambda_decay: float = 0.3       # decay length

    def current(self, tau: float) -> NDArray[float]: ...
    def summary(self) -> dict[str, float]: ...
```

### 4.6 Uncertainty State

```python
@dataclass
class UncertaintyState:
    """Per-node, per-dimension uncertainty."""
    sigma_sq: dict[NodeID, dict[str, float]]  # σ² at each node
    gamma: dict[NodeID, dict[str, float]]     # γ at each node
    sources: dict[str, float]                 # σ_e², σ_s², σ_i², σ_t², σ_m²
```

### 4.7 Explanation Trace

```python
@dataclass
class ExplanationTrace:
    signal_lineage: list[LineageEntry]       # provenance-weighted contributions
    propagation_path: list[NodeID]           # max product path
    interaction_history: list[Interaction]   # strongest κ_{d,d'} pairs
    uncertainty_evolution: list[UncertaintyAtLevel]
    resonance_contribution: float            # ||η_res·R|| / ||H||
    memory_activations: list[MemoryActivation]

@dataclass
class LineageEntry:
    node: NodeID
    dimension: str
    contribution: float
    path: list[NodeID]

@dataclass
class Interaction:
    dim_pair: tuple[str, str]
    kappa: float
    tau: float
```

---

## 5. Interfaces {#5}

### 5.1 Module Interfaces (Protocol Classes)

```python
class FeatureExtractor(Protocol):
    """Extract features for one dimension from a single sentence."""
    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[NDArray[float], float]:
        ...

class Propagator(Protocol):
    """Propagate signals up the hierarchy."""
    def aggregate_sentence_to_paragraph(
        self, signals: list[LiterarySignal]) -> LiterarySignal: ...      # Eq. (10.2)–(10.4)
    def aggregate_paragraph_to_chapter(
        self, signals: list[LiterarySignal], beta_sal: float) -> tuple[LiterarySignal, NDArray[float]]: ...  # Eq. (10.5)–(10.6)
    def interpolate_to_narrative(
        self, chapter_signals: list[tuple[float, LiterarySignal]], tau: NDArray[float]) -> LiterarySignal: ...  # Eq. (10.7)
    def project_to_discourse(
        self, narrative_signals: dict[str, LiterarySignal], W: dict[str, NDArray[float]]) -> NDArray[float]: ...  # Eq. (10.8)
    def refine_top_down(
        self, child: LiterarySignal, parent: LiterarySignal, beta_down: float) -> LiterarySignal: ...  # Eq. (10.9)

class Coupler(Protocol):
    """Compute and apply cross-dimensional coupling."""
    def compute_adaptive_weights(self, tau: float, params: CouplingParams) -> NDArray[float]: ...  # Eq. (11.1)–(11.2)
    def compute_kernel_components(self, signals: dict[str, LiterarySignal],
                                   tau: float) -> dict[str, float]: ...  # Eq. (11.3)–(11.8)
    def compute_kernel(self, components: dict[str, float],
                        weights: NDArray[float]) -> float: ...  # Eq. (11.9)
    def apply_coupling(self, signals: dict[str, LiterarySignal],
                        kappa: NDArray[float], M: NDArray[float]) -> dict[str, LiterarySignal]: ...  # Eq. (11.10)–(11.11)

class MemorySubsystem(Protocol):
    """Memory storage, retrieval, reinforcement, and decay."""
    def make_key(self, f: NDArray[float]) -> NDArray[float]: ...  # Eq. (12.2)
    def should_store(self, signal: LiterarySignal) -> bool: ...   # Eq. (12.1)
    def retrieve(self, key: NDArray[float]) -> list[MemoryEntry]: ...  # Eq. (12.3)
    def reinforce(self, entry: MemoryEntry, similarity: float) -> None: ...  # Eq. (12.4)
    def decay_all(self, delta_t: int) -> None: ...                 # Eq. (12.5)
    def purge(self) -> None: ...                                   # Eq. (12.6)
    def augment(self, signal: LiterarySignal,
                 retrieved: list[MemoryEntry]) -> LiterarySignal: ...  # Eq. (12.7)

class ResonanceEngine(Protocol):
    """Compute and apply recursive resonance."""
    def update(self, R_prev: NDArray[float], H_current: NDArray[float],
                delta_tau: float, lam: float) -> NDArray[float]: ...  # Eq. (12.8)
    def augment(self, signal: LiterarySignal,
                 resonance: NDArray[float], eta_res: float) -> LiterarySignal: ...  # Eq. (12.9)

class UncertaintyEngine(Protocol):
    """Propagate uncertainty through the hierarchy."""
    def total_variance(self, sources: dict[str, float]) -> float: ...  # Eq. (13.1)
    def confidence(self, sigma_total: float, sigma_max: float) -> float: ...  # Eq. (13.2)
    def propagate_to_paragraph(self, children: list[float],
                                 coherence: float) -> tuple[float, float]: ...  # Eq. (13.3)
    def propagate_to_chapter(self, paragraph_vars: list[float],
                               alphas: NDArray[float]) -> float: ...  # Eq. (13.4)
    def propagate_to_narrative(self, chapter_vars: list[float],
                                 phis: NDArray[float], taus: NDArray[float],
                                 tau: float) -> float: ...  # Eq. (13.5)
    def coupling_variance(self, kappa: float,
                            var_d: float, var_dp: float) -> float: ...  # Eq. (13.6)

class LDRICalculator(Protocol):
    """Compute LDRI from the Narrative Embedding and auxiliary data."""
    def compute_activations(self, H: dict[str, NDArray[float]],
                              sigma: dict[str, float]) -> NDArray[float]: ...  # LDRI Eq. (2)
    def signal_richness(self, activations: NDArray[float]) -> float: ...  # LDRI Eq. (3)–(4)
    def coupling_factor(self, H_pairs: NDArray[float]) -> float: ...  # LDRI Eq. (5), (9)
    def resonance_factor(self, H_res: NDArray[float]) -> float: ...  # LDRI Eq. (6), (9)
    def memory_factor(self, memory_store: MemoryStore) -> float: ...  # LDRI Eq. (7), (9)
    def global_confidence(self, gammas: dict[str, float]) -> float: ...  # LDRI Eq. (11)–(12)
    def ldri(self, R_signal: float, C_tilde: float, T_tilde: float,
              M_tilde: float, Gamma: float,
              alpha: float, beta: float, gamma: float, delta: float) -> float: ...  # LDRI Eq. (8), (10)

class Decomposer(Protocol):
    """Decompose LDRI into narratological components."""
    def intrinsic(self, ...) -> float: ...    # LDRI Eq. (16)
    def interaction(self, ...) -> float: ...   # LDRI Eq. (17)
    def temporal(self, ...) -> float: ...      # LDRI Eq. (18)
    def resonance(self, ...) -> float: ...     # LDRI Eq. (19)
    def memory(self, ...) -> float: ...        # LDRI Eq. (20)
    def uncertainty_adj(self, ...) -> float: ...  # LDRI Eq. (21)–(22)
    def fractions(self, ...) -> dict[str, float]: ...  # LDRI Eq. (23)
```

### 5.2 HMP-DT Engine Interface

```python
class HMPDTEngine:
    """Orchestrates the K-iteration refinement loop."""
    def __init__(self, config: HMPDTConfig,
                 propagator: Propagator,
                 coupler: Coupler,
                 memory: MemorySubsystem,
                 resonance: ResonanceEngine,
                 uncertainty: UncertaintyEngine):
        ...

    def run(self, initial_signals: SignalStore,
            hierarchy: HierarchyTree) -> tuple[SignalStore, MemoryStore, ResonanceField, UncertaintyState]:
        """Execute K iterations of bottom-up propagation with horizontal coupling,
           memory, resonance, and top-down refinement.
           
           Returns the converged state.
        """
```

### 5.3 Pipeline Interface

```python
class LITDISCPipeline:
    """End-to-end LIT-DISC pipeline."""
    def __init__(self, config_path: str):
        ...

    def process(self, text: str) -> LITDISCResult:
        """Process a single literary text end-to-end.
        
        Returns:
            LITDISCResult containing NarrativeEmbedding, LDRI score,
            ExplanationTrace, and intermediate states.
        """

@dataclass(frozen=True)
class LITDISCResult:
    text_id: str
    embedding: NarrativeEmbedding
    ldri: LDRIScore
    explanation: ExplanationTrace
    metadata: ProcessingMetadata
```

---

## 6. Storage Formats {#6}

### 6.1 Input Formats

| Format | Loader | Notes |
|--------|--------|-------|
| Plain text (.txt) | Built-in | UTF-8 encoded |
| EPUB (.epub) | epub library | Extract text content |
| TEI-XML (.xml) | xml.etree | Extract `<text>` body |
| JSON Lines (.jsonl) | Built-in | Pre-tokenized input for testing |

### 6.2 Intermediate Storage (Disk Cache)

All intermediate outputs use NumPy `.npz` files (compressed) for efficient storage and loading:

| File | Contents | Shape |
|------|----------|-------|
| `features_[dim].npz` | $\mathbf{f}$ and $\gamma$ for each node | Variable |
| `signals_level_[l].npz` | LiterarySignal components at level $\ell$ | (N_ℓ, k_d) |
| `coupling_level_[l].npz` | $\kappa_{d,d'}$, $\eta_i$, components | (6, 6) or (6,) |
| `resonance.npz` | $\mathcal{R}(\tau_i)$ for $m=100$ points | (100, d) |
| `memory.npz` | Memory entries with keys and activations | (\|M\|, 32 + 1) |
| `uncertainty.npz` | $\sigma^2$ and $\gamma$ at each level | Variable |
| `embedding.npz` | $\mathbf{H}$ with subspace labels | (120,) |
| `ldri_score.npz` | LDRI, $\Gamma$, reliability, decomposition | Scalar + vector |

### 6.3 Output Formats

| Format | Contents | Use Case |
|--------|----------|----------|
| JSON | Full LITDISCResult | Downstream analysis |
| CSV | LDRI scores + decomposition (one row per text) | Batch analysis |
| LaTeX `.tex` | Formatted tables (decomposition, explanation trace) | Publication |
| Markdown `.md` | Human-readable report | Documentation |
| SVG/PNG | Visualizations | Publication |

### 6.4 JSON Output Schema (Conceptual)

```json
{
  "text_id": "hemingway_hills_like_white_elephants",
  "ldri": {
    "score": 0.68,
    "confidence": 0.87,
    "reliability": 0.92,
    "raw": 0.78,
    "components": {
      "intrinsic": 0.26,
      "interaction": 0.11,
      "temporal": 0.09,
      "resonance": 0.05,
      "memory": 0.08,
      "uncertainty_adjustment": -0.11
    },
    "fractions": {
      "intrinsic": 0.33,
      "interaction": 0.14,
      "temporal": 0.12,
      "resonance": 0.06,
      "memory": 0.10
    }
  },
  "embedding": {
    "path": "output/hemingway_hills/embedding.npz"
  },
  "explanation": {
    "lineage": [
      {"node": [3, 6], "dim": "Em", "contribution": 0.31, "path": ["S45", "P6", "C1", "NL", "DL"]}
    ],
    "strongest_interaction": {"dims": ["Em", "Sy"], "kappa": 0.68, "tau": 0.60},
    "resonance_contribution": 0.12
  }
}
```

---

## 7. Configuration System {#7}

### 7.1 Configuration Hierarchy

```yaml
# config/default.yaml
hmp_dt:
  # Algorithm parameters (HMP-DT §15, algorithm config)
  refinement_iterations: 5          # K
  beta_salience: 2.0                # β_{sal} (Eq. 10.5)
  beta_topdown: 0.3                 # β_↓ (Eq. 10.9)
  epsilon_convergence: 1e-6         # Convergence threshold
  narrative_samples: 100            # m (Eq. 12.8)

  # Coupling parameters (HMP-DT §11.2)
  coupling:
    sigma_tau: 0.1                  # σ_τ (Eq. 11.4)
    components:
      C_sim: {mu: 0.15, omega: 0.30, alpha: 0.10, beta_i: 0.20}
      C_prox: {mu: 0.55, omega: 0.50, alpha: 0.10, beta_i: 0.20}
      C_conf: {mu: 0.50, omega: 1.00, alpha: 0.20, beta_i: 0.10}
      C_comp: {mu: 0.10, omega: 0.20, alpha: 0.05, beta_i: 0.15}
      C_sym:  {mu: 0.60, omega: 0.40, alpha: 0.05, beta_i: 0.25}
      C_rel:  {mu: 0.85, omega: 0.30, alpha: 0.05, beta_i: 0.20}

  # Memory parameters (HMP-DT §12)
  memory:
    key_dim: 32                     # k_{key} (Eq. 12.2)
    sigma_kernel: 1.0               # σ for Fourier features
    seed_W: 0                       # Fixed seed for W
    seed_b: 1                       # Fixed seed for b
    theta_store: 0.5                # θ_store (Eq. 12.1)
    theta_retrieve: 0.7             # θ_retrieve (Eq. 12.3)
    theta_forget: 0.01              # θ_forget (Eq. 12.6)
    rho_reinforce: 0.1              # ρ (Eq. 12.4)
    delta_decay: 0.05               # δ (Eq. 12.5)

  # Resonance parameters (HMP-DT §12.6–12.7)
  resonance:
    lambda_decay: 0.3               # λ (Eq. 12.8)
    eta_resonance: 0.2              # η_res (Eq. 12.9)

  # Embedding parameters (HMP-DT §14.1)
  embedding:
    dimension: 120                  # d

# LDRI parameters (LDRI Eq. 10)
ldri:
  exponents:
    alpha: 2                        # Signal richness exponent
    beta: 1                         # Coupling exponent
    gamma: 1                        # Resonance exponent
    delta: 1                        # Memory exponent
  sigma_max: 0.1                    # σ_max for reliability (LDRI Eq. 14)
  bootstrap_samples: 100            # B (LDRI Eq. 13)

# Non-Western parameter sets (HMP-DT §8)
alternate_parameter_sets:
  kishotenketsu:
    coupling:
      C_sim: {mu: 0.10, omega: 0.30, alpha: 0.10, beta_i: 0.20}
      C_prox: {mu: 0.50, omega: 0.50, alpha: 0.10, beta_i: 0.20}
      C_sym: {mu: 0.75, omega: 0.40, alpha: 0.05, beta_i: 0.25}
      C_rel: {mu: 0.90, omega: 0.30, alpha: 0.05, beta_i: 0.20}
```

### 7.2 Parameter Validation Schema

Every parameter in the configuration system must pass a validation check:

| Parameter | Type | Range | Validation |
|-----------|------|-------|------------|
| `refinement_iterations` | int | [1, 20] | $K \geq 1$ |
| `beta_topdown` | float | (0, 0.5) | Must satisfy $L < 1$ (HMP-DT A7) |
| `mu_i` | float | [0, 1] | Phase boundary within narrative |
| `omega_i` | float | (0, 1] | Window width positive |
| `key_dim` | int | {8, 16, 32, 64} | Power of 2 |
| `dimension` | int | {40, 120, 240, 480} | HMP-DT §14.1 recommended values |
| `exponents.*` | float | [0.5, 4.0] | Default $\alpha = 2$ |

---

## 8. Explainability Storage {#8}

### 8.1 Trace Structure

The explanation trace is the primary output for human interpretability. It is stored as a structured JSON object with six root-level keys, each mapping to a specific HMP-DT algorithm step:

```json
{
  "signal_lineage": [
    {
      "node_id": [3, 6],
      "dimension": "Em",
      "contribution": 0.31,
      "provenance_path": ["S45", "P6", "C1", "NL", "DL"],
      "confidence_at_source": 0.78,
      "confidence_at_root": 0.90
    }
  ],
  "propagation_path": {
    "nodes": ["S45", "P6", "C1", "NL", "DL"],
    "total_influence": 0.31
  },
  "interaction_history": {
    "all_pairs": [
      {"dims": ["Em", "Sy"], "kappa": 0.68, "tau": 0.60},
      {"dims": ["Em", "N"], "kappa": 0.71, "tau": 0.72}
    ],
    "top_pair": {"dims": ["Em", "N"], "kappa": 0.71, "tau": 0.72}
  },
  "uncertainty_evolution": [
    {"level": 1, "sigma_sq_mean": 0.15, "gamma_mean": 0.85},
    {"level": 2, "sigma_sq_mean": 0.04, "gamma_mean": 0.92},
    {"level": 3, "sigma_sq_mean": 0.06, "gamma_mean": 0.88},
    {"level": 4, "sigma_sq_mean": 0.08, "gamma_mean": 0.86},
    {"level": 5, "sigma_sq_mean": 0.10, "gamma_mean": 0.90}
  ],
  "resonance": {
    "contribution_fraction": 0.12,
    "decay_length": 0.3,
    "initial_strength": 0.42,
    "final_strength": 0.05
  },
  "memory_activations": [
    {
      "key_hash": "a1b2c3d4",
      "activation": 0.93,
      "stored_at_tau": 0.56,
      "retrieved_at_tau": 0.84,
      "content_summary": "simple operation euphemism"
    }
  ]
}
```

### 8.2 Storage Format

| Level | Format | Location |
|-------|--------|----------|
| In-memory | `ExplanationTrace` dataclass | During pipeline execution |
| Per-text persistent | JSON | `output/{text_id}/explanation.json` |
| Batch (multi-text) | NDJSON | `output/batch/explanations.jsonl` |

---

## 9. Memory Subsystem {#9}

### 9.1 Data Structures

```
MemoryStore
├── entries: list[MemoryEntry]
│   ├── key: NDArray[32]         # Fourier feature key (Eq. 12.2)
│   ├── signal: LiterarySignal   # Original signal (for augmentation, Eq. 12.7)
│   ├── beta: float              # Activation [0,1] (Eq. 12.4–12.5)
│   ├── storage_tau: float       # τ at storage time
│   ├── retrieval_count: int     # Number of successful retrievals
│   └── last_retrieved: float    # τ of most recent retrieval
│
├── config:
│   ├── theta_store: 0.5         # Storage threshold (Eq. 12.1)
│   ├── theta_retrieve: 0.7      # Retrieval threshold (Eq. 12.3)
│   ├── theta_forget: 0.01       # Purging threshold (Eq. 12.6)
│   ├── rho: 0.1                 # Reinforcement rate (Eq. 12.4)
│   ├── delta: 0.05              # Decay rate (Eq. 12.5)
│   ├── k_key: 32                # Key dimension (Eq. 12.2)
│   └── sigma: 1.0               # Kernel length scale
│
└── fourier_params:
    ├── W: NDArray[32, k_input]   # Random projection matrix (fixed seed 0)
    └── b: NDArray[32]            # Random bias vector (fixed seed 1)
```

### 9.2 Memory Operations Lifecycle

```
For each node i, dimension d, at each hierarchy level:

1. [Key Computation]
   k_i = sqrt(2/k_key) * cos(W @ f_i[d] + b)

2. [Store Decision]
   if ||f_i[d]|| * gamma_i[d] > theta_store → ENQUEUE for storage

3. [Retrieval]
   for each entry j in M:
       if cos(k_i, k_j) > theta_retrieve → ACTIVATE(j)
   
4. [Signal Augmentation]
   f_aug = f_i[d] + sum(activated_entries: beta_j * W_mem @ f_j[d])

5. [Reinforcement]   (per retrieval)
   beta_j = min(1.0, beta_j + rho * cos(k_i, k_j))

6. [Decay]           (per iteration)
   for each entry j in M:
       if not retrieved this iteration:
           beta_j *= (1 - delta)
   
7. [Purging]
   M = {j in M: beta_j > theta_forget}

8. [Storage]         (at end of iteration)
   for each enqueued signal:
       M.add(MemoryEntry(key=k_i, signal=S_i[d], beta=1.0, ...))
```

### 9.3 Complexity Bounds

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Key computation | $O(k_{\text{key}} \cdot k_{\text{input}})$ | ~$32 \cdot 12 = 384$ ops |
| Retrieval (linear scan) | $O(|\mathcal{M}| \cdot k_{\text{key}})$ | $200 \cdot 32 = 6,400$ ops max |
| Reinforcement | $O(1)$ per retrieved entry | |
| Decay | $O(|\mathcal{M}|)$ per iteration | $200$ entries max |
| Storage limit | $|\mathcal{M}| \leq 200$ | Enforced; evict lowest $\beta$ if exceeded |

---

## 10. Computational Graph {#10}

### 10.1 Directed Acyclic Graph (DAG) of the HMP-DT Engine

```
Legend: [Module] → data flow | (Eq.X) → equation reference

[SignalInitializer] (Eq. 10.1)
    |
    v
┌─────────────────────────────────────────────────────────┐
│                   REFINEMENT LOOP (K times)              │
│                                                          │
│  ┌──────────┐                                            │
│  │ Memory   │── k, retrieved_signals ──────────────────┐ │
│  │ (Eq. 12) │── reinforcement/decay signal          │ │
│  └──────────┘                                            │ │
│                                                          │ │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │ │
│  │ Coupler  │──→│Coupling  │──→│ Augmented Signal  │◄───┘ │
│  │ (Eq.11.1)│   │Operator  │   │ (Eq. 11.11)      │      │
│  └──────────┘   └──────────┘   └──────────────────┘      │
│                                                          │
│  ┌───────────┐    ┌──────────────────┐                   │
│  │Propagator │───→│Aggregated Signal │                   │
│  │ (Eq.10.2) │    │ (Eq. 10.2–10.8) │                   │
│  │ (Eq.10.5) │    └──────────────────┘                   │
│  │ (Eq.10.7) │                                           │
│  └───────────┘                                           │
│                                                          │
│  ┌───────────┐    ┌──────────────────┐                   │
│  │ Resonance │───→│ Augmented Field  │                   │
│  │ (Eq.12.8) │    │ (Eq. 12.9)      │                   │
│  └───────────┘    └──────────────────┘                   │
│                                                          │
│  ┌─────────────┐  ┌──────────────────┐                   │
│  │ Uncertainty │─→│ Confidence & σ²  │                   │
│  │ (Eq.13.1–6) │  │ (Eq. 13.2)      │                   │
│  └─────────────┘  └──────────────────┘                   │
│                                                          │
│  ┌───────────┐    ┌──────────────────┐                   │
│  │ Refiner   │───→│ Refined Signal   │                   │
│  │ (Eq.10.9) │    │ (top-down)       │                   │
│  └───────────┘    └──────────────────┘                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
    |
    v (converged)
[NarrativeEmbedding] (Eq. 14.1)
    |
    v
[LDRICalculator] (LDRI Eq. 2–10)
    |
    v
[Decomposer] [ExplanationGenerator]
```

### 10.2 Data Dependencies

| Output | Depends On | Produced By |
|--------|-----------|-------------|
| $\mathbf{f}_{S_i}^{(d)}$ | Raw text + tokens | FeatureExtractor |
| $\mathcal{S}_{i}^{(\ell)}[d]$ | $\mathbf{f}$, $\gamma$, $\tau$ | SignalInitializer |
| $\eta_i(\tau)$ | $\tau$, $(\mu_i, \omega_i, \alpha_i, \beta_i)$ | Coupler (Eq. 11.1–11.2) |
| $\kappa_{d,d'}(\tau)$ | $\mathbf{f}_d$, $\mathbf{f}_{d'}$, $\eta_i$ | Coupler (Eq. 11.3–11.9) |
| $\mathbf{k}(\mathbf{f})$ | $\mathbf{f}$, $\mathbf{W}$, $\mathbf{b}$ | Memory (Eq. 12.2) |
| $\mathcal{R}(\tau_i)$ | $\mathcal{R}(\tau_{i-1})$, $\mathbf{H}(\tau_{i-1})$, $\lambda$ | Resonance (Eq. 12.8) |
| $\sigma^2(P)$ | $\{\sigma^2(S_i)\}$, SC$(P)$ | Uncertainty (Eq. 13.3) |
| $\mathbf{H}_d$ | $\{\mathbf{f}_N^{(d)}(\tau)\}$ | NarrativeEmbedding (Eq. 14.2) |
| $\mathcal{R}_{\text{signal}}$ | $\{\|\mathbf{H}_d\|\}$ | LDRICalculator (Eq. 3–4) |

---

## 11. Parallelization Opportunities {#11}

### 11.1 Task-Level Parallelism

| Task | Granularity | Parallelism Type | Speedup Bound |
|------|-------------|------------------|---------------|
| Feature extraction (6 dimensions) | Per-dimension | Embarrassingly parallel (6×) | 6× |
| Feature extraction (N sentences) | Per-sentence | Map-reduce | $\approx$ N_cores (dim $\times$ sentence) |
| Coupling components (6 components) | Per-component | Embarrassingly parallel (6×) | 6× |
| Coupling pairs (15 pairs) | Per $(d,d')$ pair | Embarrassingly parallel (15×) | 15× (overkill for 6 dims) |
| Bootstrap ($B$ repetitions) | Per repetition | Embarrassingly parallel ($B \times$) | $B$ (100×) |
| Memory retrieval (linear scan) | Per memory entry | Map-reduce | $\approx$ N_cores |
| Batch processing (N texts) | Per text | Embarrassingly parallel ($N \times$) | $N$ |

### 11.2 Pipeline-Level Parallelism

```
Text 1:  [Preproc] → [FeatExt] → [HMPDT K=5] → [Embed] → [LDRI]
Text 2:  [Preproc] → [FeatExt] → [HMPDT K=5] → [Embed] → [LDRI]
...
Text N:  [Preproc] → [FeatExt] → [HMPDT K=5] → [Embed] → [LDRI]
         ↑ Pipeline parallelism: stages 1–5 can run concurrently
         ↑ Batch parallelism: texts 1–N are independent
```

### 11.3 Bottleneck Analysis

| Stage | Bottleneck | Mitigation |
|-------|-----------|------------|
| Preprocessing | Sentence tokenization | spaCy multiprocessing |
| Feature extraction | NLP model inference | Batching; GPU if available |
| HMP-DT refinement loop | Sequential K iterations (not parallelizable) | Each iteration is $O(10^7)$ ops; acceptable |
| Memory retrieval | Linear scan ($|\mathcal{M}| \leq 200$) | Not a bottleneck at this scale |
| Bootstrap | $B$ repetitions (embarrassingly parallel) | $B$ cores |

---

## 12. Error Handling {#12}

### 12.1 Error Taxonomy

| Category | Error Type | Source | Recovery |
|----------|-----------|--------|----------|
| **Input** | `FileNotFoundError` | Loader | Raise with path; log file search locations |
| **Input** | `EncodingError` | Loader | Try charset detection; raise if all fail |
| **Parse** | `TokenizationError` | Preprocessor | Skip sentence; log warning; continue |
| **Parse** | `HierarchyError` | Preprocessor | Degenerate hierarchy (1 sentence = 1 paragraph = 1 chapter); log warning; process as flat |
| **Feature** | `FeatureExtractionError` | FeatureExtractor | Return zero vector with $\gamma = 0$; log |
| **Feature** | `MissingFeatureError` | FeatureExtractor | Fall back to reduced feature set; log |
| **Numerical** | `NumericalInstabilityError` | Propagator/Coupler | Clip to $[0,1]$ bounds; log |
| **Numerical** | `DivergenceError` | RefinementController | Stop after K iterations; flag as non-convergent |
| **Memory** | `MemoryCapacityExceeded` | Memory | Evict lowest-$\beta$ entries; log |
| **Config** | `ConfigValidationError` | Config | Raise at init with invalid parameter details |
| **Runtime** | `PipelineTimeoutError` | Orchestrator | Timeout per text (default 600s); skip and log |

### 12.2 Error Propagation Model

```
Pipeline.process(text)
    │
    ├── Loader.load(path)
    │       └── raises FileNotFoundError → caught → logged → re-raised as PipelineError
    │
    ├── Preprocessor.process(text)
    │       └── catches TokenizationError → log warning → returns partial parse
    │
    ├── FeatureExtractor.extract(sentence)
    │       └── catches FeatureExtractionError → returns f=0, γ=0
    │
    ├── HMPDTEngine.run(...)
    │       └── catches NumericalInstability → clip to bounds
    │       └── catches Divergence → flag S[k] as non-convergent
    │
    └── LDRICalculator.compute(...)
            └── catches ValueError → return LDRI=0, Γ=0
```

### 12.3 Graceful Degradation

| Failure Scenario | Degradation Mode | Impact on Output |
|-----------------|------------------|------------------|
| One feature extractor fails | $\gamma_d \to 0$, $\Gamma \to 0$ | LDRI penalized; trace flags missing dimension |
| Memory store corruption | Reset memory, $\mathcal{M} = 0$ | LDRI recalculated without memory contribution |
| Spline interpolation fails | Linear interpolation fallback | Marginal LDRI change [Proven: Lipschitz stable] |
| Coupling kernel overflow | Clip to $[0,1]$ | Bounded by Proposition 1 |

---

## 13. Logging {#13}

### 13.1 Log Levels

| Level | Use Case | Module |
|-------|----------|--------|
| `DEBUG` | Per-node feature values, intermediate $\kappa$ values | All modules (optional, configurable) |
| `INFO` | Pipeline stage boundaries, per-text LDRI, convergence status | Orchestrator |
| `WARNING` | Degraded hierarchy, missing features, memory eviction | Preprocessor, FeatureExtractor, Memory |
| `ERROR` | Module failure, invalid config, timeout | All modules |

### 13.2 Structured Log Format

```json
{
  "timestamp": "2026-07-26T14:30:00.123Z",
  "level": "INFO",
  "pipeline": {
    "text_id": "hemingway_hills",
    "stage": "HMPDT_ENGINE",
    "iteration": 3,
    "converged": false,
    "delta": 0.042
  },
  "module": "propagator",
  "event": "paragraph_aggregation",
  "data": {
    "paragraph_id": [2, 6],
    "num_sentences": 15,
    "coherence": 0.85,
    "result_gamma": 0.88
  }
}
```

### 13.3 Per-Text Log File

```
logs/
  run_20260726_v1/
    pipeline.log              (all texts, INFO+)
    debug/
      text_hemingway_hills.log  (DEBUG level for this text)
    errors/
      error_manifest.csv        (structured: text_id, stage, error_type, message)
```

---

## 14. Reproducibility {#14}

### 14.1 Determinism Guarantees

| Component | Deterministic? | Mechanism |
|-----------|---------------|-----------|
| Preprocessing (spaCy) | No (v3.x stochastic) | Fix seed, pin spaCy version |
| Feature extraction | Yes | No randomness in extractors |
| Fourier feature generation | Yes | Seed 0 for $\mathbf{W}$, seed 1 for $\mathbf{b}$ (HMP-DT §5.4) |
| Coupling operator | Yes | Fixed seed 42 for $\mathbf{M}_{d,d'}$ |
| Memory operations | Yes | Deterministic given fixed features |
| Resonance update | Yes | Deterministic recurrence |
| Uncertainty propagation | Yes | Closed-form equations |
| LDRI calculation | Yes | Closed-form equations |
| Explanation generation | Yes | Deterministic trace |
| Bootstrap | Yes per seed | Seed = 1000 + $b$ |

### 14.2 Reproducibility Artifact

```
reproducibility/
  Dockerfile
  environment.yaml
  requirements.txt            (pinned versions)
  config/
    seeds.yaml                 (all seeds as specified in §9.2 of Evaluation Methodology)
    default.yaml               (frozen HMP-DT parameters)
  tests/
    test_known_outputs/        (regression test data: known inputs → expected outputs)
  README.md
```

### 14.3 Version Tracking

| Artifact | Version Source |
|----------|---------------|
| HMP-DT specification | Git tag `hmp-dt-v1.0` |
| LDRI methodology | Git tag `ldri-v1.0` |
| Evaluation methodology | Git tag `eval-v1.0` |
| Software implementation | Git tag `impl-v1.0` |
| Experiment run | Git tag `exp-YYYY-MM-DD-v1` |
| Annotated dataset | MD5 checksum file |

### 14.4 Configuration Checksum

Every experiment run produces a `config_checksum.txt` containing the SHA-256 hash of the active configuration file. The configuration is embedded in the output JSON metadata:

```json
{
  "metadata": {
    "config_checksum": "a1b2c3d4e5f6...",
    "config_path": "experiments/run_20260726_v1/config/default.yaml",
    "git_commit": "abc123def456",
    "pipeline_version": "1.0.0"
  }
}
```

---

## 15. Testing Strategy {#15}

### 15.1 Test Pyramid

```
                  ┌──────────────────────┐
                  │   End-to-End Tests    │  Few: 1 per scenario
                  │  (5 scenarios × 3    │
                  │   text types)        │
                  └──────────────────────┘
                  ┌──────────────────────┐
                  │   Integration Tests   │  Some: 1 per module pair
                  │  (10 module pairs)    │
                  └──────────────────────┘
                  ┌──────────────────────┐
                  │     Unit Tests        │  Many: 1 per equation
                  │  (~60 equations)      │
                  └──────────────────────┘
                  ┌──────────────────────┐
                  │  Numerical Tests      │
                  │  (bounds, invariants, │
                  │   edge cases)        │
                  └──────────────────────┘
```

### 15.2 Unit Tests (Equation-Level)

Each numbered equation in HMP-DT and LDRI maps to at least one unit test:

| Module | Equation | Test Pattern |
|--------|----------|--------------|
| SignalInitializer | Eq. (10.1) | Given known $\mathbf{f}, \gamma, \tau$, verify 7-tuple construction |
| Propagator | Eq. (10.2)–(10.4) | Given known sentence signals, verify paragraph aggregation |
| Propagator | Eq. (10.5)–(10.6) | Given known paragraph signals, verify salience weighting |
| Propagator | Eq. (10.7) | Given known chapter signals, verify spline output at known $\tau$ |
| Propagator | Eq. (10.8) | Given known narrative signals, verify linear projection |
| Propagator | Eq. (10.9) | Given known parent/child, verify refinement direction |
| Coupler | Eq. (11.1)–(11.2) | Verify $\sum \eta_i = 1$, verify raised cosine shape |
| Coupler | Eq. (11.3)–(11.8) | Verify each component on known inputs |
| Coupler | Eq. (11.9)–(11.11) | Verify $\kappa \in [0,1]$, verify update direction |
| Memory | Eq. (12.1) | Verify storage threshold decision |
| Memory | Eq. (12.2) | Verify Fourier feature key has correct dimension |
| Memory | Eq. (12.3)–(12.7) | Verify retrieval, reinforcement, decay, augmentation |
| Resonance | Eq. (12.8)–(12.9) | Verify recursion matches integral for known $\mathbf{H}$ |
| Uncertainty | Eq. (13.1)–(13.6) | Verify $\gamma \in [0,1]$, verify propagation |
| NarrativeEmbedding | Eq. (14.1)–(14.2) | Verify subspace dimensions and concatenation |
| LDRICalculator | Eq. (2)–(10) | Verify $a_d \in [0,1)$, $\mathcal{R}_{\text{signal}} \in [0,1]$, LDRI $\in [0,1]$ |
| Decomposer | Eq. (16)–(23) | Verify additive decomposition approximates LDRI |

### 15.3 Integration Tests

| Test | Modules Involved | Expected Behavior |
|------|-----------------|-------------------|
| Feature extraction → Signal initialization | FeatureExtractor × 6, SignalInitializer | 6 LiterarySignals per sentence with correct dimensions |
| Memory → Coupler | Memory, Coupler | Memory-augmented signals produce different $\kappa$ |
| Propagator → Uncertainty | Propagator, Uncertainty | $\gamma$ decreases with each hierarchical level |
| K=1 full loop | All HMP-DT modules | Converged state satisfies all bounds |
| Embedding → LDRI | NarrativeEmbedding, LDRICalculator | LDRI ∈ [0,1] for any valid embedding |
| Full pipeline → Explanation | All modules | Explanation trace has 6 complete sections |

### 15.4 Numerical Tests

| Property | Test |
|----------|------|
| Boundedness (Axiom 1, Prop 1) | On 100 random embeddings, verify $0 \leq \text{LDRI} \leq 1$ |
| Monotonicity (Axiom 2, Prop 2) | Perturb each $\|\mathbf{H}_d\|$ upward; verify LDRI non-decreasing |
| Lipschitz stability (Axiom 4, Prop 4) | Add $\epsilon \mathbf{u}$; verify $|\Delta\text{LDRI}| \leq L\epsilon$ |
| Scale invariance (Axiom 7, Prop 5) | Verify LDRI unchanged under $c \cdot \mathbf{H}$ |
| Zero-signal case | All $\mathbf{H}_d = \mathbf{0}$: verify LDRI = 0 |
| Maximum-signal case | All $a_d = 1$: verify LDRI $\approx \Gamma$ |
| Coupling-off case | $\mathbf{H}_{\text{pairs}} = \mathbf{0}$: verify $\tilde{\mathcal{C}} = 0.5$ |
| Memory-off case | $\mathcal{M} = 0$: verify $\tilde{\mathcal{M}} = 0.5$ |

### 15.5 Known-Output Regression Tests

| Test Case | Text | Expected LDRI | Source |
|-----------|------|---------------|--------|
| Simple chapter | Synthetic | 0.42 | LDRI §8.1 worked example |
| Complex chapter | Synthetic | 0.70 | LDRI §8.2 worked example |
| *Hills Like White Elephants* | Hemingway | 0.68 | HMP-DT §7.10 worked example |
| Zero-richness text | Random tokens | $\approx 0$ | Theoretical minimum |
| Full-richness text | Artificial (all dimensions max) | $\approx \Gamma$ | Theoretical maximum |

### 15.6 Testing Infrastructure

| Requirement | Specification |
|-------------|---------------|
| Framework | pytest |
| Coverage target | $\geq 90\%$ statement coverage |
| CI | GitHub Actions (Ubuntu, macOS, Windows) |
| Test data | Synthetic texts with known HMP-DT outputs |
| Performance regression | Per-commit benchmark via `pytest-benchmark` |
| Numerical precision | `pytest.approx` with rel=1e-5 tolerance |

---

## Appendix A: Equation-to-Module Mapping

| Equation | Module | Method |
|----------|--------|--------|
| HMP-DT (10.1) | SignalInitializer | `initialize_sentence_signal()` |
| HMP-DT (10.2)–(10.4) | Propagator | `aggregate_sentence_to_paragraph()` |
| HMP-DT (10.5)–(10.6) | Propagator | `aggregate_paragraph_to_chapter()` |
| HMP-DT (10.7) | Propagator | `interpolate_to_narrative()` |
| HMP-DT (10.8) | Propagator | `project_to_discourse()` |
| HMP-DT (10.9) | Propagator | `refine_top_down()` |
| HMP-DT (11.1)–(11.2) | Coupler | `compute_adaptive_weights()` |
| HMP-DT (11.3)–(11.8) | Coupler | `compute_kernel_components()` |
| HMP-DT (11.9) | Coupler | `compute_kernel()` |
| HMP-DT (11.10)–(11.11) | Coupler | `apply_coupling()` |
| HMP-DT (12.1) | Memory | `should_store()` |
| HMP-DT (12.2) | Memory | `make_key()` |
| HMP-DT (12.3) | Memory | `retrieve()` |
| HMP-DT (12.4) | Memory | `reinforce()` |
| HMP-DT (12.5) | Memory | `decay_all()` |
| HMP-DT (12.6) | Memory | `purge()` |
| HMP-DT (12.7) | Memory | `augment()` |
| HMP-DT (12.8) | Resonance | `update()` |
| HMP-DT (12.9) | Resonance | `augment()` |
| HMP-DT (13.1) | Uncertainty | `total_variance()` |
| HMP-DT (13.2) | Uncertainty | `confidence()` |
| HMP-DT (13.3) | Uncertainty | `propagate_to_paragraph()` |
| HMP-DT (13.4) | Uncertainty | `propagate_to_chapter()` |
| HMP-DT (13.5) | Uncertainty | `propagate_to_narrative()` |
| HMP-DT (13.6) | Uncertainty | `coupling_variance()` |
| HMP-DT (14.1) | NarrativeEmbedding | `construct()` |
| HMP-DT (14.2) | NarrativeEmbedding | `pool_features()` |
| LDRI (2) | LDRICalculator | `compute_activations()` |
| LDRI (3)–(4) | LDRICalculator | `signal_richness()` |
| LDRI (5), (9) | LDRICalculator | `coupling_factor()` |
| LDRI (6), (9) | LDRICalculator | `resonance_factor()` |
| LDRI (7), (9) | LDRICalculator | `memory_factor()` |
| LDRI (8), (10) | LDRICalculator | `ldri()` |
| LDRI (11)–(12) | LDRICalculator | `global_confidence()` |
| LDRI (16) | Decomposer | `intrinsic()` |
| LDRI (17) | Decomposer | `interaction()` |
| LDRI (18) | Decomposer | `temporal()` |
| LDRI (19) | Decomposer | `resonance()` |
| LDRI (20) | Decomposer | `memory()` |
| LDRI (21)–(22) | Decomposer | `uncertainty_adj()` |
| LDRI (23) | Decomposer | `fractions()` |

---

## Appendix B: File Manifest

```
lit-disc/
├── modules/
│   ├── __init__.py
│   ├── config.py              (Configuration manager)
│   ├── preprocessor.py        (Tokenization, hierarchy detection)
│   ├── narrative_position.py  (τ computation)
│   ├── signal_initializer.py  (Eq. 10.1)
│   ├── propagator.py          (Eq. 10.2–10.9)
│   ├── coupler.py             (Eq. 11.1–11.11)
│   ├── memory.py              (Eq. 12.1–12.7)
│   ├── resonance.py           (Eq. 12.8–12.9)
│   ├── uncertainty.py         (Eq. 13.1–13.6)
│   ├── embedding.py           (Eq. 14.1–14.2)
│   ├── ldri_calculator.py     (LDRI Eq. 2–10, 11–12)
│   ├── decomposer.py          (LDRI Eq. 16–23)
│   ├── explanation_generator.py
│   ├── refinement_controller.py
│   ├── report_generator.py
│   ├── visualizer.py
│   └── extractors/
│       ├── __init__.py
│       ├── semantic.py
│       ├── narrative.py
│       ├── interaction.py
│       ├── symbolic.py
│       ├── evolution.py
│       └── emotion.py
├── pipeline.py                (LITDISCPipeline orchestrator)
├── models.py                  (LiterarySignal, NarrativeEmbedding, etc.)
├── interfaces.py              (Protocol classes)
├── errors.py                  (Error taxonomy)
└── tests/
    ├── unit/
    │   ├── test_propagator.py
    │   ├── test_coupler.py
    │   ├── test_memory.py
    │   ├── test_resonance.py
    │   ├── test_uncertainty.py
    │   ├── test_embedding.py
    │   └── test_ldri.py
    ├── integration/
    │   ├── test_feature_to_signal.py
    │   ├── test_memory_to_coupler.py
    │   ├── test_propagator_uncertainty.py
    │   ├── test_full_loop.py
    │   └── test_embedding_to_ldri.py
    ├── numerical/
    │   ├── test_boundedness.py
    │   ├── test_monotonicity.py
    │   ├── test_stability.py
    │   └── test_edge_cases.py
    ├── regression/
    │   ├── test_simple_chapter.py
    │   ├── test_complex_chapter.py
    │   └── test_hemingway_example.py
    ├── fixtures/
    │   ├── simple_chapter.json
    │   ├── complex_chapter.json
    │   └── hemingway_expected.json
    └── conftest.py
```

---

*End of Software Architecture Specification — LIT-DISC v1.0*
