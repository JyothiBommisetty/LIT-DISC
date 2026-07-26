# LIT-DISC Implementation Roadmap

## Engineering Plan for HMP-DT and LDRI

**Document Type:** Implementation Roadmap  
**Framework:** LIT-DISC v1.0  
**Prerequisites:** All theoretical documents frozen (HMP-DT, LDRI, Evaluation Methodology, Architecture Specification)  
**Timeline:** 3–6 months (single developer) / 2–3 months (two developers)  

---

## Architecture Dependency Graph

The implementation order is strictly constrained by module dependencies:

```
Phase 1  ─── Infrastructure (config, models, interfaces, errors, logging)
   │
   v
Phase 2  ─── NLP Preprocessing (loader, tokenizer, hierarchy, τ)
   │
   v
Phase 3  ─── Feature Extraction (6 dimensions, independent sub-tasks)
   │
   v
Phase 4  ─── Signal Initialization (LiterarySignal construction)
   │
   v
Phase 5  ─── HMP-DT Propagation (propagator, coupler, refinement loop)
   │
   ├────────────────┬────────────────┬────────────────┐
   v                v                v                v
Phase 6         Phase 7          Phase 8          Phase 11
Memory          Resonance        Uncertainty      Explainability
   │                │                │                │
   └────────────────┴────────────────┴────────────────┘
                               │
                               v
                    Phase 9 ─── Narrative Embedding
                               │
                               v
                    Phase 10 ── LDRI Calculation
                               │
                               v
                    Phase 12 ── Report Generation & Visualization
```

---

## Phase 1 — Infrastructure (Week 1)

### Goal
Establish the foundational software skeleton: data models, configuration system, error taxonomy, logging infrastructure, and module interfaces. All subsequent phases depend on this phase.

### Modules
| Module | Files | Description |
|--------|-------|-------------|
| Models | `lit-disc/models.py` | All frozen dataclasses: `LiterarySignal`, `NarrativeEmbedding`, `LDRIScore`, `ExplanationTrace`, `HierarchyNode`, `MemoryEntry`, `ResonanceField`, `UncertaintyState`, `SignalStore`, `CouplingState`, `LITDISCResult` |
| Interfaces | `lit-disc/interfaces.py` | Protocol classes for all 17 module interfaces: `FeatureExtractor`, `Propagator`, `Coupler`, `MemorySubsystem`, `ResonanceEngine`, `UncertaintyEngine`, `LDRICalculator`, `Decomposer`, `ExplanationGenerator`, etc. |
| Config | `lit-disc/modules/config.py` | `HMPDTConfig`, `LDRIConfig` frozen dataclasses; YAML loader; parameter validation; alternate parameter sets (kishōtenketsu) |
| Errors | `lit-disc/errors.py` | Error taxonomy: `LITDISCError`, `InputError`, `ParseError`, `FeatureError`, `NumericalError`, `MemoryError`, `ConfigError`, `PipelineError`; graceful degradation strategy |
| Logging | `lit-disc/logging.py` | Structured JSON logger; 4 levels; per-text log files; error manifest |

### Dependencies
None (foundational).

### Expected Inputs
- `config/default.yaml` (architecture specification §7)
- `config/seeds.yaml` (evaluation methodology §9.2)

### Expected Outputs
- `lit-disc/models.py` — all dataclasses, importable by all other modules
- `lit-disc/interfaces.py` — all Protocol classes
- `lit-disc/modules/config.py` — `ConfigManager` class with `load()`, `validate()`, `get()`
- `lit-disc/errors.py` — error hierarchy
- `lit-disc/logging.py` — `LITDISCLogger` class

### Estimated Complexity
**3–5 person-days.** Pure data definition; no algorithmic work.

### Unit Tests
| Test | What It Verifies |
|------|------------------|
| `test_models_literary_signal.py` | 7-tuple construction, field types, frozen=True |
| `test_models_narrative_embedding.py` | Subspace dimensions sum to $d$, field access |
| `test_models_ldri_score.py` | Decomposition dict keys match spec |
| `test_config_loading.py` | YAML loads correctly with defaults |
| `test_config_validation.py` | Invalid parameters raise `ConfigError` |
| `test_config_kishotenketsu.py` | Alternate parameter set loads correctly |
| `test_errors_hierarchy.py` | All exception types chain to `LITDISCError` |
| `test_logging_structured.py` | JSON output format matches schema |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_config_to_models.py` | Config values correctly populate model defaults |

### Completion Criteria
- [ ] All frozen dataclasses defined with full type hints
- [ ] All Protocol interfaces defined with complete method signatures
- [ ] Config loader passes default.yaml with validation
- [ ] All 11 error types defined and chain to base
- [ ] Logger produces valid JSON output
- [ ] `pytest tests/unit/test_models*.py` — 100% pass
- [ ] `pytest tests/unit/test_config*.py` — 100% pass
- [ ] `pytest tests/unit/test_errors*.py` — 100% pass

---

## Phase 2 — NLP Preprocessing (Week 2)

### Goal
Implement text loading, sentence segmentation, paragraph/chapter detection, and narrative position computation. This phase transforms raw text into the hierarchical node structure that all downstream phases consume.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| Loader | `lit-disc/modules/loader.py` | — |
| Preprocessor | `lit-disc/modules/preprocessor.py` | HMP-DT §10.1–10.3 |
| NarrativePosition | `lit-disc/modules/narrative_position.py` | HMP-DT §15 algorithm step 1, Eq. (10.7) |

### Dependencies
Phase 1 (models, interfaces, config).

### Expected Inputs
- Raw text string (UTF-8)
- Config: sentence model (spaCy), tokenization settings

### Expected Outputs
- `HierarchyTree` — rooted tree with typed nodes (ℓ=1..5)
- `τ: NDArray[float, N_sentences]` — narrative positions for each sentence
- `τ_sample: NDArray[float, m=100]` — interpolated positions for spline

### Person-Days per Sub-Task
| Task | Days | Description |
|------|------|-------------|
| Loader | 0.5 | txt/epub/tei-xml readers |
| Sentence segmentation | 1.0 | spaCy/stanza integration |
| Paragraph detection | 0.5 | Blank-line and indentation heuristics |
| Chapter detection | 1.0 | Heading pattern regex, numbered chapter detection |
| τ computation | 1.0 | Verb-density velocity, cumulative integration, normalization |
| Spline base class | 1.0 | scipy.interpolate.CubicSpline wrapper |
| **Total** | **5.0** | |

### Estimated Complexity
**5–7 person-days.** Mostly NLP library integration.

### Unit Tests
| Test | What It Verifies |
|------|------------------|
| `test_loader_txt.py` | Plain text loads correctly |
| `test_loader_epub.py` | EPUB extracts text content |
| `test_preprocessor_sentences.py` | Sentence boundaries match known tokenization |
| `test_preprocessor_paragraphs.py` | Paragraph grouping correct for known text |
| `test_preprocessor_chapters.py` | Chapter detection matches annotated chapters |
| `test_narrative_position_velocity.py` | $v[t]$ decreases with event density |
| `test_narrative_position_normalization.py` | $\tau \in [0,1]$, $\tau[0]=0$, $\tau[-1]=1$ |
| `test_narrative_position_monotonic.py` | $\tau$ is strictly increasing |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_loader_to_preprocessor.py` | Loaded text → hierarchy with correct node counts |
| `test_preprocessor_to_position.py` | Hierarchy → τ assigned to every sentence node |

### Completion Criteria
- [ ] Loader handles .txt, .epub, .tei-xml
- [ ] Preprocessor produces correct sentence/paragraph/chapter boundaries for 3 test texts
- [ ] τ is strictly increasing, normalized to [0,1], $v[t] > 0$
- [ ] Spline wrapper produces correct output for known chapter τ values
- [ ] All unit tests pass
- [ ] Integration test: loader → preprocessor → τ on *Hills Like White Elephants* matches HMP-DT §7.2

---

## Phase 3 — Feature Extraction (Weeks 3–4)

### Goal
Implement all six per-dimension feature extractors. This is the most labor-intensive phase because each extractor requires NLP library integration and feature-specific logic.

### Modules
| Module | Dimension | Features | Estimated k_d | NLP Dependencies |
|--------|-----------|----------|---------------|------------------|
| `extractors/semantic.py` | $S$ | Lexical density, polysemy, concreteness, semantic coherence | 8 | spaCy, WordNet, NRC-VAD |
| `extractors/narrative.py` | $N$ | Tension, pacing, scene entropy, narrative trajectory | 8 | spaCy, dependency parser |
| `extractors/interaction.py` | $I$ | Speaker-hearer network, turn asymmetry, dialogue density | 7 | spaCy, quoted speech detector |
| `extractors/symbolic.py` | $Sy$ | Figurative density, motif persistence, symbolic coherence | 7 | spaCy, metaphor detector (optional) |
| `extractors/evolution.py` | $E$ | Arc slope, relationship state, developmental trajectory | 7 | spaCy, coreference resolver |
| `extractors/emotion.py` | $Em$ | Valence, arousal, intensity, emotional trajectory | 7 | NRC-VAD, VADER, custom lexicon |

### Dependencies
Phase 1 (models: `LiterarySignal`, `FeatureExtractor` Protocol), Phase 2 (preprocessed hierarchy, τ).

### Expected Inputs
- Per-sentence: tokens, POS tags, dependency parse, lemmas
- Per-paragraph: sentence index ranges
- Per-text: τ array

### Expected Outputs
- `f_S: dict[NodeID, NDArray[8]]`, `γ_S: dict[NodeID, float]`
- `f_N: dict[NodeID, NDArray[8]]`, `γ_N: dict[NodeID, float]`
- ... (same for all 6 dimensions)

### Person-Days per Extractor

| Extractor | Days | Risk Factors |
|-----------|------|-------------|
| Semantic | 3 | WordNet integration for polysemy; concreteness norms may need lookup table |
| Narrative | 2 | Tension heuristic requires dependency graph features |
| Interaction | 3 | Speaker attribution is an open NLP problem; use heuristic (quotation-based) |
| Symbolic | 3 | Figurative language detection is research-grade; use surface proxies (metaphor markers, lexical chains) |
| Evolution | 3 | Coreference resolution for character arcs; computationally expensive |
| Emotion | 2 | Lexicon-based (NRC-VAD); well-established |
| **Total** | **16** | |

### Estimated Complexity
**14–18 person-days.** The largest phase by effort. Each extractor is independent and can be parallelized across two developers.

### Implementation Strategy per Extractor

Each extractor follows the same pattern:

```python
class SemanticExtractor:
    def __init__(self, config):
        self.nlp = spacy.load(config.language_model)
        self.concreteness = load_norms(config.concreteness_path)
        self.wordnet = WordNetLemmatizer()

    def extract(self, sentence: str, tokens: list[str],
                pos: list[str], deps: list[tuple]) -> tuple[NDArray[8], float]:
        # 1. Compute each of 8 features
        # 2. Compute confidence γ from feature reliability heuristics
        # 3. Return (f, γ)
```

### Unit Tests (per Extractor)

| Test | What It Verifies |
|------|------------------|
| `test_[dim]_feature_shape.py` | Output is NDArray[k_d], γ is float ∈ [0,1] |
| `test_[dim]_known_input.py` | Known sentence produces expected feature values |
| `test_[dim]_empty_input.py` | Empty/null input returns zero vector, γ ≈ 0 |
| `test_[dim]_confidence.py` | Confident features (clear signal) give γ > 0.5 |

### Integration Tests

| Test | What It Verifies |
|------|------------------|
| `test_all_extractors_same_input.py` | All 6 extractors run on same 10 sentences; output dimensions match spec |

### Completion Criteria
- [ ] All 6 extractors implemented with `extract(sentence, tokens, pos, deps) → (NDArray[k_d], float)` signature
- [ ] `k_S=8, k_N=8, k_I=7, k_Sy=7, k_E=7, k_Em=7`
- [ ] All features bounded to $[0,1]$ or $[-1,1]$ per HMP-DT A3
- [ ] All γ ∈ [0,1]
- [ ] All unit tests pass
- [ ] Integration test: 6 extractors on 10 sentences of *Hills Like White Elephants* produces plausible values

---

## Phase 4 — Literary Signal Initialization (Week 4, days 3–5)

### Goal
Construct `LiterarySignal` 7-tuples from extracted features. This phase connects feature extraction to the HMP-DT engine.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| SignalInitializer | `lit-disc/modules/signal_initializer.py` | HMP-DT Eq. (10.1) |

### Dependencies
Phase 1 (models), Phase 2 (hierarchy, τ), Phase 3 (6 × f, γ).

### Expected Inputs
- `f_d: dict[NodeID, NDArray[k_d]]` — features for all nodes, dimension d
- `γ_d: dict[NodeID, float]` — confidences
- `τ: NDArray[float, N_sentences]` — narrative positions
- `HierarchyTree` — node relationships

### Expected Outputs
- `SignalStore` — maps `(NodeID, Dimension) → LiterarySignal` for every node at every level
  - Level ℓ=1: one signal per sentence per dimension (6 × N_sentences)
  - Levels ℓ=2..5: empty signals (filled by propagator in Phase 5)

### Estimated Complexity
**1–2 person-days.** Pure data assembly; no complex logic.

### Unit Tests
| Test | What It Verifies |
|------|------------------|
| `test_initialize_sentence.py` | Known f, γ, τ produce correct 7-tuple |
| `test_context_empty.py` | $\mathcal{C} = \emptyset$ initially |
| `test_provenance_sentence.py` | $\mathcal{P} = \{S_i\}$ initially |
| `test_residues_zero.py` | $\mathcal{R} = \mathbf{0}_{6\times 6}$ initially |
| `test_memory_traces_empty.py` | $\mathcal{M} = \emptyset$ initially |
| `test_signal_store_get_set.py` | Store and retrieve by (NodeID, Dimension) |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_feature_to_signal.py` | Phase 2 + 3 → Phase 4: all sentences initialized with correct dimensions |

### Completion Criteria
- [ ] SignalInitializer constructs 6 LiterarySignals per sentence
- [ ] Provenance is `{S_i}` for sentences, empty for higher levels
- [ ] All signals pass type checks
- [ ] All unit tests pass

---

## Phase 5 — HMP-DT Propagation (Weeks 5–6)

### Goal
Implement the core HMP-DT engine: bottom-up propagation (Eq. 10.2–10.8), horizontal coupling (Eq. 11.1–11.11), and the K-iteration refinement loop (Eq. 10.9). This is the algorithmic heart of the framework.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| Propagator | `lit-disc/modules/propagator.py` | HMP-DT Eq. (10.2)–(10.9) |
| Coupler | `lit-disc/modules/coupler.py` | HMP-DT Eq. (11.1)–(11.11) |
| RefinementController | `lit-disc/modules/refinement_controller.py` | HMP-DT §15 algorithm step 4 |

### Dependencies
Phase 1 (models, interfaces), Phase 4 (SignalStore).

### Expected Inputs (from Phase 4)
- `SignalStore` with ℓ=1 signals initialized

### Expected Outputs
- `SignalStore` with all levels ℓ=1..5 populated
- `CouplingState` at each level
- Convergence flag and iteration count
- Intermediate $\mathbf{H}(\tau)$ at each narrative sample point

### Person-Days per Sub-Task

| Task | Days | Complexity |
|------|------|------------|
| Sentence → Paragraph (Eq. 10.2–10.4) | 1 | Weighted mean + SC(P) |
| Paragraph → Chapter (Eq. 10.5–10.6) | 1 | Salience-weighted softmax |
| Chapter → Narrative (Eq. 10.7) | 1 | Cubic spline per dimension |
| Narrative → Discourse (Eq. 10.8) | 0.5 | Linear projection, concatenation |
| Adaptive weights (Eq. 11.1–11.2) | 1 | Raised cosine with gating |
| Kernel components (Eq. 11.3–11.8) | 2 | 6 components, each with different input types |
| Coupling operator (Eq. 11.9–11.11) | 1 | Weighted sum + element-wise update |
| Top-down refinement (Eq. 10.9) | 0.5 | Projection with β_down |
| Refinement loop (K iterations) | 1 | Convergence check, orchestration |
| **Total** | **9** | |

### Estimated Complexity
**8–12 person-days.** The most algorithmically dense phase. Each equation maps to 10–30 lines of NumPy.

### Unit Tests
| Test | Equation | What It Verifies |
|------|----------|------------------|
| `test_propagator_sentence_to_paragraph.py` | (10.2)–(10.4) | Confidence-weighted mean with known inputs |
| `test_propagator_SC_zero.py` | (10.3) | SC(P) = 0 produces correct γ |
| `test_propagator_SC_one.py` | (10.3) | SC(P) = 1 produces correct γ |
| `test_propagator_salience_weight.py` | (10.5) | α_j^{(d)} sums to 1, β_salience sharpens |
| `test_propagator_spline.py` | (10.7) | Interpolated values at known τ match expected |
| `test_propagator_projection.py` | (10.8) | Output dimension is d |
| `test_propagator_refinement.py` | (10.9) | Child signal shifts toward parent |
| `test_coupler_adaptive_weights.py` | (11.1)–(11.2) | Σ η_i = 1, η_i ∈ [0,1], raised cosine shape |
| `test_coupler_C_sim.py` | (11.3) | Cosine similarity with provenance projection |
| `test_coupler_C_prox.py` | (11.4) | Exponential decay with |τ - τ'| |
| `test_coupler_C_conf.py` | (11.5) | Product of γ's times feature similarity |
| `test_coupler_C_comp.py` | (11.6) | Bilinear form in [-1,1] |
| `test_coupler_C_sym.py` | (11.7) | Shared motif similarity |
| `test_coupler_C_rel.py` | (11.8) | Temporal averaging of past κ |
| `test_coupler_combined_kappa.py` | (11.9) | κ ∈ [0,1] |
| `test_coupler_operator.py` | (11.10)–(11.11) | Element-wise interaction with M matrix |
| `test_refinement_convergence.py` | — | ||S[k] - S[k-1]|| decreases |
| `test_refinement_early_stop.py` | — | Stops before K if converged |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_propagator_full_hierarchy.py` | Signal flows S→P→C→NL→DL with correct dimensions at each level |
| `test_coupler_with_propagator.py` | Horizontal coupling changes aggregated signals |
| `test_k1_full_loop.py` | K=1 iteration completes without error |
| `test_k5_full_loop.py` | K=5 iterations converge (||Δ|| < ε) |

### Completion Criteria
- [ ] All 6 levels populated after propagation
- [ ] Coupling κ ∈ [0,1] for all pairs
- [ ] Adaptive weights sum to 1 at all τ
- [ ] K iterations complete in < 10 seconds for a 100-sentence text
- [ ] Convergence detected before K=5 when appropriate
- [ ] All unit tests pass

---

## Phase 6 — Memory Subsystem (Week 7)

### Goal
Implement the Fourier feature key generation, storage, retrieval, reinforcement, decay, and signal augmentation. This is a self-contained subsystem that integrates into the refinement loop.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| Memory | `lit-disc/modules/memory.py` | HMP-DT Eq. (12.1)–(12.7) |

### Dependencies
Phase 1 (models: `MemoryEntry`, `MemoryStore`), Phase 5 (refinement loop calls memory operations).

### Expected Inputs
- Current `LiterarySignal` at any node/dimension
- Current `MemoryStore` state

### Expected Outputs
- Updated `MemoryStore` (new entries, reinforced activations, purged entries)
- Memory-augmented signal $\mathbf{f}_{\text{aug}}$

### Person-Days per Sub-Task

| Task | Days | Notes |
|------|------|-------|
| Fourier feature key (Eq. 12.2) | 1 | Random projection with fixed seeds; NumPy implementation |
| Storage decision (Eq. 12.1) | 0.5 | Threshold comparison |
| Retrieval (Eq. 12.3) | 0.5 | Cosine similarity, threshold filtering |
| Reinforcement and decay (Eq. 12.4–12.5) | 0.5 | In-place updates |
| Purging (Eq. 12.6) | 0.5 | List comprehension removal |
| Augmentation (Eq. 12.7) | 0.5 | Weighted sum with W_mem |
| **Total** | **3.5** | |

### Estimated Complexity
**3–5 person-days.** Self-contained; no external dependencies.

### Unit Tests
| Test | Equation | What It Verifies |
|------|----------|------------------|
| `test_memory_key_dimension.py` | (12.2) | Key shape is (32,) |
| `test_memory_key_deterministic.py` | (12.2) | Same f → same key (seeded W, b) |
| `test_memory_key_similarity.py` | (12.2) | Similar f → similar keys (cos > 0.5) |
| `test_memory_store_above_threshold.py` | (12.1) | Signal above θ_store is stored |
| `test_memory_store_below_threshold.py` | (12.1) | Signal below θ_store is not stored |
| `test_memory_retrieve_above_threshold.py` | (12.3) | Similar key retrieves stored entry |
| `test_memory_retrieve_below_threshold.py` | (12.3) | Dissimilar key does not retrieve |
| `test_memory_reinforcement.py` | (12.4) | β increases after retrieval, capped at 1.0 |
| `test_memory_decay.py` | (12.5) | β decreases by (1-δ)^Δt |
| `test_memory_purge.py` | (12.6) | Entry with β < θ_forget is removed |
| `test_memory_augmentation.py` | (12.7) | Augmented signal has larger norm than original |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_memory_in_refinement_loop.py` | Phase 5 + 6: memory-augmented signals change propagation |

### Completion Criteria
- [ ] Fourier keys are deterministic (seeded W, b)
- [ ] Storage, retrieval, reinforcement, decay, purging all correct
- [ ] Signal augmentation preserves bounds
- [ ] Memory capacity ≤ 200 entries enforced
- [ ] All unit tests pass
- [ ] Integration: memory changes LDRI for texts with recurring motifs

---

## Phase 7 — Resonance (Week 7, days 3–5)

### Goal
Implement the recursive resonance field computation and resonance-augmented signal. This module runs inside the refinement loop.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| Resonance | `lit-disc/modules/resonance.py` | HMP-DT Eq. (12.8)–(12.9) |

### Dependencies
Phase 1 (models: `ResonanceField`), Phase 5 (refinement loop calls resonance update).

### Expected Inputs
- Previous resonance field $\mathcal{R}(\tau_{i-1})$
- Current embedding $\mathbf{H}(\tau_{i-1})$
- $\Delta\tau$, $\lambda$

### Expected Outputs
- Updated $\mathcal{R}(\tau_i)$
- Resonance-augmented signal $\mathbf{f}_{\text{res}}(\tau)$

### Estimated Complexity
**1–2 person-days.** Single recurrence relation; trivial numerically.

### Unit Tests
| Test | Equation | What It Verifies |
|------|----------|------------------|
| `test_resonance_recursion.py` | (12.8) | Recursive update matches integral for constant H |
| `test_resonance_decay.py` | (12.8) | Large λ → slow decay, small λ → fast decay |
| `test_resonance_zero_input.py` | (12.8) | H = 0 → R → 0 over time |
| `test_resonance_augmentation.py` | (12.9) | Augmented signal has η_res contribution |
| `test_resonance_bounds.py` | (12.8) | R(τ) remains bounded for bounded H |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_resonance_in_loop.py` | Phase 5 + 7: resonance changes converged state |

### Completion Criteria
- [ ] Recursive update matches integral for constant H within $O(\Delta\tau^2)$ error
- [ ] Resonance field bounded for all τ
- [ ] All unit tests pass
- [ ] Integration: resonance contributes measurable ΔLDRI for texts with spatial framing

---

## Phase 8 — Uncertainty Propagation (Week 8, days 1–2)

### Goal
Implement uncertainty propagation through the hierarchy, confidence computation, and coupling uncertainty. This module runs alongside the propagator in the refinement loop.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| Uncertainty | `lit-disc/modules/uncertainty.py` | HMP-DT Eq. (13.1)–(13.6) |

### Dependencies
Phase 1 (models: `UncertaintyState`), Phase 5 (propagator calls uncertainty during aggregation).

### Expected Inputs
- Per-node uncertainty sources $\sigma_e^2, \sigma_s^2, \sigma_i^2, \sigma_t^2, \sigma_m^2$
- Coherence values SC(P) at paragraph level
- Salience weights $\alpha_j^{(d)}$ at chapter level

### Expected Outputs
- $\sigma^2$ at each node, each level
- $\gamma$ at each node, each level
- Global $\Gamma$ at discourse layer

### Estimated Complexity
**1–2 person-days.** Closed-form propagation equations; no iteration.

### Unit Tests
| Test | Equation | What It Verifies |
|------|----------|------------------|
| `test_uncertainty_total_variance.py` | (13.1) | Sum of 5 source variances |
| `test_uncertainty_confidence.py` | (13.2) | γ = 1 when σ_total = 0; γ = 0 when σ_total = σ_max |
| `test_uncertainty_paragraph_propagation.py` | (13.3) | Coherence reduces variance |
| `test_uncertainty_chapter_propagation.py` | (13.4) | Salience weighting increases variance when uneven |
| `test_uncertainty_narrative_propagation.py` | (13.5) | Interpolation adds distance-dependent variance |
| `test_uncertainty_coupling.py` | (13.6) | Coupling variance increases with κ |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_uncertainty_with_propagator.py` | Phase 5 + 8: γ decreases with hierarchy depth |

### Completion Criteria
- [ ] γ ∈ [0,1] at all nodes
- [ ] Uncertainty propagates upward correctly
- [ ] All unit tests pass

---

## Phase 9 — Narrative Embedding (Week 8, days 3–5)

### Goal
Construct the $\mathbb{R}^{120}$ Narrative Embedding from the converged HMP-DT state. This is the bridge between HMP-DT and LDRI.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| NarrativeEmbedding | `lit-disc/modules/embedding.py` | HMP-DT Eq. (14.1)–(14.2) |

### Dependencies
Phase 5 (converged signals at all levels), Phase 6 (memory store), Phase 7 (resonance field).

### Expected Inputs
- Final signals at discourse layer $\mathbf{f}_N^{(d)}(\tau)$ for all $d$, all $\tau$
- Pairwise interaction residues from the refinement loop
- Resonance field $\mathcal{R}$ at all sample points

### Expected Outputs
- `NarrativeEmbedding` with:
  - $\mathbf{H}_d \in \mathbb{R}^{12}$ for each $d \in \mathcal{D}$ (6 × 12 = 72)
  - $\mathbf{H}_{\text{pairs}} \in \mathbb{R}^{36}$
  - $\mathbf{H}_{\text{res}} \in \mathbb{R}^{12}$
  - $\mathbf{H} \in \mathbb{R}^{120}$ (concatenated)

### Estimated Complexity
**1–2 person-days.** Subspace construction from summary statistics.

### Unit Tests
| Test | Equation | What It Verifies |
|------|----------|------------------|
| `test_embedding_subspace_dimensions.py` | (14.1) | Σ dims = d (72 + 36 + 12 = 120) |
| `test_embedding_H_d_pooling.py` | (14.2) | Mean, max, slope, var, FFT computed correctly |
| `test_embedding_pairs_from_residues.py` | — | Pairs subspace populated from interaction residues |
| `test_embedding_res_from_field.py` | — | Resonance subspace from R summary stats |

### Integration Tests
| Test | What It Verifies |
|------|------------------|
| `test_embedding_from_converged.py` | Phase 5 + 6 + 7 + 9: embedding constructed from converged state |

### Completion Criteria
- [ ] $\mathbf{H} \in \mathbb{R}^{120}$
- [ ] All 8 subspaces populated with correct dimensions
- [ ] All unit tests pass

---

## Phase 10 — LDRI Calculation (Week 9)

### Goal
Implement the LDRI metric, its decomposition, and confidence estimation. This is the final analytical output of the pipeline.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| LDRICalculator | `lit-disc/modules/ldri_calculator.py` | LDRI Eq. (2)–(10), (11)–(14) |
| Decomposer | `lit-disc/modules/decomposer.py` | LDRI Eq. (16)–(23) |

### Dependencies
Phase 1 (models: `LDRIScore`), Phase 9 (NarrativeEmbedding), Phase 6 (memory store), Phase 7 (resonance field), Phase 8 (uncertainty state).

### Expected Inputs
- `NarrativeEmbedding` (from Phase 9)
- `MemoryStore` (from Phase 6)
- `ResonanceField` (from Phase 7)
- `UncertaintyState` (from Phase 8)

### Expected Outputs
- `LDRIScore`: LDRI, Γ, reliability, raw, decomposition components, fractions
- Bootstrap variance estimate

### Person-Days per Sub-Task

| Task | Days | Description |
|------|------|-------------|
| Dimension activation (Eq. 2) | 0.5 | Rayleigh CDF with σ_d null model |
| Signal richness norm (Eq. 3–4) | 0.5 | ℓ_2 norm / √6 |
| Coupling factor (Eq. 5, 9) | 0.5 | Scaled norm → [0.5,1] |
| Resonance factor (Eq. 6, 9) | 0.5 | Same pattern |
| Memory factor (Eq. 7, 9) | 0.5 | Average β activation |
| Geometric mean (Eq. 8, 10) | 0.5 | Confidence-weighted GM with exponents |
| Global confidence (Eq. 11–12) | 0.5 | Product of per-dimension γ |
| Reliability bootstrap (Eq. 13–14) | 1.0 | B=100 repetitions |
| Decomposition (Eq. 16–23) | 1.0 | 6 component equations |
| **Total** | **5.5** | |

### Estimated Complexity
**4–6 person-days.** Straightforward numerical computation; no iteration.

### Unit Tests
| Test | Equation | What It Verifies |
|------|----------|------------------|
| `test_ldri_activation.py` | (2) | $a_d \in [0,1)$, $g(0)=0$, $\lim_{x\to\infty} g(x) = 1$ |
| `test_ldri_signal_richness.py` | (3)–(4) | ℝ_signal ∈ [0,1] |
| `test_ldri_coupling_factor.py` | (5), (9) | 𝒞̃ ∈ [0.5, 1] |
| `test_ldri_resonance_factor.py` | (6), (9) | 𝒯̃ ∈ [0.5, 1] |
| `test_ldri_memory_factor.py` | (7), (9) | ℳ̃ ∈ [0.5, 1] |
| `test_ldri_compute.py` | (8), (10) | Known inputs → expected output matching §8 examples |
| `test_ldri_simple_chapter.py` | (10) | Synthetic simple chapter → LDRI = 0.42 |
| `test_ldri_complex_chapter.py` | (10) | Synthetic complex chapter → LDRI = 0.70 |
| `test_ldri_confidence.py` | (11)–(12) | Γ = Π γ_d |
| `test_ldri_bootstrap.py` | (13)–(14) | Reliability ∈ [0,1] |
| `test_decomposer_intrinsic.py` | (16) | c_int ≥ 0 |
| `test_decomposer_interaction.py` | (17) | c_intr ≥ 0, zero when 𝒞 = 0 |
| `test_decomposer_temporal.py` | (18) | c_temp ≥ 0, zero when 𝒯 = 0 |
| `test_decomposer_resonance.py` | (19) | c_res ≥ 0 |
| `test_decomposer_memory.py` | (20) | c_mem ≥ 0, zero when ℳ = 0 |
| `test_decomposer_uncertainty.py` | (21)–(22) | c_unc ≥ 0, zero when Γ = 1 |
| `test_decomposer_fractions.py` | (23) | Fractions sum ≈ 1 |

### Regression Tests
| Test | Expected LDRI | Source |
|------|---------------|--------|
| `test_regression_simple_chapter.py` | 0.42 | LDRI §8.1 |
| `test_regression_complex_chapter.py` | 0.70 | LDRI §8.2 |

### Completion Criteria
- [ ] LDRI ∈ [0,1] for all valid inputs
- [ ] All 6 decomposition components calculated
- [ ] Bootstrap reliability computed
- [ ] Regression tests match LDRI §8 worked examples within rel=1e-5
- [ ] All unit tests pass

---

## Phase 11 — Explainability (Week 10, days 1–2)

### Goal
Implement the explanation trace generator that produces the six-section structured trace consumed by the reporter and visualizer.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| ExplanationGenerator | `lit-disc/modules/explanation_generator.py` | HMP-DT §15 algorithm step 7 |

### Dependencies
Phase 5 (converged signals, provenance, interaction residues), Phase 6 (memory store), Phase 7 (resonance field), Phase 8 (uncertainty evolution).

### Expected Inputs
- Converged `SignalStore`
- `CouplingState` history (κ values per iteration)
- `MemoryStore`
- `ResonanceField`
- `UncertaintyState` per level

### Expected Outputs
- `ExplanationTrace` with 6 sections:
  1. `signal_lineage`: provenance-weighted contributions
  2. `propagation_path`: max product path
  3. `interaction_history`: strongest κ pairs
  4. `uncertainty_evolution`: σ² at each level
  5. `resonance_contribution`: fraction
  6. `memory_activations`: active entries with β

### Estimated Complexity
**1–2 person-days.** Trace assembly from existing state.

### Unit Tests
| Test | What It Verifies |
|------|------------------|
| `test_explanation_lineage.py` | Contributions sum to LDRI within tolerance |
| `test_explanation_propagation_path.py` | Path exists in hierarchy |
| `test_explanation_interaction.py` | Top pair has highest κ |
| `test_explanation_uncertainty.py` | 5 levels present |
| `test_explanation_resonance.py` | Fraction ∈ [0,1] |
| `test_explanation_memory.py` | All active memories listed |

### Completion Criteria
- [ ] 6 complete sections in trace
- [ ] All unit tests pass

---

## Phase 12 — Report Generation & Visualization (Week 10, days 3–5)

### Goal
Implement the report generator and visualization module for human-readable output.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| ReportGenerator | `lit-disc/modules/report_generator.py` | LDRI §5.4 (reporting convention) |
| Visualizer | `lit-disc/modules/visualizer.py` | — |

### Dependencies
Phase 10 (LDRIScore), Phase 11 (ExplanationTrace).

### Expected Inputs
- `LDRIScore`
- `ExplanationTrace`
- `NarrativeEmbedding`

### Expected Outputs
- Markdown report
- LaTeX tables
- JSON output
- SVG/PNG figures:
  - τ vs. dimension activation (6 lines, one per dimension)
  - Coupling heatmap (6×6 grid)
  - Uncertainty evolution (σ² vs. hierarchy level)
  - LDRI decomposition (stacked bar)
  - Embedding PCA projection (2D scatter)
  - Memory activation timeline (β vs. τ)

### Estimated Complexity
**2–3 person-days.** Library integration (matplotlib, seaborn).

### Unit Tests
| Test | What It Verifies |
|------|------------------|
| `test_report_json_schema.py` | JSON output matches schema |
| `test_report_decomposition_table.py` | LaTeX table has correct column count |
| `test_visualizer_dimension_plot.py` | Plot has 6 lines |
| `test_visualizer_coupling_heatmap.py` | Heatmap is 6×6 |
| `test_visualizer_uncertainty_plot.py` | Plot has 5 levels |

### Completion Criteria
- [ ] Report generates valid JSON, Markdown, LaTeX
- [ ] All 6 figure types render without error
- [ ] All unit tests pass

---

## Phase 13 — Evaluation Pipeline (Weeks 11–14)

### Goal
Implement the evaluation framework from the Evaluation Methodology chapter: run the full pipeline on all 5 datasets, compute all baselines, execute ablations, and produce the statistical analysis tables.

### Modules
| Module | Files | Equations |
|--------|-------|-----------|
| EvaluationRunner | `lit-disc/evaluation/runner.py` | Evaluation Methodology §3–7 |
| BaselineRunner | `lit-disc/evaluation/baselines.py` | Evaluation Methodology §4 |
| AblationRunner | `lit-disc/evaluation/ablations.py` | Evaluation Methodology §6 |
| StatisticalAnalyzer | `lit-disc/evaluation/statistics.py` | Evaluation Methodology §7 |
| BootstrapRunner | `lit-disc/evaluation/bootstrap.py` | LDRI Eq. (13)–(14) |

### Dependencies
All preceding phases (Phases 1–12).

### Datasets (Evaluation Methodology §3)
| Dataset | Texts | Source | Annotation Required |
|---------|-------|--------|---------------------|
| SNC (Standard Novel Corpus) | 100 | Project Gutenberg | 20 texts, 3 annotators |
| MNC (Modern Novel Corpus) | 50 | Booker Prize, NYT | 20 texts, 3 annotators |
| SSC (Short Story Corpus) | 100 | New Yorker, BASS | 20 texts, 3 annotators |
| MLC (Multilingual Parallel) | 30×2 | Original + translation | — |
| NWC (Non-Western) | 25 | Translations | 10 texts, experts |
| **Total** | **335** | | **70 annotated** |

### Baselines (Evaluation Methodology §4)
| Baseline | Implementation Effort |
|----------|----------------------|
| Flesch–Kincaid | Built-in (textstat library) |
| Type-Token Ratio | Built-in |
| VADER Sentiment | Built-in (vaderSentiment library) |
| LIWC | Requires LIWC dictionary license |
| Biber's Dimension 2 | Custom implementation from factor scores |
| BERT sentence embeddings | sentence-transformers library |
| LDA topic model | gensim |
| GPT-4o richness rating | OpenAI API call |

### Ablation Studies (Evaluation Methodology §6)
| Ablation | Implementation |
|----------|---------------|
| Memory off ($\mathcal{M} = 0$) | Set $\tilde{\mathcal{M}} = 0.5$ in LDRI; skip memory operations in loop |
| Resonance off ($\mathcal{T} = 0$) | Set $\tilde{\mathcal{T}} = 0.5$ in LDRI; skip resonance update in loop |
| Coupling off ($\kappa = 0$) | Set $\tilde{\mathcal{C}} = 0.5$ in LDRI; skip coupler in loop |
| Uncertainty off ($\Gamma = 1$) | Set $\Gamma = 1$ in LDRI |
| Explainability ablation | 3-group human experiment (§6.5) |

### Person-Days per Sub-Task
| Task | Days | Notes |
|------|------|-------|
| Dataset acquisition and cleaning | 3 | Project Gutenberg download, OCR fix, dedup |
| Annotation collection | 10 | Recruit, train, collect 70 annotations |
| Baseline implementation (8 baselines) | 5 | Most are library calls; LIWC is the exception |
| Ablation wrapper | 2 | Config overrides for each ablation |
| Batch pipeline runner | 2 | Parallel execution across 335 texts |
| Bootstrap runner | 1 | B=100 per text |
| Statistical analysis | 3 | RQ1–RQ7 tests, effect sizes, power |
| Interpretability experiment | 5 | 63 annotators, 3 groups, NASA-TLX |
| Reproducibility package | 2 | Docker, seeds, checksums |
| **Total** | **33** | |

### Estimated Complexity
**25–35 person-days.** Dominated by annotation collection (calendar time) and analysis.

### Completion Criteria
- [ ] All 335 texts processed through full pipeline
- [ ] All 8 baselines computed for all texts
- [ ] All 4 ablations executed
- [ ] Bootstrap reliability for all texts
- [ ] RQ1–RQ7 statistical tests computed
- [ ] Interpretability experiment complete with 63 annotators
- [ ] Reproducibility package published
- [ ] Results tables match expected patterns from Evaluation Methodology

---

## Timeline Summary

| Phase | Phase Name | Days | Calendar (Single Dev) | Calendar (Two Devs) |
|-------|-----------|------|----------------------|----------------------|
| 1 | Infrastructure | 5 | Week 1 | Week 1 |
| 2 | NLP Preprocessing | 7 | Week 2 | Week 1–2 |
| 3 | Feature Extraction | 16 | Weeks 3–4 | Weeks 2–3 |
| 4 | Signal Initialization | 2 | Week 4 | Week 3 |
| 5 | HMP-DT Propagation | 10 | Weeks 5–6 | Weeks 4–5 |
| 6 | Memory | 4 | Week 7 | Week 5 |
| 7 | Resonance | 2 | Week 7 | Week 5 |
| 8 | Uncertainty | 2 | Week 8 | Week 6 |
| 9 | Narrative Embedding | 2 | Week 8 | Week 6 |
| 10 | LDRI Calculation | 6 | Week 9 | Week 7 |
| 11 | Explainability | 2 | Week 10 | Week 7 |
| 12 | Report & Visualization | 3 | Week 10 | Week 7–8 |
| 13 | Evaluation Pipeline | 33 | Weeks 11–16 | Weeks 9–13 |
| **Total** | | **94 days** | **16 weeks** | **13 weeks** |

### Calendar Constraints
- Annotation collection (Phase 13) requires 3–4 weeks of calendar time regardless of developer count (recruiting, training, annotating)
- Phase 3 (Feature Extraction) is the only phase that benefits from parallelization across two developers (3 extractors each)
- Phases 5–11 are sequential and cannot be parallelized
- Total single-developer effort: approximately **4 months** (94 person-days)
- Total two-developer effort: approximately **3 months** (13 calendar weeks)

### Critical Path
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 9 → Phase 10
                                                ↓
                                          Phase 6, 7, 8, 11 (parallel after Phase 5)
```

Phase 5 is the critical path bottleneck. If Phase 5 slips, all subsequent phases slip.

### Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Phase 3 extractors require research-grade NLP (speaker attribution, figurative language) | Delayed or degraded feature quality | Use surface proxies initially; mark as "Feature Preview" with γ < 0.5; improve in v1.1 |
| Phase 5 spline interpolation edge case | Pipeline crash for short texts | Fallback to linear interpolation when $N_3 < 4$ |
| Phase 10 bootstrap takes too long ($B=100 \times 335$ texts) | Calendar overrun | Reduce $B$ to 50 for initial analysis; full $B=100$ only for final |
| Annotation collection yields low inter-annotator agreement ($\alpha < 0.60$) | Invalid expert correlation | Report per-annotator analyses; conduct calibration interviews |
| Phase 13 total compute (335 texts × 100 bootstrap × ~5 min) | ~28 days single-threaded | Parallelize across 64 cores → ~10 hours |

---

## Appendix: Regression Test Data Generation

Before implementing Phase 10, generate the regression test data from the LDRI §8 worked examples:

1. **Simple chapter:** Create a synthetic `NarrativeEmbedding` with $\|\mathbf{H}_d\|$ values matching §8.1. Compute LDRI manually (already done in the methodology). Save expected output as `tests/fixtures/simple_chapter_expected.json`.

2. **Complex chapter:** Same procedure for §8.2.

3. **Hemingway example:** Run Phases 2–5 on *Hills Like White Elephants*. Extract the embedding. Compute LDRI manually from HMP-DT §7.10. Save as `tests/fixtures/hemingway_expected.json`.

These three fixtures serve as the definitive regression tests for the full pipeline.

---

*End of Implementation Roadmap — LIT-DISC v1.0*
