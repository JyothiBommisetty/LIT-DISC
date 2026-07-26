# Experimental Evaluation Methodology for HMP-DT and LDRI

## A Validation Framework for Computational Literary Discourse Analysis

**Document Type:** Evaluation Methodology Chapter  
**Framework:** LIT-DISC (Literary Intelligence through Text-based Discourse Analysis)  
**Target Journal:** IEEE Transactions on Affective Computing / Digital Scholarship in the Humanities  
**Prerequisite:** HMP-DT Version 1.0 (Research Freeze) and LDRI Methodology Chapter

---

## Abstract

This chapter defines a rigorous experimental evaluation framework for validating the HMP-DT (Hierarchical Message Passing on Discourse Trees) model and the LDRI (Literary Discourse Richness Index) metric within the LIT-DISC framework. The evaluation addresses seven primary research questions spanning structural fidelity, correlation with expert judgment, interpretability, the efficacy of the memory and resonance mechanisms, confidence calibration, reliability, and robustness. For each question, testable hypotheses are formulated with explicit null and alternative hypotheses. The framework specifies five curated datasets spanning classic novels, modern novels, short stories, multilingual literature, and non-Western narratives; nine baseline methods covering traditional literary metrics and NLP approaches; and a comprehensive set of evaluation metrics for prediction quality, interpretability, robustness, stability, computational efficiency, and explainability. Five ablation studies isolate the contribution of each architectural mechanism. A full statistical analysis plan, threats-to-validity analysis, reproducibility specification, and experimental roadmap complete the framework.

---

## 1. Research Questions

### 1.1 Primary Research Questions

**RQ1 — Structural Fidelity.** Does HMP-DT capture hierarchical discourse structure more effectively than traditional flat feature aggregation?

**RQ2 — Expert Correlation.** Does LDRI correlate with expert literary judgments of discourse richness?

**RQ3 — Interpretability.** Does the HMP-DT explanation trace (§7 of the HMP-DT freeze) improve human interpretability of literary analysis outcomes?

**RQ4 — Memory Efficacy.** Does the HMP-DT memory mechanism (§12) improve the detection of recurring literary patterns?

**RQ5 — Confidence Calibration.** Is the HMP-DT confidence estimate $\gamma$ well-calibrated with actual feature extraction quality?

**RQ6 — Reliability.** Is LDRI stable under stochastic repetitions of the HMP-DT pipeline?

**RQ7 — Robustness.** Is LDRI robust to controlled perturbations of the input text?

### 1.2 Sub-Questions

**RQ1a.** Does cross-dimensional coupling (§11 of HMP-DT) improve discourse representation over independent per-dimension analysis?

**RQ1b.** Does the hierarchical propagation (§10 of HMP-DT) outperform sentence-level-only analysis?

**RQ2a.** Does the LDRI decomposition (§6 of the LDRI methodology) correspond to literary scholars' dimensional ratings?

**RQ2b.** Does the confidence-weighted LDRI ($\Gamma \cdot \text{LDRI}_{\text{raw}}$) correlate more strongly with expert judgments than the unweighted $\text{LDRI}_{\text{raw}}$?

**RQ4a.** Does memory retrieval (§12.3 of HMP-DT) contribute more to LDRI in texts with explicit recurring motifs (symbolic fiction) than in texts without (minimalist fiction)?

**RQ6a.** Which stochastic HMP-DT component (Fourier feature matrices $\mathbf{W}, \mathbf{b}$, coupling operator $\mathbf{M}_{d,d'}$) contributes most to LDRI variance?

**RQ7a.** Is LDRI more robust to synonym substitution (lexical variation) than to syntactic perturbation (constituent reordering)?

---

## 2. Hypotheses

### 2.1 RQ1: Structural Fidelity

**H1$_{0}$.** There is no statistically significant difference between HMP-DT's hierarchical representations and flat feature aggregation in predicting held-out literary annotations.

**H1$_{A}$.** HMP-DT's hierarchical representations yield significantly higher agreement with held-out literary annotations than flat feature aggregation.

**Expected outcome.** Reject H1$_{0}$. The hierarchical structure (sentence $\to$ paragraph $\to$ chapter $\to$ narrative layer $\to$ discourse layer) captures emergent discourse properties that flat aggregation misses. The mechanism for this is the coupling kernel $\kappa_{d,d'}(\tau)$, which operates at each hierarchical level and cannot be reproduced by post-hoc interaction terms on flat features.

**Effect size prediction.** Cohen's $d \geq 0.5$ (moderate effect). The expected improvement over the strongest flat baseline (BERT-mean) is $+0.08$ in Spearman $\rho$ for literary annotation prediction.

### 2.2 RQ2: Expert Correlation

**H2$_{0}$.** LDRI is uncorrelated with expert literary richness judgments ($\rho = 0$).

**H2$_{A}$.** LDRI is positively correlated with expert literary richness judgments ($\rho > 0$).

**Expected outcome.** Reject H2$_{0}$. LDRI captures a construct that overlaps with expert notions of discourse richness. The strongest correlation is expected with the "multi-dimensionality" dimension of expert ratings (a text that engages multiple literary registers), and the weakest with "aesthetic quality" (which LDRI deliberately excludes).

**Effect size prediction.** Spearman $\rho \geq 0.60$ (strong correlation). This is benchmarked against inter-annotator agreement on richness ratings, which typically ranges from $\rho = 0.40$ to $0.70$ for literary constructs (Piper, 2018; Underwood, 2019).

### 2.3 RQ3: Interpretability

**H3$_{0}$.** There is no difference in annotation accuracy between literary scholars using HMP-DT explanation traces and those using only raw LDRI scores.

**H3$_{A}$.** Literary scholars using HMP-DT explanation traces achieve significantly higher annotation accuracy (correct identification of which dimensions drive richness) than those using only raw LDRI scores.

**Expected outcome.** Reject H3$_{0}$. The explanation trace (§7 of the HMP-DT freeze) decomposes the Literary Signal into provenance paths, interaction history, uncertainty evolution, resonance contribution, and memory activation. This structured decomposition should reduce cognitive load and improve diagnostic accuracy.

**Effect size prediction.** $\geq 20\%$ improvement in dimension-identification accuracy (from $\sim 55\%$ with LDRI-only to $\sim 75\%$ with explanation traces), measured as $F_1$ score for matching attributions to expert rationales.

### 2.4 RQ4: Memory Efficacy

**H4$_{0}$.** Removing the memory mechanism ($\mathcal{M} = 0$) does not change LDRI scores significantly.

**H4$_{A}$.** Removing the memory mechanism ($\mathcal{M} = 0$) causes a significantly larger LDRI decrease for texts with explicit recurring motifs than for texts without such motifs.

**Expected outcome.** Reject H4$_{0}$. The memory mechanism (HMP-DT §12) stores and retrieves salient Literary Signals. Texts with recurring motifs (e.g., symbolic fiction) should lose more LDRI than texts without (e.g., minimalist fiction). This is a differential effect: the memory ablation produces a significant interaction between text type and LDRI change.

**Effect size prediction.** For symbolic fiction, $\Delta\text{LDRI} \geq 0.10$ when memory is ablated; for minimalist fiction, $\Delta\text{LDRI} \leq 0.03$. The interaction effect size $\eta^2_p \geq 0.10$.

### 2.5 RQ5: Confidence Calibration

**H5$_{0}$.** The confidence estimate $\gamma$ is uncorrelated with actual prediction error (measured as LDRI deviation from expert judgment).

**H5$_{A}$.** The confidence estimate $\gamma$ is negatively correlated with actual prediction error (higher confidence $\to$ lower error).

**Expected outcome.** Reject H5$_{0}$. The confidence propagation mechanism (HMP-DT §13) aggregates per-node confidences through the hierarchy, weighting down unreliable features. Texts with clear, unambiguous literary signals should simultaneously have high $\gamma$ and low error; texts with subtle or degraded signals should have low $\gamma$ and high error.

**Effect size prediction.** Spearman $\rho \leq -0.40$ (moderate negative correlation) between $\Gamma$ and $|\text{LDRI} - \text{expert LDRI}|$.

### 2.6 RQ6: Reliability

**H6$_{0}$.** The variance of LDRI across $B = 100$ bootstrap repetitions exceeds $\sigma^2_{\max} = 0.01$ (i.e., $\sigma_{\text{LDRI}} > 0.1$).

**H6$_{A}$.** The variance of LDRI across $B = 100$ bootstrap repetitions is bounded by $\sigma^2_{\max} = 0.01$ (i.e., $\sigma_{\text{LDRI}} \leq 0.1$).

**Expected outcome.** Reject H6$_{0}$ (i.e., confirm H6$_{A}$). The Fourier feature matrices $\mathbf{W}, \mathbf{b}$ (HMP-DT §5.2) are the primary stochastic component; with $k_{\text{key}} = 32$ features, the kernel approximation error is $O(1/\sqrt{32}) \approx 0.18$, but this affects only memory retrieval, not the core signal propagation. The expected $\sigma_{\text{LDRI}}$ is approximately $0.03$--$0.05$, well below the $0.1$ threshold.

### 2.7 RQ7: Robustness

**H7$_{0}$.** The mean absolute change in LDRI under controlled text perturbations exceeds $\Delta_{\max} = 0.10$.

**H7$_{A}$.** The mean absolute change in LDRI under controlled text perturbations is bounded by $\Delta_{\max} = 0.10$.

**Expected outcome.** Reject H7$_{0}$ (i.e., confirm H7$_{A}$) for lexical perturbations (synonym substitution), but potentially fail to reject for syntactic perturbations (constituent reordering). LDRI operates on literary signals, not surface form; synonym substitution preserves meaning and discourse structure, so LDRI should change minimally. Constituent reordering may alter emphasis, rhetorical structure, and narrative pacing, which legitimately changes discourse richness — the metric should be sensitive to such changes.

**Effect size prediction.** $\Delta\text{LDRI} \leq 0.05$ for $20\%$ synonym substitution; $\Delta\text{LDRI} \leq 0.12$ for constituent reordering. These expected bounds define acceptable vs. excessive sensitivity.

---

## 3. Datasets

### 3.1 Dataset 1: Standardized Novel Corpus (SNC)

**Source.** Project Gutenberg (public domain novels).

**Composition.** 100 novels spanning 1700–1923, balanced across:

| Period | Count | Examples |
|--------|-------|----------|
| 18th-century | 20 | Austen, Defoe, Swift |
| 19th-century | 50 | Dickens, Eliot, Dostoevsky, Flaubert |
| Early 20th-century | 30 | Woolf, Joyce, Faulkner, Hemingway |

**Justification.** This dataset enables comparison with existing computational literary studies (Jockers, 2013; Piper, 2018; Underwood, 2019), all of which use 19th-century novels as their primary corpus. The temporal span allows evaluation of LDRI's evolution across literary periods.

**Annotation requirement.** For a 20-novel subset, three literary scholars rate each novel on a 1–7 Likert scale for: (a) overall discourse richness, (b) each of the six HMP-DT dimensions, (c) confidence in their rating. Inter-annotator agreement target: Krippendorff's $\alpha \geq 0.60$.

**Anticipated LDRI range.** $0.30$ (Defoe, *Robinson Crusoe*) to $0.80$ (Joyce, *Ulysses$).

### 3.2 Dataset 2: Modern Novel Corpus (MNC)

**Source.** Booker Prize shortlists (2010–2025), The New York Times Notable Books.

**Composition.** 50 contemporary novels (2000–2025):

| Category | Count | Examples |
|----------|-------|----------|
| Literary fiction | 25 | Mantel, Saunders, Yanagihara |
| Genre fiction | 15 | Atwood (speculative), Tartt (crime-adjacent) |
| Experimental | 10 | Danielewski, Egan, Heti |

**Justification.** Contemporary literary fiction tests HMP-DT's applicability beyond the canon. Genre fiction tests the framework's ability to discriminate between discourse-rich genre fiction (e.g., Atwood's *The Blind Assassin*) and discourse-poor genre fiction (e.g., formulaic thrillers). Experimental fiction tests the framework's boundaries: non-linear narratives (§8 of HMP-DT freeze), multi-voiced texts, and typographic experimentation.

**Annotation requirement.** Same protocol as Dataset 1.

**Anticipated LDRI range.** $0.35$ (formulaic thriller) to $0.75$ (Smith, *White Teeth*).

### 3.3 Dataset 3: Short Story Corpus (SSC)

**Source.** *The New Yorker* fiction archive, *Best American Short Stories* (2010–2024).

**Composition.** 100 short stories:

| Category | Count | Length Range |
|----------|-------|-------------|
| Flash fiction | 25 | 500–1,500 words |
| Short story | 50 | 1,500–7,500 words |
| Novelette | 25 | 7,500–17,500 words |

**Justification.** Short stories test HMP-DT's length-robustness (Axiom 5 of LDRI). The hierarchy (sentence $\to$ paragraph $\to$ chapter) adapts: for flash fiction, there may be only one paragraph-level node, but the five-level hierarchy compresses to three active levels. The resonance mechanism (§12.6 of HMP-DT) decays over narrative position; in short texts, resonance may be weaker by the final position, which should produce lower $\mathcal{T}$ values.

**Annotation requirement.** For flash fiction, omit the "Evolution Richness" dimension (too short for character arcs). Full six-dimension rating for longer works.

**Anticipated LDRI range.** $0.25$ (plotless micro-fiction) to $0.70$ (Carver, Cheever).

### 3.4 Dataset 4: Multilingual Parallel Corpus (MLC)

**Source.** Literary works available in the original language and English translation.

**Composition.** 30 works $\times$ 2 language versions = 60 texts:

| Language | Count | Period | Examples |
|----------|-------|--------|----------|
| French | 10 | 1800–2000 | Flaubert, Proust, Beauvoir |
| German | 6 | 1800–2000 | Goethe, Kafka, Mann |
| Russian | 6 | 1800–2000 | Tolstoy, Gogol, Nabokov |
| Spanish | 8 | 1800–2000 | Cervantes, García Márquez, Bolaño |

**Justification.** Multilingual evaluation tests the language portability of HMP-DT's English-centric feature extractors. For each work, HMP-DT processes both the original and the English translation. A well-designed pipeline should produce similar LDRI scores for both versions (since discourse richness is a property of the underlying literary work, not of the particular language encoding). The difference $|\text{LDRI}_{\text{original}} - \text{LDRI}_{\text{translation}}|$ measures cross-lingual consistency.

**Anticipated result.** $|\text{LDRI}_{\text{original}} - \text{LDRI}_{\text{translation}}| \leq 0.15$ for 80\% of works. Larger discrepancies for poetry-heavy or dialect-heavy works (e.g., Bolaño's *The Savage Detectives*, which code-switches).

### 3.5 Dataset 5: Non-Western Narrative Corpus (NWC)

**Source.** Translations of canonical non-Western works.

**Composition.** 25 works:

| Tradition | Count | Examples |
|-----------|-------|----------|
| Japanese | 8 | Murasaki Shikibu, Tanizaki, Mishima, Murakami |
| Chinese | 6 | Cao Xueqin, Lu Xun, Mo Yan |
| Arabic | 5 | Mahfouz, Al-Koni, Shuaib |
| Indian (multiple languages) | 6 | Vyasa (*Mahabharata* excerpts), Tagore, Rushdie, Roy |

**Justification.** This dataset tests HMP-DT's non-Western claims (§8 of the HMP-DT freeze). The framework asserts that cultural adaptation is confined to parameters $(\mu_i, \omega_i, \alpha_i, \beta_i)$ — the architecture is universal. The dataset includes Murasaki's *The Tale of Genji* (11th-century Japanese, non-linear narrative, episodic structure) and Vyasa's *Mahabharata* (cyclic time, nested narratives). These test the multi-stream $\tau$ extension (Eq. §8.3) and the Kishōtenketsu parameterization (§8.2).

**Anticipated result.** With default Western parameters, LDRI may be under-estimated for non-Western works (e.g., $\text{LDRI} \leq 0.40$ for *Genji* despite its literary richness). With the appropriate parameterization (§8 of HMP-DT freeze), LDRI should increase significantly ($\Delta\text{LDRI} \geq 0.15$), validating the parameterization claim.

---

## 4. Baselines

### 4.1 Traditional Literary Metrics

| Baseline | Description | Relevance | Expected LDRI gap |
|----------|-------------|-----------|-------------------|
| **Flesch–Kincaid Grade Level** | Readability score based on syllable density and sentence length | Tests the claim that LDRI is not a readability proxy | Near-zero correlation ($\|rho\| \leq 0.10$) |
| **Type-Token Ratio (TTR)** | Lexical diversity: unique word types divided by tokens | Tests the claim that LDRI captures more than lexical breadth | Weak correlation ($\rho \approx 0.20$–$0.30$) |
| **VADER Sentiment (mean $\pm$ std)** | Rule-based sentiment valence and its variance | Tests the claim that LDRI subsumes emotion but is not reducible to it | Moderate correlation with $Em$ dimension ($\rho \approx 0.40$) but not with overall LDRI ($\rho \approx 0.15$) |
| **LIWC Narrative Focus** | Percentage of narrative-related words (past tense, personal pronouns) | Tests the claim that LDRI captures narrative structure beyond LIWC categories | Weak correlation ($\rho \approx 0.15$–$0.25$) |
| **Biber's Dimension 2 (Narrative vs. Non-Narrative)** | Factor score for narrative register features | Tests the claim that LDRI operates on discourse, not register | Moderate correlation for $N$ dimension ($\rho \approx 0.35$) |

### 4.2 NLP Baselines

| Baseline | Description | Relevance | Expected LDRI gap |
|----------|-------------|-----------|-------------------|
| **BERT sentence embeddings (mean pool)** | 768-d sentence-level BERT embeddings averaged over all sentences | Tests whether hierarchical structure adds value over flat context-aware representations | HMP-DT should outperform by $\rho \geq 0.08$ in expert correlation |
| **Sentiment trajectory (Reagan et al.)** | Emotional arc as a single time series | Tests whether multi-dimensional coupling adds value over single-dimension analysis | LDRI should outperform by $\rho \geq 0.15$ |
| **LDA topic model (50 topics)** | Unsupervised topic distributions per document | Tests whether LDRI captures more than topical breadth | Weak correlation ($\rho \approx 0.10$–$0.20$) |
| **LIWC-22 full profile** | ~100 psycholinguistic category scores | Tests the broadest available word-level feature baseline | LDRI should outperform by $\rho \geq 0.10$ |
| **GPT-4o richness rating (zero-shot)** | LLM prompted to rate discourse richness on $[0, 1]$ | Tests whether a general-purpose LLM can approximate LDRI without the structured pipeline | GPT-4o may achieve moderate correlation ($\rho \approx 0.40$–$0.55$) but with higher variance and lower reliability |
| **GPT-4o with chain-of-thought decomposition** | LLM prompted to analyze each dimension then combine | Tests whether structured decomposition improves LLM performance | Expected $\rho \approx 0.50$–$0.65$, still below expert inter-annotator agreement |

### 4.3 Ablated Versions of HMP-DT (Self-Baselines)

| Baseline | Description | Relevance |
|----------|-------------|-----------|
| **HMP-DT$^{\text{no-mem}}$** | HMP-DT with $\mathcal{M} = 0$ (no memory) | Isolates memory contribution (RQ4) |
| **HMP-DT$^{\text{no-res}}$** | HMP-DT with $\mathcal{T} = 0$ (no resonance) | Isolates resonance contribution |
| **HMP-DT$^{\text{no-couple}}$** | HMP-DT with $\kappa_{d,d'} = 0$ for $d \neq d'$ (no coupling) | Isolates coupling contribution (RQ1a) |
| **HMP-DT$^{\text{flat}}$** | Sentence-level signals aggregated directly to discourse layer (skip ℓ=2,3) | Tests hierarchy contribution (RQ1b) |
| **HMP-DT$^{\text{no-uncert}}$** | HMP-DT with $\Gamma = 1$ (no uncertainty) | Tests confidence calibration contribution (RQ5) |

---

## 5. Evaluation Metrics

### 5.1 Prediction Quality

**Primary metric.** Spearman rank correlation $\rho$ between LDRI and expert richness judgments (Dataset 1–3, annotated 20-novel subset).

**Secondary metrics.**
- Pearson correlation $r$ (linear association; complement to Spearman)
- Root Mean Squared Error (RMSE) after linear calibration: $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\text{LDRI}_i - \alpha - \beta \cdot \text{expert}_i)^2}$
- $F_1$ score for correctly classifying texts as "high richness" (top quartile) vs. "low richness" (bottom quartile) based on expert judgments

**Per-dimension metrics.** Spearman $\rho_d$ between each dimension activation $a_d$ and the corresponding expert dimension rating. This tests whether the HMP-DT dimensions are semantically meaningful individually, not just collectively.

### 5.2 Interpretability

**Primary metric.** Dimension-identification accuracy: the proportion of texts for which a human annotator using the HMP-DT explanation trace correctly identifies which two dimensions contribute most to LDRI.

**Secondary metrics.**
- Time-to-decision: average time (seconds) for annotators to produce a dimension-identification judgment
- Cognitive load: NASA-TLX scores for annotators using explanation traces vs. raw scores
- Faithfulness: the correlation between the explanation trace's attribution weights and the actual contribution of each dimension to LDRI (measured via leave-one-dimension-out ablation)

### 5.3 Robustness

**Primary metric.** Mean absolute perturbation effect:

$$\Delta_{\text{pert}} = \frac{1}{N}\sum_{i=1}^N |\text{LDRI}(\text{perturbed}_i) - \text{LDRI}(\text{original}_i)|$$

**Perturbation types.**
1. *Synonym substitution*: replace 10\%, 20\%, 50\% of content words with WordNet synonyms
2. *Constituent reordering*: randomly permute 5\%, 10\%, 20\% of clause-level constituents
3. *Paraphrase*: GPT-4o paraphrases of selected paragraphs
4. *Noise injection*: add Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.01)$ to $\mathbf{H}$ before LDRI computation

**Acceptable bounds.**
- $\Delta_{\text{synonym}} \leq 0.05$ for 20\% substitution
- $\Delta_{\text{paraphrase}} \leq 0.08$ for full-paragraph paraphrase
- $\Delta_{\text{noise}} \leq L \cdot \|\epsilon\|$ (Lipschitz constant $L$ from Proposition 4 of LDRI)

### 5.4 Stability

**Primary metric.** Bootstrap standard deviation:

$$\sigma_{\text{LDRI}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (\text{LDRI}^{(b)} - \overline{\text{LDRI}})^2}$$

for $B = 100$ bootstrap repetitions (resampling Fourier feature seed and coupling matrix seed).

**Secondary metric.** Spearman-Brown split-half reliability: correlation between LDRI computed on odd-numbered paragraphs vs. even-numbered paragraphs, corrected for test length.

**Acceptable bounds.** $\sigma_{\text{LDRI}} \leq 0.05$; split-half reliability $r_{SB} \geq 0.85$.

### 5.5 Computational Efficiency

**Primary metric.** Wall-clock time (seconds) for the full HMP-DT + LDRI pipeline per text.

**Secondary metrics.**
- Peak memory usage (MB)
- Number of FLOPs (theoretical, per HMP-DT §16)
- Scaling with text length: $T(N_1)$ for $N_1 \in \{10^2, 10^3, 10^4, 10^5\}$ sentences

**Acceptable bounds.** $T \leq 300$ seconds for a standard novel ($N_1 \approx 5000$) in Python/NumPy single-threaded; memory $\leq 2$ GB; FLOPs consistent with HMP-DT §16 ($\approx 1.5 \times 10^7$).

### 5.6 Explainability

**Primary metric.** Decomposition stability:

$$\text{Stab}_{\text{dec}} = \frac{1}{N}\sum_{i=1}^N \| \mathbf{c}_i - \mathbf{c}_i^{(b)} \|_2$$

where $\mathbf{c}_i$ is the decomposition vector $(c_{\text{int}}, c_{\text{intr}}, c_{\text{temp}}, c_{\text{res}}, c_{\text{mem}}, c_{\text{unc}})$ and $\mathbf{c}_i^{(b)}$ is the decomposition from the bootstrap repetition $b$.

**Secondary metrics.**
- Dominance consistency: the proportion of texts for which the same component ranks first before and after perturbation
- Additivity residual: $\|\text{LDRI} - \sum_j c_j\|$ (should be small by construction; larger values indicate approximation breakdown)

---

## 6. Ablation Studies

### 6.1 Study 1: Memory Ablation

**Design.** Compare LDRI computed with the full HMP-DT pipeline versus HMP-DT$^{\text{no-mem}}$ (memory store disabled: $\mathcal{M} = 0$, $\tilde{\mathcal{M}} = 0.5$).

**Expected effect on LDRI.** $\Delta\text{LDRI} = \text{LDRI} - \text{LDRI}^{\text{no-mem}}$ varies by text type:
- Symbolic fiction (Joyce, García Márquez): $\Delta\text{LDRI} \geq 0.10$
- Minimalist fiction (Hemingway, Carver): $\Delta\text{LDRI} \leq 0.03$
- Genre fiction (thriller, romance): $\Delta\text{LDRI} \approx 0.05$

**Expected effect on decomposition.** The $c_{\text{mem}}$ component drops to near-zero. The $c_{\text{int}}$ component may increase slightly (compensation), but total LDRI decreases.

**Statistical test.** Paired $t$-test between $\text{LDRI}$ and $\text{LDRI}^{\text{no-mem}}$ for each text type. Interaction test: $2 \times 2$ mixed ANOVA (text type $\times$ memory condition).

### 6.2 Study 2: Resonance Ablation

**Design.** Compare LDRI versus HMP-DT$^{\text{no-res}}$ (resonance disabled: $\mathcal{T} = 0$, $\tilde{\mathcal{T}} = 0.5$).

**Expected effect.** $\Delta\text{LDRI}$ is largest for:
- Works with strong spatial or temporal framing (e.g., *Wuthering Heights*: the moors persist throughout)
- Works with delayed resolution (e.g., mystery novels: early clues resonate at the denouement)
- Works where short-term fluctuations mask long-term trends (resonance acts as a low-pass filter)

**Expected null result.** $\Delta\text{LDRI} \approx 0$ for works with linear, present-tense, single-scene narratives (e.g., *Hills Like White Elephants*: 10 continuous minutes in a single location).

### 6.3 Study 3: Coupling Ablation

**Design.** Compare LDRI versus HMP-DT$^{\text{no-couple}}$ (coupling disabled: $\kappa_{d,d'} = 0$, $\tilde{\mathcal{C}} = 0.5$).

**Expected effect.** $\Delta\text{LDRI}$ is largest for:
- Works where emotional valence and symbolic meaning are tightly coupled (e.g., *The Great Gatsby*: the green light is simultaneously symbol and emotion)
- Works where character interaction and narrative tension are coupled (e.g., dialogue-heavy drama)
- Works where the coupling components interact (e.g., $C_{sim}$ and $C_{sym}$ both high for richly figurative texts)

**Interpretation.** If $\Delta\text{LDRI} \approx 0$ for a text, it means the six dimensions operate independently — each can be analyzed in isolation. If $\Delta\text{LDRI} > 0$, the coupling has detected genuine interaction.

### 6.4 Study 4: Uncertainty Ablation

**Design.** Compare LDRI ($\Gamma = \prod_d \gamma_d$) versus HMP-DT$^{\text{no-uncert}}$ ($\Gamma = 1$).

**Expected effect.** $\Delta\text{LDRI} = (1 - \Gamma) \cdot \text{LDRI}_{\text{raw}}$ is proportional to the uncertainty penalty:

| Confidence Level | Example Text Type | Expected $\Gamma$ | Expected $\Delta\text{LDRI}$ |
|-----------------|-------------------|-------------------|------------------------------|
| High | 19th-century English novel with clear feature signals | 0.85–0.95 | 0.02–0.07 |
| Medium | Contemporary experimental fiction (non-standard features) | 0.70–0.85 | 0.06–0.13 |
| Low | Non-English original (English-centric feature extractors) | 0.40–0.60 | 0.15–0.30 |

**Validation.** Compare $\Gamma$ against the variance across bootstrap repetitions ($\sigma_{\text{LDRI}}$). If $\Gamma$ and $\sigma_{\text{LDRI}}$ are correlated, then $\Gamma$ captures genuine uncertainty; if uncorrelated, $\Gamma$ may reflect feature strength rather than uncertainty.

### 6.5 Study 5: Explainability Ablation

**Design.** Compare the annotation accuracy of three groups of literary scholars:
1. *LDRI-only group*: receives only the scalar LDRI value for each text
2. *Decomposition group*: receives the LDRI decomposition ($c_{\text{int}}, c_{\text{intr}}, c_{\text{temp}}, c_{\text{res}}, c_{\text{mem}}$)
3. *Full-trace group*: receives the complete HMP-DT explanation trace (lineage, path, interaction history, uncertainty evolution, resonance contribution, memory activation)

**Task.** For each text, identify the top two contributing dimensions and provide a brief rationale.

**Primary analysis.** One-way ANOVA comparing $F_1$ scores across the three groups.

**Secondary analysis.** Qualitative content analysis of the rationales to identify whether the explanation trace produces more precise or more complex justifications.

---

## 7. Statistical Analysis Plan

### 7.1 Pre-Registration

All hypotheses, datasets, baselines, and analytic procedures will be pre-registered on the Open Science Framework prior to data collection. No post-hoc modifications to the experimental design will be permitted after annotation collection begins.

### 7.2 Primary Analyses

| RQ | Primary Test | Justification |
|----|-------------|---------------|
| RQ1 | Spearman $\rho$ between model predictions and held-out annotations; Meng's test for correlated correlation coefficients | Meng's test (Meng et al., 1992) compares two dependent Spearman correlations (HMP-DT vs. flat baseline) on the same data |
| RQ2 | Spearman $\rho$ between LDRI and expert richness judgments; one-sided test for $\rho > 0$ | One-sided because H2$_A$ specifies positive correlation |
| RQ3 | One-way ANOVA on $F_1$ scores (3 groups); post-hoc Tukey HSD | Three independent groups with a continuous outcome |
| RQ4 | $2 \times 2$ mixed ANOVA: text type (symbolic vs. minimalist) $\times$ memory condition (full vs. ablated) | One between-subjects factor (text type), one within-subjects factor (ablation) |
| RQ5 | Spearman $\rho$ between $\Gamma$ and $|\text{LDRI} - \text{expert LDRI}|$; one-sided test for $\rho < 0$ | One-sided because H5$_A$ specifies negative correlation |
| RQ6 | One-sided one-sample $t$-test: $H_6: \sigma_{\text{LDRI}} \leq 0.1$ | Testing whether bootstrap SD is below threshold |
| RQ7 | One-sided one-sample $t$-test: $H_7: \mu_{\Delta} \leq 0.10$ | Testing whether perturbation effect is below threshold |

### 7.3 Effect Size Reporting

All analyses will report:
- Point estimate with 95\% confidence interval (bootstrap percentile method)
- Standardized effect size (Cohen's $d$, Spearman $\rho$, $\eta^2_p$, Cohen's $f$)
- Bayes factor $\text{BF}_{10}$ for primary analyses (Jeffreys–Zellner–Siow prior)

### 7.4 Multiple Comparison Correction

For the primary analyses (7 tests), the Benjamini–Hochberg procedure controls the false discovery rate at $q = 0.05$. Per-dimension and per-baseline sub-analyses are explicitly designated as exploratory and reported without correction.

### 7.5 Power Analysis

| RQ | Expected Effect | $\alpha$ | Power $(1-\beta)$ | Required $N$ |
|----|----------------|----------|-------------------|--------------|
| RQ1 | $\rho = 0.60$ vs. $\rho = 0.50$ (Meng's test) | 0.05 | 0.80 | 60 texts |
| RQ2 | $\rho = 0.60$ (one-sided) | 0.05 | 0.80 | 19 texts |
| RQ3 | ANOVA $f = 0.40$ (large) | 0.05 | 0.80 | 21 per group (63 total) |
| RQ4 | $\eta^2_p = 0.10$ | 0.05 | 0.80 | 40 texts (20 per type) |
| RQ5 | $\rho = -0.40$ (one-sided) | 0.05 | 0.80 | 37 texts |
| RQ6 | $d = 0.50$ (one-sided one-sample) | 0.05 | 0.80 | 27 texts |
| RQ7 | $d = 0.50$ (one-sided one-sample) | 0.05 | 0.80 | 27 texts |

The limiting factor is the annotation requirement (RQ3 requires 63 human judgments). The primary LDRI-expert correlation (RQ2) requires only 19 annotated texts, which is feasible.

### 7.6 Sensitivity Analysis

For RQ2 (the central validation), a sensitivity analysis will examine the influence of:
- Removing each individual annotator (jackknife)
- Removing each individual text (jackknife)
- Varying the annotation aggregation method (mean vs. median vs. consensus)
- Using quadratic weighted Cohen's $\kappa$ instead of Spearman $\rho$

---

## 8. Threats to Validity

### 8.1 Internal Validity

**Confound: text length.** Longer texts have more opportunities for literary signals to accumulate. While HMP-DT normalizes via narrative position $\tau \in [0, 1]$, residual length effects may persist.

*Mitigation.* Include length as a covariate in all analyses. Report partial correlations controlling for word count. Explicitly test for length $\times$ LDRI interaction.

**Confound: genre effects.** Genre and discourse richness may be confounded (literary fiction is, by definition, richer than genre fiction). Significant LDRI differences between genres may reflect genre labels rather than discourse properties.

*Mitigation.* Include genre as a blocking factor in experimental design. Within-genre correlations (e.g., literary fiction only) should replicate the overall pattern. If LDRI distinguishes rich literary novels from impoverished literary novels, it is measuring discourse richness, not genre membership.

**Confound: annotation bias.** Expert annotators may unconsciously prefer canonical texts (which HMP-DT was designed to model) and penalize unfamiliar or experimental forms.

*Mitigation.* Include foil texts (randomly generated word salads, foreign-language text without translation) as attention checks. Annotate blind to text identity. Measure inter-annotator agreement separately for canonical and contemporary texts.

### 8.2 External Validity

**Scope limitation.** The HMP-DT freeze (Assumptions A1–A8) explicitly limits applicability to narrative prose, single-author works, monolingual text, and published literary works. The evaluation does not generalize beyond these boundaries.

*Mitigation.* Clearly delineate the scope in all publications. Report per-subset results (classic, modern, short, multilingual, non-Western) separately. If LDRI fails on non-Western texts with default parameters, this validates the parameterization claim (§8 of the freeze) rather than invalidating the model.

**Corpus representativeness.** The standardized novel corpus (Dataset 1) over-represents Anglo-American 19th-century fiction. Results may not generalize to other literary traditions.

*Mitigation.* Balanced dataset design (5 datasets covering different traditions). Explicit cross-dataset comparison in the analysis. Weight results by dataset diversity.

### 8.3 Construct Validity

**Does LDRI measure discourse richness?** The central construct validity question. Without an independent gold standard for "discourse richness," the risk of circularity is real: LDRI is validated against a construct that the framework itself defines.

*Mitigation.* The validation strategy uses **convergent evidence** from multiple sources:
1. Expert correlation (RQ2): LDRI aligns with human judgments
2. Dimensional validity (RQ1a): per-dimension activations align with per-dimension ratings
3. Discriminant validity: LDRI does NOT correlate strongly with readability, lexical diversity, or sentiment (see §4.1 baselines)
4. Ablation logic: removing mechanisms predictably degrades LDRI in theoretically-expected patterns (RQ4)

If all four converge, the construct validity claim is strengthened.

**Are expert ratings themselves valid?** Human judgments of "discourse richness" may reflect different criteria across annotators. The construct may be unstable.

*Mitigation.* Report inter-annotator agreement (Krippendorff's $\alpha$). If $\alpha < 0.60$, report per-annotator analyses. Conduct a qualitative elicitation study (structured interviews) to identify what criteria annotators use.

### 8.4 Reproducibility

**Pipeline complexity.** HMP-DT involves multiple interacting components (feature extraction, propagation, coupling, memory, resonance, uncertainty). The full pipeline is difficult to replicate independently.

*Mitigation.* See §9 (Reproducibility Specification). Containerization, documented random seeds, and version-locked dependencies.

**Stochastic components.** The Fourier feature key generation (§5.2 of HMP-DT) is the primary stochastic process. While seeded by default, different random seeds may produce different memory retrieval patterns.

*Mitigation.* Bootstrap analysis (H6) quantifies this variance. Report LDRI $\pm$ bootstrap SE, not just point estimates.

### 8.5 Dataset Bias

**Historical bias.** The canon is skewed toward Western, male, elite authors. Dataset 1 (Project Gutenberg) reflects this bias.

*Mitigation.* Dataset 5 (Non-Western) explicitly addresses this. Dataset 2 (Modern) includes diverse authors. Report demographic breakdown of authors and discuss potential bias.

**Annotation bias.** Expert annotators are likely trained in the Western literary tradition. Their judgments may systematically undervalue non-Western literary techniques.

*Mitigation.* Include at least one annotator with expertise in each literary tradition represented in Dataset 5. Compare Western-trained vs. tradition-trained annotations for non-Western texts.

---

## 9. Reproducibility Specification

### 9.1 Software Environment

| Component | Specification |
|-----------|---------------|
| Operating system | Linux (Ubuntu 22.04 LTS), macOS 14+, Windows 11 |
| Python version | 3.11+ |
| Package manager | pip with lock file (pip-tools) or conda environment |
| Containerization | Docker image with all dependencies pre-installed (Dockerfile provided) |
| Version control | Git with annotated tags for each experimental run |

### 9.2 Random Seed Management

| Component | Seed | Rationale |
|-----------|------|-----------|
| Fourier feature $\mathbf{W}$ | 0 | Fixed seed ensures cross-session reproducible keys (HMP-DT §5.4) |
| Fourier feature $\mathbf{b}$ | 1 | Fixed seed ensures cross-session reproducible keys (HMP-DT §5.4) |
| Coupling matrices $\mathbf{M}_{d,d'}$ | 42 | Fixed seed for zero-shot coupling |
| Bootstrap iteration $b$ | 1000 + $b$ | Systematic seed ensures bootstrap reproducibility |
| Synonym substitution | 2025 | Fixed seed for reproducibility of perturbation experiments |

All seeds are configurable via a central configuration file (`config/seeds.yaml`). The default seeds above are used for all primary analyses; sensitivity analyses vary seeds to test robustness.

### 9.3 Data Management

| Requirement | Specification |
|-------------|---------------|
| Raw text storage | Plain text UTF-8 files, one file per text |
| Preprocessing output | JSONL format, one line per sentence with tokens, POS tags, dependency parses |
| HMP-DT intermediate outputs | NumPy `.npz` format with named arrays for each signal component |
| LDRI scores | CSV with columns: text_id, LDRI, LDRI_raw, Gamma, Reliability, c_int, c_intr, c_temp, c_res, c_mem, c_unc |
| Annotations | CSV with columns: text_id, annotator_id, overall_richness, S_rating, N_rating, ..., confidence |
| Version tracking | MD5 checksums for all raw text files; Git LFS for large binary outputs |

### 9.4 Annotation Protocol

| Requirement | Specification |
|-------------|---------------|
| Annotator pool | Minimum 3, maximum 7 literary scholars (PhD or advanced graduate) |
| Training | 2-hour training session with 5 practice texts, calibration discussion |
| Annotation tool | Web-based interface (Qualtrics or custom React app) |
| Blind condition | Texts identified by random IDs; author and title hidden |
| Order randomization | Text presentation order randomized per annotator |
| Break scheduling | Maximum 10 texts per session, minimum 30-minute breaks |
| Inter-annotator check | Krippendorff's $\alpha$ computed after each 10-text batch; if $\alpha < 0.50$, retraining session |

### 9.5 Output Directory Structure

```
experiments/
  run_YYYY-MM-DD_v1/
    config/
      seeds.yaml
      pipeline_config.yaml
    data/
      raw/          (MD5-checked input texts)
      preprocessed/ (JSONL sentence tokens)
      features/     (extracted feature vectors)
    results/
      hmp_dt/       (.npz per text)
      ldri/         (.csv per text)
      ablations/    (per-ablation result tables)
      baselines/    (per-baseline result tables)
    analysis/
      figures/      (publication-ready .pdf and .png)
      tables/       (LaTeX .tex tables)
      stats/        (.csv of all statistical test results)
    annotations/
      raw/          (raw annotator responses)
      aggregated/   (processed and aggregated scores)
    reproducibility/
      Dockerfile
      requirements.txt
      README.md
```

### 9.6 Computational Resource Requirements

| Experiment | Time Estimate | Memory Estimate | Core Count |
|------------|---------------|-----------------|------------|
| Single text (full pipeline) | 180 sec | 1.5 GB | 1 CPU |
| Dataset 1 (100 texts) | 5 hours | 2 GB | 16 CPU (parallel) |
| All baselines (per text) | 600 sec | 4 GB | 1 CPU |
| Bootstrap ($B=100$, per text) | 5 hours | 2 GB | 16 CPU (parallel) |
| Full experimental suite | ~72 hours | 16 GB | 64 CPU cluster |


## 10. Experimental Roadmap

### 10.1 Phase 1: Corpus Preparation (Weeks 1–2)

1. Acquire and clean all five datasets
   - Download from Project Gutenberg, Booker Prize lists, *The New Yorker* archive
   - OCR correction for multilingual texts
   - Sentence tokenization (spaCy or Stanza)
   - Paragraph detection (blank-line heuristic)
   - Chapter detection (heading pattern heuristic)
2. Generate MD5 checksums for all raw files
3. Random split: 80\% discovery set, 20\% held-out test set

**Milestone 1.1.** All 335 texts tokenized, paragraph- and chapter-structured, checksum-verified.

### 10.2 Phase 2: Annotation Collection (Weeks 3–6)

1. Recruit and train annotators (3–7 literary scholars)
2. Pilot annotation on 5 texts with group calibration
3. Annotate discovery set (16 texts for Dataset 1; 16 for Dataset 2; 16 for Dataset 3; 10 for Dataset 4; 10 for Dataset 5)
4. Compute inter-annotator agreement after every 10 texts

**Milestone 2.1.** 68 annotated texts with Krippendorff's $\alpha \geq 0.60$ on overall richness.

**Milestone 2.2.** Per-dimension annotations for all 68 texts.

### 10.3 Phase 3: Pipeline Implementation and Unit Testing (Weeks 7–8)

1. Implement HMP-DT feature extractors (one per dimension)
2. Implement hierarchical propagation (§10 of HMP-DT)
3. Implement coupling kernel (§11 of HMP-DT)
4. Implement memory and resonance (§12 of HMP-DT)
5. Implement uncertainty propagation (§13 of HMP-DT)
6. Implement Narrative Embedding (§14 of HMP-DT)
7. Implement LDRI (§3–4 of LDRI methodology)
8. Unit-test each component on synthetic data

**Milestone 3.1.** All components pass unit tests (known input $\to$ expected output).

**Milestone 3.2.** Full pipeline runs end-to-end on a single text in $< 300$ seconds.

### 10.4 Phase 4: Benchmarking and Ablation (Weeks 9–11)

1. Run full pipeline on all 335 texts
2. Run all baselines (§4) on all texts
3. Run all ablations (§6) on all texts
4. Compute LDRI for each ablation condition

**Milestone 4.1.** Complete result tables for all conditions.

**Milestone 4.2.** Bootstrap reliability estimates for all conditions.

### 10.5 Phase 5: Statistical Analysis (Weeks 12–13)

1. Compute all primary and secondary metrics (§5)
2. Execute pre-registered statistical tests (§7)
3. Sensitivity analyses (§7.6)
4. Compile per-dataset results (§3)

**Milestone 5.1.** All RQs answered with effect sizes and confidence intervals.

**Milestone 5.2.** 2–3 key tables and 3–5 key figures for the publication.

### 10.6 Phase 6: Interpretability Experiment (Weeks 14–15)

1. Recruit and randomize 63 annotators into three groups (§6.5)
2. Run interpretability task (dimension identification)
3. Analyze $F_1$ scores, NASA-TLX, and rationales

**Milestone 6.1.** Interpretability results with between-group comparisons.

### 10.7 Phase 7: Final Analysis and Publication (Weeks 16–20)

1. Reveal held-out test set (20\% of texts)
2. Run full pipeline on test set
3. Confirm primary results generalize
4. Write up results
5. Prepare reproducibility package (§9)
6. Submit to target journal

**Milestone 7.1.** Pre-registered analyses confirmed on held-out set.

**Milestone 7.2.** Reproducibility package published on OSF with DOI.

**Milestone 7.3.** Manuscript submitted.

### 10.8 Roadmap Summary

```
Week  | Phase
------|------------------------------------------------------------
 1–2  | Corpus preparation (335 texts, 5 datasets)
 3–6  | Annotation collection (68 texts, 3–7 annotators)
 7–8  | Pipeline implementation and unit testing
 9–11 | Benchmarking and ablation (full suite on all texts)
12–13 | Statistical analysis
14–15 | Interpretability experiment (63 participants, 3 groups)
16–20 | Final analysis, reproducibility package, and publication
```

---

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Jockers, M. L. (2013). *Macroanalysis: Digital Methods and Literary History*. University of Illinois Press.
- Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology* (4th ed.). Sage.
- Meng, X.-L., Rosenthal, R., \& Rubin, D. B. (1992). Comparing correlated correlation coefficients. *Psychological Bulletin*, 111(1), 172–175.
- Piper, A. (2018). *Enumerations: Data and Literary Study*. University of Chicago Press.
- Reagan, A. J., Mitchell, L., Kiley, D., Danforth, C. M., \& Dodds, P. S. (2016). The emotional arcs of stories are dominated by six basic shapes. *EPJ Data Science*, 5(1), 31.
- Underwood, T. (2019). *Distant Horizons: Digital Evidence and Literary Change*. University of Chicago Press.

---

*End of Evaluation Methodology Chapter*
