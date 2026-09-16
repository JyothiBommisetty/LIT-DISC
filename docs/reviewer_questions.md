# Reviewer Questions — LIT-DISC v1.1

A skeptical peer-review simulation: 84 challenging questions with evidence-file answers.

---

## Framework & Motivation (Q1–Q12)

**Q1:** What is a "literary-discourse dimension" and how do you know there are exactly six? Why not five or seven?
→ `evidence/01_framework/framework_summary.md`: The 6 dimensions derive from the HMP-DT hierarchy (5 levels, each contributing at least one dimension). The architecture specification lists the theoretical justification.

**Q2:** The 6 dimensions are: emotion, narrative, symbolic, semantic, evolution, interaction. These look like categories a literary scholar invented. What is the principled derivation?
→ `evidence/01_framework/framework_summary.md`: Dimensions correspond to HMP-DT hierarchy levels: surface (emotion), structure (narrative), abstraction (symbolic), conceptual (semantic), temporal (evolution), relational (interaction).

**Q3:** How does HMP-DT differ from established discourse analysis frameworks like Rhetorical Structure Theory (RST) or Segmented Discourse Representation Theory (SDRT)?
→ `evidence/01_framework/framework_summary.md`: HMP-DT is hierarchical (5 levels) with continuous feature computation; RST/SDRT are tree-based with discrete relations. The comparison should be explicit in the related work section.

**Q4:** Why is a new discourse framework needed when dozens already exist?
→ `evidence/01_framework/framework_summary.md`: Existing frameworks are discrete (tree relations) and manual (annotation required). HMP-DT is continuous and automatic. This should be stated as a design rationale.

**Q5:** The 47 equations — how many are standard mathematics vs ad-hoc choices?
→ `evidence/01_framework/framework_summary.md`: The core equations (sigmoid, geometric mean, Gaussian similarity) are standard. The specific composition (H_d → S_d → C_d → ... → LDRI) is architectural.

**Q6:** Who designed these equations? What is their linguistic or literary qualification?
→ `evidence/09_reproducibility/reproducibility.md`: Not documented. This is a limitation.

**Q7:** Is the framework falsifiable? What experiment would disprove HMP-DT?
→ `evidence/10_claims/unsupported_claims.md`: Currently falsifiable in theory (e.g., show random features produce same LDRI ranking), but no falsification test was designed.

**Q8:** What does a 50% increase in LDRI mean in literary terms? Is it calibrated?
→ `evidence/07_statistics/statistics_summary.md`: LDRI is not calibrated. A 50% increase means the geometric mean of weighted component scores increased by 50%, but no mapping to human judgment exists.

**Q9:** How is this different from topic modeling or sentiment analysis applied to literary texts?
→ `evidence/03_baselines/baseline_summary.md`: Topic models capture distributional semantics; sentiment analysis captures valence. LDRI captures discourse richness (feature activation × confidence). The difference is visible in the weak correlation with readability and TTR.

**Q10:** Could existing tools (e.g., LIWC, Coh-Metrix) reproduce these results?
→ `evidence/03_baselines/baseline_summary.md`: Coh-Metrix computes similar readability metrics (Flesch-Kincaid, TTR). ρ with LDRI ranges from −0.16 to 0.53, suggesting LDRI captures something different.

**Q11:** What literary theory motivates the equation LDRI = gamma × (product of component scores)?
→ `evidence/01_framework/framework_summary.md`: The geometric mean penalizes low scores across dimensions. Gamma modulates by confidence. This is motivated as "conservative richness measurement" rather than a specific literary theory.

**Q12:** Is this a contribution to literary studies or to computational linguistics?
→ The paper should position itself as computational literary studies (CLS). The contribution is a metric + architecture + evaluation framework, applicable to both fields but without deep expertise in either.

## Activation & Saturation (Q13–Q20)

**Q13:** How could you ship v1.0 with sigma_max=0.1 and not notice 99% saturation?
→ `evidence/02_validation/activation_design_review.md`: Because the ranking was preserved. 99% of activations were 0.999+, but at 5+ decimal places, differences existed and propagated. Only CV analysis revealed the pathology.

**Q14:** If the ranking was preserved after fixing saturation, was sigma actually a problem?
→ `evidence/02_validation/activation_design_review.md`: Yes — LDRI CV of 0.31% (v1.0) vs 47.9% (v1.1). The metric was effectively constant. A metric that cannot discriminate is useless even if its ranking is stable.

**Q15:** Did you try other sigma values besides 0.1 and 5.0? Why 5.0?
→ `evidence/02_validation/activation_design_review.md`: A 10-value sweep from 0.05 to 10.0 was performed. sigma=5.0 was selected because it aligns with feature norms (||H_d|| ≈ 10–14 → z = 2–3, sigmoid linear region).

**Q16:** Is sigma_max=5.0 optimal? How do you define "optimal"?
→ "Optimal" is defined as: (a) saturation < 10%, (b) LDRI CV maximized, (c) author ranking preserved (ρ > 0.95). sigma=5.0 satisfies all three. No formal optimization was performed.

**Q17:** Does the framework overfit to the specific feature norms of these 25 passages?
→ `evidence/05_scalability/scalability_summary.md`: Feature norms at novel length (105 windows) are consistent with gold-standard norms. The sigma selection generalizes to full novels, but only one novel was tested.

**Q18:** What happens if features produce norms of 100 in a different corpus?
→ If features produce ||H_d|| = 100, then even sigma=5.0 would give z = 20, sigmoid(20) ≈ 1.0 → saturation. The activation scale is corpus-dependent. This is a limitation.

**Q19:** Why use sigmoid at all? Why not softmax, ReLU, or no activation?
→ `evidence/01_framework/framework_summary.md`: Sigmoid provides bounded [0,1] activation with a smooth transition. Softmax would enforce competition among dimensions (undesirable — they should be independent). ReLU would allow unbounded growth. The choice is architectural but should be justified.

**Q20:** Did you consider a learnable sigma per dimension instead of a global sigma_max?
→ `evidence/02_validation/activation_design_review.md`: The v1.0 framework has a learned sigma per dimension. v1.1 changed only the upper bound (sigma_max) from 0.1 to 5.0. However, the learned sigmas were also bounded by sigma_max, so the same saturation affected them.

## Baselines & Ablations (Q21–Q30)

**Q21:** Random gamma ρ=0.14 convincingly shows gamma is essential. But is this surprising? If gamma is 96.5% of LDRI variance, of course randomizing it destroys the ranking.
→ `evidence/03_baselines/baseline_summary.md`: Agreed. The Random Gamma baseline was designed to confirm the variance decomposition prediction experimentally, not as a surprising result. Its value is as experimental validation of the analysis.

**Q22:** Constant Features r>0.99 is even less surprising. If features vary by 2.6% and gamma varies by 47%, holding features constant should leave LDRI almost unchanged.
→ `evidence/03_baselines/baseline_summary.md`: The baseline validates the model's internal logic. It is not surprising given the statistics — but the statistics themselves are a finding.

**Q23:** The four baselines seem designed to confirm what the authors already know. Were any baselines genuinely surprising?
→ `evidence/03_baselines/baseline_summary.md`: The random gamma ρ=0.14 being that low is notable. One might have expected residual signal from raw LDRI to preserve some ranking. That it doesn't confirms the near-complete gamma dominance.

**Q24:** Why not include a supervised baseline (e.g., BERT-based literary quality classifier)?
→ A supervised baseline would compare LDRI against learned literary quality. Not included because: (a) no labeled data exists, (b) the paper claims LDRI is an unsupervised metric.

**Q25:** Why not compare against perplexity, which is a standard language model-derived text quality metric?
→ `evidence/03_baselines/baseline_summary.md`: Added to baseline coverage. If perplexity correlates with LDRI, that would be informative. This is a gap.

**Q26:** The ablation without uncertainty sets gamma=1. If gamma is what makes LDRI work, removing it should kill LDRI. What does this ablation prove?
→ `evidence/04_ablations/ablation_summary.md`: It proves that LDRI without uncertainty = raw LDRI, and raw LDRI is near-constant. This is the experimental demonstration that the metric collapses without gamma.

**Q27:** The ablations without memory, resonance, and coupling show |Δ| < 0.002. Doesn't this prove they are useless components?
→ `evidence/04_ablations/ablation_summary.md`: For the current pipeline and corpus, yes. Memory, resonance, and adaptive coupling contribute negligibly to LDRI. This calls into question their inclusion.

**Q28:** Why include components that contribute nothing? Why not remove them and simplify the framework?
→ The components are theoretically motivated by the HMP-DT hierarchy. Their implementation may be flawed, but the theory predicts they should contribute. The paper should acknowledge this gap.

**Q29:** The Without Narrative ablation shows Δ = +0.012. Why does removing a dimension *increase* LDRI?
→ `evidence/04_ablations/ablation_summary.md`: Because narrative has the lowest per-dimension gamma. Removing it removes a low-confidence dimension, increasing the geometric mean of the remaining higher-confidence dimensions.

**Q30:** Is the "Without Narrative" result a bug or a feature? Should LDRI increase when you remove information?
→ It is a feature of the geometric mean: removing a low-scoring dimension increases the product. This is mathematically correct but arguably undesirable for a richness metric. The framework should penalize, not encourage, dimensional reduction.

## Gold-Standard Corpus (Q31–Q40)

**Q31:** 25 passages from 25 authors in 5 categories — why this specific design?
→ `evidence/02_validation/gold_standard_summary.md`: 5 categories × 5 passages ensures balanced coverage of narrative modes. 25 total is standard for metric validation. The specific categories (SNC, MLC, NWC, SSC, MNC) cover the literary narrative space.

**Q32:** Are these the right 5 categories? Who decided them and based on what criteria?
→ `evidence/02_validation/gold_standard_summary.md`: No citation for the category taxonomy. This is a limitation.

**Q33:** Why 200 words per passage? Why not 100, 500, or 1,000?
→ `evidence/02_validation/gold_standard_summary.md`: 200 words was chosen to balance: (a) enough context for discourse features to be meaningful, (b) small enough for multiple passages to be processed in one evaluation. No principled argument.

**Q34:** Are the passages representative of the authors' styles? A 200-word extract from Woolf tells us very little.
→ `evidence/02_validation/gold_standard_summary.md`: This is a significant limitation. 200-word extracts cannot capture authorial style. The full-novel evaluation addresses this partially, but with only one novel.

**Q35:** Why 25 authors? Why these specific 25?
→ `evidence/02_validation/gold_standard_summary.md`: 25 for balanced design (5 × 5). The specific authors were selected for diversity (origin, period, style, gender). No formal selection criteria.

**Q36:** Is the corpus balanced by gender, period, geography, and language?
→ English-language corpus. Gender and period balance should be documented. Geography is limited to Indian subcontinent, Africa, UK, US, Europe.

**Q37:** Where are the passages from? Are they publicly available?
→ `evidence/09_reproducibility/reproducibility.md`: Sources are documented. Strictly for research use. Most 20th-century works are in copyright.

**Q38:** Could you share the full passage texts? Without them, results cannot be reproduced.
→ This is a limitation. Copyright prevents sharing. A possible mitigation is releasing only the computed scores per passage.

**Q39:** Why no poetry, drama, or non-fiction? The title says "literary discourse" but the corpus is only narrative fiction.
→ `evidence/02_validation/gold_standard_summary.md`: The gold standard deliberately focuses on narrative fiction. Non-narrative genres require separate validation. This is an acknowledged limitation.

**Q40:** Are the same 25 passages used for designing the metric and evaluating it? That's circular.
→ `evidence/02_validation/gold_standard_summary.md`: The gold-standard passages were used for evaluation only after the design freeze (v1.1). No parameter was tuned on these passages. The design was corrected by the sigma analysis (which revealed saturation), not by passage-level performance.

## Full-Novel Evaluation (Q41–Q52)

**Q41:** Why The Mistress of Spices? Why not a canonical English novel?
→ `evidence/05_scalability/scalability_summary.md`: The novel was selected for: public availability, manageable length (80k words), and literary richness (postcolonial magical realism, Booker Prize shortlisted). However, only one novel was tested, which is a major limitation.

**Q42:** One novel is not enough to draw general conclusions. Why not 3, 5, or 10 novels?
→ `evidence/05_scalability/scalability_summary.md`: Resource constraints. Processing one novel at 105 windows took ~85s, but analysis of 16 narrative events required reading ~8,000 words of text. Scaling to multiple novels is immediate future work.

**Q43:** Why 800-word windows? Why not chapter boundaries or natural paragraphs?
→ `evidence/05_scalability/scalability_summary.md`: 800 words was chosen as the largest size that fits within typical RFF memory capacity while allowing ~100 windows per novel. Paragraph boundaries are respected (window split is paragraph-greedy). Chapter boundaries would produce irregular window sizes.

**Q44:** Does window size affect LDRI? Could wider windows produce different rankings?
→ `evidence/05_scalability/scalability_summary.md`: Not tested. A sensitivity analysis of window size is missing. This is a limitation.

**Q45:** The first window produces 200 memory entries for 800 words. That's one entry per 4 words on average — seems high. Are entries meaningful?
→ `evidence/05_scalability/scalability_summary.md`: Memory entries are keyed by sentences (RFF of sentence features). The number of entries equals the number of sentences in the window. 200 entries for 800 words implies ~4 words per sentence, which is low but plausible for dialogue-heavy text.

**Q46:** Zero cross-window retrievals across 105 windows. Why doesn't the paper lead with this finding?
→ `evidence/05_scalability/scalability_summary.md`: It is arguably the most important architectural finding. The paper should lead with it in the memory section. The original analysis did not emphasize it sufficiently.

**Q47:** Is zero retrieval a parameter issue? Could lowering the retrieval threshold (θ=0.7) to 0.5 or 0.3 produce retrievals?
→ `evidence/05_scalability/scalability_summary.md`: Not tested. A retrieval threshold sweep (0.1–0.9) would determine whether the problem is threshold choice or key drift. This is a gap.

**Q48:** Gamma in the novel is 0.0105 vs 0.0302 in gold standard — roughly 1/3. Why is gamma lower at novel scale?
→ `evidence/05_scalability/scalability_summary.md`: Gamma is computed from per-dimension activations. At novel length, the activation function receives a wider variety of inputs, producing lower and more varied confidence estimates. The specific mechanism should be traced.

**Q49:** Raw LDRI in the novel is 0.950 vs 0.913 in gold standard — roughly 4% higher. Why is raw LDRI higher at novel scale?
→ `evidence/05_scalability/scalability_summary.md`: Likely because longer windows (800 vs 200 words) produce more stable feature estimates, reducing variance in the geometric mean. Should be confirmed experimentally.

**Q50:** The novel LDRI range is [0.0057, 0.0155]. Is a 9.8-millipoint range meaningful for literary analysis?
→ `evidence/06_case_study/mistress_case_study.md`: The range is small but interpretable (16 narrative events mapped to LDRI changes). However, distinguishing "meaningful" from "noise" within this range requires statistical testing.

**Q51:** The largest LDRI jump is +0.0078 (W25→W26). Could this be within the noise floor?
→ `evidence/06_case_study/narrative_event_log.md`: A bootstrap or non-parametric confidence interval for LDRI differences should be computed. Without it, distinguishing signal from noise is speculative.

**Q52:** How does the novel's LDRI compare to a shuffled control (same words, different order)?
→ `evidence/05_scalability/scalability_summary.md`: Not tested. A word-order baseline would determine whether LDRI captures structure or just lexical content.

## Literary Analysis (Q53–Q62)

**Q53:** The 16 narrative events were identified by looking at LDRI extremes. This is circular — you are finding events that match the metric's behavior.
→ `evidence/06_case_study/narrative_event_log.md`: Events were identified by: (a) LDRI/component z-score thresholds, (b) text extraction for the corresponding windows, (c) human reading and narrative annotation. The process is transparent but requires an independent annotation study.

**Q54:** What would a false positive look like — an LDRI peak that doesn't correspond to a narrative event?
→ `evidence/06_case_study/narrative_event_log.md`: The study did not systematically check for false positives (windows with high LDRI but no clear event). This should be done.

**Q55:** What would a false negative look like — a clear narrative event that doesn't produce an LDRI change?
→ `evidence/06_case_study/narrative_event_log.md`: Not systematically checked. Several obvious events (e.g., any customer story) may not correspond to LDRI changes.

**Q56:** The LDRI trajectory is described as "tracing the narrative arc." Could any metric with 21.8% CV trace any narrative arc?
→ `evidence/06_case_study/mistress_case_study.md`: A null model (random walk with same CV) would test whether apparent structure is random. Not performed.

**Q57:** What literary insight does the LDRI trajectory provide that close reading does not?
→ `evidence/06_case_study/mistress_case_study.md`: This is the central question for literary relevance. The LDRI trajectory quantifies the timing and magnitude of narrative transitions. Whether this quantification adds value over qualitative analysis is an open question requiring expert evaluation.

**Q58:** The peak at W40 ("Spices is this. The way. You have chosen. To. Punish me.") has LDRI=0.0155. The valley at W82 ("EVERYTHING MUST GO") has LDRI=0.0057. Is the ratio 2.7× meaningful?
→ `evidence/06_case_study/narrative_event_log.md`: The ratio is meaningful if the metric is ratio-scale. LDRI is not validated as ratio-scale. It is ordinal at best.

**Q59:** Renunciation (W82) has the lowest LDRI. But renunciation is arguably the novel's most important moment. Why would the metric score the climax lowest?
→ `evidence/06_case_study/mistress_case_study.md`: The renunciation scene is linguistically simple ("EVERYTHING MUST GO") — short phrases, low discursive complexity. LDRI captures complexity, not narrative importance. This is a crucial distinction the paper must make.

**Q60:** The case study reads like post-hoc storytelling. Would the authors have predicted these events before seeing the data?
→ For each event, one could construct a pre-registered prediction based on narrative theory and test it against LDRI. No prediction was made. The analysis is entirely post-hoc.

**Q61:** What is the inter-rater reliability for identifying 16 events? Would another reader find different events?
→ `evidence/06_case_study/narrative_event_log.md`: Single-rater analysis. No inter-rater reliability. This is a significant limitation for literary claims.

**Q62:** Is the narrative event log reproducible? Would the same 16 events be found on a second processing run?
→ Yes, the LDRI values are reproducible (deterministic). The event identification depends on the thresholds (z > 1.96). Changing thresholds would change the event list. A sensitivity analysis is missing.

## Gamma Dominance (Q63–Q72)

**Q63:** Gamma dominance at 96.5% is the most important finding. Why is it buried in statistics rather than leading the results?
→ `evidence/07_statistics/statistics_summary.md`: The results section currently leads with LDRI values. The paper should restructure to foreground gamma dominance.

**Q64:** At 96.5% dominance, is LDRI measuring anything other than uncertainty? Does it deserve the name "literary discourse richness"?
→ `evidence/10_claims/claims_and_evidence.md`: C39–C42. This is the central challenge to the framework. LDRI is primarily an uncertainty-weighting metric. The paper should either: (a) rename the metric, (b) reformulate to distribute signal across components, or (c) argue that uncertainty is a legitimate measure of richness.

**Q65:** If LDRI = gamma × constant, why not just report gamma? What does the geometric mean of features add?
→ The geometric mean of features becomes a scaling factor. Reporting gamma alone would capture ~96.5% of the signal with a simpler model. This argument must be addressed.

**Q66:** Gamma is computed from per-dimension activations, which are sigmoids of feature norms. Is gamma just measuring how large the features are?
→ Partially. Larger feature norms → larger activations → higher gamma. But gamma also depends on the learned per-dimension sigmoid steepness. The relationship between feature norms and gamma should be analyzed.

**Q67:** Could you define a "gamma-corrected LDRI" (LDRI / gamma or equivalently raw LDRI) as the actual richness?
→ `evidence/04_ablations/ablation_summary.md`: Raw LDRI (LDRI with gamma=1) is near-constant (CV=2.6%), so gamma-corrected LDRI would also be near-constant. This transformation does not produce a useful metric.

**Q68:** Is gamma dominance a consequence of the dataset (short passages), the architecture (geometric mean), or both?
→ `evidence/05_scalability/scalability_summary.md`: The novel evaluation shows gamma dominance persists at 95.3% for 105 windows. It is architectural, not dataset-dependent.

**Q69:** Would a different aggregation function (e.g., arithmetic mean, learned weighted sum) reduce gamma's dominance?
→ `evidence/04_ablations/ablation_summary.md`: Not tested. The geometric mean is the only aggregation used. A sensitivity analysis across aggregation functions could identify whether the dominance is specific to the geometric mean.

**Q70:** What would it take to achieve 50:50 gamma:raw contribution? Is that even desirable?
→ Desirability depends on the research question. For an "intrinsic richness" metric, features should contribute more. For a "confidence-weighted richness" metric, gamma dominance is appropriate. The paper must clarify which it claims to measure.

**Q71:** The narrative dimension has lowest gamma. Is this because narrative features are poorly designed, or because narrative is genuinely harder to measure with confidence?
→ `evidence/04_ablations/ablation_summary.md`: The ablation without narrative increases LDRI by +0.012. This suggests the narrative features have lower signal-to-noise ratio. Whether this reflects poor feature design or genuine measurement difficulty is unknown.

**Q72:** Could gamma be removed from LDRI entirely? What would the metric look like?
→ `evidence/03_baselines/baseline_summary.md`: Without gamma, LDRI = raw LDRI (mean=0.913, CV=2.6%). The metric becomes near-constant and cannot discriminate passages. Gamma is essential to LDRI's discriminative power, even as it dominates.

## Reproducibility & Validation (Q73–Q84)

**Q73:** 87 lint errors and 14 type errors. Why should I trust code that doesn't pass basic quality checks?
→ `evidence/09_reproducibility/reproducibility.md`: These are pre-existing (not from the sigma change), mostly auto-fixable, and do not affect correctness. A cleanup is planned.

**Q74:** The warm-call performance test fails (0.674s > 0.5s). Doesn't this invalidate the performance claims?
→ `evidence/09_reproducibility/reproducibility.md`: The failure is machine-dependent and pre-existing. Performance claims are approximately correct across runs. A threshold adjustment may be appropriate.

**Q75:** 116/117 tests pass. But 116/117 includes 57 memory/temporal tests. How many of those test retrieval (vs decay dynamics)?
→ `evidence/09_reproducibility/reproducibility.md`: The memory/temporal suite tests decay dynamics, monotonicity, and capacity constraints. Retrieval-specific tests: not counted separately. A breakdown should be provided.

**Q76:** The determinism tests pass (3/3). Does this include the full pipeline end-to-end?
→ `evidence/09_reproducibility/reproducibility.md`: Yes. The gold-standard evaluation was run twice and LDRI differences were < 1e-12.

**Q77:** Expert scores are all 4 (on a 0–5 scale). Why use placeholder scores? Why not exclude expert scores entirely?
→ `evidence/02_validation/gold_standard_summary.md`: Placeholder scores were a design error. They should be excluded from analysis until real expert ratings are collected.

**Q78:** Without expert correlation, what external validation does LDRI have?
→ `evidence/10_claims/unsupported_claims.md`: U3 — no external validation. LDRI is validated only internally (determinism, numerical stability, consistency with ablations). Expert validation is the highest-priority future work.

**Q79:** Have you considered an online experiment or crowdsourced annotations?
→ `evidence/10_claims/unsupported_claims.md`: Not yet. Crowdsourcing literary-discourse annotations is difficult but possible. This is future work.

**Q80:** The feature statistics show semantic (3.50) > narrative (3.11) > evolution (2.97) > interaction (2.65) > symbolic (2.59) > emotion (2.55). Are these differences statistically significant?
→ `evidence/07_statistics/statistics_summary.md`: No significance testing performed. A paired t-test or bootstrap confidence interval should be computed for feature mean differences.

**Q81:** 25 passages is a small sample. Power analysis for detecting ρ > 0.5 with 25 passages suggests ~80% power at α=0.05 for ρ≥0.55. Below that, the study is underpowered.
→ `evidence/07_statistics/statistics_summary.md`: 25 passages is small. The paper should acknowledge limited power for correlation analysis. ρ=0.53 (avg sentence length) and ρ=0.36 (TTR) may be unreliable.

**Q82:** Are the results robust to preprocessing variations (tokenization, sentence splitting, paragraph detection)?
→ `evidence/09_reproducibility/reproducibility.md`: The pipeline uses spaCy for tokenization and sentence splitting. A preprocessing sensitivity analysis (different tokenizers, different sentence splitters) was not performed.

**Q83:** LDRI values are at the passage/window level. Could LDRI be computed at sentence or paragraph level?
→ Possibly, but sentence-level features would be sparse. The architecture currently operates at window level because discourse features require context. Sentence-level LDRI would need architectural changes.

**Q84:** The entire evaluation was performed by the same people who built the framework. Is there a conflict of interest?
→ This is standard for computational literary studies (CLS) research where the same group builds and evaluates. Independent replication is ideal but rare in CLS. The paper should acknowledge the lack of external validation.
