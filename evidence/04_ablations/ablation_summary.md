# Ablation Study Summary — Stage 3

Five ablations test each HMP-DT component by disabling it via runtime monkey-patching (no source modifications).

## Results

| Condition | LDRI | Δ | \|Δ\| | Runtime | Effect |
|-----------|------|---|-------|---------|--------|
| Full Pipeline | 0.0277 | — | — | 0.21s | Reference |
| Without Memory | 0.0264 | −0.0013 | 0.0013 | 0.17s | SMALL |
| Without Resonance | 0.0261 | −0.0016 | 0.0016 | 0.23s | SMALL |
| Without Adaptive Coupling | 0.0292 | +0.0015 | 0.0015 | 0.14s | SMALL |
| **Without Uncertainty** | **0.9126** | **+0.8850** | **0.8850** | 0.21s | **LARGE** |
| **Without Narrative** | **0.0393** | **+0.0116** | **0.0116** | 0.16s | **MODERATE** |

## Interpretation

### Without Uncertainty (LARGE effect)
Removing uncertainty (setting Gamma=1.0) collapses LDRI to raw_ldri (mean=0.913). This confirms that LDRI = Gamma × raw_ldri and that gamma is the dominant scaling factor. LDRI is essentially uncertainty-weighted raw richness.

### Without Narrative Signals (MODERATE effect)
Removing the narrative dimension increases LDRI by +0.012 (+42%). This suggests that the narrative dimension has lower confidence (gamma) than other dimensions, and its removal reduces the gamma penalty. The effect is concentrated in passages with high narrative content (SNC category: Δ=+0.021 in SNC-02).

### Without Memory / Resonance / Adaptive Coupling (SMALL effects)
|Δ| < 0.002 in all cases. These components contribute minimally to final LDRI. Memory fills rapidly (max_size=200), resonance produces near-zero values, and adaptive coupling is dominated by the confidence-weighting scheme.

## Per-Passage Variability

| Ablation | Window with largest |Δ| | Value | Passage |
|----------|---------------------|-------|---------|
| No Memory | SNC-04 (Woolf) | 0.0038 | To the Lighthouse |
| No Resonance | NWC-04 (Roy) | 0.0089 | The God of Small Things |
| No Coupling | SNC-03 (Dostoevsky) | 0.0041 | Crime and Punishment |
| No Uncertainty | SSC-04 (Munro) | 0.9200 | The Bear Came Over... |
| No Narrative | SNC-02 (Dickens) | 0.0206 | Great Expectations |
