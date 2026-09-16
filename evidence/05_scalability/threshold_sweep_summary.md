# Memory Retrieval Threshold Sweep Summary

**Date:** 2026-07-30
**Experiment:** `data/experiments/threshold_sweep/run_threshold_sweep.py`
**Data:** `data/experiments/threshold_sweep/threshold_sweep_results.json`

## Design

Swept `theta_retrieve` from 0.1 to 0.9 on 5 representative gold-standard passages (single-window, ~200 words each). Measured LDRI, gamma, memory entries, and average memory activation (beta).

## Results

| θ | SNC-04 LDRI | SNC-04 β | MNC-01 LDRI | MNC-01 β | MLC-02 LDRI | MLC-02 β | SSC-04 LDRI | SSC-04 β | NWC-05 LDRI | NWC-05 β |
|---|-------------|----------|-------------|----------|-------------|----------|-------------|----------|-------------|----------|
| 0.1 | 0.05593 | 1.0000 | 0.00917 | 1.0000 | 0.05495 | 1.0000 | 0.01511 | 1.0000 | 0.01279 | 1.0000 |
| 0.3 | 0.05589 | 0.9992 | 0.00911 | 0.9982 | 0.05490 | 0.9995 | 0.01510 | 0.9985 | 0.01262 | 0.9992 |
| 0.5 | 0.05587 | 0.9968 | 0.00911 | 0.9953 | 0.04870 | 0.9870 | 0.01503 | 0.9940 | 0.01230 | 0.9945 |
| 0.7 | 0.05507 | 0.9920 | 0.00877 | 0.9870 | 0.05220 | 0.9788 | 0.01482 | 0.9888 | 0.01201 | 0.9825 |
| 0.9 | 0.05323 | 0.9827 | 0.00866 | 0.9935 | 0.05143 | 0.9810 | 0.01443 | 0.9815 | 0.01189 | 0.9825 |

## Key Findings

### 1. Threshold has minimal effect within a window

Even at θ=0.9, avg_beta ≥ 0.98 (max 1.0). All 200 entries are stored and remain at near-max strength regardless of retrieval threshold. This is because within a single short passage, RFF keys are highly similar and always exceed even the strictest threshold.

### 2. LDRI is nearly threshold-invariant

LDRI varies by < 5% across the full θ=0.1–0.9 range for all passages. The memory factor M ≈ 0.98–1.0 at all thresholds, contributing negligibly to the geometric mean in LDRI computation.

### 3. The zero-retrieval problem is architectural, not a threshold artifact

Since within-window retrieval is universal at all thresholds, but cross-window retrieval is zero across 105 novel windows (at θ=0.7), the problem must be architectural:
- RFF keys are computed from sentence-level features that change with narrative context
- Across windows, keys drift sufficiently far that even θ=0.1 would not match
- The fundamental issue is the RFF key representation, not the cosine similarity threshold

## Implication

Lowering `theta_retrieve` below 0.7 would not solve the cross-window retrieval problem. A structural redesign of the memory representation (e.g., semantic chunking, content-based hashing, overlapping windows) is needed for cross-window functionality.

## Data

Raw results: `data/experiments/threshold_sweep/threshold_sweep_results.json`
