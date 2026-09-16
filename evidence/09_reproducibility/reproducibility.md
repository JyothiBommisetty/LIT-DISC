# Reproducibility Report — LIT-DISC v1.1

## Environment

| Property | Value |
|----------|-------|
| Operating System | Windows (PowerShell 5.1) |
| Python version | 3.14 |
| Working directory | `D:\LIT-DISC` |
| Date | 2026-07-29 |

## Key Dependencies

- numpy
- matplotlib (for figures)
- pytest (for tests)
- ruff (linting)
- mypy (type checking)
- (other dependencies as specified in project requirements)

## Configuration

All parameters are frozen in `src/config.py`. The only change from v1.0 to v1.1 is:

```
src/config.py:59 — sigma_max: float = 0.1  →  sigma_max: float = 5.0
```

No other parameters, equations, or algorithms were modified.

## Seed Values

| Parameter | Value | Location |
|-----------|-------|----------|
| Memory key W seed | 0 | `src/config.py:MemoryParams.seed_W` |
| Memory key b seed | 1 | `src/config.py:MemoryParams.seed_b` |

## Hardware

Processing performed on standard workstation hardware. Runtime measurements are machine-dependent.

## Execution Order

```bash
# 1. Verify configuration
python -c "from src.config import LITDISCConfig; print(LITDISCConfig().ldri.sigma_max)"
# Expected: 5.0

# 2. Run verification suite
python tests/run_verification.py

# 3. Lint check
python -m ruff check src/

# 4. Type check
python -m mypy src/

# 5. Generate gold standard results
python generate_gold_standard.py

# 6. Generate descriptive statistics
python analyze_results.py

# 7. Generate baselines (Stages 1+2)
python run_baselines.py

# 8. Generate ablations (Stage 3)
python run_ablations.py

# 9. Generate full-novel evaluation
python run_novel_evaluation.py
```

## Total Runtime (all experiments)

| Experiment | Time |
|------------|------|
| Verification suite | ~40s |
| Gold standard (25 passages) | ~5s |
| Baselines (Stages 1+2) | ~30s |
| Ablations (Stage 3, 5×25) | ~70s |
| Full novel (105 windows) | ~85s |
| **Total** | **~230s** |

## Test Results

### Verification Suite

| Suite | Pass | Fail | Time |
|-------|------|------|------|
| Equation Verification | 39 | 0 | 0.25s |
| Numerical Stability | 10 | 0 | 0.33s |
| Determinism / Reproducibility | 3 | 0 | 8.35s |
| Performance | 1 | 1 | 2.33s |
| Memory / Temporal Validation | 57 | 0 | 0.49s |
| Edge Cases | 6 | 0 | 28.00s |
| **Total** | **116** | **1** | ~40s |

The single failure is the warm-call performance test (0.674s > 0.5s threshold). This is machine-dependent and pre-existing in v1.0.

### Lint (ruff)

87 pre-existing style issues found (unused imports, import ordering, etc.). None are related to the sigma change. All are auto-fixable with `ruff --fix`.

### Type Check (mypy)

14 pre-existing type issues found (missing stubs for `ebooklib`, `chardet`; untyped function bodies; etc.). None are related to the sigma change.

## Folder Locations

| Artifact | Path |
|----------|------|
| Source code | `src/` |
| Tests | `tests/` |
| Gold standard data | `data/gold_standard/` |
| Novel data | `data/mistress_spices_narrative.txt` |
| Analysis scripts | Root directory (`*.py`) |
| Novel evaluation output | `output/novel_evaluation/` |
| Evidence | `evidence/` |

## Determinism

The pipeline is fully deterministic for the same input text. The gold-standard evaluation produces identical results on re-run. Reproducibility was verified by running the complete 25-passage evaluation twice and confirming LDRI differences < 1e-12.
