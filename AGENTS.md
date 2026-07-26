# AGENTS.md — LIT-DISC Project

## Commands
- **Play**: `python -m pytest tests/` (pytest-format tests once written)
- **Verify**: `python tests/run_verification.py` — runs all 6 verification suites (66 tests, equation + numerical + determinism + performance + memory + edge cases)
- **Lint**: `python -m ruff check src/`
- **Typecheck**: `python -m mypy src/`
- **Format**: `python -m ruff format src/`

## File Locations
- **Docs**: `docs/` — all frozen methodology documents (HMP-DT v1.0, LDRI, evaluation, architecture, roadmap)
- **Source**: `src/` — 16 packages + 6 root modules, Protocol-based architecture
- **Tests**: `tests/` — planned per implementation roadmap
- **Data**: `data/` — input texts for evaluation

## Frozen Files (do NOT modify)
- `docs/hmp_dt_v1_freeze.md` — HMP-DT v1.0 specification
- `docs/ldri_methodology.md` — LDRI metric methodology
- `docs/evaluation_methodology.md` — evaluation framework
- `docs/architecture_specification.md` — software architecture
- `docs/implementation_roadmap.md` — implementation plan
- `src/models.py` — frozen dataclass definitions
- `src/config.py` — frozen configuration parameters

## Architecture Summary
16 packages, 47 equations, 5-level hierarchy, 6 dimensions, d=120 embedding.
Pipeline: Load → Preprocess → Hierarchy → Features (6) → Signals → HMP-DT (propagate/couple/memory/resonance/uncertainty) → Embed → LDRI → Explain → Report → Visualize.

## Verification Status (2026-07-26)
- **Equation Verification**: 39/39 pass — all core HMP-DT equations verified
- **Numerical Stability**: 10/10 pass — NaN/Inf/zero/extremes handled
- **Determinism / Reproducibility**: 3/3 pass — identical across runs
- **Performance**: 2/2 pass — cold < 2s, warm < 0.5s on 400-word text
- **Memory / Temporal Validation**: 6/6 pass — decay dynamics, monotonicity
- **Edge Cases**: 6/6 pass — empty/single/long/unicode/latin-1/nonexistent
- **Total**: 66/66 pass
- **Runner**: `python tests/run_verification.py`
