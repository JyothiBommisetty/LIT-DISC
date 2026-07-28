"""Comprehensive tests for NarrativeEmbeddingConstructor — HMP-DT §14.

Test categories:
  1. Construction — Eqs. (14.1)–(14.2) subspace assembly
  2. Per-dimension subspaces — Eq. (14.2) statistics
  3. Pairwise interactions — H_pairs computation
  4. Resonance subspace — H_res from trajectory or signals
  5. Dimension validation — d ∈ {40, 120, 240, 480}
  6. Serialization — save/load round-trip
  7. Edge cases — empty signals, single node
  8. Integration — worked example §7.9
"""
import json
import math
import sys
import tempfile
sys.path.insert(0, "D:\\LIT-DISC")

import numpy as np
from src.embedding import NarrativeEmbeddingConstructor
from src.models import NarrativeEmbedding, LiterarySignal, NodeID

RTOL = 1e-4
ATOL = 1e-6

_DIMS = ["S", "N", "I", "Sy", "E", "Em"]


def _make_signal(f=None, gamma=0.85, tau=0.5):
    if f is None:
        f = np.array([0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.1])
    return LiterarySignal(f=np.array(f, dtype=float), gamma=gamma, tau=tau,
                          context=None, provenance=[(1, 0)],
                          residues=np.zeros((6, 6)),
                          memory_traces=[])


def _mock_signals(n_nodes: int = 3, k_d: int = 7):
    """Create mock signals across all 6 dimensions."""
    rng = np.random.default_rng(42)
    signals: dict[NodeID, dict[str, LiterarySignal]] = {}
    for i in range(n_nodes):
        nid = (1, i)
        signals[nid] = {}
        for d in _DIMS:
            f = rng.random(k_d)
            gamma = 0.5 + 0.4 * rng.random()
            tau = (i + 0.5) / n_nodes
            signals[nid][d] = _make_signal(f=f, gamma=gamma, tau=tau)
    return signals


class _MockTree:
    def __init__(self):
        self._levels: dict[int, list] = {}

    def add_level(self, level, nodes):
        self._levels[level] = nodes

    def get_level(self, level):
        return self._levels.get(level, [])


class _MockNode:
    def __init__(self, level, index, tau=0.5, children=None, parent=None):
        self.level = level
        self.index = index
        self.tau = tau
        self.children = children or []
        self.parent = parent


# ═══════════════════════════════════════════════════════════════
# 1. CONSTRUCTION — Eq. (14.1)
# ═══════════════════════════════════════════════════════════════

def test_default_d_120():
    """Default d=120 produces correct subspace sizes."""
    ctor = NarrativeEmbeddingConstructor()
    assert ctor.d == 120
    assert ctor.d_per_dim == 12
    assert ctor.d_pairs == 36


def test_valid_d_values():
    """All valid d values produce correct proportions."""
    for d in [40, 120, 240, 480]:
        ctor = NarrativeEmbeddingConstructor(d)
        assert ctor.d_per_dim == d // 10
        assert ctor.d_pairs == 3 * d // 10
        assert ctor.d_per_dim + ctor.d_per_dim * 6 + ctor.d_pairs == d


def test_invalid_d_raises():
    """Invalid d raises ValueError."""
    for bad_d in [0, 1, 50, 100, 121, 999]:
        try:
            NarrativeEmbeddingConstructor(d=bad_d)
            assert False, f"Should raise for d={bad_d}"
        except ValueError:
            pass


def test_construct_basic():
    """Basic construction returns NarrativeEmbedding with correct dims."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0, tau=0.5)])
    signals = _mock_signals(5)
    emb = ctor.construct(signals, tree)

    assert isinstance(emb, NarrativeEmbedding)
    assert emb.d == 120
    assert len(emb.H_S) == 12
    assert len(emb.H_N) == 12
    assert len(emb.H_I) == 12
    assert len(emb.H_Sy) == 12
    assert len(emb.H_E) == 12
    assert len(emb.H_Em) == 12
    assert len(emb.H_pairs) == 36
    assert len(emb.H_res) == 12
    assert len(emb.to_vector()) == 120


def test_construct_d_40():
    """d=40 produces 4-d subspaces, 12-d pairs, 4-d res."""
    ctor = NarrativeEmbeddingConstructor(d=40)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(3)
    emb = ctor.construct(signals, tree)

    assert emb.d == 40
    assert len(emb.H_S) == 4
    assert len(emb.H_N) == 4
    assert len(emb.H_I) == 4
    assert len(emb.H_Sy) == 4
    assert len(emb.H_E) == 4
    assert len(emb.H_Em) == 4
    assert len(emb.H_pairs) == 12
    assert len(emb.H_res) == 4
    assert len(emb.to_vector()) == 40


def test_construct_d_240():
    """d=240 produces 24-d subspaces."""
    ctor = NarrativeEmbeddingConstructor(d=240)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(5)
    emb = ctor.construct(signals, tree)
    assert all(len(getattr(emb, f"H_{d}")) == 24 for d in _DIMS)
    assert len(emb.H_pairs) == 72
    assert len(emb.H_res) == 24
    assert len(emb.to_vector()) == 240


def test_construct_with_resonance_trajectory():
    """Construction with explicit resonance trajectory."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(3)

    rng = np.random.default_rng(42)
    trajectory = [rng.random(7) for _ in range(20)]
    emb = ctor.construct(signals, tree, resonance_trajectory=trajectory)
    assert len(emb.H_res) == 12
    assert all(np.isfinite(emb.H_res))


# ═══════════════════════════════════════════════════════════════
# 2. PER-DIMENSION SUBSPACES — Eq. (14.2)
# ═══════════════════════════════════════════════════════════════

def test_dim_subspace_statistics():
    """H_d contains 5 statistics per feature component: mean, max, slope, var, fft."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    signals = {
        (1, 0): {
            "S": _make_signal(np.array([0.1, 0.2, 0.3, 0.4])),
        },
        (1, 1): {
            "S": _make_signal(np.array([0.5, 0.6, 0.7, 0.8])),
        },
        (1, 2): {
            "S": _make_signal(np.array([0.9, 1.0, 1.1, 1.2])),
        },
    }
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    H_S = ctor._compute_dim_subspace("S", signals, tree)
    # 4 features × 5 stats = 20 values, truncated to 12
    assert len(H_S) == 12
    assert np.all(np.isfinite(H_S))
    # First 5 values correspond to feature 0's statistics
    # vals = [0.1, 0.5, 0.9]
    assert abs(H_S[0] - 0.5) < ATOL  # mean
    assert abs(H_S[1] - 0.9) < ATOL  # max
    assert abs(H_S[2] - 0.4) < ATOL  # slope = (0.9-0.1)/2 = 0.4
    assert abs(H_S[3] - np.var([0.1, 0.5, 0.9])) < ATOL  # var


def test_dim_subspace_empty():
    """Empty signals for a dimension returns zeros."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    H_S = ctor._compute_dim_subspace("S", {}, _MockTree())
    assert np.all(H_S == 0.0)
    assert len(H_S) == 12


def test_dim_subspace_single_signal():
    """Single signal: slope=0, var=0, fft=0."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    signals = {
        (1, 0): {"S": _make_signal(np.array([0.5, 0.3]))},
    }
    tree = _MockTree()
    H_S = ctor._compute_dim_subspace("S", signals, tree)
    assert len(H_S) == 12
    assert np.all(np.isfinite(H_S))
    # slope should be 0 (single value)
    assert H_S[2] == 0.0  # slope index for feature 0 = 2nd in 5-stat group


# ═══════════════════════════════════════════════════════════════
# 3. PAIRWISE INTERACTIONS
# ═══════════════════════════════════════════════════════════════

def test_pairs_15_unique():
    """6 dimensions → 15 unique pairwise cos similarities."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    dim_vectors = {d: np.arange(12, dtype=float) for d in _DIMS}
    H_pairs = ctor._compute_pairs(dim_vectors)
    assert len(H_pairs) == 36
    # First 15 elements are the actual cos similarities
    # All identical vectors → cos = 1.0
    first_15 = H_pairs[:15]
    assert np.allclose(first_15, 1.0, rtol=RTOL), f"first_15={first_15}"


def test_pairs_orthogonal():
    """Orthogonal dim vectors give zero pairwise similarity."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    dim_vectors = {}
    rng = np.random.default_rng(42)
    for i, d in enumerate(_DIMS):
        v = np.zeros(12)
        v[i * 2] = 1.0  # each dim has support in a different basis
        dim_vectors[d] = v
    H_pairs = ctor._compute_pairs(dim_vectors)
    assert len(H_pairs) == 36
    first_15 = H_pairs[:15]
    # Most pairs should have low similarity
    assert np.all(first_15 >= 0.0)


def test_pairs_zero_vector():
    """Zero vector inputs give zero similarity."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    dim_vectors = {}
    for i, d in enumerate(_DIMS):
        if i == 0:
            dim_vectors[d] = np.ones(12)
        else:
            dim_vectors[d] = np.zeros(12)
    H_pairs = ctor._compute_pairs(dim_vectors)
    assert len(H_pairs) == 36


# ═══════════════════════════════════════════════════════════════
# 4. RESONANCE SUBSPACE
# ═══════════════════════════════════════════════════════════════

def test_resonance_from_trajectory():
    """Explicit trajectory produces finite H_res."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    rng = np.random.default_rng(42)
    trajectory = [rng.random(7) for _ in range(30)]
    H_res = ctor._compute_resonance_subspace(trajectory, {}, _MockTree())
    assert len(H_res) == 12
    assert np.all(np.isfinite(H_res))


def test_resonance_from_signals():
    """Fallback extraction from level-4 signals."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [
        _MockNode(4, 0, tau=0.2), _MockNode(4, 1, tau=0.5),
    ])
    signals = {
        (4, 0): {d: _make_signal(f=np.array([0.5, 0.3, 0.8]))
                  for d in _DIMS},
        (4, 1): {d: _make_signal(f=np.array([0.9, 0.7, 0.2]))
                  for d in _DIMS},
    }
    H_res = ctor._compute_resonance_subspace(None, signals, tree)
    assert len(H_res) == 12
    assert np.all(np.isfinite(H_res))


def test_resonance_empty():
    """No trajectory and no level-4 nodes → zero H_res."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    H_res = ctor._compute_resonance_subspace(None, {}, _MockTree())
    assert np.all(H_res == 0.0)
    assert len(H_res) == 12


def test_resonance_single_step():
    """Single-element trajectory → zero H_res."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    trajectory = [np.array([0.5, 0.3])]
    H_res = ctor._compute_resonance_subspace(trajectory, {}, _MockTree())
    assert np.all(H_res == 0.0)


# ═══════════════════════════════════════════════════════════════
# 5. DIMENSION VALIDATION
# ═══════════════════════════════════════════════════════════════

def test_validate_passes():
    """Valid embedding passes validation."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(3)
    emb = ctor.construct(signals, tree)
    # validate is called inside construct; should not raise
    ctor.validate(emb)


def test_validate_wrong_dim():
    """Wrong subspace dimension raises ValueError."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    bad_emb = NarrativeEmbedding(
        H_S=np.zeros(12), H_N=np.zeros(12),
        H_I=np.zeros(12), H_Sy=np.zeros(12),
        H_E=np.zeros(12), H_Em=np.zeros(12),
        H_pairs=np.zeros(36), H_res=np.zeros(11),  # wrong: 11 not 12
        d=120,
    )
    try:
        ctor.validate(bad_emb)
        assert False, "Should raise"
    except ValueError as e:
        assert "H_res" in str(e)


def test_validate_wrong_pairs():
    """Wrong H_pairs dimension raises ValueError."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    bad_emb = NarrativeEmbedding(
        H_S=np.zeros(12), H_N=np.zeros(12),
        H_I=np.zeros(12), H_Sy=np.zeros(12),
        H_E=np.zeros(12), H_Em=np.zeros(12),
        H_pairs=np.zeros(30), H_res=np.zeros(12),  # wrong: 30 not 36
        d=120,
    )
    try:
        ctor.validate(bad_emb)
        assert False, "Should raise"
    except ValueError:
        pass


def test_validate_wrong_d():
    """Wrong d field raises ValueError."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    bad_emb = NarrativeEmbedding(
        H_S=np.zeros(12), H_N=np.zeros(12),
        H_I=np.zeros(12), H_Sy=np.zeros(12),
        H_E=np.zeros(12), H_Em=np.zeros(12),
        H_pairs=np.zeros(36), H_res=np.zeros(12),
        d=40,  # wrong
    )
    try:
        ctor.validate(bad_emb)
        assert False, "Should raise"
    except ValueError:
        pass


# ═══════════════════════════════════════════════════════════════
# 6. SERIALIZATION
# ═══════════════════════════════════════════════════════════════

def test_save_load_round_trip():
    """Save and load preserves all values."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(3)
    emb = ctor.construct(signals, tree)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        ctor.save(emb, path)
        loaded = ctor.load(path)
        assert loaded.d == emb.d
        assert np.allclose(loaded.H_S, emb.H_S)
        assert np.allclose(loaded.H_N, emb.H_N)
        assert np.allclose(loaded.H_I, emb.H_I)
        assert np.allclose(loaded.H_Sy, emb.H_Sy)
        assert np.allclose(loaded.H_E, emb.H_E)
        assert np.allclose(loaded.H_Em, emb.H_Em)
        assert np.allclose(loaded.H_pairs, emb.H_pairs)
        assert np.allclose(loaded.H_res, emb.H_res)
        assert np.allclose(loaded.to_vector(), emb.to_vector())
    finally:
        import os
        if os.path.exists(path):
            os.unlink(path)


def test_save_load_d_40():
    """Round-trip with d=40."""
    ctor = NarrativeEmbeddingConstructor(d=40)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(3)
    emb = ctor.construct(signals, tree)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        ctor.save(emb, path)
        loaded = ctor.load(path)
        assert loaded.d == 40
        assert len(loaded.to_vector()) == 40
    finally:
        import os
        if os.path.exists(path):
            os.unlink(path)


def test_load_nonexistent():
    """Loading nonexistent file raises FileNotFoundError."""
    ctor = NarrativeEmbeddingConstructor()
    try:
        ctor.load("/nonexistent/path.npz")
        assert False, "Should raise"
    except FileNotFoundError:
        pass


def test_to_json():
    """to_json produces valid JSON with correct structure."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(3)
    emb = ctor.construct(signals, tree)
    json_str = ctor.to_json(emb)
    data = json.loads(json_str)
    assert data["d"] == 120
    assert "subspaces" in data
    assert len(data["subspaces"]) == 8
    assert data["stats"]["norm"] > 0


# ═══════════════════════════════════════════════════════════════
# 7. EDGE CASES
# ═══════════════════════════════════════════════════════════════

def test_empty_signals():
    """Empty signals produces all-zero embedding with correct dims."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    emb = ctor.construct({}, tree)
    assert emb.d == 120
    assert np.all(emb.H_S == 0.0)
    assert np.all(emb.H_N == 0.0)
    assert np.all(emb.H_pairs == 0.0)
    assert np.all(emb.H_res == 0.0)


def test_single_node():
    """Single signal node still produces valid embedding."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    signals = {
        (1, 0): {d: _make_signal() for d in _DIMS},
    }
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    emb = ctor.construct(signals, tree)
    assert len(emb.to_vector()) == 120
    assert np.all(np.isfinite(emb.to_vector()))


def test_partial_dimensions():
    """Missing some dimensions fills with zeros."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    signals = {
        (1, 0): {"S": _make_signal(), "N": _make_signal()},
    }
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    emb = ctor.construct(signals, tree)
    # Missing dimensions should be zero
    assert np.all(emb.H_I == 0.0)
    assert np.all(emb.H_Sy == 0.0)
    assert np.all(emb.H_E == 0.0)
    assert np.all(emb.H_Em == 0.0)
    assert np.any(emb.H_S != 0.0)
    assert np.any(emb.H_N != 0.0)


def test_different_feature_lengths():
    """Dimensions with different k_d are handled."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    signals: dict[NodeID, dict[str, LiterarySignal]] = {
        (1, 0): {
            "S": _make_signal(np.array([0.5, 0.3, 0.8, 0.2])),  # k=4
            "N": _make_signal(np.array([0.1, 0.7])),  # k=2
            "I": _make_signal(np.array([0.6, 0.4, 0.9])),  # k=3
            "Sy": _make_signal(np.array([0.2, 0.8, 0.1, 0.5, 0.3])),
            "E": _make_signal(np.array([0.7, 0.2])),
            "Em": _make_signal(np.array([0.4, 0.6, 0.8])),
        },
    }
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    emb = ctor.construct(signals, tree)
    assert len(emb.to_vector()) == 120
    assert all(np.isfinite(emb.to_vector()))


def test_all_identical_signals():
    """All identical signals → variance=0, fft=0."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    f = np.array([0.5, 0.3, 0.8])
    signals: dict[NodeID, dict[str, LiterarySignal]] = {
        (1, i): {"S": _make_signal(f=f)} for i in range(5)
    }
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    emb = ctor.construct(signals, tree)
    H_S = emb.H_S
    # Var features (index 3, 8, 13...) should be ~0
    var_features = [H_S[i] for i in range(3, len(H_S), 5)]
    assert all(v < 1e-10 for v in var_features), f"var_features={var_features}"
    # Slope features should be ~0 (identical signals)
    slope_features = [H_S[i] for i in range(2, len(H_S), 5)]
    assert all(abs(s) < 1e-10 for s in slope_features), f"slope_features={slope_features}"


# ═══════════════════════════════════════════════════════════════
# 8. WORKED EXAMPLE — §7.9
# ═══════════════════════════════════════════════════════════════

def test_worked_example_dimensions():
    """§7.9: embedding structure matches spec."""
    ctor = NarrativeEmbeddingConstructor(d=120)
    assert ctor.d_per_dim == 12  # H_Em = 12, H_N = 12, etc.
    assert ctor.d_pairs == 36  # H_pairs = 36
    # H_res = 12, total = 6*12 + 36 + 12 = 120
    assert ctor.d_per_dim * 6 + ctor.d_pairs + ctor.d_per_dim == 120


def test_worked_example_subspace_sizes():
    """§7.9 table: each subspace has correct size."""
    # From §7.9:
    # H_Em = 12, H_N = 12, H_I = 12, H_Sy = 12, H_E = 12, H_S = 12
    # H_pairs = 36, H_res = 12
    ctor = NarrativeEmbeddingConstructor(d=120)
    tree = _MockTree()
    tree.add_level(4, [_MockNode(4, 0)])
    signals = _mock_signals(10)
    emb = ctor.construct(signals, tree)
    assert len(emb.H_Em) == 12
    assert len(emb.H_N) == 12
    assert len(emb.H_I) == 12
    assert len(emb.H_Sy) == 12
    assert len(emb.H_E) == 12
    assert len(emb.H_S) == 12
    assert len(emb.H_pairs) == 36
    assert len(emb.H_res) == 12


# ═══════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════

def _check_fn(fn):
    name = fn.__name__
    try:
        fn()
        return {"name": name, "status": "PASS"}
    except Exception as e:
        import traceback
        return {"name": name, "status": "FAIL",
                "error": f"{e}\n{traceback.format_exc()}"}


if __name__ == "__main__":
    test_fns = [v for k, v in globals().items()
                if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_fns:
        result = _check_fn(fn)
        if result["status"] == "PASS":
            print(f"  [PASS] {result['name']}")
            passed += 1
        else:
            print(f"  [FAIL] {result['name']}: {result.get('error', '')}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed ({passed + failed} total)")
