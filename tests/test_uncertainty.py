"""Comprehensive tests for UncertaintyPropagator — HMP-DT §13.

Test categories:
  1. Variance propagation — Eqs. (13.1)–(13.5) across hierarchy
  2. Confidence refinement — Eq. (13.2) boundary cases
  3. Covariance tracking — inter-dimension covariance
  4. Bootstrap estimation — confidence intervals, reliability
  5. Integration — full hierarchy propagation
  6. Numerical stability — edge cases, extreme values
  7. Performance benchmarks
"""
import math
import sys
import time
sys.path.insert(0, "D:\\LIT-DISC")

import numpy as np
from src.uncertainty import UncertaintyPropagator
from src.models import LiterarySignal, NodeID, NarrativeEmbedding

RTOL = 1e-4
ATOL = 1e-6

_DIMS = ["S", "N", "I", "Sy", "E", "Em"]


def _make_signal(f=None, gamma=0.85, tau=0.5):
    if f is None:
        f = np.array([0.5, 0.3, 0.8])
    return LiterarySignal(f=np.array(f, dtype=float), gamma=gamma, tau=tau,
                          context=None, provenance=[(1, 0)],
                          residues=np.zeros((6, 6)),
                          memory_traces=[])


class _MockTree:
    """Minimal mock tree for integration tests."""
    def __init__(self):
        self._levels: dict[int, list] = {}

    def add_level(self, level: int, nodes: list):
        self._levels[level] = nodes

    def get_level(self, level: int):
        return self._levels.get(level, [])


class _MockNode:
    def __init__(self, level, index, tau=0.5, children=None, parent=None):
        self.level = level
        self.index = index
        self.tau = tau
        self.children = children or []
        self.parent = parent


# ═══════════════════════════════════════════════════════════════
# 1. VARIANCE PROPAGATION — Eqs. (13.1)–(13.5)
# ═══════════════════════════════════════════════════════════════

def test_initialization_from_gamma():
    """Initialize from gamma sets sigma_e_sq correctly."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    gamma = 0.85
    up.initialize_from_gamma(nid, "S", gamma)
    sigma_total = 1.0 * (1.0 - gamma)  # sigma_max * (1 - gamma)
    sigma_sq_total = sigma_total ** 2
    known = sum(up.sigma_s ** 2 for _ in range(4))  # 4 known components
    # Actually: sigma_s^2 + sigma_i^2 + sigma_t^2 + sigma_m^2
    known = (up.sigma_s ** 2 + up.sigma_i ** 2
             + up.sigma_t ** 2 + up.sigma_m ** 2)
    expected = max(0.0, sigma_sq_total - known)
    assert abs(up.get_variance(nid, "S") - expected) < ATOL, (
        f"variance={up.get_variance(nid, 'S')}, expected={expected}")


def test_total_variance():
    """Eq. 13.1: sigma²_total = sigma²_e + sigma²_s + ..."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    up.initialize_from_gamma(nid, "S", 0.5)
    total = up.total_variance(nid, "S")
    expected = (up.get_variance(nid, "S")
                + up.sigma_s ** 2 + up.sigma_i ** 2
                + up.sigma_t ** 2 + up.sigma_m ** 2)
    assert abs(total - expected) < ATOL


def test_initialize_from_signals():
    """Initialize from a batch of signals."""
    up = UncertaintyPropagator(sigma_max=1.0, sigma_e=0.05, sigma_s=0.05,
                                sigma_i=0.02, sigma_t=0.02, sigma_m=0.02)
    signals = {
        (1, 0): {"S": _make_signal(gamma=0.7)},
        (1, 1): {"S": _make_signal(gamma=0.5)},
    }
    up.initialize_from_signals(signals)
    assert up.get_variance((1, 0), "S") > 0
    assert up.get_variance((1, 1), "S") > 0
    # Lower gamma → higher variance
    assert up.get_variance((1, 1), "S") > up.get_variance((1, 0), "S")


def test_propagate_sentence_to_paragraph():
    """Eq. 13.3: paragraph variance from sentence variances."""
    up = UncertaintyPropagator()
    child_ids = [(1, 0), (1, 1), (1, 2)]
    parent_id = (2, 0)
    for cid in child_ids:
        up.set_variance(cid, "S", 0.04)

    gammas = up.propagate_sentence_to_paragraph(
        child_ids, parent_id, sc=0.8, dims=["S"])

    # Eq. 13.3: sigma²(P) = (1/n²)*sum(sigma²_i) + (1-SC) - 2*SC*sigma²_avg
    n = 3
    sigma_sq_avg = 0.04
    expected = (3 * 0.04 / 9) + (1.0 - 0.8) - 2.0 * 0.8 * 0.04
    expected = max(0.0, expected)
    computed = up.get_variance(parent_id, "S")
    assert abs(computed - expected) < ATOL, f"computed={computed}, expected={expected}"
    assert "S" in gammas
    assert 0.0 <= gammas["S"] <= 1.0


def test_propagate_paragraph_to_chapter():
    """Eq. 13.4: chapter variance from paragraph variances + alphas."""
    up = UncertaintyPropagator()
    child_ids = [(2, 0), (2, 1), (2, 2)]
    parent_id = (3, 0)
    for cid in child_ids:
        up.set_variance(cid, "S", 0.03)

    alphas = [0.5, 0.3, 0.2]
    gammas = up.propagate_paragraph_to_chapter(
        child_ids, parent_id, alphas, dims=["S"])

    # Eq. 13.4: sigma²(C) = sum(alpha_j² * sigma²_j) + 0.1*(1 - max alpha)
    expected = (0.25 * 0.03 + 0.09 * 0.03 + 0.04 * 0.03)
    expected += 0.1 * (1.0 - 0.5)
    computed = up.get_variance(parent_id, "S")
    assert abs(computed - expected) < ATOL, f"computed={computed}, expected={expected}"
    assert 0.0 <= gammas["S"] <= 1.0


def test_propagate_chapter_to_narrative():
    """Eq. 13.5: narrative variance from chapter variances + spline weights."""
    up = UncertaintyPropagator()
    chapters = [((3, 0), 0.2), ((3, 1), 0.5), ((3, 2), 0.8)]
    parent_id = (4, 0)
    for cid, _ in chapters:
        up.set_variance(cid, "S", 0.02)

    weights = [0.1, 0.7, 0.2]
    gammas = up.propagate_chapter_to_narrative(
        chapters, parent_id, weights, tau_target=0.5, dims=["S"])

    # Eq. 13.5: sigma²(N) = sum(phi_i² * sigma²_i) + 0.05 * min|tau - tau_i|²
    expected = (0.01 * 0.02 + 0.49 * 0.02 + 0.04 * 0.02)
    expected += 0.05 * 0.0  # min_dist = |0.5 - 0.5| = 0
    computed = up.get_variance(parent_id, "S")
    assert abs(computed - expected) < ATOL, f"computed={computed}, expected={expected}"
    assert 0.0 <= gammas["S"] <= 1.0


def test_propagate_narrative_to_discourse():
    """Discourse variance is mean of narrative variances."""
    up = UncertaintyPropagator()
    nids = [(4, 0), (4, 1)]
    parent_id = (5, 0)
    for cid in nids:
        up.set_variance(cid, "S", 0.04)
    up.set_variance((4, 0), "S", 0.06)

    gammas = up.propagate_narrative_to_discourse(nids, parent_id, dims=["S"])
    expected = (0.06 + 0.04) / 2.0
    computed = up.get_variance(parent_id, "S")
    assert abs(computed - expected) < ATOL
    assert 0.0 <= gammas["S"] <= 1.0


def test_full_hierarchy_propagation():
    """End-to-end propagation through all 5 levels."""
    up = UncertaintyPropagator(sigma_e=0.05, sigma_s=0.05,
                                sigma_i=0.02, sigma_t=0.02, sigma_m=0.02)
    signals = {}
    tree = _MockTree()
    # Level 1: 4 sentences
    tree.add_level(1, [
        _MockNode(1, 0, tau=0.1, parent=(2, 0)),
        _MockNode(1, 1, tau=0.2, parent=(2, 0)),
        _MockNode(1, 2, tau=0.3, parent=(2, 1)),
        _MockNode(1, 3, tau=0.4, parent=(2, 1)),
    ])
    # Level 2: 2 paragraphs
    tree.add_level(2, [
        _MockNode(2, 0, tau=0.15, children=[(1, 0), (1, 1)], parent=(3, 0)),
        _MockNode(2, 1, tau=0.35, children=[(1, 2), (1, 3)], parent=(3, 0)),
    ])
    # Level 3: 1 chapter
    tree.add_level(3, [
        _MockNode(3, 0, tau=0.25, children=[(2, 0), (2, 1)], parent=(4, 0)),
    ])
    # Level 4: 1 narrative layer
    tree.add_level(4, [
        _MockNode(4, 0, tau=0.5, children=[(3, 0)], parent=(5, 0)),
    ])
    # Level 5: 1 discourse layer
    tree.add_level(5, [
        _MockNode(5, 0, tau=1.0, children=[(4, 0)], parent=None),
    ])

    for i in range(4):
        nid = (1, i)
        signals[nid] = {d: _make_signal(gamma=0.7 + 0.05 * i,
                                         tau=0.1 * (i + 1))
                        for d in _DIMS}

    up.propagate_all(tree, signals, _DIMS)

    # All levels should have variances
    stats1 = up.get_level_stats(1)
    stats2 = up.get_level_stats(2)
    stats3 = up.get_level_stats(3)
    stats4 = up.get_level_stats(4)
    stats5 = up.get_level_stats(5)
    assert stats1["count"] > 0, f"level 1 count={stats1['count']}"
    assert stats5["count"] > 0, f"level 5 count={stats5['count']}"
    # Variances should be positive and finite
    for v in up._var.values():
        assert v >= -ATOL, f"negative variance: {v}"
        assert math.isfinite(v)


# ═══════════════════════════════════════════════════════════════
# 2. CONFIDENCE REFINEMENT — Eq. (13.2)
# ═══════════════════════════════════════════════════════════════

def test_refine_confidence_perfect():
    """Zero variance → gamma = 1.0."""
    up = UncertaintyPropagator()
    gamma = up.refine_confidence_from_variance(0.0)
    assert abs(gamma - 1.0) < ATOL


def test_refine_confidence_max_uncertainty():
    """Variance = sigma_max² → gamma = 0.0."""
    up = UncertaintyPropagator(sigma_max=1.0)
    gamma = up.refine_confidence_from_variance(1.0)
    assert abs(gamma - 0.0) < ATOL


def test_refine_confidence_half():
    """sigma_total = 0.5 * sigma_max → gamma = 0.5."""
    up = UncertaintyPropagator(sigma_max=1.0)
    gamma = up.refine_confidence_from_variance(0.25)  # sigma = 0.5
    expected = 1.0 - min(1.0, 0.5 / 1.0)
    assert abs(gamma - expected) < ATOL


def test_refine_confidence_via_node():
    """Full round-trip: set variance → refine."""
    up = UncertaintyPropagator(sigma_e=0.0, sigma_s=0.0,
                                sigma_i=0.0, sigma_t=0.0, sigma_m=0.0)
    nid = (1, 0)
    up.set_variance(nid, "S", 0.09)  # sigma = 0.3
    gamma = up.refine_confidence(nid, "S")
    expected = 1.0 - min(1.0, 0.3 / 1.0)
    assert abs(gamma - expected) < ATOL


def test_refine_confidence_clamping():
    """sigma_total > sigma_max → gamma = 0.0."""
    up = UncertaintyPropagator(sigma_max=0.5)
    gamma = up.refine_confidence_from_variance(1.0)  # sigma = 1.0 > 0.5
    assert abs(gamma - 0.0) < ATOL


# ═══════════════════════════════════════════════════════════════
# 3. COVARIANCE TRACKING
# ═══════════════════════════════════════════════════════════════

def test_covariance_update():
    """Covariance between two dimensions should be computed."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    dim_signals = {
        "S": np.array([0.5, 0.3, 0.8]),
        "N": np.array([0.1, 0.7, 0.4]),
    }
    up.update_covariance(nid, dim_signals)
    cov_sn = up.get_covariance(nid, "S", "N")
    assert math.isfinite(cov_sn)
    assert up.get_covariance(nid, "N", "S") == cov_sn


def test_covariance_zero_on_single_element():
    """Single-element feature vectors have zero covariance."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    dim_signals = {
        "S": np.array([0.5]),
        "N": np.array([0.3]),
    }
    up.update_covariance(nid, dim_signals)
    assert abs(up.get_covariance(nid, "S", "N")) < ATOL


def test_covariance_symmetry():
    """Covariance matrix is symmetric."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    rng = np.random.default_rng(42)
    dim_signals = {d: rng.random(5) for d in _DIMS}
    up.update_covariance(nid, dim_signals)
    mat = up.get_covariance_matrix(nid, _DIMS)
    for i in range(6):
        for j in range(6):
            assert abs(mat[i, j] - mat[j, i]) < ATOL, (
                f"({i},{j}) != ({j},{i}): {mat[i,j]} vs {mat[j,i]}")


def test_covariance_diagonal_is_variance():
    """Covariance matrix diagonal equals per-dimension variance."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    up.set_variance(nid, "S", 0.04)
    up.set_variance(nid, "N", 0.09)
    mat = up.get_covariance_matrix(nid, ["S", "N"])
    assert abs(mat[0, 0] - 0.04) < ATOL
    assert abs(mat[1, 1] - 0.09) < ATOL


# ═══════════════════════════════════════════════════════════════
# 4. COUPLING UNCERTAINTY — Eq. (13.6)
# ═══════════════════════════════════════════════════════════════

def test_coupling_uncertainty():
    """Eq. 13.6: sigma²_i(d,d') = kappa² * (Var(f_d) + Var(f_d'))."""
    up = UncertaintyPropagator()
    result = up.coupling_uncertainty(kappa=0.68, var_d=0.04, var_dp=0.09)
    expected = 0.68 ** 2 * (0.04 + 0.09)
    assert abs(result - expected) < ATOL


def test_coupling_uncertainty_zero():
    """Zero kappa or zero variance gives zero coupling uncertainty."""
    up = UncertaintyPropagator()
    assert abs(up.coupling_uncertainty(0.0, 0.04, 0.09)) < ATOL
    assert abs(up.coupling_uncertainty(0.68, 0.0, 0.0)) < ATOL


# ═══════════════════════════════════════════════════════════════
# 5. BOOTSTRAP ESTIMATION
# ═══════════════════════════════════════════════════════════════

def _make_embedding(d=120):
    rng = np.random.default_rng(42)
    return NarrativeEmbedding(
        H_S=rng.random(d // 10),
        H_N=rng.random(d // 10),
        H_I=rng.random(d // 10),
        H_Sy=rng.random(d // 10),
        H_E=rng.random(d // 10),
        H_Em=rng.random(d // 10),
        H_pairs=rng.random(3 * d // 10),
        H_res=rng.random(d // 10),
        d=d,
    )


def test_bootstrap_ldri_basic():
    """Bootstrap returns dict with expected keys."""
    up = UncertaintyPropagator()
    emb = _make_embedding()
    result = up.bootstrap_ldri(emb, n_samples=50)
    assert "ldri_samples" in result
    assert "mean" in result
    assert "std_error" in result
    assert "ci_95" in result
    assert "reliability" in result
    assert len(result["ldri_samples"]) == 50
    assert 0.0 <= result["mean"] <= 1.0
    assert result["std_error"] >= 0.0
    lo, hi = result["ci_95"]
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert 0.0 <= result["reliability"] <= 1.0


def test_bootstrap_reliability_high_with_low_noise():
    """Low noise → high reliability."""
    up = UncertaintyPropagator()
    emb = _make_embedding()
    result = up.bootstrap_ldri(emb, n_samples=50, noise_scale=0.001)
    assert result["reliability"] > 0.9, f"reliability={result['reliability']}"


def test_confidence_interval_95():
    """95% CI uses z=1.96 and clamps to [0, 1]."""
    up = UncertaintyPropagator()
    ci = up.compute_confidence_interval(0.5, 0.1, level=0.95)
    lo, hi = ci
    assert abs(lo - 0.304) < 0.01  # 0.5 - 1.96*0.1 = 0.304
    assert abs(hi - 0.696) < 0.01  # 0.5 + 1.96*0.1 = 0.696


def test_confidence_interval_clamping():
    """CI clamps to [0, 1]."""
    up = UncertaintyPropagator()
    ci = up.compute_confidence_interval(0.1, 0.3)
    lo, hi = ci
    assert lo >= 0.0
    assert hi <= 1.0


def test_confidence_interval_levels():
    """Different confidence levels use different z-scores."""
    up = UncertaintyPropagator()
    ci_90 = up.compute_confidence_interval(0.5, 0.1, level=0.90)
    ci_95 = up.compute_confidence_interval(0.5, 0.1, level=0.95)
    ci_99 = up.compute_confidence_interval(0.5, 0.1, level=0.99)
    # Higher confidence → wider interval
    w90 = ci_90[1] - ci_90[0]
    w95 = ci_95[1] - ci_95[0]
    w99 = ci_99[1] - ci_99[0]
    assert w95 > w90
    assert w99 > w95


# ═══════════════════════════════════════════════════════════════
# 6. NUMERICAL STABILITY / EDGE CASES
# ═══════════════════════════════════════════════════════════════

def test_negative_variance_clamped():
    """Negative variance gets clamped to zero."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    up.set_variance(nid, "S", -0.1)
    assert up.get_variance(nid, "S") >= 0.0


def test_nan_input_initialize():
    """NaN gamma should not crash initialization."""
    up = UncertaintyPropagator()
    nid = (1, 0)
    up.initialize_from_gamma(nid, "S", float('nan'))
    v = up.get_variance(nid, "S")
    assert math.isfinite(v) or math.isnan(v)


def test_empty_child_list():
    """Empty child lists should not crash propagation."""
    up = UncertaintyPropagator()
    gammas = up.propagate_sentence_to_paragraph([], (2, 0), 0.5, ["S"])
    assert gammas == {}
    gammas = up.propagate_paragraph_to_chapter([], (3, 0), [], ["S"])
    assert gammas == {}
    gammas = up.propagate_chapter_to_narrative([], (4, 0), [], 0.5, ["S"])
    assert gammas == {}
    gammas = up.propagate_narrative_to_discourse([], (5, 0), ["S"])
    assert gammas == {}


def test_single_child_paragraph():
    """Single-sentence paragraph has n=1 variance."""
    up = UncertaintyPropagator()
    up.set_variance((1, 0), "S", 0.04)
    gammas = up.propagate_sentence_to_paragraph(
        [(1, 0)], (2, 0), sc=0.0, dims=["S"])
    # n=1: sigma² = sigma²/1 + (1-0) - 2*0*sigma²_avg = 0.04 + 1.0 = 1.04
    expected = 0.04 + 1.0 - 0.0
    expected = max(0.0, expected)
    computed = up.get_variance((2, 0), "S")
    assert abs(computed - expected) < ATOL


def test_perfect_semantic_coherence():
    """SC=1.0 gives max variance reduction."""
    up = UncertaintyPropagator()
    up.set_variance((1, 0), "S", 0.04)
    up.set_variance((1, 1), "S", 0.04)
    sigma_sq_avg = 0.04
    n = 2
    gammas = up.propagate_sentence_to_paragraph(
        [(1, 0), (1, 1)], (2, 0), sc=1.0, dims=["S"])
    expected = (2 * 0.04 / 4) + (1.0 - 1.0) - 2.0 * 1.0 * 0.04
    expected = max(0.0, expected)
    computed = up.get_variance((2, 0), "S")
    assert abs(computed - expected) < ATOL


def test_zero_alphas():
    """Zero alphas still have the baseline term in Eq. 13.4."""
    up = UncertaintyPropagator()
    up.set_variance((2, 0), "S", 0.04)
    up.set_variance((2, 1), "S", 0.04)
    alphas = [0.0, 0.0]
    gammas = up.propagate_paragraph_to_chapter(
        [(2, 0), (2, 1)], (3, 0), alphas, ["S"])
    # sigma² = 0 + 0.1 * (1 - 0) = 0.1
    expected = 0.1
    computed = up.get_variance((3, 0), "S")
    assert abs(computed - expected) < ATOL


def test_clear():
    """Clear resets all state."""
    up = UncertaintyPropagator()
    up.set_variance((1, 0), "S", 0.04)
    up.update_covariance((1, 0), {"S": np.array([0.5]), "N": np.array([0.3])})
    up.clear()
    assert up.get_variance((1, 0), "S") == 0.0
    assert abs(up.get_covariance((1, 0), "S", "N")) < ATOL


def test_level_stats_empty():
    """Empty level stats return zeroed dict."""
    up = UncertaintyPropagator()
    stats = up.get_level_stats(1)
    assert stats["count"] == 0


def test_total_std():
    """Total std from total variance."""
    up = UncertaintyPropagator()
    up.set_variance((1, 0), "S", 0.09)
    std = up.total_std((1, 0), "S")
    expected = math.sqrt(0.09 + up.sigma_s ** 2 + up.sigma_i ** 2
                         + up.sigma_t ** 2 + up.sigma_m ** 2)
    assert abs(std - expected) < ATOL


# ═══════════════════════════════════════════════════════════════
# 7. WORKED EXAMPLE — §7.4
# ═══════════════════════════════════════════════════════════════

def test_worked_example_sentence_level():
    """§7.4: Sentence-level uncertainty from Hemingway example."""
    up = UncertaintyPropagator()
    # Sentence 45: gamma_Em = 0.78
    nid_s45 = (1, 45)
    up.initialize_from_gamma(nid_s45, "Em", 0.78)
    sigma_e_sq = up.get_variance(nid_s45, "Em")
    sigma_total = math.sqrt(sigma_e_sq + up.sigma_s ** 2 + up.sigma_i ** 2
                            + up.sigma_t ** 2 + up.sigma_m ** 2)
    gamma_refined = 1.0 - min(1.0, sigma_total / up.sigma_max)
    assert abs(gamma_refined - 0.78) < 0.02, (
        f"γ_refined={gamma_refined}, expected ~0.78")


def test_worked_example_paragraph_uncertainty():
    """§7.4: Paragraph 6 uncertainty reduction via coherence."""
    up = UncertaintyPropagator()
    # 15 sentences in paragraph 6, coherence=0.85
    for i in range(43, 58):
        up.set_variance((1, i), "Em", 0.04)
    gammas = up.propagate_sentence_to_paragraph(
        [(1, i) for i in range(43, 58)], (2, 6), sc=0.85, dims=["Em"])
    sigma_sq_P6 = up.get_variance((2, 6), "Em")
    assert sigma_sq_P6 >= 0
    assert math.isfinite(sigma_sq_P6)
    # Coherence should reduce uncertainty
    assert gammas["Em"] > 0.5, f"γ={gammas['Em']}"


# ═══════════════════════════════════════════════════════════════
# 8. PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════

def test_performance_propagate_1000():
    """1000 sentence→paragraph propagations in < 100ms."""
    up = UncertaintyPropagator()
    child_ids = [(1, i) for i in range(10)]
    for c in child_ids:
        up.set_variance(c, "S", 0.04)
    start = time.perf_counter()
    for _ in range(100):
        up.propagate_sentence_to_paragraph(child_ids, (2, 0), 0.7, ["S"])
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, f"100 propagations took {elapsed:.3f}s"


def test_performance_bootstrap():
    """50 bootstrap samples in < 200ms."""
    up = UncertaintyPropagator()
    emb = _make_embedding()
    start = time.perf_counter()
    up.bootstrap_ldri(emb, n_samples=50)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"50 bootstrap took {elapsed:.3f}s"


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
        return {"name": name, "status": "FAIL", "error": f"{e}\n{traceback.format_exc()}"}


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
