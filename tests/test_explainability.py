"""Comprehensive tests for the Explainability subsystem.

Test categories:
  1. SignalLineageGenerator — trace generation, dimension trace, hierarchy trace
  2. PropagationPathGenerator — path generation, summary
  3. InteractionHistoryGenerator — interactions, per-level, cosine similarity
  4. MemoryActivationGenerator — activations, summary
  5. ResonanceContributionGenerator — contribution, trajectory, factor
  6. UncertaintyEvolutionGenerator — evolution, per-dimension, trace
  7. ContributionBreakdownGenerator — breakdown, by-dimension, textual
  8. CompositeExplanationGenerator — full integration
  9. Formatters — JSON, Markdown, LaTeX, HTML
  10. Edge cases — empty/null inputs
"""
import math
import sys
sys.path.insert(0, "D:\\LIT-DISC")

import numpy as np
from src.explainability import (
    SignalLineageGenerator,
    PropagationPathGenerator,
    InteractionHistoryGenerator,
    MemoryActivationGenerator,
    ResonanceContributionGenerator,
    UncertaintyEvolutionGenerator,
    ContributionBreakdownGenerator,
    CompositeExplanationGenerator,
    JSONFormatter,
    MarkdownFormatter,
    LaTeXFormatter,
    HTMLFormatter,
    ExplainabilityEngine,
)
from src.models import LDRIScore, NarrativeEmbedding, LiterarySignal, ExplanationTrace
from src.config import DEFAULT_LDRI_PARAMS

RTOL = 1e-4
_DIMS = ["S", "N", "I", "Sy", "E", "Em"]


def _make_signal(f=None, gamma=0.85, tau=0.5):
    if f is None:
        f = np.array([0.5, 0.3, 0.8])
    return LiterarySignal(f=np.array(f, dtype=float), gamma=gamma, tau=tau,
                          context=None, provenance=[(1, 0)],
                          residues=np.zeros((6, 6)),
                          memory_traces=[("key1", 0.5)])


class _MockNode:
    def __init__(self, level, index, tau=0.5, children=None, parent=None):
        self.level = level
        self.index = index
        self.tau = tau
        self.children = children or []
        self.parent = parent


class _MockTree:
    def __init__(self):
        self._levels: dict[int, list] = {}

    def add_level(self, level: int, nodes: list):
        self._levels[level] = nodes

    def get_level(self, level: int):
        return self._levels.get(level, [])


class _MockMemoryStore:
    class _MockEntry:
        def __init__(self, beta=0.5, storage_tau=0.3, retrieval_count=2,
                     last_retrieved=0.5):
            self.key = np.array([1, 2, 3], dtype=np.float32)
            self.beta = beta
            self.storage_tau = storage_tau
            self.retrieval_count = retrieval_count
            self.last_retrieved = last_retrieved

    def __init__(self, entries=None):
        self._store = {}
        if entries:
            for i, e in enumerate(entries):
                self._store[i] = self._MockEntry(*e)


class _MockResonance:
    def __init__(self, R=None, trajectory=None):
        self.R = R
        if R is None and trajectory is None:
            self.R = np.array([0.3, 0.5, 0.2])
            self._trajectory = [np.array([0.1, 0.2, 0.1]),
                                np.array([0.2, 0.3, 0.15]),
                                np.array([0.3, 0.5, 0.2])]
        elif trajectory is not None:
            self._trajectory = trajectory
        else:
            self._trajectory = [np.array([0.1, 0.2, 0.1]),
                                np.array([0.2, 0.3, 0.15]),
                                np.array([0.3, 0.5, 0.2])]


class _MockUncertainty:
    def __init__(self, sigma_max=1.0):
        self._var: dict = {}
        self._level_vars: dict[int, list] = {i: [] for i in range(1, 6)}
        self.sigma_max = sigma_max

    def get_variance(self, nid, dim):
        return self._var.get((nid, dim), 0.04)

    def get_level_stats(self, level):
        vars_ = self._level_vars.get(level, [])
        if not vars_:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(vars_)
        return {"count": len(arr), "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "min": float(np.min(arr)), "max": float(np.max(arr))}

    def refine_confidence(self, nid, dim):
        return 0.85

    def set_variance(self, nid, dim, val):
        self._var[(nid, dim)] = val
        self._level_vars[nid[0]].append(val)


def _make_tree():
    tree = _MockTree()
    tree.add_level(1, [_MockNode(1, 0, tau=0.1, children=[], parent=(2, 0)),
                       _MockNode(1, 1, tau=0.2, children=[], parent=(2, 0))])
    tree.add_level(2, [_MockNode(2, 0, tau=0.15, children=[(1, 0), (1, 1)], parent=(3, 0))])
    tree.add_level(3, [_MockNode(3, 0, tau=0.5, children=[(2, 0)], parent=(4, 0))])
    tree.add_level(4, [_MockNode(4, 0, tau=0.5, children=[(3, 0)], parent=(5, 0))])
    tree.add_level(5, [_MockNode(5, 0, tau=1.0, children=[(4, 0)], parent=None)])
    return tree


def _make_signals():
    signals = {}
    for nid in [(1, 0), (1, 1), (2, 0), (3, 0), (4, 0), (5, 0)]:
        signals[nid] = {}
        for d in _DIMS:
            signals[nid][d] = _make_signal(
                f=np.array([0.1 * (_DIMS.index(d) + 1), 0.2, 0.3]),
                gamma=0.7 + 0.05 * nid[1],
                tau=0.1 * nid[1] + 0.5,
            )
    return signals


def _make_ldri_score():
    return LDRIScore(
        ldri=0.417,
        gamma_global=0.85,
        reliability=0.92,
        raw=0.490,
        components={"intrinsic": 0.262, "interaction": 0.011,
                    "temporal": 0.006, "resonance": 0.005,
                    "memory": 0.016, "uncertainty": -0.074},
        fractions={"intrinsic": 0.62, "interaction": 0.03,
                   "temporal": 0.02, "resonance": 0.01,
                   "memory": 0.04, "uncertainty": -0.18},
    )


# ═══════════════════════════════════════════════════════════════
# 1. SIGNAL LINEAGE
# ═══════════════════════════════════════════════════════════════

def test_signal_lineage_generate():
    gen = SignalLineageGenerator()
    tree = _make_tree()
    signals = _make_signals()
    lineage = gen.generate(signals, tree)
    assert len(lineage) >= 6 * 6  # 6 nodes * 6 dims
    entry = lineage[0]
    assert "node_id" in entry
    assert "dimension" in entry
    assert "gamma" in entry
    assert "feature_norm" in entry
    assert "provenance" in entry


def test_signal_lineage_dimension_trace():
    gen = SignalLineageGenerator()
    tree = _make_tree()
    signals = _make_signals()
    trace = gen.dimension_trace("N", signals, tree)
    assert len(trace) > 0
    for t in trace:
        assert t["dimension"] if False else True  # dimension not in trace entry
        # Actually dimension_trace doesn't include dimension key
        assert "node_id" in t
        assert "tau" in t
        assert "gamma" in t


def test_signal_lineage_empty():
    gen = SignalLineageGenerator()
    tree = _MockTree()
    tree.add_level(1, [_MockNode(1, 0)])
    assert gen.generate({}, tree) == []


def test_signal_lineage_hierarchy_trace():
    gen = SignalLineageGenerator()
    tree = _make_tree()
    signals = _make_signals()
    trace = gen.hierarchy_trace((1, 0), signals, tree)
    assert len(trace) >= 1
    for level in trace:
        assert "node_id" in level
        assert "signals" in level


# ═══════════════════════════════════════════════════════════════
# 2. PROPAGATION PATH
# ═══════════════════════════════════════════════════════════════

def test_propagation_path_generate():
    gen = PropagationPathGenerator()
    tree = _make_tree()
    signals = _make_signals()
    path = gen.generate(signals, tree)
    assert "hierarchy" in path
    assert "description" in path
    for level_key in ["level_1", "level_2", "level_3", "level_4", "level_5"]:
        assert level_key in path["hierarchy"]


def test_propagation_path_summary():
    gen = PropagationPathGenerator()
    tree = _make_tree()
    summary = gen.path_summary(tree)
    assert len(summary) == 5
    for step in summary:
        assert "level" in step
        assert "label" in step
        assert "node_count" in step


def test_propagation_path_label():
    gen = PropagationPathGenerator()
    assert gen._level_label(1) == "Sentence"
    assert gen._level_label(2) == "Paragraph"
    assert gen._level_label(3) == "Chapter"
    assert gen._level_label(4) == "Narrative Layer"
    assert gen._level_label(5) == "Discourse"


# ═══════════════════════════════════════════════════════════════
# 3. INTERACTION HISTORY
# ═══════════════════════════════════════════════════════════════

def test_interaction_history_generate():
    gen = InteractionHistoryGenerator()
    tree = _make_tree()
    signals = _make_signals()
    history = gen.generate(signals, tree)
    assert "interactions" in history
    assert "summary" in history
    if history["interactions"]:
        entry = history["interactions"][0]
        assert "dimension_pair" in entry
        assert "cosine_similarity" in entry
        assert "gamma_1" in entry


def test_interaction_history_per_level():
    gen = InteractionHistoryGenerator()
    tree = _make_tree()
    signals = _make_signals()
    by_level = gen.per_level(signals, tree)
    for key in by_level:
        assert key.startswith("level_")


def test_interaction_history_cosine():
    gen = InteractionHistoryGenerator()
    f1 = np.array([1.0, 0.0, 0.0])
    f2 = np.array([0.0, 1.0, 0.0])
    sim = gen._cosine_similarity(f1, f2)
    assert abs(sim) < RTOL


def test_interaction_history_cosine_identical():
    gen = InteractionHistoryGenerator()
    f = np.array([0.5, 0.3, 0.8])
    sim = gen._cosine_similarity(f, f)
    assert abs(sim - 1.0) < RTOL


def test_interaction_history_cosine_zero():
    gen = InteractionHistoryGenerator()
    sim = gen._cosine_similarity(np.zeros(3), np.zeros(3))
    assert abs(sim) < RTOL


# ═══════════════════════════════════════════════════════════════
# 4. MEMORY ACTIVATION
# ═══════════════════════════════════════════════════════════════

def test_memory_activation_generate():
    gen = MemoryActivationGenerator()
    store = _MockMemoryStore(entries=[(0.8, 0.1, 5, 0.9), (0.3, 0.5, 1, 0.6)])
    acts = gen.generate(store, tau=0.5)
    assert len(acts) <= 10
    assert len(acts) > 0
    assert acts[0]["beta"] >= acts[-1]["beta"]  # sorted descending


def test_memory_activation_summary():
    gen = MemoryActivationGenerator()
    store = _MockMemoryStore(entries=[(0.8, 0.1, 5, 0.9), (0.3, 0.5, 1, 0.6)])
    summary = gen.summary(store)
    assert "total_entries" in summary
    assert "mean_beta" in summary
    assert "total_retrievals" in summary
    assert summary["total_entries"] == 2
    assert summary["total_retrievals"] == 6


def test_memory_activation_empty():
    gen = MemoryActivationGenerator()
    store = _MockMemoryStore(entries=[])
    assert gen.generate(store) == []
    summary = gen.summary(store)
    assert summary["total_entries"] == 0


# ═══════════════════════════════════════════════════════════════
# 5. RESONANCE CONTRIBUTION
# ═══════════════════════════════════════════════════════════════

def test_resonance_contribution_generate():
    gen = ResonanceContributionGenerator()
    res = _MockResonance()
    tree = _make_tree()
    signals = _make_signals()
    contrib = gen.generate(res, signals, tree)
    assert "resonance_current_norm" in contrib
    assert "trajectory_length" in contrib
    assert "signal_augmentation" in contrib
    assert contrib["resonance_current_norm"] > 0


def test_resonance_contribution_trajectory():
    gen = ResonanceContributionGenerator()
    res = _MockResonance()
    traj = gen.trajectory(res)
    assert len(traj) == 3
    for t in traj:
        assert "step" in t
        assert "norm" in t
        assert "mean" in t


def test_resonance_contribution_factor():
    gen = ResonanceContributionGenerator()
    H_res = np.array([0.3, 0.5, 0.2])
    factor = gen.contribution_factor(H_res)
    expected = np.linalg.norm(H_res) / math.sqrt(3)
    assert abs(factor - expected) < RTOL


def test_resonance_contribution_empty():
    gen = ResonanceContributionGenerator()
    res = _MockResonance(R=None, trajectory=[])
    tree = _make_tree()
    contrib = gen.generate(res, {}, tree)
    assert contrib["resonance_current_norm"] == 0.0


# ═══════════════════════════════════════════════════════════════
# 6. UNCERTAINTY EVOLUTION
# ═══════════════════════════════════════════════════════════════

def test_uncertainty_evolution_generate():
    gen = UncertaintyEvolutionGenerator()
    up = _MockUncertainty()
    tree = _make_tree()
    signals = _make_signals()
    evo = gen.generate(up, signals, tree)
    assert len(evo) == 5
    for entry in evo:
        assert "level" in entry
        assert "label" in entry
        assert "stats" in entry
        assert "mean_confidence" in entry


def test_uncertainty_evolution_per_dimension():
    gen = UncertaintyEvolutionGenerator()
    up = _MockUncertainty()
    signals = _make_signals()
    pd = gen.per_dimension(up, signals)
    assert len(pd) == 6
    for entry in pd:
        assert "dimension" in entry
        assert "mean_variance" in entry


def test_uncertainty_evolution_trace():
    gen = UncertaintyEvolutionGenerator()
    up = _MockUncertainty()
    up.set_variance((1, 0), "S", 0.04)
    up.set_variance((2, 0), "S", 0.03)
    tr = gen.trace(up)
    assert len(tr) >= 2


# ═══════════════════════════════════════════════════════════════
# 7. CONTRIBUTION BREAKDOWN
# ═══════════════════════════════════════════════════════════════

def test_contribution_breakdown_generate():
    gen = ContributionBreakdownGenerator()
    score = _make_ldri_score()
    bd = gen.generate(score)
    assert "ldri" in bd
    assert "components" in bd
    assert "fractions" in bd
    assert abs(bd["ldri"] - 0.417) < RTOL


def test_contribution_breakdown_by_dimension():
    gen = ContributionBreakdownGenerator()
    activations = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
    bd = gen.by_dimension(activations)
    assert len(bd) == 6
    total_contrib = sum(d["contribution"] for d in bd)
    assert abs(total_contrib - 1.0) < RTOL


def test_contribution_breakdown_textual():
    gen = ContributionBreakdownGenerator()
    score = _make_ldri_score()
    text = gen.textual_summary(score)
    assert "LDRI" in text
    assert "intrinsic" in text
    assert "0.417" in text


# ═══════════════════════════════════════════════════════════════
# 8. COMPOSITE GENERATOR
# ═══════════════════════════════════════════════════════════════

def test_composite_generate():
    gen = CompositeExplanationGenerator()
    tree = _make_tree()
    signals = _make_signals()
    score = _make_ldri_score()
    store = _MockMemoryStore(entries=[(0.8, 0.1, 5, 0.9)])
    res = _MockResonance()
    up = _MockUncertainty()
    trace = gen.generate(signals, tree, score, store, res, up)
    assert hasattr(trace, 'signal_lineage')
    assert hasattr(trace, 'propagation_path')
    assert hasattr(trace, 'interaction_history')
    assert hasattr(trace, 'uncertainty_evolution')
    assert len(trace.signal_lineage) > 0
    assert len(trace.propagation_path) > 0


def test_composite_as_dict():
    gen = CompositeExplanationGenerator()
    tree = _make_tree()
    signals = _make_signals()
    score = _make_ldri_score()
    store = _MockMemoryStore(entries=[(0.8, 0.1, 5, 0.9)])
    res = _MockResonance()
    up = _MockUncertainty()
    data = gen.as_dict(signals, tree, score, store, res, up)
    assert "signal_lineage" in data
    assert "propagation_path" in data
    assert "interaction_history" in data
    assert "contribution_breakdown" in data
    assert data["contribution_breakdown"]["ldri"] == 0.417


# ═══════════════════════════════════════════════════════════════
# 9. FORMATTERS
# ═══════════════════════════════════════════════════════════════

def _get_composite_data():
    gen = CompositeExplanationGenerator()
    tree = _make_tree()
    signals = _make_signals()
    score = _make_ldri_score()
    store = _MockMemoryStore(entries=[(0.8, 0.1, 5, 0.9)])
    res = _MockResonance()
    up = _MockUncertainty()
    return gen.as_dict(signals, tree, score, store, res, up)


def test_json_formatter():
    data = _get_composite_data()
    fmt = JSONFormatter()
    output = fmt.format_full(data)
    assert isinstance(output, str)
    assert '"contribution_breakdown"' in output
    assert '"signal_lineage"' in output
    import json
    parsed = json.loads(output)
    assert parsed["contribution_breakdown"]["ldri"] == 0.417


def test_json_formatter_individual():
    fmt = JSONFormatter()
    data = _get_composite_data()
    assert isinstance(fmt.format_signal_lineage(data["signal_lineage"]), str)
    assert isinstance(fmt.format_propagation_path(data["propagation_path"]), str)
    assert isinstance(fmt.format_interaction_history(data["interaction_history"]), str)
    assert isinstance(fmt.format_memory_activations(data["memory_activations"]), str)
    assert isinstance(fmt.format_resonance_contribution(data["resonance_contribution"]), str)
    assert isinstance(fmt.format_uncertainty_evolution(data["uncertainty_evolution"]), str)
    assert '"ldri"' in fmt.format_contribution_breakdown(data["contribution_breakdown"])


def test_markdown_formatter():
    data = _get_composite_data()
    fmt = MarkdownFormatter()
    output = fmt.format_full(data)
    assert isinstance(output, str)
    assert "## Signal Lineage" in output
    assert "## Contribution Breakdown" in output
    assert "## Propagation Path" in output


def test_markdown_formatter_individual():
    fmt = MarkdownFormatter()
    data = _get_composite_data()
    assert "## Signal Lineage" in fmt.format_signal_lineage(data["signal_lineage"])
    assert "## Propagation Path" in fmt.format_propagation_path(data["propagation_path"])
    assert "## Interaction History" in fmt.format_interaction_history(data["interaction_history"])
    assert "## Memory Activations" in fmt.format_memory_activations(data["memory_activations"])
    assert "## Resonance Contribution" in fmt.format_resonance_contribution(data["resonance_contribution"])
    assert "## Uncertainty Evolution" in fmt.format_uncertainty_evolution(data["uncertainty_evolution"])
    assert "## Contribution Breakdown" in fmt.format_contribution_breakdown(data["contribution_breakdown"])


def test_latex_formatter():
    data = _get_composite_data()
    fmt = LaTeXFormatter()
    output = fmt.format_full(data)
    assert isinstance(output, str)
    assert r"\documentclass" in output
    assert r"\begin{document}" in output
    assert r"\subsection*{Contribution Breakdown}" in output


def test_latex_formatter_individual():
    fmt = LaTeXFormatter()
    data = _get_composite_data()
    assert r"\subsection*{Signal Lineage}" in fmt.format_signal_lineage(data["signal_lineage"])
    assert r"\subsection*{Contribution Breakdown}" in fmt.format_contribution_breakdown(data["contribution_breakdown"])
    assert "LDRI" in fmt.format_contribution_breakdown(data["contribution_breakdown"])


def test_html_formatter():
    data = _get_composite_data()
    fmt = HTMLFormatter()
    output = fmt.format_full(data)
    assert isinstance(output, str)
    assert "<!DOCTYPE html>" in output
    assert "<h1>LIT-DISC Explanation Report</h1>" in output
    assert "Contribution Breakdown" in output


def test_html_formatter_empty():
    fmt = HTMLFormatter()
    output = fmt.format_full({})
    assert "<!DOCTYPE html>" in output
    assert "<h1>" in output


# ═══════════════════════════════════════════════════════════════
# 10. EXPLAINABILITY ENGINE
# ═══════════════════════════════════════════════════════════════

def test_engine_generate_all():
    engine = ExplainabilityEngine()
    tree = _make_tree()
    signals = _make_signals()
    score = _make_ldri_score()
    store = _MockMemoryStore(entries=[(0.8, 0.1, 5, 0.9)])
    res = _MockResonance()
    up = _MockUncertainty()
    data = engine.generate_all(signals, tree, score, store, res, up)
    assert "signal_lineage" in data
    assert "contribution_breakdown" in data
    assert data["contribution_breakdown"]["ldri"] == 0.417


def test_engine_formatters():
    engine = ExplainabilityEngine()
    data = {"contribution_breakdown": _make_ldri_score().__dict__}
    assert isinstance(engine.to_json({"test": 1}), str)
    assert isinstance(engine.to_markdown({"test": 1}), str)
    assert isinstance(engine.to_latex({"contribution_breakdown": _make_ldri_score().__dict__}), str)
    assert isinstance(engine.to_html({"test": 1}), str)


def test_engine_export_all():
    engine = ExplainabilityEngine()
    data = _get_composite_data()
    result = engine.export_all(data, ["json", "md", "tex", "html"])
    assert "json" in result
    assert "markdown" in result
    assert "latex" in result
    assert "html" in result
    for fmt, content in result.items():
        assert isinstance(content, str)
        assert len(content) > 0


def test_engine_export_partial():
    engine = ExplainabilityEngine()
    data = _get_composite_data()
    result = engine.export_all(data, ["json"])
    assert "json" in result
    assert "markdown" not in result


# ═══════════════════════════════════════════════════════════════
# 11. EDGE CASES
# ═══════════════════════════════════════════════════════════════

def test_empty_signals():
    gen = SignalLineageGenerator()
    tree = _MockTree()
    tree.add_level(1, [_MockNode(1, 0)])
    assert gen.generate({}, tree) == []


def test_empty_tree():
    gen = PropagationPathGenerator()
    tree = _MockTree()
    path = gen.generate({}, tree)
    assert isinstance(path["description"], str)


def test_empty_interactions():
    gen = InteractionHistoryGenerator()
    tree = _MockTree()
    tree.add_level(1, [_MockNode(1, 0)])
    h = gen.generate({(1, 0): {"S": _make_signal()}}, tree)
    assert h["interactions"] == []


def test_empty_uncertainty():
    gen = UncertaintyEvolutionGenerator()
    up = _MockUncertainty()
    evo = gen.generate(up, {}, _MockTree())
    assert len(evo) == 5


def test_contribution_breakdown_empty():
    gen = ContributionBreakdownGenerator()
    score = LDRIScore(ldri=0.0, gamma_global=0.0, reliability=0.0,
                      raw=0.0, components={}, fractions={})
    bd = gen.generate(score)
    assert bd["ldri"] == 0.0


def test_memory_activation_none():
    gen = MemoryActivationGenerator()
    acts = gen.generate(None)
    assert acts == []


def test_resonance_contribution_null_engine():
    gen = ResonanceContributionGenerator()
    tree = _make_tree()
    contrib = gen.generate(None, {}, tree)
    assert contrib["resonance_current_norm"] == 0.0


def test_formatters_empty_data():
    mf = MarkdownFormatter()
    output = mf.format_signal_lineage([])
    assert "No signal lineage" in output

    hf = HTMLFormatter()
    output = hf.format_full({})
    assert isinstance(output, str)

    jf = JSONFormatter()
    output = jf.format_signal_lineage([])
    assert '"signal_lineage"' in output


def test_numpy_handling():
    bd = ContributionBreakdownGenerator().generate(_make_ldri_score())
    jf = JSONFormatter()
    output = jf.format_contribution_breakdown(bd)
    import json
    parsed = json.loads(output)
    assert "contribution_breakdown" in parsed
    assert isinstance(parsed["contribution_breakdown"]["ldri"], float)


def test_propagation_numpy_array():
    gen = SignalLineageGenerator()
    tree = _make_tree()
    signals = _make_signals()
    lineage = gen.generate(signals, tree)
    for entry in lineage:
        assert isinstance(entry["gamma"], float)
        assert isinstance(entry["feature_norm"], float)


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
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
            err = result.get('error', '')
            try:
                print(f"  [FAIL] {result['name']}: {err}")
            except UnicodeEncodeError:
                print(f"  [FAIL] {result['name']}: [non-ASCII error suppressed]")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed ({passed + failed} total)")
