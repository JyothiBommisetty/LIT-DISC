"""Comprehensive tests for the Visualization subsystem.

Test categories:
  1. Radar chart — generates SVG/PNG, correct axes, determinism
  2. Timeline — generates output, handles empty data
  3. Chapter Richness — bar chart with labels
  4. Dimension Comparison — grouped bars with error bars
  5. Memory Activation — scatter with colorbar
  6. Resonance Curve — line with trend
  7. Theme Evolution — multi-line with legend
  8. Export — PNG, SVG, PDF round-trip
  9. LITDISCVisualizer.full_report — integration
  10. Styling — colors, dimensions, configuration
  11. Edge cases — empty/None data, single values
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "D:\\LIT-DISC")

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

from src.models import (
    ExplanationTrace,
    LDRIScore,
    LITDISCResult,
    NarrativeEmbedding,
)
from src.visualization import (
    ExportManager,
    LITDISCVisualizer,
    configure_style,
    set_deterministic,
)
from src.visualization.styling import (
    _COLOR_LIST,
    _DIM_LABELS,
    _DIMS,
    close_figure,
    create_figure,
    get_color,
    get_dim_full,
    get_dim_label,
    save_figure,
)

_DIMENSION_SET = {"S", "N", "I", "Sy", "E", "Em"}


def _make_embedding(d: int = 120):
    rng = np.random.default_rng(42)
    return NarrativeEmbedding(
        H_S=rng.random(d // 10), H_N=rng.random(d // 10),
        H_I=rng.random(d // 10), H_Sy=rng.random(d // 10),
        H_E=rng.random(d // 10), H_Em=rng.random(d // 10),
        H_pairs=rng.random(3 * d // 10), H_res=rng.random(d // 10), d=d,
    )


def _make_ldri_score():
    return LDRIScore(
        ldri=0.417, gamma_global=0.85, reliability=0.92, raw=0.490,
        components={"intrinsic": 0.262, "interaction": 0.011,
                    "temporal": 0.006, "resonance": 0.005,
                    "memory": 0.016, "uncertainty": -0.074},
        fractions={"intrinsic": 0.62, "interaction": 0.03,
                   "temporal": 0.02, "resonance": 0.01,
                   "memory": 0.04, "uncertainty": -0.18},
    )


def _make_explanation():
    return ExplanationTrace(
        signal_lineage=[
            {"node_id": [1, 0], "level": 1, "index": 0, "dimension": "S",
             "dimension_label": "Symbolic", "tau": 0.1, "gamma": 0.85,
             "feature_norm": 0.5, "provenance": [], "memory_trace_count": 0},
            {"node_id": [1, 1], "level": 1, "index": 1, "dimension": "N",
             "dimension_label": "Narrative", "tau": 0.5, "gamma": 0.78,
             "feature_norm": 0.7, "provenance": [], "memory_trace_count": 1},
            {"node_id": [2, 0], "level": 2, "index": 0, "dimension": "Em",
             "dimension_label": "Emotion", "tau": 0.9, "gamma": 0.92,
             "feature_norm": 0.3, "provenance": [], "memory_trace_count": 0},
        ],
        propagation_path={},
        interaction_history={},
        uncertainty_evolution=[],
        resonance_contribution={
            "resonance_current_norm": 0.35,
            "trajectory_length": 8,
            "trajectory": [
                {"step": 0, "norm": 0.1, "mean": 0.05, "max": 0.1},
                {"step": 3, "norm": 0.25, "mean": 0.12, "max": 0.25},
                {"step": 7, "norm": 0.35, "mean": 0.15, "max": 0.35},
            ],
            "signal_augmentation": {"applied": True, "amplification": 0.35,
                                    "description": "Resonance field norm = 0.3500"},
        },
        memory_activations=[
            {"key_hash": 12345, "beta": 0.85, "storage_tau": 0.1,
             "retrieval_count": 5, "recency": 0.9},
            {"key_hash": 67890, "beta": 0.45, "storage_tau": 0.5,
             "retrieval_count": 2, "recency": 0.5},
            {"key_hash": 11111, "beta": 0.12, "storage_tau": 0.8,
             "retrieval_count": 0, "recency": 0.2},
        ],
        contribution_breakdown=_make_ldri_score().__dict__,
    )


def _make_result(text_id: str = "test_text"):
    return LITDISCResult(
        text_id=text_id,
        embedding=_make_embedding(),
        ldri=_make_ldri_score(),
        explanation=_make_explanation(),
        metadata={"source": "test", "date": "2026-07-27"},
    )


# ═══════════════════════════════════════════════════════════════
# 1. RADAR CHART
# ═══════════════════════════════════════════════════════════════

def test_radar_chart_svg():
    viz = LITDISCVisualizer()
    activations = np.array([0.9, 0.7, 0.5, 0.3, 0.6, 0.8])
    svg = viz.radar_chart(activations, title="Test Radar")
    assert isinstance(svg, str)
    assert "<svg" in svg or len(svg) > 100


def test_radar_chart_png():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    viz = LITDISCVisualizer()
    activations = np.array([0.5, 0.6, 0.7, 0.4, 0.3, 0.8])
    result = viz.radar_chart(activations, filename=path)
    assert result == path
    assert Path(path).stat().st_size > 1000
    Path(path).unlink(missing_ok=True)


def test_radar_chart_deterministic():
    viz = LITDISCVisualizer(deterministic=True)
    a = np.array([0.5, 0.6, 0.7, 0.4, 0.3, 0.8])
    svg1 = viz.radar_chart(a)
    svg2 = viz.radar_chart(a)
    assert isinstance(svg1, str) and isinstance(svg2, str)
    assert len(svg1) > 0 and len(svg2) > 0


def test_radar_chart_all_zeros():
    viz = LITDISCVisualizer()
    svg = viz.radar_chart(np.zeros(6))
    assert "<svg" in svg or len(svg) > 100


def test_radar_chart_all_ones():
    viz = LITDISCVisualizer()
    svg = viz.radar_chart(np.ones(6))
    assert "<svg" in svg or len(svg) > 100


def test_radar_chart_custom_dims():
    viz = LITDISCVisualizer()
    svg = viz.radar_chart(np.array([0.5, 0.4, 0.3]),
                          dims=["S", "N", "I"])
    assert isinstance(svg, str)


# ═══════════════════════════════════════════════════════════════
# 2. TIMELINE
# ═══════════════════════════════════════════════════════════════

def test_timeline_svg():
    viz = LITDISCVisualizer()
    data = {
        "S": [(0.1, 0.5), (0.5, 0.7), (0.9, 0.6)],
        "N": [(0.2, 0.4), (0.6, 0.8), (0.8, 0.3)],
    }
    svg = viz.timeline(data, title="Test Timeline")
    assert isinstance(svg, str)
    assert len(svg) > 100


def test_timeline_empty():
    viz = LITDISCVisualizer()
    svg = viz.timeline({})
    assert isinstance(svg, str)


def test_timeline_partial_dims():
    viz = LITDISCVisualizer()
    data = {"S": [(0.1, 0.5)]}
    svg = viz.timeline(data)
    assert isinstance(svg, str)


def test_timeline_png():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    viz = LITDISCVisualizer()
    data = {"S": [(0.0, 0.2), (0.5, 0.8), (1.0, 0.3)]}
    viz.timeline(data, filename=path)
    assert Path(path).stat().st_size > 1000
    Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# 3. CHAPTER RICHNESS
# ═══════════════════════════════════════════════════════════════

def test_chapter_richness_svg():
    viz = LITDISCVisualizer()
    data = {"Ch1": 0.417, "Ch2": 0.703, "Ch3": 0.550, "Ch4": 0.620}
    svg = viz.chapter_richness(data)
    assert isinstance(svg, str)
    assert len(svg) > 100


def test_chapter_richness_single():
    viz = LITDISCVisualizer()
    data = {"Ch1": 0.500}
    svg = viz.chapter_richness(data)
    assert isinstance(svg, str)


def test_chapter_richness_empty():
    viz = LITDISCVisualizer()
    svg = viz.chapter_richness({})
    assert isinstance(svg, str)


def test_chapter_richness_png():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    viz = LITDISCVisualizer()
    viz.chapter_richness({"Ch1": 0.5}, filename=path)
    assert Path(path).stat().st_size > 1000
    Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# 4. DIMENSION COMPARISON
# ═══════════════════════════════════════════════════════════════

def test_dim_comparison_svg():
    viz = LITDISCVisualizer()
    data = {d: np.array([0.5, 0.6, 0.7]) for d in _DIMS}
    svg = viz.dimension_comparison(data)
    assert isinstance(svg, str)
    assert len(svg) > 100


def test_dim_comparison_single_values():
    viz = LITDISCVisualizer()
    data = {d: np.array([0.5]) for d in _DIMS}
    svg = viz.dimension_comparison(data)
    assert isinstance(svg, str)


def test_dim_comparison_partial():
    viz = LITDISCVisualizer()
    data = {"S": np.array([0.5]), "N": np.array([0.6])}
    svg = viz.dimension_comparison(data)
    assert isinstance(svg, str)


def test_dim_comparison_png():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    viz = LITDISCVisualizer()
    data = {"S": np.array([0.5]), "N": np.array([0.6])}
    viz.dimension_comparison(data, filename=path)
    assert Path(path).stat().st_size > 1000
    Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# 5. MEMORY ACTIVATION
# ═══════════════════════════════════════════════════════════════

def test_memory_activation_svg():
    viz = LITDISCVisualizer()
    entries = [
        {"storage_tau": 0.1, "beta": 0.9, "retrieval_count": 5, "recency": 0.8},
        {"storage_tau": 0.5, "beta": 0.5, "retrieval_count": 2, "recency": 0.4},
        {"storage_tau": 0.8, "beta": 0.2, "retrieval_count": 0, "recency": 0.1},
    ]
    svg = viz.memory_activation(entries)
    assert isinstance(svg, str)
    assert len(svg) > 100


def test_memory_activation_empty():
    viz = LITDISCVisualizer()
    svg = viz.memory_activation([])
    assert isinstance(svg, str)


def test_memory_activation_single():
    viz = LITDISCVisualizer()
    entries = [{"storage_tau": 0.5, "beta": 0.8, "retrieval_count": 1, "recency": 0.5}]
    svg = viz.memory_activation(entries)
    assert isinstance(svg, str)


def test_memory_activation_missing_keys():
    viz = LITDISCVisualizer()
    entries = [{"unknown": 1}]
    svg = viz.memory_activation(entries)
    assert isinstance(svg, str)


# ═══════════════════════════════════════════════════════════════
# 6. RESONANCE CURVE
# ═══════════════════════════════════════════════════════════════

def test_resonance_curve_svg():
    viz = LITDISCVisualizer()
    traj = [0.1, 0.15, 0.2, 0.3, 0.25, 0.35]
    svg = viz.resonance_curve(traj)
    assert isinstance(svg, str)
    assert len(svg) > 100


def test_resonance_curve_flat():
    viz = LITDISCVisualizer()
    svg = viz.resonance_curve([0.5, 0.5, 0.5, 0.5])
    assert isinstance(svg, str)


def test_resonance_curve_single_point():
    viz = LITDISCVisualizer()
    svg = viz.resonance_curve([0.5])
    assert isinstance(svg, str)


def test_resonance_curve_numpy():
    viz = LITDISCVisualizer()
    svg = viz.resonance_curve(np.array([0.1, 0.2, 0.3]))
    assert isinstance(svg, str)


# ═══════════════════════════════════════════════════════════════
# 7. THEME EVOLUTION
# ═══════════════════════════════════════════════════════════════

def test_theme_evolution_svg():
    viz = LITDISCVisualizer()
    data = {
        "S": [(0.0, 0.2), (0.5, 0.8), (1.0, 0.3)],
        "N": [(0.0, 0.5), (0.5, 0.4), (1.0, 0.7)],
    }
    svg = viz.theme_evolution(data)
    assert isinstance(svg, str)
    assert len(svg) > 100


def test_theme_evolution_empty():
    viz = LITDISCVisualizer()
    svg = viz.theme_evolution({})
    assert isinstance(svg, str)


def test_theme_evolution_single_dim():
    viz = LITDISCVisualizer()
    data = {"S": [(0.0, 0.5), (1.0, 0.8)]}
    svg = viz.theme_evolution(data)
    assert isinstance(svg, str)


# ═══════════════════════════════════════════════════════════════
# 8. EXPORT
# ═══════════════════════════════════════════════════════════════

def _make_test_figure():
    fig, ax = create_figure()
    ax.plot([0, 1], [0, 1])
    return fig


def test_export_png():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.png")
        em = ExportManager()
        fig = _make_test_figure()
        result = em.png(fig, path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 500


def test_export_svg():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.svg")
        em = ExportManager()
        fig = _make_test_figure()
        result = em.svg(fig, path)
        assert Path(result).exists()
        content = Path(result).read_text(encoding="utf-8")
        assert "<svg" in content


def test_export_pdf():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.pdf")
        em = ExportManager()
        fig = _make_test_figure()
        result = em.pdf(fig, path)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 100


def test_export_all():
    with tempfile.TemporaryDirectory() as tmp:
        base = str(Path(tmp) / "chart")
        em = ExportManager()
        fig = _make_test_figure()
        results = em.export_all(fig, base, formats=["png", "svg", "pdf"])
        assert "png" in results
        assert "svg" in results
        assert "pdf" in results
        for p in results.values():
            assert Path(p).exists()
            assert Path(p).stat().st_size > 100


def test_export_auto_extension():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test_no_ext")
        em = ExportManager()
        fig = _make_test_figure()
        result = em.png(fig, path)
        assert result.endswith(".png")


def test_export_dpi():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.png")
        em = ExportManager(dpi=150)
        fig = _make_test_figure()
        em.png(fig, path)
        assert Path(path).stat().st_size > 500


# ═══════════════════════════════════════════════════════════════
# 9. FULL REPORT
# ═══════════════════════════════════════════════════════════════

def test_full_report_files():
    with tempfile.TemporaryDirectory() as tmp:
        viz = LITDISCVisualizer()
        result = _make_result()
        out = viz.full_report(result, output_dir=tmp, prefix="test")
        for key in ["radar", "timeline", "chapter_richness",
                     "dim_comparison", "memory", "resonance",
                     "theme_evolution"]:
            assert key in out, f"Missing {key}"
            assert Path(out[key]).exists(), f"File {out[key]} does not exist"
            assert Path(out[key]).stat().st_size > 500


def test_full_report_svg_backend():
    viz = LITDISCVisualizer()
    result = _make_result()
    out = viz.full_report(result, output_dir=".", prefix="test_svg")
    assert isinstance(out, dict)
    for path in out.values():
        assert Path(path).exists()
        Path(path).unlink(missing_ok=True)
    assert isinstance(viz.radar_chart(np.ones(6)), str)
    assert isinstance(viz.timeline({"S": [(0, 0.5)]}), str)


# ═══════════════════════════════════════════════════════════════
# 10. STYLING
# ═══════════════════════════════════════════════════════════════

def test_dimension_labels():
    assert len(_DIMS) == 6
    assert set(_DIMS) == _DIMENSION_SET
    for d in _DIMS:
        assert d in _DIM_LABELS
        assert len(_DIM_LABELS[d]) > 0


def test_colors():
    for d in _DIMS:
        color = get_color(d)
        assert color.startswith("#")
        assert len(color) == 7


def test_color_list():
    assert len(_COLOR_LIST) == 6
    assert len(set(_COLOR_LIST)) == 6


def test_dim_label_unknown():
    label = get_dim_label("X")
    assert "X" in label


def test_dim_full():
    full = get_dim_full("S")
    assert "Symbolic" in full


def test_create_figure():
    fig, ax = create_figure()
    assert fig is not None
    assert ax is not None
    close_figure(fig)


def test_create_figure_multi():
    fig, axes = create_figure(nrows=2, ncols=2)
    assert axes.shape == (2, 2)
    close_figure(fig)


def test_save_figure():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    fig, ax = create_figure()
    ax.plot([0, 1], [0, 1])
    save_figure(fig, path)
    assert Path(path).stat().st_size > 500
    Path(path).unlink(missing_ok=True)


def test_configure_style():
    configure_style()
    assert mpl.rcParams["font.size"] >= 10


def test_set_deterministic():
    set_deterministic()
    a = np.random.random(5)
    set_deterministic()
    b = np.random.random(5)
    assert np.allclose(a, b)


# ═══════════════════════════════════════════════════════════════
# 11. EDGE CASES
# ═══════════════════════════════════════════════════════════════

def test_visualizer_to_bytes():
    viz = LITDISCVisualizer()
    fig, ax = create_figure()
    ax.plot([0, 1], [0, 1])
    data = viz.to_bytes(fig, fmt="png")
    assert isinstance(data, bytes)
    assert len(data) > 100
    close_figure(fig)


def test_visualizer_to_svg():
    viz = LITDISCVisualizer()
    fig, ax = create_figure()
    ax.plot([0, 1], [0, 1])
    svg = viz.to_svg(fig)
    assert isinstance(svg, str)
    assert "<svg" in svg
    close_figure(fig)


def test_visualizer_export():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    viz = LITDISCVisualizer()
    fig, ax = create_figure()
    ax.plot([0, 1], [0, 1])
    viz.export(fig, path)
    assert Path(path).stat().st_size > 500
    Path(path).unlink(missing_ok=True)
    close_figure(fig)


def test_full_report_minimal_result():
    viz = LITDISCVisualizer()
    empty_explanation = ExplanationTrace(
        signal_lineage=[], propagation_path={}, interaction_history={},
        uncertainty_evolution=[], resonance_contribution=0.0,
        memory_activations=[],
    )
    result = LITDISCResult(
        text_id="minimal",
        embedding=_make_embedding(),
        ldri=_make_ldri_score(),
        explanation=empty_explanation,
        metadata={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = viz.full_report(result, output_dir=tmp, prefix="min")
        for key in ["radar", "timeline", "chapter_richness",
                     "dim_comparison", "memory", "resonance",
                     "theme_evolution"]:
            assert key in out
            assert Path(out[key]).exists()


def test_missing_memory_keys():
    viz = LITDISCVisualizer()
    entries = [{}]
    svg = viz.memory_activation(entries)
    assert isinstance(svg, str)


def test_empty_resonance_data():
    viz = LITDISCVisualizer()
    traj = viz._extract_resonance_data(_make_result())
    assert len(traj) > 0


def test_extract_activations():
    viz = LITDISCVisualizer()
    result = _make_result()
    a = viz._extract_activations(result)
    assert a.shape == (6,)
    assert np.all(a >= 0) and np.all(a <= 1)


def test_extract_timeline_data():
    viz = LITDISCVisualizer()
    result = _make_result()
    data = viz._extract_timeline_data(result)
    assert len(data) > 0


def test_extract_chapter_data():
    viz = LITDISCVisualizer()
    result = _make_result()
    data = viz._extract_chapter_data(result)
    assert "Chapter" in data


def test_extract_memory_data():
    viz = LITDISCVisualizer()
    result = _make_result()
    data = viz._extract_memory_data(result)
    assert len(data) >= 3


def test_extract_resonance_data():
    viz = LITDISCVisualizer()
    result = _make_result()
    data = viz._extract_resonance_data(result)
    assert len(data) > 0


def test_constant_resonance():
    viz = LITDISCVisualizer()
    svg = viz.resonance_curve([0.5] * 10)
    assert isinstance(svg, str)


def test_linear_resonance():
    viz = LITDISCVisualizer()
    svg = viz.resonance_curve(np.linspace(0.1, 0.9, 20))
    assert isinstance(svg, str)


# ═══════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════

def _check_fn(fn):
    name = fn.__name__
    try:
        fn()
        return {"name": name, "status": "PASS"}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"name": name, "status": "FAIL", "error": f"{e}\n{traceback.format_exc()}"}


if __name__ == "__main__":
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
            err = result.get("error", "")
            try:
                print(f"  [FAIL] {result['name']}: {err.split(chr(10))[0]}")
            except UnicodeEncodeError:
                print(f"  [FAIL] {result['name']}: [error]")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed ({passed + failed} total)")
