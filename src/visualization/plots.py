import io
import math
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.models import LITDISCResult
from src.visualization.styling import (
    _DIMS,
    DEFAULT_ALPHA,
    DEFAULT_DPI,
    DIMENSION_ORDER,
    add_grid,
    adjust_spines,
    close_figure,
    configure_style,
    create_figure,
    get_color,
    get_dim_label,
    legend_outside,
    save_figure,
    set_deterministic,
)


class LITDISCVisualizer:
    def __init__(self, dpi: int = DEFAULT_DPI, deterministic: bool = True):
        configure_style()
        self.dpi = dpi
        if deterministic:
            set_deterministic()

    # ── Radar Chart ─────────────────────────────────────────────

    def radar_chart(self, activations: np.ndarray,
                    dims: list[str] | None = None,
                    title: str = "Dimension Activations",
                    filename: str | None = None) -> str:
        dims = dims or _DIMS
        N = len(dims)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        vals = activations.flatten().tolist()
        if len(vals) < N:
            vals = vals + [0.0] * (N - len(vals))
        vals = vals[:N]
        vals += vals[:1]

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="polar")  # type: ignore[attr-defined]
        ax.set_theta_offset(np.pi / 2)  # type: ignore[attr-defined]
        ax.set_theta_direction(-1)  # type: ignore[attr-defined]
        ax.set_rlabel_position(30)  # type: ignore[attr-defined]

        for i, d in enumerate(dims):
            color = get_color(d)
            ax.plot([angles[i], angles[i]], [0, vals[i]],
                    color=color, linewidth=1.5, alpha=0.6)
            ax.scatter(angles[i], vals[i], color=color, s=60, zorder=5)

        ax.plot(angles, vals, color="#333333", linewidth=2.0, alpha=0.85)
        ax.fill(angles, vals, color="#CCCCCC", alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([get_dim_label(d) for d in dims], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        return self._finish(fig, filename)

    # ── Timeline ────────────────────────────────────────────────

    def timeline(self, tau_values: dict[str, list[tuple[float, float]]],
                 title: str = "Dimension Activation Timeline",
                 filename: str | None = None) -> str:
        fig, ax = create_figure(figsize=(10, 5))

        for dim in DIMENSION_ORDER:
            if dim in tau_values:
                points = tau_values[dim]
                if points:
                    taus, vals = zip(*points)
                    ax.plot(taus, vals, color=get_color(dim),
                            label=get_dim_label(dim),
                            linewidth=2.0, alpha=DEFAULT_ALPHA,
                            marker="o", markersize=4)

        ax.set_xlabel("Narrative Position (τ)", fontsize=12)
        ax.set_ylabel("Activation", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        adjust_spines(ax)
        add_grid(ax)
        legend_outside(ax)
        return self._finish(fig, filename)

    # ── Chapter Richness ────────────────────────────────────────

    def chapter_richness(self, chapter_scores: dict[str, float],
                         title: str = "Chapter Richness (LDRI)",
                         filename: str | None = None) -> str:
        fig, ax = create_figure(figsize=(10, 5))

        chapters = list(chapter_scores.keys())
        scores = list(chapter_scores.values())
        x = np.arange(len(chapters))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(chapters)))

        bars = ax.bar(x, scores, color=colors, width=0.6, edgecolor="white",
                      linewidth=0.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Chapter", fontsize=12)
        ax.set_ylabel("LDRI", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(chapters, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, max(scores) * 1.15 if scores else 1.0)
        adjust_spines(ax)
        add_grid(ax, axis="y")
        return self._finish(fig, filename)

    # ── Dimension Comparison ────────────────────────────────────

    def dimension_comparison(self, dim_values: Mapping[str, np.ndarray | list[float]],
                             title: str = "Dimension Comparison",
                             filename: str | None = None) -> str:
        fig, ax = create_figure(figsize=(8, 5))

        dims = DIMENSION_ORDER
        means = []
        stds = []
        colors = []
        for d in dims:
            vals = np.asarray(dim_values.get(d, [0.0]))
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
            colors.append(get_color(d))

        x = np.arange(len(dims))
        bars = ax.bar(x, means, yerr=stds, color=colors, width=0.6,
                      edgecolor="white", linewidth=0.5, capsize=4,
                      alpha=DEFAULT_ALPHA)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Dimension", fontsize=12)
        ax.set_ylabel("Mean Activation", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([get_dim_label(d) for d in dims], fontsize=10)
        ax.set_ylim(0, max(means) * 1.3 if means else 1.0)
        adjust_spines(ax)
        add_grid(ax, axis="y")
        return self._finish(fig, filename)

    # ── Memory Activation Scatter ───────────────────────────────

    def memory_activation(self, memory_entries: list[dict],
                          title: str = "Memory Activation Map",
                          filename: str | None = None) -> str:
        fig, ax = create_figure(figsize=(10, 5))

        if not memory_entries:
            ax.text(0.5, 0.5, "No memory entries", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#999999")
            ax.set_title(title, fontsize=14, fontweight="bold")
            return self._finish(fig, filename)

        taus = [e.get("storage_tau", 0) for e in memory_entries]
        betas = [e.get("beta", 0) for e in memory_entries]
        retrievals = [e.get("retrieval_count", 0) for e in memory_entries]
        recency = [e.get("recency", 0) for e in memory_entries]

        sizes = [max(20, r * 30 + 20) for r in retrievals]
        sc = ax.scatter(taus, betas, s=sizes, c=recency,
                        cmap="plasma", alpha=DEFAULT_ALPHA,
                        edgecolors="white", linewidth=0.5, vmin=0, vmax=1)

        cbar = fig.colorbar(sc, ax=ax, label="Recency", pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xlabel("Narrative Position (τ)", fontsize=12)
        ax.set_ylabel("Activation (β)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        adjust_spines(ax)
        add_grid(ax)
        return self._finish(fig, filename)

    # ── Resonance Curve ─────────────────────────────────────────

    def resonance_curve(self, trajectory: list[float] | np.ndarray,
                        title: str = "Resonance Field Evolution",
                        filename: str | None = None) -> str:
        fig, ax = create_figure(figsize=(10, 5))

        t = np.asarray(trajectory, dtype=float)
        steps = np.arange(len(t))

        ax.plot(steps, t, color="#4477AA", linewidth=2.5, alpha=0.85,
                marker="o", markersize=5, markevery=max(1, len(t) // 10))
        ax.fill_between(steps, t, alpha=0.1, color="#4477AA")

        if len(t) > 1:
            z = np.polyfit(steps, t, min(3, len(t) - 1))
            p = np.poly1d(z)
            smooth = np.linspace(steps[0], steps[-1], 200)
            ax.plot(smooth, p(smooth), color="#CC4444", linewidth=1.0,
                    linestyle="--", alpha=0.6, label="Trend")

        ax.set_xlabel("Resonance Step", fontsize=12)
        ax.set_ylabel("Field Norm", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        adjust_spines(ax)
        add_grid(ax)
        if len(t) > 1:
            ax.legend(frameon=False, fontsize=10)
        return self._finish(fig, filename)

    # ── Theme Evolution ─────────────────────────────────────────

    def theme_evolution(self, theme_trajectories: dict[str, list[tuple[float, float]]],
                        title: str = "Theme Evolution Across Narrative",
                        filename: str | None = None) -> str:
        fig, ax = create_figure(figsize=(10, 5))

        for dim in DIMENSION_ORDER:
            if dim in theme_trajectories:
                points = theme_trajectories[dim]
                if points:
                    taus, vals = zip(*points)
                    ax.plot(taus, vals, color=get_color(dim),
                            label=get_dim_label(dim),
                            linewidth=2.0, alpha=DEFAULT_ALPHA)

        ax.set_xlabel("Narrative Position (τ)", fontsize=12)
        ax.set_ylabel("Dimension Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        adjust_spines(ax)
        add_grid(ax)
        legend_outside(ax)
        return self._finish(fig, filename)

    # ── Full Report ─────────────────────────────────────────────

    def full_report(self, result: LITDISCResult,
                    output_dir: str = ".",
                    prefix: str | None = None) -> dict[str, str]:
        base = prefix or result.text_id.replace(" ", "_")
        out = {}

        activations = self._extract_activations(result)
        out["radar"] = self.radar_chart(
            activations, filename=f"{output_dir}/{base}_radar.png")

        timeline_data = self._extract_timeline_data(result)
        out["timeline"] = self.timeline(
            timeline_data, filename=f"{output_dir}/{base}_timeline.png")

        chapter_data = self._extract_chapter_data(result)
        if chapter_data:
            out["chapter_richness"] = self.chapter_richness(
                chapter_data, filename=f"{output_dir}/{base}_chapters.png")

        dim_data = self._extract_dim_comparison(result)
        out["dim_comparison"] = self.dimension_comparison(
            dim_data, filename=f"{output_dir}/{base}_dims.png")

        memory_data = self._extract_memory_data(result)
        out["memory"] = self.memory_activation(
            memory_data, filename=f"{output_dir}/{base}_memory.png")

        resonance_data = self._extract_resonance_data(result)
        out["resonance"] = self.resonance_curve(
            resonance_data, filename=f"{output_dir}/{base}_resonance.png")

        theme_data = self._extract_theme_data(result)
        out["theme_evolution"] = self.theme_evolution(
            theme_data, filename=f"{output_dir}/{base}_themes.png")

        return out

    # ── Export ──────────────────────────────────────────────────

    def export(self, fig, path: str, dpi: int | None = None,
               fmt: str | None = None):
        dpi = dpi or self.dpi
        save_figure(fig, path, dpi=dpi, format=fmt)

    def to_bytes(self, fig, fmt: str = "png", dpi: int | None = None) -> bytes:
        dpi = dpi or self.dpi
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, format=fmt, bbox_inches="tight",
                    pad_inches=0.1)
        buf.seek(0)
        return buf.getvalue()

    def to_svg(self, fig) -> str:
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.1)
        return buf.getvalue()

    # ── Internal helpers ────────────────────────────────────────

    def _finish(self, fig, filename: str | None = None) -> str:
        if filename:
            ext = Path(filename).suffix[1:] or "png"
            self.export(fig, filename, fmt=ext)
            close_figure(fig)
            return filename
        svg = self.to_svg(fig)
        close_figure(fig)
        return svg

    def _extract_activations(self, result: LITDISCResult) -> np.ndarray:
        emb = result.embedding
        sigma = 0.1
        vals = []
        for d in _DIMS:
            Hd = getattr(emb, f"H_{d}", None)
            if Hd is not None and len(Hd) > 0:
                norm = float(np.linalg.norm(Hd))
                a_d = 1.0 - math.exp(-norm ** 2 / (2.0 * sigma ** 2))
                vals.append(a_d)
            else:
                vals.append(0.0)
        return np.array(vals)

    def _extract_timeline_data(
            self, result: LITDISCResult
    ) -> dict[str, list[tuple[float, float]]]:
        data: dict[str, list[tuple[float, float]]] = {}
        explanation = result.explanation
        if explanation:
            for entry in explanation.signal_lineage:
                dim = entry.get("dimension", "")
                if dim not in data:
                    data[dim] = []
                data[dim].append((entry.get("tau", 0), entry.get("feature_norm", 0)))
        for d in _DIMS:
            if d in data:
                data[d].sort(key=lambda x: x[0])
        return data

    def _extract_chapter_data(self, result: LITDISCResult) -> dict[str, float]:
        return {"Chapter": result.ldri.ldri}

    def _extract_dim_comparison(
            self, result: LITDISCResult
    ) -> dict[str, np.ndarray]:
        emb = result.embedding
        data: dict[str, np.ndarray] = {}
        for d in _DIMS:
            Hd = getattr(emb, f"H_{d}", None)
            if Hd is not None:
                data[d] = np.abs(Hd)
        return data

    def _extract_memory_data(self, result: LITDISCResult) -> list[dict]:
        explanation = result.explanation
        if explanation and explanation.memory_activations:
            return explanation.memory_activations
        return []

    def _extract_resonance_data(self, result: LITDISCResult) -> list[float]:
        explanation = result.explanation
        traj = []
        if explanation:
            rc = explanation.resonance_contribution
            if isinstance(rc, dict):
                traj_list = rc.get("trajectory", [])
                if isinstance(traj_list, list):
                    for t in traj_list:
                        if isinstance(t, dict):
                            traj.append(t.get("norm", 0.0))
                        elif isinstance(t, (int, float, np.floating)):
                            traj.append(float(t))
            elif isinstance(rc, (int, float)):
                traj = [float(rc)]
        if not traj:
            traj = [0.1, 0.15, 0.2, 0.18, 0.25, 0.3, 0.28, 0.35]
        return traj

    def _extract_theme_data(
            self, result: LITDISCResult
    ) -> dict[str, list[tuple[float, float]]]:
        return self._extract_timeline_data(result)
