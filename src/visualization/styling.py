"""Shared styling configuration for publication-quality LIT-DISC visualizations."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_DIMS = ["S", "N", "I", "Sy", "E", "Em"]
_DIM_LABELS = {
    "S": "Symbolic", "N": "Narrative", "I": "Interaction",
    "Sy": "Semantic", "E": "Evolution", "Em": "Emotion",
}

_DIM_FULL = {d: f"{d}: {_DIM_LABELS[d]}" for d in _DIMS}

_COLORS = {
    "S": "#4477AA",
    "N": "#EE6677",
    "I": "#228833",
    "Sy": "#CCBB44",
    "E": "#66CCEE",
    "Em": "#AA3377",
}

_COLOR_LIST = [_COLORS[d] for d in _DIMS]

DIMENSION_ORDER = _DIMS

DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (8.0, 6.0)
DEFAULT_FONTFAMILY = "sans-serif"
DEFAULT_TITLE_SIZE = 14
DEFAULT_LABEL_SIZE = 12
DEFAULT_TICK_SIZE = 10
DEFAULT_LEGEND_SIZE = 10
DEFAULT_LINE_WIDTH = 2.0
DEFAULT_MARKER_SIZE = 6.0
DEFAULT_ALPHA = 0.85
DPI_LOW = 72
DPI_HIGH = 600


def configure_style():
    mpl.rcParams.update({
        "font.family": DEFAULT_FONTFAMILY,
        "font.size": DEFAULT_LABEL_SIZE,
        "axes.titlesize": DEFAULT_TITLE_SIZE,
        "axes.labelsize": DEFAULT_LABEL_SIZE,
        "xtick.labelsize": DEFAULT_TICK_SIZE,
        "ytick.labelsize": DEFAULT_TICK_SIZE,
        "legend.fontsize": DEFAULT_LEGEND_SIZE,
        "lines.linewidth": DEFAULT_LINE_WIDTH,
        "lines.markersize": DEFAULT_MARKER_SIZE,
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


configure_style()


def get_color(dim: str) -> str:
    return _COLORS.get(dim, "#333333")


def get_dim_label(dim: str) -> str:
    return f"{dim}: {_DIM_LABELS.get(dim, dim)}"


def get_dim_full(dim: str) -> str:
    return _DIM_FULL.get(dim, dim)


def colorbar_cmap(name: str = "viridis", n: int = 256):
    return mpl.colormaps[name]


def set_deterministic():
    np.random.seed(42)


def create_figure(nrows: int = 1, ncols: int = 1,
                  figsize: tuple[float, float] | None = None):
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        return fig, axes
    return fig, axes


def close_figure(fig):
    plt.close(fig)


def adjust_spines(ax, spine_positions: tuple[str, ...] = ("left", "bottom")):
    for spine in ax.spines.values():
        spine.set_visible(False)
    for pos in spine_positions:
        if pos in ax.spines:
            ax.spines[pos].set_visible(True)
            ax.spines[pos].set_linewidth(0.5)
            ax.spines[pos].set_color("#888888")


def add_grid(ax, axis: str = "y", alpha: float = 0.3):
    ax.grid(True, axis=axis, alpha=alpha, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def legend_outside(ax, loc: str = "lower center",
                   bbox_to_anchor: tuple[float, float] = (0.5, -0.25),
                   ncol: int = 3):
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor,
                  ncol=ncol, frameon=False)


def save_figure(fig, path: str, dpi: int = DEFAULT_DPI,
                format: str | None = None):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                pad_inches=0.1, format=format)
