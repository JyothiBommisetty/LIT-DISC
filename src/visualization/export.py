from pathlib import Path

import matplotlib.pyplot as plt

from src.visualization.styling import DEFAULT_DPI


class ExportManager:
    def __init__(self, dpi: int = DEFAULT_DPI):
        self.dpi = dpi

    def png(self, fig, path: str, dpi: int | None = None) -> str:
        path = self._ensure_ext(path, ".png")
        fig.savefig(path, dpi=dpi or self.dpi, format="png",
                    bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return path

    def svg(self, fig, path: str) -> str:
        path = self._ensure_ext(path, ".svg")
        fig.savefig(path, format="svg", bbox_inches="tight",
                    pad_inches=0.1)
        plt.close(fig)
        return path

    def pdf(self, fig, path: str, dpi: int | None = None) -> str:
        path = self._ensure_ext(path, ".pdf")
        fig.savefig(path, dpi=dpi or self.dpi, format="pdf",
                    bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return path

    def export_all(self, fig, base_path: str,
                   formats: list[str] | None = None,
                   dpi: int | None = None) -> dict[str, str]:
        formats = formats or ["png", "svg", "pdf"]
        results = {}
        stem = Path(base_path).stem
        parent = Path(base_path).parent

        for fmt in formats:
            out_path = str(parent / f"{stem}.{fmt}")
            if fmt == "png":
                results["png"] = self.png(fig, out_path, dpi=dpi)
            elif fmt == "svg":
                results["svg"] = self.svg(fig, out_path)
            elif fmt == "pdf":
                results["pdf"] = self.pdf(fig, out_path, dpi=dpi)
        return results

    @staticmethod
    def _ensure_ext(path: str, ext: str) -> str:
        p = Path(path)
        if p.suffix.lower() != ext:
            p = p.with_suffix(ext)
        return str(p)
