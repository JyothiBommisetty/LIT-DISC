import csv
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from src.explainability import ExplainabilityEngine
from src.models import LITDISCResult
from src.visualization import LITDISCVisualizer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ReportGenerator:
    def __init__(self):
        self.explainability = ExplainabilityEngine()
        self.visualizer = LITDISCVisualizer()

    def to_json(self, result: LITDISCResult, path: str | None = None) -> str:
        data = self._build_data(result)
        output = json.dumps(data, cls=NumpyEncoder, indent=2)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
        return output

    def to_markdown(self, result: LITDISCResult, path: str | None = None) -> str:
        data = self._build_data(result)
        output = self.explainability.to_markdown(data)
        header = self._markdown_header(result)
        output = header + output
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
        return output

    def to_latex(self, result: LITDISCResult, path: str | None = None) -> str:
        data = self._build_data(result)
        output = self.explainability.to_latex(data)
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
        return output

    def to_html(self, result: LITDISCResult, path: str | None = None) -> str:
        data = self._build_data(result)
        meta = result.metadata or {}
        header = (
            f"<p><strong>Text:</strong> {result.text_id} | "
            f"<strong>LDRI:</strong> {result.ldri.ldri:.4f} | "
            f"<strong>Sentences:</strong> {meta.get('sentences', '?')} | "
            f"<strong>Paragraphs:</strong> {meta.get('paragraphs', '?')} | "
            f"<strong>Chapters:</strong> {meta.get('chapters', '?')}</p>\n"
        )
        output = self.explainability.to_html(data)
        output = output.replace("<h1>LIT-DISC Explanation Report</h1>",
                                f"<h1>LIT-DISC Report: {result.text_id}</h1>\n{header}")
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
        return output

    def to_svg(self, result: LITDISCResult, path: str | None = None) -> str:
        activations = self.visualizer._extract_activations(result)
        return self.visualizer.radar_chart(
            activations, filename=path,
            title=f"LIT-DISC Dimension Activations: {result.text_id}")

    def to_pdf(self, result: LITDISCResult, path: str | None = None) -> str:
        """Generate PDF report. Tries pdflatex on LaTeX output; falls back to matplotlib PDF."""
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Try pdflatex first
        latex = self.to_latex(result)
        if self._try_pdflatex(latex, path):
            return path

        # Fallback: generate a summary PDF via matplotlib
        md = self.to_markdown(result)
        fig = self._md_to_pdf_figure(md, result)
        if path:
            fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.5)
            import matplotlib.pyplot as plt
            plt.close(fig)
            return path
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.5)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return buf.getvalue().decode("latin-1")

    def to_csv(self, result: LITDISCResult, path: str | None = None) -> str:
        """Flatten key metrics to CSV."""
        meta = result.metadata or {}
        components = result.ldri.components or {}
        fractions = result.ldri.fractions or {}
        row = {
            "text_id": result.text_id,
            "ldri": f"{result.ldri.ldri:.6f}",
            "raw": f"{result.ldri.raw:.6f}",
            "confidence": f"{result.ldri.gamma_global:.6f}",
            "reliability": f"{result.ldri.reliability:.6f}",
            "embedding_dim": result.embedding.d,
            "sentences": meta.get("sentences", ""),
            "paragraphs": meta.get("paragraphs", ""),
            "chapters": meta.get("chapters", ""),
            "refinement_iters": meta.get("refinement_iterations", ""),
            "memory_entries": meta.get("memory_entries", ""),
            "total_time_s": f"{meta.get('total_time_s', 0):.4f}",
        }
        for k in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            row[f"comp_{k}"] = f"{components.get(k, 0):.6f}"
            row[f"frac_{k}"] = f"{fractions.get(k, 0):.6f}"

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
        csv_str = output.getvalue()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_str)
        return csv_str

    def export_all(self, result: LITDISCResult,
                   formats: list[str] | None = None,
                   output_dir: str = ".") -> dict[str, str]:
        """Export to all specified formats, writing files to output_dir."""
        formats = formats or ["json", "md", "tex", "html", "csv"]
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, result.text_id)
        out: dict[str, str] = {}
        fmt_map = {
            "json": ("json", self.to_json),
            "md": ("md", self.to_markdown),
            "markdown": ("md", self.to_markdown),
            "tex": ("tex", self.to_latex),
            "latex": ("tex", self.to_latex),
            "html": ("html", self.to_html),
            "csv": ("csv", self.to_csv),
            "svg": ("svg", self.to_svg),
            "pdf": ("pdf", self.to_pdf),
        }
        for fmt in formats:
            if fmt in fmt_map:
                ext, func = fmt_map[fmt]
                fpath = f"{base}.{ext}"
                func(result, fpath)
                out[fmt] = fpath
        return out

    def batch_export(self, results: list[LITDISCResult],
                     formats: list[str],
                     output_dir: str = ".") -> dict[str, list[str]]:
        """Export multiple results to specified formats."""
        out: dict[str, list[str]] = {}
        for fmt in formats:
            out[fmt] = []
            for r in results:
                base = os.path.join(output_dir, r.text_id)
                fpath = self._export_single(r, fmt, base)
                out[fmt].append(fpath)
        return out

    def _export_single(self, result: LITDISCResult, fmt: str,
                       base_path: str) -> str:
        ext_map = {
            "json": "json", "md": "md", "markdown": "md",
            "tex": "tex", "latex": "tex", "html": "html",
            "csv": "csv", "svg": "svg", "pdf": "pdf",
        }
        ext = ext_map.get(fmt, fmt)
        fpath = f"{base_path}.{ext}"
        if fmt == "json":
            self.to_json(result, fpath)
        elif fmt in ("md", "markdown"):
            self.to_markdown(result, fpath)
        elif fmt in ("tex", "latex"):
            self.to_latex(result, fpath)
        elif fmt == "html":
            self.to_html(result, fpath)
        elif fmt == "csv":
            self.to_csv(result, fpath)
        elif fmt == "svg":
            self.to_svg(result, fpath)
        elif fmt == "pdf":
            self.to_pdf(result, fpath)
        return fpath

    def _markdown_header(self, result: LITDISCResult) -> str:
        meta = result.metadata or {}
        return (
            f"# LIT-DISC Report: {result.text_id}\n\n"
            f"**LDRI:** {result.ldri.ldri:.4f} &nbsp;|&nbsp; "
            f"**Raw:** {result.ldri.raw:.4f} &nbsp;|&nbsp; "
            f"**Γ:** {result.ldri.gamma_global:.4f} &nbsp;|&nbsp; "
            f"**Reliability:** {result.ldri.reliability:.4f}\n\n"
            f"**Sentences:** {meta.get('sentences', '?')} | "
            f"**Paragraphs:** {meta.get('paragraphs', '?')} | "
            f"**Chapters:** {meta.get('chapters', '?')} | "
            f"**Refinement iterations:** {meta.get('refinement_iterations', '?')}\n\n"
            "---\n\n"
        )

    @staticmethod
    def _try_pdflatex(latex_content: str, pdf_path: str | None) -> bool:
        """Try to compile LaTeX to PDF using pdflatex."""
        if pdf_path is None:
            return False
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = os.path.join(tmpdir, "report.tex")
                with open(tex_file, "w", encoding="utf-8") as f:
                    f.write(latex_content)
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode",
                     "-output-directory", tmpdir, tex_file],
                    capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    src_pdf = os.path.join(tmpdir, "report.pdf")
                    if os.path.exists(src_pdf):
                        import shutil
                        shutil.copy2(src_pdf, pdf_path)
                        return True
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    @staticmethod
    def _md_to_pdf_figure(markdown_text: str, result: LITDISCResult):
        """Create a matplotlib figure with report summary for PDF fallback."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")

        meta = result.metadata or {}
        lines = [
            f"LIT-DISC Report: {result.text_id}",
            "",
            f"LDRI: {result.ldri.ldri:.4f}   (raw: {result.ldri.raw:.4f})",
            f"Confidence Γ: {result.ldri.gamma_global:.4f}",
            f"Reliability: {result.ldri.reliability:.4f}",
            "",
            f"Sentences: {meta.get('sentences', '?')}   "
            f"Paragraphs: {meta.get('paragraphs', '?')}   "
            f"Chapters: {meta.get('chapters', '?')}",
            f"Refinement iterations: {meta.get('refinement_iterations', '?')}   "
            f"Memory entries: {meta.get('memory_entries', '?')}",
            f"Total time: {meta.get('total_time_s', 0):.2f}s",
            "",
            "Decomposition:",
        ]
        comp = result.ldri.components or {}
        frac = result.ldri.fractions or {}
        for key in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            val = comp.get(key, 0)
            fr = frac.get(key, 0)
            lines.append(f"  {key}: {val:.4f}  ({fr:.1%})")

        y = 0.95
        for line in lines:
            if line.startswith("  "):
                ax.text(0.08, y, line, fontsize=9, family="monospace",
                        transform=ax.transAxes, verticalalignment="top")
            elif line == "":
                y -= 0.02
            else:
                weight = "bold" if y > 0.9 else "normal"
                ax.text(0.05, y, line, fontsize=11 if y > 0.9 else 10,
                        weight=weight, transform=ax.transAxes,
                        verticalalignment="top")
            y -= 0.035

        ax.text(0.05, 0.02,
                "Generated by LIT-DISC v1.0 · HMP-DT v1.0 · LDRI v1.0",
                fontsize=7, color="gray", transform=ax.transAxes)
        return fig

    def _build_data(self, result: LITDISCResult) -> dict:
        explanation = result.explanation
        # Flatten explanation keys to top level for formatters
        data = {
            "text_id": result.text_id,
            "ldri": {
                "score": result.ldri.ldri,
                "raw": result.ldri.raw,
                "confidence": result.ldri.gamma_global,
                "reliability": result.ldri.reliability,
                "components": result.ldri.components,
                "fractions": result.ldri.fractions,
            },
            "embedding": {
                "dimension": result.embedding.d,
                "H_S": result.embedding.H_S,
                "H_N": result.embedding.H_N,
                "H_I": result.embedding.H_I,
                "H_Sy": result.embedding.H_Sy,
                "H_E": result.embedding.H_E,
                "H_Em": result.embedding.H_Em,
                "H_pairs": result.embedding.H_pairs,
                "H_res": result.embedding.H_res,
            },
            # Both nested (for JSON) and flat (for formatters)
            "explanation": {
                "signal_lineage": (explanation.signal_lineage
                                   if explanation else []),
                "propagation_path": (explanation.propagation_path
                                     if explanation else {}),
                "interaction_history": (explanation.interaction_history
                                        if explanation else {}),
                "uncertainty_evolution": (explanation.uncertainty_evolution
                                          if explanation else []),
                "resonance_contribution": (explanation.resonance_contribution
                                           if explanation else 0.0),
                "memory_activations": (explanation.memory_activations
                                       if explanation else []),
            },
            "signal_lineage": (explanation.signal_lineage
                               if explanation else []),
            "propagation_path": (explanation.propagation_path
                                 if explanation else {}),
            "interaction_history": (explanation.interaction_history
                                    if explanation else {}),
            "uncertainty_evolution": (explanation.uncertainty_evolution
                                      if explanation else []),
            "resonance_contribution": (explanation.resonance_contribution
                                       if explanation else 0.0),
            "memory_activations": (explanation.memory_activations
                                   if explanation else []),
            "contribution_breakdown": (explanation.contribution_breakdown
                                       if explanation else {}),
            "metadata": {
                "model": "HMP-DT v1.0",
                "metric": "LDRI v1.0",
                "dimensions": ["S", "N", "I", "Sy", "E", "Em"],
                "embedding_dim": result.embedding.d,
                **result.metadata,
            },
        }
        return data
