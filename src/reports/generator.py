import json
import numpy as np
from src.models import LITDISCResult


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
    def to_json(self, result: LITDISCResult, path: str | None = None) -> str:
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
            "explanation": {
                "signal_lineage": result.explanation.signal_lineage if result.explanation else [],
                "resonance_contribution": result.explanation.resonance_contribution if result.explanation else 0.0,
                "memory_activations": result.explanation.memory_activations if result.explanation else [],
            },
            "metadata": {
                "model": "HMP-DT v1.0",
                "metric": "LDRI v1.0",
                "dimensions": ["S", "N", "I", "Sy", "E", "Em"],
                "embedding_dim": result.embedding.d,
                **result.metadata,
            },
        }
        output = json.dumps(data, cls=NumpyEncoder, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
        return output

    def to_latex(self, result: LITDISCResult, path: str | None = None) -> str:
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{LDRI Analysis for " + result.text_id.replace("_", r"\_") + "}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Component & Value & Fraction \\",
            r"\midrule",
        ]
        comps = result.ldri.components
        fracs = result.ldri.fractions
        for key in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            val = comps.get(key, 0.0)
            frac = fracs.get(key, 0.0)
            lines.append(f"{key.capitalize()} & {val:.4f} & {frac:.4f} \\\\")
        lines.append(r"\midrule")
        lines.append(f"LDRI & {result.ldri.ldri:.4f} & --- \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        output = "\n".join(lines)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(output)
        return output

    def to_svg(self, visualization_data: dict, path: str | None = None) -> str:
        raise NotImplementedError

    def batch_export(self, results: list[LITDISCResult],
                     formats: list[str]) -> dict[str, list[str]]:
        out = {}
        for fmt in formats:
            out[fmt] = []
            for i, r in enumerate(results):
                fpath = f"{r.text_id}_{i}.{fmt}"
                if fmt == "json":
                    self.to_json(r, fpath)
                elif fmt == "latex":
                    self.to_latex(r, fpath)
                out[fmt].append(fpath)
        return out
