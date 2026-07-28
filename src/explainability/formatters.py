import json

import numpy as np


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


class JSONFormatter:
    def format(self, data: dict) -> str:
        return json.dumps(data, cls=NumpyEncoder, indent=2)

    def format_signal_lineage(self, lineage: list[dict]) -> str:
        return self.format({"signal_lineage": lineage})

    def format_propagation_path(self, path: dict) -> str:
        return self.format({"propagation_path": path})

    def format_interaction_history(self, history: dict) -> str:
        return self.format({"interaction_history": history})

    def format_memory_activations(self, activations: list[dict]) -> str:
        return self.format({"memory_activations": activations})

    def format_resonance_contribution(self, contribution: dict) -> str:
        return self.format({"resonance_contribution": contribution})

    def format_uncertainty_evolution(self, evolution: list[dict]) -> str:
        return self.format({"uncertainty_evolution": evolution})

    def format_contribution_breakdown(self, breakdown: dict) -> str:
        return self.format({"contribution_breakdown": breakdown})

    def format_full(self, data: dict) -> str:
        return self.format(data)


class MarkdownFormatter:
    def format_signal_lineage(self, lineage: list[dict]) -> str:
        if not lineage:
            return "_No signal lineage available._\n"
        lines = ["## Signal Lineage\n", "| Node ID | Level | Dimension | τ | γ | Norm | Provenance |",
                 "|---------|-------|-----------|----|----|------|------------|"]
        for entry in lineage:
            nid = f"({entry['node_id'][0]},{entry['node_id'][1]})" if entry.get('node_id') else "-"
            prov = ", ".join(f"({p[0]},{p[1]})" for p in entry.get('provenance', [])) or "-"
            lines.append(
                f"| {nid} | {entry.get('level', '-')} | "
                f"{entry.get('dimension', '-')} ({entry.get('dimension_label', '')}) | "
                f"{entry.get('tau', 0):.3f} | {entry.get('gamma', 0):.3f} | "
                f"{entry.get('feature_norm', 0):.3f} | {prov} |")
        return "\n".join(lines) + "\n"

    def format_propagation_path(self, path: dict) -> str:
        lines = ["## Propagation Path\n"]
        hierarchy = path.get("hierarchy", {})
        for level_key in sorted(hierarchy.keys()):
            level_data = hierarchy[level_key]
            lines.append(f"### {level_data['label']} ({level_data['node_count']} nodes)\n")
            for node in level_data.get("nodes", []):
                nid = node.get("node_id", [])
                tau = node.get("tau", 0.5)
                dims = node.get("dimensions", {})
                if dims:
                    dim_parts = [f"{d}: γ={info['gamma']:.3f}, ‖f‖={info['norm']:.3f}"
                                 for d, info in sorted(dims.items())]
                    lines.append(f"- Node ({nid[0]},{nid[1]}) τ={tau:.3f}: {', '.join(dim_parts)}")
                else:
                    lines.append(f"- Node ({nid[0]},{nid[1]}) τ={tau:.3f}: _no signals_")
        lines.append("")
        desc = path.get("description", "")
        if desc:
            lines.append(f"**Path:** {desc}\n")
        return "\n".join(lines)

    def format_interaction_history(self, history: dict) -> str:
        lines = ["## Interaction History\n"]
        interactions = history.get("interactions", [])
        if not interactions:
            lines.append("_No cross-dimensional interactions recorded._\n")
        else:
            lines.append("| Node ID | Level | Pair | Cosine Similarity | γ₁ | γ₂ |")
            lines.append("|---------|-------|------|------------------|-----|-----|")
            for entry in interactions:
                nid = f"({entry['node_id'][0]},{entry['node_id'][1]})"
                labels = entry.get("dimension_labels", ["", ""])
                pair = f"{entry['dimension_pair']} ({labels[0]}-{labels[1]})"
                lines.append(
                    f"| {nid} | {entry.get('level', '-')} | {pair} | "
                    f"{entry.get('cosine_similarity', 0):.4f} | "
                    f"{entry.get('gamma_1', 0):.3f} | {entry.get('gamma_2', 0):.3f} |")
        summary = history.get("summary", {})
        if summary:
            lines.append(
                f"\n**Summary:** {summary.get('count', 0)} interactions, "
                f"mean cosine = {summary.get('mean_cosine', 0):.4f}, "
                f"max = {summary.get('max_cosine', 0):.4f}")
            if summary.get("pairs"):
                lines.append("\n**Pair counts:**\n")
                for pair, count in sorted(summary["pairs"].items()):
                    lines.append(f"- {pair}: {count}")
        return "\n".join(lines) + "\n"

    def format_memory_activations(self, activations: list[dict]) -> str:
        lines = ["## Memory Activations\n"]
        if not activations:
            lines.append("_No memory entries._\n")
        else:
            lines.append("| Key Hash | β (Activation) | Stored at τ | Retrievals | Recency |")
            lines.append("|----------|----------------|-------------|------------|---------|")
            for entry in activations:
                lines.append(
                    f"| {entry.get('key_hash', 0)} | {entry.get('beta', 0):.4f} | "
                    f"{entry.get('storage_tau', 0):.3f} | "
                    f"{entry.get('retrieval_count', 0)} | {entry.get('recency', 0):.3f} |")
        return "\n".join(lines) + "\n"

    def format_resonance_contribution(self, contribution: dict) -> str:
        lines = ["## Resonance Contribution\n"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Current Norm | {contribution.get('resonance_current_norm', 0):.4f} |")
        lines.append(f"| Trajectory Length | {contribution.get('trajectory_length', 0)} |")
        aug = contribution.get("signal_augmentation", {})
        lines.append(f"| Augmentation Applied | {aug.get('applied', False)} |")
        lines.append(f"| Amplification | {aug.get('amplification', 0):.4f} |")
        if aug.get("description"):
            lines.append(f"| Description | {aug['description']} |")
        traj = contribution.get("trajectory", [])
        if traj:
            lines.append("\n### Trajectory Samples\n")
            lines.append("| Step | Norm | Mean | Max |")
            lines.append("|------|------|------|-----|")
            for t in traj:
                lines.append(
                    f"| {t.get('step', 0)} | {t.get('norm', 0):.4f} | "
                    f"{t.get('mean', 0):.4f} | {t.get('max', 0):.4f} |")
        return "\n".join(lines) + "\n"

    def format_uncertainty_evolution(self, evolution: list[dict]) -> str:
        lines = ["## Uncertainty Evolution\n"]
        if not evolution:
            lines.append("_No uncertainty data._\n")
            return "\n".join(lines)
        lines.append("| Level | Label | Count | Mean Variance | Mean Confidence |")
        lines.append("|-------|-------|-------|---------------|-----------------|")
        for entry in evolution:
            stats = entry.get("stats", {})
            count = stats.get("count", 0)
            mean_var = stats.get("mean", 0)
            mean_conf = entry.get("mean_confidence", 0)
            lines.append(
                f"| {entry.get('level', '-')} | {entry.get('label', '')} | {count} | "
                f"{mean_var:.4f} | {mean_conf:.4f} |")
        return "\n".join(lines) + "\n"

    def format_contribution_breakdown(self, breakdown: dict) -> str:
        lines = ["## Contribution Breakdown\n"]
        lines.append("| Component | Value | Fraction |")
        lines.append("|-----------|-------|----------|")
        components = breakdown.get("components", {})
        fractions = breakdown.get("fractions", {})
        for key in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            val = components.get(key, 0.0)
            frac = fractions.get(key, 0.0)
            lines.append(f"| {key.capitalize()} | {val:.4f} | {frac:.4f} |")
        lines.append(f"\n**LDRI:** {breakdown.get('ldri', 0):.4f}  |  "
                     f"**Raw:** {breakdown.get('raw', 0):.4f}  |  "
                     f"**Γ:** {breakdown.get('gamma_global', 0):.4f}  |  "
                     f"**Reliability:** {breakdown.get('reliability', 0):.4f}")
        return "\n".join(lines) + "\n"

    def format_full(self, data: dict) -> str:
        parts = []
        if "signal_lineage" in data:
            parts.append(self.format_signal_lineage(data["signal_lineage"]))
        if "propagation_path" in data:
            parts.append(self.format_propagation_path(data["propagation_path"]))
        if "interaction_history" in data:
            parts.append(self.format_interaction_history(data["interaction_history"]))
        if "memory_activations" in data:
            parts.append(self.format_memory_activations(data["memory_activations"]))
        if "resonance_contribution" in data:
            parts.append(self.format_resonance_contribution(data["resonance_contribution"]))
        if "uncertainty_evolution" in data:
            parts.append(self.format_uncertainty_evolution(data["uncertainty_evolution"]))
        if "contribution_breakdown" in data:
            parts.append(self.format_contribution_breakdown(data["contribution_breakdown"]))
        return "\n---\n".join(parts)


class LaTeXFormatter:
    def format_signal_lineage(self, lineage: list[dict]) -> str:
        lines = [
            r"\subsection*{Signal Lineage}",
            r"\begin{longtable}{lllllll}",
            r"\toprule",
            r"Node & Level & Dim & $\tau$ & $\gamma$ & $\|f\|$ & Provenance \\",
            r"\midrule",
        ]
        for entry in lineage[:50]:
            nid = f"({entry['node_id'][0]},{entry['node_id'][1]})" if entry.get('node_id') else "-"
            prov = ", ".join(f"({p[0]},{p[1]})" for p in entry.get('provenance', [])) or "-"
            lines.append(
                f"{nid} & {entry.get('level', '-')} & "
                f"{entry.get('dimension', '-')} & "
                f"{entry.get('tau', 0):.3f} & {entry.get('gamma', 0):.3f} & "
                f"{entry.get('feature_norm', 0):.3f} & {prov} \\\\")
        lines.extend([r"\bottomrule", r"\end{longtable}"])
        return "\n".join(lines) + "\n"

    def format_contribution_breakdown(self, breakdown: dict) -> str:
        lines = [
            r"\subsection*{Contribution Breakdown}",
            r"\begin{table}[ht]",
            r"\centering",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Component & Value & Fraction \\",
            r"\midrule",
        ]
        components = breakdown.get("components", {})
        fractions = breakdown.get("fractions", {})
        for key in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            val = components.get(key, 0.0)
            frac = fractions.get(key, 0.0)
            lines.append(f"{key.capitalize()} & {val:.4f} & {frac:.4f} \\\\")
        lines.extend([
            r"\midrule",
            f"LDRI & {breakdown.get('ldri', 0):.4f} & --- \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            (f"\nConfidence $\\Gamma$ = {breakdown.get('gamma_global', 0):.4f}, "
             f"Reliability = {breakdown.get('reliability', 0):.4f}\n"),
        ])
        return "\n".join(lines) + "\n"

    def format_full(self, data: dict) -> str:
        parts = [r"\documentclass{article}",
                 r"\usepackage{booktabs,longtable}",
                 r"\begin{document}",
                 r"\title{LIT-DISC Explanation Report}",
                 r"\maketitle"]
        if "contribution_breakdown" in data:
            parts.append(self.format_contribution_breakdown(data["contribution_breakdown"]))
        if "signal_lineage" in data:
            parts.append(self.format_signal_lineage(data["signal_lineage"]))
        if "uncertainty_evolution" in data:
            parts.append(self._format_uncertainty_evolution(data["uncertainty_evolution"]))
        parts.append(r"\end{document}")
        return "\n".join(parts) + "\n"

    def _format_uncertainty_evolution(self, evolution: list[dict]) -> str:
        lines = [
            r"\subsection*{Uncertainty Evolution}",
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"Level & Label & Mean Var & Mean $\gamma$ \\",
            r"\midrule",
        ]
        for entry in evolution:
            stats = entry.get("stats", {})
            lines.append(
                f"{entry.get('level', '-')} & {entry.get('label', '')} & "
                f"{stats.get('mean', 0):.4f} & {entry.get('mean_confidence', 0):.4f} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(lines) + "\n"


class HTMLFormatter:
    _CSS = """
    <style>
      body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #222; background: #fafafa; }
      h2 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: .3em; margin-top: 2em; }
      h3 { color: #2c3e50; margin-top: 1.5em; }
      table { border-collapse: collapse; width: 100%; margin: 1em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
      th, td { padding: .5em .75em; text-align: left; border-bottom: 1px solid #ddd; font-size: .9em; }
      th { background: #2980b9; color: white; font-weight: 600; }
      tr:hover { background: #f0f7ff; }
      .section { margin: 2em 0; padding: 1em; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
      .summary { display: inline-block; margin: .5em; padding: .75em 1.25em; background: #eaf2f8; border-radius: 4px; }
      .summary strong { display: block; font-size: .85em; color: #555; }
      .summary span { font-size: 1.4em; font-weight: 700; color: #1a5276; }
      .empty { color: #999; font-style: italic; }
    </style>"""

    def format_full(self, data: dict) -> str:
        parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'>",
            "<title>LIT-DISC Explanation Report</title>",
            self._CSS, "</head><body>",
            "<h1>LIT-DISC Explanation Report</h1>",
        ]
        if "contribution_breakdown" in data:
            parts.append(self._section("Contribution Breakdown",
                                       self._format_contribution_breakdown(data["contribution_breakdown"])))
        if "signal_lineage" in data:
            parts.append(self._section("Signal Lineage",
                                       self._format_signal_lineage(data["signal_lineage"])))
        if "propagation_path" in data:
            parts.append(self._section("Propagation Path",
                                       self._format_propagation_path(data["propagation_path"])))
        if "interaction_history" in data:
            parts.append(self._section("Interaction History",
                                       self._format_interaction_history(data["interaction_history"])))
        if "memory_activations" in data:
            parts.append(self._section("Memory Activations",
                                       self._format_memory_activations(data["memory_activations"])))
        if "resonance_contribution" in data:
            parts.append(self._section("Resonance Contribution",
                                       self._format_resonance_contribution(data["resonance_contribution"])))
        if "uncertainty_evolution" in data:
            parts.append(self._section("Uncertainty Evolution",
                                       self._format_uncertainty_evolution(data["uncertainty_evolution"])))
        parts.append("</body></html>")
        return "\n".join(parts)

    @staticmethod
    def _section(title: str, content: str) -> str:
        return f"<div class='section'><h2>{title}</h2>{content}</div>"

    @staticmethod
    def _format_contribution_breakdown(breakdown: dict) -> str:
        lines = ["<table>",
                 "<tr><th>Component</th><th>Value</th><th>Fraction</th></tr>"]
        components = breakdown.get("components", {})
        fractions = breakdown.get("fractions", {})
        for key in ["intrinsic", "interaction", "temporal", "resonance", "memory", "uncertainty"]:
            val = components.get(key, 0.0)
            frac = fractions.get(key, 0.0)
            lines.append(f"<tr><td>{key.capitalize()}</td><td>{val:.4f}</td><td>{frac:.4f}</td></tr>")
        lines.append("</table>")
        lines.append(
            f"<div class='summary'><strong>LDRI</strong><span>{breakdown.get('ldri', 0):.4f}</span></div>"
            f"<div class='summary'><strong>Raw</strong><span>{breakdown.get('raw', 0):.4f}</span></div>"
            f"<div class='summary'><strong>&Gamma;</strong><span>{breakdown.get('gamma_global', 0):.4f}</span></div>"
            f"<div class='summary'><strong>Reliability</strong><span>{breakdown.get('reliability', 0):.4f}</span></div>"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_signal_lineage(lineage: list[dict]) -> str:
        if not lineage:
            return "<p class='empty'>No signal lineage available.</p>"
        lines = ["<table>",
                 "<tr><th>Node</th><th>Level</th><th>Dim</th><th>&tau;</th><th>&gamma;</th><th>&#x2016;f&#x2016;</th><th>Provenance</th></tr>"]
        for entry in lineage:
            nid = f"({entry['node_id'][0]},{entry['node_id'][1]})" if entry.get('node_id') else "-"
            prov = ", ".join(f"({p[0]},{p[1]})" for p in entry.get('provenance', [])) or "-"
            lines.append(
                f"<tr><td>{nid}</td><td>{entry.get('level', '-')}</td>"
                f"<td>{entry.get('dimension', '-')}</td>"
                f"<td>{entry.get('tau', 0):.3f}</td><td>{entry.get('gamma', 0):.3f}</td>"
                f"<td>{entry.get('feature_norm', 0):.3f}</td><td>{prov}</td></tr>")
        lines.append("</table>")
        return "\n".join(lines)

    @staticmethod
    def _format_propagation_path(path: dict) -> str:
        lines = []
        hierarchy = path.get("hierarchy", {})
        for level_key in sorted(hierarchy.keys()):
            level_data = hierarchy[level_key]
            lines.append(f"<h3>{level_data['label']} ({level_data['node_count']} nodes)</h3>")
            for node in level_data.get("nodes", []):
                nid = node.get("node_id", [])
                tau = node.get("tau", 0.5)
                dims = node.get("dimensions", {})
                if dims:
                    dim_parts = [f"{d}: &gamma;={info['gamma']:.3f}, &#x2016;f&#x2016;={info['norm']:.3f}"
                                 for d, info in sorted(dims.items())]
                    lines.append(f"<p>Node ({nid[0]},{nid[1]}) &tau;={tau:.3f}: {', '.join(dim_parts)}</p>")
                else:
                    lines.append(f"<p>Node ({nid[0]},{nid[1]}) &tau;={tau:.3f}: <em>no signals</em></p>")
        return "\n".join(lines)

    @staticmethod
    def _format_interaction_history(history: dict) -> str:
        interactions = history.get("interactions", [])
        if not interactions:
            return "<p class='empty'>No cross-dimensional interactions recorded.</p>"
        lines = ["<table>",
                 "<tr><th>Node</th><th>Level</th><th>Pair</th><th>Cosine</th><th>&gamma;&#x2081;</th><th>&gamma;&#x2082;</th></tr>"]
        for entry in interactions:
            nid = f"({entry['node_id'][0]},{entry['node_id'][1]})"
            lines.append(
                f"<tr><td>{nid}</td><td>{entry.get('level', '-')}</td>"
                f"<td>{entry['dimension_pair']}</td>"
                f"<td>{entry.get('cosine_similarity', 0):.4f}</td>"
                f"<td>{entry.get('gamma_1', 0):.3f}</td><td>{entry.get('gamma_2', 0):.3f}</td></tr>")
        lines.append("</table>")
        summary = history.get("summary", {})
        if summary:
            lines.append(f"<p><strong>Summary:</strong> {summary.get('count', 0)} interactions, "
                         f"mean cosine = {summary.get('mean_cosine', 0):.4f}</p>")
        return "\n".join(lines)

    @staticmethod
    def _format_memory_activations(activations: list[dict]) -> str:
        if not activations:
            return "<p class='empty'>No memory entries.</p>"
        lines = ["<table>",
                 "<tr><th>&beta;</th><th>Stored &tau;</th><th>Retrievals</th><th>Recency</th></tr>"]
        for entry in activations:
            lines.append(
                f"<tr><td>{entry.get('beta', 0):.4f}</td>"
                f"<td>{entry.get('storage_tau', 0):.3f}</td>"
                f"<td>{entry.get('retrieval_count', 0)}</td>"
                f"<td>{entry.get('recency', 0):.3f}</td></tr>")
        lines.append("</table>")
        return "\n".join(lines)

    @staticmethod
    def _format_resonance_contribution(contribution: dict) -> str:
        lines = [f"<p><strong>Current Norm:</strong> {contribution.get('resonance_current_norm', 0):.4f}</p>",
                 f"<p><strong>Trajectory Length:</strong> {contribution.get('trajectory_length', 0)}</p>"]
        aug = contribution.get("signal_augmentation", {})
        lines.append(f"<p><strong>Augmentation:</strong> {aug.get('description', 'N/A')}</p>")
        return "\n".join(lines)

    @staticmethod
    def _format_uncertainty_evolution(evolution: list[dict]) -> str:
        if not evolution:
            return "<p class='empty'>No uncertainty data.</p>"
        lines = ["<table>",
                 "<tr><th>Level</th><th>Label</th><th>Count</th><th>Mean Var</th><th>Mean &gamma;</th></tr>"]
        for entry in evolution:
            stats = entry.get("stats", {})
            lines.append(
                f"<tr><td>{entry.get('level', '-')}</td><td>{entry.get('label', '')}</td>"
                f"<td>{stats.get('count', 0)}</td><td>{stats.get('mean', 0):.4f}</td>"
                f"<td>{entry.get('mean_confidence', 0):.4f}</td></tr>")
        lines.append("</table>")
        return "\n".join(lines)
