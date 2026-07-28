from src.explainability.formatters import (
    HTMLFormatter,
    JSONFormatter,
    LaTeXFormatter,
    MarkdownFormatter,
)
from src.explainability.trace import (
    CompositeExplanationGenerator,
    ContributionBreakdownGenerator,
    InteractionHistoryGenerator,
    MemoryActivationGenerator,
    PropagationPathGenerator,
    ResonanceContributionGenerator,
    SignalLineageGenerator,
    UncertaintyEvolutionGenerator,
)


class ExplainabilityEngine:
    def __init__(self):
        self.generator = CompositeExplanationGenerator()
        self.json = JSONFormatter()
        self.markdown = MarkdownFormatter()
        self.latex = LaTeXFormatter()
        self.html = HTMLFormatter()

        self.signal_lineage = SignalLineageGenerator()
        self.propagation_path = PropagationPathGenerator()
        self.interaction_history = InteractionHistoryGenerator()
        self.memory_activation = MemoryActivationGenerator()
        self.resonance_contribution = ResonanceContributionGenerator()
        self.uncertainty_evolution = UncertaintyEvolutionGenerator()
        self.contribution_breakdown = ContributionBreakdownGenerator()

    def generate_all(self, signals, tree, ldri_score,
                     memory_store=None, resonance_engine=None,
                     uncertainty_propagator=None) -> dict:
        return self.generator.as_dict(
            signals, tree, ldri_score,
            memory_store, resonance_engine, uncertainty_propagator,
        )

    def generate_contribution_breakdown(self, ldri_score) -> dict:
        return self.contribution_breakdown.generate(ldri_score)

    def generate_signal_lineage(self, signals, tree) -> list[dict]:
        return self.signal_lineage.generate(signals, tree)

    def generate_propagation_path(self, signals, tree) -> dict:
        return self.propagation_path.generate(signals, tree)

    def generate_interaction_history(self, signals, tree) -> dict:
        return self.interaction_history.generate(signals, tree)

    def generate_memory_activations(self, memory_store, **kw) -> list[dict]:
        return self.memory_activation.generate(memory_store, **kw)

    def generate_resonance_contribution(self, resonance_engine, signals, tree) -> dict:
        return self.resonance_contribution.generate(resonance_engine, signals, tree)

    def generate_uncertainty_evolution(self, up, signals, tree) -> list[dict]:
        return self.uncertainty_evolution.generate(up, signals, tree)

    def to_json(self, data: dict) -> str:
        return self.json.format_full(data)

    def to_markdown(self, data: dict) -> str:
        return self.markdown.format_full(data)

    def to_latex(self, data: dict) -> str:
        return self.latex.format_full(data)

    def to_html(self, data: dict) -> str:
        return self.html.format_full(data)

    def export_all(self, data: dict, formats: list[str]) -> dict[str, str]:
        results = {}
        if "json" in formats:
            results["json"] = self.to_json(data)
        if "md" in formats or "markdown" in formats:
            results["markdown"] = self.to_markdown(data)
        if "tex" in formats or "latex" in formats:
            results["latex"] = self.to_latex(data)
        if "html" in formats:
            results["html"] = self.to_html(data)
        return results
