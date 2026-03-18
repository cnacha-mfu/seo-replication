"""
Manifest Generator for SENSO Framework — Paper 1, Phase 2

Generates evolution-aware CLAUDE.md sections by querying a populated
Software Evolution Ontology (SEO) in Apache Jena Fuseki and synthesizing
natural language summaries via NVIDIA NIM LLM.

Pipeline:
  1. FusekiQuerier — SPARQL queries for per-module evolution data
  2. ManifestSynthesizer — NIM LLM or template-based natural language generation
  3. ManifestGenerator — orchestrates querying + synthesis for all modules
  4. ManifestWriter — assembles and writes the final CLAUDE.md file

Used by: Paper 1 (RQ3 manifest generation, RQ4 agentic experiment conditions)

Usage:
    python generate.py --repo owner/name --fuseki-url http://localhost:3030 --dataset seo --output manifests/generated/
    python generate.py --repo owner/name --fuseki-url http://localhost:3030 --dataset seo --no-llm --output manifests/generated/
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("senso.manifest_generator")

# ── Constants ────────────────────────────────────────────────────────────────

SEO_IRI = "http://senso-framework.org/ontology/seo#"
SEO_PREFIX = f"PREFIX seo: <{SEO_IRI}>"

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "meta/llama-3.3-70b-instruct"

# Lehman's law labels for human-readable output
LEHMAN_LAW_LABELS = {
    "LawI": "Continuing Change",
    "LawII": "Increasing Complexity",
    "LawIII": "Self Regulation",
    "LawIV": "Conservation of Organizational Stability",
    "LawV": "Conservation of Familiarity",
    "LawVI": "Continuing Growth",
    "LawVII": "Declining Quality",
    "LawVIII": "Feedback System",
}

# Anti-pattern display names
PATTERN_LABELS = {
    "ShadowTechDebt": "Shadow Tech Debt",
    "DependencySprawl": "Dependency Sprawl",
    "RefactoringAvoidance": "Refactoring Avoidance",
    "CopyPasteOverReuse": "Copy-Paste Over Reuse",
    "ArchitecturalErosion": "Architectural Erosion",
    "BigBangRefactoring": "Big-Bang Refactoring",
    "GradualDecay": "Gradual Decay",
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ComplexityData:
    """Complexity metrics for a module. Used by Paper 1 manifest generation."""
    current: float = 0.0
    six_month_delta_pct: float = 0.0
    project_percentile: float = 0.0


@dataclass
class CouplingData:
    """Coupling metrics for a module. Used by Paper 1 manifest generation."""
    fan_in: int = 0
    fan_out: int = 0
    dependency_count: int = 0
    project_p75_fan_out: Optional[float] = None
    decoupling_targets: list[str] = field(default_factory=list)


@dataclass
class RefactoringData:
    """Refactoring history for a module. Used by Paper 1 manifest generation."""
    total_count: int = 0
    major_count: int = 0
    last_impact_pct: Optional[float] = None
    regression_count: int = 0
    last_date: Optional[str] = None


@dataclass
class ModuleEvolutionData:
    """All evolution data for a single module. Used by Paper 1 manifest generation."""
    module_name: str
    evolution_phase: str = "unknown"
    phase_since: Optional[str] = None
    complexity: ComplexityData = field(default_factory=ComplexityData)
    coupling: CouplingData = field(default_factory=CouplingData)
    refactoring: RefactoringData = field(default_factory=RefactoringData)
    agent_contribution_pct: float = 0.0
    law_conformance: dict[str, bool] = field(default_factory=dict)
    active_laws: list[str] = field(default_factory=list)
    detected_patterns: list[str] = field(default_factory=list)


# ── SPARQL Queries ───────────────────────────────────────────────────────────

SPARQL_LIST_MODULES = """
{seo_prefix}
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?module ?moduleName WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label ?moduleName .
    ?project seo:repoFullName "{repo}" .
}}
ORDER BY ?moduleName
"""

SPARQL_EVOLUTION_PHASE = """
{seo_prefix}
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?phase ?phaseSince WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?episode a ?phaseClass ;
             seo:appliesTo ?module ;
             seo:startDate ?phaseSince .
    VALUES ?phaseClass {{ seo:GrowthPhase seo:StabilizationPhase seo:DeclinePhase }}
    BIND(
        IF(?phaseClass = seo:GrowthPhase, "growth",
        IF(?phaseClass = seo:StabilizationPhase, "stabilization",
        "decline"))
        AS ?phase
    )
}}
ORDER BY DESC(?phaseSince)
LIMIT 1
"""

SPARQL_COMPLEXITY = """
{seo_prefix}
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?currentCC ?deltaPercent ?percentile WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?metric a seo:ComplexityMetric ;
            seo:measuredOn ?module ;
            seo:currentValue ?currentCC ;
            seo:sixMonthDeltaPercent ?deltaPercent ;
            seo:projectPercentile ?percentile .
}}
LIMIT 1
"""

SPARQL_COUPLING = """
{seo_prefix}

SELECT ?fanIn ?fanOut ?depCount ?p75FanOut WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?metric a seo:CouplingMetric ;
            seo:measuredOn ?module ;
            seo:fanIn ?fanIn ;
            seo:fanOut ?fanOut ;
            seo:dependencyCount ?depCount .
    OPTIONAL {{ ?metric seo:projectP75FanOut ?p75FanOut . }}
}}
LIMIT 1
"""

SPARQL_COUPLING_TARGETS = """
{seo_prefix}
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?targetName WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?module seo:decouplingTarget ?target .
    ?target rdfs:label ?targetName .
}}
"""

SPARQL_REFACTORING = """
{seo_prefix}
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT (COUNT(?refactoring) AS ?totalCount)
       (SUM(IF(?isMajor, 1, 0)) AS ?majorCount)
       (MAX(?lastImpact) AS ?lastImpactPct)
       (SUM(?regressions) AS ?regressionCount)
       (MAX(?refDate) AS ?lastDate)
WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?refactoring a seo:RefactoringEvent ;
                 seo:appliesTo ?module ;
                 seo:eventDate ?refDate .
    OPTIONAL {{ ?refactoring seo:isMajor ?isMajor . }}
    OPTIONAL {{ ?refactoring seo:complexityImpactPercent ?lastImpact . }}
    OPTIONAL {{ ?refactoring seo:regressionCount ?regressions . }}
}}
"""

SPARQL_AGENT_CONTRIBUTION = """
{seo_prefix}

SELECT ?agentPct WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?ratio a seo:AgentContributionRatio ;
           seo:measuredOn ?module ;
           seo:ratioValue ?agentPct .
}}
LIMIT 1
"""

SPARQL_LAW_CONFORMANCE = """
{seo_prefix}

SELECT ?lawName ?isConforming WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?conformance a seo:LawConformance ;
                 seo:appliesTo ?module ;
                 seo:lawName ?lawName ;
                 seo:isConforming ?isConforming .
}}
"""

SPARQL_DETECTED_PATTERNS = """
{seo_prefix}

SELECT ?patternType WHERE {{
    ?module a seo:Module ;
            seo:belongsToProject ?project ;
            rdfs:label "{module_name}" .
    ?project seo:repoFullName "{repo}" .
    ?pattern seo:detectedIn ?module ;
             a ?patternType .
    FILTER(?patternType != <http://www.w3.org/2002/07/owl#NamedIndividual>)
}}
"""


# ── FusekiQuerier ────────────────────────────────────────────────────────────

class FusekiQuerier:
    """Executes SPARQL queries against Apache Jena Fuseki to retrieve
    per-module evolution data from the populated Software Evolution Ontology.

    Used by: Paper 1 (manifest generation pipeline)
    """

    def __init__(self, fuseki_url: str, dataset: str):
        """Initialize Fuseki connection.

        Args:
            fuseki_url: Base URL of the Fuseki server (e.g. http://localhost:3030).
            dataset: Name of the Fuseki dataset containing SEO triples.
        """
        self.fuseki_url = fuseki_url.rstrip("/")
        self.dataset = dataset
        self.endpoint = f"{self.fuseki_url}/{self.dataset}/sparql"
        logger.info("FusekiQuerier initialized: endpoint=%s", self.endpoint)

    def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a SPARQL SELECT query and return results as list of dicts.

        Args:
            query: SPARQL SELECT query string.

        Returns:
            List of result bindings as dictionaries.
        """
        try:
            from SPARQLWrapper import SPARQLWrapper, JSON
        except ImportError:
            logger.error("SPARQLWrapper is required: pip install sparqlwrapper")
            raise

        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
        except Exception:
            logger.exception("SPARQL query failed against %s", self.endpoint)
            raise

        bindings = results.get("results", {}).get("bindings", [])
        parsed: list[dict[str, Any]] = []
        for binding in bindings:
            row: dict[str, Any] = {}
            for var_name, val_info in binding.items():
                raw = val_info.get("value", "")
                dtype = val_info.get("datatype", "")
                if "integer" in dtype:
                    row[var_name] = int(raw)
                elif "float" in dtype or "double" in dtype or "decimal" in dtype:
                    row[var_name] = float(raw)
                elif "boolean" in dtype:
                    row[var_name] = raw.lower() in ("true", "1")
                else:
                    row[var_name] = raw
            parsed.append(row)

        return parsed

    def _format_query(self, template: str, **kwargs: str) -> str:
        """Format a SPARQL query template with the SEO prefix and parameters."""
        return template.format(seo_prefix=SEO_PREFIX, **kwargs)

    def list_modules(self, repo: str) -> list[str]:
        """List all module names for a given repository.

        Args:
            repo: Repository full name (owner/name).

        Returns:
            List of module name strings.
        """
        query = self._format_query(SPARQL_LIST_MODULES, repo=repo)
        results = self._execute_query(query)
        modules = [r["moduleName"] for r in results if "moduleName" in r]
        logger.info("Found %d modules for repo %s", len(modules), repo)
        return modules

    def query_module_data(self, repo: str, module_name: str) -> ModuleEvolutionData:
        """Query all evolution data for a single module.

        Args:
            repo: Repository full name (owner/name).
            module_name: Module name (as stored in ontology rdfs:label).

        Returns:
            ModuleEvolutionData with all available evolution context.
        """
        data = ModuleEvolutionData(module_name=module_name)

        # Evolution phase
        phase_results = self._execute_query(
            self._format_query(SPARQL_EVOLUTION_PHASE, repo=repo, module_name=module_name)
        )
        if phase_results:
            data.evolution_phase = phase_results[0].get("phase", "unknown")
            raw_since = phase_results[0].get("phaseSince", "")
            if raw_since:
                data.phase_since = raw_since[:10]  # YYYY-MM-DD or YYYY-MM

        # Complexity
        cx_results = self._execute_query(
            self._format_query(SPARQL_COMPLEXITY, repo=repo, module_name=module_name)
        )
        if cx_results:
            r = cx_results[0]
            data.complexity = ComplexityData(
                current=r.get("currentCC", 0.0),
                six_month_delta_pct=r.get("deltaPercent", 0.0),
                project_percentile=r.get("percentile", 0.0),
            )

        # Coupling
        coup_results = self._execute_query(
            self._format_query(SPARQL_COUPLING, repo=repo, module_name=module_name)
        )
        if coup_results:
            r = coup_results[0]
            data.coupling = CouplingData(
                fan_in=r.get("fanIn", 0),
                fan_out=r.get("fanOut", 0),
                dependency_count=r.get("depCount", 0),
                project_p75_fan_out=r.get("p75FanOut"),
            )

        # Decoupling targets
        target_results = self._execute_query(
            self._format_query(SPARQL_COUPLING_TARGETS, repo=repo, module_name=module_name)
        )
        data.coupling.decoupling_targets = [
            r["targetName"] for r in target_results if "targetName" in r
        ]

        # Refactoring history
        ref_results = self._execute_query(
            self._format_query(SPARQL_REFACTORING, repo=repo, module_name=module_name)
        )
        if ref_results:
            r = ref_results[0]
            data.refactoring = RefactoringData(
                total_count=r.get("totalCount", 0),
                major_count=r.get("majorCount", 0),
                last_impact_pct=r.get("lastImpactPct"),
                regression_count=r.get("regressionCount", 0),
                last_date=r.get("lastDate", "")[:10] if r.get("lastDate") else None,
            )

        # Agent contribution
        agent_results = self._execute_query(
            self._format_query(SPARQL_AGENT_CONTRIBUTION, repo=repo, module_name=module_name)
        )
        if agent_results:
            data.agent_contribution_pct = agent_results[0].get("agentPct", 0.0)

        # Law conformance
        law_results = self._execute_query(
            self._format_query(SPARQL_LAW_CONFORMANCE, repo=repo, module_name=module_name)
        )
        for r in law_results:
            law_name = r.get("lawName", "")
            # Strip IRI prefix if present
            if "#" in law_name:
                law_name = law_name.split("#")[-1]
            is_conforming = r.get("isConforming", True)
            data.law_conformance[law_name] = is_conforming
            if not is_conforming:
                data.active_laws.append(law_name)

        # Detected patterns
        pattern_results = self._execute_query(
            self._format_query(SPARQL_DETECTED_PATTERNS, repo=repo, module_name=module_name)
        )
        for r in pattern_results:
            pt = r.get("patternType", "")
            if "#" in pt:
                pt = pt.split("#")[-1]
            if pt:
                data.detected_patterns.append(pt)

        logger.info(
            "Queried module %s: phase=%s, complexity_delta=%.1f%%, agent_pct=%.1f%%",
            module_name,
            data.evolution_phase,
            data.complexity.six_month_delta_pct,
            data.agent_contribution_pct,
        )
        return data


# ── ManifestSynthesizer ─────────────────────────────────────────────────────

class ManifestSynthesizer:
    """Synthesizes natural language evolution summaries from structured module data.

    Supports two modes:
    - LLM mode: calls NVIDIA NIM (meta/llama-3.3-70b-instruct) for rich prose
    - Template mode (--no-llm): deterministic template-based generation for testing

    Used by: Paper 1 (manifest generation, RQ3 comparison, RQ4 experiment conditions)
    """

    def __init__(self, use_llm: bool = True, model: str = NIM_MODEL):
        """Initialize the synthesizer.

        Args:
            use_llm: If True, use NIM LLM for synthesis. If False, use templates.
            model: NIM model identifier for LLM synthesis.
        """
        self.use_llm = use_llm
        self.model = model
        self._client = None

        if self.use_llm:
            self._init_nim_client()

    def _init_nim_client(self) -> None:
        """Initialize the NVIDIA NIM OpenAI-compatible client."""
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            logger.error("NVIDIA_API_KEY not set. Use --no-llm or set it in .env")
            raise EnvironmentError("NVIDIA_API_KEY environment variable is required for LLM mode")

        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)
            logger.info("NIM client initialized: model=%s", self.model)
        except ImportError:
            logger.error("openai package required: pip install openai")
            raise

    def synthesize(self, module_data: ModuleEvolutionData) -> str:
        """Generate a natural language evolution summary for a module.

        Args:
            module_data: Structured evolution data for one module.

        Returns:
            Markdown-formatted evolution summary string.
        """
        if self.use_llm:
            return self._synthesize_llm(module_data)
        return self._synthesize_template(module_data)

    def _build_prompt(self, data: ModuleEvolutionData) -> str:
        """Build an LLM prompt from structured module evolution data.

        Args:
            data: Module evolution data.

        Returns:
            Prompt string for the LLM.
        """
        # Build structured context block
        context_parts = [
            f"Module: {data.module_name}",
            f"Evolution Phase: {data.evolution_phase}",
        ]
        if data.phase_since:
            context_parts.append(f"Phase Since: {data.phase_since}")

        context_parts.extend([
            f"Cyclomatic Complexity (current): {data.complexity.current:.1f}",
            f"Complexity 6-month delta: {data.complexity.six_month_delta_pct:+.1f}%",
            f"Complexity project percentile: p{data.complexity.project_percentile:.0f}",
            f"Fan-in: {data.coupling.fan_in}",
            f"Fan-out: {data.coupling.fan_out}",
            f"Dependency count: {data.coupling.dependency_count}",
        ])

        if data.coupling.project_p75_fan_out is not None:
            context_parts.append(
                f"Project p75 fan-out: {data.coupling.project_p75_fan_out:.0f}"
            )
        if data.coupling.decoupling_targets:
            context_parts.append(
                f"Decoupling in progress from: {', '.join(data.coupling.decoupling_targets)}"
            )

        context_parts.extend([
            f"Total refactorings: {data.refactoring.total_count}",
            f"Major refactorings: {data.refactoring.major_count}",
        ])
        if data.refactoring.last_impact_pct is not None:
            context_parts.append(
                f"Last refactoring complexity impact: {data.refactoring.last_impact_pct:+.1f}%"
            )
        if data.refactoring.regression_count > 0:
            context_parts.append(
                f"Refactoring-caused regressions: {data.refactoring.regression_count}"
            )

        context_parts.append(f"Agent contribution: {data.agent_contribution_pct:.1f}% of recent commits")

        if data.active_laws:
            law_labels = [
                f"{law} ({LEHMAN_LAW_LABELS.get(law, 'Unknown')})"
                for law in data.active_laws
            ]
            context_parts.append(f"Active Lehman's laws (non-conforming): {', '.join(law_labels)}")

        if data.detected_patterns:
            pattern_labels = [PATTERN_LABELS.get(p, p) for p in data.detected_patterns]
            context_parts.append(f"Detected anti-patterns: {', '.join(pattern_labels)}")

        context_block = "\n".join(context_parts)

        prompt = f"""You are an expert software evolution analyst. Given the following structured evolution data for a software module, generate a concise markdown summary for a CLAUDE.md evolution manifest. The summary should help AI coding agents understand the module's evolution state and make architecture-aware decisions.

The summary must include exactly these sections as markdown bullet points:
- **Evolution Phase**: phase name, when it started, what it implies
- **Complexity Trend**: current complexity, recent trend, which Lehman law is active if relevant
- **Coupling Risk**: fan-out level, whether above project average, any decoupling efforts
- **Refactoring History**: count and outcomes of past refactoring attempts, regressions
- **Agent Contribution**: percentage of recent commits by coding agents
- **Recommendation**: 1-2 actionable sentences for a coding agent working in this module

Output ONLY the bullet points (no heading, no intro, no extra commentary). Each bullet must start with "- **Label**: ".

=== STRUCTURED DATA ===
{context_block}
=== END DATA ==="""
        return prompt

    def _synthesize_llm(self, data: ModuleEvolutionData) -> str:
        """Synthesize using NIM LLM.

        Args:
            data: Module evolution data.

        Returns:
            LLM-generated markdown bullet points.
        """
        prompt = self._build_prompt(data)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You produce concise, factual CLAUDE.md evolution summaries for AI coding agents. Output markdown bullet points only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            content = response.choices[0].message.content.strip()
            logger.info("LLM synthesis complete for module %s (%d chars)", data.module_name, len(content))
            return content
        except Exception:
            logger.exception("LLM synthesis failed for module %s, falling back to template", data.module_name)
            return self._synthesize_template(data)

    def _synthesize_template(self, data: ModuleEvolutionData) -> str:
        """Generate a template-based summary without LLM (--no-llm mode).

        Args:
            data: Module evolution data.

        Returns:
            Template-generated markdown bullet points.
        """
        lines: list[str] = []

        # Evolution Phase
        phase_str = data.evolution_phase.capitalize()
        if data.phase_since:
            phase_str += f" (since {data.phase_since})"
        lines.append(f"- **Evolution Phase**: {phase_str}")

        # Complexity Trend
        delta = data.complexity.six_month_delta_pct
        direction = "up" if delta > 0 else "down" if delta < 0 else "stable"
        law_note = ""
        if "LawII" in data.active_laws:
            law_note = " (Law II active)"
        lines.append(
            f"- **Complexity Trend**: Cyclomatic complexity {delta:+.0f}% in 6 months, "
            f"currently {data.complexity.current:.1f} (project p{data.complexity.project_percentile:.0f})"
            f"{law_note}"
        )

        # Coupling Risk
        coupling_parts = [f"Fan-out = {data.coupling.fan_out}"]
        if (
            data.coupling.project_p75_fan_out is not None
            and data.coupling.fan_out > data.coupling.project_p75_fan_out
        ):
            coupling_parts.append(f"above project p75 ({data.coupling.project_p75_fan_out:.0f})")
        if data.coupling.decoupling_targets:
            coupling_parts.append(
                f"Decoupling from {', '.join(data.coupling.decoupling_targets)} in progress"
            )
        coupling_parts.append(f"Fan-in = {data.coupling.fan_in}")
        coupling_parts.append(f"{data.coupling.dependency_count} dependencies")
        lines.append(f"- **Coupling Risk**: {'. '.join(coupling_parts)}")

        # Refactoring History
        if data.refactoring.total_count > 0:
            ref_str = f"{data.refactoring.major_count} major refactoring(s) out of {data.refactoring.total_count} total"
            if data.refactoring.last_impact_pct is not None:
                ref_str += f". Last reduced complexity {abs(data.refactoring.last_impact_pct):.0f}%"
            if data.refactoring.regression_count > 0:
                ref_str += f" but caused {data.refactoring.regression_count} regression(s)"
            lines.append(f"- **Refactoring History**: {ref_str}")
        else:
            lines.append("- **Refactoring History**: No refactoring events recorded")

        # Agent Contribution
        lines.append(
            f"- **Agent Contribution**: {data.agent_contribution_pct:.0f}% of recent commits by coding agents"
        )

        # Recommendation
        recommendations = self._generate_template_recommendations(data)
        lines.append(f"- **Recommendation**: {' '.join(recommendations)}")

        return "\n".join(lines)

    def _generate_template_recommendations(self, data: ModuleEvolutionData) -> list[str]:
        """Generate rule-based recommendations from structured data.

        Args:
            data: Module evolution data.

        Returns:
            List of recommendation sentences.
        """
        recs: list[str] = []

        # Phase-based recommendations
        if data.evolution_phase == "stabilization":
            recs.append("Avoid new dependencies.")
        elif data.evolution_phase == "decline":
            recs.append("Consider deprecation or major refactoring.")
        elif data.evolution_phase == "growth":
            recs.append("Ensure new features include tests.")

        # Complexity recommendations
        if data.complexity.six_month_delta_pct > 20:
            recs.append("Prefer decomposition over extension.")
        if data.complexity.project_percentile > 75:
            recs.append("Module is among the most complex in the project.")

        # Coupling recommendations
        if (
            data.coupling.project_p75_fan_out is not None
            and data.coupling.fan_out > data.coupling.project_p75_fan_out
        ):
            recs.append("Reduce outgoing dependencies before adding new ones.")

        # Refactoring recommendations
        if data.refactoring.regression_count > 0:
            recs.append("Full integration tests required for changes.")

        # Pattern-based recommendations
        if "ShadowTechDebt" in data.detected_patterns:
            recs.append("Review agent-generated code for hidden tech debt.")
        if "DependencySprawl" in data.detected_patterns:
            recs.append("Consolidate dependencies before adding new ones.")

        if not recs:
            recs.append("No specific concerns. Follow standard development practices.")

        return recs


# ── ManifestGenerator ────────────────────────────────────────────────────────

class ManifestGenerator:
    """Orchestrates querying Fuseki and synthesizing evolution manifests
    for all modules in a repository.

    Used by: Paper 1 (manifest generation pipeline, RQ3, RQ4)
    """

    def __init__(
        self,
        querier: FusekiQuerier,
        synthesizer: ManifestSynthesizer,
    ):
        """Initialize the manifest generator.

        Args:
            querier: FusekiQuerier instance for SPARQL queries.
            synthesizer: ManifestSynthesizer instance for natural language generation.
        """
        self.querier = querier
        self.synthesizer = synthesizer

    def generate(self, repo: str) -> dict[str, str]:
        """Generate evolution summaries for all modules in a repository.

        Args:
            repo: Repository full name (owner/name).

        Returns:
            Dict mapping module name to its markdown summary string.
        """
        modules = self.querier.list_modules(repo)

        if not modules:
            logger.warning("No modules found for repo %s in Fuseki", repo)
            return {}

        summaries: dict[str, str] = {}
        for module_name in modules:
            logger.info("Generating manifest for module: %s", module_name)
            module_data = self.querier.query_module_data(repo, module_name)
            summary = self.synthesizer.synthesize(module_data)
            summaries[module_name] = summary

        logger.info(
            "Generated manifests for %d modules in %s", len(summaries), repo
        )
        return summaries


# ── ManifestWriter ───────────────────────────────────────────────────────────

class ManifestWriter:
    """Assembles per-module summaries into a complete CLAUDE.md evolution section
    and writes it to disk.

    Used by: Paper 1 (manifest generation pipeline)
    """

    HEADER = "## Evolution Context (auto-generated by SENSO)\n"
    FOOTER = (
        "\n---\n"
        "*Generated by SENSO Framework — Software Evolution Ontology. "
        "See: Paper 1, \"From Tribal Knowledge to Machine-Readable Evolution Context.\"*\n"
    )

    def __init__(self, output_dir: str):
        """Initialize the writer.

        Args:
            output_dir: Directory where generated CLAUDE.md files are written.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ManifestWriter output dir: %s", self.output_dir)

    def write(
        self,
        repo: str,
        module_summaries: dict[str, str],
        timestamp: Optional[str] = None,
    ) -> Path:
        """Assemble and write a complete CLAUDE.md evolution section.

        Args:
            repo: Repository full name (owner/name).
            module_summaries: Dict mapping module name to markdown summary.
            timestamp: Optional ISO timestamp; defaults to now.

        Returns:
            Path to the written file.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        sections: list[str] = [self.HEADER]
        sections.append(f"**Repository**: `{repo}`  ")
        sections.append(f"**Generated**: {timestamp}\n")

        for module_name in sorted(module_summaries.keys()):
            summary = module_summaries[module_name]
            sections.append(f"### Module: {module_name}\n{summary}\n")

        sections.append(self.FOOTER)

        content = "\n".join(sections)

        # Use repo name (sanitized) as filename
        safe_name = repo.replace("/", "__")
        output_path = self.output_dir / f"{safe_name}_CLAUDE.md"

        output_path.write_text(content, encoding="utf-8")
        logger.info("Manifest written to %s (%d bytes)", output_path, len(content))

        # Also save structured data as JSON sidecar
        json_path = self.output_dir / f"{safe_name}_manifest.json"
        json_data = {
            "repo": repo,
            "generated_at": timestamp,
            "modules": {
                name: summary for name, summary in sorted(module_summaries.items())
            },
        }
        json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
        logger.info("JSON sidecar written to %s", json_path)

        return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str]) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config. If None, returns empty dict.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s — using defaults", config_path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    logger.info("Loaded config from %s", config_path)
    return config


def main() -> None:
    """CLI entry point for manifest generation.

    Usage:
        python generate.py --repo owner/name --fuseki-url http://localhost:3030 --dataset seo --output manifests/generated/
        python generate.py --repo owner/name --no-llm --output manifests/generated/
    """
    parser = argparse.ArgumentParser(
        description="SENSO Manifest Generator — Paper 1, Phase 2. "
        "Generates evolution-aware CLAUDE.md sections from the SEO triple store.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository full name (owner/name).",
    )
    parser.add_argument(
        "--fuseki-url",
        default="http://localhost:3030",
        help="Fuseki server URL (default: http://localhost:3030).",
    )
    parser.add_argument(
        "--dataset",
        default="seo",
        help="Fuseki dataset name (default: seo).",
    )
    parser.add_argument(
        "--output",
        default="manifests/generated/",
        help="Output directory for generated manifests.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use template-based generation (no LLM call). For testing.",
    )
    parser.add_argument(
        "--model",
        default=NIM_MODEL,
        help=f"NIM model for LLM synthesis (default: {NIM_MODEL}).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (overrides CLI args where applicable).",
    )

    args = parser.parse_args()

    # Load .env for API keys
    load_dotenv()

    # Load optional YAML config and merge (CLI args take precedence)
    config = load_config(args.config)
    fuseki_url = config.get("fuseki", {}).get("fuseki_url", args.fuseki_url)
    dataset = config.get("fuseki", {}).get("dataset", args.dataset)
    output_dir = config.get("output", {}).get("output_dir", args.output)
    model = config.get("llm", {}).get("model", args.model)
    use_llm = not args.no_llm
    if not use_llm:
        logger.info("Running in template mode (--no-llm): no LLM calls will be made")

    # Initialize pipeline components
    querier = FusekiQuerier(fuseki_url=fuseki_url, dataset=dataset)
    synthesizer = ManifestSynthesizer(use_llm=use_llm, model=model)
    generator = ManifestGenerator(querier=querier, synthesizer=synthesizer)
    writer = ManifestWriter(output_dir=output_dir)

    # Generate manifests
    logger.info("Starting manifest generation for %s", args.repo)
    module_summaries = generator.generate(args.repo)

    if not module_summaries:
        logger.warning("No module summaries generated for %s — exiting", args.repo)
        sys.exit(1)

    # Write output
    output_path = writer.write(args.repo, module_summaries)
    logger.info("Manifest generation complete: %s", output_path)


if __name__ == "__main__":
    main()
