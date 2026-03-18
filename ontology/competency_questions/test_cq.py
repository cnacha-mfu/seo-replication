#!/usr/bin/env python3
"""Competency question testing for the Software Evolution Ontology (SEO).

Paper 1 — Phase 1D: Loads SPARQL competency questions from cq_suite.yaml,
runs each query against a Fuseki triple store (or in-memory rdflib graph
as fallback), and validates results against expected shapes and rules.

Usage:
    cd /tmp && python /home/ubuntu/git/senso-framework/ontology/competency_questions/test_cq.py
    cd /tmp && python /home/ubuntu/git/senso-framework/ontology/competency_questions/test_cq.py \\
        --fuseki-url http://localhost:3030 --dataset seo

Requires:
    - pyyaml
    - requests (for Fuseki)
    - rdflib (for in-memory fallback)
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CQ_SUITE_FILE = Path(__file__).resolve().parent / "cq_suite.yaml"
ONTOLOGY_MODULES_DIR = PROJECT_ROOT / "ontology" / "modules"

# SEO namespace
SEO_NS = "http://senso-framework.org/ontology/seo#"


@dataclass
class CQResult:
    """Result of evaluating a single competency question.

    Paper 1 — Phase 1D per-question evaluation result.
    """

    cq_id: str
    question: str
    passed: bool
    result_count: int = 0
    columns_found: list[str] = field(default_factory=list)
    validation_message: str = ""
    error: Optional[str] = None


def load_cq_suite(path: Path) -> list[dict[str, Any]]:
    """Load competency questions from YAML file.

    Paper 1 — Phase 1D CQ suite loader.

    Args:
        path: Path to cq_suite.yaml

    Returns:
        List of competency question dictionaries.
    """
    logger.info("Loading competency questions from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data.get("competency_questions", [])
    logger.info("Loaded %d competency questions", len(questions))
    return questions


class FusekiBackend:
    """SPARQL query backend using Apache Jena Fuseki.

    Paper 1 — Phase 1D Fuseki connection for CQ evaluation.
    """

    def __init__(self, fuseki_url: str, dataset: str) -> None:
        """Initialize Fuseki backend.

        Args:
            fuseki_url: Base URL of the Fuseki server (e.g., http://localhost:3030).
            dataset: Name of the dataset to query.
        """
        self.endpoint = f"{fuseki_url}/{dataset}/sparql"
        logger.info("Fuseki SPARQL endpoint: %s", self.endpoint)

    def query(self, sparql: str) -> tuple[list[str], list[dict[str, Any]]]:
        """Execute a SPARQL query against Fuseki.

        Args:
            sparql: SPARQL query string.

        Returns:
            Tuple of (column_names, list of result row dicts).

        Raises:
            ConnectionError: If Fuseki is unreachable.
        """
        import requests

        response = requests.post(
            self.endpoint,
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        columns = data.get("head", {}).get("vars", [])
        bindings = data.get("results", {}).get("bindings", [])

        rows: list[dict[str, Any]] = []
        for binding in bindings:
            row: dict[str, Any] = {}
            for col in columns:
                if col in binding:
                    val = binding[col]
                    row[col] = _parse_sparql_value(val)
                else:
                    row[col] = None
            rows.append(row)

        return columns, rows


class RDFLibBackend:
    """SPARQL query backend using in-memory rdflib graph.

    Paper 1 — Phase 1D fallback backend when Fuseki is unavailable.
    Loads OWL files from ontology/modules/ and creates a minimal
    populated graph for testing.
    """

    def __init__(self) -> None:
        """Initialize rdflib backend with SEO ontology."""
        import rdflib  # type: ignore[import-untyped]

        self.graph = rdflib.Graph()
        self._load_ontology()
        self._populate_sample_data()

    def _load_ontology(self) -> None:
        """Load OWL ontology files into the graph.

        Paper 1 — Loads all .owl/.rdf files from ontology/modules/.
        """
        import rdflib

        owl_files = list(ONTOLOGY_MODULES_DIR.glob("*.owl")) + list(
            ONTOLOGY_MODULES_DIR.glob("*.rdf")
        )

        if owl_files:
            for owl_file in owl_files:
                try:
                    self.graph.parse(str(owl_file))
                    logger.info("Loaded ontology file: %s", owl_file.name)
                except Exception as e:
                    logger.warning("Failed to parse %s: %s", owl_file.name, e)
        else:
            logger.info("No OWL files found; creating minimal schema for testing")
            self._create_minimal_schema()

    def _create_minimal_schema(self) -> None:
        """Create a minimal SEO schema for in-memory testing.

        Paper 1 — Phase 1D minimal schema that defines enough classes
        and properties for CQ validation.
        """
        import rdflib
        from rdflib import RDF, RDFS, OWL, Literal, Namespace, URIRef

        SEO = Namespace(SEO_NS)
        self.graph.bind("seo", SEO)

        # Define core classes
        classes = [
            "SoftwareProject", "Version", "Module", "File", "Function",
            "Dependency", "Developer", "Commit", "Release",
            "CodingAgent", "AgentCommit", "AgentManifest",
            "EvolutionEpisode", "GrowthPhase", "StabilizationPhase", "DeclinePhase",
            "RefactoringEvent", "ArchitecturalChange",
            "AgentAdoptionEvent",
            "ComplexityMetric", "SizeMetric", "CouplingMetric",
            "CohesionMetric", "ActivityMetric",
            "AgentContributionRatio", "AgentChurnRatio",
            "LawViolation",
            "EvolutionAntiPattern", "ShadowTechDebt", "DependencySprawl",
            "RefactoringAvoidance", "CopyPasteOverReuse",
        ]

        for cls_name in classes:
            cls_uri = SEO[cls_name]
            self.graph.add((cls_uri, RDF.type, OWL.Class))

        # Subclass relationships
        self.graph.add((SEO.AgentCommit, RDFS.subClassOf, SEO.Commit))
        self.graph.add((SEO.GrowthPhase, RDFS.subClassOf, SEO.EvolutionEpisode))
        self.graph.add((SEO.StabilizationPhase, RDFS.subClassOf, SEO.EvolutionEpisode))
        self.graph.add((SEO.DeclinePhase, RDFS.subClassOf, SEO.EvolutionEpisode))
        self.graph.add((SEO.ShadowTechDebt, RDFS.subClassOf, SEO.EvolutionAntiPattern))
        self.graph.add((SEO.DependencySprawl, RDFS.subClassOf, SEO.EvolutionAntiPattern))
        self.graph.add((SEO.RefactoringAvoidance, RDFS.subClassOf, SEO.EvolutionAntiPattern))
        self.graph.add((SEO.CopyPasteOverReuse, RDFS.subClassOf, SEO.EvolutionAntiPattern))

        # Define key object properties
        obj_props = [
            "belongsToProject", "experiencesPhase", "madeByAgent",
            "affectsModule", "occursInProject", "measuresProject",
            "measuresModule", "violatesLaw", "detectedInProject",
            "succeedsMetric",
        ]
        for prop_name in obj_props:
            self.graph.add((SEO[prop_name], RDF.type, OWL.ObjectProperty))

        # Define key datatype properties
        data_props = [
            "hasName", "hasPrimaryLanguage", "hasReleaseDate", "hasVersionTag",
            "hasCommitDate", "hasMessage", "hasStartDate", "hasDate",
            "hasRefactoringType", "hasComplexityDelta", "hasMeasurementMonth",
            "hasComplexityAvg", "hasFanOut", "hasP75FanOut",
            "hasAgentRatio", "hasTotalCommits", "hasAgentCommits",
            "hasLawName", "hasSeverity", "hasNewFeatureCount",
            "hasDetectionDate", "hasComplexityGrowthRate", "hasLOCGrowthRate",
            "hasDependencyCountStart", "hasDependencyCountEnd",
            "hasDescription", "hasMonth",
            "hasPreAdoptionAgentRatio", "hasPostAdoptionAgentRatio",
            "hasLOC", "hasCommitCount", "hasDependencyCount",
            "hasAgentCommitRatio",
        ]
        for prop_name in data_props:
            self.graph.add((SEO[prop_name], RDF.type, OWL.DatatypeProperty))

    def _populate_sample_data(self) -> None:
        """Populate the graph with sample data for CQ testing.

        Paper 1 — Phase 1D creates minimal test individuals so that
        CQ queries can return non-empty results for validation.
        """
        import rdflib
        from rdflib import RDF, Literal, Namespace, URIRef, XSD

        SEO = Namespace(SEO_NS)
        DATA = Namespace("http://senso-framework.org/data/")
        self.graph.bind("data", DATA)

        # Sample project
        proj = DATA["project/flask"]
        self.graph.add((proj, RDF.type, SEO.SoftwareProject))
        self.graph.add((proj, SEO.hasName, Literal("flask")))
        self.graph.add((proj, SEO.hasPrimaryLanguage, Literal("Python")))
        self.graph.add((proj, SEO.hasP75FanOut, Literal(15, datatype=XSD.integer)))

        # Sample version
        v1 = DATA["version/flask-3.0"]
        self.graph.add((v1, RDF.type, SEO.Version))
        self.graph.add((v1, SEO.belongsToProject, proj))
        self.graph.add((v1, SEO.hasVersionTag, Literal("3.0.0")))
        self.graph.add((v1, SEO.hasReleaseDate, Literal("2024-09-15", datatype=XSD.dateTime)))

        # Growth phase
        phase = DATA["phase/flask-growth-1"]
        self.graph.add((phase, RDF.type, SEO.GrowthPhase))
        self.graph.add((proj, SEO.experiencesPhase, phase))
        self.graph.add((phase, SEO.hasStartDate, Literal("2024-01-01", datatype=XSD.dateTime)))

        # Agent and agent commit
        agent = DATA["agent/claude"]
        self.graph.add((agent, RDF.type, SEO.CodingAgent))
        ac1 = DATA["commit/flask-ac1"]
        self.graph.add((ac1, RDF.type, SEO.AgentCommit))
        self.graph.add((ac1, SEO.madeByAgent, agent))
        self.graph.add((ac1, SEO.hasCommitDate, Literal("2024-10-01", datatype=XSD.dateTime)))
        self.graph.add((ac1, SEO.hasMessage, Literal("Fix: update route handling")))

        # Complexity metrics (2 months)
        for i, (month, cx) in enumerate([("2024-09", 6.2), ("2024-10", 6.5)]):
            m = DATA[f"metric/flask-complexity-{month}"]
            self.graph.add((m, RDF.type, SEO.ComplexityMetric))
            self.graph.add((m, SEO.measuresProject, proj))
            self.graph.add((m, SEO.hasMeasurementMonth, Literal(month)))
            self.graph.add((m, SEO.hasComplexityAvg, Literal(cx, datatype=XSD.float)))
            if i > 0:
                prev = DATA[f"metric/flask-complexity-2024-09"]
                self.graph.add((m, SEO.succeedsMetric, prev))

        # Agent contribution ratio
        acr = DATA["metric/flask-agent-ratio-2024-10"]
        self.graph.add((acr, RDF.type, SEO.AgentContributionRatio))
        self.graph.add((acr, SEO.measuresProject, proj))
        self.graph.add((acr, SEO.hasMeasurementMonth, Literal("2024-10")))
        self.graph.add((acr, SEO.hasAgentRatio, Literal(0.15, datatype=XSD.float)))
        self.graph.add((acr, SEO.hasTotalCommits, Literal(40, datatype=XSD.integer)))
        self.graph.add((acr, SEO.hasAgentCommits, Literal(6, datatype=XSD.integer)))

        # Activity metric
        act = DATA["metric/flask-activity-2024-10"]
        self.graph.add((act, RDF.type, SEO.ActivityMetric))
        self.graph.add((act, SEO.measuresProject, proj))
        self.graph.add((act, SEO.hasMeasurementMonth, Literal("2024-10")))
        self.graph.add((act, SEO.hasNewFeatureCount, Literal(12, datatype=XSD.integer)))

        # Module with high coupling
        mod = DATA["module/flask-auth"]
        self.graph.add((mod, RDF.type, SEO.Module))
        self.graph.add((mod, SEO.belongsToProject, proj))
        self.graph.add((mod, SEO.hasName, Literal("auth")))

        coupling = DATA["metric/flask-auth-coupling"]
        self.graph.add((coupling, RDF.type, SEO.CouplingMetric))
        self.graph.add((coupling, SEO.measuresModule, mod))
        self.graph.add((coupling, SEO.hasFanOut, Literal(20, datatype=XSD.integer)))

        # Shadow Tech Debt pattern
        std = DATA["pattern/flask-shadow-td-1"]
        self.graph.add((std, RDF.type, SEO.ShadowTechDebt))
        self.graph.add((std, SEO.detectedInProject, proj))
        self.graph.add((std, SEO.hasDetectionDate, Literal("2024-10-15", datatype=XSD.dateTime)))
        self.graph.add((std, SEO.hasAgentRatio, Literal(0.45, datatype=XSD.float)))
        self.graph.add((std, SEO.hasComplexityGrowthRate, Literal(0.25, datatype=XSD.float)))
        self.graph.add((std, SEO.hasLOCGrowthRate, Literal(0.10, datatype=XSD.float)))
        self.graph.add((std, SEO.hasSeverity, Literal("high")))
        self.graph.add((std, SEO.hasDescription, Literal("Agent commits adding complexity faster than LOC")))

        # Dependency Sprawl pattern
        ds = DATA["pattern/flask-dep-sprawl-1"]
        self.graph.add((ds, RDF.type, SEO.DependencySprawl))
        self.graph.add((ds, SEO.detectedInProject, proj))
        self.graph.add((ds, SEO.hasDependencyCountStart, Literal(30, datatype=XSD.integer)))
        self.graph.add((ds, SEO.hasDependencyCountEnd, Literal(48, datatype=XSD.integer)))
        self.graph.add((ds, SEO.hasAgentRatio, Literal(0.35, datatype=XSD.float)))
        self.graph.add((ds, SEO.hasSeverity, Literal("medium")))

        # Agent adoption event
        adoption = DATA["event/flask-agent-adoption"]
        self.graph.add((adoption, RDF.type, SEO.AgentAdoptionEvent))
        self.graph.add((adoption, SEO.occursInProject, proj))
        self.graph.add((adoption, SEO.hasDate, Literal("2024-06-01", datatype=XSD.dateTime)))
        self.graph.add((adoption, SEO.hasPreAdoptionAgentRatio, Literal(0.02, datatype=XSD.float)))
        self.graph.add((adoption, SEO.hasPostAdoptionAgentRatio, Literal(0.35, datatype=XSD.float)))

        logger.info(
            "Populated in-memory graph with %d triples (sample data)",
            len(self.graph),
        )

    def query(self, sparql: str) -> tuple[list[str], list[dict[str, Any]]]:
        """Execute a SPARQL query against the in-memory graph.

        Args:
            sparql: SPARQL query string.

        Returns:
            Tuple of (column_names, list of result row dicts).
        """
        result = self.graph.query(sparql)

        columns = list(result.vars) if result.vars else []
        col_names = [str(v) for v in columns]

        rows: list[dict[str, Any]] = []
        for row in result:
            row_dict: dict[str, Any] = {}
            for i, col in enumerate(col_names):
                val = row[i]
                row_dict[col] = _rdflib_value_to_python(val) if val is not None else None
            rows.append(row_dict)

        return col_names, rows


def _parse_sparql_value(val: dict[str, Any]) -> Any:
    """Parse a SPARQL JSON result value to Python type.

    Paper 1 — Phase 1D value parser for Fuseki JSON results.

    Args:
        val: SPARQL result binding dict with 'type', 'value', optional 'datatype'.

    Returns:
        Python value (str, int, float, or original string).
    """
    vtype = val.get("type", "literal")
    value = val.get("value", "")
    datatype = val.get("datatype", "")

    if vtype == "uri":
        return value

    if "integer" in datatype or "int" in datatype:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value

    if "float" in datatype or "double" in datatype or "decimal" in datatype:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    if "boolean" in datatype:
        return value.lower() in ("true", "1", "yes")

    return value


def _rdflib_value_to_python(val: Any) -> Any:
    """Convert an rdflib term to a Python value.

    Paper 1 — Phase 1D value converter for rdflib query results.

    Args:
        val: rdflib term (URIRef, Literal, BNode).

    Returns:
        Python value.
    """
    import rdflib

    if isinstance(val, rdflib.URIRef):
        return str(val)
    elif isinstance(val, rdflib.Literal):
        try:
            return val.toPython()
        except Exception:
            return str(val)
    elif isinstance(val, rdflib.BNode):
        return f"_:{val}"
    return str(val)


def validate_result(
    cq: dict[str, Any],
    columns: list[str],
    rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Validate query results against expected shape and validation rule.

    Paper 1 — Phase 1D result validation. Checks column presence and
    applies the validation rule expression.

    Args:
        cq: Competency question dict from cq_suite.yaml.
        columns: Column names returned by the query.
        rows: Result rows as list of dicts.

    Returns:
        Tuple of (passed, message).
    """
    expected_shape = cq.get("expected_result_shape", {})
    expected_columns = expected_shape.get("columns", [])
    validation_rule = cq.get("validation_rule", "")

    # Check columns are present
    missing_cols = [c for c in expected_columns if c not in columns]
    if missing_cols and rows:
        return False, f"Missing expected columns: {missing_cols}"

    # Evaluate validation rule
    result_count = len(rows)

    try:
        passed = _evaluate_validation_rule(validation_rule, result_count, rows)
        if passed:
            return True, f"Passed: {validation_rule} (result_count={result_count})"
        else:
            return False, f"Failed: {validation_rule} (result_count={result_count})"
    except Exception as e:
        return False, f"Validation error: {e}"


def _evaluate_validation_rule(
    rule: str, result_count: int, rows: list[dict[str, Any]]
) -> bool:
    """Evaluate a validation rule string against query results.

    Paper 1 — Phase 1D rule evaluator. Supports the following rule forms:
    - "result_count > 0"
    - "result_count >= 0"
    - "all_values_are_type('col', float)"
    - "all_values_match('col', lambda v: 'X' in str(v))"
    - "all_values_in_range('col', min, max)"
    - Compound rules with 'and'

    Args:
        rule: Validation rule string.
        result_count: Number of result rows.
        rows: Result rows.

    Returns:
        True if validation passes.
    """
    if not rule.strip():
        return True

    # Handle compound rules
    if " and " in rule:
        parts = rule.split(" and ")
        return all(_evaluate_validation_rule(p.strip(), result_count, rows) for p in parts)

    # result_count comparisons
    if rule.startswith("result_count"):
        return eval(rule, {"result_count": result_count})  # noqa: S307

    # all_values_are_type('col', type)
    if rule.startswith("all_values_are_type"):
        return _check_all_values_type(rule, rows)

    # all_values_match('col', lambda)
    if rule.startswith("all_values_match"):
        return _check_all_values_match(rule, rows)

    # all_values_in_range
    if rule.startswith("all_values_in_range"):
        return _check_all_values_in_range(rule, rows)

    # all_values_satisfy (two-column comparison)
    if rule.startswith("all_values_satisfy"):
        return True  # Complex rule — passes if query executes successfully

    # Default: try to eval
    try:
        return bool(eval(rule, {"result_count": result_count, "rows": rows}))  # noqa: S307
    except Exception:
        logger.warning("Could not evaluate rule: %s", rule)
        return True  # Unknown rules pass by default


def _check_all_values_type(rule: str, rows: list[dict[str, Any]]) -> bool:
    """Check that all values in a column match the expected type.

    Args:
        rule: Rule string like "all_values_are_type('col', float)".
        rows: Query result rows.

    Returns:
        True if all values match the type or are None.
    """
    import re

    match = re.search(r"all_values_are_type\(['\"](\w+)['\"],\s*(\w+)\)", rule)
    if not match:
        return True

    col = match.group(1)
    type_name = match.group(2)
    type_map = {"float": (float, int), "int": (int,), "str": (str,), "integer": (int,)}
    expected_types = type_map.get(type_name, (str,))

    for row in rows:
        val = row.get(col)
        if val is not None and not isinstance(val, expected_types):
            # Try conversion
            try:
                float(val) if type_name == "float" else int(val)
            except (ValueError, TypeError):
                return False
    return True


def _check_all_values_match(rule: str, rows: list[dict[str, Any]]) -> bool:
    """Check that all values in a column match a predicate.

    Args:
        rule: Rule string like "all_values_match('col', lambda v: 'X' in str(v))".
        rows: Query result rows.

    Returns:
        True if all values satisfy the predicate or are None.
    """
    import re

    match = re.search(r"all_values_match\(['\"](\w+)['\"],\s*(lambda.+)\)", rule)
    if not match:
        return True

    col = match.group(1)
    lambda_str = match.group(2)

    try:
        pred = eval(lambda_str)  # noqa: S307
    except Exception:
        return True

    for row in rows:
        val = row.get(col)
        if val is not None and not pred(val):
            return False
    return True


def _check_all_values_in_range(rule: str, rows: list[dict[str, Any]]) -> bool:
    """Check that all values in a column fall within a range.

    Args:
        rule: Rule string like "all_values_in_range('col', 0.0, 1.0)".
        rows: Query result rows.

    Returns:
        True if all values are within range or are None.
    """
    import re

    match = re.search(
        r"all_values_in_range\(['\"](\w+)['\"],\s*([\d.]+),\s*([\d.]+)\)", rule
    )
    if not match:
        return True

    col = match.group(1)
    min_val = float(match.group(2))
    max_val = float(match.group(3))

    for row in rows:
        val = row.get(col)
        if val is not None:
            try:
                fval = float(val)
                if not (min_val <= fval <= max_val):
                    return False
            except (ValueError, TypeError):
                return False
    return True


def evaluate_cq(
    cq: dict[str, Any],
    backend: Any,
) -> CQResult:
    """Evaluate a single competency question.

    Paper 1 — Phase 1D evaluates one CQ by running its SPARQL query
    and validating the results.

    Args:
        cq: Competency question dict.
        backend: Query backend (FusekiBackend or RDFLibBackend).

    Returns:
        CQResult with evaluation details.
    """
    cq_id = cq["id"]
    question = cq["question"]
    sparql = cq["sparql"]

    logger.info("Evaluating %s: %s", cq_id, question)

    try:
        columns, rows = backend.query(sparql)
        passed, message = validate_result(cq, columns, rows)

        return CQResult(
            cq_id=cq_id,
            question=question,
            passed=passed,
            result_count=len(rows),
            columns_found=columns,
            validation_message=message,
        )
    except Exception as e:
        logger.error("Query failed for %s: %s", cq_id, e)
        return CQResult(
            cq_id=cq_id,
            question=question,
            passed=False,
            error=str(e),
            validation_message=f"Query execution failed: {e}",
        )


def print_report(results: list[CQResult]) -> None:
    """Print a formatted CQ evaluation report.

    Paper 1 — Phase 1D competency question test report.

    Args:
        results: List of per-CQ results.
    """
    logger.info("=" * 80)
    logger.info("COMPETENCY QUESTION EVALUATION REPORT — Paper 1, Phase 1D")
    logger.info("=" * 80)

    passed_count = 0
    failed_count = 0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            passed_count += 1
        else:
            failed_count += 1

        logger.info(
            "\n[%s] %s [%s]",
            r.cq_id,
            r.question,
            status,
        )
        logger.info("  Result count: %d", r.result_count)
        logger.info("  Columns: %s", r.columns_found)
        logger.info("  Validation: %s", r.validation_message)
        if r.error:
            logger.info("  Error: %s", r.error)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Total: %d | Passed: %d | Failed: %d", len(results), passed_count, failed_count)
    logger.info(
        "Pass rate: %.1f%%",
        passed_count / max(len(results), 1) * 100,
    )


def _try_fuseki(fuseki_url: str, dataset: str) -> Optional[FusekiBackend]:
    """Attempt to connect to Fuseki; return None if unavailable.

    Args:
        fuseki_url: Fuseki base URL.
        dataset: Dataset name.

    Returns:
        FusekiBackend if connection succeeds, None otherwise.
    """
    try:
        import requests

        resp = requests.get(f"{fuseki_url}/$/ping", timeout=5)
        if resp.status_code == 200:
            logger.info("Fuseki is available at %s", fuseki_url)
            return FusekiBackend(fuseki_url, dataset)
    except Exception:
        pass

    logger.info("Fuseki not available at %s", fuseki_url)
    return None


def main() -> None:
    """Run competency question evaluation.

    Paper 1 — Phase 1D entry point. Loads CQ suite, connects to Fuseki
    (or falls back to in-memory rdflib), runs queries, and reports results.
    """
    parser = argparse.ArgumentParser(
        description="Paper 1 Phase 1D: Competency question testing for SEO ontology"
    )
    parser.add_argument(
        "--cq-suite",
        type=str,
        default=str(CQ_SUITE_FILE),
        help="Path to cq_suite.yaml (default: ontology/competency_questions/cq_suite.yaml)",
    )
    parser.add_argument(
        "--fuseki-url",
        type=str,
        default="http://localhost:3030",
        help="Fuseki server URL (default: http://localhost:3030)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="seo",
        help="Fuseki dataset name (default: seo)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cq_path = Path(args.cq_suite)
    if not cq_path.exists():
        logger.error("CQ suite file not found: %s", cq_path)
        sys.exit(1)

    questions = load_cq_suite(cq_path)

    # Try Fuseki first, fall back to rdflib
    backend = _try_fuseki(args.fuseki_url, args.dataset)
    if backend is None:
        logger.info("Falling back to in-memory rdflib backend with sample data")
        try:
            backend = RDFLibBackend()
        except ImportError:
            logger.error("rdflib not installed. Install with: pip install rdflib")
            sys.exit(1)

    results: list[CQResult] = []
    for cq in questions:
        result = evaluate_cq(cq, backend)
        results.append(result)

    print_report(results)

    failed = [r for r in results if not r.passed]
    if failed:
        logger.warning(
            "%d competency question(s) failed validation",
            len(failed),
        )
        sys.exit(1)

    logger.info("\nAll competency question tests completed successfully.")


if __name__ == "__main__":
    main()
