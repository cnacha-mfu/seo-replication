"""
Phase 3 — LLM-as-Judge Manifest Comparator for Paper 1

Rates both human-written and SENSO-generated agent manifests using 5 NVIDIA NIM
models for cross-model agreement, following the Paper 0 methodology.

Rubric dimensions (4-point scale):
  1. evolution_coverage  — evolution context richness
  2. architectural_specificity — architectural guidance detail
  3. actionability — recommendation concreteness
  4. phase_awareness — adaptation to current evolution phase

Used by: Paper 1 (Ontology + Agent Manifests)
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so shared.* imports resolve when running
# from /tmp or any other working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.llm_judge.judge import JudgmentRubric, Judgment  # noqa: E402
from shared.evaluation.statistics import holm_bonferroni  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────────

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODEL_REGISTRY: list[dict[str, str]] = [
    {"env_var": "NVIDIA_API_KEY",   "model": "meta/llama-3.3-70b-instruct",           "org": "Meta"},
    {"env_var": "NVIDIA_API_KEY_2", "model": "nvidia/llama-3.1-nemotron-70b-instruct", "org": "NVIDIA"},
    {"env_var": "NVIDIA_API_KEY_3", "model": "mistralai/mistral-large-latest",         "org": "Mistral AI"},
    {"env_var": "NVIDIA_API_KEY_4", "model": "google/gemma-2-27b-it",                  "org": "Google"},
    {"env_var": "NVIDIA_API_KEY_5", "model": "qwen/qwen2.5-72b-instruct",             "org": "Alibaba"},
]

# ── Rubric definitions ────────────────────────────────────────────────────

RUBRIC_DIMENSIONS: list[JudgmentRubric] = [
    JudgmentRubric(
        name="evolution_coverage",
        description="How much evolution context (phases, trends, history) does the manifest contain?",
        scale_min=1, scale_max=4,
        anchors={
            1: "No evolution info",
            2: "Mentions some history",
            3: "Covers phases and trends",
            4: "Comprehensive evolution context with quantitative data",
        },
    ),
    JudgmentRubric(
        name="architectural_specificity",
        description="How specific is the architectural guidance?",
        scale_min=1, scale_max=4,
        anchors={
            1: "Generic/no architecture info",
            2: "Mentions components",
            3: "Describes relationships and risks",
            4: "Detailed coupling, cohesion, dependency analysis",
        },
    ),
    JudgmentRubric(
        name="actionability",
        description="How actionable are the recommendations?",
        scale_min=1, scale_max=4,
        anchors={
            1: "No recommendations",
            2: "Vague guidance",
            3: "Specific recommendations",
            4: "Concrete, context-aware recommendations with rationale",
        },
    ),
    JudgmentRubric(
        name="phase_awareness",
        description=(
            "Does the manifest show awareness of the project's current "
            "evolution phase?"
        ),
        scale_min=1, scale_max=4,
        anchors={
            1: "No phase awareness",
            2: "Implicit phase hints",
            3: "Explicit phase identification",
            4: "Phase-aware recommendations that adapt to current state",
        },
    ),
]

# ── Helper: load manifests from a directory ───────────────────────────────


def _load_manifests(directory: Path) -> dict[str, str]:
    """Return {repo_slug: manifest_text} for every .md / .yaml / .json in *directory*."""
    manifests: dict[str, str] = {}
    if not directory.is_dir():
        logger.warning("Directory does not exist: %s", directory)
        return manifests
    for path in sorted(directory.iterdir()):
        if path.suffix in {".md", ".yaml", ".yml", ".json", ".txt"}:
            manifests[path.stem] = path.read_text(encoding="utf-8")
    return manifests


# ═══════════════════════════════════════════════════════════════════════════
# Core classes
# ═══════════════════════════════════════════════════════════════════════════


class NIMJudge:
    """Wraps a single NVIDIA NIM model via the OpenAI-compatible endpoint.

    Scores one manifest on one rubric dimension (Paper 1).
    """

    def __init__(self, model_id: str, api_key: str):
        self.model_id = model_id
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=NIM_BASE_URL, api_key=self._api_key)
        return self._client

    # ── public API ────────────────────────────────────────────────────────

    def score(
        self,
        manifest_text: str,
        rubric: JudgmentRubric,
        item_id: str,
        passes: int = 5,
    ) -> list[Judgment]:
        """Score *manifest_text* on *rubric* over multiple independent passes.

        Returns one ``Judgment`` per successful pass.
        """
        prompt = self._build_prompt(manifest_text, rubric)
        judgments: list[Judgment] = []

        for pass_num in range(passes):
            try:
                score, reasoning = self._call(prompt)
                judgments.append(Judgment(
                    item_id=item_id,
                    model=self.model_id,
                    dimension=rubric.name,
                    score=score,
                    reasoning=reasoning,
                ))
            except Exception as exc:
                logger.warning(
                    "Model %s pass %d failed on %s/%s: %s",
                    self.model_id, pass_num + 1, item_id, rubric.name, exc,
                )
        return judgments

    # ── internals ─────────────────────────────────────────────────────────

    def _build_prompt(self, manifest_text: str, rubric: JudgmentRubric) -> str:
        anchors_text = "\n".join(
            f"  {s}: {d}" for s, d in sorted(rubric.anchors.items())
        )
        return (
            "You are an expert software-evolution researcher evaluating an "
            "AI coding-agent manifest.\n\n"
            f"## Dimension: {rubric.name}\n"
            f"{rubric.description}\n\n"
            f"## Scoring Scale ({rubric.scale_min}–{rubric.scale_max}):\n"
            f"{anchors_text}\n\n"
            "## Manifest to Evaluate:\n"
            f"{manifest_text}\n\n"
            "## Instructions:\n"
            "1. Reason step-by-step about the manifest's quality on this dimension.\n"
            "2. Provide your score.\n\n"
            "Respond ONLY with valid JSON:\n"
            f'{{"reasoning": "…", "score": <integer {rubric.scale_min}-{rubric.scale_max}>}}'
        )

    def _call(self, prompt: str) -> tuple[int, str]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0.3,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        # Tolerate markdown-wrapped JSON (```json … ```)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match is None:
            raise ValueError(f"No JSON object found in response: {text[:200]}")
        parsed = json.loads(json_match.group())
        return int(parsed["score"]), parsed.get("reasoning", "")


# ───────────────────────────────────────────────────────────────────────────


class ManifestComparator:
    """Orchestrates pairwise comparison of human vs SENSO manifests across
    all 5 NIM models (Paper 1).

    For each repo that appears in *both* the human and generated directories
    the comparator scores both manifests on every rubric dimension using every
    available NIM model, then aggregates the results.
    """

    def __init__(
        self,
        human_dir: Path,
        generated_dir: Path,
        output_dir: Path,
        models: Optional[list[str]] = None,
        passes: int = 5,
        dry_run: bool = False,
    ):
        self.human_dir = Path(human_dir)
        self.generated_dir = Path(generated_dir)
        self.output_dir = Path(output_dir)
        self.passes = passes
        self.dry_run = dry_run
        self.judges: list[NIMJudge] = self._init_judges(models)
        self.all_judgments: list[Judgment] = []

    # ── initialisation ────────────────────────────────────────────────────

    def _init_judges(self, model_filter: Optional[list[str]]) -> list["NIMJudge"]:
        judges: list[NIMJudge] = []
        for entry in MODEL_REGISTRY:
            if model_filter and entry["model"] not in model_filter:
                continue
            api_key = os.getenv(entry["env_var"])
            if not api_key:
                logger.warning(
                    "Skipping %s (%s): env var %s not set",
                    entry["model"], entry["org"], entry["env_var"],
                )
                continue
            judges.append(NIMJudge(model_id=entry["model"], api_key=api_key))
            logger.info("Loaded judge: %s (%s)", entry["model"], entry["org"])
        if not judges and not self.dry_run:
            logger.error("No NIM API keys found — set NVIDIA_API_KEY* in .env")
        return judges

    # ── main entry point ──────────────────────────────────────────────────

    def run(self) -> dict:
        """Score all paired manifests and persist results.

        Returns a summary dict suitable for reporting.
        """
        human_manifests = _load_manifests(self.human_dir)
        generated_manifests = _load_manifests(self.generated_dir)

        paired_repos = sorted(set(human_manifests) & set(generated_manifests))
        if not paired_repos:
            logger.error(
                "No paired repos found between %s and %s",
                self.human_dir, self.generated_dir,
            )
            return {}

        logger.info("Found %d paired repos: %s", len(paired_repos), paired_repos)

        for repo in paired_repos:
            for source_tag, text in [
                ("human", human_manifests[repo]),
                ("senso", generated_manifests[repo]),
            ]:
                item_id = f"{repo}:{source_tag}"
                for rubric in RUBRIC_DIMENSIONS:
                    if self.dry_run:
                        # Produce synthetic scores for pipeline testing
                        for judge in self.judges or [NIMJudge.__new__(NIMJudge)]:
                            model_name = (
                                judge.model_id
                                if hasattr(judge, "model_id")
                                else "dry-run-model"
                            )
                            self.all_judgments.append(Judgment(
                                item_id=item_id,
                                model=model_name,
                                dimension=rubric.name,
                                score=random.randint(
                                    rubric.scale_min, rubric.scale_max
                                ),
                                reasoning="[dry-run]",
                            ))
                        continue

                    for judge in self.judges:
                        new = judge.score(
                            manifest_text=text,
                            rubric=rubric,
                            item_id=item_id,
                            passes=self.passes,
                        )
                        self.all_judgments.extend(new)

        # Persist raw judgments
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.output_dir / "raw_judgments.json"
        with open(raw_path, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {
                        "item_id": j.item_id,
                        "model": j.model,
                        "dimension": j.dimension,
                        "score": j.score,
                        "reasoning": j.reasoning,
                    }
                    for j in self.all_judgments
                ],
                fh,
                indent=2,
            )
        logger.info("Saved %d raw judgments → %s", len(self.all_judgments), raw_path)

        # Build summary
        summary = self._build_summary(paired_repos)
        summary_path = self.output_dir / "comparison_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("Saved comparison summary → %s", summary_path)
        return summary

    # ── summary builder ───────────────────────────────────────────────────

    def _build_summary(self, repos: list[str]) -> dict:
        """Aggregate per-repo, per-dimension mean scores for human vs SENSO."""
        summary: dict = {"repos": repos, "dimensions": {}}
        for dim in RUBRIC_DIMENSIONS:
            human_means: list[float] = []
            senso_means: list[float] = []
            for repo in repos:
                for tag, bucket in [("human", human_means), ("senso", senso_means)]:
                    item_id = f"{repo}:{tag}"
                    scores = [
                        j.score
                        for j in self.all_judgments
                        if j.item_id == item_id and j.dimension == dim.name
                    ]
                    bucket.append(float(np.mean(scores)) if scores else float("nan"))
            summary["dimensions"][dim.name] = {
                "human_means": human_means,
                "senso_means": senso_means,
                "human_grand_mean": float(np.nanmean(human_means)),
                "senso_grand_mean": float(np.nanmean(senso_means)),
            }
        return summary


# ───────────────────────────────────────────────────────────────────────────


class PerturbationTester:
    """Degrades manifest inputs and verifies that LLM-judge scores drop
    correspondingly (Paper 1 sensitivity testing).

    Three perturbation types:
      1. remove_refactoring — strip all refactoring-related sections
      2. truncate_history   — keep only the last 3 months of history data
      3. scramble_deps      — randomly shuffle dependency lists
    """

    PERTURBATION_TYPES = [
        "remove_refactoring",
        "truncate_history",
        "scramble_deps",
    ]

    def __init__(self, judges: list[NIMJudge], passes: int = 3, dry_run: bool = False):
        self.judges = judges
        self.passes = passes
        self.dry_run = dry_run

    # ── public API ────────────────────────────────────────────────────────

    def run(self, manifests: dict[str, str]) -> dict:
        """Run all perturbation types on every manifest and compare scores.

        Returns dict keyed by perturbation type → list of (original, degraded)
        score pairs per dimension.
        """
        results: dict[str, dict[str, list[tuple[float, float]]]] = {
            pt: {r.name: [] for r in RUBRIC_DIMENSIONS}
            for pt in self.PERTURBATION_TYPES
        }

        for repo, text in manifests.items():
            for pt in self.PERTURBATION_TYPES:
                degraded = self._apply(pt, text)
                for rubric in RUBRIC_DIMENSIONS:
                    orig_score = self._score_aggregate(text, rubric, f"{repo}:orig:{pt}")
                    degr_score = self._score_aggregate(degraded, rubric, f"{repo}:degr:{pt}")
                    results[pt][rubric.name].append((orig_score, degr_score))

        return self._summarise(results)

    # ── perturbation strategies ───────────────────────────────────────────

    def _apply(self, perturbation_type: str, text: str) -> str:
        if perturbation_type == "remove_refactoring":
            return self._remove_refactoring(text)
        elif perturbation_type == "truncate_history":
            return self._truncate_history(text)
        elif perturbation_type == "scramble_deps":
            return self._scramble_deps(text)
        raise ValueError(f"Unknown perturbation: {perturbation_type}")

    @staticmethod
    def _remove_refactoring(text: str) -> str:
        """Remove lines/sections mentioning refactoring."""
        lines = text.splitlines()
        return "\n".join(
            l for l in lines
            if "refactor" not in l.lower()
        )

    @staticmethod
    def _truncate_history(text: str) -> str:
        """Keep only the last ~20 lines of history/trend data."""
        lines = text.splitlines()
        if len(lines) <= 30:
            return text
        # Keep first 10 (header) + last 20 (recent history)
        return "\n".join(lines[:10] + ["[... history truncated to 3 months ...]"] + lines[-20:])

    @staticmethod
    def _scramble_deps(text: str) -> str:
        """Randomly shuffle dependency/import lists."""
        lines = text.splitlines()
        dep_block: list[int] = []
        result = list(lines)
        for i, line in enumerate(lines):
            lower = line.lower()
            if any(kw in lower for kw in ("depend", "import", "require", "package")):
                dep_block.append(i)
        if dep_block:
            contents = [result[i] for i in dep_block]
            random.shuffle(contents)
            for idx, i in enumerate(dep_block):
                result[i] = contents[idx]
        return "\n".join(result)

    # ── scoring helper ────────────────────────────────────────────────────

    def _score_aggregate(
        self, text: str, rubric: JudgmentRubric, item_id: str
    ) -> float:
        """Return the grand-mean score across all judges and passes."""
        if self.dry_run:
            return float(random.randint(rubric.scale_min, rubric.scale_max))

        all_scores: list[int] = []
        for judge in self.judges:
            judgments = judge.score(text, rubric, item_id, passes=self.passes)
            all_scores.extend(j.score for j in judgments)
        return float(np.mean(all_scores)) if all_scores else float("nan")

    # ── summarise ─────────────────────────────────────────────────────────

    @staticmethod
    def _summarise(
        results: dict[str, dict[str, list[tuple[float, float]]]]
    ) -> dict:
        summary: dict = {}
        for pt, dims in results.items():
            summary[pt] = {}
            for dim, pairs in dims.items():
                if not pairs:
                    continue
                orig = [p[0] for p in pairs]
                degr = [p[1] for p in pairs]
                drop = [o - d for o, d in pairs]
                summary[pt][dim] = {
                    "mean_original": float(np.nanmean(orig)),
                    "mean_degraded": float(np.nanmean(degr)),
                    "mean_drop": float(np.nanmean(drop)),
                    "all_dropped": all(d > 0 for d in drop),
                    "n": len(pairs),
                }
        return summary


# ───────────────────────────────────────────────────────────────────────────


class AgreementAnalyzer:
    """Computes inter-rater reliability across the 5 NIM models (Paper 1).

    Metrics:
      - Krippendorff's alpha (ordinal) per dimension
      - Pairwise Spearman rho between every model pair
    """

    def __init__(self, judgments: list[Judgment]):
        self.judgments = judgments

    def compute_krippendorff_alpha(self, dimension: str) -> float:
        """Krippendorff's alpha for one rubric dimension."""
        try:
            import krippendorff
        except ImportError:
            logger.error("pip install krippendorff")
            return float("nan")

        dim_j = [j for j in self.judgments if j.dimension == dimension]
        models = sorted({j.model for j in dim_j})
        items = sorted({j.item_id for j in dim_j})

        if len(models) < 2 or len(items) < 2:
            return float("nan")

        matrix = np.full((len(models), len(items)), np.nan)
        for j in dim_j:
            m_idx = models.index(j.model)
            i_idx = items.index(j.item_id)
            # If multiple passes per model×item, keep the mean
            existing = matrix[m_idx, i_idx]
            if np.isnan(existing):
                matrix[m_idx, i_idx] = j.score
            else:
                matrix[m_idx, i_idx] = (existing + j.score) / 2.0

        return float(krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="ordinal",
        ))

    def compute_pairwise_spearman(self, dimension: str) -> dict:
        """Pairwise Spearman rho between every model pair for one dimension."""
        from scipy.stats import spearmanr

        dim_j = [j for j in self.judgments if j.dimension == dimension]
        models = sorted({j.model for j in dim_j})

        # Aggregate: mean score per model × item
        model_scores: dict[str, dict[str, float]] = {m: {} for m in models}
        model_counts: dict[str, dict[str, list[int]]] = {m: {} for m in models}
        for j in dim_j:
            model_counts[j.model].setdefault(j.item_id, []).append(j.score)
        for m in models:
            for item, scores in model_counts[m].items():
                model_scores[m][item] = float(np.mean(scores))

        results: dict[str, dict] = {}
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                common = sorted(set(model_scores[m1]) & set(model_scores[m2]))
                if len(common) < 3:
                    continue
                s1 = [model_scores[m1][c] for c in common]
                s2 = [model_scores[m2][c] for c in common]
                rho, p = spearmanr(s1, s2)
                results[f"{m1} vs {m2}"] = {
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                    "n_items": len(common),
                }
        return results

    def summary(self) -> dict:
        """Full agreement summary across all dimensions."""
        dimensions = sorted({j.dimension for j in self.judgments})
        return {
            dim: {
                "krippendorff_alpha": self.compute_krippendorff_alpha(dim),
                "pairwise_spearman": self.compute_pairwise_spearman(dim),
            }
            for dim in dimensions
        }


# ───────────────────────────────────────────────────────────────────────────


class StatisticalReporter:
    """Runs Wilcoxon signed-rank tests (human vs SENSO, paired by repo) per
    rubric dimension and applies Holm-Bonferroni correction (Paper 1).
    """

    def __init__(self, judgments: list[Judgment]):
        self.judgments = judgments

    def run(self, repos: list[str]) -> dict:
        """Return per-dimension Wilcoxon test results with correction."""
        from scipy.stats import wilcoxon

        dimension_results: dict[str, dict] = {}
        p_values: list[float] = []
        dim_names: list[str] = []

        for rubric in RUBRIC_DIMENSIONS:
            human_scores: list[float] = []
            senso_scores: list[float] = []

            for repo in repos:
                for tag, bucket in [
                    ("human", human_scores),
                    ("senso", senso_scores),
                ]:
                    item_id = f"{repo}:{tag}"
                    scores = [
                        j.score
                        for j in self.judgments
                        if j.item_id == item_id and j.dimension == rubric.name
                    ]
                    bucket.append(float(np.mean(scores)) if scores else float("nan"))

            # Drop pairs with NaN
            pairs = [
                (h, s)
                for h, s in zip(human_scores, senso_scores)
                if not (np.isnan(h) or np.isnan(s))
            ]
            if len(pairs) < 5:
                logger.warning(
                    "Dimension %s: only %d valid pairs, skipping Wilcoxon",
                    rubric.name, len(pairs),
                )
                dimension_results[rubric.name] = {"skipped": True, "n_pairs": len(pairs)}
                p_values.append(1.0)
                dim_names.append(rubric.name)
                continue

            h_arr = np.array([p[0] for p in pairs])
            s_arr = np.array([p[1] for p in pairs])

            try:
                stat, p = wilcoxon(h_arr, s_arr)
            except ValueError:
                # All differences are zero
                stat, p = 0.0, 1.0

            from shared.evaluation.statistics import cliffs_delta

            delta, magnitude = cliffs_delta(s_arr.tolist(), h_arr.tolist())

            dimension_results[rubric.name] = {
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
                "cliffs_delta": delta,
                "effect_magnitude": magnitude,
                "human_mean": float(np.mean(h_arr)),
                "senso_mean": float(np.mean(s_arr)),
                "n_pairs": len(pairs),
            }
            p_values.append(float(p))
            dim_names.append(rubric.name)

        # Holm-Bonferroni correction
        rejected = holm_bonferroni(p_values)
        for dim_name, rej in zip(dim_names, rejected):
            if dim_name in dimension_results and not dimension_results[dim_name].get("skipped"):
                dimension_results[dim_name]["significant_after_correction"] = rej

        return dimension_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 — LLM-as-Judge Manifest Comparator (Paper 1)",
    )
    p.add_argument(
        "--human-dir",
        type=Path,
        default=_PROJECT_ROOT / "manifests" / "collected",
        help="Directory of human-written manifests",
    )
    p.add_argument(
        "--generated-dir",
        type=Path,
        default=_PROJECT_ROOT / "manifests" / "generated",
        help="Directory of SENSO-generated manifests",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "experiments" / "paper1_ontology_manifests" / "results",
        help="Output directory for results",
    )
    p.add_argument(
        "--perturbation",
        action="store_true",
        help="Run perturbation sensitivity testing after comparison",
    )
    p.add_argument(
        "--models",
        nargs="*",
        help="Subset of NIM model IDs to use (default: all available)",
    )
    p.add_argument(
        "--passes",
        type=int,
        default=5,
        help="Number of independent scoring passes per model (default: 5)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Produce synthetic scores without API calls (pipeline testing)",
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config (overrides CLI flags)",
    )
    return p.parse_args(argv)


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    """Overlay YAML config onto parsed CLI args (CLI wins where specified)."""
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        for key in ("human_dir", "generated_dir", "output", "passes"):
            if key in cfg and getattr(args, key.replace("-", "_"), None) is None:
                setattr(args, key.replace("-", "_"), cfg[key])
        if "models" in cfg and args.models is None:
            args.models = cfg["models"]
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    args = _apply_config(args)

    logger.info("=== Phase 3: Manifest Comparator (Paper 1) ===")
    logger.info("Human dir : %s", args.human_dir)
    logger.info("Generated : %s", args.generated_dir)
    logger.info("Output    : %s", args.output)
    logger.info("Dry run   : %s", args.dry_run)

    # ── 1. Main comparison ────────────────────────────────────────────────
    comparator = ManifestComparator(
        human_dir=args.human_dir,
        generated_dir=args.generated_dir,
        output_dir=args.output,
        models=args.models,
        passes=args.passes,
        dry_run=args.dry_run,
    )
    summary = comparator.run()
    if not summary:
        logger.error("No comparison results — aborting.")
        return

    repos = summary["repos"]

    # ── 2. Agreement analysis ─────────────────────────────────────────────
    logger.info("Computing cross-model agreement …")
    analyzer = AgreementAnalyzer(comparator.all_judgments)
    agreement = analyzer.summary()
    agreement_path = args.output / "agreement.json"
    with open(agreement_path, "w", encoding="utf-8") as fh:
        json.dump(agreement, fh, indent=2, default=str)
    logger.info("Agreement results → %s", agreement_path)

    # ── 3. Statistical tests ──────────────────────────────────────────────
    logger.info("Running Wilcoxon signed-rank tests …")
    reporter = StatisticalReporter(comparator.all_judgments)
    stats = reporter.run(repos)
    stats_path = args.output / "statistical_tests.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    logger.info("Statistical results → %s", stats_path)

    # ── 4. Perturbation testing (optional) ────────────────────────────────
    if args.perturbation:
        logger.info("Running perturbation sensitivity tests …")
        generated_manifests = _load_manifests(Path(args.generated_dir))
        tester = PerturbationTester(
            judges=comparator.judges,
            passes=max(1, args.passes // 2),
            dry_run=args.dry_run,
        )
        pert_results = tester.run(generated_manifests)
        pert_path = args.output / "perturbation_results.json"
        with open(pert_path, "w", encoding="utf-8") as fh:
            json.dump(pert_results, fh, indent=2, default=str)
        logger.info("Perturbation results → %s", pert_path)

    logger.info("=== Phase 3 complete ===")


if __name__ == "__main__":
    main()
