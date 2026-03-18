"""
LLM-as-Judge Pipeline for SENSO Framework

Implements the human-free evaluation methodology validated in Paper 0 (ADR study):
- Cross-model agreement (multiple LLMs as independent raters)
- Perturbation-based sensitivity testing
- Convergent validity with objective metrics

Used by: Paper 1 (manifest quality comparison), Paper 4 (guardrail quality assessment)

Usage:
    from shared.llm_judge import LLMJudge, CrossModelAgreement
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class JudgmentRubric:
    """A rubric dimension for LLM-as-Judge evaluation."""
    name: str
    description: str
    scale_min: int = 1
    scale_max: int = 4
    anchors: dict = field(default_factory=dict)  # score -> description


@dataclass
class Judgment:
    """A single LLM judgment on one item."""
    item_id: str
    model: str
    dimension: str
    score: int
    reasoning: Optional[str] = None


class LLMJudge:
    """Base class for LLM-as-Judge evaluation.

    Supports multiple LLM providers (Anthropic, OpenAI, Google).
    Uses structured output (JSON mode) and chain-of-thought prompting.
    """

    def __init__(self, model_name: str, provider: str = "anthropic"):
        self.model_name = model_name
        self.provider = provider
        self.client = self._init_client()

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic()
            except ImportError:
                logger.warning("anthropic package not installed")
                return None
        elif self.provider == "openai":
            try:
                import openai
                return openai.OpenAI()
            except ImportError:
                logger.warning("openai package not installed")
                return None
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                return genai
            except ImportError:
                logger.warning("google-generativeai package not installed")
                return None
        return None

    def judge(self, item: str, rubric: JudgmentRubric,
              context: str = "", passes: int = 5) -> list[Judgment]:
        """Score an item on a rubric dimension using multiple passes.

        Args:
            item: The content to evaluate
            rubric: The scoring rubric
            context: Additional context for the judge
            passes: Number of independent scoring passes (for stability)

        Returns:
            List of Judgment objects (one per pass)
        """
        prompt = self._build_prompt(item, rubric, context)
        judgments = []

        for pass_num in range(passes):
            try:
                score, reasoning = self._call_llm(prompt)
                judgments.append(Judgment(
                    item_id=str(hash(item))[:12],
                    model=self.model_name,
                    dimension=rubric.name,
                    score=score,
                    reasoning=reasoning,
                ))
            except Exception as e:
                logger.warning(f"Pass {pass_num+1} failed: {e}")

        return judgments

    def _build_prompt(self, item: str, rubric: JudgmentRubric, context: str) -> str:
        anchors_text = "\n".join(
            f"  {score}: {desc}" for score, desc in sorted(rubric.anchors.items())
        )
        return f"""You are an expert evaluator. Score the following item on the dimension described below.

## Dimension: {rubric.name}
{rubric.description}

## Scoring Scale ({rubric.scale_min}–{rubric.scale_max}):
{anchors_text}

## Context:
{context}

## Item to Evaluate:
{item}

## Instructions:
1. First, reason step-by-step about the item's quality on this dimension.
2. Then provide your score.

Respond in JSON format:
{{"reasoning": "your step-by-step reasoning", "score": <integer {rubric.scale_min}-{rubric.scale_max}>}}
"""

    def _call_llm(self, prompt: str) -> tuple[int, str]:
        """Call the LLM and parse the response. Override for each provider."""
        if self.provider == "anthropic" and self.client:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        elif self.provider == "openai" and self.client:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content
        else:
            raise RuntimeError(f"No client for {self.provider}")

        # Parse JSON response
        parsed = json.loads(text)
        return int(parsed["score"]), parsed.get("reasoning", "")


class CrossModelAgreement:
    """Compute cross-model agreement metrics (from Paper 0 methodology).

    Uses Krippendorff's alpha for ordinal data across multiple LLM raters.
    """

    def __init__(self, judgments: list[Judgment]):
        self.judgments = judgments

    def compute_krippendorff_alpha(self, dimension: str) -> float:
        """Compute Krippendorff's alpha for a specific dimension."""
        try:
            import krippendorff
            import numpy as np
        except ImportError:
            logger.error("Install krippendorff: pip install krippendorff")
            return float("nan")

        # Build reliability matrix: models × items
        dim_judgments = [j for j in self.judgments if j.dimension == dimension]
        models = sorted(set(j.model for j in dim_judgments))
        items = sorted(set(j.item_id for j in dim_judgments))

        matrix = np.full((len(models), len(items)), np.nan)
        for j in dim_judgments:
            m_idx = models.index(j.model)
            i_idx = items.index(j.item_id)
            matrix[m_idx, i_idx] = j.score

        return krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="ordinal",
        )

    def compute_pairwise_agreement(self, dimension: str) -> dict:
        """Compute pairwise agreement between all model pairs."""
        from scipy.stats import spearmanr
        import numpy as np

        dim_judgments = [j for j in self.judgments if j.dimension == dimension]
        models = sorted(set(j.model for j in dim_judgments))
        items = sorted(set(j.item_id for j in dim_judgments))

        # Build score vectors per model
        model_scores = {}
        for model in models:
            scores = {}
            for j in dim_judgments:
                if j.model == model:
                    scores[j.item_id] = j.score
            model_scores[model] = scores

        results = {}
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                common = set(model_scores[m1].keys()) & set(model_scores[m2].keys())
                if len(common) < 3:
                    continue
                s1 = [model_scores[m1][item] for item in sorted(common)]
                s2 = [model_scores[m2][item] for item in sorted(common)]
                rho, p = spearmanr(s1, s2)
                exact_agree = sum(a == b for a, b in zip(s1, s2)) / len(s1)
                results[f"{m1}_vs_{m2}"] = {
                    "spearman_rho": rho,
                    "p_value": p,
                    "exact_agreement": exact_agree,
                    "n_items": len(common),
                }
        return results

    def summary(self) -> dict:
        """Compute agreement summary across all dimensions."""
        dimensions = sorted(set(j.dimension for j in self.judgments))
        return {
            dim: {
                "krippendorff_alpha": self.compute_krippendorff_alpha(dim),
                "pairwise": self.compute_pairwise_agreement(dim),
            }
            for dim in dimensions
        }
