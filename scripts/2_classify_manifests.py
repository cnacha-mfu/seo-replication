"""
Content Classifier for JCSSE 2026 Paper.

Classifies sections of human-authored CLAUDE.md files into 6 content categories
using 3 NVIDIA NIM LLM models (cross-model agreement).

Paper: "What Do Human Agent Manifests Miss? An Empirical Content Analysis of CLAUDE.md Files"
Venue: JCSSE 2026
Deadline: 30 March 2026

Used by: JCSSE 2026 paper (all RQs)

Usage:
    python experiments/jcsse2026/content_classifier.py \
        --config experiments/jcsse2026/configs/jcsse.yaml
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Category constants (mirrors jcsse.yaml) ───────────────────────────────────
CATEGORIES = [
    "coding_style",
    "project_structure",
    "workflow_process",
    "architecture",
    "evolution_context",
    "agent_rules",
]

CATEGORY_LABELS = {
    "coding_style": "Coding Style",
    "project_structure": "Project Structure",
    "workflow_process": "Workflow & Process",
    "architecture": "Architecture",
    "evolution_context": "Evolution Context",
    "agent_rules": "Agent-Specific Rules",
}

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Section:
    """A single Markdown heading + its body text."""
    heading: str
    level: int          # 1=H1, 2=H2, 3=H3
    text: str           # full text including heading line
    body: str           # body only (no heading line)
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.body.strip())


@dataclass
class SectionClassification:
    """Classification result for one section from one model."""
    repo: str
    section_heading: str
    model: str
    categories: list[str]
    reasoning: str
    raw_response: str = ""


@dataclass
class ManifestClassification:
    """Aggregated classification for one CLAUDE.md manifest."""
    repo: str
    full_name: str
    n_sections: int
    n_chars: int
    sections: list[dict] = field(default_factory=list)
    # Majority-vote category presence per section (across 3 models)
    section_votes: list[dict] = field(default_factory=list)
    # Manifest-level: which categories are present (True/False)
    category_present: dict = field(default_factory=dict)
    # Proportion of sections tagged with each category
    category_section_fraction: dict = field(default_factory=dict)
    # Proportion of characters attributed to each category
    category_char_fraction: dict = field(default_factory=dict)


# ── Markdown splitter ─────────────────────────────────────────────────────────

def split_into_sections(text: str) -> list[Section]:
    """Split Markdown text into sections by heading.

    Returns one Section per heading block, plus a preamble section
    for any content before the first heading.
    """
    heading_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    sections: list[Section] = []

    # Find all heading positions
    matches = list(heading_re.finditer(text))

    if not matches:
        # No headings — treat entire file as one section
        if text.strip():
            sections.append(Section(
                heading="(no heading)",
                level=1,
                text=text,
                body=text,
            ))
        return sections

    # Preamble (before first heading)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(
            heading="(preamble)",
            level=0,
            text=preamble,
            body=preamble,
        ))

    for idx, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        full_text = text[start:end]
        body = text[m.end():end].strip()

        # Skip very short sections (< 20 chars body) — likely empty stubs
        if len(body) < 20 and heading not in ("(preamble)", "(no heading)"):
            continue

        sections.append(Section(
            heading=heading,
            level=level,
            text=full_text,
            body=body,
        ))

    return sections


# ── LLM classifier ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant classifying sections of CLAUDE.md files
(AI agent instruction manifests) into content categories for an empirical study.

The six categories are:
1. coding_style       — Naming conventions, formatting, linting, import order, code style tools
2. project_structure  — Directory layout, key files, module descriptions, where things live
3. workflow_process   — Commit format, PR rules, testing requirements, CI/CD, review guidelines
4. architecture       — Module roles, dependencies, design patterns, component boundaries, ADRs
5. evolution_context  — Project phase, complexity trends, technical debt, history, churn, refactoring plans
6. agent_rules        — Claude plan mode, subagent strategy, tool use constraints, AI-specific directives

Instructions:
- A section may belong to multiple categories.
- Focus on the dominant content of the section.
- If a section contains only boilerplate (e.g., "Table of Contents") assign an empty list.
- Be concise in your reasoning.

Respond ONLY with valid JSON in this exact format:
{"categories": ["category_id", ...], "reasoning": "one sentence explanation"}

Valid category IDs: coding_style, project_structure, workflow_process, architecture, evolution_context, agent_rules"""


def build_classification_prompt(section: Section, repo_name: str) -> str:
    """Build the user prompt for a section classification request."""
    return f"""Repository: {repo_name}
Section heading: {section.heading!r}

Section content:
---
{section.body[:2000]}
---

Classify this section into one or more of the six content categories."""


def call_nim(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
) -> tuple[str, str]:
    """Call a NVIDIA NIM model and return (raw_text, parsed_json_str).

    Returns ("", "{}") on failure.
    """
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            return raw, raw
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                "Model %s attempt %d/%d failed: %s — retrying in %ds",
                model, attempt + 1, retries, exc, wait,
            )
            time.sleep(wait)
    return "", "{}"


def parse_classification(raw: str) -> tuple[list[str], str]:
    """Parse the JSON classification response.

    Returns (categories_list, reasoning_str).
    Falls back to empty list on parse error.
    """
    if not raw:
        return [], "parse error: empty response"

    # Try to extract JSON from the response (model may add preamble)
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    text = json_match.group(0) if json_match else raw

    try:
        parsed = json.loads(text)
        cats = [c for c in parsed.get("categories", []) if c in CATEGORIES]
        reasoning = parsed.get("reasoning", "")
        return cats, reasoning
    except json.JSONDecodeError:
        logger.debug("JSON parse failed for: %s", raw[:200])
        # Fallback: look for category keywords in raw text
        found = [c for c in CATEGORIES if c in raw.lower()]
        return found, f"fallback parse: {raw[:100]}"


def majority_vote(votes: list[list[str]]) -> list[str]:
    """Return categories that appear in majority (≥2 of 3) model votes."""
    from collections import Counter
    counts: Counter = Counter()
    for cats in votes:
        for cat in cats:
            counts[cat] += 1
    n_models = len(votes)
    threshold = n_models / 2  # >50%
    return [cat for cat, cnt in counts.items() if cnt > threshold]


# ── Main classifier ────────────────────────────────────────────────────────────

class ContentClassifier:
    """Classify CLAUDE.md sections using 3 NVIDIA NIM models.

    For each of the 23 human manifests:
    1. Split into sections by Markdown heading.
    2. For each section, query 3 LLM models → 3 category lists.
    3. Majority vote → final categories per section.
    4. Aggregate to manifest-level coverage and fraction metrics.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self._init_clients()

    def _init_clients(self) -> None:
        """Initialise one OpenAI client per configured NVIDIA NIM model."""
        from openai import OpenAI

        self.clients: list[tuple[str, str, object]] = []  # (org, model, client)
        for entry in self.config["models"]:
            api_key = os.environ.get(entry["key_env"])
            if not api_key:
                logger.warning(
                    "Missing env var %s — skipping model %s",
                    entry["key_env"], entry["model"],
                )
                continue
            client = OpenAI(
                base_url=self.config["nim_base_url"],
                api_key=api_key,
            )
            self.clients.append((entry["org"], entry["model"], client))
            logger.info("Loaded model: %s (%s)", entry["model"], entry["org"])

        if not self.clients:
            raise RuntimeError(
                "No NVIDIA NIM models available. "
                "Set NVIDIA_API_KEY, NVIDIA_API_KEY_2, NVIDIA_API_KEY_3."
            )

    def classify_section(
        self, section: Section, repo_name: str
    ) -> tuple[list[str], list[SectionClassification]]:
        """Classify one section with all models → majority vote categories.

        Returns (voted_categories, raw_classifications).
        """
        user_prompt = build_classification_prompt(section, repo_name)
        all_classifications: list[SectionClassification] = []
        all_votes: list[list[str]] = []

        for org, model, client in self.clients:
            raw, _ = call_nim(
                client=client,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
            )
            cats, reasoning = parse_classification(raw)
            all_votes.append(cats)
            all_classifications.append(SectionClassification(
                repo=repo_name,
                section_heading=section.heading,
                model=model,
                categories=cats,
                reasoning=reasoning,
                raw_response=raw,
            ))

        voted = majority_vote(all_votes)
        return voted, all_classifications

    def classify_manifest(self, repo: dict) -> ManifestClassification:
        """Classify all sections of one manifest."""
        full_name = repo["full_name"]
        text = repo.get("claude_md_content", "")
        repo_key = full_name.replace("/", "__")

        logger.info("Classifying %s (%d chars)", full_name, len(text))

        sections = split_into_sections(text)
        logger.info("  → %d sections", len(sections))

        result = ManifestClassification(
            repo=repo_key,
            full_name=full_name,
            n_sections=len(sections),
            n_chars=len(text),
        )

        section_records = []
        all_raw_classifications: list[dict] = []
        total_chars = sum(s.char_count for s in sections) or 1

        for section in sections:
            voted_cats, raw_cls = self.classify_section(section, full_name)
            section_records.append({
                "heading": section.heading,
                "level": section.level,
                "char_count": section.char_count,
                "voted_categories": voted_cats,
                "model_classifications": [asdict(c) for c in raw_cls],
            })
            all_raw_classifications.extend([asdict(c) for c in raw_cls])

        result.sections = section_records
        result.section_votes = section_records  # same reference

        # ── Aggregate metrics ──────────────────────────────────────────────────
        n_sections = len(sections) or 1
        cat_section_count: dict[str, int] = {c: 0 for c in CATEGORIES}
        cat_char_count: dict[str, int] = {c: 0 for c in CATEGORIES}

        for sec_rec, sec in zip(section_records, sections):
            for cat in sec_rec["voted_categories"]:
                cat_section_count[cat] += 1
                cat_char_count[cat] += sec.char_count

        result.category_present = {
            cat: cat_section_count[cat] > 0 for cat in CATEGORIES
        }
        result.category_section_fraction = {
            cat: cat_section_count[cat] / n_sections for cat in CATEGORIES
        }
        result.category_char_fraction = {
            cat: cat_char_count[cat] / total_chars for cat in CATEGORIES
        }

        return result

    def _process_one(self, repo: dict, out_dir: Path, idx: int, total: int) -> ManifestClassification:
        """Classify a single repo and save the result. Thread-safe."""
        full_name = repo["full_name"]
        repo_key = full_name.replace("/", "__")
        out_file = out_dir / f"{repo_key}_classification.json"

        # Resume: skip if already done
        if out_file.exists():
            logger.info("[%d/%d] Skipping %s (already done)", idx, total, full_name)
            with open(out_file) as f:
                saved = json.load(f)
            return ManifestClassification(
                repo=repo_key,
                full_name=full_name,
                n_sections=saved["n_sections"],
                n_chars=saved["n_chars"],
                sections=saved["sections"],
                section_votes=saved["sections"],
                category_present=saved["category_present"],
                category_section_fraction=saved["category_section_fraction"],
                category_char_fraction=saved["category_char_fraction"],
            )

        logger.info("[%d/%d] Processing %s", idx, total, full_name)
        mc = self.classify_manifest(repo)

        # Save per-manifest JSON
        with open(out_file, "w") as f:
            json.dump(asdict(mc), f, indent=2)
        logger.info("  Saved → %s", out_file)
        return mc

    def run(self, output_dir: Optional[str] = None, workers: int = 4) -> list[ManifestClassification]:
        """Classify all manifests and save per-manifest JSONs.

        Args:
            output_dir: Override output directory from config.
            workers: Number of parallel workers (default 4).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cfg = self.config
        data_path = cfg["data"]["phase1_repos"]
        out_dir = Path(output_dir or cfg["data"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(data_path) as f:
            repos = json.load(f)

        total = len(repos)
        logger.info("Loaded %d repositories from %s (workers=%d)", total, data_path, workers)

        results: list[ManifestClassification] = [None] * total  # type: ignore

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._process_one, repo, out_dir, i, total): i
                for i, repo in enumerate(repos, 1)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx - 1] = future.result()
                except Exception as exc:
                    logger.error("Repo index %d failed: %s", idx, exc)

        # Filter out any None entries from failures
        results = [r for r in results if r is not None]

        # Save aggregated summary
        summary_path = out_dir / "classification_summary.json"
        summary = self._build_summary(results)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved → %s", summary_path)

        return results

    def _build_summary(self, results: list[ManifestClassification]) -> dict:
        """Build aggregated summary across all manifests."""
        n = len(results)
        cat_present_count = {cat: 0 for cat in CATEGORIES}
        cat_section_frac_sum = {cat: 0.0 for cat in CATEGORIES}
        cat_char_frac_sum = {cat: 0.0 for cat in CATEGORIES}

        for mc in results:
            for cat in CATEGORIES:
                if mc.category_present.get(cat, False):
                    cat_present_count[cat] += 1
                cat_section_frac_sum[cat] += mc.category_section_fraction.get(cat, 0.0)
                cat_char_frac_sum[cat] += mc.category_char_fraction.get(cat, 0.0)

        return {
            "n_manifests": n,
            "categories": CATEGORIES,
            "category_present_count": cat_present_count,
            "category_present_fraction": {
                cat: cat_present_count[cat] / n for cat in CATEGORIES
            },
            "category_mean_section_fraction": {
                cat: cat_section_frac_sum[cat] / n for cat in CATEGORIES
            },
            "category_mean_char_fraction": {
                cat: cat_char_frac_sum[cat] / n for cat in CATEGORIES
            },
            "repos": [mc.full_name for mc in results],
        }


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify CLAUDE.md sections using NVIDIA NIM LLMs (JCSSE 2026)"
    )
    parser.add_argument(
        "--config",
        default="experiments/jcsse2026/configs/jcsse.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Classify a single repo by full_name (for testing)",
    )
    args = parser.parse_args()

    classifier = ContentClassifier(config_path=args.config)

    if args.single:
        cfg = classifier.config
        with open(cfg["data"]["phase1_repos"]) as f:
            repos = json.load(f)
        target = next((r for r in repos if r["full_name"] == args.single), None)
        if not target:
            logger.error("Repo %s not found in phase1_repos.json", args.single)
            return
        mc = classifier.classify_manifest(target)
        print(json.dumps(asdict(mc), indent=2))
        return

    classifier.run(output_dir=args.output, workers=args.workers)


if __name__ == "__main__":
    main()
