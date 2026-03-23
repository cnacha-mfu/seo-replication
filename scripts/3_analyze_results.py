"""
Content Analysis for JCSSE 2026 Paper.

Loads per-manifest classification JSONs from content_classifier.py and produces:
  RQ1: Category frequency table + stacked bar chart per manifest
  RQ2: Evolution content gap — % of manifests with zero evolution content
  RQ3: Correlation heatmap: manifest category vs. project age/phase/agent ratio

Outputs:
  results/rq1_category_distribution.json
  results/rq2_evolution_gap.json
  results/rq3_correlations.json
  results/figures/fig_stacked_bar.pgf
  results/figures/fig_evolution_gap.pgf
  results/figures/fig_correlation_heatmap.pgf

Paper: "What Do Human Agent Manifests Miss? An Empirical Content Analysis of CLAUDE.md Files"
Venue: JCSSE 2026

Used by: JCSSE 2026 paper (all RQs)

Usage:
    python experiments/jcsse2026/analyze_content.py \
        --config experiments/jcsse2026/configs/jcsse.yaml
"""

import json
import logging
import math
import os
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

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
    "workflow_process": "Workflow \\& Process",
    "architecture": "Architecture",
    "evolution_context": "Evolution Context",
    "agent_rules": "Agent Rules",
}

PHASES = ["growth", "stabilization", "decline"]


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_classifications(results_dir: Path) -> list[dict]:
    """Load all per-manifest classification JSONs."""
    files = sorted(results_dir.glob("*_classification.json"))
    if not files:
        raise FileNotFoundError(f"No classification JSONs found in {results_dir}")
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    logger.info("Loaded %d manifest classifications", len(data))
    return data


def load_population_data(pop_dir: Path) -> dict[str, dict]:
    """Load per-repo population JSONs, keyed by repo slug (owner__name)."""
    pop_files = sorted(pop_dir.glob("*.json"))
    pop: dict[str, dict] = {}
    for f in pop_files:
        with open(f) as fh:
            d = json.load(fh)
        # Key by owner__name to match classifier output
        repo_slug = f.stem  # filename without .json
        pop[repo_slug] = d
    logger.info("Loaded %d population records", len(pop))
    return pop


def load_phase1_repos(repos_path: str) -> dict[str, dict]:
    """Load phase1_repos.json, keyed by full_name."""
    with open(repos_path) as f:
        repos = json.load(f)
    return {r["full_name"]: r for r in repos}


# ── RQ1: Category distribution ────────────────────────────────────────────────

def compute_rq1(classifications: list[dict]) -> dict:
    """Compute category frequency table and per-manifest distributions."""
    n = len(classifications)
    cat_present_count = {cat: 0 for cat in CATEGORIES}
    cat_section_fracs: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    cat_char_fracs: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}

    per_manifest = []
    for mc in classifications:
        row = {
            "repo": mc["full_name"],
            "n_sections": mc["n_sections"],
            "n_chars": mc["n_chars"],
        }
        for cat in CATEGORIES:
            present = mc["category_present"].get(cat, False)
            sec_frac = mc["category_section_fraction"].get(cat, 0.0)
            char_frac = mc["category_char_fraction"].get(cat, 0.0)
            if present:
                cat_present_count[cat] += 1
            cat_section_fracs[cat].append(sec_frac)
            cat_char_fracs[cat].append(char_frac)
            row[f"{cat}_present"] = present
            row[f"{cat}_sec_frac"] = round(sec_frac, 4)
            row[f"{cat}_char_frac"] = round(char_frac, 4)
        per_manifest.append(row)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def stdev(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    category_stats = {}
    for cat in CATEGORIES:
        category_stats[cat] = {
            "label": CATEGORY_LABELS[cat],
            "n_present": cat_present_count[cat],
            "pct_present": round(cat_present_count[cat] / n * 100, 1),
            "mean_section_frac": round(mean(cat_section_fracs[cat]), 4),
            "std_section_frac": round(stdev(cat_section_fracs[cat]), 4),
            "mean_char_frac": round(mean(cat_char_fracs[cat]), 4),
            "std_char_frac": round(stdev(cat_char_fracs[cat]), 4),
        }

    return {
        "n_manifests": n,
        "category_stats": category_stats,
        "per_manifest": per_manifest,
    }


# ── RQ2: Evolution gap ────────────────────────────────────────────────────────

EVOLUTION_DIMENSIONS = [
    "phase_mentioned",
    "complexity_trend",
    "law_compliance",
    "churn_trajectory",
    "tech_debt",
    "refactoring_plans",
]

# Keywords to detect each evolution sub-dimension in section text
EVOLUTION_KEYWORDS: dict[str, list[str]] = {
    "phase_mentioned": [
        "growth phase", "stabilization", "decline", "mature", "maintenance phase",
        "early stage", "scaling", "phase", "lifecycle",
    ],
    "complexity_trend": [
        "complexity", "cyclomatic", "technical complexity", "growing complex",
        "getting complex", "complexity metric",
    ],
    "law_compliance": [
        "lehman", "law of", "continuing change", "increasing complexity",
        "self-regulation", "conservation", "evolution law",
    ],
    "churn_trajectory": [
        "churn", "turnover", "contributor", "bus factor", "knowledge",
        "team change", "ownership",
    ],
    "tech_debt": [
        "tech debt", "technical debt", "debt", "legacy", "refactor", "cleanup",
        "todo", "fixme", "workaround",
    ],
    "refactoring_plans": [
        "refactor", "rewrite", "migration", "modernization", "upgrade path",
        "planned improvement", "future work",
    ],
}


def detect_evolution_dimensions(mc: dict) -> dict[str, bool]:
    """Detect which evolution sub-dimensions appear anywhere in the manifest."""
    # Collect all section body text
    all_text = " ".join(
        sec.get("heading", "") + " " +
        " ".join(
            cls.get("reasoning", "")
            for cls in sec.get("model_classifications", [])
        )
        for sec in mc.get("sections", [])
    ).lower()

    # Also include original manifest text if available (not stored — use classification labels)
    # Primary signal: sections classified as evolution_context + keyword scan of reasonings
    results = {}
    for dim, keywords in EVOLUTION_KEYWORDS.items():
        results[dim] = any(kw in all_text for kw in keywords)
    return results


def compute_rq2(classifications: list[dict]) -> dict:
    """Compute evolution content gap analysis."""
    n = len(classifications)
    evo_absent_count = 0
    dim_present_counts = {dim: 0 for dim in EVOLUTION_DIMENSIONS}
    per_manifest = []

    for mc in classifications:
        evo_present = mc["category_present"].get("evolution_context", False)
        evo_char_frac = mc["category_char_fraction"].get("evolution_context", 0.0)
        evo_sec_frac = mc["category_section_fraction"].get("evolution_context", 0.0)

        if not evo_present:
            evo_absent_count += 1

        dim_flags = detect_evolution_dimensions(mc)
        for dim, flag in dim_flags.items():
            if flag:
                dim_present_counts[dim] += 1

        per_manifest.append({
            "repo": mc["full_name"],
            "evolution_context_present": evo_present,
            "evolution_char_frac": round(evo_char_frac, 4),
            "evolution_sec_frac": round(evo_sec_frac, 4),
            **{f"dim_{dim}": flag for dim, flag in dim_flags.items()},
        })

    return {
        "n_manifests": n,
        "n_evolution_absent": evo_absent_count,
        "pct_evolution_absent": round(evo_absent_count / n * 100, 1),
        "evolution_dimension_counts": dim_present_counts,
        "evolution_dimension_pct": {
            dim: round(dim_present_counts[dim] / n * 100, 1)
            for dim in EVOLUTION_DIMENSIONS
        },
        "per_manifest": per_manifest,
    }


# ── RQ3: Correlations ─────────────────────────────────────────────────────────

def compute_rq3(
    classifications: list[dict],
    pop_data: dict[str, dict],
    phase1_repos: dict[str, dict],
) -> dict:
    """Correlate category presence with project characteristics.

    Project characteristics:
      - age_months (from created_at to last processed month)
      - stars (from phase1_repos)
      - total_commits
      - phase (encoded: growth=0, stabilization=1, decline=2)
      - law_conformance_score (fraction of 8 laws conforming)
      - agent_ratio (proxied by manifest length / total_commits)

    Returns Spearman correlations between each category's char_frac
    and each project characteristic.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        logger.error("scipy required: pip install scipy")
        return {}

    import datetime

    rows = []
    for mc in classifications:
        full_name = mc["full_name"]
        repo_slug = mc["repo"]

        pop = pop_data.get(repo_slug, {})
        phase1 = phase1_repos.get(full_name, {})

        if not pop:
            logger.warning("No population data for %s", full_name)
            continue

        # Project age in months from created_at
        created_str = phase1.get("created_at", "")
        if created_str:
            try:
                created = datetime.datetime.fromisoformat(
                    created_str.replace("Z", "+00:00")
                )
                now = datetime.datetime(2026, 3, 19, tzinfo=datetime.timezone.utc)
                age_months = (now.year - created.year) * 12 + (now.month - created.month)
            except ValueError:
                age_months = pop.get("total_months", 0)
        else:
            age_months = pop.get("total_months", 0)

        # Phase encoding
        phase_str = pop.get("phase", "stabilization").lower()
        phase_enc = {"growth": 0, "stabilization": 1, "decline": 2}.get(phase_str, 1)

        # Law conformance score
        law_conf = pop.get("law_conformance", {})
        if law_conf:
            n_laws = len(law_conf)
            conf_score = sum(1 for v in law_conf.values() if v) / n_laws
        else:
            conf_score = float("nan")

        # Manifest length proxy for agent investment
        manifest_chars = mc["n_chars"]
        total_commits = pop.get("total_commits", 1) or 1
        agent_ratio = manifest_chars / total_commits  # chars per commit

        row = {
            "full_name": full_name,
            "age_months": age_months,
            "stars": phase1.get("stars", 0),
            "total_commits": total_commits,
            "phase_enc": phase_enc,
            "law_conformance_score": conf_score,
            "agent_ratio": agent_ratio,
        }
        for cat in CATEGORIES:
            row[f"{cat}_char_frac"] = mc["category_char_fraction"].get(cat, 0.0)
            row[f"{cat}_present"] = int(mc["category_present"].get(cat, False))
        rows.append(row)

    if not rows:
        logger.warning("No rows for RQ3 correlation — check population data")
        return {"n": 0, "correlations": {}}

    # Build vectors
    char_vars = [f"{cat}_char_frac" for cat in CATEGORIES]
    proj_vars = ["age_months", "stars", "total_commits", "phase_enc",
                 "law_conformance_score", "agent_ratio"]

    import math as _math

    def clean(vals: list) -> list[float]:
        return [float(v) if not _math.isnan(float(v)) else 0.0 for v in vals]

    correlations = {}
    for cvar in char_vars:
        correlations[cvar] = {}
        x = clean([r[cvar] for r in rows])
        for pvar in proj_vars:
            y_raw = [r.get(pvar, 0.0) for r in rows]
            try:
                y = clean(y_raw)
                rho, pval = spearmanr(x, y)
                correlations[cvar][pvar] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(pval), 4),
                    "significant": float(pval) < 0.05,
                }
            except Exception as exc:
                logger.warning("Correlation %s vs %s failed: %s", cvar, pvar, exc)
                correlations[cvar][pvar] = {"spearman_rho": float("nan"), "p_value": 1.0}

    return {
        "n": len(rows),
        "category_vars": char_vars,
        "project_vars": proj_vars,
        "correlations": correlations,
        "rows": rows,
    }


# ── PGF figure generators ─────────────────────────────────────────────────────

def generate_stacked_bar_pgf(rq1: dict, out_path: Path) -> None:
    """Generate PGFPlots stacked bar chart: category fractions per manifest."""
    per_manifest = rq1["per_manifest"]
    # Sort by evolution_context_char_frac descending, then other cats
    per_manifest_sorted = sorted(
        per_manifest,
        key=lambda r: r.get("evolution_context_char_frac", 0),
        reverse=True,
    )

    # Short repo labels (last part of full_name)
    labels = [r["repo"].split("/")[-1][:20] for r in per_manifest_sorted]
    label_str = "{" + ",".join(labels) + "}"

    # Build addplot lines per category
    colors = {
        "coding_style": "blue!60",
        "project_structure": "teal!60",
        "workflow_process": "orange!70",
        "architecture": "red!60",
        "evolution_context": "purple!70",
        "agent_rules": "gray!60",
    }
    cat_labels = {
        "coding_style": "Coding Style",
        "project_structure": "Proj. Structure",
        "workflow_process": "Workflow",
        "architecture": "Architecture",
        "evolution_context": "Evolution",
        "agent_rules": "Agent Rules",
    }

    addplots = []
    for cat in CATEGORIES:
        fracs = [r.get(f"{cat}_char_frac", 0.0) for r in per_manifest_sorted]
        coords = " ".join(f"({i+1},{v:.4f})" for i, v in enumerate(fracs))
        addplots.append(
            f"    \\addplot+[ybar stacked, fill={colors[cat]}, draw={colors[cat]}]\n"
            f"        coordinates {{{coords}}};\n"
            f"    \\addlegendentry{{{cat_labels[cat]}}}"
        )

    n = len(per_manifest_sorted)
    pgf = f"""% Figure: Stacked bar chart — category content fractions per manifest
% Generated by analyze_content.py (JCSSE 2026)
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar stacked,
    bar width=6pt,
    width=\\columnwidth,
    height=6cm,
    xlabel={{Repository (sorted by Evolution Context fraction)}},
    ylabel={{Fraction of Manifest Content}},
    xtick={{1,...,{n}}},
    xticklabels={label_str},
    xticklabel style={{rotate=45, anchor=east, font=\\tiny}},
    ymin=0, ymax=1,
    ymajorgrids=true,
    legend style={{
        at={{(0.5,-0.35)}},
        anchor=north,
        legend columns=3,
        font=\\footnotesize,
    }},
    enlarge x limits=0.05,
]
{chr(10).join(addplots)}
\\end{{axis}}
\\end{{tikzpicture}}
"""
    out_path.write_text(pgf, encoding="utf-8")
    logger.info("Stacked bar chart → %s", out_path)


def generate_evolution_gap_pgf(rq2: dict, out_path: Path) -> None:
    """Generate PGFPlots horizontal bar chart — evolution dimension coverage."""
    dims = EVOLUTION_DIMENSIONS
    dim_labels = {
        "phase_mentioned": "Phase Mentioned",
        "complexity_trend": "Complexity Trend",
        "law_compliance": "Law Compliance",
        "churn_trajectory": "Churn Trajectory",
        "tech_debt": "Tech Debt",
        "refactoring_plans": "Refactoring Plans",
    }
    pcts = [rq2["evolution_dimension_pct"].get(d, 0.0) for d in dims]
    label_str = "{" + ",".join(dim_labels[d] for d in dims) + "}"
    coords = " ".join(f"({p:.1f},{i+1})" for i, p in enumerate(pcts))

    pgf = f"""% Figure: Evolution dimension coverage gap
% Generated by analyze_content.py (JCSSE 2026)
\\begin{{tikzpicture}}
\\begin{{axis}}[
    xbar,
    bar width=10pt,
    width=\\columnwidth,
    height=5cm,
    xlabel={{\\% of Manifests Containing Dimension}},
    ytick={{1,...,{len(dims)}}},
    yticklabels={label_str},
    yticklabel style={{font=\\small}},
    xmin=0, xmax=100,
    xmajorgrids=true,
    nodes near coords,
    nodes near coords style={{font=\\tiny}},
]
    \\addplot+[xbar, fill=purple!60, draw=purple!80]
        coordinates {{{coords}}};
\\end{{axis}}
\\end{{tikzpicture}}
"""
    out_path.write_text(pgf, encoding="utf-8")
    logger.info("Evolution gap chart → %s", out_path)


def generate_correlation_heatmap_pgf(rq3: dict, out_path: Path) -> None:
    """Generate PGFPlots matrix plot — Spearman ρ heatmap."""
    if not rq3.get("correlations"):
        logger.warning("No RQ3 data — skipping heatmap")
        return

    cat_vars = [f"{cat}_char_frac" for cat in CATEGORIES]
    proj_vars = rq3.get("project_vars", [])
    proj_labels = {
        "age_months": "Age (months)",
        "stars": "Stars",
        "total_commits": "Commits",
        "phase_enc": "Phase",
        "law_conformance_score": "Law Score",
        "agent_ratio": "Agent Ratio",
    }
    cat_labels_short = {
        "coding_style_char_frac": "Coding",
        "project_structure_char_frac": "Structure",
        "workflow_process_char_frac": "Workflow",
        "architecture_char_frac": "Arch.",
        "evolution_context_char_frac": "Evolution",
        "agent_rules_char_frac": "Agent",
    }

    correlations = rq3["correlations"]
    n_rows = len(cat_vars)
    n_cols = len(proj_vars)

    # Build matrix data for pgfplots matrix plot
    matrix_data = []
    for i, cvar in enumerate(cat_vars):
        for j, pvar in enumerate(proj_vars):
            rho = correlations.get(cvar, {}).get(pvar, {}).get("spearman_rho", 0.0)
            if rho != rho:  # NaN check
                rho = 0.0
            matrix_data.append(f"({j+1},{i+1}) [{rho:.2f}]")

    coords = "\n    ".join(matrix_data)
    col_labels = "{" + ",".join(proj_labels.get(p, p) for p in proj_vars) + "}"
    row_labels = "{" + ",".join(cat_labels_short.get(c, c) for c in cat_vars) + "}"

    pgf = f"""% Figure: Spearman rho correlation heatmap
% Generated by analyze_content.py (JCSSE 2026)
\\begin{{tikzpicture}}
\\begin{{axis}}[
    matrix plot,
    width=\\columnwidth,
    height=6cm,
    xtick={{1,...,{n_cols}}},
    xticklabels={col_labels},
    xticklabel style={{rotate=45, anchor=east, font=\\small}},
    ytick={{1,...,{n_rows}}},
    yticklabels={row_labels},
    yticklabel style={{font=\\small}},
    colormap/RdBu,
    colorbar,
    colorbar style={{ylabel={{Spearman $\\rho$}}}},
    point meta min=-1, point meta max=1,
]
    \\addplot[
        matrix plot*,
        mesh/cols={n_cols},
        point meta=explicit,
    ] coordinates {{
    {coords}
    }};
\\end{{axis}}
\\end{{tikzpicture}}
"""
    out_path.write_text(pgf, encoding="utf-8")
    logger.info("Correlation heatmap → %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze CLAUDE.md content classifications (JCSSE 2026)"
    )
    parser.add_argument(
        "--config",
        default="experiments/jcsse2026/configs/jcsse.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["data"]["output_dir"])
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    classifications = load_classifications(out_dir)

    pop_dir = Path(config["data"]["phase2_population"])
    phase1_path = config["data"]["phase1_repos"]
    pop_data = load_population_data(pop_dir)
    phase1_repos = load_phase1_repos(phase1_path)

    # ── RQ1 ───────────────────────────────────────────────────────────────────
    logger.info("Computing RQ1: category distribution")
    rq1 = compute_rq1(classifications)
    rq1_path = out_dir / "rq1_category_distribution.json"
    with open(rq1_path, "w") as f:
        json.dump(rq1, f, indent=2)
    logger.info("RQ1 → %s", rq1_path)

    # Print category table to console
    print("\n=== RQ1: Category Distribution ===")
    print(f"{'Category':<30} {'N present':>10} {'% present':>10} {'Mean char%':>12}")
    print("-" * 65)
    for cat, stats in rq1["category_stats"].items():
        print(
            f"{stats['label']:<30} {stats['n_present']:>10} "
            f"{stats['pct_present']:>9.1f}% {stats['mean_char_frac']*100:>11.1f}%"
        )

    # ── RQ2 ───────────────────────────────────────────────────────────────────
    logger.info("Computing RQ2: evolution gap")
    rq2 = compute_rq2(classifications)
    rq2_path = out_dir / "rq2_evolution_gap.json"
    with open(rq2_path, "w") as f:
        json.dump(rq2, f, indent=2)
    logger.info("RQ2 → %s", rq2_path)

    print(f"\n=== RQ2: Evolution Content Gap ===")
    print(
        f"Manifests with NO evolution content: "
        f"{rq2['n_evolution_absent']}/{rq2['n_manifests']} "
        f"({rq2['pct_evolution_absent']:.1f}%)"
    )
    print("\nEvolution sub-dimension coverage:")
    for dim in EVOLUTION_DIMENSIONS:
        pct = rq2["evolution_dimension_pct"].get(dim, 0)
        print(f"  {dim:<25} {pct:>5.1f}%")

    # ── RQ3 ───────────────────────────────────────────────────────────────────
    logger.info("Computing RQ3: correlations")
    rq3 = compute_rq3(classifications, pop_data, phase1_repos)
    rq3_path = out_dir / "rq3_correlations.json"
    with open(rq3_path, "w") as f:
        json.dump(rq3, f, indent=2)
    logger.info("RQ3 → %s", rq3_path)

    if rq3.get("correlations"):
        print("\n=== RQ3: Significant Correlations (p < 0.05) ===")
        for cvar, proj_dict in rq3["correlations"].items():
            for pvar, stats in proj_dict.items():
                if stats.get("significant"):
                    print(
                        f"  {cvar} × {pvar}: "
                        f"ρ={stats['spearman_rho']:.3f}, p={stats['p_value']:.3f}"
                    )

    # ── Figures ───────────────────────────────────────────────────────────────
    logger.info("Generating PGF figures")
    generate_stacked_bar_pgf(rq1, fig_dir / "fig_stacked_bar.pgf")
    generate_evolution_gap_pgf(rq2, fig_dir / "fig_evolution_gap.pgf")
    generate_correlation_heatmap_pgf(rq3, fig_dir / "fig_correlation_heatmap.pgf")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
