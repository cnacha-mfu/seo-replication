# Replication Package — JCSSE 2026

**"What Do Human Agent Manifests Miss? An Empirical Content Analysis of AI Coding Agent Instruction Files"**

This package contains all data, scripts, and results needed to replicate the study reported in the paper.

---

## Repository Structure

```
replication/
├── data/
│   ├── expanded_corpus.json          # 80-repository corpus (metadata + manifest text)
│   ├── classifications/              # Per-repository LLM classification outputs (80 files)
│   └── results/
│       ├── rq1_category_distribution.json   # RQ1: category presence & fractions
│       ├── rq2_evolution_gap.json           # RQ2: evolution sub-dimension breakdown
│       ├── rq3_correlations.json            # RQ3: Spearman ρ table
│       └── classification_summary.json     # Aggregated classification results
├── figures/
│   └── fig_evolution_gap.pgf         # PGFPlots figure for RQ2 (used in paper)
├── scripts/
│   ├── 1_expand_corpus.py            # Step 1: collect manifests from GitHub
│   ├── 2_classify_manifests.py       # Step 2: LLM classification (3-model majority vote)
│   ├── 3_analyze_results.py          # Step 3: compute RQ1–RQ3 statistics
│   ├── gen_appendix.py               # Generate LaTeX appendix table rows
│   └── jcsse.yaml                    # Configuration (taxonomy, models, paths)
└── requirements.txt
```

---

## Reproducing the Results

### Prerequisites

- Python 3.11+
- A GitHub personal access token (for corpus collection only)
- Three NVIDIA NIM API keys (free tier — apply at https://build.nvidia.com/explore/discover)

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
export GITHUB_TOKEN=your_github_token
export NVIDIA_API_KEY=your_key_1       # meta/llama-3.3-70b-instruct
export NVIDIA_API_KEY_2=your_key_2     # nvidia/llama-3.3-nemotron-super-49b-v1
export NVIDIA_API_KEY_3=your_key_3     # mistralai/mixtral-8x22b-instruct-v0.1
```

### Step 1 — Corpus Collection (optional, results already included)

Collect 80 repositories containing agent manifest files:

```bash
python scripts/1_expand_corpus.py \
    --token $GITHUB_TOKEN \
    --target 80 \
    --output data/expanded_corpus.json
```

The corpus used in the paper is already provided in `data/expanded_corpus.json`.

### Step 2 — LLM Classification (optional, results already included)

Classify each manifest section using three NVIDIA NIM models with majority-vote consensus:

```bash
python scripts/2_classify_manifests.py \
    --corpus data/expanded_corpus.json \
    --config scripts/jcsse.yaml \
    --output data/classifications/
```

Individual per-repository classification files are already provided in `data/classifications/`.

### Step 3 — Statistical Analysis

Compute RQ1–RQ3 results from the classification outputs:

```bash
python scripts/3_analyze_results.py \
    --corpus data/expanded_corpus.json \
    --classifications data/classifications/ \
    --output data/results/
```

This regenerates `rq1_category_distribution.json`, `rq2_evolution_gap.json`, and
`rq3_correlations.json`. All three are already provided for direct inspection.

---

## Pre-computed Results

The `data/results/` directory contains all figures reported in the paper:

| File | Content |
|------|---------|
| `rq1_category_distribution.json` | Category presence % and mean content fractions (Table 1, Fig. 1) |
| `rq2_evolution_gap.json` | Evolution sub-dimension breakdown (Fig. 2) |
| `rq3_correlations.json` | 36 Spearman ρ values with p-values (Table 2) |
| `classification_summary.json` | Raw majority-vote labels for all 80 × N_sections entries |

---

## LLM Models Used

| Model | Organization | NVIDIA NIM ID |
|-------|-------------|---------------|
| Llama-3.3-70B | Meta | `meta/llama-3.3-70b-instruct` |
| Nemotron-Super-49B | NVIDIA | `nvidia/llama-3.3-nemotron-super-49b-v1` |
| Mixtral-8x22B | Mistral AI | `mistralai/mixtral-8x22b-instruct-v0.1` |

All three models are available on the NVIDIA NIM free tier. A category is assigned when
at least two of the three models agree (majority vote ≥ 2/3).

---

## Corpus Summary

- **80 repositories** from public GitHub
- **5 manifest types**: CLAUDE.md (46), AGENTS.md (21), copilot-instructions.md (12), .clinerules (1)
- **Languages**: TypeScript (20), Python (17), C# (7), Rust (6), Go (5), and others
- **Stars**: 30–49,611 (median 250); **Age**: 5–183 months (median 38)

---

## Citation

```bibtex
@inproceedings{jcsse2026manifests,
  title     = {What Do Human Agent Manifests Miss? An Empirical Content
               Analysis of {AI} Coding Agent Instruction Files},
  author    = {[Author names omitted for blind review]},
  booktitle = {Proceedings of the International Joint Conference on
               Computer Science and Software Engineering (JCSSE)},
  year      = {2026}
}
```
