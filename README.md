# SEO Replication Package

**Replication package for:**

> N. Chondamrongkul, *From Tribal Knowledge to Machine-Readable Evolution Context: An Ontology-Based Approach to Grounding AI Coding Agents in Software History*, Journal of Systems and Software (under review), 2026.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This repository contains all artifacts required to reproduce or independently verify the four research questions reported in the paper:

| RQ | Description | Pre-computed result |
|----|-------------|---------------------|
| RQ1 | Ontology correctness & convergent validity | `experiments/results/real/` (scenario bank + CQ) |
| RQ2 | Population pipeline accuracy | `experiments/results/real/rq2_real_population.json` |
| RQ3 | Manifest quality — LLM-as-Judge (918 judgments) | `experiments/results/real/rq3_real_manifest_quality.json` |
| RQ4 | Agentic task parity (42 tasks, 252 responses) | `experiments/results/real/rq4_real_agentic.json` |

---

## Repository Structure

```
seo-replication/
│
├── ontology/
│   ├── modules/                    # OWL 2 DL source files (5 modules)
│   │   ├── seo-entities.owl        # M1: SoftwareProject, Version, Commit, AgentCommit
│   │   ├── seo-processes.owl       # M2: EvolutionPhase (Growth/Stabilization/Decline)
│   │   ├── seo-metrics.owl         # M3: ComplexityMetric, ActivityMetric
│   │   ├── seo-laws.owl            # M4: 8 LehmanLaw individuals + 4 SWRL rules
│   │   └── seo-patterns.owl        # M5: ShadowTechDebt, DependencySprawl, ...
│   ├── scenario_bank/
│   │   ├── scenarios.yaml          # 20 evaluation scenarios with ground-truth labels
│   │   └── test_scenarios.py       # RQ1 runner: scenario bank evaluation
│   ├── competency_questions/
│   │   ├── cq_suite.yaml           # 15 SPARQL competency questions
│   │   └── test_cq.py              # RQ1 runner: CQ evaluation against Fuseki
│   └── population/
│       └── populate.py             # RQ2: six-stage population pipeline
│
├── shared/
│   └── llm_judge/
│       └── judge.py                # RQ3: LLM-as-Judge pipeline (5 models, NVIDIA NIM)
│
├── manifests/
│   ├── generator/
│   │   └── generate.py             # SPARQL → evolution-aware manifest generator
│   └── comparator/
│       └── compare.py              # Human vs. SENSO manifest comparison
│
├── experiments/
│   ├── run_real_experiments.py     # Full experiment runner (RQ2–RQ4)
│   ├── rq4_harness.py              # RQ4: agentic task evaluation harness
│   ├── configs/
│   │   ├── real_experiments.yaml   # Repository list, LLM config, lookback window
│   │   └── rq4.yaml                # RQ4 task and generator configuration
│   └── results/
│       └── real/
│           ├── phase1_repos.json               # 23-repository corpus metadata
│           ├── rq2_real_population.json        # RQ2: population statistics
│           ├── rq3_real_manifest_quality.json  # RQ3: 918 LLM judgments (raw)
│           ├── rq4_real_agentic.json           # RQ4: 252 external-judge scores
│           ├── phase2_population/              # Per-repo population data (23 files)
│           ├── phase3_manifests/               # SENSO-generated manifests (23 .md files)
│           └── phase4_agentic/
│               └── tasks.json                  # 42 coding task definitions
│
└── requirements.txt
```

---

## OWL Ontology Modules

Load into Protégé or Apache Jena Fuseki by importing `seo-patterns.owl` — it transitively imports all other modules via OWL `owl:imports` declarations.

| Module | File | Classes | Properties | Key Contents |
|--------|------|---------|------------|--------------|
| M1 — Evolution Entity | `seo-entities.owl` | 13 | 11 | `SoftwareProject`, `Version`, `Commit`, `AgentCommit` |
| M2 — Evolution Process | `seo-processes.owl` | 10 | 4 | `GrowthPhase`, `StabilizationPhase`, `DeclinePhase` |
| M3 — Metric | `seo-metrics.owl` | 8 | 7 | `ComplexityMetric`, `ActivityMetric`, `metricValue` |
| M4 — Evolution Law | `seo-laws.owl` | 3 | 4 | 8 `LehmanLaw` individuals, `LawConformance`, 4 SWRL rules |
| M5 — Pattern | `seo-patterns.owl` | 11 | 3 | `ShadowTechDebt`, `DependencySprawl`, `RefactoringAvoidance`, `CopyPasteOverReuse` |

**Total: 45 classes, 35 properties, 653 axioms, 4 SWRL rules.**

---

## Prerequisites

```bash
git clone https://github.com/cnacha-mfu/seo-replication.git
cd seo-replication
pip install -r requirements.txt
```

**Environment variables:**

```bash
export NVIDIA_API_KEY=<your-nim-key>     # free at https://build.nvidia.com  (RQ3)
export GITHUB_TOKEN=<your-github-token>  # GitHub API rate limits            (RQ2–RQ4)
```

A Fuseki triple store is required for RQ1 CQ evaluation and RQ2 population:

```bash
# Download Apache Jena Fuseki from https://jena.apache.org/download/
./fuseki-server --update --mem /seo    # starts at http://localhost:3030
```

---

## Reproducing Each RQ

### RQ1 — Ontology Correctness (Section 5.2)

```bash
# Scenario bank (20 scenarios, ground-truth labels)
python ontology/scenario_bank/test_scenarios.py

# SPARQL competency questions (15 CQs against populated Fuseki)
python ontology/competency_questions/test_cq.py \
  --endpoint http://localhost:3030/seo/sparql
```

Expected output: phase detection 95%, law conformance 84.4% (135/160), CQ pass rate 100%, false positive rate 0%.

### RQ2 — Population Pipeline (Section 5.3)

```bash
# Single repository example
python ontology/population/populate.py \
  --repo https://github.com/pallets/flask \
  --lookback 60 \
  --fuseki http://localhost:3030/seo \
  --output experiments/results/

# Full 23-repository corpus
python experiments/run_real_experiments.py \
  --config experiments/configs/real_experiments.yaml \
  --rq rq2
```

Pre-computed result: `experiments/results/real/rq2_real_population.json`

### RQ3 — Manifest Quality / LLM-as-Judge (Section 5.4)

```bash
python experiments/run_real_experiments.py \
  --config experiments/configs/real_experiments.yaml \
  --rq rq3
```

Pre-computed result: `experiments/results/real/rq3_real_manifest_quality.json`
(918 judgments: 23 pairs × 2 conditions × 4 dimensions × 5 LLM judges)

SENSO-generated manifests for all 23 repositories: `experiments/results/real/phase3_manifests/`

### RQ4 — Agentic Task Parity (Section 5.5)

```bash
python experiments/rq4_harness.py \
  --config experiments/configs/rq4.yaml
```

Pre-computed result: `experiments/results/real/rq4_real_agentic.json`
(252 responses: 42 tasks × 3 conditions × 2 generator models)

---

## Inspecting Pre-Computed Results Without Re-Running

All raw result files are committed to this repository. Reviewers can inspect the reported numbers directly:

```python
import json

# RQ3: Inter-rater agreement and effect sizes
with open("experiments/results/real/rq3_real_manifest_quality.json") as f:
    rq3 = json.load(f)
print(rq3["n_judgments"])          # 918
print(rq3["krippendorff_alpha"])   # per-dimension alpha values

# RQ4: Friedman test results and Cliff's delta
with open("experiments/results/real/rq4_real_agentic.json") as f:
    rq4 = json.load(f)
print(rq4["per_judge_rating"])     # mean scores per condition
```

---

## LLM Configuration

All LLM experiments use [NVIDIA NIM](https://build.nvidia.com) (free public API) with five models from five organizations. Edit `experiments/configs/real_experiments.yaml` to change model selection:

| Model | Organization | NIM identifier |
|-------|-------------|---------------|
| Llama 3.3 70B | Meta | `meta/llama-3.3-70b-instruct` |
| Nemotron Super 49B | NVIDIA | `nvidia/llama-3.3-nemotron-super-49b-v1` |
| Mixtral 8x22B | Mistral AI | `mistralai/mixtral-8x22b-instruct-v0.1` |
| Gemma 2 27B | Google | `google/gemma-2-27b-it` |
| Qwen 2 7B | Alibaba | `qwen/qwen2-7b-instruct` |

All calls use `temperature=0` for deterministic, reproducible outputs.

---

## License

MIT — see [LICENSE](LICENSE)

## Citation

If you use these artifacts, please cite:

```
N. Chondamrongkul, "From Tribal Knowledge to Machine-Readable Evolution Context:
An Ontology-Based Approach to Grounding AI Coding Agents in Software History,"
Journal of Systems and Software, under review, 2026.
https://github.com/cnacha-mfu/seo-replication
```
