# Replication Package — JCSSE 2026

**"Evolution Context Gap: An Empirical Study of Agent Manifests"**

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

**80 repositories** from public GitHub across 4 manifest types.
**Languages**: TypeScript (20), Python (17), C# (7), Rust (6), Go (5), and others.
**Stars**: 30–49,611 (median 250); **Age**: 5–183 months (median 38).

| # | Repository | Manifest Type | Language | Stars | Age (mo) | Evolution |
|--:|:----------|:-------------:|:--------:|------:|---------:|:---------:|
| 1 | [acepanel/panel](https://github.com/acepanel/panel) | `CLAUDE.md` | Go | 2,626 | 39 | — |
| 2 | [ada-url/ada](https://github.com/ada-url/ada) | `CLAUDE.md` | C++ | 1,706 | 38 | — |
| 3 | [apache/skywalking](https://github.com/apache/skywalking) | `CLAUDE.md` | Java | 24,747 | 125 | — |
| 4 | [aptos-labs/aptos-docs](https://github.com/aptos-labs/aptos-docs) | `CLAUDE.md` | Mdx | 54 | 13 | — |
| 5 | [astrolabsoftware/fink-broker](https://github.com/astrolabsoftware/fink-broker) | `CLAUDE.md` | Python | 79 | 85 | — |
| 6 | [AudiusProject/apps](https://github.com/AudiusProject/apps) | `CLAUDE.md` | Typescript | 607 | 79 | — |
| 7 | [bpowers/simlin](https://github.com/bpowers/simlin) | `CLAUDE.md` | Rust | 92 | 94 | ✓ |
| 8 | [braintree/braintree-web](https://github.com/braintree/braintree-web) | `CLAUDE.md` | Javascript | 448 | 137 | — |
| 9 | [carpenike/k8s-gitops](https://github.com/carpenike/k8s-gitops) | `CLAUDE.md` | Shell | 310 | 74 | — |
| 10 | [cbyrohl/scida](https://github.com/cbyrohl/scida) | `CLAUDE.md` | Python | 41 | 50 | — |
| 11 | [cellwebb/gac](https://github.com/cellwebb/gac) | `CLAUDE.md` | Python | 303 | 14 | — |
| 12 | [cypress-io/cypress](https://github.com/cypress-io/cypress) | `CLAUDE.md` | Typescript | 49,611 | 133 | — |
| 13 | [derrickburns/generalized-kmeans-clustering](https://github.com/derrickburns/generalized-kmeans-clustering) | `CLAUDE.md` | Scala | 342 | 139 | ✓ |
| 14 | [felipebarcelospro/igniter-js](https://github.com/felipebarcelospro/igniter-js) | `CLAUDE.md` | Typescript | 241 | 15 | ✓ |
| 15 | [getsentry/sentry-docs](https://github.com/getsentry/sentry-docs) | `CLAUDE.md` | Mdx | 418 | 132 | — |
| 16 | [hapifhir/org.hl7.fhir.validator-wrapper](https://github.com/hapifhir/org.hl7.fhir.validator-wrapper) | `CLAUDE.md` | Kotlin | 47 | 67 | ✓ |
| 17 | [HassanZahirnia/laravel-package-ocean](https://github.com/HassanZahirnia/laravel-package-ocean) | `CLAUDE.md` | Blade | 223 | 35 | ✓ |
| 18 | [huangwb8/ChineseResearchLaTeX](https://github.com/huangwb8/ChineseResearchLaTeX) | `CLAUDE.md` | Python | 1,273 | 24 | — |
| 19 | [javascript-obfuscator/javascript-obfuscator](https://github.com/javascript-obfuscator/javascript-obfuscator) | `CLAUDE.md` | Typescript | 15,908 | 119 | ✓ |
| 20 | [jonit-dev/vibe-coder-3d](https://github.com/jonit-dev/vibe-coder-3d) | `CLAUDE.md` | Typescript | 42 | 10 | — |
| 21 | [kubernetes-sigs/cloud-provider-azure](https://github.com/kubernetes-sigs/cloud-provider-azure) | `CLAUDE.md` | Go | 292 | 96 | — |
| 22 | [lambrospetrou/durable-utils](https://github.com/lambrospetrou/durable-utils) | `CLAUDE.md` | Typescript | 85 | 16 | — |
| 23 | [lerna/lerna](https://github.com/lerna/lerna) | `CLAUDE.md` | Typescript | 36,093 | 124 | — |
| 24 | [managedcode/Storage](https://github.com/managedcode/Storage) | `CLAUDE.md` | C# | 132 | 59 | — |
| 25 | [nanobrowser/nanobrowser](https://github.com/nanobrowser/nanobrowser) | `CLAUDE.md` | Typescript | 12,462 | 14 | ✓ |
| 26 | [nats-io/nats.go](https://github.com/nats-io/nats.go) | `CLAUDE.md` | Go | 6,498 | 164 | — |
| 27 | [noahlh/celestite](https://github.com/noahlh/celestite) | `CLAUDE.md` | Javascript | 236 | 92 | — |
| 28 | [pcingola/SnpEff](https://github.com/pcingola/SnpEff) | `CLAUDE.md` | Java | 303 | 147 | — |
| 29 | [popstas/telegram-download-chat](https://github.com/popstas/telegram-download-chat) | `CLAUDE.md` | Python | 83 | 9 | — |
| 30 | [rabbitmq/rabbitmqadmin-ng](https://github.com/rabbitmq/rabbitmqadmin-ng) | `CLAUDE.md` | Rust | 47 | 33 | — |
| 31 | [rpmuller/pyquante2](https://github.com/rpmuller/pyquante2) | `CLAUDE.md` | Python | 159 | 181 | ✓ |
| 32 | [run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS) | `CLAUDE.md` | Typescript | 3,068 | 33 | ✓ |
| 33 | [rydmike/flex_seed_scheme](https://github.com/rydmike/flex_seed_scheme) | `CLAUDE.md` | Dart | 35 | 42 | ✓ |
| 34 | [saleor/macaw-ui](https://github.com/saleor/macaw-ui) | `CLAUDE.md` | Typescript | 115 | 80 | — |
| 35 | [scopewu/qrcode.vue](https://github.com/scopewu/qrcode.vue) | `CLAUDE.md` | Typescript | 810 | 108 | — |
| 36 | [shiguredo/mp4-rs](https://github.com/shiguredo/mp4-rs) | `CLAUDE.md` | Rust | 151 | 18 | — |
| 37 | [simstudioai/sim](https://github.com/simstudioai/sim) | `CLAUDE.md` | Typescript | 27,023 | 13 | — |
| 38 | [snowplow/snowplow-javascript-tracker](https://github.com/snowplow/snowplow-javascript-tracker) | `CLAUDE.md` | Typescript | 581 | 153 | — |
| 39 | [the-cafe/git-ai-commit](https://github.com/the-cafe/git-ai-commit) | `CLAUDE.md` | Python | 82 | 19 | — |
| 40 | [tobymao/sqlglot](https://github.com/tobymao/sqlglot) | `CLAUDE.md` | Python | 9,047 | 60 | — |
| 41 | [TopSwagCode/MinimalWorker](https://github.com/TopSwagCode/MinimalWorker) | `CLAUDE.md` | C# | 276 | 10 | — |
| 42 | [tuist/tuist](https://github.com/tuist/tuist) | `CLAUDE.md` | Swift | 5,579 | 95 | ✓ |
| 43 | [VCnoC/Claude-Code-Zen-mcp-Skill-Work](https://github.com/VCnoC/Claude-Code-Zen-mcp-Skill-Work) | `CLAUDE.md` | Python | 114 | 4 | — |
| 44 | [Vortiago/mcp-outline](https://github.com/Vortiago/mcp-outline) | `CLAUDE.md` | Python | 116 | 11 | — |
| 45 | [xoofx/Tomlyn](https://github.com/xoofx/Tomlyn) | `CLAUDE.md` | C# | 535 | 86 | — |
| 46 | [zigflow/zigflow](https://github.com/zigflow/zigflow) | `CLAUDE.md` | Go | 106 | 7 | ✓ |
| 47 | [agentjido/jido_ai](https://github.com/agentjido/jido_ai) | `AGENTS.md` | Elixir | 163 | 14 | — |
| 48 | [akirak/flake-templates](https://github.com/akirak/flake-templates) | `AGENTS.md` | Nix | 205 | 52 | — |
| 49 | [bbc/simorgh](https://github.com/bbc/simorgh) | `AGENTS.md` | Typescript | 1,678 | 94 | ✓ |
| 50 | [byteowlz/eaRS](https://github.com/byteowlz/eaRS) | `AGENTS.md` | Rust | 57 | 8 | — |
| 51 | [chmouel/nextmeeting](https://github.com/chmouel/nextmeeting) | `AGENTS.md` | Python | 57 | 37 | — |
| 52 | [DataDog/lading](https://github.com/DataDog/lading) | `AGENTS.md` | Rust | 94 | 60 | ✓ |
| 53 | [dermatologist/fhiry](https://github.com/dermatologist/fhiry) | `AGENTS.md` | Jupyter Notebook | 44 | 63 | — |
| 54 | [funnyzak/weread-bot](https://github.com/funnyzak/weread-bot) | `AGENTS.md` | Python | 124 | 5 | — |
| 55 | [iflow-ai/NioPD](https://github.com/iflow-ai/NioPD) | `AGENTS.md` | Javascript | 113 | 6 | — |
| 56 | [ipspace/netlab](https://github.com/ipspace/netlab) | `AGENTS.md` | Python | 651 | 63 | — |
| 57 | [Kaptensanders/skolmat](https://github.com/Kaptensanders/skolmat) | `AGENTS.md` | Python | 30 | 48 | ✓ |
| 58 | [Lakr233/ColorfulX](https://github.com/Lakr233/ColorfulX) | `AGENTS.md` | Swift | 439 | 27 | — |
| 59 | [mscraftsman/generative-ai](https://github.com/mscraftsman/generative-ai) | `AGENTS.md` | C# | 207 | 24 | — |
| 60 | [openai/openai-agents-js](https://github.com/openai/openai-agents-js) | `AGENTS.md` | Typescript | 2,486 | 9 | ✓ |
| 61 | [Peiiii/AgentVerse](https://github.com/Peiiii/AgentVerse) | `AGENTS.md` | Typescript | 281 | 12 | — |
| 62 | [reliverse/experiments](https://github.com/reliverse/experiments) | `AGENTS.md` | Typescript | 118 | 30 | — |
| 63 | [rstackjs/rstack-examples](https://github.com/rstackjs/rstack-examples) | `AGENTS.md` | Typescript | 155 | 27 | — |
| 64 | [sebastien/monitoring](https://github.com/sebastien/monitoring) | `AGENTS.md` | Python | 437 | 182 | — |
| 65 | [sunny-chung/hello-http](https://github.com/sunny-chung/hello-http) | `AGENTS.md` | Kotlin | 110 | 28 | — |
| 66 | [vincent-uden/miro](https://github.com/vincent-uden/miro) | `AGENTS.md` | Rust | 426 | 11 | — |
| 67 | [WordPress/wp-ai-client](https://github.com/WordPress/wp-ai-client) | `AGENTS.md` | Php | 115 | 6 | — |
| 68 | [a16z/halmos](https://github.com/a16z/halmos) | `copilot-instructions.md` | Python | 979 | 39 | ✓ |
| 69 | [chef/chef-vault](https://github.com/chef/chef-vault) | `copilot-instructions.md` | Ruby | 408 | 156 | ✓ |
| 70 | [coffebar/dotfiles](https://github.com/coffebar/dotfiles) | `copilot-instructions.md` | Lua | 250 | 73 | — |
| 71 | [daohainam/microservice-patterns](https://github.com/daohainam/microservice-patterns) | `copilot-instructions.md` | C# | 322 | 12 | — |
| 72 | [dotnet/macios](https://github.com/dotnet/macios) | `copilot-instructions.md` | C# | 2,837 | 120 | — |
| 73 | [flightphp/core](https://github.com/flightphp/core) | `copilot-instructions.md` | Php | 2,848 | 181 | — |
| 74 | [github/gh-models](https://github.com/github/gh-models) | `copilot-instructions.md` | Go | 186 | 18 | — |
| 75 | [jaxen-xpath/jaxen](https://github.com/jaxen-xpath/jaxen) | `copilot-instructions.md` | Java | 87 | 131 | ✓ |
| 76 | [linux-do/cdk](https://github.com/linux-do/cdk) | `copilot-instructions.md` | Typescript | 649 | 9 | — |
| 77 | [microsoftgraph/msgraph-sdk-dotnet](https://github.com/microsoftgraph/msgraph-sdk-dotnet) | `copilot-instructions.md` | C# | 779 | 121 | — |
| 78 | [vinhnx/VT.ai](https://github.com/vinhnx/VT.ai) | `copilot-instructions.md` | Python | 106 | 22 | ✓ |
| 79 | [yf-yang/v0-copilot](https://github.com/yf-yang/v0-copilot) | `copilot-instructions.md` | Typescript | 43 | 14 | — |
| 80 | [JunDamin/hwpapi](https://github.com/JunDamin/hwpapi) | `.clinerules` | Jupyter Notebook | 30 | 37 | ✓ |

*80 repositories. ✓ = evolution context detected; — = absent. Sorted by manifest type then alphabetically.*

---

## Citation

```bibtex
@inproceedings{jcsse2026manifests,
  title     = {Evolution Context Gap: An Empirical Study of Agent Manifests},
  author    = {[Author names omitted for blind review]},
  booktitle = {Proceedings of the International Joint Conference on
               Computer Science and Software Engineering (JCSSE)},
  year      = {2026}
}
```
