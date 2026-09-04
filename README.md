# OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents

<p align="center">
  <a href="https://arxiv.org/abs/2601.13722">
    <img src="https://img.shields.io/badge/arXiv-2601.13722-b31b1b.svg?style=flat-square" alt="arXiv: 2601.13722">
  </a>
  <a href="https://2026.emnlp.org/">
    <img src="https://img.shields.io/badge/EMNLP-2026-6f42c1.svg?style=flat-square" alt="EMNLP 2026">
  </a>
  <a href="https://yulinlp.github.io/OP-Bench/">
    <img src="https://img.shields.io/badge/Website-live-2563eb.svg?style=flat-square" alt="Website live">
  </a>
</p>

<p align="center">
  <a href="README.md">
    <img src="https://img.shields.io/badge/English-README-1f6feb.svg?style=flat-square" alt="English README">
  </a>
  <a href="README.zh-CN.md">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-README.zh--CN-16a34a.svg?style=flat-square" alt="中文 README">
  </a>
</p>

<p align="center">
  <strong>A diagnostic benchmark for measuring when long-term memory makes a conversational agent personalize too much, too often, or in the wrong way.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.13722">Paper</a>
  ·
  <a href="#quick-start">Quick start</a>
  ·
  <a href="data/README.md">Data</a>
  ·
  <a href="#citation">Citation</a>
</p>

## Overview

Memory-augmented conversational agents are designed to use long-term user
information to provide helpful personalization. OP-Bench studies the other
side of this capability: **over-personalization**, where an agent introduces
personal information unnecessarily, agrees with a user at the expense of
accuracy or neutrality, or repeats the same personalized content across
different questions.

OP-Bench contains 1,700 human-verified instances constructed from long-horizon
dialogue histories. It evaluates three failure modes—Irrelevance, Sycophancy,
and Repetition—using complementary model-based and embedding-based metrics.

| At a glance | |
| --- | --- |
| Benchmark size | 1,700 verified instances |
| Users | 20 |
| Failure modes | 3 primary categories, 6 subcategories |
| Evaluation signal | Higher score means less over-personalization |
| Paper evaluation | 6 language models, 5 memory settings, 36 configurations |
| Public release | Benchmark data, prompts, evaluator, selected agents, MemOS adapter, and interactive demo |

### Failure modes

| Category | What it measures | Subcategories |
| --- | --- | --- |
| **Irrelevance** | Whether the response injects personal information when the query does not call for it | Fully irrelevant; baiting (deceptively relevant) |
| **Sycophancy** | Whether personalization causes excessive agreement instead of factual or value-sensitive responses | Fact-level; value-level; memory-level |
| **Repetition** | Whether semantically distinct queries receive nearly identical personalized responses | One repetition setting |

<p align="center">
  <img src="docs/figures/figure-1-taxonomy.png" alt="Illustration of the three OP-Bench failure modes: irrelevance, sycophancy, and repetition" width="720">
</p>
<p align="center"><em>Illustration of the three over-personalization categories.</em></p>

## Data construction

The benchmark is built in three stages:

1. **Initialization:** derive a structured user profile and user topics from
   long-horizon LoCoMo conversations.
2. **Task construction:** generate controlled probes for Irrelevance,
   Sycophancy, and Repetition, including the relevant subcategories.
3. **Human review:** independently review each candidate, adjudicate
   disagreements, and retain only verified instances.

<p align="center">
  <img src="docs/figures/figure-3-pipeline.png" alt="Three-stage OP-Bench construction pipeline from initialization to task construction and human review" width="100%">
</p>
<p align="center"><em>OP-Bench construction pipeline (paper Figure 3).</em></p>

The final dataset distribution is:

| Category | Subcategory | Count | Share |
| --- | --- | ---: | ---: |
| Irrelevance | Fully irrelevant | 318 | 18.7% |
| Irrelevance | Baiting | 100 | 5.9% |
| Sycophancy | Fact-level | 100 | 5.9% |
| Sycophancy | Value-level | 100 | 5.9% |
| Sycophancy | Memory-level | 200 | 11.8% |
| Repetition | — | 882 | 51.9% |
| **Total** | — | **1,700** | **100%** |

## Experimental findings

OP-Bench scores are designed so that **higher is better**: a higher score
indicates that the response avoids the corresponding over-personalization
behavior.

### Main result

The following is the GPT-4o-mini block from the paper's main results table.
The percentage in parentheses is the relative drop from the memory-free
<code>BASE</code> setting.

| Memory setting | OP-Bench average |
| --- | ---: |
| <code>BASE</code> | **83.10** |
| <code>RAG</code> | 55.96 (↓32.7%) |
| <code>Mem0</code> | 46.32 (↓44.3%) |
| <code>MemU</code> | 40.46 (↓51.3%) |
| <code>MEMOS</code> | 41.86 (↓49.6%) |

Across models and memory systems, the paper reports relative drops of
**26.2%–61.1%** compared with <code>BASE</code>.

### Cross-model comparison

The four panels below reproduce the paper's radar comparison for the
evaluated model families. The common legend is shown above the panels.

<p align="center">
  <img src="docs/figures/figure-2-legend.png" alt="Legend for BASE, RAG, Mem0, MEMOS, and MemU" width="620">
</p>
<p align="center">
  <img src="docs/figures/figure-2-gpt4o-mini.png" alt="Radar chart for GPT-4o-mini" width="47%">
  <img src="docs/figures/figure-2-gemini-2-5-flash.png" alt="Radar chart for Gemini-2.5-flash" width="47%">
</p>
<p align="center">
  <img src="docs/figures/figure-2-qwen3-235b.png" alt="Radar chart for Qwen3-235B" width="47%">
  <img src="docs/figures/figure-2-qwen3-32b.png" alt="Radar chart for Qwen3-32B" width="47%">
</p>
<p align="center"><em>OP-Bench scores across model and memory configurations (paper Figure 2).</em></p>

### Research questions

| Question | Conclusion |
| --- | --- |
| **RQ1 — Does over-personalization exist?** | Yes. All evaluated memory-augmented settings show substantial degradation relative to <code>BASE</code>; more elaborate memory mechanisms tend to suffer larger drops. |
| **RQ2 — Why does it occur?** | Memory can be retrieved aggressively and receive disproportionate attention. The average memory-to-query attention ratio exceeds 2×, which can produce memory hijacking and response collapse. |
| **RQ3 — Can it be mitigated?** | Only partially. Post-processing improves OP-Bench by up to **+20.0%**, but some configurations reduce LoCoMo personalization performance by up to **−8.5%**. No evaluated method closes the gap to the memory-free oracle. |
| **RQ4 — What is the timing cost?** | The supplementary timing view shows retrieval averaging roughly **38–44 ms**, while post-processing ranges from about **0.97–4.13 s** and model response generation from about **2.71–3.41 s**. |

### Retrieval is a major part of the problem

Even fully irrelevant queries can trigger memory retrieval. Baiting queries
may have high superficial similarity to stored memories, making retrieval
alone an insufficient safeguard.

<p align="center">
  <img src="docs/figures/figure-5-retrieval.png" alt="Bar chart comparing retrieval relevance for fully irrelevant and baiting queries across RAG, MemOS, MemU, and Mem0" width="760">
</p>
<p align="center"><em>Retrieved-memory similarity on the Irrelevance task (paper Figure 5).</em></p>

## What's included

This public repository is organized around the reproducible benchmark path:

- benchmark data and construction/evaluation prompts;
- a resumable benchmark builder;
- search, generation, scoring, and post-processing utilities;
- the in-scope <code>LDAgent</code> and <code>SimpleRAGAgent</code> implementations;
- an HTTP adapter for an external MemOS service;
- a dependency-light web demo for the benchmark story and results;
- tests for configuration, scoring, data types, and the web demo.

The repository does **not** vendor the upstream MemOS server, model weights,
private deployment scripts, unrelated experimental agents, or generated
results. Plotting scripts from the original research workspace are also
excluded; the key paper figures used here are stored under
[docs/figures/](docs/figures/).

## Repository layout

~~~text
OPBench-public/
├── configs/
│   └── evaluation.example.yaml
├── data/
│   ├── locomo10.json
│   ├── locomo10_overpersonalized.json
│   └── README.md
├── docs/
│   └── figures/                  # Selected figures from the paper
├── prompts/
│   ├── benchmark/                # Task-construction templates
│   └── evaluation/               # Scoring prompts and archived source
├── scripts/
│   ├── prepare_locomo_histories.py
│   ├── start_agent.py
│   ├── start_agents.py
│   └── ingest_memos.py
├── src/opbench/
│   ├── agents/                   # LDAgent and SimpleRAGAgent
│   ├── baselines/                # MemOS API adapter
│   ├── benchmark_builder.py      # Build or extend benchmark tasks
│   ├── evaluation.py             # Search, generation, scoring, metrics
│   ├── postprocessing.py         # Context filtering/compression hooks
│   └── config.py                 # Environment-backed configuration
├── tests/
├── web/                          # Local full-stack research explorer
├── .env.example
├── pyproject.toml
└── NOTICE.md
~~~

## Quick start

### 1. Install

Python 3.10 or newer is required.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[agents]"
~~~

Create a local environment file from the template:

~~~bash
cp .env.example .env
~~~

Fill in the provider variables locally. Credentials and endpoint URLs must
remain in <code>.env</code> or another ignored local configuration; never put
them in source files, YAML committed to Git, or shell scripts.

### 2. Launch the local web demo

The demo is self-contained and does not require model credentials:

~~~bash
python web/backend/server.py --host 127.0.0.1 --port 8765
~~~

Then open <http://127.0.0.1:8765>. The interface contains three chapters:
data construction, representative QA examples, and experimental results.
The construction chapter supports automatic playback as well as manual
previous/next navigation. See [web/README.md](web/README.md) for API routes
and implementation notes.

### Publish the demo with GitHub Pages

The demo can also be published as a static site. The frontend falls back to
the checked-in JSON payloads when the Python API is unavailable. The included
GitHub Actions workflow builds the Pages artifact and deploys it when relevant
files are pushed to <code>main</code>. Enable **GitHub Actions** under Settings
→ Pages, then visit <https://yulinlp.github.io/OP-Bench/> after the first
successful deployment.

### 3. Run the included agents

Prepare one history file per LoCoMo user:

~~~bash
python scripts/prepare_locomo_histories.py
~~~

Start all ten agents for one of the two public memory implementations:

~~~bash
python scripts/start_agents.py --agent ldagent
# or
python scripts/start_agents.py --agent simplerag
~~~

Each server exposes:

- <code>POST /v1/chat/completions</code>
- <code>GET /v1/models</code>
- <code>GET /health</code>

To start a single process, use <code>scripts/start_agent.py</code> with a
history file and port.

### 4. Evaluate OP-Bench

The checked-in task file is
<code>data/locomo10_overpersonalized.json</code>. For the in-process agents:

~~~bash
python -m opbench.cli generate --frame ldagent
python -m opbench.cli generate --frame simplerag
~~~

For MemOS, first ingest histories into an already configured external
service, then search, generate, and score:

~~~bash
python scripts/ingest_memos.py --online
python -m opbench.cli search --frame memos-api-online
python -m opbench.cli generate --frame memos-api-online --postprocess none
python -m opbench.cli score --frame memos-api-online --responses results/responses/memos-api-online_gpt-4o-mini_none.json
python -m opbench.cli metrics --frame memos-api-online --judged results/judged/memos-api-online_gpt-4o-mini.json
~~~

For all commands and options:

~~~bash
python -m opbench.cli --help
~~~

The available context methods include <code>none</code>,
<code>self_recheck</code>, <code>reminder</code>, <code>self_critic</code>,
<code>few_shot_chain_of_thought</code>, <code>comorag</code>, and
<code>marag</code>. The public implementation records filtered context rather
than hidden chain-of-thought traces. Generated artifacts and results are
written to ignored directories by default.

### Rebuild the benchmark

The builder is resumable: an existing output file is normalized and completed
instead of being silently overwritten.

~~~bash
python -m opbench.cli build-benchmark --source data/locomo10.json --output data/locomo10_overpersonalized.generated.json
~~~

The model endpoint and credentials for construction are read from the local
environment.

### Run tests

~~~bash
python -m pip install -e ".[dev]"
pytest
~~~

## Reproducibility notes

- <code>BASE</code> is the memory-free reference setting; <code>RAG</code>,
  <code>Mem0</code>, <code>MemU</code>, and <code>MEMOS</code> are
  memory-augmented settings evaluated in the paper.
- The checked-in prompts preserve the task and scoring definitions used by the
  benchmark.
- Repetition uses cosine similarity after explicit L2 normalization.
- API failures are recorded per question, retries are bounded, and result
  files are written atomically so long runs can be resumed.
- Set <code>use_both_personas: true</code> in a local configuration when both
  personas are required; the default follows the original scripts and uses the
  first persona in each conversation.

## Website

The hosted OP-Bench website is available at
<https://yulinlp.github.io/OP-Bench/>. The local interactive demo in
[web/](web/) provides the benchmark construction, QA examples, and results
views.

## License and data

See [NOTICE.md](NOTICE.md) for provenance and redistribution notes. The
repository currently does not declare a software license; add or confirm the
appropriate license before redistributing the code. LoCoMo data, model
checkpoints, and third-party services remain subject to their original terms.
No model weights or credentials are bundled in this release.

## Citation

If OP-Bench is useful in your research, please cite:

~~~bibtex
@misc{hu2026opbench,
  title         = {OP-Bench: Benchmarking Over-Personalization for Memory-Augmented Personalized Conversational Agents},
  author        = {Hu, Yulin and Long, Zimo and Guo, Jiahe and Sui, Xingyu and Fu, Xing and Zhao, Weixiang and Zhao, Yanyan and Qin, Bing},
  year          = {2026},
  eprint        = {2601.13722},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.13722}
}
~~~
