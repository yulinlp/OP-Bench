# OPBench

OPBench evaluates over-personalization and response timing in memory-augmented
conversation agents. This directory is the cleaned public release prepared from
the paper implementation. It keeps the benchmark data and prompts, the
evaluation pipeline, the MemOS HTTP adapter, and the two in-scope memory
methods: `LDAgent` and `SimpleRAGAgent`.

The original directories are not modified by this cleanup. Generated answers,
judge outputs, vector stores, model caches, and credentials are intentionally
excluded from this repository.

## Repository layout

```text
OPBench-public/
├── data/                         # LoCoMo input and generated OPBench tasks
├── prompts/benchmark/            # Benchmark-construction prompt templates
├── prompts/evaluation/           # Archived prompt source for traceability
├── src/opbench/
│   ├── agents/                   # LDAgent and SimpleRAGAgent
│   ├── baselines/                # MemOS adapter entry point
│   ├── benchmark_builder.py      # Rebuild/extend the task file
│   ├── evaluation.py             # search, generation, scoring, metrics
│   ├── scoring.py                # judge and embedding metrics
│   └── config.py                 # secret-free configuration
├── scripts/                      # history preparation, servers, ingestion
├── configs/evaluation.example.yaml
├── .env.example
└── pyproject.toml
```

Plotting scripts and unrelated agents/experiments from the original research
workspace are not part of the public release. MemOS itself is used as an
external service through `MemosAPIClient`; the full upstream MemOS server is
not vendored here.

## Installation and credentials

Run the following commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[agents]"
cp .env.example .env
```

Fill the local `.env` with the key and, when needed, endpoint URL for the model
provider. The repository contains only variable names and empty values. Never
commit `.env` or paste credentials into YAML, shell scripts, or Python files.

For LDAgent, the same model endpoint is used for memory summarization,
personality extraction, and response generation. SimpleRAGAgent additionally
downloads the embedding model named by `--embedding-model` and automatically
uses CUDA when available, otherwise CPU.

## Run the included agents

First create one history file per LoCoMo user:

```bash
python scripts/prepare_locomo_histories.py
```

The generated files are placed under `artifacts/`, which is ignored by Git.
Start all ten LDAgent or SimpleRAGAgent servers with the port map in the example
configuration:

```bash
python scripts/start_agents.py --agent ldagent
# or
python scripts/start_agents.py --agent simplerag
```

Each server implements:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

To start just one process, use `scripts/start_agent.py` and provide a history
file and port. No API key is passed on the command line.

## Run OPBench

The default task file is `data/locomo10_overpersonalized.json`. By default,
evaluation uses the first persona in each LoCoMo conversation, matching the
original scripts; set `use_both_personas: true` in a local config when both
personas are required.

For the in-process agent servers:

```bash
python -m opbench.cli generate --frame ldagent
python -m opbench.cli generate --frame simplerag
```

For MemOS, ingest the histories into the configured service, retrieve the top-k
memories, then generate and score responses:

```bash
python scripts/ingest_memos.py --online
python -m opbench.cli search --frame memos-api-online
python -m opbench.cli generate --frame memos-api-online --postprocess none
python -m opbench.cli score --frame memos-api-online \
  --responses results/responses/memos-api-online_gpt-4o-mini_none.json
python -m opbench.cli metrics --frame memos-api-online \
  --judged results/judged/memos-api-online_gpt-4o-mini.json
```

Use `scripts/run_opbench.sh` as a shorter wrapper around the same CLI. All
output paths can be changed with `--output-dir`; the default `results/` is
ignored by Git.

The optional context methods are `none`, `self_recheck`, `reminder`,
`self_critic`, `few_shot_chain_of_thought`, `comorag`, and `marag`. The public
implementation records the filtered context rather than hidden chain-of-thought
traces.

## Rebuild the benchmark

The checked-in task file is the benchmark used for the paper experiments. To
rebuild it from `locomo10.json` or extend it with a new model:

```bash
python -m opbench.cli build-benchmark \
  --source data/locomo10.json \
  --output data/locomo10_overpersonalized.generated.json
```

The builder is resumable: an existing output file is normalized and completed
instead of being silently overwritten. Its model endpoint and credentials come
from the configured environment variables.

## Reproducibility notes

- The paper evaluates BASE/RAG and several external memory systems. This public
  tree contains the reusable OPBench evaluator, the MemOS API integration, and
  the requested LDAgent/SimpleRAGAgent implementations. Other third-party SDKs
  and private deployment scripts are intentionally not copied.
- The checked-in prompts preserve the paper's task and scoring definitions.
- Diversity uses cosine similarity after explicit L2 normalization. This fixes
  the original scorer's implicit assumption that provider embeddings were already
  normalized.
- API failures are recorded per question and retries are bounded. Results are
  written atomically so a long run can be resumed safely.
