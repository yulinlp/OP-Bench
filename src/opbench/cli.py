"""Command line entry point for the public OPBench repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_builder import BenchmarkBuilder
from .config import load_config
from .env import load_dotenv
from .evaluation import run_generate, run_metrics, run_score, run_search
from .postprocessing import METHODS

ROOT_DEFAULT = Path(".")
BENCHMARK_DEFAULT = ROOT_DEFAULT / "data" / "locomo10_overpersonalized.json"
CONFIG_DEFAULT = ROOT_DEFAULT / "configs" / "evaluation.example.yaml"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--benchmark", default=str(BENCHMARK_DEFAULT))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--workers", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OPBench public evaluation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-benchmark", help="build the task file from LoCoMo")
    build.add_argument("--source", default="data/locomo10.json")
    build.add_argument("--output", default=str(BENCHMARK_DEFAULT))
    build.add_argument("--prompt-dir", default="prompts/benchmark")
    build.add_argument("--config", default=str(CONFIG_DEFAULT))
    build.add_argument("--workers", type=int)
    build.add_argument("--overwrite", action="store_true")

    search = subparsers.add_parser("search", help="retrieve memories from MemOS")
    _common(search)
    search.add_argument("--frame", choices=["memos-api", "memos-api-online"], default="memos-api-online")
    search.add_argument("--version", default="default")

    generate = subparsers.add_parser("generate", help="generate responses")
    _common(generate)
    generate.add_argument("--frame", choices=["ldagent", "simplerag", "agent", "memos-api", "memos-api-online"], required=True)
    generate.add_argument("--version", default="default")
    generate.add_argument("--search-path")
    generate.add_argument("--postprocess", choices=list(METHODS), default="none")
    generate.add_argument("--no-memory", action="store_true")

    score = subparsers.add_parser("score", help="score generated responses")
    _common(score)
    score.add_argument("--responses", required=True)
    score.add_argument("--frame", default="agent")

    metrics = subparsers.add_parser("metrics", help="aggregate a judged result file")
    metrics.add_argument("--judged", required=True)
    metrics.add_argument("--frame", default="agent")
    metrics.add_argument("--output-dir", default="results")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if args.command == "build-benchmark":
        config = load_config(args.config)
        builder = BenchmarkBuilder(
            args.source,
            args.output,
            args.prompt_dir,
            config.agent,
            workers=args.workers or config.workers,
        )
        builder.build_all(overwrite=args.overwrite)
        print(f"Benchmark written to {args.output}")
        return

    config = load_config(args.config)
    if args.workers:
        config = config.__class__(
            agent=config.agent,
            scorer=config.scorer,
            embedding_model=config.embedding_model,
            top_k=config.top_k,
            workers=max(1, args.workers),
            task_types=config.task_types,
            use_both_personas=config.use_both_personas,
            agent_ports=config.agent_ports,
            agent_host_env=config.agent_host_env,
        )

    if args.command == "search":
        path = run_search(
            args.benchmark,
            args.output_dir,
            config,
            frame=args.frame,
            version=args.version,
            online=args.frame.endswith("online"),
        )
    elif args.command == "generate":
        path = run_generate(
            args.benchmark,
            args.output_dir,
            config,
            frame=args.frame,
            version=args.version,
            search_path=args.search_path,
            use_memory=not args.no_memory,
            postprocess=args.postprocess,
        )
    elif args.command == "score":
        path = run_score(args.benchmark, args.responses, args.output_dir, config, frame=args.frame)
    elif args.command == "metrics":
        path = run_metrics(args.judged, args.output_dir, frame=args.frame)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    print(path)


if __name__ == "__main__":
    main()
