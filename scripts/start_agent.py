#!/usr/bin/env python3
"""Start one OpenAI-compatible LDAgent or SimpleRAGAgent server."""

from __future__ import annotations

import argparse
from dataclasses import replace

import uvicorn

from opbench.agents.common import AgentSettings
from opbench.agents.server import build_agent, create_app
from opbench.config import load_config
from opbench.env import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["ldagent", "simplerag"], required=True)
    parser.add_argument("--history", required=True, help="JSON history file for this agent")
    parser.add_argument("--data-format", choices=["conversation", "locomo"], default="conversation")
    parser.add_argument("--config", default="configs/evaluation.example.yaml")
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--user-name", default="user")
    parser.add_argument("--agent-name", default="assistant")
    parser.add_argument("--mode", choices=["uni", "chat", "qa"], default="uni")
    parser.add_argument("--max-memories", type=int, default=5)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-personality", action="store_true")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument("--persist-path")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--max-user-traits", type=int, default=5)
    parser.add_argument("--max-agent-traits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    endpoint = replace(config.agent, model=args.model or config.agent.model)
    settings = AgentSettings(
        history_path=args.history,
        data_format=args.data_format,
        model=args.model or config.agent.model,
        max_tokens=endpoint.max_tokens,
        temperature=endpoint.temperature,
        user_name=args.user_name,
        agent_name=args.agent_name,
        mode=args.mode,
        max_memories=max(1, args.max_memories),
        no_memory=args.no_memory,
        no_personality=args.no_personality,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        persist_path=args.persist_path,
        spacy_model=args.spacy_model,
        max_user_traits=max(1, args.max_user_traits),
        max_agent_traits=max(1, args.max_agent_traits),
    )
    agent = build_agent(args.agent, settings, endpoint)
    app = create_app(agent, settings.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
