#!/usr/bin/env python3
"""Launch one LDAgent or SimpleRAGAgent process for each LoCoMo user."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prepare_locomo_histories import prepare

from opbench.config import load_config
from opbench.env import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["ldagent", "simplerag"], required=True)
    parser.add_argument("--config", default="configs/evaluation.example.yaml")
    parser.add_argument("--source", default="data/locomo10.json")
    parser.add_argument("--history-dir", default="artifacts/agent_histories")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-personality", action="store_true")
    parser.add_argument("--mode", choices=["uni", "chat", "qa"], default="uni")
    args = parser.parse_args()

    config = load_config(args.config)
    history_dir = Path(args.history_dir)
    if not (history_dir / "manifest.json").exists():
        prepare(Path(args.source), history_dir)

    script = Path(__file__).with_name("start_agent.py")
    processes: list[subprocess.Popen] = []
    try:
        for person, port in config.agent_ports.items():
            history = history_dir / f"{person}.json"
            if not history.exists():
                print(f"Skipping {person}: missing {history}", file=sys.stderr)
                continue
            command = [
                sys.executable,
                str(script),
                "--agent",
                args.agent,
                "--history",
                str(history),
                "--config",
                args.config,
                "--host",
                args.host,
                "--port",
                str(port),
                "--mode",
                args.mode,
            ]
            if args.no_personality:
                command.append("--no-personality")
            if args.agent == "simplerag":
                persist = Path("artifacts/rag") / f"{person}.json"
                command.extend(["--persist-path", str(persist)])
            print(f"Starting {person} on port {port}")
            processes.append(subprocess.Popen(command, env=os.environ.copy()))
        if not processes:
            raise RuntimeError("No agent processes were started; check the port map and histories.")
        print("Agents are running. Press Ctrl-C to stop them.")
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
