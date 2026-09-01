#!/usr/bin/env python3
"""Split native LoCoMo conversations into one history file per speaker A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def session_keys(conversation: dict) -> list[str]:
    keys = [key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))


def prepare(source: Path, destination: Path) -> list[dict[str, str]]:
    data = json.loads(source.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in data:
        conversation = item.get("conversation", {})
        if not isinstance(conversation, dict):
            continue
        user_name = str(conversation.get("speaker_a", "User"))
        assistant_name = str(conversation.get("speaker_b", "Assistant"))
        sessions = []
        for key in session_keys(conversation):
            turns = []
            for turn in conversation.get(key, []):
                if not isinstance(turn, dict) or not turn.get("text"):
                    continue
                speaker = turn.get("speaker")
                role = "user" if speaker == user_name else "assistant" if speaker == assistant_name else "user"
                turns.append({"role": role, "content": str(turn["text"])})
            if turns:
                sessions.append(turns)
        output = destination / f"{user_name}.json"
        output.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"person": user_name, "path": str(output), "sessions": str(len(sessions))})
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/locomo10.json")
    parser.add_argument("--destination", default="artifacts/agent_histories")
    args = parser.parse_args()
    manifest = prepare(Path(args.source), Path(args.destination))
    print(f"Prepared {len(manifest)} agent histories in {args.destination}")


if __name__ == "__main__":
    main()

