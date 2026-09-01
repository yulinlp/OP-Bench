#!/usr/bin/env python3
"""Ingest the LoCoMo histories into a MemOS API before running ``opbench search``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opbench.clients import MemosAPIClient
from opbench.env import load_dotenv
from opbench.types import PERSON_TO_CONV_IDX


def session_keys(conversation: dict) -> list[str]:
    keys = [key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/locomo10.json")
    parser.add_argument("--version", default="default")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    client = MemosAPIClient(online=args.online)
    for index, item in enumerate(data):
        conversation = item.get("conversation", {})
        if not isinstance(conversation, dict):
            continue
        user_name = str(conversation.get("speaker_a", "User"))
        conv_index = PERSON_TO_CONV_IDX.get(user_name, index)
        user_id = f"locomo_exp_user_{conv_index}_speaker_a_{args.version}"
        for key in session_keys(conversation):
            date = conversation.get(f"{key}_date_time", "")
            messages = []
            for turn in conversation.get(key, []):
                if not isinstance(turn, dict) or not turn.get("text"):
                    continue
                messages.append(
                    {
                        "role": "user" if turn.get("speaker") == user_name else "assistant",
                        "content": str(turn["text"]),
                        "chat_time": date,
                    }
                )
            if messages:
                client.add(messages, user_id, f"{user_id}_{key}")
        print(f"Ingested {user_name}")


if __name__ == "__main__":
    main()
