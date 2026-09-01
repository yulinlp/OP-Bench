"""Prompt composition for LDAgent and SimpleRAGAgent."""

from __future__ import annotations

from typing import Any

from ..prompts import ANSWER_SYSTEM_PROMPT


def _memory_block(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No relevant memories."
    return "\n".join(
        str(memory.get("summary") or memory.get("dialog") or "")
        for memory in memories
        if memory.get("summary") or memory.get("dialog")
    ) or "No relevant memories."


class LDAgentComposer:
    def __init__(self, user_name: str, agent_name: str):
        self.user_name = user_name
        self.agent_name = agent_name

    def compose(
        self,
        question: str,
        memories: list[dict[str, Any]],
        user_traits: str,
        agent_traits: str,
    ) -> list[dict[str, str]]:
        memory = _memory_block(memories)
        user_prompt = (
            "Reply in a natural, spoken tone. When relevant, appropriately incorporate "
            "the user's memory and personality information to make the response "
            "personalized and engaging.\n\n"
            f"Memory:\n{memory}\n\n"
            f"User's personality:\n{user_traits or 'No user traits.'}\n\n"
            f"User's Latest Input:\n{question}"
        )
        return [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]


class SimpleRAGComposer:
    def __init__(self, user_name: str = "user"):
        self.user_name = user_name

    def compose(self, question: str, memories: list[dict[str, Any]], mode: str) -> list[dict[str, str]]:
        memory = _memory_block(memories)
        if mode == "qa":
            user_prompt = (
                "Based on the context below, answer the question with a short phrase. "
                "Use exact words from the context whenever possible.\n\n"
                f"{memory}\n\nQuestion: {question}\nShort answer:"
            )
            return [{"role": "user", "content": user_prompt}]
        user_prompt = (
            "Reply in a natural, spoken tone. When relevant, appropriately incorporate "
            "the user's memory and personality information to make the response "
            "personalized and engaging.\n\n"
            f"Memory:\n{memory}\n\nUser's Latest Input:\n{question}"
        )
        return [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

