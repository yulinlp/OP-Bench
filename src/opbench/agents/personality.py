"""Configurable personality extraction used by the LD-Agent baseline."""

from __future__ import annotations

from typing import Any


class Personality:
    def __init__(self, client: Any, settings, logger=None):
        self.client = client
        self.settings = settings
        self.logger = logger
        self.user_traits: list[str] = []
        self.agent_traits: list[str] = []

    def _extract(self, sentence: str) -> str:
        prompt = (
            "Extract a concise personal trait from the sentence below. "
            "If it contains no stable personal trait, return NO_TRAIT. "
            "Use no more than 20 words and do not add facts.\n\n"
            f"Sentence: {sentence}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            return (getattr(response.choices[0].message, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001 - a failed optional trait call is non-fatal
            if self.logger:
                self.logger.warning("Could not extract personality trait: %s", exc)
            return ""

    @staticmethod
    def _append(values: list[str], trait: str, limit: int) -> None:
        if not trait or "NO_TRAIT" in trait.upper() or len(trait) <= 3:
            return
        if trait not in values:
            values.append(trait)
        del values[:-limit]

    def update(self, conversations: list[list[dict[str, str]]]) -> None:
        for conversation in conversations:
            for turn in conversation:
                content = turn.get("content", "")
                if not content:
                    continue
                trait = self._extract(content)
                if turn.get("role") == "user":
                    self._append(self.user_traits, trait, self.settings.max_user_traits)
                elif turn.get("role") == "assistant":
                    self._append(self.agent_traits, trait, self.settings.max_agent_traits)

    def get_traits(self) -> dict[str, str]:
        return {
            "user_traits": "\n".join(self.user_traits),
            "agent_traits": "\n".join(self.agent_traits),
        }
