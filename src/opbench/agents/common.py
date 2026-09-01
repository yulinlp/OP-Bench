"""Common history loading and OpenAI-compatible generation helpers for agents."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import EndpointConfig, create_openai_client


@dataclass
class AgentSettings:
    history_path: str | None
    data_format: str = "conversation"
    model: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.0
    user_name: str = "user"
    agent_name: str = "assistant"
    mode: str = "uni"
    max_memories: int = 5
    no_memory: bool = False
    no_personality: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.3
    persist_path: str | None = None
    spacy_model: str = "en_core_web_sm"
    max_user_traits: int = 5
    max_agent_traits: int = 5


def _sorted_session_keys(conversation: dict[str, Any]) -> list[str]:
    keys = [
        key
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))


def _locomo_sessions(
    item: dict[str, Any], user_name: str, agent_name: str
) -> list[list[dict[str, str]]]:
    conversation = item.get("conversation", {})
    if not isinstance(conversation, dict):
        return []
    speaker_a = conversation.get("speaker_a", user_name)
    speaker_b = conversation.get("speaker_b", agent_name)
    sessions: list[list[dict[str, str]]] = []
    for key in _sorted_session_keys(conversation):
        turns: list[dict[str, str]] = []
        for turn in conversation.get(key, []):
            if not isinstance(turn, dict) or not turn.get("text"):
                continue
            speaker = turn.get("speaker")
            role = "user" if speaker == speaker_a else "assistant" if speaker == speaker_b else "user"
            turns.append({"role": role, "content": str(turn["text"])})
        if turns:
            sessions.append(turns)
    return sessions


def as_conversations(
    history: Any,
    *,
    data_format: str,
    user_name: str,
    agent_name: str,
) -> list[list[dict[str, str]]]:
    """Normalize native LoCoMo and simple conversation JSON into sessions."""

    if not isinstance(history, list):
        return []
    if data_format.lower() == "locomo":
        sessions: list[list[dict[str, str]]] = []
        for item in history:
            if isinstance(item, dict):
                sessions.extend(_locomo_sessions(item, user_name, agent_name))
        return sessions

    conversations: list[list[dict[str, str]]] = []
    for value in history:
        # A list of turns is one conversation. A list of conversations is also
        # accepted when each child is a list.
        if not isinstance(value, list):
            continue
        turns: list[dict[str, str]] = []
        for turn in value:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content", turn.get("text", ""))
            if not content:
                continue
            role = turn.get("role")
            if role not in {"user", "assistant", "system"}:
                role = "user" if turn.get("speaker") == user_name else "assistant"
            turns.append({"role": role, "content": str(content)})
        if turns:
            conversations.append(turns)
    return conversations


def load_conversations(settings: AgentSettings) -> list[list[dict[str, str]]]:
    if not settings.history_path:
        return []
    path = Path(settings.history_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return as_conversations(
        raw,
        data_format=settings.data_format,
        user_name=settings.user_name,
        agent_name=settings.agent_name,
    )


class AgentBase:
    def __init__(self, settings: AgentSettings, endpoint: EndpointConfig):
        self.settings = settings
        self.endpoint = endpoint
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = create_openai_client(endpoint)
        self.is_initialized = False

    def complete(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=model or self.settings.model or self.endpoint.model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            return ""
        return getattr(choices[0].message, "content", "") or ""

    @staticmethod
    def last_user_message(messages: list[Any]) -> str:
        for message in reversed(messages):
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role == "user" and content:
                return str(content)
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

