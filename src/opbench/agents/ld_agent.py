"""LD-Agent baseline: summarized memories plus keyword-overlap retrieval."""

from __future__ import annotations

import time
from typing import Any

from .common import AgentBase, AgentSettings, load_conversations
from .composer import LDAgentComposer
from .ld_memory import LDAgentMemory
from .personality import Personality


class LDAgent(AgentBase):
    def __init__(self, settings: AgentSettings, endpoint):
        super().__init__(settings, endpoint)
        self.memory = LDAgentMemory(self.client, settings, self.logger)
        self.personality = Personality(self.client, settings, self.logger)
        conversations = load_conversations(settings)
        self.memory.load_memory(conversations, data_format=settings.data_format)
        if not settings.no_personality:
            self.personality.update(conversations)
        self.composer = LDAgentComposer(settings.user_name, settings.agent_name)
        self.is_initialized = True

    def generate_response(self, messages: list[Any], model: str | None = None) -> str:
        question = self.last_user_message(messages)
        now = time.time() + self.memory.current_time_pass
        retrieved = [] if self.settings.no_memory else self.memory.retrieve(
            question, current_time=now, limit=self.settings.max_memories
        )
        traits = {"user_traits": "", "agent_traits": ""}
        if not self.settings.no_personality:
            traits = self.personality.get_traits()
        composed = self.composer.compose(
            question,
            retrieved,
            traits["user_traits"],
            traits["agent_traits"],
        )
        return self.complete(composed, model=model)
