"""SimpleRAGAgent baseline: one dense embedding per conversation turn."""

from __future__ import annotations

from typing import Any

from .common import AgentBase, AgentSettings, load_conversations
from .composer import SimpleRAGComposer
from .rag_memory import SimpleRAGMemory


class SimpleRAGAgent(AgentBase):
    def __init__(self, settings: AgentSettings, endpoint):
        super().__init__(settings, endpoint)
        self.memory = SimpleRAGMemory(settings, self.logger)
        conversations = load_conversations(settings)
        self.memory.load_memory(conversations)
        self.composer = SimpleRAGComposer(settings.user_name)
        self.is_initialized = True

    def generate_response(self, messages: list[Any], model: str | None = None) -> str:
        question = self.last_user_message(messages)
        memories = [] if self.settings.no_memory else self.memory.retrieve(
            question,
            limit=self.settings.max_memories,
            threshold=self.settings.similarity_threshold,
        )
        composed = self.composer.compose(question, memories, self.settings.mode)
        return self.complete(composed, model=model)

