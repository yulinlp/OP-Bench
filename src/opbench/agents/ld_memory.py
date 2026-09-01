"""Lightweight implementation of the LD-Agent memory used in OPBench.

The research version used Chroma plus a spaCy noun-overlap stage.  The public
version keeps that retrieval idea but stores records in a small in-process
index, making the data flow easier to inspect and avoiding machine-specific
database/model paths.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MemoryRecord:
    idx: int
    dialog: str
    summary: str
    topics: list[str]
    timestamp: float

    def as_result(self, score: float, overlap: int) -> dict[str, Any]:
        return {
            "idx": self.idx,
            "dialog": self.dialog,
            "summary": self.summary,
            "topics": ",".join(self.topics),
            "time": self.timestamp,
            "overall_score": score,
            "overlap_count": overlap,
        }


class LDAgentMemory:
    def __init__(self, client: Any, settings, logger=None):
        self.client = client
        self.settings = settings
        self.logger = logger
        self.records: list[MemoryRecord] = []
        self._embeddings: list[np.ndarray] = []
        self.short_term: list[dict[str, Any]] = []
        self.current_time_pass = 0.0
        self._nlp = self._load_nlp()
        self._embedder = self._load_embedder()

    def _load_nlp(self):
        try:
            import spacy

            return spacy.load(self.settings.spacy_model)
        except Exception as exc:  # noqa: BLE001 - optional spaCy/model failures use a fallback
            if self.logger:
                self.logger.info("spaCy model unavailable; using the keyword fallback (%s)", exc)
            return None

    def _load_embedder(self):
        """Load the same sentence embedding family used by the original Chroma index."""

        try:
            from sentence_transformers import SentenceTransformer

            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass
            return SentenceTransformer(self.settings.embedding_model, device=device)
        except Exception as exc:  # noqa: BLE001 - lexical retrieval remains available
            if self.logger:
                self.logger.info("Embedding model unavailable; using keyword candidates (%s)", exc)
            return None

    def _keywords(self, text: str) -> list[str]:
        if self._nlp is not None:
            doc = self._nlp(text)
            words = [token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}]
        else:
            words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
        stopwords = {
            "the", "and", "that", "this", "with", "from", "your", "about", "have",
            "what", "when", "where", "which", "would", "could", "there", "their",
            "they", "them", "were", "been", "into", "also", "just", "like", "user",
            "assistant", "conversation",
        }
        return sorted({word for word in words if word and word not in stopwords})

    def _summary(self, dialog: str, turn_count: int) -> str:
        prompt = (
            f"You are good at extracting events and summarizing a conversation between "
            f"{self.settings.user_name} and {self.settings.agent_name}.\n\n"
            f"Conversation:\n{dialog}\n\n"
            "Summarize the main points in brief English sentences within 20 words.\n"
            "SUMMARY:"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.settings.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            text = getattr(response.choices[0].message, "content", "") or ""
            return text.strip() or f"Summary of {turn_count} conversation turns"
        except Exception as exc:  # noqa: BLE001 - provider failures use a deterministic fallback
            if self.logger:
                self.logger.warning("Could not summarize memory: %s", exc)
            return f"Summary of {turn_count} conversation turns"

    def _append_record(self, record: MemoryRecord) -> None:
        self.records.append(record)
        if self._embedder is not None:
            try:
                embedding = self._embedder.encode(
                    [record.summary], normalize_embeddings=True, convert_to_numpy=True
                )[0]
                self._embeddings.append(np.asarray(embedding, dtype=np.float32))
            except Exception as exc:  # noqa: BLE001 - retrieval can fall back to keywords
                if self.logger:
                    self.logger.warning("Could not embed LD-Agent memory: %s", exc)
                self._embedder = None
                self._embeddings.clear()

    def load_memory(self, conversations: list[list[dict[str, str]]], *, data_format: str) -> None:
        if self.records:
            return
        interval = 7 * 24 * 60 * 60 if data_format.lower() == "locomo" else 2 * 60 * 60
        now = time.time()
        total = len(conversations)
        for index, conversation in enumerate(conversations):
            lines = []
            for turn in conversation:
                role = self.settings.user_name if turn.get("role") == "user" else self.settings.agent_name
                lines.append(f"{role}: {turn.get('content', '')}")
            dialog = "\n".join(lines).strip()
            if not dialog:
                continue
            record = MemoryRecord(
                idx=len(self.records),
                dialog=dialog,
                summary=self._summary(dialog, len(conversation)),
                topics=self._keywords(dialog),
                timestamp=now - (total - index) * interval,
            )
            self._append_record(record)
        if self.logger:
            self.logger.info("Loaded %d LD-Agent memory records", len(self.records))

    def add_memory(self, message: dict[str, Any]) -> None:
        content = str(message.get("content", "")).strip()
        if not content:
            return
        self._append_record(
            MemoryRecord(
                idx=len(self.records),
                dialog=content,
                summary=content,
                topics=self._keywords(content),
                timestamp=float(message.get("timestamp", time.time())),
            )
        )

    def retrieve(self, query: str, *, current_time: float | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        now = current_time or time.time()
        limit = limit or self.settings.max_memories
        query_topics = set(self._keywords(query))
        if not query_topics:
            return []
        candidate_indices = range(len(self.records))
        dense_scores: dict[int, float] = {}
        if self._embedder is not None and self._embeddings:
            try:
                query_embedding = self._embedder.encode(
                    [query], normalize_embeddings=True, convert_to_numpy=True
                )[0]
                matrix = np.asarray(self._embeddings, dtype=np.float32)
                similarities = matrix @ np.asarray(query_embedding, dtype=np.float32)
                candidate_count = min(len(self.records), max(10, limit))
                candidate_indices = np.argsort(-similarities)[:candidate_count]
                dense_scores = {int(index): float(similarities[index]) for index in candidate_indices}
            except Exception as exc:  # noqa: BLE001 - lexical retrieval remains available
                if self.logger:
                    self.logger.warning("Could not embed query: %s", exc)
        scored: list[tuple[float, int, MemoryRecord, int, float]] = []
        for record_index in candidate_indices:
            record = self.records[int(record_index)]
            overlap = len(query_topics.intersection(record.topics))
            if overlap == 0:
                continue
            denom = len(query_topics) + len(record.topics)
            overlap_score = 2 * overlap / denom if denom else 0.0
            time_gap = max(0.0, now - record.timestamp)
            score = math.exp(-1e-7 * time_gap) * overlap_score
            scored.append((score, record.idx, record, overlap, dense_scores.get(int(record_index), 0.0)))
        scored.sort(key=lambda value: (-value[0], value[1]))
        results = []
        for score, _, record, overlap, dense_score in scored[:limit]:
            item = record.as_result(score, overlap)
            item["distance"] = 1.0 - dense_score if dense_scores else 0.0
            results.append(item)
        return results

    def retrieve_memories(self, query: str, *, current_time: float | None = None) -> dict[str, Any]:
        now = current_time or time.time()
        self.short_term.append({"idx": len(self.short_term), "time": now, "dialog": f"{self.settings.user_name}: {query}"})
        recent = self.short_term[-self.settings.max_memories :]
        return {"context_memories": recent, "relevant_memories": self.retrieve(query, current_time=now)}

    def reset(self) -> None:
        self.records.clear()
        self._embeddings.clear()
        self.short_term.clear()
