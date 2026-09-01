"""Dense vector memory for the SimpleRAGAgent baseline."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Document:
    content: str
    doc_id: str
    metadata: dict[str, Any]
    timestamp: str
    embedding: list[float] | None = None

    @classmethod
    def create(cls, content: str, metadata: dict[str, Any] | None = None) -> Document:
        return cls(
            content=content,
            doc_id=str(uuid.uuid4()),
            metadata=metadata or {},
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


class SimpleRAGMemory:
    def __init__(self, settings, logger=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SimpleRAGAgent requires the optional 'agents' dependencies; "
                "install with `pip install -e '.[agents]'`."
            ) from exc
        self.settings = settings
        self.logger = logger
        self.documents: list[Document] = []
        device = self._device()
        self.embedding_model = SentenceTransformer(settings.embedding_model, device=device)
        if logger:
            logger.info("Loaded embedding model %s on %s", settings.embedding_model, device)
        if settings.persist_path and Path(settings.persist_path).exists():
            self._load(Path(settings.persist_path))

    @staticmethod
    def _device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _encode(self, texts: list[str]) -> np.ndarray:
        values = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        matrix = np.asarray(values, dtype=np.float32)
        return matrix if matrix.ndim == 2 else matrix[None, :]

    def _add(self, document: Document) -> None:
        document.embedding = self._encode([document.content])[0].tolist()
        self.documents.append(document)

    def load_memory(self, conversations: list[list[dict[str, str]]]) -> None:
        if self.documents:
            return
        for conversation_index, conversation in enumerate(conversations):
            for turn_index, turn in enumerate(conversation):
                content = turn.get("content", "").strip()
                if not content:
                    continue
                role = "User" if turn.get("role") == "user" else "Agent"
                self._add(
                    Document.create(
                        f"{role}: {content}",
                        {
                            "type": "conversation",
                            "conversation_id": conversation_index,
                            "turn_index": turn_index,
                            "role": turn.get("role", ""),
                        },
                    )
                )
        self.persist()
        if self.logger:
            self.logger.info("Loaded %d SimpleRAG documents", len(self.documents))

    def add_memory(self, message: dict[str, Any]) -> None:
        content = str(message.get("content", "")).strip()
        if not content:
            return
        self._add(Document.create(content, {"type": "memory", "role": message.get("role", "unknown")}))
        self.persist()

    def retrieve(self, query: str, limit: int | None = None, threshold: float | None = None) -> list[dict[str, Any]]:
        if not self.documents:
            return []
        limit = limit or self.settings.max_memories
        threshold = self.settings.similarity_threshold if threshold is None else threshold
        query_embedding = self._encode([query])[0]
        matrix = np.asarray([doc.embedding for doc in self.documents], dtype=np.float32)
        scores = matrix @ query_embedding
        indices = np.argsort(-scores)[:limit]
        result = []
        for index in indices:
            score = float(scores[index])
            if score < threshold:
                continue
            doc = self.documents[int(index)]
            result.append(
                {
                    "idx": int(index),
                    "dialog": doc.content,
                    "summary": doc.content,
                    "similarity": score,
                    "time": doc.timestamp,
                    "metadata": doc.metadata,
                }
            )
        return result

    def reset(self) -> None:
        self.documents.clear()
        self.persist()

    def persist(self) -> None:
        if not self.settings.persist_path:
            return
        path = Path(self.settings.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(document) for document in self.documents], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.documents = [Document(**item) for item in raw if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            if self.logger:
                self.logger.warning("Could not load RAG cache %s: %s", path, exc)
            self.documents = []

