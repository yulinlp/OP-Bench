"""OpenAI-compatible HTTP server for the two public OPBench agents."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .common import AgentSettings
from .ld_agent import LDAgent
from .simple_rag import SimpleRAGAgent


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[Message] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


def _response(content: str, model: str, request_messages: list[Message]) -> dict[str, Any]:
    prompt_tokens = sum(len(message.content.split()) for message in request_messages)
    completion_tokens = len(content.split())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream(content: str, model: str):
    async def iterator():
        identifier = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        words = content.split()
        for index, word in enumerate(words):
            chunk = {
                "id": identifier,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word + (" " if index + 1 < len(words) else "")}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return iterator()


def create_app(agent: Any, model: str) -> FastAPI:
    app = FastAPI(title="OPBench memory agent")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages cannot be empty")
        try:
            content = await asyncio.to_thread(
                agent.generate_response,
                request.messages,
                request.model or model,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if request.stream:
            return StreamingResponse(_stream(content, request.model or model), media_type="text/event-stream")
        return _response(content, request.model or model, request.messages)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "opbench"}]}

    @app.get("/health")
    async def health():
        healthy = bool(getattr(agent, "is_initialized", False))
        return {"status": "healthy" if healthy else "unhealthy", "model": model}

    return app


def build_agent(kind: str, settings: AgentSettings, endpoint):
    normalized = kind.lower()
    if normalized in {"ldagent", "ld"}:
        return LDAgent(settings, endpoint)
    if normalized in {"simplerag", "rag"}:
        return SimpleRAGAgent(settings, endpoint)
    raise ValueError(f"Unknown agent kind: {kind}")

