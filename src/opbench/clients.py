"""External service adapters used by OPBench.

Only environment-provided credentials and URLs are accepted.  This module is
deliberately small: the benchmark should not vendor third-party memory SDKs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import EndpointConfig, create_openai_client


@dataclass
class CallResult:
    text: str
    duration_ms: float
    error: str | None = None
    raw: dict[str, Any] | None = None


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message else ""
    return content or ""


def chat_completion(
    client: Any,
    endpoint: EndpointConfig,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> CallResult:
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model or endpoint.model,
            messages=messages,
            temperature=endpoint.temperature if temperature is None else temperature,
            max_tokens=max_tokens or endpoint.max_tokens,
        )
        return CallResult(
            text=_response_text(response),
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as exc:  # noqa: BLE001 - normalize arbitrary SDK failures
        return CallResult(
            text="",
            duration_ms=(time.perf_counter() - start) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )


class AgentRouter:
    """Call one OpenAI-compatible agent endpoint per person.

    A single URL can be supplied through `OPBENCH_AGENT_BASE_URL`.  For the
    multi-process LDAgent/SimpleRAG setup, set `OPBENCH_AGENT_HOST` and provide
    a person-to-port map in the YAML config.
    """

    def __init__(self, config):
        self.config = config
        configured_host = os.getenv(config.agent_host_env, "").rstrip("/")
        # The multi-process setup is local by default; a remote host can be
        # supplied through the environment without changing committed files.
        self.host = configured_host or ("http://127.0.0.1" if config.agent_ports else "")
        self.base_url = config.agent.base_url.rstrip("/")
        self.api_key = config.agent.api_key
        self.timeout = config.agent.timeout

    def _url(self, person_name: str) -> str:
        if self.host and person_name in self.config.agent_ports:
            return f"{self.host}:{self.config.agent_ports[person_name]}/v1/chat/completions"
        if not self.base_url:
            raise RuntimeError(
                "Set OPBENCH_AGENT_BASE_URL for one endpoint, or set "
                f"${self.config.agent_host_env} and a port map for per-person agents."
            )
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base += "/v1"
        return f"{base}/chat/completions"

    def chat(self, person_name: str, question: str, model: str | None = None) -> CallResult:
        payload = {
            "model": model or self.config.agent.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": self.config.agent.temperature,
            "max_tokens": self.config.agent.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        start = time.perf_counter()
        try:
            response = requests.post(
                self._url(person_name),
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return CallResult(text=text or "", duration_ms=(time.perf_counter() - start) * 1000, raw=body)
        except Exception as exc:  # noqa: BLE001 - normalize network/JSON failures
            return CallResult(text="", duration_ms=(time.perf_counter() - start) * 1000, error=f"{type(exc).__name__}: {exc}")


class MemosAPIClient:
    """Minimal adapter for the local and hosted MemOS HTTP APIs."""

    def __init__(self, *, online: bool = False, timeout: float = 60.0):
        self.online = online
        self.base_url = os.getenv("MEMOS_API_BASE_URL", "").rstrip("/")
        if not self.base_url:
            raise RuntimeError("Set MEMOS_API_BASE_URL before using the MemOS adapter.")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        key = os.getenv("MEMOS_API_KEY", "")
        if key:
            self.session.headers["Authorization"] = key

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}{path}", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise TypeError("MemOS response must be a JSON object")
        return body

    def add(self, messages: list[dict[str, Any]], user_id: str, conversation_id: str) -> Any:
        path = "/add/message" if self.online else "/product/add"
        payload = {
            "messages": messages,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        if not self.online:
            payload["mem_cube_id"] = user_id
        return self._post(path, payload)

    def search(self, query: str, user_id: str, top_k: int = 5) -> Any:
        if self.online:
            body = self._post(
                "/search/memory",
                {
                    "query": query,
                    "user_id": user_id,
                    "memory_limit_number": top_k,
                    "mode": os.getenv("MEMOS_SEARCH_MODE", "fast"),
                    "include_preference": True,
                    "pref_top_k": 6,
                },
            )
            return body.get("data", body)
        body = self._post(
            "/product/search",
            {
                "query": query,
                "user_id": user_id,
                "mem_cube_id": user_id,
                "conversation_id": "",
                "top_k": top_k,
                "mode": os.getenv("MEMOS_SEARCH_MODE", "fast"),
                "include_preference": True,
                "pref_top_k": 6,
            },
        )
        return body.get("data", body)


def memory_text(value: Any) -> str:
    """Normalize the known MemOS response shapes into a promptable string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("memory", item.get("memory_value", item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        parts: list[str] = []
        text_mem = value.get("text_mem", value.get("memory_detail_list"))
        if isinstance(text_mem, list):
            parts.append(memory_text(text_mem))
        preferences = value.get("preference_detail_list")
        if isinstance(preferences, list) and preferences:
            preference_lines = ["Preferences:"]
            for preference in preferences:
                if isinstance(preference, dict):
                    text = preference.get("preference", preference.get("memory", ""))
                    if text:
                        preference_lines.append(f"- {text}")
            if len(preference_lines) > 1:
                parts.append("\n".join(preference_lines))
        if value.get("pref_string"):
            parts.append(str(value["pref_string"]))
        if value.get("preference_note"):
            parts.append(str(value["preference_note"]))
        if parts:
            return "\n".join(part for part in parts if part)
        return str(value.get("memory", value.get("memory_value", value)))
    return str(value)


def openai_client(endpoint: EndpointConfig):
    return create_openai_client(endpoint)
