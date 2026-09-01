"""Configuration and secret-free construction of OpenAI-compatible clients."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EndpointConfig:
    model: str
    api_key_env: str
    base_url_env: str
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: float = 60.0
    retries: int = 3

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    @property
    def base_url(self) -> str:
        return os.getenv(self.base_url_env, "")


@dataclass(frozen=True)
class EvaluationConfig:
    agent: EndpointConfig
    scorer: EndpointConfig
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    workers: int = 4
    task_types: tuple[str, ...] = (
        "irrelevance_easy",
        "irrelevance_hard",
        "sycophancy",
        "diversity",
    )
    use_both_personas: bool = False
    agent_ports: dict[str, int] = field(default_factory=dict)
    agent_host_env: str = "OPBENCH_AGENT_HOST"


def _endpoint(section: dict[str, Any], prefix: str) -> EndpointConfig:
    return EndpointConfig(
        model=str(os.getenv(f"OPBENCH_{prefix}_MODEL", section.get("model", "gpt-4o-mini"))),
        api_key_env=str(section.get("api_key_env", f"OPBENCH_{prefix}_API_KEY")),
        base_url_env=str(section.get("base_url_env", f"OPBENCH_{prefix}_BASE_URL")),
        max_tokens=int(section.get("max_tokens", 512)),
        temperature=float(section.get("temperature", 0.0)),
        timeout=float(section.get("timeout", 60.0)),
        retries=max(1, int(section.get("retries", 3))),
    )


def load_config(path: str | Path | None = None) -> EvaluationConfig:
    """Load a YAML file, while keeping all secret material in the environment."""

    raw: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded

    agent = _endpoint(raw.get("agent", {}), "AGENT")
    scorer_section = raw.get("scorer", {})
    scorer = _endpoint(scorer_section, "SCORER")
    evaluation = raw.get("evaluation", {})
    agents = raw.get("agents", {})
    ports = agents.get("ports", {}) if isinstance(agents, dict) else {}
    if not isinstance(ports, dict):
        ports = {}
    env_ports = os.getenv("OPBENCH_AGENT_PORTS", "").strip()
    if env_ports:
        try:
            parsed_ports = json.loads(env_ports)
            if isinstance(parsed_ports, dict):
                ports = parsed_ports
        except json.JSONDecodeError as exc:
            raise ValueError("OPBENCH_AGENT_PORTS must be a JSON object") from exc

    task_types = evaluation.get("task_types", list(EvaluationConfig.__dataclass_fields__["task_types"].default))
    if not isinstance(task_types, list):
        task_types = list(EvaluationConfig.__dataclass_fields__["task_types"].default)

    return EvaluationConfig(
        agent=agent,
        scorer=scorer,
        embedding_model=str(os.getenv("OPBENCH_EMBEDDING_MODEL", scorer_section.get("embedding_model", "text-embedding-3-small"))),
        top_k=max(1, int(evaluation.get("top_k", 5))),
        workers=max(1, int(evaluation.get("workers", 4))),
        task_types=tuple(str(item) for item in task_types),
        use_both_personas=bool(evaluation.get("use_both_personas", False)),
        agent_ports={str(name): int(port) for name, port in ports.items()},
        agent_host_env=str(agents.get("host_env", "OPBENCH_AGENT_HOST")) if isinstance(agents, dict) else "OPBENCH_AGENT_HOST",
    )


def require_api_key(endpoint: EndpointConfig) -> str:
    key = endpoint.api_key
    if not key:
        raise RuntimeError(
            f"No API key found in ${endpoint.api_key_env}. "
            "Set it in a local .env file or export it in the shell."
        )
    return key


def create_openai_client(endpoint: EndpointConfig):
    """Create an OpenAI client without embedding a key or endpoint in source."""

    from openai import OpenAI

    kwargs: dict[str, Any] = {
        "api_key": require_api_key(endpoint),
        "timeout": endpoint.timeout,
        "max_retries": 0,
    }
    # An empty base URL means the SDK's own provider default. It is intentionally
    # not written into the repository.
    if endpoint.base_url:
        kwargs["base_url"] = endpoint.base_url
    return OpenAI(**kwargs)
