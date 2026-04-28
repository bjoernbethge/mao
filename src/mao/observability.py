"""Central LangSmith observability helpers for MAO."""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from langsmith import Client, tracing_context

MAO_VERSION = "1.0.0"

_REQUEST_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "mao_request_context", default={}
)
_CLIENT: Client | None = None


def _env_flag(*names: str, default: bool = False) -> bool:
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        return raw.strip().lower() in {"1", "true", "yes", "on", "local"}
    return default


def _env_float(*names: str) -> float | None:
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


@dataclass(slots=True)
class LangSmithSettings:
    enabled: bool
    api_key: str | None
    project: str
    endpoint: str | None
    workspace_id: str | None
    sampling_rate: float | None

    @classmethod
    def from_env(cls) -> "LangSmithSettings":
        return cls(
            enabled=_env_flag("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
            api_key=os.environ.get("LANGSMITH_API_KEY"),
            project=os.environ.get("LANGSMITH_PROJECT", "mao-agents"),
            endpoint=os.environ.get("LANGSMITH_ENDPOINT"),
            workspace_id=os.environ.get("LANGSMITH_WORKSPACE_ID"),
            sampling_rate=_env_float(
                "LANGSMITH_TRACING_SAMPLING_RATE",
                "LANGCHAIN_TRACING_SAMPLING_RATE",
            ),
        )

    def tracing_enabled(self) -> bool:
        return bool(self.enabled and self.api_key)


def get_langsmith_settings() -> LangSmithSettings:
    return LangSmithSettings.from_env()


def get_langsmith_client() -> Client | None:
    global _CLIENT
    settings = get_langsmith_settings()
    if not settings.tracing_enabled():
        return None
    if _CLIENT is None:
        kwargs: dict[str, Any] = {
            "api_key": settings.api_key,
            "workspace_id": settings.workspace_id,
            "tracing_sampling_rate": settings.sampling_rate,
        }
        if settings.endpoint:
            kwargs["api_url"] = settings.endpoint
        _CLIENT = Client(**kwargs)
    return _CLIENT


def current_request_context() -> dict[str, Any]:
    return dict(_REQUEST_CONTEXT.get({}))


def current_request_id() -> str | None:
    return current_request_context().get("request_id")


def next_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def set_request_context(**values: Any) -> Token:
    current = current_request_context()
    current.update({k: v for k, v in values.items() if v is not None})
    return _REQUEST_CONTEXT.set(current)


def reset_request_context(token: Token) -> None:
    _REQUEST_CONTEXT.reset(token)


def build_trace_metadata(
    *,
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_context = current_request_context()
    merged: dict[str, Any] = {
        "service": "mao",
        "service_version": MAO_VERSION,
    }
    merged.update(request_context)
    if thread_id:
        merged["thread_id"] = thread_id
    if metadata:
        merged.update({k: v for k, v in metadata.items() if v is not None})
    return merged


def merge_trace_tags(*tag_groups: list[str] | None) -> list[str]:
    tags: list[str] = ["mao"]
    for tag_group in tag_groups:
        if not tag_group:
            continue
        tags.extend(tag_group)
    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag and tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


@contextmanager
def langsmith_trace(
    *,
    run_name: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    settings = get_langsmith_settings()
    client = get_langsmith_client()
    with tracing_context(
        project_name=settings.project,
        tags=merge_trace_tags(tags),
        metadata=build_trace_metadata(metadata=metadata),
        enabled=settings.tracing_enabled(),
        client=client,
    ):
        yield
