"""Shared doubles for the graphiti extraction-route tests (task 9943).

RecordingLLMClient stands in for one of CategoryRoutedLLMClient's inner
clients. decisions_routed_to and scrub_ambient_openai_base_url set a
decisions_and_rationale route up onto a local mock server.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig as GraphitiLLMConfig
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

from fused_memory.config.schema import ExtractionRouteConfig

ROUTE_MODEL = 'gpt-6.1-sol'
PLAN_KEY = 'plan-key'


class RecordingSdk:
    """A stand-in SDK client that records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class RecordingLLMClient(LLMClient):
    """Records every call and tracer it receives; its ``client`` is a RecordingSdk.

    A call whose response_model *answers* holds gets that payload back; any
    other call gets ``{'client': name}``.
    """

    def __init__(self, name: str, answers: Mapping[type[BaseModel], dict[str, Any]] | None = None) -> None:
        super().__init__(GraphitiLLMConfig(model=f'{name}-model'))
        self.name = name
        self.client = RecordingSdk()
        self.answers = dict(answers or {})
        self.calls: list[dict[str, Any]] = []
        self.tracers: list[Any] = []

    @property
    def prompt_names(self) -> list[str | None]:
        return [call['prompt_name'] for call in self.calls]

    @property
    def response_models(self) -> list[type[BaseModel] | None]:
        return [call['response_model'] for call in self.calls]

    def set_tracer(self, tracer: Any) -> None:
        self.tracers.append(tracer)
        super().set_tracer(tracer)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
        await asyncio.sleep(0)
        self.calls.append({
            'messages': messages, 'response_model': response_model, 'max_tokens': max_tokens,
            'model_size': model_size, 'group_id': group_id, 'prompt_name': prompt_name,
        })
        if response_model in self.answers:
            return self.answers[response_model]
        return {'client': self.name}

    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError('generate_response is overridden')


def decisions_routed_to(api_url: str) -> dict[str, ExtractionRouteConfig]:
    """``llm.extraction_routes`` sending decisions_and_rationale to ROUTE_MODEL at effort low, on *api_url* with PLAN_KEY."""
    return {
        'decisions_and_rationale': ExtractionRouteConfig(
            model=ROUTE_MODEL, reasoning_effort='low', api_url=api_url, api_key=PLAN_KEY,
        ),
    }


def scrub_ambient_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the ambient base-URL variables, so no client warns that a mock server's URL overrides one."""
    for name in ('OPENAI_BASE_URL', 'OPENAI_API_BASE', 'OPENAI_API_URL'):
        monkeypatch.delenv(name, raising=False)
