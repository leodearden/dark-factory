"""Thin graphiti-core LLM client subclasses owned by fused-memory.

It also holds the category router, CategoryRoutedLLMClient, which sends the
extraction prompts made inside an extraction_category_scope to that
category's route.

Kept out of the already-large ``graphiti_client.py``, following the
``backends/falkor_fulltext.py`` convention for focused helpers.
"""

import contextlib
import json
from collections.abc import Iterator, Mapping
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import Reasoning
from pydantic import BaseModel

from fused_memory.models.enums import MemoryCategory

# Prefixed to the inlined schema. Phrased as an instruction rather than a bare
# schema dump because the model is being asked to honour it without any
# server-side enforcement behind it.
_INLINE_SCHEMA_INSTRUCTION = (
    '\n\nRespond with a single JSON object conforming to this JSON Schema. '
    'Use exactly these top-level keys and field names; resolve any "$ref" '
    'against the "$defs" section. Output only the JSON object — no prose, no '
    'code fences.\n\nJSON Schema:\n'
)


def _with_inline_schema(
    messages: list[Message], response_model: type[BaseModel],
) -> list[Message]:
    """Return a copy of ``messages`` with the schema appended to the last one.

    A COPY, never an in-place append: upstream's ``generate_response`` retry
    loop appends its error-context message to the same list and calls
    ``_generate_response`` again, so mutating here would stack one schema block
    per attempt.
    """
    if not messages:
        return messages
    last = messages[-1]
    schema_text = json.dumps(response_model.model_json_schema(), indent=2)
    return [
        *messages[:-1],
        Message(role=last.role, content=last.content + _INLINE_SCHEMA_INSTRUCTION + schema_text),
    ]


class _CallUsage:
    """The ``usage`` block of the latest completion inside one ``generate_response``."""

    def __init__(self) -> None:
        self.latest: Any = None

    def tokens(self) -> tuple[int, int]:
        return (
            getattr(self.latest, 'prompt_tokens', 0) or 0,
            getattr(self.latest, 'completion_tokens', 0) or 0,
        )


_CALL_USAGE: ContextVar[_CallUsage | None] = ContextVar('generic_client_call_usage', default=None)


class _Delegate:
    """Forwards every attribute to ``target`` except the ``overrides``."""

    def __init__(self, target: Any, **overrides: Any) -> None:
        self._target = target
        vars(self).update(overrides)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _UsageCapturingCompletions(_Delegate):
    async def create(self, **kwargs: Any) -> Any:
        response = await self._target.create(**kwargs)
        call_usage = _CALL_USAGE.get()
        if call_usage is not None:
            call_usage.latest = getattr(response, 'usage', None)
        return response


def _with_usage_capture(openai_client: Any) -> _Delegate:
    chat = openai_client.chat
    return _Delegate(
        openai_client,
        chat=_Delegate(chat, completions=_UsageCapturingCompletions(chat.completions)),
    )


class TokenRecordingOpenAIGenericClient(OpenAIGenericClient):
    """OpenAIGenericClient that records token usage, which upstream drops.

    graphiti-core 0.28.2's ``OpenAIGenericClient`` discards ``response.usage``
    and never calls ``token_tracker.record``, so this arm would otherwise report
    a fabricated zero. Usage is captured from the wire response through a
    delegating wrapper around ``self.client`` (nothing upstream is vendored) and
    recorded once per successful ``generate_response``, for the successful
    attempt only, exactly as ``BaseOpenAIClient.generate_response`` records it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = _with_usage_capture(self.client)

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
        call_usage = _CallUsage()
        reset_token = _CALL_USAGE.set(call_usage)
        try:
            response = await super().generate_response(
                messages,
                response_model=response_model,
                max_tokens=max_tokens,
                model_size=model_size,
                group_id=group_id,
                prompt_name=prompt_name,
            )
        finally:
            _CALL_USAGE.reset(reset_token)
        self.token_tracker.record(prompt_name, *call_usage.tokens())
        return response


class ForceJsonObjectOpenAIGenericClient(TokenRecordingOpenAIGenericClient):
    """OpenAIGenericClient that always requests ``response_format=json_object``.

    Selected by ``llm.structured_output_mode='json_object'`` (only meaningful
    alongside ``llm.client_class='openai_generic'``).

    WHY WE OWN THIS
    ---------------
    graphiti-core 0.28.2 ships **no** ``structured_output_mode`` knob — the
    mode is purely response_model-driven. ``_generate_response`` builds
    ``{'type': 'json_object'}`` and then *replaces* it with
    ``{'type': 'json_schema', ...}`` whenever ``response_model is not None``
    (openai_generic_client.py:111-121), and ``generate_response`` forwards
    ``response_model`` straight through. There is no upstream way to ask for
    the weaker mode while still passing a response_model.

    THE MOTIVATING FAILURE
    ----------------------
    The llama.cpp MoE arm SILENTLY ignores ``$ref``/``$defs`` inside a
    json_schema ``response_format`` (llama.cpp#21228). It returns 200 OK with
    off-schema JSON rather than erroring, so the failure surfaces far
    downstream as bad extractions. Because it is silent, the accompanying test
    carries a loud control assertion that the stock client really does emit
    ``$defs``/``$ref`` — otherwise a future wheel could make this wrapper inert
    without anything going red.

    WHY THIS SEAM
    -------------
    Dropping ``response_model`` at the ``_generate_response`` boundary is the
    minimal, upstream-faithful forcing mechanism: the retry loop, the tracing
    span, message construction and the prompt text are all in
    ``generate_response``, above this call, and stay completely untouched.
    Rewriting ``response_format`` after the fact, or vendoring the client,
    would duplicate upstream code that a wheel bump could silently diverge
    from.

    WHAT DROPPING response_model COSTS, AND HOW IT IS PAID BACK
    ----------------------------------------------------------
    ``response_model`` is not only the server-side enforcement switch — under
    graphiti-core 0.28.2 it is the ONLY schema signalling there is. Its stock
    prompts (``prompts/extract_nodes.py``) describe the *task* but never name
    the response envelope (``extracted_entities``) or the per-entity field
    names; the structure came entirely from the json_schema response_format.
    So the naive override would ask a local endpoint for "some valid JSON"
    with nothing telling it what shape, and downstream that splits two ways:
    ``node_operations.py`` does ``ExtractedEntities(**llm_response)`` — a hard
    failure *outside* the retry loop — while ``community_operations.py`` does
    ``llm_response.get('summary', '')``, which degrades to empty strings with
    no error at all. The second is precisely the silent-wrong-output class this
    whole knob exists to prevent.

    This override therefore replaces both halves of what it removes:

    1. **In-band schema.** The schema is appended to the last message, so the
       model is still told the envelope and field names — just in the prompt
       rather than in ``response_format``.
    2. **Client-side validation.** The decoded payload is validated against
       ``response_model`` and a mismatch RAISES. Upstream's ``generate_response``
       catches that (it retries on any non-rate-limit, non-transport
       ``Exception``), appends the error text as a new user message and
       re-prompts — so an off-envelope response self-corrects, and an
       unfixable one fails loudly after MAX_RETRIES instead of silently
       reaching downstream ``.get()`` calls.

    DELETE ME when a future graphiti-core exposes the mode natively.
    """

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        if response_model is None:
            # Nothing to force and nothing to compensate for — upstream already
            # uses json_object when no response_model is passed.
            return await super()._generate_response(
                messages,
                response_model=None,
                max_tokens=max_tokens,
                model_size=model_size,
            )

        # Passing response_model=None is the forcing mechanism: it makes
        # upstream keep its `{'type': 'json_object'}` default instead of
        # swapping in a json_schema response_format. The schema goes into the
        # prompt instead, so the model is not left guessing the envelope.
        result = await super()._generate_response(
            _with_inline_schema(messages, response_model),
            response_model=None,
            max_tokens=max_tokens,
            model_size=model_size,
        )

        # Validate what the server was no longer asked to enforce. A
        # ValidationError here is deliberately allowed to propagate: upstream's
        # generate_response retry loop turns it into a re-prompt carrying the
        # error text, and raises it after MAX_RETRIES. Returning the payload
        # unchecked would hand off-envelope JSON to downstream `.get()` calls
        # that silently read it as empty.
        response_model.model_validate(result)
        return result


def _responses_input(messages: list[ChatCompletionMessageParam]) -> ResponseInputParam:
    """graphiti's converted messages as Responses input. graphiti emits only system and user
    messages with text content, so anything else is a defect, refused before it is sent."""
    converted: ResponseInputParam = []
    for message in messages:
        role, content = message['role'], message.get('content')
        if role not in ('system', 'user') or not isinstance(content, str):
            raise TypeError(f'a {role} message with {type(content).__name__} content is not Responses input')
        converted.append(EasyInputMessageParam(role=role, content=content))
    return converted


class ExplicitEffortOpenAIClient(OpenAIClient):
    """graphiti's OpenAIClient for a reasoning model graphiti_core 0.28.2 does not recognise by name.

    graphiti treats only gpt-5, o1 and o3 names as reasoning models: any other
    model is sent temperature and never sent reasoning, and gpt-6.1-sol answers
    a temperature with HTTP 400. This client sends no temperature, and exactly
    the chosen reasoning effort (esc-9065-2). It wraps an already-built client,
    reusing its SDK client, config, max_tokens and token tracker. graphiti's own
    ``reasoning`` stays at its default and is never read: ``reasoning_effort``
    is the one copy.
    """

    def __init__(self, built: OpenAIClient, effort: ReasoningEffort) -> None:
        if effort is None:
            raise ValueError('an effort client sends a chosen effort; build the configured client instead')
        super().__init__(config=built.config, client=built.client, max_tokens=built.max_tokens)
        self.token_tracker = built.token_tracker
        self._reasoning_effort: ReasoningEffort = effort

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """The effort every structured call sends."""
        return self._reasoning_effort

    async def _create_structured_completion(
        self, model: str, messages: list[ChatCompletionMessageParam], temperature: float | None,
        max_tokens: int, response_model: type[BaseModel], reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        return await self.client.responses.parse(
            model=model, input=_responses_input(messages), max_output_tokens=max_tokens,
            text_format=response_model, reasoning=Reasoning(effort=self._reasoning_effort),
        )

    async def _create_completion(
        self, model: str, messages: list[ChatCompletionMessageParam], temperature: float | None,
        max_tokens: int, response_model: type[BaseModel] | None = None, reasoning: str | None = None,
        verbosity: str | None = None,
    ) -> Any:
        raise NotImplementedError(
            'a JSON-mode chat completion cannot carry reasoning.effort, so it is never sent without it',
        )


EXTRACTION_PROMPT_NAMES: frozenset[str] = frozenset({
    'extract_nodes.extract_message',
    'extract_nodes.extract_text',
    'extract_nodes.extract_json',
    'extract_edges.edge',
})
"""graphiti_core 0.28.2's node_operations.py::_call_extraction_llm and edge_operations.py::extract_edges prompts."""

_EXTRACTION_CATEGORY: ContextVar[MemoryCategory | None] = ContextVar('extraction_category', default=None)


@contextlib.contextmanager
def extraction_category_scope(category: MemoryCategory | None) -> Iterator[None]:
    """Route the extraction calls made inside this scope, and in tasks it spawns, by *category*."""
    reset_token = _EXTRACTION_CATEGORY.set(category)
    try:
        yield
    finally:
        _EXTRACTION_CATEGORY.reset(reset_token)


class CategoryRoutedLLMClient(LLMClient):
    """Sends a scoped category's extraction prompts to that category's route, everything else to the default.

    The category is the one extraction_category_scope set for the calling
    context. The default's token tracker is the router's, so a measured window
    over the router credits every inner client whose tracker attributes.
    """

    def __init__(self, default: LLMClient, routes: Mapping[MemoryCategory, LLMClient]) -> None:
        super().__init__(config=default.config)
        self.token_tracker = default.token_tracker
        self._default = default
        self._routes: Mapping[MemoryCategory, LLMClient] = MappingProxyType(dict(routes))

    @property
    def default(self) -> LLMClient:
        return self._default

    @property
    def routes(self) -> Mapping[MemoryCategory, LLMClient]:
        return self._routes

    def client_for(self, prompt_name: str | None) -> LLMClient:
        if prompt_name in EXTRACTION_PROMPT_NAMES and (category := _EXTRACTION_CATEGORY.get()) is not None:
            return self._routes.get(category, self._default)
        return self._default

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
    ) -> dict[str, Any]:
        return await self.client_for(prompt_name).generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
        )

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        raise NotImplementedError('generate_response delegates whole calls to an inner client')

    def set_tracer(self, tracer: Any) -> None:
        super().set_tracer(tracer)
        for client in (self._default, *self._routes.values()):
            client.set_tracer(tracer)


def wrapped_llm_clients(client: LLMClient | None) -> tuple[LLMClient | None, ...]:
    """The clients that actually send *client*'s calls: a router's default then its routes, else *client*."""
    if isinstance(client, CategoryRoutedLLMClient):
        return (client.default, *client.routes.values())
    return (client,)
