"""Thin graphiti-core LLM client subclasses owned by fused-memory.

Kept out of the already-large ``graphiti_client.py``, following the
``backends/falkor_fulltext.py`` convention for focused helpers.
"""

from typing import Any

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel


class ForceJsonObjectOpenAIGenericClient(OpenAIGenericClient):
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
    ``generate_response``, above this call, and stay completely untouched. In
    particular the prompt still describes the expected schema in-band, so the
    model is still told what to produce — we only stop asking the *server* to
    enforce it. Rewriting ``response_format`` after the fact, or vendoring the
    client, would duplicate upstream code that a wheel bump could silently
    diverge from.

    SCOPE
    -----
    Client-side REQUEST-mode selection only. Validating the response against
    the expected schema is a separate, harness-side concern.

    DELETE ME when a future graphiti-core exposes the mode natively.
    """

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        # Passing response_model=None is the entire override: it makes
        # upstream keep its `{'type': 'json_object'}` default instead of
        # swapping in a json_schema response_format.
        return await super()._generate_response(
            messages,
            response_model=None,
            max_tokens=max_tokens,
            model_size=model_size,
        )
