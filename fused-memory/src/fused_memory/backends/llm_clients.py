"""Thin graphiti-core LLM client subclasses owned by fused-memory.

Kept out of the already-large ``graphiti_client.py``, following the
``backends/falkor_fulltext.py`` convention for focused helpers.
"""

import json
from typing import Any

from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

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
