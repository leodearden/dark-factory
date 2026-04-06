"""vLLM → Anthropic protocol bridge.

Provides:
- Pure translation functions (testable without a server):
  - ``_normalize_tool_use_block`` — normalises a single tool_use block
  - ``_translate_messages_response`` — normalises a full /v1/messages response body
- ``VllmBridge`` — async-context-manager aiohttp proxy that starts a local
  HTTP server, translates POST /v1/messages responses, and passes all other
  traffic straight through to the configured upstream URL.
"""

from __future__ import annotations

import json
import uuid


# ── pure translation helpers ─────────────────────────────────────────────────


def _normalize_tool_use_block(block: dict) -> dict:
    """Return a normalised copy of a tool_use content block.

    Handles the following malformations produced by vLLM:
    - ``input`` serialised as a JSON string instead of a dict → json.loads'd
    - ``id`` missing → generated as ``'toolu_' + uuid4().hex[:24]``
    - ``id`` present but not prefixed with ``'toolu_'`` → wrapped as
      ``'toolu_' + existing_id``

    The function is idempotent: a well-formed Anthropic block is returned
    unchanged (value-equal).  The input dict is never mutated; a new dict
    is always returned.
    """
    result = dict(block)

    # ── normalise `input` ────────────────────────────────────────────────────
    raw_input = result.get('input')
    if isinstance(raw_input, str):
        try:
            result['input'] = json.loads(raw_input)
        except json.JSONDecodeError:
            result['input'] = {'_raw': raw_input}

    # ── normalise `id` ──────────────────────────────────────────────────────
    existing_id = result.get('id')
    if not existing_id:
        result['id'] = 'toolu_' + uuid.uuid4().hex[:24]
    elif not str(existing_id).startswith('toolu_'):
        result['id'] = 'toolu_' + existing_id

    return result


def _translate_messages_response(body: dict) -> dict:
    """Return a normalised copy of a /v1/messages response body.

    Handles the following vLLM malformations:
    - OpenAI-style top-level ``tool_calls`` list → Anthropic content[] blocks
    - ``stop_reason='tool_calls'`` → ``'tool_use'`` when content has tool_use blocks
    - tool_use blocks with JSON-string ``input`` or non-Anthropic ``id``

    The function is idempotent: a well-formed Anthropic response body is
    returned unchanged.  Error bodies and non-assistant responses are passed
    through unchanged.  The input dict is never mutated; a new dict is returned.
    """
    # ── pass through non-assistant / error bodies ────────────────────────────
    if body.get('type') == 'error' or body.get('role') != 'assistant':
        return dict(body)

    result = dict(body)

    # ── convert OpenAI-style top-level tool_calls ────────────────────────────
    if 'tool_calls' in result:
        raw_content = result.get('content', '')
        content_list: list[dict] = []

        # Wrap any existing string content into a text block
        if isinstance(raw_content, str) and raw_content:
            content_list.append({'type': 'text', 'text': raw_content})
        elif isinstance(raw_content, list):
            content_list.extend(raw_content)

        # Convert each OpenAI tool_call to an Anthropic tool_use block
        for tc in result['tool_calls']:
            fn = tc.get('function', {})
            block = {
                'type': 'tool_use',
                'id': tc.get('id', ''),
                'name': fn.get('name', ''),
                'input': fn.get('arguments', '{}'),
            }
            content_list.append(_normalize_tool_use_block(block))

        result['content'] = content_list
        del result['tool_calls']

    # ── normalise any tool_use blocks already in content[] ───────────────────
    if isinstance(result.get('content'), list):
        normalised: list[dict] = []
        for block in result['content']:
            if block.get('type') == 'tool_use':
                normalised.append(_normalize_tool_use_block(block))
            else:
                normalised.append(block)
        result['content'] = normalised

    # ── fix stop_reason ──────────────────────────────────────────────────────
    has_tool_use = isinstance(result.get('content'), list) and any(
        b.get('type') == 'tool_use' for b in result['content']
    )
    if has_tool_use:
        result['stop_reason'] = 'tool_use'

    return result
