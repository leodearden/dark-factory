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
