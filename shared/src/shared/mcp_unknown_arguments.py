"""The refusal of a tool call that carries an argument its tool does not declare.

The FastMCP bundled in the ``mcp`` SDK validates arguments with a pydantic
model that IGNORES extras, so a misnamed argument (``metadata_patch`` for
``metadata``) vanishes and the call runs as if it had never been sent. That is
how escalations esc-3147-7 and esc-3671-7 began (task 4867).

Both halves of the wire contract live here, so a server and its callers read
one definition: :func:`unknown_argument_refusal` builds the structured refusal
keyed by :data:`UNKNOWN_ARGUMENT_CODE`, and :func:`refused_arguments` reads the
undeclared names back out of a refusal's text. The module imports no fastmcp,
so a stdlib client can depend on it.
"""
from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from typing import Any

UNKNOWN_ARGUMENT_CODE: str = 'unknown_argument'

_HINT = (
    'The call was rejected BEFORE the tool ran: nothing was executed, enqueued '
    'or written, so resubmitting is safe. Each argument under `unknown` is not '
    'declared by this tool and would otherwise have been silently ignored. '
    'Rename it to the declared name you meant, or remove it. `closest_declared` '
    'is only a spelling guess and can name a parameter with a different meaning, '
    'so check it against the tool\'s documentation before resubmitting.'
)


def unknown_argument_refusal(
    tool: str, declared: Sequence[str], arguments: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The refusal for *arguments* sent to *tool*, or ``None`` when every name is declared.

    *declared* is the tool's parameter names in declaration order.
    """
    unknown = [name for name in arguments if name not in declared]
    if not unknown:
        return None
    closest = {name: _closest(name, declared) for name in unknown}
    return {
        'error': (
            f'{tool} was called with argument(s) it does not declare: {", ".join(unknown)}'
        ),
        'code': UNKNOWN_ARGUMENT_CODE,
        'tool': tool,
        'unknown': [{'name': name, 'closest_declared': closest[name]} for name in unknown],
        'declared': list(declared),
        'provided': list(arguments),
        'hint': _HINT,
    }


def refused_arguments(text: str) -> frozenset[str]:
    """The undeclared argument names a tool-error *text* refuses, or nothing if it is no such refusal."""
    try:
        refusal = json.loads(text)
    except ValueError:
        return frozenset()
    if not isinstance(refusal, Mapping) or refusal.get('code') != UNKNOWN_ARGUMENT_CODE:
        return frozenset()
    entries = refusal.get('unknown')
    if not isinstance(entries, list):
        return frozenset()
    return frozenset(
        entry['name'] for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get('name'), str)
    )


def _closest(name: str, declared: Sequence[str]) -> str | None:
    matches = difflib.get_close_matches(name, declared, n=1)
    return matches[0] if matches else None

