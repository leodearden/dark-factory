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
    'Rename it to a name under `declared` (`closest_declared` is the likeliest) '
    'or remove it. `repaired_call`, when not null, is your complete argument map '
    'with each unknown argument renamed to its closest declared name: confirm '
    'each rename is what you meant, then resubmit it.'
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
        'repaired_call': _repaired_call(arguments, closest),
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


def _repaired_call(
    arguments: Mapping[str, Any], closest: Mapping[str, str | None]
) -> dict[str, Any] | None:
    """*arguments* with every unknown name renamed, or ``None`` if any rename would lose a value.

    A rename loses a value when there is no close declared name, when the
    caller already sent that name, or when two unknown names would land on it.
    """
    targets = list(closest.values())
    for target in targets:
        if target is None or target in arguments or targets.count(target) > 1:
            return None
    return {closest.get(name) or name: value for name, value in arguments.items()}
