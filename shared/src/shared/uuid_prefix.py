"""Detection of truncated-uuid prefixes in MCP tool arguments (PRD contract C1).

``plans/uuid-prefix-resolution-prd.md`` §4-C1 OWNS the token grammar and is
normative. Do not re-derive it here, and do not "improve" it against a hunch:
every rule below was decided against a measured corpus of 47,209 hex-8
occurrences, and the numbers cited are the paper's, not a second measurement
(INV-9).

WHY THE GLUE RULE EXISTS. A run preceded by ``-`` or ``_`` is not a token.
Composite identifiers minted by the factory itself — ``recon-<hex>`` handles,
``episode_<hex>`` names, ``STAGE2_<hex>`` labels — truncate OTHER namespaces
and would resolve to nothing, and they are 29% of the noise class. The rule
costs 2.8% of true references, nearly all of which are ``/``-separated lists
that the rule keeps because a ``/`` is not glue.

WHY THERE IS NO FULL-UUID PREDICATE HERE. The grammar excludes every group of
a 36-char uuid on its own: group 1 is 8 hex followed by ``-``, groups 2-4 are
4 hex (below the 8-char floor), and group 5 is 12 hex preceded by ``-``. A
32-char undashed uuid is above the 31-char ceiling. ``fused_memory.utils.
validation::is_full_uuid`` is not importable from ``shared`` (fused-memory
depends on shared, not the reverse), and writing a second copy of it here is
exactly the lockstep duplication INV-5 forbids — so the cheapest correct
answer is to write no predicate at all and pin the behaviour by test.

LOOP-THREAD OCCUPANCY BOUND (INV-8). :func:`find_prefix_tokens` is synchronous
and runs on the event-loop thread inside a middleware hook, so its cost is
stated rather than assumed. It performs a single linear pass per string with
one compiled regex whose body is a bounded character-class repetition and
whose lookarounds are fixed-width, so there is no backtracking and no
per-token re-scan. Worst-case work is therefore O(total characters across all
strings reachable from the MCP argument map) + O(number of container nodes),
with no dependence on how many tokens are found.

Stdlib only, deliberately: no fastmcp, no pydantic. Leaf β's resolver and leaf
γ's ``tools.py`` both need this module, and neither should pull a middleware
dependency to reach it. The module is not re-exported from
``shared/__init__`` — the ``mcp_envelope`` / ``storm_counter`` /
``toolcall_markup`` convention.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

__all__ = ['PrefixToken', 'find_prefix_tokens']


#: The PRD §4-C1 grammar as ONE compiled expression. Both lookbehinds are
#: fixed-width, so the whole pattern is linear and backtrack-free — the
#: mechanical basis for the INV-8 bound in the module docstring.
#:
#: ``(?<![0-9a-f])``  not preceded by a hex char (no mid-run match)
#: ``(?<![-_])``      the GLUE RULE: not preceded by '-' or '_'
#: ``[0-9a-f]{8,31}`` lowercase hex only, floor 8, ceiling 31 inclusive
#: ``(?![0-9a-f-])``  not followed by a hex char, and not followed by '-'
#:                    (so group 1 of a full uuid never matches)
_PREFIX_RE = re.compile(r'(?<![0-9a-f])(?<![-_])[0-9a-f]{8,31}(?![0-9a-f-])')


class PrefixToken(NamedTuple):
    """One prefix-shaped run, located precisely enough to replace it.

    ``path`` is the structured route to the string that contains the token —
    ``('content',)`` for a top-level argument, ``('metadata',
    'cluster_memory_ids', 2)`` for a nested one — never an encoded string a
    consumer would need an ad-hoc parser for (heuristic 12).

    ``start``/``end`` are the span WITHIN that string, so
    ``value[start:end] == token`` always holds.
    """

    path: tuple[str | int, ...]
    token: str
    start: int
    end: int


def find_prefix_tokens(arguments: Any) -> tuple[PrefixToken, ...]:
    """Return every prefix-shaped token in *arguments*, in document order.

    Pure, synchronous, and never raises for any JSON-shaped input.
    """
    if not isinstance(arguments, dict):
        return ()
    found: list[PrefixToken] = []
    for name, value in arguments.items():
        if isinstance(value, str):
            found.extend(_scan(value, (name,)))
    return tuple(found)


def _scan(value: str, path: tuple[str | int, ...]) -> list[PrefixToken]:
    return [
        PrefixToken(path=path, token=m.group(), start=m.start(), end=m.end())
        for m in _PREFIX_RE.finditer(value)
    ]
