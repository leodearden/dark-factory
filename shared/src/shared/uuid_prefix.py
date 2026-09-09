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
from collections.abc import Mapping
from typing import Any, NamedTuple

__all__ = ['PrefixToken', 'find_prefix_tokens', 'substitute']


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

    Every string value reachable through dicts AND lists is scanned, not only
    the top-level ones that ``mcp_markup_middleware.py::_first_markup_argument``
    inspects. D11 is the reason: the DF 4643 instance lives one level down at
    ``metadata.cluster_memory_ids[i]``, and the measured field spread is 20+
    paths, so a top-level-only scan would have missed the incident that
    motivated the contract.

    Only VALUES are scanned. A dict key that happens to look like a prefix is
    a field name, not a citation.

    Document order is argument insertion order, then container order, then
    span within a string. It falls out of pushing children onto the stack in
    REVERSE so they pop in order.

    The traversal is ITERATIVE with an explicit stack, never recursive: this
    runs on the event-loop thread inside a middleware hook, and a deeply
    nested argument map must not turn a guard into a RecursionError.

    PRECONDITION: *arguments* is JSON-derived, and therefore finite and
    acyclic. No cycle detection is written — a cycle cannot arrive through the
    MCP wire, and none is testable without hanging the suite.

    Pure, synchronous, and never raises for any JSON-shaped input.
    """
    if not isinstance(arguments, dict):
        return ()
    found: list[PrefixToken] = []
    stack: list[tuple[tuple[str | int, ...], Any]] = [((), arguments)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, str):
            found.extend(_scan(node, path))
        elif isinstance(node, dict):
            stack.extend((path + (k,), v) for k, v in reversed(list(node.items())))
        elif isinstance(node, list):
            stack.extend((path + (i,), v) for i, v in reversed(list(enumerate(node))))
    return tuple(found)


def _scan(value: str, path: tuple[str | int, ...]) -> list[PrefixToken]:
    return [
        PrefixToken(path=path, token=m.group(), start=m.start(), end=m.end())
        for m in _PREFIX_RE.finditer(value)
    ]


def substitute(arguments: Mapping[str, Any], token: PrefixToken, full_id: str) -> dict[str, Any]:
    """Return *arguments* with *token*'s span replaced by *full_id*.

    MONOTONE, and enforced here rather than trusted: *full_id* must have
    ``token.token`` as a prefix, or this raises ``ValueError`` naming both
    values. That check is not decoration, it is what makes the design safe to
    deploy. A wrong expansion under this rule is still VISIBLE in the stored
    text and REVERSIBLE by truncation, because every character the author
    actually wrote is still there. A replacement free to drop them would be
    silent information loss that no reader downstream could detect. This
    function is the only door through which that could happen, so the
    invariant is enforced at the door (heuristic 10).

    A PATH COPY, not a deep copy: only the containers along ``token.path`` are
    rebuilt — O(depth), not O(size) — and every untouched substructure is
    shared with the input by reference.

    The input is NEVER mutated. The guard depends on that: the original
    argument map is its fallback when a later token in the same call turns out
    to be ambiguous or unresolvable, which is what makes "forwarded unchanged"
    structurally true rather than incidental.

    Single-token by design. Applying several is the caller's fold in REVERSE
    document order, which keeps every remaining span valid without re-scanning;
    a batch API would need its own ordering contract for no gain.
    """
    if not full_id.startswith(token.token):
        raise ValueError(
            f'refusing a non-monotone substitution: {full_id!r} does not start with '
            f'the token it would replace, {token.token!r}. An expansion must keep the '
            f'original characters so that a wrong one stays visible and reversible.'
        )
    trail: list[tuple[Any, str | int]] = []
    node: Any = arguments
    for key in token.path:
        trail.append((node, key))
        node = node[key]
    rebuilt: Any = node[: token.start] + full_id + node[token.end :]
    for container, key in reversed(trail):
        rebuilt = _with_child(container, key, rebuilt)
    return rebuilt


def _with_child(container: Any, key: str | int, child: Any) -> Any:
    """A shallow copy of *container* with one slot replaced.

    Shallow is the point: every sibling comes across by reference, so the copy
    costs O(width of this one node) and nothing deeper is touched.
    """
    if isinstance(key, int):
        copied = list(container)
        copied[key] = child
        return copied
    return {**container, key: child}
