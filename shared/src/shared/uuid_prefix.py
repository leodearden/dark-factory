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

The SAME bound is owed by the write side, and until this module had a batch
door it was not kept. :func:`substitute_all` is O(total characters of the
strings it touches + number of replacements), plus one O(depth x width) path
copy per DISTINCT path — with no dependence on how many times any one id is
cited beyond a single linear pass. The door it replaced, a caller-side fold of
single-token :func:`substitute` calls, was O(occurrences x string length),
because every occurrence rebuilt the whole containing string: MEASURED on one
``content`` string carrying 16 distinct tokens each repeated, 0.69s at 6,400
occurrences (275KB), 3.00s at 12,800 (550KB) and 16.91s at 25,600 (1.1MB) —
4x the time per 2x the input — against 0.00s / 0.01s / 0.02s grouped. A
middleware calling that fold stalls a shared single-threaded server for those
seconds on one ordinary large write, which is why the batch door exists and
why the bound is stated here rather than assumed.

Stdlib only, deliberately: no fastmcp, no pydantic. Leaf β's resolver and leaf
γ's ``tools.py`` both need this module, and neither should pull a middleware
dependency to reach it. The module is not re-exported from
``shared/__init__`` — the ``mcp_envelope`` / ``storm_counter`` /
``toolcall_markup`` convention.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

__all__ = [
    'UUID_PREFIX_OVERRIDE_KEY',
    'PrefixToken',
    'find_prefix_tokens',
    'strip_uuid_prefix_override',
    'substitute',
    'substitute_all',
    'uuid_prefix_override_requested',
]


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

    The ONE-ELEMENT case of :func:`substitute_all`, delegated rather than
    reimplemented, so this module holds one splice and one monotonicity check
    (SPOT). Its single-token contract is unchanged; a caller with several
    replacements should call the batch door directly rather than folding this
    one, which costs a full rebuild of the containing string per occurrence.
    """
    return substitute_all(arguments, ((token, full_id),))


def substitute_all(
    arguments: Mapping[str, Any],
    replacements: Iterable[tuple[PrefixToken, str]],
) -> dict[str, Any]:
    """Return *arguments* with every ``(token, full_id)`` span replaced at once.

    ONE rebuild per containing string, not one per occurrence. The spans in a
    string are spliced in a single ``''.join`` over alternating gap and
    replacement slices, and each DISTINCT path pays one path copy. That is the
    whole reason this door exists: the caller-side fold it replaced was
    O(occurrences x string length) and stalled the event loop on an ordinary
    large write — see the module docstring's INV-8 section for the numbers.

    ORDER-FREE. *replacements* may arrive in any order; spans are sorted per
    path here. The fold demanded REVERSE document order and silently corrupted
    the later span without it, so removing that precondition is a safety
    property of this door and not merely a convenience.

    MONOTONE for EVERY pair, validated BEFORE anything is rebuilt, so a
    refusal is all-or-nothing: a batch that refuses one pair has applied none
    of them, which is what lets a boundary guard keep "nothing was written"
    structurally true. The message is the single-token door's, because it is
    the same invariant enforced at the same door (heuristic 10).

    OVERLAPPING spans within one path are REFUSED. No detector produces them —
    ``re.finditer`` yields non-overlapping matches — but a silently corrupted
    string is a far worse failure than a loud refusal, and this door cannot
    tell a caller's bug from a detector's.

    Non-mutating, like the single-token door and for the same reason: the
    original argument map is a guard's fallback when a later token in the same
    call turns out to be ambiguous or unresolvable.
    """
    pairs = list(replacements)
    for token, full_id in pairs:
        if not full_id.startswith(token.token):
            raise ValueError(
                f'refusing a non-monotone substitution: {full_id!r} does not start with '
                f'the token it would replace, {token.token!r}. An expansion must keep the '
                f'original characters so that a wrong one stays visible and reversible.'
            )
    by_path: dict[tuple[str | int, ...], list[tuple[PrefixToken, str]]] = {}
    for token, full_id in pairs:
        by_path.setdefault(token.path, []).append((token, full_id))

    result: Any = arguments
    for path, group in by_path.items():
        trail: list[tuple[Any, str | int]] = []
        node: Any = result
        for key in path:
            trail.append((node, key))
            node = node[key]
        rebuilt: Any = _spliced(node, path, sorted(group, key=lambda pair: pair[0].start))
        for container, key in reversed(trail):
            rebuilt = _with_child(container, key, rebuilt)
        result = rebuilt
    return result


def _spliced(value: str, path: tuple[str | int, ...], group: list[tuple[PrefixToken, str]]) -> str:
    """*value* with every span in *group* (ascending, non-overlapping) replaced.

    One pass, one join: the gaps between spans are each copied exactly once,
    which is the linear bound the module docstring states.
    """
    parts: list[str] = []
    cursor = 0
    for token, full_id in group:
        if token.start < cursor:
            raise ValueError(
                f'refusing a batch with overlapping spans at {path!r}: '
                f'{token.token!r} starts at {token.start} inside a replacement that '
                f'already covers up to {cursor}. Splicing them would corrupt the '
                f'string rather than expand it.'
            )
        parts.append(value[cursor : token.start])
        parts.append(full_id)
        cursor = token.end
    parts.append(value[cursor:])
    return ''.join(parts)


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


#: Declared opt-out for a deliberate bare prefix — a correction record quoting
#: a bad citation verbatim, say.
#:
#: It lives in the DETECTOR module rather than the guard, which is the exact
#: placement of ``MARKUP_OVERRIDE_KEY`` in ``shared/toolcall_markup.py`` and
#: matters for the same reason: the flag has ONE lifecycle across TWO layers.
#: The boundary guard honours it, and leaf γ's ``fused_memory/server/tools.py``
#: must additionally strip it at its own write-time layer, exactly as
#: ``allow_mcp_markup`` is stripped at both. Putting these three names in the
#: guard would force a tool body to import a middleware module — and fastmcp
#: with it — to reach a string constant and a five-line stripper.
UUID_PREFIX_OVERRIDE_KEY = 'allow_uuid_prefix'


def _as_metadata_dict(metadata: object) -> dict[str, Any] | None:
    """Best-effort read of *metadata* as a dict, else ``None``.

    ``submit_task``/``update_task`` accept metadata as an object OR a JSON
    string, so both shapes are understood. Anything unparseable — malformed
    JSON, a non-dict JSON payload, a wrong type entirely — yields ``None``
    without raising: validating metadata is not this module's job, and a write
    must never fail because an override helper choked on a field it does not
    own.
    """
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def uuid_prefix_override_requested(metadata: object) -> bool:
    """Return True iff *metadata* carries an explicit opt-in.

    FAIL-CLOSED: only a literal boolean ``True`` counts, mirroring
    ``markup_override_requested`` and add_memory's ``allow_near_duplicate``
    check. A truthy-but-not-``True`` value (``'yes'``, ``1``) is far more
    likely to be unrelated data than a considered decision to keep a bare
    prefix out of the guard's reach — and failing closed costs the author one
    resubmit, where failing open costs a silently unexpanded citation.

    Never raises, for any input.
    """
    parsed = _as_metadata_dict(metadata)
    if parsed is None:
        return False
    return parsed.get(UUID_PREFIX_OVERRIDE_KEY) is True


def strip_uuid_prefix_override(metadata: Any) -> Any:
    """Return *metadata* without :data:`UUID_PREFIX_OVERRIDE_KEY`, in the same shape.

    The override is a call-time-only control flag: it must never be persisted
    into stored memory metadata or the task metadata vocabulary. Returning the
    shape it was given (dict in / dict out, JSON string in / JSON string out)
    lets a call site substitute the result inline before forwarding downstream.

    NON-mutating — the caller's own dict is left intact, since the handler may
    still need the original and quietly mutating caller-owned metadata is
    action-at-a-distance a guard should not introduce.

    Unparseable input passes straight through unchanged, never raising.
    """
    if isinstance(metadata, dict):
        if UUID_PREFIX_OVERRIDE_KEY not in metadata:
            return metadata
        return {k: v for k, v in metadata.items() if k != UUID_PREFIX_OVERRIDE_KEY}
    if isinstance(metadata, str):
        parsed = _as_metadata_dict(metadata)
        if parsed is None or UUID_PREFIX_OVERRIDE_KEY not in parsed:
            return metadata
        return json.dumps({k: v for k, v in parsed.items() if k != UUID_PREFIX_OVERRIDE_KEY})
    return metadata
