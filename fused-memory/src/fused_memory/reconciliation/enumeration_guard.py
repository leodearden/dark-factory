"""Write-time liveness gate for memory ids a recon stage enumerates (task 3056).

**The invariant this module owns.** An enumerated memory id relayed into a
recon-stage task write is shape-valid and re-resolved AT WRITE TIME, never
copied forward from Stage-1 analysis time. ``citation_verifier`` owns the same
invariant for a finding's ``cited_memories``; this is its Stage-2 half, for the
id lists a recon agent writes into ``submit_task`` / ``update_task`` metadata.
Both halves share one verdict, ``citation_verifier.resolve_memory_liveness``.

**Provenance.** Reify gate 5537 was filed by Stage 2 from Stage-1 finding
f11018d4 enumerating four Mem0 entries. One of them, 0d542614, had been deleted
about eleven minutes before the gate shipped, and the gate nominated that very
row as the likeliest deletion target. The same list carried ``'21f78de1'``, an
8-char fragment no tool resolves. Reify gate 5533 listed its ten legacy entries
as bare 8-char fragments with no full id anywhere beside them.

**The rules, cheapest first** (no lookup runs before a rejection needing none):

1. *Shape.* A metadata LIST element, at any depth, that is a UUID-prefix
   fragment is rejected when filing and recorded as a ``truncated`` exception
   when updating: for ``update_task`` losing the write is worse than the defect
   (decision D3 of ``plans/uuid-prefix-resolution-prd.md``). Measured across
   both task stores, 70 of 70 list elements that are exactly 8 hex chars are
   truncated UUIDs, while commit-SHA lists are 10 or 40 chars. 8-char commit
   SHAs DO appear as scalars and in prose, so scalars, dict keys, prose and
   the description are never shape-checked.
2. *Scope shape.* A present ``x_enumeration_scope`` must be the structured
   ``{stores, categories, exhaustive, note?}`` object, on every path.
3. *Liveness.* Each canonical-UUID string element of a TOP-LEVEL list is
   resolved once. A list is a Mem0 enumeration iff at least one member is
   LIVE. A canonical UUID is not necessarily a Mem0 id (recon metadata carries
   finding, run and Graphiti id lists, and gate 5537 deliberately records a
   deleted id), and ``get_memory_by_id`` is a Mem0-only point read, so a miss
   proves absence only beside a live sibling. Key names cannot type a list:
   recon gates use 17+ spellings for memory lists. Inside an enumeration,
   ABSENT members are dropped and INCONCLUSIVE ones kept; both are recorded.
   Nested lists belong to records another module builds (a consolidation
   gate's provenance, a predicate task's cited ids) and are never looked up.
4. *Scope required.* Filing a gate that carries an enumeration without
   ``x_enumeration_scope`` is rejected: a gate cannot be filed with an
   unqualified "these are the N entries" claim.
5. *Stamp.* ``x_liveness_reverified_at`` and ``x_liveness_exceptions`` are
   gate-owned: set from this write's own evidence, incoming copies overwritten
   or stripped, so a copied-forward stamp never poses as a fresh verification.
   Description lines naming an absent or inconclusive id are annotated.

**Scope of the gate.** Recon-stage callers only: the defect is a property of
the writer's provenance, and interactive writes legitimately cite deleted ids
as incident evidence. It filters the OUTGOING payload only and never reads or
repairs stored task state. Named residuals: a list mixing memory ids with
non-memory UUIDs can lose its non-memory members; a Mem0 list whose members
are all deleted is not proven and stays unstamped; an enumeration carried only
in description prose is not checked, which is why the rendered Stage-2 section
demands top-level metadata lists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any, NamedTuple

from fused_memory.utils.validation import validate_full_uuid

X_ENUMERATION_SCOPE_KEY = 'x_enumeration_scope'
X_LIVENESS_REVERIFIED_AT_KEY = 'x_liveness_reverified_at'
X_LIVENESS_EXCEPTIONS_KEY = 'x_liveness_exceptions'

ENUMERATED_ID_TRUNCATED = 'EnumeratedIdTruncated'

_TRUNCATED_OUTCOME = 'truncated'

_STAT_KEYS: tuple[str, ...] = (
    'mem0_enumerations',
    'unproven_lists',
    'verified',
    'dropped_absent',
    'inconclusive',
    'truncated',
    'annotated_lines',
)


class EnumerationGateResult(NamedTuple):
    """``rejection`` is a structured error dict to return to the caller, or
    None; ``description`` and ``metadata`` are what to write (the caller's own
    objects when nothing changed); ``stats`` is None when nothing was checked."""

    rejection: dict[str, Any] | None
    description: str | None
    metadata: Any
    stats: dict[str, int] | None


def _strict_uuid_prefix_pattern() -> re.Pattern[str]:
    """Every strict prefix of the canonical 8-4-4-4-12 layout that keeps its
    whole first group: ``21f78de1``, ``21f78de1-a5f3``, ``21f78de1-a5f3-42``...
    A full UUID never matches; ``utils.validation.is_full_uuid`` owns that shape."""
    rest = '[0-9a-f]{0,11}'
    for _ in range(3):
        rest = f'(?:[0-9a-f]{{4}}(?:-{rest})?|[0-9a-f]{{0,3}})'
    return re.compile(f'[0-9a-f]{{8}}(?:-{rest})?', re.IGNORECASE)


_UUID_FRAGMENT = _strict_uuid_prefix_pattern()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_recon_stage(agent_id: Any) -> bool:
    return isinstance(agent_id, str) and agent_id.startswith('recon-stage-')


def _metadata_as_dict(metadata: Any) -> dict[str, Any] | None:
    """The metadata as a dict, or None when it is not one — never ``{}``, so the
    gate can hand an unusable blob back to the caller untouched."""
    if isinstance(metadata, dict):
        return metadata
    if not isinstance(metadata, str) or not metadata:
        return None
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _list_element_fragments(node: Any, path: str = '') -> Iterator[tuple[str, str]]:
    """``(path, value)`` for each list element, at any depth, that is a UUID
    fragment. Paths use ``citation_verifier.find_citation_occurrences``'
    vocabulary (``key.child[0]``)."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _list_element_fragments(value, f'{path}.{key}' if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            element_path = f'{path}[{index}]'
            if isinstance(value, str) and _UUID_FRAGMENT.fullmatch(value):
                yield element_path, value
            else:
                yield from _list_element_fragments(value, element_path)


def _truncated_rejection(fragments: list[tuple[str, str]]) -> dict[str, Any]:
    problems = [
        problem
        for path, value in fragments
        if (problem := validate_full_uuid(value, field_name=path)) is not None
    ]
    return {
        'error': (
            'recon-stage metadata enumerates memory ids by truncated fragment: '
            + '; '.join(problem['error'] for problem in problems)
        ),
        'error_type': ENUMERATED_ID_TRUNCATED,
        'offenders': [{'path': path, 'value': value} for path, value in fragments],
        'hint': problems[0]['hint'],
    }


def _truncated_record(path: str, value: str) -> dict[str, str]:
    return {'memory_id': value, 'path': path, 'outcome': _TRUNCATED_OUTCOME}


async def enumeration_liveness_gate(
    *,
    agent_id: str | None,
    description: str | None,
    metadata: Any,
    memory_service: Any,
    project_root: str,
    is_filing: bool,
    now: Callable[[], datetime] = _utc_now,
) -> EnumerationGateResult:
    """Shape-check, re-resolve and stamp the memory ids a recon write enumerates.

    ``is_filing`` is True for ``submit_task`` and False for ``update_task``;
    the module docstring states which rules differ between the two. Never
    raises: a failed lookup is an inconclusive verdict, not an error.
    """
    passthrough = EnumerationGateResult(None, description, metadata, None)
    if not _is_recon_stage(agent_id):
        return passthrough
    parsed = _metadata_as_dict(metadata)
    if parsed is None:
        return passthrough

    fragments = list(_list_element_fragments(parsed))
    if fragments and is_filing:
        return passthrough._replace(rejection=_truncated_rejection(fragments))

    records = [_truncated_record(path, value) for path, value in fragments]
    stats = dict.fromkeys(_STAT_KEYS, 0)
    stats['truncated'] = len(records)
    if not any(stats.values()):
        return passthrough

    rewritten = dict(parsed)
    rewritten[X_LIVENESS_EXCEPTIONS_KEY] = records
    return EnumerationGateResult(None, description, rewritten, stats)
