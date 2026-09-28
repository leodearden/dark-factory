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
import logging
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, NamedTuple

from fused_memory.middleware.recurring_gate_guard import is_gate_submission
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.scope import resolve_project_id_for_root
from fused_memory.reconciliation.citation_verifier import (
    LivenessVerdict,
    MemoryLiveness,
    resolve_memory_liveness,
)
from fused_memory.utils.validation import is_full_uuid, validate_full_uuid

logger = logging.getLogger(__name__)

X_ENUMERATION_SCOPE_KEY = 'x_enumeration_scope'
X_LIVENESS_REVERIFIED_AT_KEY = 'x_liveness_reverified_at'
X_LIVENESS_EXCEPTIONS_KEY = 'x_liveness_exceptions'

ENUMERATED_ID_TRUNCATED = 'EnumeratedIdTruncated'
ENUMERATION_SCOPE_MISSING = 'EnumerationScopeMissing'
ENUMERATION_SCOPE_MALFORMED = 'EnumerationScopeMalformed'

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


def enumeration_scope_example() -> dict[str, Any]:
    """A well-formed ``x_enumeration_scope``; ``note`` is the one optional field."""
    return {
        'stores': [SourceStore.mem0.value],
        'categories': [MemoryCategory.procedural_knowledge.value],
        'exhaustive': False,
        'note': 'how the sweep was bounded, e.g. one search per category',
    }


def _vocabulary_problem(value: Any, vocabulary: type[StrEnum]) -> str | None:
    allowed = [member.value for member in vocabulary]
    if not isinstance(value, list) or not value:
        return f'must be a non-empty list drawn from {allowed}'
    unknown = [item for item in value if item not in allowed]
    if unknown:
        return f'unknown value(s) {unknown!r}; allowed: {allowed}'
    return None


def _scope_problems(scope: Any) -> list[dict[str, str]]:
    """Every problem with a present scope, per field — never just the first."""
    if not isinstance(scope, dict):
        return [{
            'field': X_ENUMERATION_SCOPE_KEY,
            'problem': f'must be an object, got {type(scope).__name__}',
        }]
    problems = {
        'stores': _vocabulary_problem(scope.get('stores'), SourceStore),
        'categories': _vocabulary_problem(scope.get('categories'), MemoryCategory),
        'exhaustive': (
            None if isinstance(scope.get('exhaustive'), bool) else 'must be a bool'
        ),
        'note': (
            None if isinstance(scope.get('note', ''), str)
            else 'must be a string when present'
        ),
    }
    return [
        {'field': field, 'problem': problem}
        for field, problem in problems.items()
        if problem is not None
    ]


def _malformed_scope_rejection(problems: list[dict[str, str]]) -> dict[str, Any]:
    return {
        'error': f'{X_ENUMERATION_SCOPE_KEY} is malformed: ' + '; '.join(
            f"{problem['field']} {problem['problem']}" for problem in problems
        ),
        'error_type': ENUMERATION_SCOPE_MALFORMED,
        'problems': problems,
        'expected_shape': enumeration_scope_example(),
        'hint': (
            'State the searched stores and categories and whether the sweep was '
            'exhaustive as structured fields; a prose scope cannot be checked.'
        ),
    }


def _missing_scope_rejection(enumerations: list[str]) -> dict[str, Any]:
    return {
        'error': (
            f'a gate enumerating Mem0 entries in {enumerations} cannot be filed '
            f'without {X_ENUMERATION_SCOPE_KEY}'
        ),
        'error_type': ENUMERATION_SCOPE_MISSING,
        'enumerations': enumerations,
        'expected_shape': enumeration_scope_example(),
        'hint': (
            'Say which stores and categories the enumeration was drawn from and '
            'whether that sweep was exhaustive, so the gate never claims "these '
            'are the N entries" unqualified.'
        ),
    }


def _truncated_record(path: str, value: str) -> dict[str, str]:
    return {'memory_id': value, 'path': path, 'outcome': _TRUNCATED_OUTCOME}


class _LivenessPass(NamedTuple):
    """The proven Mem0 enumerations, rebuilt without their absent members,
    plus the exception records and counters of the pass that proved them."""

    enumerations: dict[str, list[Any]]
    records: list[dict[str, str]]
    stats: dict[str, int]


def _candidate_lists(parsed: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        key: value
        for key, value in parsed.items()
        if isinstance(value, list) and any(is_full_uuid(member) for member in value)
    }


async def _resolve_members(
    candidates: dict[str, list[Any]],
    memory_service: Any,
    project_root: str,
) -> dict[str, LivenessVerdict]:
    """One verdict per DISTINCT candidate id, memoised for this call only: a
    longer-lived cache would reintroduce the stale read this gate closes."""
    distinct = dict.fromkeys(
        member
        for members in candidates.values()
        for member in members
        if is_full_uuid(member)
    )
    if not distinct:
        return {}
    project_id = resolve_project_id_for_root(project_root)
    verdicts = {
        memory_id: await resolve_memory_liveness(memory_service, project_id, memory_id)
        for memory_id in distinct
    }
    inconclusive = {
        memory_id: verdict.error_type
        for memory_id, verdict in verdicts.items()
        if verdict.outcome is MemoryLiveness.INCONCLUSIVE
    }
    if inconclusive:
        logger.warning(
            'enumeration liveness inconclusive for %d id(s), kept unverified: %s',
            len(inconclusive), inconclusive,
        )
    return verdicts


def _member_verdict(
    member: Any, verdicts: dict[str, LivenessVerdict],
) -> LivenessVerdict | None:
    return verdicts.get(member) if is_full_uuid(member) else None


def _liveness_record(path: str, memory_id: str, verdict: LivenessVerdict) -> dict[str, str]:
    record = {'memory_id': memory_id, 'path': path, 'outcome': verdict.outcome.value}
    if verdict.error_type is not None:
        record['error_type'] = verdict.error_type
    return record


def _rebuild_enumeration(
    key: str, members: list[Any], verdicts: dict[str, LivenessVerdict],
) -> tuple[list[Any], list[dict[str, str]]]:
    """The list without its ABSENT members, and a record for every member that
    is not LIVE. An INCONCLUSIVE member stays in place: unknown is not absent."""
    kept: list[Any] = []
    records: list[dict[str, str]] = []
    for index, member in enumerate(members):
        verdict = _member_verdict(member, verdicts)
        if verdict is not None and verdict.outcome is not MemoryLiveness.LIVE:
            records.append(_liveness_record(f'{key}[{index}]', member, verdict))
        if verdict is None or verdict.outcome is not MemoryLiveness.ABSENT:
            kept.append(member)
    return kept, records


def _is_live(member: Any, verdicts: dict[str, LivenessVerdict]) -> bool:
    verdict = _member_verdict(member, verdicts)
    return verdict is not None and verdict.outcome is MemoryLiveness.LIVE


async def _liveness_pass(
    parsed: dict[str, Any], memory_service: Any, project_root: str,
) -> _LivenessPass:
    candidates = _candidate_lists(parsed)
    verdicts = await _resolve_members(candidates, memory_service, project_root)
    proven = {
        key: members for key, members in candidates.items()
        if any(_is_live(member, verdicts) for member in members)
    }
    enumerations: dict[str, list[Any]] = {}
    records: list[dict[str, str]] = []
    verified = 0
    for key, members in proven.items():
        enumerations[key], list_records = _rebuild_enumeration(key, members, verdicts)
        records.extend(list_records)
        verified += sum(_is_live(member, verdicts) for member in members)
    outcomes = [record['outcome'] for record in records]
    stats = {
        'mem0_enumerations': len(proven),
        'unproven_lists': len(candidates) - len(proven),
        'verified': verified,
        'dropped_absent': outcomes.count(MemoryLiveness.ABSENT.value),
        'inconclusive': outcomes.count(MemoryLiveness.INCONCLUSIVE.value),
    }
    return _LivenessPass(enumerations, records, stats)


def _mention_pattern(memory_id: str) -> re.Pattern[str]:
    """The full id, or its 8-char prefix as a whole word — but not as the head
    of a DIFFERENT dashed id that merely shares the prefix."""
    return re.compile(
        rf'(?<![0-9a-f]){re.escape(memory_id)}(?![0-9a-f])'
        rf'|(?<![0-9a-f]){re.escape(memory_id[:8])}(?![0-9a-f]|-[0-9a-f])',
        re.IGNORECASE,
    )


def _annotation_marker(memory_id: str) -> str:
    return f'[{memory_id[:8]}: '


def _annotation(record: dict[str, str], verified_at: str) -> str:
    if record['outcome'] == MemoryLiveness.ABSENT.value:
        note = f'already deleted, verified {verified_at}'
    else:
        note = f'liveness inconclusive ({record.get("error_type")}), checked {verified_at}'
    return f'{_annotation_marker(record["memory_id"])}{note}]'


def _annotate_description(
    description: str | None, records: list[dict[str, str]], verified_at: str,
) -> tuple[str | None, int]:
    """Append one annotation per (line, id) for each line naming a recorded id;
    a line already carrying that id's marker is skipped, so re-running is a
    no-op. Returns the caller's own string when no line changed."""
    if description is None or not records:
        return description, 0
    annotations: dict[str, str] = {}
    for record in records:
        annotations.setdefault(record['memory_id'], _annotation(record, verified_at))
    mentions = {memory_id: _mention_pattern(memory_id) for memory_id in annotations}
    lines = description.split('\n')
    annotated = 0
    for index, line in enumerate(lines):
        additions = [
            annotation
            for memory_id, annotation in annotations.items()
            if mentions[memory_id].search(line)
            and _annotation_marker(memory_id) not in line
        ]
        if additions:
            lines[index] = ' '.join([line, *additions])
            annotated += 1
    return ('\n'.join(lines) if annotated else description), annotated


def _with_gate_owned_keys(
    metadata: dict[str, Any],
    records: list[dict[str, str]],
    *,
    proven: bool,
    verified_at: str,
) -> dict[str, Any]:
    """Only this write's own evidence sets the two liveness keys; an incoming
    copy never survives, so a copied-forward stamp cannot pose as fresh."""
    rewritten = {
        key: value
        for key, value in metadata.items()
        if key not in (X_LIVENESS_REVERIFIED_AT_KEY, X_LIVENESS_EXCEPTIONS_KEY)
    }
    if proven:
        rewritten[X_LIVENESS_REVERIFIED_AT_KEY] = verified_at
    if records:
        rewritten[X_LIVENESS_EXCEPTIONS_KEY] = records
    return rewritten


def _rejection_before_lookup(
    parsed: dict[str, Any], fragments: list[tuple[str, str]], *, is_filing: bool,
) -> dict[str, Any] | None:
    if fragments and is_filing:
        return _truncated_rejection(fragments)
    if X_ENUMERATION_SCOPE_KEY not in parsed:
        return None
    problems = _scope_problems(parsed[X_ENUMERATION_SCOPE_KEY])
    return _malformed_scope_rejection(problems) if problems else None


def _scope_required(parsed: dict[str, Any], liveness: _LivenessPass) -> bool:
    return (
        bool(liveness.enumerations)
        and X_ENUMERATION_SCOPE_KEY not in parsed
        and is_gate_submission(parsed)
    )


def _rewrite(
    passthrough: EnumerationGateResult,
    parsed: dict[str, Any],
    fragments: list[tuple[str, str]],
    liveness: _LivenessPass,
    *,
    verified_at: str,
) -> EnumerationGateResult:
    description, annotated_lines = _annotate_description(
        passthrough.description, liveness.records, verified_at,
    )
    records = [_truncated_record(path, value) for path, value in fragments]
    records.extend(liveness.records)
    stats = dict.fromkeys(_STAT_KEYS, 0) | liveness.stats | {
        'truncated': len(fragments),
        'annotated_lines': annotated_lines,
    }
    reported = stats if any(stats.values()) else None
    metadata = _with_gate_owned_keys(
        parsed | liveness.enumerations,
        records,
        proven=bool(liveness.enumerations),
        verified_at=verified_at,
    )
    if metadata == parsed and description is passthrough.description:
        return passthrough._replace(stats=reported)
    return EnumerationGateResult(None, description, metadata, reported)


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
    rejection = _rejection_before_lookup(parsed, fragments, is_filing=is_filing)
    if rejection is not None:
        return passthrough._replace(rejection=rejection)

    liveness = await _liveness_pass(parsed, memory_service, project_root)
    if is_filing and _scope_required(parsed, liveness):
        return passthrough._replace(
            rejection=_missing_scope_rejection(list(liveness.enumerations)),
        )
    return _rewrite(
        passthrough, parsed, fragments, liveness, verified_at=now().isoformat(),
    )
