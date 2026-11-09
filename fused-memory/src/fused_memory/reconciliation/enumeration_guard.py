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
   the description are never shape-checked. Two of those 70 are all-decimal,
   so digits-only elements are not exempt.
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
   Reads are bounded because this is a write path: a few in flight at once,
   a per-write budget, and a stop once reads keep failing. An id left
   unread is INCONCLUSIVE with ``error_type`` ``LIVENESS_LOOKUP_SKIPPED``.
4. *Overwriting update.* An update carrying an enumeration must be one whose
   metadata merge lets every incoming top-level key replace the stored one.
   The additive union keeps every stored member and, on a scalar collision,
   the stored value, so neither a drop nor the fresh stamp would land.
5. *Scope required.* Filing a gate that carries an enumeration without
   ``x_enumeration_scope`` is rejected: a gate cannot be filed with an
   unqualified "these are the N entries" claim. A gate is any filing
   ``operational_routing_guard.inject_operational_routing`` stores as
   always-escalating, so ``decision`` filings count as well as
   ``operational`` ones.
6. *Stamp.* ``x_liveness_reverified_at`` and ``x_liveness_exceptions`` are
   gate-owned: set from this write's own evidence, incoming copies overwritten
   or stripped, so a copied-forward stamp never poses as a fresh verification.
   One exception keeps a removal visible (INV-11): ``replace`` swaps in the
   whole blob, so under it an incoming exception record for an id this write
   did not observe is carried forward. A record's ``outcome`` is an
   ``EnumeratedIdOutcome``. Description lines naming an absent or
   inconclusive id are annotated, and an annotation whose outcome a later
   write changed is rewritten in place.

**Scope of the gate.** Recon-stage callers only: the defect is a property of
the writer's provenance, and interactive writes legitimately cite deleted ids
as incident evidence. It filters the OUTGOING payload only and never reads or
repairs stored task state, which is why it refuses an update whose merge would
not let that payload land. Named residuals: a list mixing memory ids with
non-memory UUIDs can lose its non-memory members; a Mem0 list whose members
are all deleted is not proven and stays unstamped; an enumeration carried only
in description prose is not checked, which is why the rendered Stage-2 section
demands top-level metadata lists; under ``merge`` a stored
``x_liveness_exceptions`` from an earlier write survives a later write that
records nothing but is overwritten by one that records something; a list whose
live members all fall past the lookup budget stays unproven; any 8-hex list
element reads as a fragment, so a list of ``YYYYMMDD`` dates or a hex word
such as ``deadbeef`` is rejected on filing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, NamedTuple

from fused_memory.middleware.operational_routing_guard import inject_operational_routing
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
ENUMERATION_UPDATE_NOT_OVERWRITING = 'EnumerationUpdateNotOverwriting'

MAX_ENUMERATED_IDS_PER_WRITE = 100
MAX_CONCURRENT_LIVENESS_READS = 8
MAX_CONSECUTIVE_LIVENESS_READ_ERRORS = 3
LIVENESS_LOOKUP_SKIPPED = 'LivenessLookupSkipped'

_REPLACE_MODE = 'replace'
_OVERWRITING_MODES: tuple[str, ...] = ('merge', _REPLACE_MODE)


class EnumeratedIdOutcome(StrEnum):
    """The ``outcome`` of one ``x_liveness_exceptions`` record: why an
    enumerated id did not verify LIVE at write time."""

    ABSENT = 'absent'
    INCONCLUSIVE = 'inconclusive'
    TRUNCATED = 'truncated'


_OUTCOME_OF_VERDICT = {
    MemoryLiveness.ABSENT: EnumeratedIdOutcome.ABSENT,
    MemoryLiveness.INCONCLUSIVE: EnumeratedIdOutcome.INCONCLUSIVE,
}
_OUTCOME_VALUES = frozenset(outcome.value for outcome in EnumeratedIdOutcome)

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


def _overwrites_top_level_keys(metadata_mode: str | None, append: bool | None) -> bool:
    """Whether the backend's merge lets every incoming top-level key replace
    the stored one. The authority is
    ``fused-memory/src/fused_memory/backends/sqlite_task_backend.py::_resolve_metadata_mode``;
    this is a fail-closed allowlist of its overwriting outcomes, so a mode the
    backend adds later is classified non-overwriting and rejected loudly, never
    silently misreported as landed. ``('merge', True)`` passes here only
    because the backend refuses that pair itself, so nothing lands."""
    if metadata_mode in _OVERWRITING_MODES:
        return True
    return metadata_mode is None and append is None


def _not_overwriting_rejection(
    enumerations: list[str], metadata_mode: str | None, append: bool | None,
) -> dict[str, Any]:
    return {
        'error': (
            f'an update re-asserting the Mem0 enumeration(s) {enumerations} '
            f'under metadata_mode={metadata_mode!r}, append={append!r} does not '
            f'resolve to one of {list(_OVERWRITING_MODES)}, the only merges under '
            "which the gate's drops and fresh stamp replace the stored list and stamp"
        ),
        'error_type': ENUMERATION_UPDATE_NOT_OVERWRITING,
        'enumerations': enumerations,
        'metadata_mode': metadata_mode,
        'append': append,
        'accepted_modes': list(_OVERWRITING_MODES),
        'hint': (
            'Read the task with get_task, then re-send the COMPLETE enumeration '
            "with metadata_mode='merge' (the default when both flags are "
            "omitted). 'merge' replaces each supplied top-level key wholesale, "
            'so a list holding only the ids being added erases every other '
            "stored member. 'replace' swaps in the whole metadata blob, so it "
            'must carry every stored key. Attach memory_hints in a separate '
            'append=True update.'
        ),
    }


def _truncated_record(path: str, value: str) -> dict[str, str]:
    return {'memory_id': value, 'path': path, 'outcome': EnumeratedIdOutcome.TRUNCATED.value}


class _LivenessPass(NamedTuple):
    """The proven Mem0 enumerations, rebuilt without their absent members,
    plus the exception records and counters of the pass that proved them.
    ``observed`` is every member id of those enumerations, lowercased."""

    enumerations: dict[str, list[Any]]
    records: list[dict[str, str]]
    stats: dict[str, int]
    observed: frozenset[str]


def _candidate_lists(parsed: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        key: value
        for key, value in parsed.items()
        if isinstance(value, list) and any(is_full_uuid(member) for member in value)
    }


_SKIPPED = LivenessVerdict(MemoryLiveness.INCONCLUSIVE, LIVENESS_LOOKUP_SKIPPED)


async def _bounded_verdicts(
    memory_ids: list[str], memory_service: Any, project_id: str,
) -> list[LivenessVerdict]:
    """One verdict per id, in order, with at most
    ``MAX_CONCURRENT_LIVENESS_READS`` point reads in flight. This runs on a
    write path, and on a sick store every further read only waits out its own
    timeout, so an id is never read, and answers INCONCLUSIVE, when it falls
    past ``MAX_ENUMERATED_IDS_PER_WRITE`` or its read would start after
    ``MAX_CONSECUTIVE_LIVENESS_READ_ERRORS`` reads in a row came back
    inconclusive. The primitive never raises, so no read aborts the rest."""
    in_flight = asyncio.Semaphore(MAX_CONCURRENT_LIVENESS_READS)
    consecutive_errors = 0

    async def _read(memory_id: str) -> LivenessVerdict:
        nonlocal consecutive_errors
        async with in_flight:
            if consecutive_errors >= MAX_CONSECUTIVE_LIVENESS_READ_ERRORS:
                return _SKIPPED
            verdict = await resolve_memory_liveness(memory_service, project_id, memory_id)
            failed = verdict.outcome is MemoryLiveness.INCONCLUSIVE
            consecutive_errors = consecutive_errors + 1 if failed else 0
            return verdict

    checked = memory_ids[:MAX_ENUMERATED_IDS_PER_WRITE]
    verdicts = await asyncio.gather(*(_read(memory_id) for memory_id in checked))
    skipped = len(memory_ids) - len(checked)
    if skipped:
        logger.warning(
            'enumeration liveness: %d id(s) past the %d-lookup budget left unchecked',
            skipped, MAX_ENUMERATED_IDS_PER_WRITE,
        )
    return [*verdicts, *[_SKIPPED] * skipped]


async def _resolve_members(
    candidates: dict[str, list[Any]],
    memory_service: Any,
    project_root: str,
) -> dict[str, LivenessVerdict]:
    """One verdict per DISTINCT candidate id, memoised for this call only: a
    longer-lived cache would reintroduce the stale read this gate closes."""
    distinct = list(dict.fromkeys(
        member
        for members in candidates.values()
        for member in members
        if is_full_uuid(member)
    ))
    if not distinct:
        return {}
    project_id = await asyncio.to_thread(resolve_project_id_for_root, project_root)
    resolved = await _bounded_verdicts(distinct, memory_service, project_id)
    verdicts = dict(zip(distinct, resolved, strict=True))
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
    record = {
        'memory_id': memory_id,
        'path': path,
        'outcome': _OUTCOME_OF_VERDICT[verdict.outcome].value,
    }
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
    outcomes = Counter(record['outcome'] for record in records)
    stats = {
        'mem0_enumerations': len(proven),
        'unproven_lists': len(candidates) - len(proven),
        'verified': verified,
        'dropped_absent': outcomes[EnumeratedIdOutcome.ABSENT],
        'inconclusive': outcomes[EnumeratedIdOutcome.INCONCLUSIVE],
    }
    observed = frozenset(
        member.lower()
        for members in proven.values()
        for member in members
        if is_full_uuid(member)
    )
    return _LivenessPass(enumerations, records, stats, observed)


def _mention_pattern(memory_id: str) -> re.Pattern[str]:
    """The full id, or its 8-char prefix as a whole word — but not as the head
    of a DIFFERENT dashed id that merely shares the prefix."""
    return re.compile(
        rf'(?<![0-9a-f]){re.escape(memory_id)}(?![0-9a-f])'
        rf'|(?<![0-9a-f]){re.escape(memory_id[:8])}(?![0-9a-f]|-[0-9a-f])',
        re.IGNORECASE,
    )


_ANNOTATION_PHRASE = {
    EnumeratedIdOutcome.ABSENT: 'already deleted',
    EnumeratedIdOutcome.INCONCLUSIVE: 'liveness inconclusive',
}


def _annotation_marker(record: dict[str, str]) -> str:
    """Names the id AND the outcome, so a line annotated under one outcome is
    rewritten, never skipped, once a later write proves another."""
    phrase = _ANNOTATION_PHRASE[EnumeratedIdOutcome(record['outcome'])]
    return f'[{record["memory_id"][:8]}: {phrase}'


def _annotation(record: dict[str, str], verified_at: str) -> str:
    if record['outcome'] == EnumeratedIdOutcome.ABSENT:
        detail = f', verified {verified_at}'
    else:
        detail = f' ({record.get("error_type")}), checked {verified_at}'
    return f'{_annotation_marker(record)}{detail}]'


def _any_annotation_pattern(memory_id: str) -> re.Pattern[str]:
    return re.compile(rf'\[{re.escape(memory_id[:8])}: [^\]]*\]', re.IGNORECASE)


def _with_annotation(line: str, record: dict[str, str], verified_at: str) -> str:
    """*line* carrying this record's annotation: unchanged when it already
    states this outcome, rewritten in place when it states another, and
    appended otherwise."""
    if _annotation_marker(record) in line:
        return line
    annotation = _annotation(record, verified_at)
    rewritten, replaced = _any_annotation_pattern(record['memory_id']).subn(
        lambda _stale: annotation, line, count=1,
    )
    return rewritten if replaced else f'{line} {annotation}'


def _annotate_description(
    description: str | None, records: list[dict[str, str]], verified_at: str,
) -> tuple[str | None, int]:
    """Annotate each line naming a recorded id with that id's outcome; re-running
    with the same outcome is a no-op. Returns the caller's own string when no
    line changed."""
    if description is None or not records:
        return description, 0
    by_id: dict[str, dict[str, str]] = {}
    for record in records:
        by_id.setdefault(record['memory_id'], record)
    mentions = {memory_id: _mention_pattern(memory_id) for memory_id in by_id}
    lines = description.split('\n')
    annotated = 0
    for index, line in enumerate(lines):
        rewritten = line
        for memory_id, record in by_id.items():
            if mentions[memory_id].search(line):
                rewritten = _with_annotation(rewritten, record, verified_at)
        if rewritten != line:
            lines[index] = rewritten
            annotated += 1
    return ('\n'.join(lines) if annotated else description), annotated


def _is_exception_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    memory_id, outcome = record.get('memory_id'), record.get('outcome')
    return (
        isinstance(memory_id, str)
        and isinstance(outcome, str)
        and outcome in _OUTCOME_VALUES
    )


def _carried_records(
    incoming: Any, observed: frozenset[str],
) -> list[dict[str, Any]]:
    """The well-formed incoming exception records for ids this write did not
    observe; this write's own record supersedes any for an id it did."""
    if not isinstance(incoming, list):
        return []
    return [
        record
        for record in incoming
        if _is_exception_record(record) and record['memory_id'].lower() not in observed
    ]


def _with_gate_owned_keys(
    metadata: dict[str, Any],
    records: list[dict[str, str]],
    *,
    proven: bool,
    verified_at: str,
    carried: list[dict[str, Any]],
) -> dict[str, Any]:
    """Only this write's own evidence sets the stamp, so a copied-forward stamp
    cannot pose as fresh. The exceptions are this write's records after any
    *carried* earlier ones."""
    rewritten = {
        key: value
        for key, value in metadata.items()
        if key not in (X_LIVENESS_REVERIFIED_AT_KEY, X_LIVENESS_EXCEPTIONS_KEY)
    }
    if proven:
        rewritten[X_LIVENESS_REVERIFIED_AT_KEY] = verified_at
    if carried or records:
        rewritten[X_LIVENESS_EXCEPTIONS_KEY] = [*carried, *records]
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


def _files_an_escalating_gate(parsed: dict[str, Any]) -> bool:
    """Judged on the metadata the filing is stored as: the operational
    routing coerces an ``operational`` or ``decision`` declaration into an
    always-escalating gate, and is idempotent on one already coerced."""
    routed = inject_operational_routing(parsed)
    return isinstance(routed, dict) and routed.get('always_escalates') is True


def _scope_required(parsed: dict[str, Any], liveness: _LivenessPass) -> bool:
    return (
        bool(liveness.enumerations)
        and X_ENUMERATION_SCOPE_KEY not in parsed
        and _files_an_escalating_gate(parsed)
    )


def _rewrite(
    passthrough: EnumerationGateResult,
    parsed: dict[str, Any],
    fragments: list[tuple[str, str]],
    liveness: _LivenessPass,
    *,
    verified_at: str,
    metadata_mode: str | None,
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
    carried = (
        _carried_records(
            parsed.get(X_LIVENESS_EXCEPTIONS_KEY),
            liveness.observed | {value.lower() for _path, value in fragments},
        )
        if metadata_mode == _REPLACE_MODE else []
    )
    metadata = _with_gate_owned_keys(
        parsed | liveness.enumerations,
        records,
        proven=bool(liveness.enumerations),
        verified_at=verified_at,
        carried=carried,
    )
    if metadata == parsed and description is passthrough.description:
        return passthrough._replace(stats=reported)
    return EnumerationGateResult(None, description, metadata, reported)


def render_enumeration_guard_section() -> str:
    """The Stage-2 prompt section for this gate, rendered from the names it
    enforces (the ``render_*_section`` convention of ``recon_self_model``), so
    the guidance cannot drift from the code that rejects a violation."""
    stores = ', '.join(f'`{store.value}`' for store in SourceStore)
    categories = ', '.join(f'`{category.value}`' for category in MemoryCategory)
    example = json.dumps(enumeration_scope_example())
    return (
        '## Enumerated Memory IDs\n'
        'When a task you file or update enumerates Mem0 memories ("these are the '
        'N entries to decide on"), the write gate re-checks that list at write '
        'time. Write it so the gate can:\n\n'
        '1. Carry the enumeration as a TOP-LEVEL metadata list of full 36-char '
        'UUIDs, each copied verbatim from a resolved record, never an 8-char or '
        'partial fragment. A fragment list element is rejected when filing '
        f'(`{ENUMERATED_ID_TRUNCATED}`) and recorded as a '
        f'`{EnumeratedIdOutcome.TRUNCATED.value}` exception when updating. An '
        'enumeration that exists only in description prose is never checked.\n'
        '2. Re-resolve every id with `get_memory_by_id` immediately before filing '
        'or updating. Never copy a Stage-1 enumeration forward verbatim.\n'
        '3. The gate drops ids that are genuinely absent and records them in '
        f'`{X_LIVENESS_EXCEPTIONS_KEY}`. It keeps and annotates ids whose lookup '
        'was inconclusive, because inconclusive is not absence. It stamps '
        f'`{X_LIVENESS_REVERIFIED_AT_KEY}`. Never author either key: the stamp '
        "is always overwritten or stripped. Under `'replace'`, send the stored "
        f'`{X_LIVENESS_EXCEPTIONS_KEY}` back as `get_task` returned it, so the '
        'gate can carry forward the records for ids this write did not '
        're-check.\n'
        "4. Update an enumeration with `metadata_mode='merge'` (the default when "
        "both flags are omitted) or `'replace'`, never with `append=True` or "
        "`metadata_mode='additive'`. The additive union keeps every stored member "
        'and the stored stamp, so the gate could neither drop an absent id nor '
        'refresh the stamp; such an update is rejected '
        f'(`{ENUMERATION_UPDATE_NOT_OVERWRITING}`). Both allowed modes replace '
        "what you send wholesale: `'merge'` replaces each top-level key you "
        'supply, so read the task with `get_task` and send the COMPLETE list, '
        "never only the ids being added; `'replace'` replaces the whole metadata "
        'blob, so send every stored key. If the same update also attaches '
        '`memory_hints`, send the hints in a separate `append=True` update.\n'
        '5. Filing a GATE that carries a Mem0 enumeration requires '
        f'`{X_ENUMERATION_SCOPE_KEY}`, a structured object stating what was '
        f'searched: `stores` (from {stores}), `categories` (from {categories}), '
        '`exhaustive` (a bool: true only if every store and category was swept) '
        f'and an optional string `note`. Example: `{example}`. Omitting it on a '
        f'gate filing is rejected (`{ENUMERATION_SCOPE_MISSING}`). A present value '
        'of any other shape, the legacy prose string included, is rejected on '
        f'every write (`{ENUMERATION_SCOPE_MALFORMED}`).\n\n'
        'WHY: reify gate 5537 shipped about 11 minutes after Mem0 entry 0d542614 '
        'was deleted, nominating the deleted row as the likeliest deletion '
        "target, and carried the truncated id '21f78de1'."
    )


async def enumeration_gate_for_filing(
    *,
    agent_id: str | None,
    description: str | None,
    metadata: Any,
    memory_service: Any,
    project_root: str,
    now: Callable[[], datetime] = _utc_now,
) -> EnumerationGateResult:
    """Shape-check, re-resolve and stamp the memory ids a recon ``submit_task``
    enumerates. A new task has no stored metadata, so every key lands as sent;
    the filing-only rules are the fragment rejection and the required scope.
    Never raises: a failed lookup is an inconclusive verdict, not an error."""
    return await _enumeration_gate(
        agent_id=agent_id,
        description=description,
        metadata=metadata,
        memory_service=memory_service,
        project_root=project_root,
        now=now,
        is_filing=True,
        metadata_mode=None,
        append=None,
    )


async def enumeration_gate_for_update(
    *,
    agent_id: str | None,
    description: str | None,
    metadata: Any,
    memory_service: Any,
    project_root: str,
    metadata_mode: str | None,
    append: bool | None,
    now: Callable[[], datetime] = _utc_now,
) -> EnumerationGateResult:
    """Shape-check, re-resolve and stamp the memory ids a recon ``update_task``
    enumerates, judged against the merge its ``metadata_mode`` / ``append``
    flags make the backend apply. A fragment is recorded, never rejected.
    Never raises: a failed lookup is an inconclusive verdict, not an error."""
    return await _enumeration_gate(
        agent_id=agent_id,
        description=description,
        metadata=metadata,
        memory_service=memory_service,
        project_root=project_root,
        now=now,
        is_filing=False,
        metadata_mode=metadata_mode,
        append=append,
    )


async def _enumeration_gate(
    *,
    agent_id: str | None,
    description: str | None,
    metadata: Any,
    memory_service: Any,
    project_root: str,
    now: Callable[[], datetime],
    is_filing: bool,
    metadata_mode: str | None,
    append: bool | None,
) -> EnumerationGateResult:
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
    if liveness.enumerations and not _overwrites_top_level_keys(metadata_mode, append):
        return passthrough._replace(rejection=_not_overwriting_rejection(
            list(liveness.enumerations), metadata_mode, append,
        ))
    if is_filing and _scope_required(parsed, liveness):
        return passthrough._replace(
            rejection=_missing_scope_rejection(list(liveness.enumerations)),
        )
    return _rewrite(
        passthrough, parsed, fragments, liveness,
        verified_at=now().isoformat(), metadata_mode=metadata_mode,
    )
