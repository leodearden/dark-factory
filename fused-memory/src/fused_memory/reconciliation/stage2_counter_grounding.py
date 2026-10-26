"""Grounds Stage 2's self-reported task counters in code-side evidence."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeAlias

from shared.task_metadata_wire import coerce_task_metadata

from fused_memory.middleware.task_creation_journal import CreationOutcomes, TaskKey, task_key
from fused_memory.reconciliation.task_corroboration import (
    LookupVerdict,
    TaskLookup,
    confirm_task_present,
    corroborate_task_lookups,
)

logger = logging.getLogger(__name__)

HINTS_UPDATED_ACTION = 'task_hints_updated'


def _dict_records(records: object) -> list[dict[Any, Any]]:
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _record_key(record: dict[Any, Any], default_project_id: str | None) -> TaskKey | None:
    return task_key(record.get('project_id'), record.get('task_id'), default_project_id)


def action_record_keys(
    records: object,
    default_project_id: str | None,
    valid_statuses: frozenset[str],
) -> frozenset[TaskKey]:
    """The deduped keys of the action records whose ``status`` is in *valid_statuses*.

    Status matching ignores case and surrounding whitespace. A malformed list
    or record is skipped, never raised: a bad record must not corrupt an
    otherwise-good stage report.
    """
    keys: set[TaskKey] = set()
    for record in _dict_records(records):
        try:
            status = record.get('status')
            if not isinstance(status, str) or status.strip().lower() not in valid_statuses:
                continue
            key = _record_key(record, default_project_id)
        except Exception:
            continue
        if key is not None:
            keys.add(key)
    return frozenset(keys)


def coerce_count(value: object) -> int:
    """Best-effort int coercion of a self-reported counter; anything unusable is 0.

    Accepts an int, a finite float, or a numeric string; a bool, a non-finite
    number (task 4140) and anything else is 0. Coercing rather than rejecting
    keeps a legitimately larger ``"3"`` from looking like an undercount and
    being repaired downward, which the upward-only repair forbids
    (task 2230 / W5-mu).
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else 0
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            pass
        try:
            return coerce_count(float(stripped))
        except (ValueError, OverflowError):
            return 0
    return 0


@dataclass(frozen=True)
class HintsClaim:
    """The memory_hints entries Stage 2 says it attached to one task."""

    entities: frozenset[str]
    queries: frozenset[str]


def _hint_entries(value: object) -> frozenset[str]:
    """A hints list as comparable strings; non-str entries compare by canonical JSON."""
    if not isinstance(value, list):
        return frozenset()
    return frozenset(
        entry if isinstance(entry, str) else json.dumps(entry, sort_keys=True)
        for entry in value
    )


def hints_update_claims(
    records: object,
    default_project_id: str | None,
) -> Mapping[TaskKey, HintsClaim]:
    """Stage 2's ``task_hints_updated`` records as one claim per task key.

    Records for the same key union their entries. A record naming no entries
    at all claims nothing. Malformed input is skipped, never raised.
    """
    entities: dict[TaskKey, set[str]] = {}
    queries: dict[TaskKey, set[str]] = {}
    for record in _dict_records(records):
        try:
            if record.get('action') != HINTS_UPDATED_ACTION:
                continue
            key = _record_key(record, default_project_id)
            record_entities = _hint_entries(record.get('entities'))
            record_queries = _hint_entries(record.get('queries'))
        except Exception:
            continue
        if key is None or not (record_entities or record_queries):
            continue
        entities.setdefault(key, set()).update(record_entities)
        queries.setdefault(key, set()).update(record_queries)
    return {
        key: HintsClaim(entities=frozenset(entities[key]), queries=frozenset(queries[key]))
        for key in entities
    }


def hints_claim_confirmed(claim: HintsClaim, get_task_result: object) -> bool:
    """True iff the re-read task's memory_hints contain every claimed entry.

    This is the SUPERSET rule the Stage 2 prompt makes the agent apply before
    it counts a hints update. Fails closed and never raises.
    """
    if not isinstance(get_task_result, dict) or not confirm_task_present(get_task_result):
        return False
    try:
        metadata = coerce_task_metadata(get_task_result.get('metadata')) or {}
        hints = metadata.get('memory_hints')
        if not isinstance(hints, dict):
            return False
        return (
            claim.entities <= _hint_entries(hints.get('entities'))
            and claim.queries <= _hint_entries(hints.get('queries'))
        )
    except Exception:
        return False


@dataclass(frozen=True)
class CounterEvidence:
    """What can raise one counter: journal keys (code truth) and LLM record claims.

    Each claim carries the predicate a re-read task must satisfy to confirm it.
    """

    journaled: frozenset[TaskKey]
    claims: Mapping[TaskKey, Callable[[object], bool]]


@dataclass(frozen=True)
class CounterGrounding:
    """How much of one counter's evidence was confirmed.

    ``record_confirmed``, ``uncorroborated`` and ``unresolvable`` partition the
    record claims the journal did not already cover: confirmed, asked but not
    confirmed, and could not ask.
    """

    journaled: int
    record_confirmed: int
    uncorroborated: int
    unresolvable: int

    @property
    def valid(self) -> int:
        return self.journaled + self.record_confirmed


async def ground_counter(
    taskmaster: Any,
    known_projects: Mapping[str, str] | None,
    evidence: CounterEvidence,
) -> CounterGrounding:
    """Confirm one counter's evidence. Journal keys are trusted and never looked up."""
    record_only = [
        TaskLookup(project_id=key[0], task_id=key[1], accept=accept)
        for key, accept in evidence.claims.items()
        if key not in evidence.journaled
    ]
    verdicts = Counter(await corroborate_task_lookups(taskmaster, known_projects, record_only))
    return CounterGrounding(
        journaled=len(evidence.journaled),
        record_confirmed=verdicts[LookupVerdict.CONFIRMED],
        uncorroborated=verdicts[LookupVerdict.UNCONFIRMED],
        unresolvable=verdicts[LookupVerdict.UNRESOLVABLE],
    )


def publish_counter(
    counter: str,
    prefix: str,
    reported_raw: object,
    grounding: CounterGrounding,
    *,
    run_id: str,
    project_id: str | None,
) -> dict[str, Any]:
    """The stats one grounded counter publishes; repairs UPWARD only.

    The journal is complete only for submissions that carried
    ``_causation_id``, so it never proves a larger self-report false
    (task 2230 / W5-mu): a counter is raised to its confirmed evidence and
    never lowered.
    """
    reported = coerce_count(reported_raw)
    updates: dict[str, Any] = {
        counter: reported,
        f'{prefix}_valid': grounding.valid,
        f'{prefix}_journaled': grounding.journaled,
        f'{prefix}_uncorroborated': grounding.uncorroborated,
        f'{prefix}_unresolvable': grounding.unresolvable,
    }
    if grounding.valid > reported:
        updates[counter] = grounding.valid
        updates[f'{counter}_reported'] = reported_raw
        logger.warning(
            'reconciliation.stage2_%s_undercount: run_id=%s project_id=%s self-reported '
            '%s=%r but %d were confirmed (journaled=%d, corroborated records=%d) — '
            'repairing upward to %d.',
            counter, run_id, project_id, counter, reported_raw, grounding.valid,
            grounding.journaled, grounding.record_confirmed, grounding.valid,
        )
    shortfall = grounding.uncorroborated + grounding.unresolvable
    if shortfall:
        logger.warning(
            'reconciliation.stage2_%s_uncorroborated: run_id=%s project_id=%s %d record '
            'claim(s) could not be confirmed via get_task (uncorroborated=%d '
            'unresolvable=%d) — they did NOT raise %s, which stands at %r.',
            prefix, run_id, project_id, shortfall, grounding.uncorroborated,
            grounding.unresolvable, counter, updates[counter],
        )
    return updates


EvidenceBuilder: TypeAlias = Callable[
    [Mapping[str, Any], str | None, CreationOutcomes], CounterEvidence,
]


def _creation_records_evidence(
    stats: Mapping[str, Any],
    project_id: str | None,
    status: str,
    journaled: frozenset[TaskKey],
) -> CounterEvidence:
    records = action_record_keys(stats.get('task_created_records'), project_id, frozenset({status}))
    return CounterEvidence(journaled=journaled, claims=dict.fromkeys(records, confirm_task_present))


def _created_evidence(
    stats: Mapping[str, Any], project_id: str | None, journal: CreationOutcomes,
) -> CounterEvidence:
    return _creation_records_evidence(stats, project_id, 'created', journal.created)


def _combined_evidence(
    stats: Mapping[str, Any], project_id: str | None, journal: CreationOutcomes,
) -> CounterEvidence:
    return _creation_records_evidence(stats, project_id, 'combined', journal.combined)


def _hints_updated_evidence(
    stats: Mapping[str, Any], project_id: str | None, journal: CreationOutcomes,
) -> CounterEvidence:
    claims = hints_update_claims(stats.get('task_hints_updated_records'), project_id)
    return CounterEvidence(
        journaled=frozenset(),
        claims={key: partial(hints_claim_confirmed, claim) for key, claim in claims.items()},
    )


@dataclass(frozen=True)
class _GroundedCounter:
    counter: str
    prefix: str
    evidence: EvidenceBuilder


_GROUNDED_COUNTERS: tuple[_GroundedCounter, ...] = (
    _GroundedCounter('tasks_created', 'task_created_records', _created_evidence),
    _GroundedCounter('tasks_combined', 'tasks_combined_evidence', _combined_evidence),
    _GroundedCounter('tasks_hints_updated', 'task_hints_updated_records', _hints_updated_evidence),
)

_NO_JOURNAL = CreationOutcomes(created=frozenset(), combined=frozenset())


async def ground_stage2_counters(
    stats: Mapping[str, Any],
    *,
    taskmaster: Any,
    known_projects: Mapping[str, str] | None,
    project_id: str | None,
    run_id: str,
    journal: CreationOutcomes | None,
) -> dict[str, Any]:
    """The stat updates that ground every Stage 2 task counter; *stats* is not mutated.

    *journal* is the run's creation outcomes, or None when the journal could
    not be read, in which case grounding falls back to records alone.
    """
    if journal is None:
        logger.warning(
            'reconciliation.stage2_creation_journal_unavailable: run_id=%s project_id=%s '
            '— the creation journal could not be read; Stage 2 task counters are '
            'grounded on corroborated records only.',
            run_id, project_id,
        )
        journal = _NO_JOURNAL
    updates: dict[str, Any] = {}
    for spec in _GROUNDED_COUNTERS:
        grounding = await ground_counter(
            taskmaster, known_projects, spec.evidence(stats, project_id, journal),
        )
        updates.update(publish_counter(
            spec.counter, spec.prefix, stats.get(spec.counter, 0), grounding,
            run_id=run_id, project_id=project_id,
        ))
    return updates
