"""The answer to "are this task's dependencies satisfied?" (task 6518).

:func:`deps_verdict` is THE dependency predicate, and it answers in three
states.  Dispatch reads only ``.satisfied``, so an UNKNOWN dependency waits
exactly as an UNSATISFIED one does.  Park GC expires a park only on
UNSATISFIED and keeps it on UNKNOWN: expiry needs affirmative evidence
(esc-6040-7 D2, INV-3 corroborate before acting).

The predicate is pure; escalations and counters belong to its callers in the
scheduler.  Its DEBUG lines explain why dispatch waits, so they log under the
scheduler's logger.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

from shared.task_metadata import parse_metadata

from orchestrator.task_status import TERMINAL_STATUSES

logger = logging.getLogger('orchestrator.scheduler')


class DepKind(StrEnum):
    """Which dependency arm produced a non-satisfied verdict."""

    LOCAL = 'local'
    EXTERNAL = 'external'
    DELIVERED_CHECK = 'delivered_check'


class DepState(StrEnum):
    SATISFIED = 'satisfied'
    UNSATISFIED = 'unsatisfied'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class DepsVerdict:
    """A dependency answer; ``kind`` names the arm iff the answer is not SATISFIED."""

    state: DepState
    kind: DepKind | None = None

    def __post_init__(self) -> None:
        if (self.kind is None) != (self.state is DepState.SATISFIED):
            raise ValueError(
                f'a {self.state} dependency verdict cannot carry kind {self.kind}: '
                'a kind names the arm of an unsatisfied or unknown verdict only'
            )

    @classmethod
    def unsatisfied(cls, kind: DepKind) -> DepsVerdict:
        return cls(DepState.UNSATISFIED, kind)

    @classmethod
    def unknown(cls, kind: DepKind) -> DepsVerdict:
        return cls(DepState.UNKNOWN, kind)

    @property
    def satisfied(self) -> bool:
        return self.state is DepState.SATISFIED

    @staticmethod
    def combine(verdicts: Iterable[DepsVerdict]) -> DepsVerdict:
        """The first UNSATISFIED verdict, else the first UNKNOWN, else SATISFIED."""
        first_unknown: DepsVerdict | None = None
        for verdict in verdicts:
            if verdict.state is DepState.UNSATISFIED:
                return verdict
            if verdict.state is DepState.UNKNOWN and first_unknown is None:
                first_unknown = verdict
        return DEPS_SATISFIED if first_unknown is None else first_unknown


DEPS_SATISFIED = DepsVerdict(DepState.SATISFIED)


def task_external_deps(task: dict) -> list[str]:
    """Return *task*'s ``metadata.external_deps`` as canonical wire strings.

    THE single seam every scheduler external_deps read routes through
    (task 2167 — W3-δ SEAM A).  Delegates to the shared, versioned
    ``shared.task_metadata.parse_metadata`` parser instead of the old untyped
    ``(task.get('metadata') or {}).get('external_deps') or []`` idiom, which
    ``AttributeError``s on JSON-string metadata (str has no ``.get``).

    ``direction='read'`` makes this best-effort — it never raises, even on
    corrupt/unparseable metadata — so a bad historical metadata row cannot
    break a scheduler dispatch tick (matches the scheduler's existing
    fail-safe philosophy).

    The return value stays ``list[str]`` — the canonical "project_id:task_id"
    wire form fed to ``get_external_statuses`` / the external-status cache —
    and is intentionally NOT converted to ``list[ExternalDep]`` (PRD Open
    Question #4): a malformed dep must still reach ``get_external_statuses``
    so it can resolve to the ``malformed`` sentinel and escalate, which
    pre-filtering with ``ExternalDep.parse`` would silently prevent.
    """
    metadata, _warnings = parse_metadata(task.get('metadata'), direction='read')
    return metadata.external_deps


def deps_verdict(
    task: dict,
    status_map: dict[str, str],
    tasks_by_id: dict[str, dict] | None = None,
    *,
    external_status_cache: dict[str, str] | None = None,
    external_resolver_failed: bool = False,
    delivered_check_cache: dict[str, bool] | None = None,
    terminal_dep_records: dict[str, dict] | None = None,
) -> DepsVerdict:
    """THE dependency predicate: are *task*'s dependencies satisfied?

    Three arms, local then external then delivered-check, combined by
    :meth:`DepsVerdict.combine`: the first arm known UNSATISFIED wins, else
    the first UNKNOWN.  The arms run lazily in that order, so none runs, or
    logs, after an UNSATISFIED one.

    **Local**: a dep is satisfied when its status is in
    :data:`TERMINAL_STATUSES` (``done`` or ``cancelled``).  ``cancelled``
    represents an obsolete or duplicate task and should not block its
    dependents — the dependent re-architects against current main and either
    finds the work already merged or escalates for a different reason.  A dep
    with no *status_map* entry is UNSATISFIED.

    **Intra-train allowance (PRD § 9.3):** when *tasks_by_id* is supplied,
    a dep in status ``merge-deferred`` is also treated as satisfied provided
    that BOTH the dependent (*task*) and the dep carry ``metadata.train.id``
    AND their train IDs match.  This allows the next member of an atomic
    train to dispatch as soon as its predecessor has been parked by the
    merge-deferred gate, without waiting for a full ``done`` transition.
    When *tasks_by_id* is ``None`` (the default), the allowance is disabled
    and ``merge-deferred`` blocks like any other non-terminal status.  The
    allowance also requires the dep record to be present in *tasks_by_id*;
    a missing dep (stale snapshot) is treated conservatively as blocking.

    **Cross-project external deps**: consulted only when
    *external_status_cache* is not ``None``.  An external dep in
    ``task.metadata.external_deps`` is satisfied iff its cached status is
    exactly ``'done'``; any other reported value, live status or sentinel
    (``unknown_project`` / ``unknown_task`` / ``malformed``), is UNSATISFIED.
    A dep missing from the cache, or any external dep when
    *external_resolver_failed* is ``True``, is UNKNOWN.

    **Delivered checks** (capability-delivered-checks PRD, task delta):
    consulted only when *delivered_check_cache* is not ``None`` AND
    *tasks_by_id* is supplied.  Each TERMINAL (done/cancelled) local dep whose
    record carries a truthy ``metadata.delivered_checks`` must resolve to
    exactly ``True`` in *delivered_check_cache* (keyed by dep task id) — this
    applies to a ``cancelled`` dep just as much as a ``done`` one, since the
    check trusts the state of ``main``, not the status label.  ``False`` (the
    check ran and FAILED) is UNSATISFIED; a dep absent from the cache
    (errored, over budget, or not yet evaluated this tick) is UNKNOWN.  A dep
    whose record carries no ``metadata.delivered_checks`` is never consulted.

    **``terminal_dep_records`` fallback (task 2692, δ/ε follow-up):** the
    active-only ``get_tasks`` fetch that seeds *tasks_by_id* excludes
    DONE/CANCELLED tasks, so a just-completed dep is often genuinely ABSENT
    from *tasks_by_id* even though its TERMINAL status is already in
    *status_map*.  When the delivered-check arm's ``tasks_by_id.get(dep_id)``
    misses (``None``) and *terminal_dep_records* is not ``None``, the dep
    record is looked up there instead — a purely ADDITIVE fallback consulted
    ONLY in that arm (never the intra-train merge-deferred allowance above,
    which deals exclusively with non-terminal deps already present in
    *tasks_by_id*).  If the dep is present in *tasks_by_id*,
    *terminal_dep_records* is never consulted (*tasks_by_id* wins).  If the
    dep is absent from BOTH, the arm skips it.

    Handles three dependency formats:
      - dict with 'id' key: ``{'id': 1}`` or ``{'id': '1'}``
      - integer: ``1``
      - string: ``'1'``

    Emits a DEBUG log when a dependency blocks dispatch, naming the dep ID
    and its current status to aid diagnosis of premature-dispatch issues.
    Emits a separate DEBUG log when the intra-train allowance fires.
    """
    dep_ids = [
        str(d.get('id', d) if isinstance(d, dict) else d)
        for d in task.get('dependencies', [])
    ]
    arms = (
        partial(_local_verdict, task, dep_ids, status_map, tasks_by_id),
        partial(_external_verdict, task, external_status_cache, external_resolver_failed),
        partial(
            _delivered_check_verdict,
            task, dep_ids, status_map, tasks_by_id,
            delivered_check_cache, terminal_dep_records,
        ),
    )
    return DepsVerdict.combine(arm() for arm in arms)


def _local_verdict(
    task: dict,
    dep_ids: list[str],
    status_map: dict[str, str],
    tasks_by_id: dict[str, dict] | None,
) -> DepsVerdict:
    """The local arm of :func:`deps_verdict`: SATISFIED or UNSATISFIED, never UNKNOWN."""
    task_id = str(task.get('id', '?'))
    # Resolve this task's train id once (None if not a train member).
    # Coerce empty string to None so that '' does not accidentally match
    # another task whose id is also '' (ill-formed but defensively handled).
    task_train = (task.get('metadata') or {}).get('train')
    task_train_id: str | None = (
        task_train.get('id') or None if isinstance(task_train, dict) else None
    )
    for dep_id in dep_ids:
        dep_status = status_map.get(dep_id, 'unknown')
        if dep_status in TERMINAL_STATUSES:
            continue
        # Intra-train allowance: merge-deferred predecessor in same train.
        if (
            dep_status == 'merge-deferred'
            and task_train_id is not None
            and tasks_by_id is not None
        ):
            dep_task = tasks_by_id.get(dep_id)
            if dep_task is not None:
                dep_train = (dep_task.get('metadata') or {}).get('train')
                # Coerce empty string to None (see task_train_id note above).
                dep_train_id: str | None = (
                    dep_train.get('id') or None if isinstance(dep_train, dict) else None
                )
                if dep_train_id == task_train_id:
                    logger.debug(
                        'Task %s: intra-train dep satisfied: dep=%s train_id=%s',
                        task_id,
                        dep_id,
                        task_train_id,
                    )
                    continue
        logger.debug(
            'Task %s blocked: dep %s has status %s, '
            'need done or cancelled',
            task_id,
            dep_id,
            dep_status,
        )
        return DepsVerdict.unsatisfied(DepKind.LOCAL)
    return DEPS_SATISFIED


def _external_verdict(
    task: dict,
    external_status_cache: dict[str, str] | None,
    external_resolver_failed: bool,
) -> DepsVerdict:
    """The cross-project arm of :func:`deps_verdict`; inactive without a cache."""
    if external_status_cache is None:
        return DEPS_SATISFIED
    task_id = str(task.get('id', '?'))
    verdict = DEPS_SATISFIED
    for ext_dep in task_external_deps(task):
        if external_resolver_failed:
            logger.debug(
                'Task %s blocked: external dep %r not checked '
                '(resolver failed — fail-safe wait)',
                task_id,
                ext_dep,
            )
            return DepsVerdict.unknown(DepKind.EXTERNAL)
        ext_status = external_status_cache.get(ext_dep)
        if ext_status == 'done':
            continue
        logger.debug(
            'Task %s blocked: external dep %r has status %r, '
            'need done',
            task_id,
            ext_dep,
            ext_status,
        )
        if ext_dep in external_status_cache:
            return DepsVerdict.unsatisfied(DepKind.EXTERNAL)
        verdict = DepsVerdict.unknown(DepKind.EXTERNAL)
    return verdict


def _delivered_check_verdict(
    task: dict,
    dep_ids: list[str],
    status_map: dict[str, str],
    tasks_by_id: dict[str, dict] | None,
    delivered_check_cache: dict[str, bool] | None,
    terminal_dep_records: dict[str, dict] | None,
) -> DepsVerdict:
    """The delivered-check arm of :func:`deps_verdict`; inactive without a cache or *tasks_by_id*."""
    if delivered_check_cache is None or tasks_by_id is None:
        return DEPS_SATISFIED
    task_id = str(task.get('id', '?'))
    verdict = DEPS_SATISFIED
    for dep_id in dep_ids:
        if status_map.get(dep_id, 'unknown') not in TERMINAL_STATUSES:
            continue
        dep_task = tasks_by_id.get(dep_id)
        if dep_task is None and terminal_dep_records is not None:
            # Fallback (task 2692): the active-only get_tasks fetch
            # excludes this TERMINAL dep from tasks_by_id — consult
            # the per-tick backfilled record instead. Only reached
            # when tasks_by_id genuinely misses (tasks_by_id wins).
            dep_task = terminal_dep_records.get(dep_id)
        if dep_task is None:
            # Conscious fail-OPEN, not a gap: see the identical skip site in
            # scheduler.py::Scheduler._compute_delivered_check_cache (task
            # 2692, reviewer_comprehensive robustness amendment) for the full
            # rationale. In short — an unreadable record (fetch failed in
            # scheduler.py::Scheduler._phase_backfill_terminal_dep_records,
            # which logs a bounded WARNING for it) means this dep is simply
            # never gated here, rather than fail-safe-blocking a dependent
            # whose dep record may never be fetchable.
            continue
        if not (dep_task.get('metadata') or {}).get('delivered_checks'):
            continue
        confirmed = delivered_check_cache.get(dep_id)
        if confirmed is True:
            continue
        logger.debug(
            'Task %s blocked: dep %s delivered_checks not confirmed on '
            'main (cache=%r)',
            task_id,
            dep_id,
            confirmed,
        )
        if confirmed is False:
            return DepsVerdict.unsatisfied(DepKind.DELIVERED_CHECK)
        verdict = DepsVerdict.unknown(DepKind.DELIVERED_CHECK)
    return verdict
