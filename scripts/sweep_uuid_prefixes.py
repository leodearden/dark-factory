"""Expand stored truncated-uuid citations to the full ids they name.

PRD contract C6, ``plans/uuid-prefix-resolution-prd.md`` §4-C6, which owns the
lane table and the rules; this docstring does not restate them (INV-9). Every
token comes from ``shared.uuid_prefix`` (C1) and every answer from the
``PrefixResolver`` port (C2), resolved against the record's own project. Only
``unique`` tokens are expanded; ``ambiguous`` ones are reported by record and
left byte-identical, ``none`` ones are counted and left alone.

One deliberate departure from §4-C6: the tasks lane sends ``update_task`` a
``merge`` of only the changed metadata keys, not a ``replace`` of the
just-read blob. A replace strips stored wait anchors and reverts any
concurrent write to a sibling key (``build_task_update``).

Run it from the checkout root, one project per sitting, dry run first (the
default), then one lane at a time::

    uv run --project fused-memory python scripts/sweep_uuid_prefixes.py \\
        --config fused-memory/config/config.yaml \\
        --project <id> [--project-root <checkout>] [--lane all]
    uv run --project fused-memory python scripts/sweep_uuid_prefixes.py \\
        --config fused-memory/config/config.yaml \\
        --project <id> [--project-root <checkout>] --apply --lane <lane>

A second ``--apply`` of the same lane must report ``0 expandable remaining``
and exit 0.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import json
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, NamedTuple, Protocol, TypeVar

import _checkout as _checkout
import httpx
import sweep_toolcall_markup as markup_sweep
from _task_db_scan import UNREADABLE_STORE_ERRORS, connect_ro, decode_metadata, tasks_db_path
from fused_memory.models.scope import Scope, build_known_projects_map, resolve_project_id_for_root
from migrate_metadata_modules_to_files import (
    DEFAULT_SERVER,
    FusedMemoryClient,
    write_failure_reason,
)
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV
from shared.uuid_prefix import PrefixToken, find_prefix_tokens, substitute_all
from shared.uuid_prefix_guard import PrefixResolver, Resolution, ResolverUnavailable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The planner: one record, its tokens' resolutions, what to write.
# ---------------------------------------------------------------------------


class AmbiguousSite(NamedTuple):
    """A token that names more than one id, so the sweep must not choose."""

    path: tuple[str | int, ...]
    token: str
    candidates: tuple[str, ...]


class RecordPlan(NamedTuple):
    """What one record holds, and the document to write when anything expands.

    ``after`` is ``None`` exactly when ``expansions`` is empty, so "nothing to
    write" is never encoded as an unchanged copy.
    """

    expansions: tuple[tuple[PrefixToken, str], ...]
    ambiguous: tuple[AmbiguousSite, ...]
    #: Unique tokens under a field the lane's write path cannot reach.
    unwritable: tuple[PrefixToken, ...]
    untouched: int
    after: dict[str, Any] | None


def plan_record(
    fields: Mapping[str, Any],
    resolutions: Mapping[str, Resolution],
    unwritable_fields: frozenset[str],
) -> RecordPlan:
    """Classify every token in *fields* and build the expanded document.

    Pure. A token with no entry in *resolutions* raises ``KeyError``: every
    token was meant to be resolved first, and silently treating a missing
    answer as ``none`` would leave a live citation unexpanded (INV-11).
    """
    expansions: list[tuple[PrefixToken, str]] = []
    ambiguous: list[AmbiguousSite] = []
    unwritable: list[PrefixToken] = []
    untouched = 0
    for token in find_prefix_tokens(dict(fields)):
        resolution = resolutions[token.token]
        if resolution.outcome == 'unique':
            if token.path[0] in unwritable_fields:
                unwritable.append(token)
            else:
                expansions.append((token, resolution.candidates[0].id))
        elif resolution.outcome == 'ambiguous':
            ambiguous.append(
                AmbiguousSite(
                    path=token.path,
                    token=token.token,
                    candidates=tuple(candidate.id for candidate in resolution.candidates),
                )
            )
        else:
            untouched += 1
    return RecordPlan(
        expansions=tuple(expansions),
        ambiguous=tuple(ambiguous),
        unwritable=tuple(unwritable),
        untouched=untouched,
        after=substitute_all(fields, expansions) if expansions else None,
    )


# ---------------------------------------------------------------------------
# Resolution: once per distinct token, a bounded number at a time.
# ---------------------------------------------------------------------------

#: Resolutions in flight at once. Each one walks the project's whole id space
#: (D9); the plan's two-phase resolution decision sets this bound. Distinct
#: from ``uuid_prefix_guard.MAX_CONCURRENT_RESOLUTIONS``, which is a per-call
#: latency policy for the MCP guard, not a corpus-walk load policy.
RESOLVE_CONCURRENCY = 4


async def resolve_distinct(
    project_id: str,
    tokens: Iterable[str],
    resolve: PrefixResolver,
    memo: Mapping[str, Resolution],
    *,
    concurrency: int = RESOLVE_CONCURRENCY,
) -> dict[str, Resolution]:
    """*memo* extended with a resolution for every token it lacks.

    Each distinct token is resolved exactly once. *memo* is never mutated. A
    store outage on any token raises ``ResolverUnavailable`` itself, after the
    task group has cancelled every other resolution still in flight.
    """
    pending = [token for token in dict.fromkeys(tokens) if token not in memo]
    resolved: dict[str, Resolution] = {}
    slots = asyncio.Semaphore(concurrency)

    async def resolve_one(token: str) -> None:
        async with slots:
            resolved[token] = await resolve(project_id, token)

    try:
        async with asyncio.TaskGroup() as group:
            for token in pending:
                group.create_task(resolve_one(token))
    except* ResolverUnavailable as outage:
        raise outage.exceptions[0] from None
    return {**memo, **resolved}


# ---------------------------------------------------------------------------
# What a lane is, and what one run of it reports.
# ---------------------------------------------------------------------------


class SweepScope(NamedTuple):
    """One project, and the checkout holding its file-backed stores, if any."""

    project_id: str
    project_root: Path | None


#: A lane's own locator type for the record it writes back.
_Source = TypeVar('_Source')
_Source_co = TypeVar('_Source_co', covariant=True)


class Record(NamedTuple, Generic[_Source_co]):
    """One sweepable document, and where its lane writes it back."""

    key: str
    fields: dict[str, Any]
    source: _Source_co


class WriteFailureRecord(NamedTuple):
    key: str
    reason: str
    error: str


class AmbiguousRecordSite(NamedTuple):
    record: str
    path: tuple[str | int, ...]
    token: str
    candidates: tuple[str, ...]


class UnwritableSite(NamedTuple):
    record: str
    path: tuple[str | int, ...]
    token: str


class LaneReport(NamedTuple):
    """One lane's outcome for one project.

    ``expanded`` counts only occurrences that are on disk, so a dry run always
    leaves ``remaining == expandable``. ``aborted`` names the store whose
    outage stopped the lane before anything was written.
    """

    lane: str
    project_id: str
    scanned: int
    expandable: int
    expanded: int
    untouched: int
    ambiguous: tuple[AmbiguousRecordSite, ...]
    unwritable: tuple[UnwritableSite, ...]
    refused: Mapping[str, int]
    failures: tuple[WriteFailureRecord, ...]
    aborted: str | None

    @property
    def remaining(self) -> int:
        return self.expandable - self.expanded

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._asdict(),
            'remaining': self.remaining,
            'ambiguous': [site._asdict() for site in self.ambiguous],
            'unwritable': [site._asdict() for site in self.unwritable],
            'refused': dict(self.refused),
            'failures': [failure._asdict() for failure in self.failures],
        }


class StoreUnavailable(Exception):
    """A lane's own store could not be read: the population's ``ResolverUnavailable``."""

    def __init__(self, store: str) -> None:
        super().__init__(f'the {store} store could not be read')
        self.store = store


#: Every outage that stops a lane before it writes. Each carries ``.store``.
_STORE_OUTAGES = (ResolverUnavailable, StoreUnavailable)


class TaskClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> object: ...


TaskClientOpener = Callable[[str | None], contextlib.AbstractAsyncContextManager[TaskClient]]


class TaskClientSlot:
    """The run's one MCP task client, opened by the first write that asks for it.

    A dry run never asks, so it never dials the server. The client closes with
    the run's exit stack.
    """

    def __init__(
        self,
        opener: TaskClientOpener,
        server_url: str | None,
        stack: contextlib.AsyncExitStack,
    ) -> None:
        self._opener = opener
        self._server_url = server_url
        self._stack = stack
        self._client: TaskClient | None = None

    async def client(self) -> TaskClient:
        if self._client is None:
            self._client = await self._stack.enter_async_context(self._opener(self._server_url))
        return self._client


class Mem0CallFailed(Exception):
    """One mem0 read or amend failed in the backend, not in this script."""


class Mem0Corpus(Protocol):
    """A project's Mem0 bodies, read and amended in place by memory id.

    ``bodies`` raises ``StoreUnavailable`` when the store cannot be walked;
    ``read_body`` and ``amend_body`` raise ``Mem0CallFailed`` when the backend
    fails. Any other exception is a bug and propagates.
    """

    def bodies(self, project_id: str) -> AsyncIterator[tuple[str, str]]: ...

    async def read_body(self, project_id: str, memory_id: str) -> str | None: ...

    async def amend_body(self, project_id: str, memory_id: str, body: str) -> Mapping[str, Any]: ...


class LaneContext(NamedTuple):
    """What a lane's population and writer may reach during one run."""

    scope: SweepScope
    mem0: Mem0Corpus | None
    tasks: TaskClientSlot | None


Population = Callable[[LaneContext], AsyncIterator[Record[_Source] | markup_sweep.Refusal]]
Writer = Callable[
    [LaneContext, Record[_Source], dict[str, Any]], Awaitable[WriteFailureRecord | None]
]


class Lane(NamedTuple, Generic[_Source]):
    """A store's sweep: its population and the writer for exactly those records."""

    name: str
    #: The lane reads a checkout's files, so a run needs a project root.
    needs_root: bool
    #: Top-level fields scanned for the report but never written.
    unwritable_fields: frozenset[str]
    population: Population[_Source]
    write: Writer[_Source]


#: A lane could not finish because a store it reads was unreachable.
EXIT_STORE_UNAVAILABLE = 4


def exit_code(reports: Iterable[LaneReport]) -> int:
    """Fold lane reports into one status: outage, then write failure, then work left."""
    reports = tuple(reports)
    if any(report.aborted is not None for report in reports):
        return EXIT_STORE_UNAVAILABLE
    if any(report.failures for report in reports):
        return markup_sweep.EXIT_WRITE_FAILED
    if any(report.remaining > 0 for report in reports):
        return markup_sweep.EXIT_REPAIRABLE_REMAINS
    return markup_sweep.EXIT_CLEAN


class WritePacer:
    """Holds each write until at least ``1 / rate`` seconds after the last one."""

    def __init__(
        self,
        rate: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._last_release: float | None = None

    async def wait(self) -> None:
        delay = 0.0
        if self._last_release is not None:
            delay = max(0.0, self._last_release + self._interval - self._clock())
        await self._sleep(delay)
        self._last_release = self._clock()


# ---------------------------------------------------------------------------
# Dry-run diffs: one per changed string.
# ---------------------------------------------------------------------------


def format_path(path: tuple[str | int, ...]) -> str:
    """``('evidence', 0, 'observation')`` as ``evidence[0].observation``."""
    parts = [f'[{key}]' if isinstance(key, int) else f'.{key}' for key in path]
    return ''.join(parts).removeprefix('.')


def _value_at(document: Mapping[str, Any], path: tuple[str | int, ...]) -> Any:
    node: Any = document
    for key in path:
        node = node[key]
    return node


def render_diffs(lane: str, record: Record[object], plan: RecordPlan) -> list[str]:
    """A unified diff of each string *plan* changes, labelled by lane, record and path."""
    if plan.after is None:
        return []
    diffs = []
    for path in dict.fromkeys(token.path for token, _ in plan.expansions):
        label = f'{lane} {record.key} {format_path(path)}'
        before = str(_value_at(record.fields, path)).splitlines()
        after = str(_value_at(plan.after, path)).splitlines()
        diffs.append(
            '\n'.join(difflib.unified_diff(before, after, fromfile=label, tofile=label, lineterm=''))
        )
    return diffs


# ---------------------------------------------------------------------------
# The run loop: population, resolution, plan, then a diff or a paced write.
# ---------------------------------------------------------------------------

#: Writes per second in every lane. Each mem0 amend re-embeds, and a bulk
#: burst of task writes has restarted fused-memory before.
DEFAULT_RATE = 5.0


async def run_sweep(
    scope: SweepScope,
    *,
    lanes: Iterable[str],
    apply: bool,
    resolve: PrefixResolver,
    mem0: Mem0Corpus | None = None,
    open_task_client: TaskClientOpener | None = None,
    server_url: str | None = None,
    rate: float = DEFAULT_RATE,
) -> tuple[tuple[LaneReport, ...], tuple[str, ...]]:
    """Sweep the selected *lanes* in declared order; return their reports and dry-run diffs.

    Resolutions are memoised for this one run, shared across its lanes, and
    discarded with it.
    """
    selected = frozenset(lanes)
    unknown = selected - LANES.keys()
    if unknown:
        raise ValueError(f'unknown lanes {sorted(unknown)}; known: {list(LANES)}')
    pacer = WritePacer(rate)
    memo: dict[str, Resolution] = {}
    reports: list[LaneReport] = []
    diffs: list[str] = []
    async with contextlib.AsyncExitStack() as stack:
        tasks = None if open_task_client is None else TaskClientSlot(open_task_client, server_url, stack)
        context = LaneContext(scope=scope, mem0=mem0, tasks=tasks)
        for lane in (lane for name, lane in LANES.items() if name in selected):
            report, lane_diffs, memo = await _sweep_lane(
                lane, context, resolve, memo, apply=apply, pacer=pacer
            )
            reports.append(report)
            diffs.extend(lane_diffs)
    return tuple(reports), tuple(diffs)


async def _sweep_lane(
    lane: Lane[_Source],
    context: LaneContext,
    resolve: PrefixResolver,
    memo: dict[str, Resolution],
    *,
    apply: bool,
    pacer: WritePacer,
) -> tuple[LaneReport, tuple[str, ...], dict[str, Resolution]]:
    project_id = context.scope.project_id
    try:
        records, refused = await _collect(lane, context)
        tokens = [token.token for record in records for token in find_prefix_tokens(record.fields)]
        memo = await resolve_distinct(project_id, tokens, resolve, memo)
    except _STORE_OUTAGES as outage:
        return _aborted_report(lane.name, project_id, outage.store), (), memo
    plans = [(record, plan_record(record.fields, memo, lane.unwritable_fields)) for record in records]
    if not apply:
        diffs = tuple(diff for record, plan in plans for diff in render_diffs(lane.name, record, plan))
        return _lane_report(lane.name, project_id, plans, refused, expanded=0, failures=()), diffs, memo
    expanded, failures = await _write_all(lane, context, plans, pacer)
    report = _lane_report(lane.name, project_id, plans, refused, expanded=expanded, failures=failures)
    return report, (), memo


async def _collect(
    lane: Lane[_Source], context: LaneContext
) -> tuple[list[Record[_Source]], Counter[str]]:
    records: list[Record[_Source]] = []
    refused: Counter[str] = Counter()
    async for item in lane.population(context):
        if isinstance(item, Record):
            records.append(item)
        else:
            refused[item.reason] += 1
    return records, refused


async def _write_all(
    lane: Lane[_Source],
    context: LaneContext,
    plans: list[tuple[Record[_Source], RecordPlan]],
    pacer: WritePacer,
) -> tuple[int, tuple[WriteFailureRecord, ...]]:
    """Write every planned record; count expansions only once they are on disk."""
    expanded = 0
    failures: list[WriteFailureRecord] = []
    for record, plan in plans:
        if plan.after is None:
            continue
        await pacer.wait()
        failure = await lane.write(context, record, plan.after)
        if failure is None:
            expanded += len(plan.expansions)
        else:
            failures.append(failure)
    return expanded, tuple(failures)


def _lane_report(
    lane: str,
    project_id: str,
    plans: Sequence[tuple[Record[object], RecordPlan]],
    refused: Mapping[str, int],
    *,
    expanded: int,
    failures: tuple[WriteFailureRecord, ...],
) -> LaneReport:
    return LaneReport(
        lane=lane,
        project_id=project_id,
        scanned=len(plans),
        expandable=sum(len(plan.expansions) for _, plan in plans),
        expanded=expanded,
        untouched=sum(plan.untouched for _, plan in plans),
        ambiguous=tuple(
            AmbiguousRecordSite(record.key, site.path, site.token, site.candidates)
            for record, plan in plans
            for site in plan.ambiguous
        ),
        unwritable=tuple(
            UnwritableSite(record.key, token.path, token.token)
            for record, plan in plans
            for token in plan.unwritable
        ),
        refused=MappingProxyType(dict(refused)),
        failures=failures,
        aborted=None,
    )


def _aborted_report(lane: str, project_id: str, store: str) -> LaneReport:
    """Nothing is known about the corpus, so every count is zero."""
    return _lane_report(lane, project_id, [], {}, expanded=0, failures=())._replace(aborted=store)


# ---------------------------------------------------------------------------
# Lane plumbing shared by the writers.
# ---------------------------------------------------------------------------

def _required_root(scope: SweepScope, lane: str) -> Path:
    if scope.project_root is None:
        raise ValueError(f'the {lane} lane needs a project root and the scope has none')
    return scope.project_root


# ---------------------------------------------------------------------------
# The escalations lane: the markup sweep's engine, end to end.
# ---------------------------------------------------------------------------


class EscalationSource(NamedTuple):
    resolved: markup_sweep.ResolvedTarget
    #: The bytes the plan was computed from; the engine re-checks them before the swap.
    raw: str


async def _escalation_population(
    context: LaneContext,
) -> AsyncIterator[Record[EscalationSource] | markup_sweep.Refusal]:
    root = _required_root(context.scope, markup_sweep.LANE_ESCALATIONS)
    targets = [
        target
        for target in markup_sweep.discover_targets(root)
        if target.lane == markup_sweep.LANE_ESCALATIONS
    ]
    for target in markup_sweep.dedupe_by_realpath(targets):
        yield _escalation_record(target, root)


def _escalation_record(
    target: markup_sweep.Target, root: Path
) -> Record[EscalationSource] | markup_sweep.Refusal:
    loaded = markup_sweep.load_target(target)
    if isinstance(loaded, markup_sweep.Refusal):
        return loaded
    if not markup_sweep.round_trips(loaded.raw, loaded.obj):
        return markup_sweep.Refusal(
            path=target.path,
            lane=target.lane,
            reason=markup_sweep.REASON_FORMAT_NOT_REPRODUCIBLE,
        )
    resolved = markup_sweep.resolve_write_target(target, root)
    if isinstance(resolved, markup_sweep.Refusal):
        return resolved
    relpath = target.path.relative_to(root).as_posix()
    return Record(
        key=f"{loaded.obj['id']} ({relpath})",
        fields=loaded.obj,
        source=EscalationSource(resolved=resolved, raw=loaded.raw),
    )


async def _write_escalation(
    context: LaneContext, record: Record[EscalationSource], after: dict[str, Any]
) -> WriteFailureRecord | None:
    failure = markup_sweep.write_repaired(record.source.resolved, record.source.raw, after)
    if failure is None:
        return None
    return WriteFailureRecord(key=record.key, reason=failure.reason, error=failure.error)


# ---------------------------------------------------------------------------
# The tasks lane: read-only sqlite scan, MCP update_task write.
# ---------------------------------------------------------------------------

LANE_TASKS = 'tasks'
TASK_FIELDS = ('title', 'description', 'details', 'test_strategy', 'metadata')
#: update_task has no test_strategy parameter, so that column is scanned only.
_TASK_TEXT_WRITABLE = ('title', 'description', 'details')
_TASK_SELECT = f"SELECT tag, id, {', '.join(TASK_FIELDS)} FROM tasks"
TASKS_STORE = 'tasks.db'
#: The server answered, and its answer was not an acceptance.
REASON_WRITE_REJECTED = 'write-rejected'
#: A task write that never got a readable answer: transport, HTTP status, a
#: JSON-RPC error, or a body that is not JSON.
_TASK_CALL_FAILURES = (httpx.HTTPError, OSError, RuntimeError, ValueError)


class WriteRejectedError(RuntimeError):
    pass


def assert_write_accepted(reply: object) -> None:
    """Raise unless *reply* carries a positive write signal. The only judge of a task reply."""
    reason = write_failure_reason(reply)
    if reason is not None:
        raise WriteRejectedError(reason)


class SweepFusedMemoryClient(FusedMemoryClient):
    _client_name = 'sweep-uuid-prefixes'


def open_live_task_client(server_url: str | None) -> SweepFusedMemoryClient:
    return SweepFusedMemoryClient(server_url or DEFAULT_SERVER)


class TaskSource(NamedTuple):
    root: Path
    tag: str
    task_id: int


def _task_fields(row: Sequence[Any]) -> dict[str, Any]:
    title, description, details, test_strategy, metadata = row[2:]
    return {
        'title': title or '',
        'description': description or '',
        'details': details or '',
        'test_strategy': test_strategy or '',
        'metadata': decode_metadata(metadata),
    }


def _task_rows(root: Path) -> list[Any]:
    with contextlib.closing(connect_ro(tasks_db_path(str(root)))) as conn:
        return conn.execute(f'{_TASK_SELECT} ORDER BY tag, id').fetchall()


def _required_task_client(context: LaneContext) -> TaskClientSlot:
    if context.tasks is None:
        raise ValueError('a tasks-lane write needs a task client and none was injected')
    return context.tasks


def _current_task_fields(source: TaskSource) -> dict[str, Any] | None:
    with contextlib.closing(connect_ro(tasks_db_path(str(source.root)))) as conn:
        row = conn.execute(
            f'{_TASK_SELECT} WHERE tag = ? AND id = ?', (source.tag, source.task_id)
        ).fetchone()
    return None if row is None else _task_fields(row)


async def _task_population(
    context: LaneContext,
) -> AsyncIterator[Record[TaskSource] | markup_sweep.Refusal]:
    root = _required_root(context.scope, LANE_TASKS)
    try:
        rows = _task_rows(root)
    except UNREADABLE_STORE_ERRORS as exc:
        raise StoreUnavailable(TASKS_STORE) from exc
    for row in rows:
        tag, task_id = row[0], row[1]
        yield Record(
            key=f'{tag}#{task_id}',
            fields=_task_fields(row),
            source=TaskSource(root=root, tag=tag, task_id=task_id),
        )


def build_task_update(
    root: Path,
    tag: str,
    task_id: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, str]:
    """The ``update_task`` arguments for exactly what changed between *before* and *after*.

    Metadata goes as a shallow ``merge`` of the changed top-level keys only, so
    no sibling key a concurrent writer set, and no wait anchor a ``replace``
    would strip, is ever touched.
    """
    payload = {'id': str(task_id), 'project_root': str(root), 'tag': tag}
    for column in _TASK_TEXT_WRITABLE:
        if after[column] != before[column]:
            payload[column] = after[column]
    changed = {
        key: value
        for key, value in after['metadata'].items()
        if key not in before['metadata'] or before['metadata'][key] != value
    }
    if changed:
        payload['metadata'] = json.dumps(changed)
        payload['metadata_mode'] = 'merge'
    return payload


async def _write_task(
    context: LaneContext, record: Record[TaskSource], after: dict[str, Any]
) -> WriteFailureRecord | None:
    source = record.source
    tasks = _required_task_client(context)
    try:
        current = _current_task_fields(source)
    except UNREADABLE_STORE_ERRORS as exc:
        return WriteFailureRecord(
            key=record.key, reason=markup_sweep.REASON_CHANGED_UNDER_US, error=f're-read failed: {exc}'
        )
    if current != record.fields:
        return WriteFailureRecord(
            key=record.key,
            reason=markup_sweep.REASON_CHANGED_UNDER_US,
            error='the task changed between the scan and the write',
        )
    payload = build_task_update(source.root, source.tag, source.task_id, record.fields, after)
    try:
        client = await tasks.client()
        assert_write_accepted(await client.call_tool('update_task', payload))
    except WriteRejectedError as exc:
        return WriteFailureRecord(key=record.key, reason=REASON_WRITE_REJECTED, error=str(exc))
    except _TASK_CALL_FAILURES as exc:
        return WriteFailureRecord(
            key=record.key,
            reason=markup_sweep.REASON_WRITE_FAILED,
            error=f'{type(exc).__name__}: {exc}',
        )
    return None


# ---------------------------------------------------------------------------
# The mem0 lane: raw point bodies, content-only amends that keep the point id.
# ---------------------------------------------------------------------------

LANE_MEM0 = 'mem0'
MEM0_STORE = 'mem0'
#: Attribution on every amend: the write's agent_id and its _source.
WRITE_SOURCE = 'sweep_uuid_prefixes'
WRITE_REASON = 'uuid-prefix corpus sweep (PRD C6, task 5325)'
_SCROLL_PAGE_SIZE = 1000


class MemoryStore(Protocol):
    """The slice of ``MemoryService`` the adapter uses."""

    config: Any
    mem0: Any

    async def get_memory_by_id(self, *, project_id: str, memory_id: str) -> dict | None: ...

    async def update_memory(
        self,
        *,
        memory_id: str,
        project_id: str,
        content: str,
        reason: str,
        agent_id: str,
        _source: str,
    ) -> dict: ...


class Mem0Failures(NamedTuple):
    """How the adapter tells a backend failure from a bug in this script."""

    #: What the backend raises when a call cannot complete.
    backend: tuple[type[BaseException], ...]
    #: True for the backend's "that collection does not exist" failure.
    is_missing_collection: Callable[[BaseException], bool]


def live_mem0_failures() -> Mem0Failures:
    """The live backend's failure types.

    The imports are deferred for the reason ``open_live_stores`` gives.
    """
    import openai
    from fused_memory.backends.mem0_client import is_missing_collection_error
    from qdrant_client.http.exceptions import ApiException

    return Mem0Failures(
        backend=(ApiException, openai.APIError, httpx.HTTPError, TimeoutError, OSError),
        is_missing_collection=is_missing_collection_error,
    )


class ServiceMem0Corpus:
    """``Mem0Corpus`` over a live ``MemoryService``."""

    def __init__(self, memory: MemoryStore, failures: Mem0Failures) -> None:
        self._memory = memory
        self._failures = failures

    async def bodies(self, project_id: str) -> AsyncIterator[tuple[str, str]]:
        """Every point whose ``data`` is text, in scroll order.

        A collection that was never created holds no memories. Any other
        backend failure is an outage, never an empty corpus.
        """
        collection = Scope(project_id=project_id).mem0_collection_name(
            self._memory.config.mem0.collection_prefix
        )
        points = self._memory.mem0.scroll_collection_pages(
            collection, page_size=_SCROLL_PAGE_SIZE, with_payload=True
        )
        try:
            async with contextlib.aclosing(points):
                async for point in points:
                    body = (point.payload or {}).get('data')
                    if isinstance(body, str):
                        yield str(point.id), body
        except self._failures.backend as exc:
            if not self._failures.is_missing_collection(exc):
                raise StoreUnavailable(MEM0_STORE) from exc
            logger.warning('mem0 collection %s does not exist, so it holds no memories', collection)

    async def read_body(self, project_id: str, memory_id: str) -> str | None:
        with self._backend_call():
            record = await self._memory.get_memory_by_id(project_id=project_id, memory_id=memory_id)
        if record is None:
            return None
        body = record['metadata'].get('data')
        return body if isinstance(body, str) else None

    async def amend_body(self, project_id: str, memory_id: str, body: str) -> Mapping[str, Any]:
        """A content-only amend: same point id, every metadata key left as stored."""
        with self._backend_call():
            return await self._memory.update_memory(
                memory_id=memory_id,
                project_id=project_id,
                content=body,
                reason=WRITE_REASON,
                agent_id=WRITE_SOURCE,
                _source=WRITE_SOURCE,
            )

    @contextlib.contextmanager
    def _backend_call(self) -> Iterator[None]:
        try:
            yield
        except self._failures.backend as exc:
            raise Mem0CallFailed(f'{type(exc).__name__}: {exc}') from exc


class Mem0Source(NamedTuple):
    memory_id: str


def _required_corpus(context: LaneContext) -> Mem0Corpus:
    if context.mem0 is None:
        raise ValueError('the mem0 lane needs a corpus and none was injected')
    return context.mem0


async def _mem0_population(
    context: LaneContext,
) -> AsyncIterator[Record[Mem0Source] | markup_sweep.Refusal]:
    corpus = _required_corpus(context)
    async for memory_id, body in corpus.bodies(context.scope.project_id):
        yield Record(key=memory_id, fields={'data': body}, source=Mem0Source(memory_id))


def _amend_rejection(reply: Mapping[str, Any]) -> str | None:
    """Why *reply* is not an applied amend, or ``None`` when it is one."""
    if reply.get('error_type'):
        return f"{reply['error_type']}: {reply.get('error', '')}"
    if reply.get('status') != 'updated':
        return f'the reply carries no positive write signal: {dict(reply)!r}'
    return None


async def _write_mem0(
    context: LaneContext, record: Record[Mem0Source], after: dict[str, Any]
) -> WriteFailureRecord | None:
    source = record.source
    corpus = _required_corpus(context)
    project_id = context.scope.project_id
    try:
        current = await corpus.read_body(project_id, source.memory_id)
        if current != record.fields['data']:
            return WriteFailureRecord(
                key=record.key,
                reason=markup_sweep.REASON_CHANGED_UNDER_US,
                error='the memory body changed between the scan and the write',
            )
        reply = await corpus.amend_body(project_id, source.memory_id, after['data'])
    except Mem0CallFailed as exc:
        return WriteFailureRecord(key=record.key, reason=markup_sweep.REASON_WRITE_FAILED, error=str(exc))
    rejection = _amend_rejection(reply)
    if rejection is not None:
        return WriteFailureRecord(key=record.key, reason=REASON_WRITE_REJECTED, error=rejection)
    return None


#: The declared lane table. Order is the order ``--lane all`` runs them in.
#: Each lane's source type is checked where the lane is built; the table only
#: holds them side by side.
LANES: Mapping[str, Lane[Any]] = MappingProxyType({
    lane.name: lane
    for lane in (
        Lane(
            name=LANE_MEM0,
            needs_root=False,
            unwritable_fields=frozenset(),
            population=_mem0_population,
            write=_write_mem0,
        ),
        Lane(
            name=LANE_TASKS,
            needs_root=True,
            unwritable_fields=frozenset({'test_strategy'}),
            population=_task_population,
            write=_write_task,
        ),
        Lane(
            name=markup_sweep.LANE_ESCALATIONS,
            needs_root=True,
            unwritable_fields=frozenset(),
            population=_escalation_population,
            write=_write_escalation,
        ),
    )
})


# ---------------------------------------------------------------------------
# Production stores, and the CLI that composes everything.
# ---------------------------------------------------------------------------

_CHECKOUT_ROOT = Path(__file__).resolve().parent.parent


class LiveStores(NamedTuple):
    resolve: PrefixResolver
    mem0: Mem0Corpus


class StoresOpener(Protocol):
    def __call__(
        self, config_path: str | None, *, mem0_apply: bool
    ) -> contextlib.AbstractAsyncContextManager[LiveStores]: ...


@contextlib.asynccontextmanager
async def open_live_stores(config_path: str | None, *, mem0_apply: bool) -> AsyncIterator[LiveStores]:
    """An initialized ``MemoryService``'s resolver and mem0 corpus, closed on the way out.

    The imports are deferred because they pull graphiti_core, which the
    scripts test environment does not install; nothing else here needs them.
    The store-mutation preflight runs before ``initialize`` when mem0 writes
    are requested, so a process that cannot write the store never starts.
    """
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.services.id_resolver import IdResolver
    from fused_memory.services.memory_service import MemoryService
    from fused_memory.utils.store_mutation_preflight import assert_store_mutation_allowed

    if config_path is not None:
        os.environ['CONFIG_PATH'] = config_path
    if mem0_apply:
        assert_store_mutation_allowed(operation='sweep_uuid_prefixes --apply --lane mem0')
    memory = MemoryService(FusedMemoryConfig())
    try:
        await memory.initialize()
        yield LiveStores(
            resolve=IdResolver.for_memory_service(memory).resolve_prefix,
            mem0=ServiceMem0Corpus(memory, live_mem0_failures()),
        )
    finally:
        await memory.close()


def _positive_rate(text: str) -> float:
    try:
        rate = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a number: {text!r}') from None
    if not rate > 0:
        raise argparse.ArgumentTypeError(f'must be more than 0 writes per second, got {text}')
    return rate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Expand stored truncated-uuid citations to full ids (PRD C6).'
    )
    parser.add_argument(
        '--project', required=True, help='the project whose stores are swept and resolved against'
    )
    parser.add_argument(
        '--project-root', help='the checkout holding the project tasks.db and data/escalations'
    )
    parser.add_argument('--lane', choices=(*LANES, 'all'), default='all')
    parser.add_argument(
        '--apply',
        action='store_true',
        help='write the expansions; without it the run is a dry run that prints diffs',
    )
    parser.add_argument(
        '--rate',
        type=_positive_rate,
        default=DEFAULT_RATE,
        help=f'writes per second in every lane (default {DEFAULT_RATE})',
    )
    parser.add_argument(
        '--json', action='store_true', help='print one JSON document instead of diffs and summaries'
    )
    parser.add_argument('--config', help='fused-memory config file; sets CONFIG_PATH')
    parser.add_argument(
        '--server-url', help=f'fused-memory MCP server for task writes (default {DEFAULT_SERVER})'
    )
    return parser


def _selected_lanes(choice: str) -> tuple[str, ...]:
    return tuple(LANES) if choice == 'all' else (choice,)


def resolve_scope(args: argparse.Namespace, parser: argparse.ArgumentParser) -> SweepScope:
    """The canonical project and its checkout, or a usage error naming the remedy."""
    try:
        project_id = Scope(project_id=args.project).project_id
    except ValueError as exc:
        parser.error(f'--project {args.project!r} is not a project id: {exc}')
    root = _project_root(args.project_root, project_id, parser)
    needing = [name for name in _selected_lanes(args.lane) if LANES[name].needs_root]
    if root is None and needing:
        parser.error(
            f'lanes {needing} read files under a project root, and none is known for '
            f'{project_id!r}: pass --project-root, or list the checkout in {KNOWN_PROJECT_ROOTS_ENV}'
        )
    return SweepScope(project_id=project_id, project_root=root)


def _project_root(
    given: str | None, project_id: str, parser: argparse.ArgumentParser
) -> Path | None:
    if given is None:
        known = build_known_projects_map(str(_CHECKOUT_ROOT)).get(project_id)
        return None if known is None else Path(known)
    root = Path(given).resolve()
    declared = resolve_project_id_for_root(root)
    if declared != project_id:
        parser.error(
            f'--project-root {root} belongs to project {declared!r}, not --project {project_id!r}'
        )
    return root


def render_report(report: LaneReport) -> list[str]:
    """The operator's text view of one lane: a summary line, then one line per site."""
    lines = [
        f'lane={report.lane} project={report.project_id} scanned={report.scanned} '
        f'expandable={report.expandable} expanded={report.expanded} '
        f'ambiguous={len(report.ambiguous)} untouched={report.untouched} '
        f'unwritable={len(report.unwritable)} refused={sum(report.refused.values())} '
        f'failed={len(report.failures)}'
    ]
    lines += [f'refused: {reason} {count}' for reason, count in sorted(report.refused.items())]
    lines += [
        f'ambiguous: {site.record} {format_path(site.path)} {site.token} '
        f"candidates={','.join(site.candidates)}"
        for site in report.ambiguous
    ]
    lines += [
        f'unwritable: {site.record} {format_path(site.path)} {site.token}'
        for site in report.unwritable
    ]
    lines += [f'failed: {failure.key} {failure.reason}: {failure.error}' for failure in report.failures]
    if report.aborted is not None:
        lines.append(f'aborted: store={report.aborted}')
    lines.append(f'{report.lane}: {report.remaining} expandable remaining')
    return lines


async def _sweep_with_stores(
    args: argparse.Namespace,
    scope: SweepScope,
    open_stores: StoresOpener,
    open_task_client: TaskClientOpener,
) -> tuple[tuple[LaneReport, ...], tuple[str, ...]]:
    lanes = _selected_lanes(args.lane)
    mem0_apply = args.apply and LANE_MEM0 in lanes
    async with open_stores(args.config, mem0_apply=mem0_apply) as stores:
        return await run_sweep(
            scope,
            lanes=lanes,
            apply=args.apply,
            resolve=stores.resolve,
            mem0=stores.mem0,
            open_task_client=open_task_client,
            server_url=args.server_url,
            rate=args.rate,
        )


def main(
    argv: list[str] | None = None,
    *,
    open_stores: StoresOpener = open_live_stores,
    open_task_client: TaskClientOpener = open_live_task_client,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    scope = resolve_scope(args, parser)
    reports, diffs = asyncio.run(_sweep_with_stores(args, scope, open_stores, open_task_client))
    code = exit_code(reports)
    if args.json:
        print(json.dumps({'lanes': [report.as_dict() for report in reports], 'exit_code': code}))
        return code
    for diff in diffs:
        print(diff)
    for report in reports:
        print('\n'.join(render_report(report)))
    return code


if __name__ == '__main__':
    sys.exit(main())
