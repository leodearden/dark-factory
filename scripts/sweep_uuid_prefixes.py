"""Expand stored truncated-uuid citations to the full ids they name.

PRD contract C6, ``plans/uuid-prefix-resolution-prd.md`` §4-C6, which owns the
lane table and the rules; this docstring does not restate them (INV-9). Every
token comes from ``shared.uuid_prefix`` (C1) and every answer from the
``PrefixResolver`` port (C2), resolved against the record's own project. Only
``unique`` tokens are expanded; ``ambiguous`` ones are reported by record and
left byte-identical, ``none`` ones are counted and left alone.

Run it one project per sitting, dry run first (the default), then one lane at
a time::

    uv run --project fused-memory python scripts/sweep_uuid_prefixes.py \\
        --project <id> [--project-root <checkout>] [--lane all]
    uv run --project fused-memory python scripts/sweep_uuid_prefixes.py \\
        --project <id> [--project-root <checkout>] --apply --lane <lane>

A second ``--apply`` of the same lane must report ``0 expandable remaining``
and exit 0.
"""
from __future__ import annotations

import asyncio
import difflib
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, TypeVar

# Bind `shared` and `fused_memory` to THIS checkout, the
# scripts/check_consolidation_closure.py idiom: the editable installs are
# ordinary .pth entries, so sys.path order decides which copy a worktree run
# imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _src in (_REPO_ROOT / 'fused-memory' / 'src', _REPO_ROOT / 'shared' / 'src'):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import sweep_toolcall_markup as markup_sweep  # noqa: E402
from shared.uuid_prefix import PrefixToken, find_prefix_tokens, substitute_all  # noqa: E402
from shared.uuid_prefix_guard import (  # noqa: E402
    PrefixResolver,
    Resolution,
    ResolverUnavailable,
)

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


class Record(NamedTuple):
    """One sweepable document.

    ``source`` is the owning lane's locator for the write; only that lane's
    writer reads it.
    """

    key: str
    fields: dict[str, Any]
    source: object


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


class LaneContext(NamedTuple):
    """What a lane's population and writer may reach during one run."""

    scope: SweepScope


Population = Callable[[LaneContext], AsyncIterator['Record | markup_sweep.Refusal']]
Writer = Callable[[LaneContext, Record, dict[str, Any]], Awaitable['WriteFailureRecord | None']]


class Lane(NamedTuple):
    name: str
    #: The lane reads a checkout's files, so a run needs a project root.
    needs_root: bool
    #: Top-level fields scanned for the report but never written.
    unwritable_fields: frozenset[str]
    population: Population
    write: Writer


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


def render_diffs(lane: str, record: Record, plan: RecordPlan) -> list[str]:
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
    mem0: object | None = None,
    open_task_client: object | None = None,
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
    context = LaneContext(scope=scope)
    pacer = WritePacer(rate)
    memo: dict[str, Resolution] = {}
    reports: list[LaneReport] = []
    diffs: list[str] = []
    for lane in (lane for name, lane in LANES.items() if name in selected):
        report, lane_diffs, memo = await _sweep_lane(
            lane, context, resolve, memo, apply=apply, pacer=pacer
        )
        reports.append(report)
        diffs.extend(lane_diffs)
    return tuple(reports), tuple(diffs)


async def _sweep_lane(
    lane: Lane,
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
    except ResolverUnavailable as outage:
        return _aborted_report(lane.name, project_id, outage.store), (), memo
    plans = [(record, plan_record(record.fields, memo, lane.unwritable_fields)) for record in records]
    if not apply:
        diffs = tuple(diff for record, plan in plans for diff in render_diffs(lane.name, record, plan))
        return _lane_report(lane.name, project_id, plans, refused, expanded=0, failures=()), diffs, memo
    expanded, failures = await _write_all(lane, context, plans, pacer)
    report = _lane_report(lane.name, project_id, plans, refused, expanded=expanded, failures=failures)
    return report, (), memo


async def _collect(lane: Lane, context: LaneContext) -> tuple[list[Record], Counter[str]]:
    records: list[Record] = []
    refused: Counter[str] = Counter()
    async for item in lane.population(context):
        if isinstance(item, Record):
            records.append(item)
        else:
            refused[item.reason] += 1
    return records, refused


async def _write_all(
    lane: Lane,
    context: LaneContext,
    plans: list[tuple[Record, RecordPlan]],
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
    plans: list[tuple[Record, RecordPlan]],
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

_Source = TypeVar('_Source')


def _source_of(record: Record, kind: type[_Source], lane: str) -> _Source:
    """*record*'s locator, narrowed to the type the *lane* writer owns."""
    if not isinstance(record.source, kind):
        raise TypeError(
            f'the {lane} lane was handed a {type(record.source).__name__} source, '
            f'not its own {kind.__name__}'
        )
    return record.source


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
) -> AsyncIterator[Record | markup_sweep.Refusal]:
    root = _required_root(context.scope, markup_sweep.LANE_ESCALATIONS)
    targets = [
        target
        for target in markup_sweep.discover_targets(root)
        if target.lane == markup_sweep.LANE_ESCALATIONS
    ]
    for target in markup_sweep.dedupe_by_realpath(targets):
        yield _escalation_record(target, root)


def _escalation_record(target: markup_sweep.Target, root: Path) -> Record | markup_sweep.Refusal:
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
    context: LaneContext, record: Record, after: dict[str, Any]
) -> WriteFailureRecord | None:
    source = _source_of(record, EscalationSource, markup_sweep.LANE_ESCALATIONS)
    failure = markup_sweep.write_repaired(source.resolved, source.raw, after)
    if failure is None:
        return None
    return WriteFailureRecord(key=record.key, reason=failure.reason, error=failure.error)


#: The declared lane table. Order is the order ``--lane all`` runs them in.
LANES: Mapping[str, Lane] = MappingProxyType({
    lane.name: lane
    for lane in (
        Lane(
            name=markup_sweep.LANE_ESCALATIONS,
            needs_root=True,
            unwritable_fields=frozenset(),
            population=_escalation_population,
            write=_write_escalation,
        ),
    )
})
