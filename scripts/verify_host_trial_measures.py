"""Compute the verify-host trial report's per-block figures from one project's typed store rows.

Each figure is one small pure function of a block, returning a Figure: a
measured value, or a state saying why there is none (no host verify, a
producer that never wrote, or an unavailable source). It is never a silent 0.
FIGURE_RULES is the one home of each figure's rule text, which both renderers
print.
"""
from __future__ import annotations

import bisect
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

from verify_host_trial_blocks import Block
from verify_host_trial_stores import (
    CPU_ARM_SPECS,
    LOCK_CONFLICT_REASONS,
    REQUEUED,
    SOURCES,
    CpuRead,
    LegHold,
    ProjectRead,
    Source,
    SourceState,
    TaskCompletion,
    TaskStart,
)

if TYPE_CHECKING:
    from sampler.metrics import MetricSpec

__all__ = [
    'FIGURE_RULES',
    'CpuStat',
    'DeferralHours',
    'Figure',
    'FigureId',
    'Measured',
    'NeverWritten',
    'NoHostVerify',
    'RunnerMedian',
    'TurnWait',
    'Unavailable',
    'cpu_pressure',
    'deferral_hours_by_metric',
    'host_verifies',
    'landings',
    'landings_per_day',
    'lock_conflict_requeues',
    'merge_verify_median_by_runner',
    'project_figures',
    'spend_per_day',
    'turn_wait_share',
]

V = TypeVar('V')


@dataclass(frozen=True)
class Measured(Generic[V]):
    value: V


@dataclass(frozen=True)
class NoHostVerify:
    """No merge_verify started on the host in this block and project."""


@dataclass(frozen=True)
class NeverWritten:
    source: str
    """The rows whose producer never wrote one in this store."""


@dataclass(frozen=True)
class Unavailable:
    reason: str


Figure: TypeAlias = Measured[V] | NoHostVerify | NeverWritten | Unavailable


class FigureId(StrEnum):
    HOST_VERIFIES = 'host_verifies'
    LANDINGS_PER_DAY = 'landings_per_day'
    MERGE_VERIFY_MEDIAN = 'merge_verify_median_by_runner'
    TURN_WAIT_SHARE = 'turn_wait_share'
    DEFERRAL_HOURS = 'deferral_hours_by_metric'
    LOCK_CONFLICT_REQUEUES = 'lock_conflict_requeues'
    SPEND_PER_DAY = 'spend_per_day'
    CPU_PRESSURE = 'cpu_pressure'


_MERGE_VERIFY_START = 'start = event timestamp (the END) - data.duration_ms'
_CPU_ARMS = ', '.join(f'{spec.arm} ({spec.name})' for spec in CPU_ARM_SPECS)

FIGURE_RULES: Mapping[FigureId, str] = MappingProxyType({
    FigureId.HOST_VERIFIES: (
        f'host verifies: merge_verify rows whose runner is --host, each in the block holding its start '
        f'({_MERGE_VERIFY_START}); none is "no verify ran on this host", never 0'
    ),
    FigureId.LANDINGS_PER_DAY: (
        'landings per day: distinct tasks whose first done row over the whole store '
        '(merge_finalized state done, or task_results outcome done) falls in the block, over its days'
    ),
    FigureId.MERGE_VERIFY_MEDIAN: (
        f'merge_verify median wall by runner, in seconds, with n: data.duration_ms of each row, '
        f'in the block holding its start ({_MERGE_VERIFY_START})'
    ),
    FigureId.TURN_WAIT_SHARE: (
        'share of run-slot time waiting for a task-verify turn: the time in [plan_arrival_at, completed_at] '
        'of role=task verify_admission_plan rows that no leg held a slot ([granted_at, granted_at + held_secs]), '
        'over task_started to next task_completed time of the same run and task, both clipped to the block; '
        'an unpaired task_started is excluded and counted; a task verify a remote host won leaves no '
        'admission row, so its (near-zero) wait is not counted'
    ),
    FigureId.DEFERRAL_HOURS: (
        'dispatch_deferred hours by gate arm: each row spans from the first scheduler_tick of its run at or '
        'after it (the end of its own tick) to the next scheduler_tick of that run, in the block of its own '
        'timestamp; a row with no following tick in its run adds no hours and is counted as unspanned'
    ),
    FigureId.LOCK_CONFLICT_REQUEUES: (
        f'lock-conflict requeues by reason: task_completed rows with outcome {REQUEUED} and reason in '
        f'{", ".join(sorted(LOCK_CONFLICT_REASONS))}, in the block of their timestamp'
    ),
    FigureId.SPEND_PER_DAY: (
        'agent spend per day: the sum of invocations.cost_usd (list USD) with completed_at in the block, '
        'over its days'
    ),
    FigureId.CPU_PRESSURE: (
        f'workstation CPU pressure: mean and nearest-rank p90 of each CPU gate arm\'s host metric ({_CPU_ARMS}, '
        'named through sampler.metrics.ARM_METRICS) over the samples in the block; a sample whose readability '
        'metric read 0 at the same ts is dropped'
    ),
})


# ---------------------------------------------------------------------------
# Producer proof
# ---------------------------------------------------------------------------


def _gate(project: ProjectRead, sources: Sequence[Source], *, needs_all: bool = True) -> Unavailable | NeverWritten | None:
    """Why the figure over *sources* cannot be measured, or None.

    Any unavailable source makes it unavailable. With *needs_all* any source
    never written makes it never written; otherwise only all of them do.
    """
    reads = [project.sources.get(source) for source in sources]
    for source, read in zip(sources, reads, strict=True):
        if read is None:
            return Unavailable(f'{source} was not read')
        if read.state is SourceState.UNAVAILABLE:
            return Unavailable(read.reason or str(source))
    never = [source for source, read in zip(sources, reads, strict=True) if read and read.state is SourceState.NEVER_WRITTEN]
    if never and (needs_all or len(never) == len(sources)):
        return NeverWritten(' or '.join(SOURCES[source].producer for source in (never if needs_all else sources)))
    return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def landings(project: ProjectRead, block: Block) -> Figure[int]:
    """The tasks whose first landing falls in *block*."""
    blocked = _gate(project, (Source.MERGE_FINALIZED, Source.TASK_RESULTS), needs_all=False)
    if blocked is not None:
        return blocked
    return Measured(sum(1 for landing in project.landings if block.contains(landing.at)))


def landings_per_day(project: ProjectRead, block: Block) -> Figure[float]:
    count = landings(project, block)
    return Measured(count.value / block.days) if isinstance(count, Measured) else count


@dataclass(frozen=True)
class RunnerMedian:
    median_secs: float
    n: int


def merge_verify_median_by_runner(project: ProjectRead, block: Block) -> Figure[Mapping[str, RunnerMedian]]:
    blocked = _gate(project, (Source.MERGE_VERIFY,))
    if blocked is not None:
        return blocked
    walls: dict[str, list[float]] = defaultdict(list)
    for verify in project.merge_verifies:
        if block.contains(verify.start):
            walls[verify.runner].append((verify.end - verify.start).total_seconds())
    return Measured(MappingProxyType({
        runner: RunnerMedian(statistics.median(secs), len(secs)) for runner, secs in sorted(walls.items())
    }))


def host_verifies(project: ProjectRead, host: str, block: Block) -> Figure[int]:
    blocked = _gate(project, (Source.MERGE_VERIFY,))
    if blocked is not None:
        return blocked
    count = sum(1 for verify in project.merge_verifies if verify.runner == host and block.contains(verify.start))
    return Measured(count) if count else NoHostVerify()


@dataclass(frozen=True)
class TurnWait:
    share: float
    wait_secs: float
    slot_secs: float
    unpaired_starts: int


def _clip(start: datetime, end: datetime, block: Block) -> tuple[datetime, datetime] | None:
    start, end = max(start, block.start), min(end, block.end)
    return (start, end) if start < end else None


def _union_secs(spans: Iterable[tuple[datetime, datetime]]) -> float:
    total = 0.0
    reach: datetime | None = None
    for start, end in sorted(spans):
        if reach is not None and start < reach:
            start = reach
        if start < end:
            total += (end - start).total_seconds()
            reach = end if reach is None else max(reach, end)
    return total


def _uncovered_secs(span: tuple[datetime, datetime] | None, legs: Iterable[LegHold]) -> float:
    """The measure of *span* that no leg's hold covers."""
    if span is None:
        return 0.0
    start, end = span
    held = [(max(leg.start, start), min(leg.end, end)) for leg in legs]
    return (end - start).total_seconds() - _union_secs(held)


def _run_slots(
    starts: Iterable[TaskStart], completions: Iterable[TaskCompletion]
) -> tuple[list[tuple[datetime, datetime]], list[TaskStart]]:
    """Each start paired with the next completion of its run and task before its next start; and the unpaired."""
    ends: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for completion in completions:
        ends[completion.run_id, completion.task_id].append(completion.at)
    by_key: dict[tuple[str, str], list[TaskStart]] = defaultdict(list)
    for start in starts:
        by_key[start.run_id, start.task_id].append(start)
    slots: list[tuple[datetime, datetime]] = []
    unpaired: list[TaskStart] = []
    for key, key_starts in by_key.items():
        key_ends = sorted(ends.get(key, ()))
        key_starts.sort(key=lambda start: start.at)
        for start, following in zip(key_starts, [*key_starts[1:], None], strict=True):
            i = bisect.bisect_left(key_ends, start.at)
            if i < len(key_ends) and (following is None or key_ends[i] <= following.at):
                slots.append((start.at, key_ends[i]))
            else:
                unpaired.append(start)
    return slots, unpaired


def turn_wait_share(project: ProjectRead, block: Block) -> Figure[TurnWait]:
    blocked = _gate(project, (Source.ADMISSION_PLAN, Source.TASK_STARTED, Source.TASK_COMPLETED))
    if blocked is not None:
        return blocked
    wait = sum(_uncovered_secs(_clip(plan.arrival, plan.completed, block), plan.legs) for plan in project.admission_plans)
    slots, unpaired = _run_slots(project.task_starts, project.task_completions)
    clipped = [span for span in (_clip(start, end, block) for start, end in slots) if span is not None]
    slot_secs = sum((end - start).total_seconds() for start, end in clipped)
    if slot_secs <= 0:
        return Unavailable('no task run-slot time (task_started to task_completed) in this block')
    unpaired_here = sum(1 for start in unpaired if block.contains(start.at))
    return Measured(TurnWait(wait / slot_secs, wait, slot_secs, unpaired_here))


@dataclass(frozen=True)
class DeferralHours:
    hours: Mapping[str, float]
    """Hours held, by the shared.psi gate arm that held them."""
    unspanned: int


def _tick_span_secs(run_ticks: Sequence[datetime], at: datetime) -> float | None:
    i = bisect.bisect_left(run_ticks, at)
    return (run_ticks[i + 1] - run_ticks[i]).total_seconds() if i + 1 < len(run_ticks) else None


def deferral_hours_by_metric(project: ProjectRead, block: Block) -> Figure[DeferralHours]:
    blocked = _gate(project, (Source.DISPATCH_DEFERRED, Source.SCHEDULER_TICK))
    if blocked is not None:
        return blocked
    ticks: dict[str, list[datetime]] = defaultdict(list)
    for tick in project.ticks:
        ticks[tick.run_id].append(tick.at)
    for run_ticks in ticks.values():
        run_ticks.sort()
    hours: dict[str, float] = defaultdict(float)
    unspanned = 0
    for deferral in project.deferrals:
        if not block.contains(deferral.at):
            continue
        span = _tick_span_secs(ticks.get(deferral.run_id, []), deferral.at)
        if span is None:
            unspanned += 1
        else:
            hours[deferral.metric] += span / 3600
    return Measured(DeferralHours(MappingProxyType(dict(sorted(hours.items()))), unspanned))


def lock_conflict_requeues(project: ProjectRead, block: Block) -> Figure[Mapping[str, int]]:
    blocked = _gate(project, (Source.TASK_COMPLETED,))
    if blocked is not None:
        return blocked
    counts = Counter(
        completion.reason
        for completion in project.task_completions
        if block.contains(completion.at) and completion.outcome == REQUEUED and completion.reason in LOCK_CONFLICT_REASONS
    )
    return Measured(MappingProxyType(dict(sorted(counts.items()))))


def spend_per_day(project: ProjectRead, block: Block) -> Figure[float]:
    blocked = _gate(project, (Source.INVOCATIONS,))
    if blocked is not None:
        return blocked
    return Measured(sum(i.cost_usd for i in project.invocations if block.contains(i.completed_at)) / block.days)


@dataclass(frozen=True)
class CpuStat:
    mean: float
    p90: float
    n: int


def _cpu_stat(values: Sequence[float]) -> CpuStat | None:
    if not values:
        return None
    ordered = sorted(values)
    return CpuStat(statistics.fmean(ordered), ordered[max(1, math.ceil(0.9 * len(ordered))) - 1], len(ordered))


def cpu_pressure(cpu: CpuRead, block: Block) -> Figure[Mapping[MetricSpec, CpuStat | None]]:
    """Per CPU arm spec, the stat over the block's samples; None for an arm with no sample there."""
    if cpu.reason is not None:
        return Unavailable(cpu.reason)
    stats = {
        spec: _cpu_stat([sample.value for sample in samples if block.contains(sample.at)])
        for spec, samples in cpu.samples.items()
    }
    if all(stat is None for stat in stats.values()):
        return Unavailable('no CPU sample in this block')
    return Measured(MappingProxyType(stats))


def project_figures(project: ProjectRead, host: str, block: Block) -> Mapping[FigureId, Figure[Any]]:
    """Every project-level figure of *project* in *block*, in report order."""
    return MappingProxyType({
        FigureId.HOST_VERIFIES: host_verifies(project, host, block),
        FigureId.LANDINGS_PER_DAY: landings_per_day(project, block),
        FigureId.MERGE_VERIFY_MEDIAN: merge_verify_median_by_runner(project, block),
        FigureId.TURN_WAIT_SHARE: turn_wait_share(project, block),
        FigureId.DEFERRAL_HOURS: deferral_hours_by_metric(project, block),
        FigureId.LOCK_CONFLICT_REQUEUES: lock_conflict_requeues(project, block),
        FigureId.SPEND_PER_DAY: spend_per_day(project, block),
    })
