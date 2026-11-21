"""The ``holds`` report rendered: the text an operator reads, and the JSON a script reads.

It renders what scripts/resource_report_holds.py reads and analyses, and is
never imported back. Every consumer name is encoded by ``sampler.consumers``;
no metric name is spelled here (sampler/tests/test_vocabulary_one_table.py
scans it).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

import _checkout as _checkout
from resource_report_holds import (
    RANKINGS,
    Aside,
    ConsumerWindow,
    Hold,
    HoldConsumers,
    HoldReading,
    HoldsReport,
    HoldState,
    ProjectConsumer,
    RootRead,
    RootStatus,
    SamplerGap,
    Summary,
    Totals,
    Window,
    aside,
    holding_project,
    incomplete,
    reclaim_threads_share,
    summarise,
    who_grew,
    who_paid,
)
from sampler.consumers import Consumer, Role, encode_consumer

__all__ = ['format_utc', 'render_text', 'report_json']


def format_utc(moment: datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ``: the one rendering of a time in the report."""
    return moment.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


_STATE_TEXT: Mapping[HoldState, str] = MappingProxyType({
    HoldState.RANKED: 'ranked over the window',
    HoldState.NOT_LIVE: 'recorder not live at this hold',
    HoldState.NO_SAMPLES: (
        'no samples: under two consumer collections in the window, '
        'not counting one just after a gap'
    ),
    HoldState.GAP: 'sampler gap in the window, so it is not ranked',
})


def _root_line(read: RootRead) -> str:
    if read.status is RootStatus.MISSING:
        found = f'runs.db missing: {read.runs_db}'
    elif read.status is RootStatus.UNREADABLE:
        found = f'runs.db unreadable: {read.runs_db} ({read.detail})'
    else:
        found = f'runs.db {read.runs_db}'
    skipped = f'; {read.skipped} hold events skipped, unparseable' if read.skipped else ''
    return f'  {read.root}: {found}{skipped}'


def _samples_line(report: HoldsReport) -> str:
    first, last = report.first_collection, report.last_collection
    if first is None or last is None:
        span = 'no consumer collections'
    else:
        span = f'consumer collections {format_utc(first)} to {format_utc(last)}'
    return f'samples db: {report.samples_db} ({span})'


def _hold_count_line(report: HoldsReport) -> str:
    count = len(report.readings)
    noun = 'memory hold' if count == 1 else 'memory holds'
    return f'{count} {noun} since {format_utc(report.since)}'


def _header(report: HoldsReport) -> list[str]:
    return [
        f'roots ({len(report.roots)}):',
        *(_root_line(read) for read in report.roots),
        _samples_line(report),
        f'window: {report.window_seconds} s before each hold',
        _hold_count_line(report),
    ]


def _heading(hold: Hold) -> str:
    value = 'n/a' if hold.arm_value is None else f'{hold.arm_value:g}'
    return f'hold {format_utc(hold.at)}  {hold.project_root}  task {hold.task_id}  {hold.arm} {value}'


_TOP = 5
_USEC_PER_SECOND = 1_000_000
_KiB = 2**10
_MiB = 2**20
_GiB = 2**30


def _bytes(value: float) -> str:
    sign = '-' if value < 0 else '+'
    size = abs(value)
    if size >= _GiB:
        return f'{sign}{size / _GiB:.2f} GiB'
    if size >= _MiB:
        return f'{sign}{size / _MiB:.1f} MiB'
    if size >= _KiB:
        return f'{sign}{size / _KiB:.0f} KiB'
    return f'{sign}{size:.0f} B'


def _pages(value: float) -> str:
    return f'{value:.0f} pages'


def _percent(share: float | None) -> str:
    return 'n/a' if share is None else f'{share:.1%}'


def _growth(c: ConsumerWindow) -> str:
    return (
        f'{_bytes(c.grew)} (program {_bytes(c.program_growth_bytes)}, '
        f'swap {_bytes(c.swap_growth_bytes)})'
    )


def _paid(c: ConsumerWindow) -> str:
    return (
        f'{_pages(c.reclaimed_pages)}, stall some {c.memory_some_stall_seconds:.1f} s, '
        f'full {c.memory_full_stall_seconds:.1f} s'
    )


def _ranking_lines(
    title: str,
    empty: str,
    ranking: Sequence[ConsumerWindow],
    figures: Callable[[ConsumerWindow], str],
    marks: Mapping[Consumer, str],
) -> list[str]:
    if not ranking:
        return [f'  {empty}']
    return [f'  {title}:'] + [
        f'    {i}. {encode_consumer(c.consumer)}  {figures(c)}{marks.get(c.consumer, "")}'
        for i, c in enumerate(ranking[:_TOP], start=1)
    ]


def _reclaim_threads(window: Window) -> str:
    usec = window.reclaim_threads_cpu_usec
    if usec is None:
        return 'reclaim threads unavailable'
    share = _percent(reclaim_threads_share(window))
    return f'reclaim threads {usec / _USEC_PER_SECOND:.1f} s CPU ({share} of host CPU)'


def _aside_line(row: Aside, window: Window) -> str:
    line = (
        f'  {row.label}: grew {_bytes(row.positive_growth)} ({_percent(row.growth_share)} of growth, '
        f'net {_bytes(row.net_grew)}), '
        f'{_pages(row.reclaimed_pages)} reclaimed ({_percent(row.reclaim_share)} of pages)'
    )
    return f'{line}; {_reclaim_threads(window)}' if row.role is Role.KERNEL else line


_INCOMPLETE_MARK = '  [incomplete]'


def _completeness(c: ConsumerWindow) -> str:
    return '' if c.complete else _INCOMPLETE_MARK


def _ranks(row: ProjectConsumer) -> str:
    def rank(value: int | None) -> str:
        return 'not ranked' if value is None else str(value)

    return f'who grew: {rank(row.who_grew_rank)}; who paid: {rank(row.who_paid_rank)}'


def _project_line(row: ProjectConsumer) -> str:
    if row.consumer is None:
        return f'    {row.role.value}: not mapped ({row.reason})'
    name = encode_consumer(row.consumer)
    if row.figures is None:
        return f'    {row.role.value}: {name} absent, no rows in the window'
    return (
        f'    {row.role.value}: {name} grew {_bytes(row.figures.grew)}, '
        f'{_pages(row.figures.reclaimed_pages)}; {_ranks(row)}{_completeness(row.figures)}'
    )


def _marks(mapping: HoldConsumers, window: Window) -> dict[Consumer, str]:
    project = {
        consumer: f"  [this project's {role.value}]"
        for role, consumer in mapping.by_role().items()
        if consumer is not None
    }
    return {
        consumer: project.get(consumer, '') + _completeness(row)
        for consumer, row in window.consumers.items()
    }


def _window_lines(mapping: HoldConsumers, window: Window) -> list[str]:
    marks = _marks(mapping, window)
    unreadable = incomplete(window)
    return [
        f'  window {format_utc(window.start)} to {format_utc(window.end)}, '
        f'{window.collections} collections',
        *_ranking_lines(
            'who grew (program + swap)', 'who grew: no consumer grew', who_grew(window), _growth, marks
        ),
        *_ranking_lines(
            'who paid (pages reclaimed, own memory stall)',
            'who paid: no consumer paid',
            who_paid(window),
            _paid,
            marks,
        ),
        *(_aside_line(row, window) for row in aside(window)),
        '  this project:',
        *(_project_line(row) for row in holding_project(mapping, window)),
        *(
            [f'  incomplete (unreadable in the window): {", ".join(map(encode_consumer, unreadable))}']
            if unreadable
            else []
        ),
    ]


def _hold_block(reading: HoldReading) -> list[str]:
    if reading.window is not None:
        return [_heading(reading.hold), *_window_lines(reading.mapping, reading.window)]
    gap, last = reading.gap, reading.last_collection
    between = (
        '' if gap is None
        else f'; no collection between {format_utc(gap.before)} and {format_utc(gap.after)}'
    )
    after = '' if last is None else f'; the last collection was {format_utc(last)}'
    return [_heading(reading.hold), f'  {_STATE_TEXT[reading.state]}{between}{after}']


def _state_counts(counts: Mapping[HoldState, int]) -> str:
    return ', '.join(f'{state.value} {count}' for state, count in counts.items())


def _tops_line(title: str, tops: Mapping[Consumer, int]) -> str:
    topped = ', '.join(f'{encode_consumer(c)} {n}' for c, n in Counter(tops).most_common())
    return f'  {title}: {topped or "none"}'


def _totals_line(consumer: Consumer, totals: Totals) -> str:
    left_out = totals.incomplete_holds
    return (
        f'    {encode_consumer(consumer)}  grew {_bytes(totals.grew)}, '
        f'{_pages(totals.reclaimed_pages)}, stall some {totals.memory_some_stall_seconds:.1f} s, '
        f'full {totals.memory_full_stall_seconds:.1f} s'
        + (f"; {left_out} hold{'' if left_out == 1 else 's'} left out, incomplete" if left_out else '')
    )


def _summary_lines(report: HoldsReport) -> list[str]:
    summary = summarise(report)
    ranked = sum(reading.window is not None for reading in report.readings)
    return [
        f'summary across {len(report.readings)} memory holds ({ranked} ranked):',
        '  holds per project:',
        *(
            f'    {root}: {sum(counts.values())} ({_state_counts(counts)})'
            for root, counts in summary.holds_per_project.items()
        ),
        *(_tops_line(f'times topping {name}', tops) for name, tops in summary.tops.items()),
        *(
            _tops_line(f'times topping {name} while incomplete, left out above', tops)
            for name, tops in summary.incomplete_tops.items()
        ),
        '  totals per consumer over the ranked holds, by growth:',
        *(_totals_line(consumer, totals) for consumer, totals in summary.totals.items()),
        *(
            f'  {label}: {_percent(growth)} of growth, {_percent(pages)} of pages'
            for label, (growth, pages) in summary.aside_shares.items()
        ),
        f'  reclaim threads: {_percent(summary.reclaim_thread_share)} of host CPU',
    ]


def render_text(report: HoldsReport, *, summary: bool = False) -> str:
    sections = [_header(report), *(_hold_block(reading) for reading in report.readings)]
    if summary:
        sections.append(_summary_lines(report))
    return '\n\n'.join('\n'.join(lines) for lines in sections)


def _time_json(moment: datetime | None) -> str | None:
    return None if moment is None else format_utc(moment)


def _consumer_json(consumer: Consumer | None) -> str | None:
    return None if consumer is None else encode_consumer(consumer)


def _root_json(read: RootRead) -> dict[str, Any]:
    return {
        'root': str(read.root),
        'runs_db': str(read.runs_db),
        'status': read.status.value,
        'detail': read.detail,
        'skipped': read.skipped,
    }


def _mapping_json(mapping: HoldConsumers) -> dict[str, Any]:
    return {
        'service': _consumer_json(mapping.service),
        'agents': _consumer_json(mapping.agents),
        'project': _consumer_json(mapping.project),
        'verify': encode_consumer(mapping.verify),
        'unmapped_reason': mapping.unmapped_reason,
    }


def _figures_json(c: ConsumerWindow) -> dict[str, Any]:
    return {
        'consumer': encode_consumer(c.consumer),
        'grew_bytes': c.grew,
        'program_growth_bytes': c.program_growth_bytes,
        'swap_growth_bytes': c.swap_growth_bytes,
        'reclaimed_pages': c.reclaimed_pages,
        'memory_some_stall_seconds': c.memory_some_stall_seconds,
        'memory_full_stall_seconds': c.memory_full_stall_seconds,
        'cpu_usec': c.cpu_usec,
        'complete': c.complete,
    }


def _aside_json(row: Aside) -> dict[str, Any]:
    return {
        'label': row.label,
        'positive_growth_bytes': row.positive_growth,
        'net_grew_bytes': row.net_grew,
        'reclaimed_pages': row.reclaimed_pages,
        'growth_share': row.growth_share,
        'reclaim_share': row.reclaim_share,
    }


def _project_json(row: ProjectConsumer) -> dict[str, Any]:
    return {
        'consumer': _consumer_json(row.consumer),
        'status': row.status.value,
        'reason': row.reason,
        'figures': None if row.figures is None else _figures_json(row.figures),
        'who_grew_rank': row.who_grew_rank,
        'who_paid_rank': row.who_paid_rank,
    }


def _window_json(mapping: HoldConsumers, window: Window) -> dict[str, Any]:
    return {
        'window': {
            'start': format_utc(window.start),
            'end': format_utc(window.end),
            'collections': window.collections,
        },
        **{name: [_figures_json(c) for c in rank(window)] for name, rank in RANKINGS.items()},
        'aside': [_aside_json(row) for row in aside(window)],
        'holding_project': {
            row.role.value: _project_json(row) for row in holding_project(mapping, window)
        },
        'incomplete': [encode_consumer(c) for c in incomplete(window)],
        'reclaim_threads_cpu_usec': window.reclaim_threads_cpu_usec,
        'host_cpu_usec': window.host_cpu_usec,
        'reclaim_threads_share': reclaim_threads_share(window),
    }


def _gap_json(gap: SamplerGap) -> dict[str, str]:
    return {'before': format_utc(gap.before), 'after': format_utc(gap.after)}


def _reading_json(reading: HoldReading) -> dict[str, Any]:
    hold = reading.hold
    entry: dict[str, Any] = {
        'at': format_utc(hold.at),
        'project_root': str(hold.project_root),
        'event_id': hold.event_id,
        'task_id': hold.task_id,
        'arm': hold.arm,
        'arm_value': hold.arm_value,
        'own_cgroup': hold.own_cgroup,
        'mapping': _mapping_json(reading.mapping),
        'state': reading.state.value,
        'last_collection': _time_json(reading.last_collection),
        'gap': None if reading.gap is None else _gap_json(reading.gap),
        'window': None,
    }
    if reading.window is not None:
        entry.update(_window_json(reading.mapping, reading.window))
    return entry


def _totals_json(totals: Totals) -> dict[str, Any]:
    return {
        'grew_bytes': totals.grew,
        'reclaimed_pages': totals.reclaimed_pages,
        'memory_some_stall_seconds': totals.memory_some_stall_seconds,
        'memory_full_stall_seconds': totals.memory_full_stall_seconds,
        'incomplete_holds': totals.incomplete_holds,
    }


def _tops_json(tops: Mapping[str, Mapping[Consumer, int]]) -> dict[str, dict[str, int]]:
    return {
        name: {encode_consumer(c): n for c, n in Counter(counts).most_common()}
        for name, counts in tops.items()
    }


def _summary_json(summary: Summary) -> dict[str, Any]:
    return {
        'holds_per_project': {
            str(root): {'total': sum(counts.values()), **{s.value: n for s, n in counts.items()}}
            for root, counts in summary.holds_per_project.items()
        },
        'tops': _tops_json(summary.tops),
        'incomplete_tops': _tops_json(summary.incomplete_tops),
        'totals': {encode_consumer(c): _totals_json(t) for c, t in summary.totals.items()},
        'aside_shares': {
            label: {'growth_share': growth, 'reclaim_share': pages}
            for label, (growth, pages) in summary.aside_shares.items()
        },
        'reclaim_thread_share': summary.reclaim_thread_share,
    }


def report_json(report: HoldsReport, *, summary: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        'roots': [_root_json(read) for read in report.roots],
        'samples_db': str(report.samples_db),
        'first_collection': _time_json(report.first_collection),
        'last_collection': _time_json(report.last_collection),
        'since': format_utc(report.since),
        'window_seconds': report.window_seconds,
        'holds': [_reading_json(reading) for reading in report.readings],
    }
    if summary:
        out['summary'] = _summary_json(summarise(report))
    return out
