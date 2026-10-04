"""Part 1 of the suite census (task 5414): never-failed tests ranked by per-run cost.

Pure: the readers in ``suite_census_evidence`` turn retained verify artefacts
into the typed records below; this module folds a stream of them into an
``OutcomeCensus`` and renders it.
"""
from __future__ import annotations

import enum
import math
import statistics
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

_UNRESOLVED_SAMPLES = 5


class Outcome(enum.Enum):
    PASSED = 'passed'
    FAILED = 'failed'
    SKIPPED = 'skipped'


@dataclass(frozen=True, order=True, slots=True)
class TestId:
    """One test FUNCTION: params and xdist group suffixes are already stripped."""

    package: str
    name: str


@dataclass(frozen=True, slots=True)
class Observation:
    source: str
    run: str
    test: TestId
    outcome: Outcome
    seconds: float | None


@dataclass(frozen=True, slots=True)
class Unresolved:
    """A record a reader could not attribute to a tracked test, kept so it is counted."""

    source: str
    raw_id: str


Record = Observation | Unresolved


@dataclass(frozen=True)
class SourceWindow:
    source: str
    pattern: str
    present: bool
    artefacts: int
    first: str | None
    last: str | None


@dataclass(frozen=True)
class Evidence:
    """Everything one ecosystem's artefacts say about test outcomes.

    ``records`` returns a fresh iterator each call, and it MUST yield every
    Observation of one (source, run) contiguously: ``census_outcomes`` folds a
    run when the next one starts and refuses a run that reappears. Unresolved
    records may appear anywhere. Every record's source has a window here.
    """

    windows: tuple[SourceWindow, ...]
    records: Callable[[], Iterator[Record]]
    uncosted_universe: frozenset[TestId]


@dataclass(frozen=True)
class TestCost:
    test: TestId
    runs: int
    median_s: float
    max_s: float
    total_s: float


@dataclass(frozen=True)
class SourceTally:
    source: str
    runs: int
    red_runs: int
    unresolved: int
    unresolved_samples: tuple[str, ...]


@dataclass(frozen=True)
class PackageSummary:
    package: str
    observed: int
    failed: int
    never_failed: int
    uncosted_never_failed: int
    p50_s: float | None
    top1_share: float | None


@dataclass(frozen=True)
class OutcomeCensus:
    windows: tuple[SourceWindow, ...]
    sources: tuple[SourceTally, ...]
    failed_floor: frozenset[TestId]
    failed_without_cost: tuple[TestId, ...]
    ranking: tuple[TestCost, ...]
    packages: tuple[PackageSummary, ...]
    never_failed_uncosted: tuple[TestId, ...]


class _Tally:
    def __init__(self) -> None:
        self.runs = 0
        self.red_runs = 0
        self.unresolved = 0
        self.samples: list[str] = []

    def frozen(self, source: str) -> SourceTally:
        return SourceTally(
            source=source, runs=self.runs, red_runs=self.red_runs,
            unresolved=self.unresolved, unresolved_samples=tuple(self.samples),
        )


class _Fold:
    """The accumulators of ONE census_outcomes call, folding a run as it ends."""

    def __init__(self, sources: Iterable[str]) -> None:
        self.tallies = {source: _Tally() for source in sources}
        self.costs: dict[TestId, array[float]] = {}
        self.failed: set[TestId] = set()
        self._finished: set[tuple[str, str]] = set()
        self._run: tuple[str, str] | None = None
        self._run_seconds: dict[TestId, float] = {}
        self._run_failed: set[TestId] = set()

    def add(self, record: Record) -> None:
        tally = self._tally(record.source)
        if isinstance(record, Unresolved):
            tally.unresolved += 1
            if len(tally.samples) < _UNRESOLVED_SAMPLES:
                tally.samples.append(record.raw_id)
            return
        key = (record.source, record.run)
        if key != self._run:
            self._start(key)
        if record.outcome is Outcome.FAILED:
            self._run_failed.add(record.test)
        if record.outcome is not Outcome.SKIPPED and record.seconds is not None:
            self._run_seconds[record.test] = (
                self._run_seconds.get(record.test, 0.0) + record.seconds
            )

    def close(self) -> None:
        if self._run is None:
            return
        tally = self.tallies[self._run[0]]
        tally.runs += 1
        tally.red_runs += bool(self._run_failed)
        for test, seconds in self._run_seconds.items():
            self.costs.setdefault(test, array('d')).append(seconds)
        self.failed |= self._run_failed
        self._finished.add(self._run)
        self._run = None
        self._run_seconds = {}
        self._run_failed = set()

    def _start(self, key: tuple[str, str]) -> None:
        self.close()
        if key in self._finished:
            raise ValueError(
                f'run {key[1]!r} of source {key[0]!r} reappeared after another run '
                'started; a reader must yield one run\'s records contiguously'
            )
        self._run = key

    def _tally(self, source: str) -> _Tally:
        try:
            return self.tallies[source]
        except KeyError:
            raise ValueError(
                f'a record came from source {source!r}, which has no SourceWindow; '
                'every source must state its window'
            ) from None


def _test_cost(test: TestId, run_seconds: array[float]) -> TestCost:
    return TestCost(
        test=test, runs=len(run_seconds), median_s=statistics.median(run_seconds),
        max_s=max(run_seconds), total_s=math.fsum(run_seconds),
    )


def _rank_key(cost: TestCost) -> tuple[float, float, str, str]:
    return (-cost.median_s, -cost.total_s, cost.test.name, cost.test.package)


def _top1_share(medians: list[float]) -> float | None:
    total = math.fsum(medians)
    if not total:
        return None
    top = sorted(medians, reverse=True)[: max(1, math.ceil(len(medians) / 100))]
    return math.fsum(top) / total


def _package_summaries(
    costs: dict[TestId, TestCost], floor: frozenset[TestId],
    never_failed_uncosted: tuple[TestId, ...],
) -> tuple[PackageSummary, ...]:
    medians: defaultdict[str, list[float]] = defaultdict(list)
    for test, cost in costs.items():
        medians[test.package].append(cost.median_s)
    never_failed = Counter(test.package for test in costs if test not in floor)
    failed = Counter(test.package for test in floor)
    uncosted = Counter(test.package for test in never_failed_uncosted)
    return tuple(
        PackageSummary(
            package=package, observed=len(medians[package]), failed=failed[package],
            never_failed=never_failed[package], uncosted_never_failed=uncosted[package],
            p50_s=statistics.median(medians[package]) if medians[package] else None,
            top1_share=_top1_share(medians[package]),
        )
        for package in sorted({*medians, *failed, *uncosted})
    )


def census_outcomes(evidence: Evidence) -> OutcomeCensus:
    fold = _Fold(window.source for window in evidence.windows)
    for record in evidence.records():
        fold.add(record)
    fold.close()
    floor = frozenset(fold.failed)
    costs = {test: _test_cost(test, runs) for test, runs in fold.costs.items()}
    never_failed_uncosted = tuple(sorted(evidence.uncosted_universe - floor))
    return OutcomeCensus(
        windows=evidence.windows,
        sources=tuple(tally.frozen(source) for source, tally in fold.tallies.items()),
        failed_floor=floor,
        failed_without_cost=tuple(sorted(floor.difference(costs))),
        ranking=tuple(
            sorted((cost for test, cost in costs.items() if test not in floor), key=_rank_key)
        ),
        packages=_package_summaries(costs, floor, never_failed_uncosted),
        never_failed_uncosted=never_failed_uncosted,
    )


# ---------------------------------------------------------------------------
# Rendering.

_CAVEATS = """\
### What this ranking is not

- No test is retired by this census.
- Retiring any test needs a planted-defect check first (INV-10
  `guards-exercise-behaviour`): show the test goes red on the defect it guards.
- A slow test that never failed is an offline-lane candidate, not a deletion.
- "Never failed" is weak evidence: it means "never seen failing in the windows
  above", and a guard whose invariant nobody broke also never fails."""


def _cell(value: object) -> str:
    return str(value).replace('|', '\\|')


def _row(*cells: object) -> str:
    return '| ' + ' | '.join(_cell(cell) for cell in cells) + ' |'


def _table(header: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    lines = [_row(*header), _row(*('---' for _ in header))]
    lines.extend(_row(*row) for row in rows)
    return '\n'.join(lines)


def _seconds(value: float | None) -> str:
    return '—' if value is None else f'{value:.3f}'


def _share(numerator: int, denominator: int) -> str:
    return f'{numerator / denominator:.1%}' if denominator else '—'


def _windows_table(census: OutcomeCensus) -> str:
    tallies = {tally.source: tally for tally in census.sources}
    rows = (
        (
            window.source, f'`{window.pattern}`', 'yes' if window.present else 'no',
            window.artefacts, tallies[window.source].runs, tallies[window.source].red_runs,
            _share(tallies[window.source].red_runs, tallies[window.source].runs),
            window.first or '—', window.last or '—', tallies[window.source].unresolved,
        )
        for window in census.windows
    )
    return '### Evidence windows\n\n' + _table(
        ('source', 'pattern / query', 'present', 'artefacts', 'runs', 'red runs',
         'red share', 'first', 'last', 'unresolved'),
        rows,
    )


def _floor_statement(census: OutcomeCensus) -> str:
    return (
        f'The failed floor holds {len(census.failed_floor)} tests: every test one of the '
        'sources above recorded failing inside its window. Failures outside those '
        'windows, or never retained, are unseen, so the true failed set is at least '
        f'this large. {len(census.failed_without_cost)} floor members have no cost '
        'observation (their failure is known only from a log or ledger line).'
    )


def _packages_table(census: OutcomeCensus) -> str:
    rows = (
        (
            summary.package, summary.observed, summary.failed, summary.never_failed,
            summary.uncosted_never_failed, _seconds(summary.p50_s),
            '—' if summary.top1_share is None else f'{summary.top1_share:.1%}',
        )
        for summary in census.packages
    )
    return '### Per package\n\n' + _table(
        ('package', 'costed tests', 'failed (floor)', 'never failed, costed',
         'never failed, uncosted', 'p50 of medians s', 'top 1% share of time'),
        rows,
    )


def _ranking_table(census: OutcomeCensus, top: int) -> str:
    shown = census.ranking[:top]
    rows = (
        (rank, f'`{cost.test.name}`', cost.test.package, cost.runs,
         _seconds(cost.median_s), _seconds(cost.max_s), _seconds(cost.total_s))
        for rank, cost in enumerate(shown, start=1)
    )
    return (
        f'### Never failed × per-run cost (top {len(shown)} of {len(census.ranking)})\n\n'
        + _table(('rank', 'test', 'package', 'runs', 'median s', 'max s', 'total s'), rows)
    )


def _details(summary: str, items: Iterable[str]) -> str:
    body = '\n'.join(f'- {item}' for item in items)
    return f'<details><summary>{summary}</summary>\n\n{body}\n\n</details>'


def _test_items(tests: Iterable[TestId]) -> Iterator[str]:
    return (f'`{test.name}` ({test.package})' for test in tests)


def _unresolved_items(census: OutcomeCensus) -> Iterator[str]:
    for tally in census.sources:
        samples = ', '.join(f'`{sample}`' for sample in tally.unresolved_samples)
        yield f'{tally.source}: {tally.unresolved} unresolved; samples: {samples or "—"}'


def render_markdown(census: OutcomeCensus, *, top: int) -> str:
    sections = (
        _windows_table(census),
        _floor_statement(census),
        _packages_table(census),
        _ranking_table(census, top),
        _details(
            f'Failed floor ({len(census.failed_floor)} tests)',
            _test_items(sorted(census.failed_floor)),
        ),
        _details(
            f'Never failed, uncosted ({len(census.never_failed_uncosted)} tests)',
            _test_items(census.never_failed_uncosted),
        ),
        _details('Unresolved records per source', _unresolved_items(census)),
        _CAVEATS,
    )
    return '\n\n'.join(sections) + '\n'
