"""Read the durable verify archive's junit reports into typed rows, each run tied to the tree it verified.

The archive is ``data/verify-logs/<task>/`` in the main checkout.
:class:`ArchiveName` is the one public grammar for the artefact names read
here; :func:`read_runs` streams every archived junit into a :class:`JunitRun`
holding the rows at or above a duration floor.

Two facts are decided once, here at the boundary, so no caller re-parses them:

* a row whose ``<error>``/``<failure>`` message reports an xdist worker crash
  is :attr:`CaseOutcome.CENSORED`, and one whose message IS pytest-timeout's
  failure (``pytest_timeout.PYTEST_FAILURE_MESSAGE``, as junitxml words a
  call failure or a setup or teardown error) is :attr:`CaseOutcome.TIMED_OUT`;
  a message that merely quotes that text, say a child pytest's output, is not.
  Neither a censored nor a timed-out row's junit ``time`` is the duration of a
  test that ran to its own end, so such a row is kept whatever the floor and
  must never be read as one;
* each run's tree is the ``tip_sha`` of the attempt's first dispatch sidecar
  stamped at or after the junit, else that dispatch's ``base_sha``, else the
  caller's HEAD, recorded as a typed :class:`TreeSource`.

An unreadable junit is a counted :class:`Skip`, never an exception, so
``runs`` plus ``skipped`` account for every junit the archive holds.

Stdlib only.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

__all__ = [
    'ArchiveName',
    'ArtefactKind',
    'CaseOutcome',
    'JunitCase',
    'JunitCorpus',
    'JunitRun',
    'Skip',
    'TreeRef',
    'TreeSource',
    'read_runs',
]

_UNREADABLE_JUNIT = (OSError, EOFError, zlib.error, ET.ParseError)
_WORKER_CRASH = re.compile(r"worker '[^']+' crashed while running")
_PYTEST_TIMEOUT = re.compile(
    r'(?:failed on (?:setup|teardown) with ")?Failed: Timeout \(>[0-9.]+s\) from pytest-timeout\.'
)


class ArtefactKind(StrEnum):
    """The archived verify artefacts this module names."""

    JUNIT = 'junit'
    SUMMARY = 'summary'
    DISPATCH = 'dispatch'


_SUFFIX = {
    ArtefactKind.JUNIT: '.xml.gz',
    ArtefactKind.SUMMARY: '.json',
    ArtefactKind.DISPATCH: '.json',
}
_NAME = re.compile(
    r'^attempt-(?P<attempt>\d+)'
    r'(?:\.(?P<infix>[^.]+))?(?:\.(?P<segment>[^.]+))?'
    r'\.(?P<kind>junit|summary|dispatch)-'
    r'(?P<stamp>\d{8}T\d{6})(?:_(?P<micros>\d{1,6}))?Z'
    r'(?P<suffix>\.xml\.gz|\.json)$'
)


@dataclass(frozen=True)
class ArchiveName:
    """One archived artefact's name: ``attempt-N[.<infix>][.<segment>].<kind>-<stamp><suffix>``.

    ``infix`` is the module prefix as ``verify._make_infix`` sanitises it;
    ``segment`` names one segment of a segmented leg and only follows an
    infix.  ``stamp`` is UTC, in either the microsecond or the older
    whole-second spelling.  The producers are
    ``orchestrator/src/orchestrator/verify.py::_archive_junit_report``,
    ``orchestrator/src/orchestrator/verify.py::_archive_attempt_summary`` and
    ``orchestrator/src/orchestrator/task_verify_dispatch.py::_write_sidecar``;
    the grammar reads what they write today and is not a promise about
    names they wrote before.
    """

    kind: ArtefactKind
    attempt: int
    infix: str | None
    segment: str | None
    stamp: datetime

    @classmethod
    def parse(cls, filename: str) -> ArchiveName | None:
        """The parsed *filename*, or None when it is not an artefact this grammar names."""
        match = _NAME.match(filename)
        if match is None:
            return None
        kind = ArtefactKind(match['kind'])
        if match['suffix'] != _SUFFIX[kind]:
            return None
        if kind is not ArtefactKind.JUNIT and match['segment'] is not None:
            return None
        if kind is ArtefactKind.DISPATCH and match['infix'] is not None:
            return None
        stamp = datetime.strptime(match['stamp'], '%Y%m%dT%H%M%S').replace(
            microsecond=int((match['micros'] or '0').ljust(6, '0')), tzinfo=UTC
        )
        return cls(kind, int(match['attempt']), match['infix'], match['segment'], stamp)


class CaseOutcome(StrEnum):
    """What a junit row says happened; TIMED_OUT and CENSORED rows carry no duration of their own."""

    PASSED = 'passed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    TIMED_OUT = 'timed_out'
    CENSORED = 'censored'

    @property
    def has_duration(self) -> bool:
        """Whether the row's junit ``time`` is the duration of a test that ran to its own end."""
        return self not in (CaseOutcome.TIMED_OUT, CaseOutcome.CENSORED)


class TreeSource(StrEnum):
    """Where a run's tree sha came from."""

    TIP = 'tip'
    BASE = 'base'
    HEAD_FALLBACK = 'head_fallback'


@dataclass(frozen=True)
class TreeRef:
    """The commit a run's code is read at, and how it was chosen."""

    sha: str
    source: TreeSource


@dataclass(frozen=True)
class JunitCase:
    """One kept junit row: ``seconds`` is the per-test total, setup + call + teardown."""

    classname: str
    name: str
    seconds: float
    outcome: CaseOutcome


@dataclass(frozen=True)
class JunitRun:
    """One archived junit report and the rows of it at or above the floor."""

    task_id: str
    attempt: int
    infix: str | None
    segment: str | None
    stamp: datetime
    path: Path
    tree: TreeRef
    cases: tuple[JunitCase, ...]
    rows_below_floor: int


@dataclass(frozen=True)
class Skip:
    """An archived junit that could not be read, and why."""

    path: Path
    reason: str


@dataclass(frozen=True)
class JunitCorpus:
    """Every archived junit, read or skipped."""

    runs: tuple[JunitRun, ...]
    skipped: tuple[Skip, ...]


@dataclass(frozen=True)
class _Artefact:
    path: Path
    name: ArchiveName


def read_runs(
    archive_root: Path,
    *,
    min_seconds: float,
    commit_exists: Callable[[str], bool],
    head_sha: str,
) -> JunitCorpus:
    """Stream every junit under *archive_root*'s task directories into runs, keeping rows of at least *min_seconds*.

    *commit_exists* answers whether a sha is a commit the caller can read
    code at; *head_sha* is the tree used when no dispatched commit is.
    """
    runs: list[JunitRun] = []
    skipped: list[Skip] = []
    for task_dir in sorted(path for path in archive_root.iterdir() if path.is_dir()):
        artefacts = list(_artefacts(task_dir))
        dispatches = sorted(
            (a for a in artefacts if a.name.kind is ArtefactKind.DISPATCH), key=lambda a: a.name.stamp
        )
        for junit in sorted(
            (a for a in artefacts if a.name.kind is ArtefactKind.JUNIT), key=lambda a: a.path.name
        ):
            tree = _tree_for(junit.name, dispatches, commit_exists, head_sha)
            try:
                cases, below = _read_cases(junit.path, min_seconds)
            except _UNREADABLE_JUNIT as exc:
                skipped.append(Skip(junit.path, f'{type(exc).__name__}: {exc}'))
                continue
            runs.append(
                JunitRun(
                    task_id=task_dir.name,
                    attempt=junit.name.attempt,
                    infix=junit.name.infix,
                    segment=junit.name.segment,
                    stamp=junit.name.stamp,
                    path=junit.path,
                    tree=tree,
                    cases=cases,
                    rows_below_floor=below,
                )
            )
    return JunitCorpus(tuple(runs), tuple(skipped))


def _artefacts(task_dir: Path) -> Iterator[_Artefact]:
    for path in task_dir.iterdir():
        name = ArchiveName.parse(path.name)
        if name is not None and path.is_file():
            yield _Artefact(path, name)


def _tree_for(
    junit: ArchiveName,
    dispatches: Iterable[_Artefact],
    commit_exists: Callable[[str], bool],
    head_sha: str,
) -> TreeRef:
    dispatch = next(
        (d for d in dispatches if d.name.attempt == junit.attempt and d.name.stamp >= junit.stamp),
        None,
    )
    shas = {} if dispatch is None else _dispatched_shas(dispatch.path)
    for source, key in ((TreeSource.TIP, 'tip_sha'), (TreeSource.BASE, 'base_sha')):
        sha = shas.get(key)
        if sha and commit_exists(sha):
            return TreeRef(sha, source)
    return TreeRef(head_sha, TreeSource.HEAD_FALLBACK)


def _dispatched_shas(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if isinstance(value, str)}


def _read_cases(path: Path, min_seconds: float) -> tuple[tuple[JunitCase, ...], int]:
    kept: list[JunitCase] = []
    below = 0
    with gzip.open(path, 'rb') as report:
        for _event, element in ET.iterparse(report, events=('end',)):
            if element.tag != 'testcase':
                continue
            case = _case(element)
            element.clear()
            if not case.outcome.has_duration or case.seconds >= min_seconds:
                kept.append(case)
            else:
                below += 1
    return tuple(kept), below


def _case(element: ET.Element) -> JunitCase:
    return JunitCase(
        classname=element.get('classname', ''),
        name=element.get('name', ''),
        seconds=_seconds(element.get('time')),
        outcome=_outcome(element),
    )


def _seconds(raw: str | None) -> float:
    try:
        return float(raw) if raw is not None else 0.0
    except ValueError:
        return 0.0


def _outcome(element: ET.Element) -> CaseOutcome:
    messages = [child.get('message', '') for child in element if child.tag in ('error', 'failure')]
    if any(_WORKER_CRASH.search(message) for message in messages):
        return CaseOutcome.CENSORED
    if any(_PYTEST_TIMEOUT.match(message) for message in messages):
        return CaseOutcome.TIMED_OUT
    if messages:
        return CaseOutcome.FAILED
    if any(child.tag == 'skipped' for child in element):
        return CaseOutcome.SKIPPED
    return CaseOutcome.PASSED
