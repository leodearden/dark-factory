"""The client defect bridge end to end (plans/client-feed-prd.md §14.6 B36): the portal's vendored inbox, the real filer, the real tool layer.

The real ``client_feed`` filer runs over its production HTTP session against the fused-memory
tool layer served by ``_served_task_stack``, dispatch guards included. Every assertion reads
the rows that store wrote through its own ``get_tasks`` tool, and the publisher then reads
the same store by its own production path. The vendored inboxes have one home,
``client-feed/tests/fixtures``, read here by path.
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from _served_task_stack import ServedTaskStack, served_task_stack
from client_feed.contract import INBOX_FILENAME, DefectFiled, FeedLine
from client_feed.defect_ref import portal_ref
from client_feed.feed_io import OutDir, read_feed_lines
from client_feed.filer import PassReport, file_once
from client_feed.filer_tasks import FIX_TITLE, GATE_TITLE, RESTATED_HEADING
from client_feed.filer_tools import FILER_IDENTITY, OpenTools, ToolCaller, open_session
from client_feed.policy import Policy
from client_feed.publisher import run_pass

PROJECT = 'demo-client'
CLIENT_FEED_FIXTURES = Path(__file__).resolve().parents[2] / 'client-feed' / 'tests' / 'fixtures'
IN_V2 = CLIENT_FEED_FIXTURES / 'in-v2' / INBOX_FILENAME
NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
RESTATED = portal_ref(PROJECT, 2, 1)
QUARANTINED = portal_ref(PROJECT, 2, 2)
CRITERIA = {RESTATED: 'C-12', QUARANTINED: 'C-14'}
QUARANTINE_REASONS = ('suspected_injection', 'not_restatable')

Calls = list[tuple[str, dict[str, Any]]]


class SimulatedCrash(BaseException):
    """The filer process dying mid-pass: no handler of the filer's catches it."""


class RecordingCaller:
    """A ToolCaller that forwards every call and keeps its name and arguments."""

    def __init__(self, caller: ToolCaller, calls: Calls) -> None:
        self._caller = caller
        self.calls = calls

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        return self._caller.call_tool(name, arguments)


class CrashingCaller:
    """A ToolCaller that raises SimulatedCrash before or after forwarding the nth call of `tool`."""

    def __init__(
        self, caller: ToolCaller, *, tool: str, nth: int, when: Literal['before', 'after']
    ) -> None:
        self._caller = caller
        self._tool = tool
        self._nth = nth
        self._when = when
        self._seen = 0

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        crashes = name == self._tool and self._count() == self._nth
        if crashes and self._when == 'before':
            raise SimulatedCrash(f'before {name} #{self._nth}')
        reply = self._caller.call_tool(name, arguments)
        if crashes:
            raise SimulatedCrash(f'after {name} #{self._nth}')
        return reply

    def _count(self) -> int:
        self._seen += 1
        return self._seen


def over_session(wrap: Callable[[ToolCaller], ToolCaller]) -> OpenTools:
    """The filer's open_tools: the production session, wrapped."""

    @contextlib.contextmanager
    def open_tools(policy: Policy) -> Iterator[ToolCaller]:
        with open_session(policy) as session:
            yield wrap(session)

    return open_tools


class Bridge:
    """One client project over the served store: its dirs, its policy, the filer's passes and the publisher's."""

    def __init__(self, tmp_path: Path, served: ServedTaskStack, *, auto_dispatch: bool) -> None:
        self.root = served.root
        self.in_dir = tmp_path / 'in'
        self.in_dir.mkdir()
        self.out_dir = tmp_path / 'out'
        self.out_dir.mkdir()
        self.policy = Policy.model_validate({
            'project_id': PROJECT,
            'project_root': str(self.root),
            'out_dir': str(self.out_dir),
            'in_dir': str(self.in_dir),
            'visible_globs': ['src/**'],
            'filer': {
                'fused_memory_url': served.url,
                'state_dir': str(tmp_path / 'state'),
                'auto_dispatch': auto_dispatch,
            },
        })
        self._published = 0

    @property
    def inbox(self) -> Path:
        return self.in_dir / INBOX_FILENAME

    def file(self, open_tools: OpenTools = open_session) -> PassReport:
        """One filer pass on a fresh session from the stored cursor: a restarted filer."""
        return file_once(self.policy, open_tools=open_tools)

    def tasks(self) -> list[dict[str, Any]]:
        with open_session(self.policy) as reader:
            reply = reader.call_tool('get_tasks', {'project_root': str(self.root)})
        return reply['tasks']

    def fixes(self) -> list[dict[str, Any]]:
        return [
            task
            for task in self.tasks()
            if task['metadata'].get('source') == FILER_IDENTITY and task['metadata'].get('defects')
        ]

    def gates(self) -> list[dict[str, Any]]:
        return [task for task in self.tasks() if 'x_client_defect_gate_for' in task['metadata']]

    def fix_of(self, ref: str) -> dict[str, Any]:
        [fix] = [task for task in self.fixes() if task['metadata']['defects'][0]['ref'] == ref]
        return fix

    def gate_of(self, ref: str) -> dict[str, Any]:
        [gate] = [task for task in self.gates() if task['metadata']['x_client_defect_gate_for'] == ref]
        return gate

    def publish(self) -> tuple[FeedLine, ...]:
        """One publisher pass over the same store; the whole feed after it."""
        self._published += 1
        out = OutDir.open(self.out_dir, project_id=PROJECT)
        run_pass(self.policy, out, now=NOW.replace(hour=NOW.hour + self._published))
        return read_feed_lines(self.out_dir)


@pytest.fixture
def served(tmp_path: Path) -> Iterator[ServedTaskStack]:
    with served_task_stack(tmp_path) as stack:
        yield stack


@pytest.fixture
def gated(tmp_path: Path, served: ServedTaskStack) -> Bridge:
    return Bridge(tmp_path, served, auto_dispatch=False)


def defects_filed(feed: tuple[FeedLine, ...]) -> set[tuple[str, int, int, int, bool]]:
    return {
        (line.ref, line.block, line.report, line.task_id, line.gated)
        for line in feed
        if isinstance(line, DefectFiled)
    }


def assert_one_gated_fix_per_ref(bridge: Bridge, refs: tuple[str, ...]) -> None:
    assert len(bridge.fixes()) == len(bridge.gates()) == len(refs)
    for ref in refs:
        fix, gate = bridge.fix_of(ref), bridge.gate_of(ref)
        assert [str(task_id) for task_id in fix['dependencies']] == [gate['id']]
        assert (fix['status'], gate['status']) == ('pending', 'pending')
    assert [task['id'] for task in bridge.tasks() if task['status'] == 'deferred'] == []


def test_b36_b25_the_vendored_inbox_is_filed_as_gated_fixes_on_the_real_store(gated: Bridge) -> None:
    shutil.copyfile(IN_V2, gated.inbox)
    calls: Calls = []

    report = gated.file(over_session(lambda session: RecordingCaller(session, calls)))

    assert set(report.converged) == {RESTATED, QUARANTINED}
    assert report.offset_to == gated.inbox.stat().st_size
    for ref, criterion_id in CRITERIA.items():
        fix, gate = gated.fix_of(ref), gated.gate_of(ref)
        [entry] = fix['metadata']['defects']
        assert (entry['ref'], entry['origin'], entry['status'], entry['sightings']) == (ref, 'client', 'open', 1)
        assert fix['title'] == FIX_TITLE.format(ref=ref, criterion_id=criterion_id)
        assert gate['metadata']['task_kind'] == 'deterministic'
        assert gate['metadata']['always_escalates'] is True
        assert gate['title'] == GATE_TITLE.format(ref=ref)
    assert RESTATED_HEADING in gated.fix_of(RESTATED)['description']
    quarantined = gated.fix_of(QUARANTINED)['description']
    assert RESTATED_HEADING not in quarantined
    assert all(reason in quarantined for reason in QUARANTINE_REASONS)
    assert_one_gated_fix_per_ref(gated, (RESTATED, QUARANTINED))
    names = [name for name, _ in calls]
    assert names.index('lookup_defect_ref') < names.index('submit_task')
    assert all(arguments['planning_mode'] is True for name, arguments in calls if name == 'submit_task')

    feed = gated.publish()

    assert defects_filed(feed) == {
        (ref, 2, report_no, int(gated.fix_of(ref)['id']), True)
        for ref, report_no in ((RESTATED, 1), (QUARANTINED, 2))
    }


@pytest.mark.parametrize(
    ('tool', 'nth', 'when'),
    [
        pytest.param('submit_task', 1, 'after', id='after-fix-submit'),
        pytest.param('submit_task', 2, 'after', id='after-gate-submit'),
        pytest.param('add_dependency', 1, 'after', id='after-edge'),
        pytest.param('commit_planning', 1, 'before', id='before-commit'),
    ],
)
def test_b36_b29_every_crash_point_converges_to_one_gated_fix_per_report(
    gated: Bridge, tool: str, nth: int, when: Literal['before', 'after']
) -> None:
    shutil.copyfile(IN_V2, gated.inbox)
    crashing = over_session(lambda session: CrashingCaller(session, tool=tool, nth=nth, when=when))

    with pytest.raises(SimulatedCrash):
        gated.file(crashing)
    report = gated.file()

    assert set(report.converged) == {RESTATED, QUARANTINED}
    assert_one_gated_fix_per_ref(gated, (RESTATED, QUARANTINED))
