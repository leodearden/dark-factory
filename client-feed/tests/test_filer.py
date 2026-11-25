"""The defect filer's pass (plans/client-feed-prd.md §14.4, §14.5): B25-B29, filing each portal report once as a fix and its gate."""

import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _defects import carrier_for, declared, quarantined, reported
from _fake_fused_memory import FakeFusedMemory, SimulatedCrash
from _in_dir import append_inbox, request_reveals
from _stores import TasksDb

from client_feed.contract import INBOX_FILENAME, QuarantineReason
from client_feed.defect_ref import portal_ref
from client_feed.filer import run_pass
from client_feed.filer_state import FilerCursor
from client_feed.filer_tasks import FIX_TITLE, GATE_TITLE, RESTATED_HEADING
from client_feed.filer_tools import FILER_IDENTITY, WRITE_TOOLS, ClosedTools
from client_feed.policy import Policy
from client_feed.sources.tasks import DefectCarrier, read_defect_carriers

PROJECT = 'acme-widgets'
REF = portal_ref(PROJECT, 2, 3)
AT = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)


class World:
    """One client project's dirs, its policy, and the fake fused-memory the filer talks to."""

    def __init__(self, tmp_path: Path, *, auto_dispatch: bool) -> None:
        self.tmp_path = tmp_path
        self.root = tmp_path / 'root'
        self.root.mkdir()
        self.in_dir = tmp_path / 'in'
        self.policy = Policy.model_validate({
            'project_id': PROJECT,
            'project_root': str(self.root),
            'out_dir': str(tmp_path / 'out'),
            'in_dir': str(self.in_dir),
            'visible_globs': ['src/**'],
            'filer': {'state_dir': str(tmp_path / 'state'), 'auto_dispatch': auto_dispatch},
        })
        self.fake = FakeFusedMemory(self.root)
        self.tools = ClosedTools(self.fake, self.root)

    def run(self, cursor: FilerCursor | None = None) -> tuple[Any, FilerCursor]:
        return run_pass(self.policy, self.tools, cursor or FilerCursor())

    def fixes(self) -> list[dict[str, Any]]:
        return [task for task in self.fake.tasks() if task['metadata'].get('defects') and task['metadata'].get('source') == FILER_IDENTITY]

    def gates(self) -> list[dict[str, Any]]:
        return [task for task in self.fake.tasks() if 'x_client_defect_gate_for' in task['metadata']]

    def fix_and_gate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        [fix], [gate] = self.fixes(), self.gates()
        return fix, gate

    def writes(self) -> list[str]:
        return [name for name, _ in self.fake.calls if name in WRITE_TOOLS]

    def carriers(self) -> tuple[DefectCarrier, ...]:
        """What the publisher's carrier reader sees of the store the filer wrote."""
        root = Path(tempfile.mkdtemp(dir=self.tmp_path))
        self.fake.write_tasks_db(TasksDb.create(root), updated_at='2026-11-20T09:00:00+00:00')
        return read_defect_carriers(root, project_id=PROJECT)

    def inbox_size(self) -> int:
        return (self.in_dir / INBOX_FILENAME).stat().st_size


@pytest.fixture
def gated(tmp_path: Path) -> World:
    return World(tmp_path, auto_dispatch=False)


@pytest.fixture
def auto(tmp_path: Path) -> World:
    return World(tmp_path, auto_dispatch=True)


def _assert_filed_and_gated(world: World) -> None:
    fix, gate = world.fix_and_gate()
    assert fix['dependencies'] == [gate['id']]
    assert (fix['status'], gate['status']) == ('pending', 'pending')
    assert [task for task in world.fake.tasks() if task['status'] == 'deferred'] == []
    assert world.carriers() == (carrier_for(REF, project_id=PROJECT, task_id=fix['id'], gated=True),)


def test_b25_a_restated_report_is_filed_as_a_fix_waiting_on_its_gate(gated: World) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT))

    report, cursor = gated.run()

    fix, gate = gated.fix_and_gate()
    [entry] = fix['metadata']['defects']
    assert (entry['ref'], entry['origin'], entry['status'], entry['sightings']) == (REF, 'client', 'open', 1)
    assert fix['title'] == FIX_TITLE.format(ref=REF, criterion_id='C-1')
    assert RESTATED_HEADING in fix['description']
    assert gate['task_kind'] == 'deterministic'
    assert gate['metadata']['always_escalates'] is True
    assert gate['title'] == GATE_TITLE.format(ref=REF)
    _assert_filed_and_gated(gated)
    names = [name for name, _ in gated.fake.calls]
    assert names.index('lookup_defect_ref') < names.index('submit_task')
    assert all(arguments['planning_mode'] is True for name, arguments in gated.fake.calls if name == 'submit_task')
    assert report.converged == (REF,)
    assert cursor.inbox_offset == gated.inbox_size()


def test_b26_a_quarantined_report_is_gated_even_under_auto_dispatch(auto: World) -> None:
    reasons = (QuarantineReason.SUSPECTED_INJECTION,)
    append_inbox(auto.in_dir, reported(2, 3, at=AT, outcome=quarantined(*reasons)))

    auto.run()

    fix, _ = auto.fix_and_gate()
    assert RESTATED_HEADING not in fix['description']
    assert 'Observed:' not in fix['description']
    assert all(reason.value in fix['description'] for reason in reasons)
    _assert_filed_and_gated(auto)


def test_b27_a_restated_report_under_auto_dispatch_is_released_ungated(auto: World) -> None:
    append_inbox(auto.in_dir, reported(2, 3, at=AT))

    auto.run()

    [fix] = auto.fixes()
    assert (fix['status'], fix['dependencies']) == ('pending', [])
    assert auto.gates() == []
    assert auto.carriers() == (carrier_for(REF, project_id=PROJECT, task_id=fix['id'], gated=False),)


def test_b28_a_filed_report_is_filed_once_and_costs_only_reads(gated: World) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT))
    request_reveals(gated.in_dir, 'src/a.py', at=AT)
    with (gated.in_dir / INBOX_FILENAME).open('a') as inbox:
        inbox.write('{"type": "not_a_line"}\n')

    first, cursor = gated.run()
    writes_after_first = len(gated.writes())
    gated.run(cursor)
    gated.run(FilerCursor())

    assert len(gated.fixes()) == len(gated.gates()) == 1
    assert len(gated.writes()) == writes_after_first
    assert first.rejected_lines == 1
    assert first.converged == (REF,)


@pytest.mark.parametrize(
    'crash',
    [
        lambda fake: fake.crash_after('submit_task', 1),
        lambda fake: fake.crash_after('submit_task', 2),
        lambda fake: fake.crash_after('add_dependency', 1),
        lambda fake: fake.crash_before('commit_planning', 1),
    ],
    ids=['after-fix-submit', 'after-gate-submit', 'after-edge', 'before-commit'],
)
def test_b29_every_crash_point_converges_to_one_gated_fix(
    gated: World, crash: Callable[[FakeFusedMemory], None]
) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT))
    crash(gated.fake)

    with pytest.raises(SimulatedCrash):
        gated.run()
    gated.run()

    _assert_filed_and_gated(gated)


def test_a_hand_filed_carrier_counts_as_filed(gated: World) -> None:
    gated.fake.seed_task(metadata={'defects': [declared(REF)]})
    append_inbox(gated.in_dir, reported(2, 3, at=AT))

    report, _ = gated.run()

    assert gated.writes() == []
    assert report.converged == (REF,)


def test_the_cursor_advances_only_when_every_ref_converged(gated: World) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT), reported(2, 4, at=AT, seq=2))
    held = FilerCursor()
    gated.fake.refuse_next('submit_task', {'error': 'busy', 'error_type': 'ReconciliationBacklogExceeded'})

    stuck, stuck_cursor = gated.run(held)
    clean, clean_cursor = gated.run(stuck_cursor)

    assert [item.ref for item in stuck.unconverged] == [REF]
    assert stuck.converged == (portal_ref(PROJECT, 2, 4),)
    assert stuck_cursor.inbox_offset == held.inbox_offset
    assert clean.converged == (REF, portal_ref(PROJECT, 2, 4))
    assert clean_cursor.inbox_offset == gated.inbox_size()
