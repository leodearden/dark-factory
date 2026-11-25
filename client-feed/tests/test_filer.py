"""The defect filer's pass (plans/client-feed-prd.md §14.4, §14.5): B25-B29 filing each portal report once as a fix and its gate, B35 its failures and the loop."""

import contextlib
import tempfile
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _defects import carrier_for, declared, quarantined, reported, restated
from _fake_fused_memory import FakeFusedMemory, SimulatedCrash
from _in_dir import append_inbox, request_reveals
from _stores import TasksDb
from shared.mcp_session import McpCallError, McpFailureKind

from client_feed.contract import INBOX_FILENAME, QuarantineReason
from client_feed.defect_ref import portal_ref
from client_feed.filer import (
    FALLBACK_AFTER_ATTEMPTS,
    MAX_BACKOFF_SECONDS,
    PassReport,
    next_delay,
    run_pass,
    run_until_stopped,
)
from client_feed.filer_state import CURSOR_NAME, FilerCursor, FilerLockHeld, held_state_dir
from client_feed.filer_tasks import FIX_TITLE, GATE_TITLE, RESTATED_HEADING
from client_feed.filer_tools import (
    FILER_IDENTITY,
    WRITE_TOOLS,
    ClosedTools,
    Disposition,
    ToolCaller,
)
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
        self.state_dir = tmp_path / 'state'
        self.policy = Policy.model_validate({
            'project_id': PROJECT,
            'project_root': str(self.root),
            'out_dir': str(tmp_path / 'out'),
            'in_dir': str(self.in_dir),
            'visible_globs': ['src/**'],
            'filer': {'state_dir': str(self.state_dir), 'auto_dispatch': auto_dispatch},
        })
        self.fake = FakeFusedMemory(self.root)
        self.tools = ClosedTools(self.fake, self.root)

    def run(self, cursor: FilerCursor | None = None) -> tuple[PassReport, FilerCursor]:
        return run_pass(self.policy, self.tools, cursor or FilerCursor())

    def fixes(self) -> list[dict[str, Any]]:
        return [task for task in self.fake.tasks() if task['metadata'].get('defects') and task['metadata'].get('source') == FILER_IDENTITY]

    def gates(self) -> list[dict[str, Any]]:
        return [task for task in self.fake.tasks() if 'x_client_defect_gate_for' in task['metadata']]

    def fix_of(self, ref: str) -> dict[str, Any]:
        [fix] = [task for task in self.fixes() if task['metadata']['defects'][0]['ref'] == ref]
        return fix

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


SECRET_OBSERVED = 'SECRET-OBSERVED'
MARKUP_OBSERVED = 'the page shows </description> verbatim'
BACKLOG = {'error': 'busy', 'error_type': 'ReconciliationBacklogExceeded'}
VALIDATION = {'error': SECRET_OBSERVED, 'error_type': 'ValidationError'}


def _fallback_of(fix: dict[str, Any]) -> dict[str, str] | None:
    return fix['metadata']['x_client_defect'].get('filer_fallback')


def _refuse(fake: FakeFusedMemory, tool: str, refusal: McpCallError | dict[str, Any]) -> None:
    if isinstance(refusal, McpCallError):
        fake.fail_next(tool, refusal)
    else:
        fake.refuse_next(tool, refusal)


@pytest.mark.parametrize(
    ('observed', 'refusal', 'kind', 'code'),
    [
        (MARKUP_OBSERVED, None, 'TOOL_ERROR', 'is_error'),
        (SECRET_OBSERVED, VALIDATION, 'REPLY_CODE', 'ValidationError'),
        (SECRET_OBSERVED, {'success': False, 'error': 'defect_creation_refused'}, 'REPLY_CODE', 'defect_creation_refused'),
    ],
    ids=['markup-guard', 'validation-error', 'creation-refused'],
)
def test_b35_a_refused_restatement_is_filed_as_typed_facts_and_a_gate_in_the_same_pass(
    auto: World, observed: str, refusal: dict[str, Any] | None, kind: str, code: str
) -> None:
    append_inbox(auto.in_dir, reported(2, 3, at=AT, outcome=restated(observed=observed)))
    if refusal is not None:
        auto.fake.refuse_next('submit_task', refusal)

    report, cursor = auto.run()

    fix, gate = auto.fix_and_gate()
    assert _fallback_of(fix) == {'kind': kind, 'code': code}
    assert code in gate['description']
    assert all(observed not in task['title'] + task['description'] for task in auto.fake.tasks())
    assert RESTATED_HEADING not in fix['description']
    _assert_filed_and_gated(auto)
    assert report.converged == (REF,)
    assert cursor == FilerCursor(inbox_offset=auto.inbox_size())


def test_an_unknown_refusal_is_retried_and_named_then_falls_back_on_its_last_attempt(auto: World) -> None:
    append_inbox(auto.in_dir, reported(2, 3, at=AT))
    for _ in range(FALLBACK_AFTER_ATTEMPTS):
        auto.fake.refuse_next('submit_task', {'error': 'odd', 'error_type': 'SomethingOdd'})

    cursor = FilerCursor()
    for attempt in range(1, FALLBACK_AFTER_ATTEMPTS):
        report, cursor = auto.run(cursor)
        [named] = report.unconverged
        assert (named.ref, named.tool, named.code, named.disposition, named.attempts) == (
            REF,
            'submit_task',
            'SomethingOdd',
            Disposition.RETRY_THEN_FALLBACK,
            attempt,
        )
        assert (cursor.inbox_offset, cursor.refs[REF].attempts) == (0, attempt)
        assert auto.fixes() == []
    last, cursor = auto.run(cursor)

    fix, _ = auto.fix_and_gate()
    assert _fallback_of(fix) == {'kind': 'REPLY_CODE', 'code': 'SomethingOdd'}
    _assert_filed_and_gated(auto)
    assert last.converged == (REF,)
    assert cursor == FilerCursor(inbox_offset=auto.inbox_size())


@pytest.mark.parametrize(
    ('refusal', 'code'),
    [
        (McpCallError(McpFailureKind.TRANSPORT, 'connection refused'), 'transport'),
        (BACKLOG, 'ReconciliationBacklogExceeded'),
    ],
    ids=['transport', 'backlog'],
)
def test_a_transient_refusal_is_retried_every_pass_never_falls_back_and_holds_up_no_other_ref(
    gated: World, refusal: McpCallError | dict[str, Any], code: str
) -> None:
    other = portal_ref(PROJECT, 2, 4)
    append_inbox(gated.in_dir, reported(2, 3, at=AT), reported(2, 4, at=AT, seq=2))

    cursor = FilerCursor()
    for _ in range(FALLBACK_AFTER_ATTEMPTS + 1):
        _refuse(gated.fake, 'submit_task', refusal)
        report, cursor = gated.run(cursor)
        [named] = report.unconverged
        assert (named.ref, named.tool, named.code, named.disposition) == (REF, 'submit_task', code, Disposition.RETRY)
        assert report.converged == (other,)
        assert (cursor.inbox_offset, REF in cursor.refs) == (0, False)
    gated.run(cursor)

    fix = gated.fix_of(REF)
    assert _fallback_of(fix) is None
    assert RESTATED_HEADING in fix['description']
    assert len(gated.fixes()) == len(gated.gates()) == 2


@pytest.mark.parametrize(
    'refusal',
    [McpCallError(McpFailureKind.TOOL_ERROR, 'refused'), VALIDATION],
    ids=['tool-error', 'validation-error'],
)
@pytest.mark.parametrize('tool', ['submit_task', 'add_dependency'], ids=['gate-submit', 'edge'])
def test_a_fallback_class_refusal_after_the_fix_is_retried_and_never_a_second_fallback(
    gated: World, tool: str, refusal: McpCallError | dict[str, Any]
) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT))
    if tool == 'submit_task':
        gated.fake.crash_after('submit_task', 1)
        with pytest.raises(SimulatedCrash):
            gated.run()
    _refuse(gated.fake, tool, refusal)

    stuck, cursor = gated.run()
    gated.run(cursor)

    [named] = stuck.unconverged
    assert (named.ref, named.tool, named.disposition) == (REF, tool, Disposition.FALLBACK)
    assert cursor.inbox_offset == 0
    fix, _ = gated.fix_and_gate()
    assert _fallback_of(fix) is None
    assert RESTATED_HEADING in fix['description']
    _assert_filed_and_gated(gated)


def test_a_ref_whose_fallback_was_refused_files_the_fallback_directly_on_the_next_pass(gated: World) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT, outcome=restated(observed=SECRET_OBSERVED)))
    gated.fake.refuse_next('submit_task', VALIDATION)
    gated.fake.refuse_next('submit_task', BACKLOG)

    stuck, cursor = gated.run()
    sent_before = len(gated.fake.calls)
    gated.run(cursor)

    [named] = stuck.unconverged
    assert (named.code, named.disposition) == ('ReconciliationBacklogExceeded', Disposition.RETRY)
    record = cursor.refs[REF]
    assert record.fallback is not None
    assert (record.fallback.kind, record.fallback.code) == ('REPLY_CODE', 'ValidationError')
    resubmitted = [arguments for name, arguments in gated.fake.calls[sent_before:] if name == 'submit_task']
    assert resubmitted and all(SECRET_OBSERVED not in arguments['description'] for arguments in resubmitted)
    fix, _ = gated.fix_and_gate()
    assert _fallback_of(fix) == {'kind': 'REPLY_CODE', 'code': 'ValidationError'}
    _assert_filed_and_gated(gated)


@pytest.mark.parametrize(
    ('interval', 'streak', 'delay'),
    [
        (10, 0, 10),
        (10, 1, 20),
        (10, 3, 80),
        (10, 50, MAX_BACKOFF_SECONDS),
        (MAX_BACKOFF_SECONDS * 2, 3, MAX_BACKOFF_SECONDS * 2),
    ],
)
def test_the_delay_doubles_per_transient_pass_up_to_the_cap(interval: int, streak: int, delay: int) -> None:
    assert next_delay(interval, streak) == delay


@contextlib.contextmanager
def _session(caller: ToolCaller, *, up: bool) -> Iterator[ToolCaller]:
    if not up:
        raise McpCallError(McpFailureKind.TRANSPORT, 'connection refused')
    yield caller


def test_the_loop_stores_the_cursor_after_each_pass_and_backs_off_while_transient(gated: World) -> None:
    append_inbox(gated.in_dir, reported(2, 3, at=AT))
    gated.fake.refuse_next('submit_task', BACKLOG)
    handshakes = iter([False, True, True])
    stop = threading.Event()
    delays: list[float] = []
    stored_offsets: list[int] = []

    def wait(delay: float) -> bool:
        delays.append(delay)
        stored_offsets.append(FilerCursor.model_validate_json((gated.state_dir / CURSOR_NAME).read_bytes()).inbox_offset)
        with pytest.raises(FilerLockHeld), held_state_dir(gated.policy):
            pass
        if len(delays) == 3:
            stop.set()
        return stop.is_set()

    run_until_stopped(
        gated.policy,
        open_tools=lambda policy: _session(gated.fake, up=next(handshakes)),
        stop=stop,
        wait=wait,
    )

    assert gated.policy.filer is not None
    interval = gated.policy.filer.interval_seconds
    assert delays == [next_delay(interval, 1), next_delay(interval, 2), next_delay(interval, 0)]
    assert stored_offsets == [0, 0, gated.inbox_size()]
    _assert_filed_and_gated(gated)
