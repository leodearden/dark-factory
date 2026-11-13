"""Row 26 of plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md: the C10 L0 settle check."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import check_l0_settle_outcomes as settle
import pytest
from escalation.archive import ARCHIVE_SUBDIR, DATE_FORMAT, archive_dir_for_date
from escalation.models import RESOLUTION_CLASSES, Escalation
from escalation.queue import EscalationQueue
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(settle.__file__).resolve()
CATEGORY = 'design_concern'

STEWARD = ('resolved', 'steward', 'steward fixed the plan', None)
PROMOTION = (
    'dismissed',
    'harness-orphan-reaper',
    'Auto-promoted to level 1 — orphan L0 for task_id={task_id} (no active workflow)',
    None,
)
STALE_STRAND = (
    'dismissed',
    'auto-dismissed',
    'Auto-dismissed: orchestrator restarted — stale from prior run [pending_secs=900 severity=blocking]',
    'stale-strand',
)
RESTART_DISMISSAL = (
    'dismissed',
    'auto-dismissed',
    'Auto-dismissed: orchestrator restarted — stale from prior run [pending_secs=40 severity=blocking]',
    None,
)
REAPER_DISMISSAL = (
    'dismissed',
    'harness-orphan-reaper',
    'Dismissed by harness-orphan-reaper — open L1 already covers this task '
    '(task_id={task_id}, category=design_concern, covering=esc-{task_id}-90)',
    None,
)
ARCHITECT_PROMOTION = (
    'dismissed',
    'orchestrator-workflow',
    'Auto-promoted to level 1 as esc-{task_id}-91 — architect filed it in lieu of a plan',
    None,
)
WORKFLOW_DISMISSAL = (
    'dismissed',
    'auto-dismissed',
    'Auto-dismissed: steward interrupted (timeout) with WIP present — task blocked for review',
    None,
)
CLOSE_DELAY = timedelta(hours=1)
PERMISSIVE: dict[str, str] = {
    'window_days': '14',
    'baseline_stranded_per_100': '3.6',
    'ratio': '100',
    'min_stranded': '99',
    'max_reaper_wrong': '99',
    'max_pending_hours': '48',
}


def iso(moment: datetime) -> str:
    return moment.isoformat()


def write_record(queue_dir: Path, **fields: Any) -> Path:
    """Write one escalation where the queue keeps it: pending at the root, closed under its archive day."""
    record: dict[str, Any] = {
        'agent_role': 'implementer',
        'summary': f'escalation {fields["id"]}',
        'detail': '',
        'resolution': None,
        'resolved_at': None,
        'resolved_by': None,
        'resolution_class': None,
        'late_resolutions': [],
        **fields,
    }
    directory = queue_dir if record['status'] == 'pending' else archive_dir_for_date(queue_dir, record['resolved_at'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{record["id"]}.json'
    path.write_text(json.dumps(record))
    return path


def late_resolution(at: datetime) -> dict[str, Any]:
    return {
        'timestamp': iso(at),
        'resolution': 'steward finding that lost the race to the sweep',
        'resolved_by': 'steward',
        'dismiss': False,
        'prior_resolution_class': 'benign',
    }


@dataclass
class FixtureQueue:
    """Writes escalations into one tmp queue directory, each id unique per task."""

    queue_dir: Path
    now: datetime
    _sequence: count = field(default_factory=lambda: count(1))

    def ago(self, *, days: float = 0, hours: float = 0) -> datetime:
        return self.now - timedelta(days=days, hours=hours)

    def closed(
        self,
        task_id: str,
        filed_at: datetime,
        closure: tuple[str, str, str, str | None],
        *,
        severity: str = 'blocking',
        level: int = 0,
        late: bool = False,
    ) -> Path:
        status, resolved_by, resolution, resolution_class = closure
        resolved_at = filed_at + CLOSE_DELAY
        return write_record(
            self.queue_dir,
            id=f'esc-{task_id}-{next(self._sequence)}',
            task_id=task_id,
            severity=severity,
            category=CATEGORY,
            status=status,
            level=level,
            timestamp=iso(filed_at),
            resolved_at=iso(resolved_at),
            resolved_by=resolved_by,
            resolution=resolution.format(task_id=task_id),
            resolution_class=resolution_class,
            late_resolutions=[late_resolution(resolved_at + timedelta(minutes=5))] if late else [],
        )

    def pending(self, task_id: str, filed_at: datetime) -> Path:
        return write_record(
            self.queue_dir,
            id=f'esc-{task_id}-{next(self._sequence)}',
            task_id=task_id,
            severity='blocking',
            category=CATEGORY,
            status='pending',
            level=0,
            timestamp=iso(filed_at),
        )


@dataclass(frozen=True)
class SettleQueue:
    queue_dir: Path
    post: tuple[Path, ...]
    pre: tuple[Path, ...]
    excluded: tuple[Path, ...]


def build_settle_queue(queue_dir: Path) -> SettleQueue:
    """Counted blocking L0s per window, plus records every count must ignore."""
    build = FixtureQueue(queue_dir, datetime.now(UTC))
    post: list[Path] = []
    pre: list[Path] = []
    excluded: list[Path] = []

    dismissed_then_refiled = build.ago(days=5)
    dismissed_then_refiled_late = build.ago(days=8)
    dismissed_then_l2 = build.ago(days=12)
    post += [
        build.closed('11', build.ago(days=2), STEWARD),
        build.closed('12', build.ago(days=3), PROMOTION),
        build.closed('13', build.ago(days=4), PROMOTION),
        build.closed('14', build.ago(days=6), STALE_STRAND),
        build.closed('15', build.ago(days=7), REAPER_DISMISSAL, late=True),
        build.closed('16', dismissed_then_refiled, REAPER_DISMISSAL),
        build.closed('16', dismissed_then_refiled + CLOSE_DELAY + timedelta(hours=2), STEWARD),
        build.closed('17', build.ago(days=9), REAPER_DISMISSAL),
        build.closed('18', dismissed_then_refiled_late, REAPER_DISMISSAL),
        build.closed('18', dismissed_then_refiled_late + CLOSE_DELAY + timedelta(hours=30), STEWARD),
        build.closed('19', build.ago(days=10), RESTART_DISMISSAL, late=True),
        build.closed('20', build.ago(days=11), WORKFLOW_DISMISSAL, late=True),
        build.pending('21', build.ago(hours=30)),
        build.closed('22', dismissed_then_l2, REAPER_DISMISSAL),
        build.closed('24', build.ago(days=13), ARCHITECT_PROMOTION),
    ]
    excluded += [
        build.closed('30', build.ago(days=3), PROMOTION, severity='info'),
        build.closed('__starvation_auto_pin_cap__', build.ago(days=4), PROMOTION),
        build.closed('31', build.ago(days=5), PROMOTION, level=1),
        build.closed('22', dismissed_then_l2 + CLOSE_DELAY + timedelta(hours=2), STEWARD, level=2),
    ]
    pre += [build.closed(str(40 + n), build.ago(days=15 + n), STEWARD) for n in range(7)]
    pre += [build.closed('50', build.ago(days=25), PROMOTION)]
    return SettleQueue(queue_dir, tuple(post), tuple(pre), tuple(excluded))


@pytest.fixture
def settle_queue(tmp_path: Path) -> SettleQueue:
    return build_settle_queue(tmp_path / 'escalations')


def argv_for(queue_dir: Path, **overrides: str) -> list[str]:
    flags = {**PERMISSIVE, **overrides}
    argv = ['--queue-dir', str(queue_dir)]
    for name, value in flags.items():
        argv += [f'--{name.replace("_", "-")}', value]
    return argv


def run_main(capsys: pytest.CaptureFixture[str], queue_dir: Path, **overrides: str) -> tuple[int, list[str]]:
    code = settle.main(argv_for(queue_dir, **overrides))
    return code, capsys.readouterr().out.splitlines()


def trailing_json(lines: Sequence[str]) -> dict[str, Any]:
    return json.loads(lines[-1])


def rate(stranded: int, filed: int) -> float:
    return round(100 * stranded / filed, 2)


def test_trailing_json_equals_fixture_counts(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    code, lines = run_main(capsys, settle_queue.queue_dir)

    payload = trailing_json(lines)
    post_filed = len(settle_queue.post)
    pre_filed = len(settle_queue.pre)
    assert code == 0
    assert payload['post'] == {
        'filed': post_filed,
        'reaper_promotions': 2,
        'stale_strands': 1,
        'other_promotions': 1,
        'stranded': 3,
        'stranded_per_100': rate(3, post_filed),
        'reaper_dismissals': 5,
        'reaper_wrong': 2,
        'startup_dismissals': 2,
        'startup_wrong': 1,
        'workflow_auto_dismissals': 1,
        'workflow_auto_wrong': 1,
    }
    assert payload['pre'] == {
        'filed': pre_filed,
        'reaper_promotions': 1,
        'stale_strands': 0,
        'other_promotions': 0,
        'stranded': 1,
        'stranded_per_100': rate(1, pre_filed),
        'reaper_dismissals': 0,
        'reaper_wrong': 0,
        'startup_dismissals': 0,
        'startup_wrong': 0,
        'workflow_auto_dismissals': 0,
        'workflow_auto_wrong': 0,
    }
    assert payload['pending_over_h'] == 0


def test_uncounted_records_change_no_count(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    _, with_excluded = run_main(capsys, settle_queue.queue_dir)
    for path in settle_queue.excluded:
        path.unlink()
    _, without_excluded = run_main(capsys, settle_queue.queue_dir)

    assert trailing_json(with_excluded) == trailing_json(without_excluded)


def test_pending_counts_against_now(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    _, under_24 = run_main(capsys, settle_queue.queue_dir, max_pending_hours='24')
    _, under_48 = run_main(capsys, settle_queue.queue_dir, max_pending_hours='48')

    assert trailing_json(under_24)['pending_over_h'] == 1
    assert trailing_json(under_48)['pending_over_h'] == 0


REDESIGN_TEXT = 'open the joint L0 end-of-life redesign with the harness plan (S8)'
STRANDED_BREACH = {'ratio': '1.0', 'min_stranded': '3'}
STRANDED_BELOW_MIN_COUNT = {'ratio': '1.0', 'min_stranded': '4'}
STRANDED_UNDER_FLOOR = {**STRANDED_BREACH, 'baseline_stranded_per_100': '50'}
STRANDED_UNDER_PRE_BAR = {'ratio': '2.0', 'min_stranded': '3', 'baseline_stranded_per_100': '3.6'}
REAPER_WRONG_BREACH = {'max_reaper_wrong': '2'}
REAPER_WRONG_UNDER = {'max_reaper_wrong': '3'}
PENDING_BREACH = {'max_pending_hours': '24'}
THRESHOLD_CASES = (
    {},
    STRANDED_BREACH,
    STRANDED_BELOW_MIN_COUNT,
    STRANDED_UNDER_FLOOR,
    STRANDED_UNDER_PRE_BAR,
    REAPER_WRONG_BREACH,
    REAPER_WRONG_UNDER,
    PENDING_BREACH,
)


def assert_breached(code: int, lines: Sequence[str], *rules: str) -> None:
    payload = trailing_json(lines)
    assert code == 1
    assert payload['breaches'] == list(rules)
    assert payload['follow_up'] == REDESIGN_TEXT
    assert REDESIGN_TEXT in lines[-1]
    assert lines[-2] == f'follow-up: {REDESIGN_TEXT}'
    for rule in rules:
        assert any(line.startswith(f'breach: {rule}') for line in lines)


def assert_clean(code: int, lines: Sequence[str]) -> None:
    payload = trailing_json(lines)
    assert code == 0
    assert payload['breaches'] == []
    assert payload['follow_up'] is None
    assert REDESIGN_TEXT not in '\n'.join(lines)


def test_exit_0_below_every_threshold(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    assert_clean(*run_main(capsys, settle_queue.queue_dir))


def test_stranded_rate_breach(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    assert_breached(*run_main(capsys, settle_queue.queue_dir, **STRANDED_BREACH), settle.Breach.STRANDED_RATE)


def test_stranded_rate_needs_min_count(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    assert_clean(*run_main(capsys, settle_queue.queue_dir, **STRANDED_BELOW_MIN_COUNT))


def test_stranded_rate_uses_baseline_floor(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    _, permissive = run_main(capsys, settle_queue.queue_dir)
    post_rate = trailing_json(permissive)['post']['stranded_per_100']
    pre_rate = trailing_json(permissive)['pre']['stranded_per_100']
    assert pre_rate < post_rate < float(STRANDED_UNDER_FLOOR['baseline_stranded_per_100'])

    assert_clean(*run_main(capsys, settle_queue.queue_dir, **STRANDED_UNDER_FLOOR))


def test_stranded_rate_bar_scales_the_pre_window_rate(
    settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str],
) -> None:
    _, permissive = run_main(capsys, settle_queue.queue_dir)
    post_rate = trailing_json(permissive)['post']['stranded_per_100']
    pre_rate = trailing_json(permissive)['pre']['stranded_per_100']
    ratio = float(STRANDED_UNDER_PRE_BAR['ratio'])
    floor = float(STRANDED_UNDER_PRE_BAR['baseline_stranded_per_100'])
    assert ratio * floor < post_rate < ratio * pre_rate

    code, lines = run_main(capsys, settle_queue.queue_dir, **STRANDED_UNDER_PRE_BAR)

    assert_clean(code, lines)
    assert trailing_json(lines)['stranded_bar_per_100'] == pytest.approx(ratio * pre_rate)


def test_reaper_wrong_breach(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    assert_breached(*run_main(capsys, settle_queue.queue_dir, **REAPER_WRONG_BREACH), settle.Breach.REAPER_WRONG)
    assert_clean(*run_main(capsys, settle_queue.queue_dir, **REAPER_WRONG_UNDER))


def test_pending_breach(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    assert_breached(*run_main(capsys, settle_queue.queue_dir, **PENDING_BREACH), settle.Breach.PENDING_OVER_H)


def test_startup_and_workflow_wrong_are_never_judged(
    settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str],
) -> None:
    build = FixtureQueue(settle_queue.queue_dir, datetime.now(UTC))
    for n in range(5):
        build.closed(str(60 + n), build.ago(days=2, hours=n), RESTART_DISMISSAL, late=True)
        build.closed(str(70 + n), build.ago(days=3, hours=n), WORKFLOW_DISMISSAL, late=True)

    code, lines = run_main(capsys, settle_queue.queue_dir, **REAPER_WRONG_UNDER)

    post = trailing_json(lines)['post']
    assert post['startup_wrong'] == 6
    assert post['workflow_auto_wrong'] == 6
    assert_clean(code, lines)


def snapshot(root: Path) -> dict[str, tuple[bytes | None, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes() if path.is_file() else None, path.stat().st_mtime_ns)
        for path in sorted(root.rglob('*'))
    }


def test_queue_dir_is_unchanged(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    before = snapshot(settle_queue.queue_dir)

    for overrides in THRESHOLD_CASES:
        run_main(capsys, settle_queue.queue_dir, **overrides)

    assert snapshot(settle_queue.queue_dir) == before


def test_executable_entry_point_end_to_end(settle_queue: SettleQueue) -> None:
    completed = subprocess.run(
        [str(SCRIPT), *argv_for(settle_queue.queue_dir, **PENDING_BREACH)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.stderr == ''
    assert_breached(completed.returncode, completed.stdout.splitlines(), settle.Breach.PENDING_OVER_H)


UNREADABLE = 'unreadable queue directory'
NOTHING_FILED = 'nothing to judge'


def assert_cannot_judge(code: int, lines: Sequence[str], headline: str, path: Path) -> None:
    assert code == 2
    assert lines[-1].startswith(f'{headline}: {path} (')
    assert REDESIGN_TEXT not in '\n'.join(lines)


def test_missing_queue_dir_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    absent = tmp_path / 'absent'

    assert_cannot_judge(*run_main(capsys, absent), UNREADABLE, absent)


def test_queue_dir_is_a_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    not_a_dir = tmp_path / 'escalations'
    not_a_dir.write_text('')

    assert_cannot_judge(*run_main(capsys, not_a_dir), UNREADABLE, not_a_dir)


def test_missing_archive_dir_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build = FixtureQueue(tmp_path / 'escalations', datetime.now(UTC))
    build.pending('21', build.ago(hours=2))

    assert_cannot_judge(*run_main(capsys, build.queue_dir), UNREADABLE, build.queue_dir / 'archive')


def test_corrupt_record_exits_2(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    corrupt = archive_dir_for_date(settle_queue.queue_dir, iso(datetime.now(UTC))) / 'esc-1-9.json'
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text('{not json')

    assert_cannot_judge(*run_main(capsys, settle_queue.queue_dir), UNREADABLE, corrupt)


def test_record_missing_required_key_exits_2(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    build = FixtureQueue(settle_queue.queue_dir, datetime.now(UTC))
    keyless = build.closed('61', build.ago(days=2), STEWARD)
    record = json.loads(keyless.read_text())
    del record['timestamp']
    keyless.write_text(json.dumps(record))

    assert_cannot_judge(*run_main(capsys, settle_queue.queue_dir), UNREADABLE, keyless)


def test_empty_post_window_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build = FixtureQueue(tmp_path / 'escalations', datetime.now(UTC))
    for n in range(3):
        build.closed(str(40 + n), build.ago(days=15 + n), PROMOTION)

    code, lines = run_main(capsys, build.queue_dir)

    assert_cannot_judge(code, lines, NOTHING_FILED, build.queue_dir)
    assert 'no blocking L0 filed in the last 14 days' in lines[-1]


def vanish_on_first_read(monkeypatch: pytest.MonkeyPatch, target: Path, make_vanish: Callable[[], object]) -> None:
    """Run make_vanish just before target is first read, as a queue writer racing the scan would."""
    read_text = Path.read_text
    pending = [make_vanish]

    def racing_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == target and pending:
            pending.pop()()
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', racing_read_text)


def test_archived_record_vanishing_mid_scan_exits_2(
    settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    vanishing = settle_queue.pre[0]
    vanish_on_first_read(monkeypatch, vanishing, vanishing.unlink)

    assert_cannot_judge(*run_main(capsys, settle_queue.queue_dir), UNREADABLE, vanishing)


def test_live_record_vanishing_without_an_archived_copy_exits_2(
    settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = next(path for path in settle_queue.post if path.parent == settle_queue.queue_dir)
    vanish_on_first_read(monkeypatch, live, live.unlink)

    assert_cannot_judge(*run_main(capsys, settle_queue.queue_dir), UNREADABLE, live)


def test_non_record_files_are_ignored(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
    queue_dir = settle_queue.queue_dir
    _, before = run_main(capsys, queue_dir)
    unreadable_if_read = '{not json'
    (queue_dir / 'b3-state.json').write_text(unreadable_if_read)
    (queue_dir / 'esc-1.seq').write_text('7')
    (queue_dir / 'esc-1-1.json.lock').write_text('')
    (queue_dir / 'archive' / 'esc-508-96.json').write_text(unreadable_if_read)
    (queue_dir / 'archive' / 'not-a-date').mkdir()
    (queue_dir / 'archive' / 'not-a-date' / 'x.json').write_text(unreadable_if_read)
    write_record(
        queue_dir,
        id='esc-reconciliation-backlog-dark_factory-2026-11-10T225816_4098180000',
        task_id=None,
        agent_role='fused-memory',
        severity='blocking',
        category='infra_issue',
        status='pending',
        level=1,
        timestamp=iso(datetime.now(UTC) - timedelta(days=1)),
    )

    code, after = run_main(capsys, queue_dir)

    assert code == 0
    assert trailing_json(after) == trailing_json(before)


def test_root_copy_wins_over_an_archived_copy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    queue_dir = tmp_path / 'escalations'
    filed_at = datetime.now(UTC) - timedelta(hours=3)
    same_id = {
        'id': 'esc-23-1',
        'task_id': '23',
        'severity': 'blocking',
        'category': CATEGORY,
        'level': 0,
        'timestamp': iso(filed_at),
    }
    status, resolved_by, resolution, resolution_class = PROMOTION
    archived = write_record(
        queue_dir,
        **same_id,
        status=status,
        resolved_at=iso(filed_at + CLOSE_DELAY),
        resolved_by=resolved_by,
        resolution=resolution.format(task_id='23'),
        resolution_class=resolution_class,
    )
    live = write_record(queue_dir, **same_id, status='pending')
    assert archived.parent != live.parent

    _, lines = run_main(capsys, queue_dir, max_pending_hours='1')

    payload = trailing_json(lines)
    assert (payload['post']['filed'], payload['post']['reaper_promotions']) == (1, 0)
    assert payload['pending_over_h'] == 1


def test_mirrored_archive_layout_equals_the_escalation_producer() -> None:
    assert settle.ARCHIVE_SUBDIR == ARCHIVE_SUBDIR
    assert settle.ARCHIVE_DAY_FORMAT == DATE_FORMAT
    assert settle.STALE_STRAND_CLASS in RESOLUTION_CLASSES


def file_escalation(queue: EscalationQueue, task_id: str, summary: str, *, level: int = 0, filed_at: datetime) -> str:
    escalation_id = queue.make_id(task_id)
    queue.submit(
        Escalation(
            id=escalation_id,
            task_id=task_id,
            agent_role='implementer',
            severity='blocking',
            category=CATEGORY,
            summary=summary,
            level=level,
            timestamp=iso(filed_at),
        )
    )
    return escalation_id


def test_live_record_the_queue_archives_mid_scan_is_counted_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_dir = tmp_path / 'q'
    queue = EscalationQueue(queue_dir)
    now = datetime.now(UTC)
    resolved_id = file_escalation(queue, '104', 'ordinary steward work', filed_at=now - timedelta(minutes=40))
    queue.resolve(resolved_id, 'steward fixed it', resolved_by='steward')
    orphan_id = file_escalation(queue, '105', 'orphan the reaper promotes', filed_at=now - timedelta(minutes=30))
    live = queue_dir / f'{orphan_id}.json'
    _, resolved_by, resolution, _ = PROMOTION
    vanish_on_first_read(
        monkeypatch,
        live,
        lambda: queue.resolve(orphan_id, resolution.format(task_id='105'), dismiss=True, resolved_by=resolved_by),
    )

    code, lines = run_main(capsys, queue_dir)

    post = trailing_json(lines)['post']
    assert not live.exists()
    assert code == 0
    assert (post['filed'], post['reaper_promotions']) == (2, 1)


def test_counts_records_written_by_the_real_queue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    queue_dir = tmp_path / 'q'
    queue = EscalationQueue(queue_dir)
    now = datetime.now(UTC)

    file_escalation(queue, '101', 'stranded across a restart', filed_at=now - timedelta(minutes=20))
    queue.dismiss_all_pending('Auto-dismissed: orchestrator restarted — stale from prior run', strand_age_secs=600)

    file_escalation(queue, '102', 'open L1 for the task', level=1, filed_at=now - timedelta(minutes=10))
    covered_id = file_escalation(queue, '102', 'L0 the reaper sees as covered', filed_at=now - timedelta(minutes=5))
    covered = queue.get(covered_id)
    assert covered is not None
    assert queue.dismiss_if_covered(covered, resolver='harness-orphan-reaper')
    queue.resolve(covered_id, 'steward finding after the sweep', resolved_by='steward')

    resolved_id = file_escalation(queue, '103', 'ordinary steward work', filed_at=now - timedelta(minutes=2))
    queue.resolve(resolved_id, 'steward fixed it', resolved_by='steward')

    code, lines = run_main(capsys, queue_dir)

    post = trailing_json(lines)['post']
    assert code == 0
    assert (post['stale_strands'], post['stranded']) == (1, 1)
    assert (post['reaper_dismissals'], post['reaper_wrong']) == (1, 1)
    assert post['startup_dismissals'] == 1
    assert post['filed'] == 3
