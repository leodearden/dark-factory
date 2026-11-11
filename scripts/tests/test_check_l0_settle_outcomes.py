"""Row 26 of plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md: the C10 L0 settle check."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import check_l0_settle_outcomes as settle
import pytest
from escalation.archive import archive_dir_for_date

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
    ]
    excluded += [
        build.closed('30', build.ago(days=3), PROMOTION, severity='info'),
        build.closed('__starvation_auto_pin_cap__', build.ago(days=4), PROMOTION),
        build.closed('31', build.ago(days=5), PROMOTION, level=1),
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
        'stranded': 3,
        'stranded_per_100': rate(3, post_filed),
        'reaper_dismissals': 4,
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


def test_info_and_non_numeric_l0s_are_excluded(settle_queue: SettleQueue, capsys: pytest.CaptureFixture[str]) -> None:
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
