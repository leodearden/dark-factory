"""Tests for scripts/dismiss_recon_integrity_noise.py.

This suite was created by task 4319 alongside the target-store preflight — the
script had no test file at all before, so the baseline classes below close that
gap as well as pinning the new guard.

The guard case is the interesting one. ``EscalationQueue.__init__`` does
``mkdir(parents=True, exist_ok=True)``, so WITHOUT the preflight this script
pointed at a missing queue dir manufactures an empty queue, reports
``"pending_before": 0, "to_dismiss": 0``, and ``main()`` returns 0 — a false
all-clear indistinguishable from a genuinely quiet queue. The script's
``--queue-dir`` default is the RELATIVE ``./data/reconciliation/escalations``,
so this is exactly what a run from a task worktree does.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from fused_memory.utils.target_store_preflight import TargetStoreMissing

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'dismiss_recon_integrity_noise.py'

_mod = load_script_module(SCRIPT_PATH)
run = _mod.run
main = _mod.main
TARGET_CATEGORY = _mod.TARGET_CATEGORY
_apply_exit_code = _mod._apply_exit_code


def _esc(
    *,
    category: str = TARGET_CATEGORY,
    severity: str = 'info',
    task_id: str = '1',
) -> Escalation:
    """Build an Escalation suitable for these tests."""
    return Escalation(
        id=f'esc-{task_id}-{uuid.uuid4().hex[:8]}',
        task_id=task_id,
        agent_role='reconciler',
        severity=severity,
        category=category,
        summary='Non-actionable integrity finding: test.',
        detail=json.dumps({'category': 'systemic_pattern', 'description': 'test'}),
        timestamp=datetime.now(UTC).isoformat(),
    )


class TestRunTargetStorePreflight:
    """The refusal, and the specific lie it replaces."""

    def test_dry_run_refuses_a_missing_queue_dir(self, tmp_path: Path):
        """The DRY RUN refuses too — its report is exactly as false as an apply.

        ``"pending_before": 0`` from a queue that did not exist until the
        script looked at it IS the defect, so gating on ``--apply`` would let
        the misleading report print and exit 0.
        """
        with pytest.raises(TargetStoreMissing):
            run(tmp_path / 'data' / 'reconciliation' / 'escalations', apply=False)

    def test_apply_refuses_a_missing_queue_dir(self, tmp_path: Path):
        with pytest.raises(TargetStoreMissing):
            run(tmp_path / 'data' / 'reconciliation' / 'escalations', apply=True)

    @pytest.mark.parametrize('apply', [False, True])
    def test_refusal_leaves_no_litter(self, tmp_path: Path, apply: bool):
        """``EscalationQueue.__init__`` was never reached, so nothing was created.

        This is why the guard sits before the construction rather than at the
        ``--apply`` gate: the ``mkdir`` happens before ``apply`` is consulted.
        """
        target = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing):
            run(target, apply=apply)

        assert not target.exists()
        assert not (tmp_path / 'data').exists()

    def test_main_does_not_return_zero_for_a_missing_queue_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The regression pin: ``main()`` returns 0 UNCONDITIONALLY.

        There is no error accounting in this script's ``main()`` at all, so a
        refusal routed through the normal report path would exit 0 — the very
        ``no-silent-fail-soft`` defect the guard exists to fix. It must raise
        instead.
        """
        missing = tmp_path / 'data' / 'reconciliation' / 'escalations'
        monkeypatch.setattr(
            'sys.argv',
            ['dismiss_recon_integrity_noise.py', '--queue-dir', str(missing)],
        )

        with pytest.raises(TargetStoreMissing):
            main()

    def test_existing_queue_dir_passes_the_guard(self, tmp_path: Path):
        """The guard does not require absoluteness, or a non-empty queue."""
        report = run(tmp_path, apply=False)

        assert report['pending_before'] == 0


class TestRunBaseline:
    """Baseline coverage this script never had, against a real queue."""

    def test_dry_run_reports_without_writing(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path)
        noise = [_esc(task_id=str(i)) for i in range(3)]
        keeper = _esc(category='infra_issue', severity='blocking', task_id='9')
        for esc in [*noise, keeper]:
            queue.submit(esc)

        report = run(tmp_path, apply=False)

        assert report['dry_run'] is True
        assert report['pending_before'] == 4
        assert report['to_dismiss'] == 3
        assert report['kept'] == 1
        assert report['kept_by_category'] == {'infra_issue': 1}
        assert 'dismissed' not in report
        assert len(EscalationQueue(tmp_path).get_pending()) == 4

    def test_apply_dismisses_only_the_target_category(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path)
        for i in range(3):
            queue.submit(_esc(task_id=str(i)))
        keeper = _esc(category='infra_issue', severity='blocking', task_id='9')
        queue.submit(keeper)

        report = run(tmp_path, apply=True)

        assert report['dry_run'] is False
        assert report['dismissed'] == 3
        assert report['pending_after'] == 1
        still_pending = EscalationQueue(tmp_path).get_pending()
        assert [e.id for e in still_pending] == [keeper.id]


# ---------------------------------------------------------------------------
# TestVanishedTargetAccounting (task 4996)
# ---------------------------------------------------------------------------

class TestVanishedTargetAccounting:
    """A target that vanishes between get_pending() and resolve() must be
    counted, not just logged — INV-11 (a log is not a return value)."""

    def test_vanished_target_is_counted_and_undercounts_dismissed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """State drift: a targeted escalation vanishes *during* the apply
        loop, between ``get_pending()`` (which built ``to_dismiss``) and this
        specific target's own ``resolve()`` call.  ``dismissed`` must fall
        short of ``to_dismiss`` and the shortfall must be named in the
        report, not just a WARNING log line.
        """
        queue = EscalationQueue(tmp_path)
        targets = [_esc(task_id=str(i)) for i in range(3)]
        for esc in targets:
            queue.submit(esc)

        vanished_id = targets[0].id
        real_resolve = EscalationQueue.resolve

        def _resolve_with_one_vanish(self, escalation_id, *args, **kwargs):
            if escalation_id == vanished_id:
                return None
            return real_resolve(self, escalation_id, *args, **kwargs)

        monkeypatch.setattr(EscalationQueue, 'resolve', _resolve_with_one_vanish)

        report = run(tmp_path, apply=True)

        assert report['to_dismiss'] == 3
        assert report['dismissed'] == 2
        assert report['vanished'] == 1
        # The "vanished" target's own file was never touched by resolve(), so
        # it is still sitting in the queue, pending.
        assert report['pending_after'] == 1


class TestApplyExitCode:
    """_apply_exit_code(report) turns a run() report into a loud, non-zero
    process exit whenever a targeted escalation vanished before resolve —
    for CI/operator wiring."""

    def test_clean_apply_report_exits_zero(self) -> None:
        report = {'dry_run': False, 'dismissed': 3, 'vanished': 0}
        assert _apply_exit_code(report) == 0

    def test_vanished_present_exits_non_zero(self) -> None:
        report = {'dry_run': False, 'dismissed': 2, 'vanished': 1}
        assert _apply_exit_code(report) != 0

    def test_dry_run_report_has_no_vanished_key_and_exits_zero(self) -> None:
        """Dry-run reports never populate 'vanished' — the default keeps it clean."""
        report = {'dry_run': True, 'to_dismiss': 3, 'kept': 1}
        assert _apply_exit_code(report) == 0
