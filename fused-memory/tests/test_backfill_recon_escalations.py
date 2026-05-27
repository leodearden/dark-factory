"""Tests for backfill_recon_escalations.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import types
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from escalation.dedupe import compute_content_fingerprint
from escalation.models import Escalation
from escalation.queue import EscalationQueue

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'backfill_recon_escalations.py'


def _load_module() -> types.ModuleType:
    """Load backfill_recon_escalations.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'backfill_recon_escalations'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
finding_fingerprint = _mod.finding_fingerprint
build_plan = _mod.build_plan
apply_plan = _mod.apply_plan
run = _mod.run
GroupCollapse = _mod.GroupCollapse
BackfillPlan = _mod.BackfillPlan
RESOLVED_BY = _mod.RESOLVED_BY
DEFAULT_NOTE = _mod.DEFAULT_NOTE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(
    *,
    category: str = 'recon_integrity_issue',
    severity: str = 'info',
    finding_category: str = 'systemic_pattern',
    affected_ids: list[str] | None = None,
    description: str = 'Test finding description.',
    task_id: str = '1',
    esc_id: str | None = None,
    timestamp: str | None = None,
) -> Escalation:
    """Build an Escalation suitable for the backfill tests."""
    if affected_ids is None:
        affected_ids = ['1478', '1480']
    detail = json.dumps({
        'category': finding_category,
        'affected_ids': affected_ids,
        'description': description,
        'severity': severity,
        'actionable': True,
        'suggested_action': 'investigate',
    })
    return Escalation(
        id=esc_id or f'esc-{task_id}-{uuid.uuid4().hex[:8]}',
        task_id=task_id,
        agent_role='reconciler',
        severity=severity,
        category=category,
        summary='Systemic pattern detected in tasks.',
        detail=detail,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
    )


def _submit_esc(queue: EscalationQueue, esc: Escalation) -> Escalation:
    """Submit *esc* to *queue* and return it."""
    queue.submit(esc)
    return esc


# ---------------------------------------------------------------------------
# TestFindingFingerprint
# ---------------------------------------------------------------------------

class TestFindingFingerprint:
    """finding_fingerprint extracts fields from detail and calls compute_content_fingerprint."""

    def test_basic_match(self) -> None:
        """Fingerprint equals compute_content_fingerprint with parsed fields."""
        esc = _esc(
            finding_category='systemic_pattern',
            affected_ids=['1478', '1480'],
            description='Duplicate task IDs detected across sprints.',
        )
        expected = compute_content_fingerprint(
            'recon_integrity_issue',
            'systemic_pattern',
            ['1478', '1480'],
            'Duplicate task IDs detected across sprints.',
        )
        assert finding_fingerprint(esc) == expected

    def test_two_identical_findings_share_fingerprint(self) -> None:
        """Two escalations with identical finding fields share a fingerprint."""
        esc1 = _esc(
            finding_category='orphan_dependency',
            affected_ids=['42', '99'],
            description='Orphan found.',
            task_id='100',
        )
        esc2 = _esc(
            finding_category='orphan_dependency',
            affected_ids=['42', '99'],
            description='Orphan found.',
            task_id='200',
        )
        assert finding_fingerprint(esc1) == finding_fingerprint(esc2)

    def test_different_affected_ids_produce_different_fingerprints(self) -> None:
        """Differing affected_ids produce distinct fingerprints."""
        esc1 = _esc(affected_ids=['1', '2'], finding_category='gap')
        esc2 = _esc(affected_ids=['3', '4'], finding_category='gap')
        assert finding_fingerprint(esc1) != finding_fingerprint(esc2)

    def test_malformed_detail_falls_back(self) -> None:
        """Malformed/empty detail falls back to a deterministic fingerprint from esc.summary."""
        esc = Escalation(
            id='esc-bad-1',
            task_id='1',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='Orphan nodes in graph.',
            detail='not-json',
        )
        # Should not raise; should return a deterministic string.
        fp = finding_fingerprint(esc)
        assert isinstance(fp, str)
        assert len(fp) == 64  # sha256 hex digest length

        # Calling again returns the same value (deterministic).
        assert finding_fingerprint(esc) == fp

    def test_empty_detail_falls_back(self) -> None:
        """Empty string detail falls back gracefully."""
        esc = Escalation(
            id='esc-empty-1',
            task_id='1',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='Empty detail escalation.',
            detail='',
        )
        fp = finding_fingerprint(esc)
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_non_dict_detail_falls_back(self) -> None:
        """JSON list as detail (non-dict) falls back gracefully."""
        esc = Escalation(
            id='esc-list-1',
            task_id='1',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary='List detail.',
            detail=json.dumps(['a', 'b', 'c']),
        )
        fp = finding_fingerprint(esc)
        assert isinstance(fp, str)
        assert len(fp) == 64

    def test_fallback_is_compute_content_fingerprint_based(self) -> None:
        """Fallback fingerprint equals compute_content_fingerprint('...', '', [], summary)."""
        summary = 'Orphan nodes in graph.'
        esc = Escalation(
            id='esc-bad-2',
            task_id='1',
            agent_role='reconciler',
            severity='info',
            category='recon_integrity_issue',
            summary=summary,
            detail='bad json{',
        )
        expected = compute_content_fingerprint('recon_integrity_issue', '', [], summary)
        assert finding_fingerprint(esc) == expected


# ---------------------------------------------------------------------------
# TestBuildPlan
# ---------------------------------------------------------------------------

class TestBuildPlan:
    """build_plan groups eligible escalations by fingerprint and plans collapses."""

    def _make_group(
        self,
        n: int,
        *,
        finding_category: str = 'systemic_pattern',
        affected_ids: list[str] | None = None,
        base_offset_seconds: int = 0,
    ) -> list[Escalation]:
        """Create *n* escalations with the same finding (same fingerprint)."""
        if affected_ids is None:
            affected_ids = ['1', '2']
        escs = []
        base = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(n):
            ts = (base + timedelta(seconds=base_offset_seconds + i * 60)).isoformat()
            escs.append(_esc(
                finding_category=finding_category,
                affected_ids=affected_ids,
                timestamp=ts,
                task_id=str(i + 1),
            ))
        return escs

    def test_only_eligible_categories_are_considered(self) -> None:
        """Blocking escalations and recon_stale_run are never in any group."""
        dupes = self._make_group(3)
        infra = Escalation(
            id='esc-infra-1',
            task_id='99',
            agent_role='reconciler',
            severity='blocking',
            category='infra_issue',
            summary='Infra down.',
        )
        stale = Escalation(
            id='esc-stale-1',
            task_id='98',
            agent_role='reconciler',
            severity='info',
            category='recon_stale_run',
            summary='Stale run.',
        )
        pending = dupes + [infra, stale]
        plan = build_plan(pending)

        # Blocking and other categories must never appear in any collapse.
        collapsed_ids = {cid for c in plan.collapses for cid in [c.canonical_id] + c.child_ids}
        assert infra.id not in collapsed_ids
        assert stale.id not in collapsed_ids

    def test_oldest_is_canonical(self) -> None:
        """The escalation with the minimum timestamp becomes canonical."""
        group = self._make_group(4)
        # Shuffle to ensure we're not relying on list order.
        import random
        random.shuffle(group)
        plan = build_plan(group)

        assert len(plan.collapses) == 1
        collapse = plan.collapses[0]
        # Oldest: the one with base_offset_seconds=0 + index=0 (i.e. seconds=0)
        # All _make_group members have base 2026-01-01; the one at index 0 has smallest ts.
        oldest = min(group, key=lambda e: e.timestamp)
        assert collapse.canonical_id == oldest.id

    def test_singleton_groups_not_collapsed(self) -> None:
        """Unique fingerprints (singleton groups) produce no collapse action."""
        unique1 = _esc(affected_ids=['10'], finding_category='gap')
        unique2 = _esc(affected_ids=['20'], finding_category='gap')
        plan = build_plan([unique1, unique2])
        assert plan.collapses == []
        assert plan.groups_collapsed == 0
        assert plan.to_dismiss == 0

    def test_counts_are_correct(self) -> None:
        """Plan counts match actual group distribution."""
        group_a = self._make_group(5, affected_ids=['1', '2'])  # 1 collapse, 4 dismissed
        group_b = self._make_group(3, affected_ids=['3', '4'])  # 1 collapse, 2 dismissed
        singleton = _esc(affected_ids=['99'], finding_category='unique_thing')
        blocking = Escalation(
            id='esc-rf-1',
            task_id='50',
            agent_role='reconciler',
            severity='blocking',
            category='recon_failure',
            summary='Recon failed.',
        )
        pending = group_a + group_b + [singleton, blocking]
        plan = build_plan(pending)

        assert plan.pending_before == len(pending)
        assert plan.eligible_total == len(group_a) + len(group_b) + 1  # +1 for singleton
        assert plan.distinct_fingerprints == 3  # group_a fp, group_b fp, singleton fp
        assert plan.groups_collapsed == 2
        assert plan.to_dismiss == 4 + 2  # children only
        # survivors = 3 canonical/singleton eligible + 1 blocking non-eligible
        assert plan.expected_survivors == plan.pending_before - plan.to_dismiss

    def test_children_are_non_canonical_members(self) -> None:
        """child_ids contains all members of the group except the canonical."""
        group = self._make_group(4)
        plan = build_plan(group)

        assert len(plan.collapses) == 1
        c = plan.collapses[0]
        assert c.group_size == 4
        assert len(c.child_ids) == 3
        assert c.canonical_id not in c.child_ids
        assert set(c.child_ids) | {c.canonical_id} == {e.id for e in group}


# ---------------------------------------------------------------------------
# TestApplyPlan
# ---------------------------------------------------------------------------

class TestApplyPlan:
    """apply_plan performs the actual dismiss-and-archive operations."""

    def test_children_are_dismissed_and_archived(self, tmp_path: Path) -> None:
        """Each child is dismissed, archived (out of queue root), not deleted."""
        queue = EscalationQueue(tmp_path)
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        escs = []
        for i in range(3):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(timestamp=ts, task_id=str(i))
            _submit_esc(queue, e)
            escs.append(e)

        plan = build_plan(queue.get_pending())
        apply_plan(queue, plan)

        child_ids = plan.collapses[0].child_ids

        # Children are no longer pending.
        pending_ids = {e.id for e in queue.get_pending()}
        for cid in child_ids:
            assert cid not in pending_ids

        # Each child is archived (not deleted).
        for cid in child_ids:
            child = queue.get(cid)
            assert child is not None
            assert child.status == 'dismissed'
            assert child.resolved_by == RESOLVED_BY
            assert child.resolution  # non-empty note
            # File should NOT be in the queue root (it was archived).
            root_file = tmp_path / f'{cid}.json'
            assert not root_file.exists(), f'Child {cid} was not archived'

    def test_canonical_is_still_pending_with_dedupe_fields(self, tmp_path: Path) -> None:
        """Canonical remains pending with dedupe_count, dedupe_children, dedupe_fingerprint set."""
        queue = EscalationQueue(tmp_path)
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        escs = []
        n = 4
        for i in range(n):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(timestamp=ts, task_id=str(i), affected_ids=['5', '6'])
            _submit_esc(queue, e)
            escs.append(e)

        plan = build_plan(queue.get_pending())
        apply_plan(queue, plan)

        canonical_id = plan.collapses[0].canonical_id
        fp = plan.collapses[0].fingerprint

        canonical = queue.get(canonical_id)
        assert canonical is not None
        assert canonical.status == 'pending'
        assert canonical.dedupe_count == n
        assert set(canonical.dedupe_children) == set(plan.collapses[0].child_ids)
        assert canonical.dedupe_fingerprint == fp

    def test_blocking_escalations_untouched(self, tmp_path: Path) -> None:
        """Blocking escalations remain pending and unmodified after apply_plan."""
        queue = EscalationQueue(tmp_path)
        blocking = Escalation(
            id='esc-infra-block-1',
            task_id='99',
            agent_role='reconciler',
            severity='blocking',
            category='infra_issue',
            summary='Infra is down.',
        )
        _submit_esc(queue, blocking)

        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(3):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(timestamp=ts, task_id=str(i))
            _submit_esc(queue, e)

        plan = build_plan(queue.get_pending())
        apply_plan(queue, plan)

        reloaded = queue.get(blocking.id)
        assert reloaded is not None
        assert reloaded.status == 'pending'


# ---------------------------------------------------------------------------
# TestRunDryRun
# ---------------------------------------------------------------------------

class TestRunDryRun:
    """run(apply=False) is the default; it must not write anything."""

    def test_dry_run_returns_correct_report(self, tmp_path: Path) -> None:
        """Dry-run report has dry_run=True and correct counts."""
        queue = EscalationQueue(tmp_path)
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(4):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            _submit_esc(queue, _esc(timestamp=ts, task_id=str(i)))

        report = run(tmp_path, apply=False)
        assert report['dry_run'] is True
        assert 'pending_before' in report
        assert 'expected_survivors' in report

    def test_dry_run_performs_zero_writes(self, tmp_path: Path) -> None:
        """Dry-run leaves pending count unchanged and no record modified."""
        queue = EscalationQueue(tmp_path)
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        escs = []
        for i in range(4):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(timestamp=ts, task_id=str(i))
            _submit_esc(queue, e)
            escs.append(e)

        pending_before = len(queue.get_pending())
        run(tmp_path, apply=False)

        # Pending count unchanged.
        assert len(queue.get_pending()) == pending_before

        # No record had its status changed (still pending, dedupe_count still 0).
        for e in escs:
            reloaded = queue.get(e.id)
            assert reloaded is not None
            assert reloaded.status == 'pending'
            assert reloaded.dedupe_count == 0


# ---------------------------------------------------------------------------
# TestApplyAndIdempotency
# ---------------------------------------------------------------------------

class TestApplyAndIdempotency:
    """Acceptance scenario: apply collapses correctly; second run is a no-op."""

    def _seed_queue(self, queue: EscalationQueue) -> dict:
        """Seed queue with two fingerprint groups, singletons, and blocking records."""
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        state: dict = {
            'group_a': [],  # 3 members, same fingerprint
            'group_b': [],  # 2 members, same fingerprint
            'singleton': None,
            'blocking': [],
        }

        # Group A: 3 duplicates
        for i in range(3):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(affected_ids=['a1', 'a2'], timestamp=ts, task_id=f'a{i}')
            queue.submit(e)
            state['group_a'].append(e)

        # Group B: 2 duplicates
        for i in range(2):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(affected_ids=['b1', 'b2'], timestamp=ts, task_id=f'b{i}')
            queue.submit(e)
            state['group_b'].append(e)

        # Singleton
        s = _esc(affected_ids=['singleton'], task_id='s1')
        queue.submit(s)
        state['singleton'] = s

        # Blocking
        for cat in ('infra_issue', 'recon_failure'):
            b = Escalation(
                id=f'esc-{cat}-1',
                task_id='block',
                agent_role='reconciler',
                severity='blocking',
                category=cat,
                summary=f'{cat} occurred.',
            )
            queue.submit(b)
            state['blocking'].append(b)

        return state

    def test_apply_collapses_to_expected_survivor_count(self, tmp_path: Path) -> None:
        """After apply, pending count = distinct fingerprints + non-eligible."""
        queue = EscalationQueue(tmp_path)
        self._seed_queue(queue)

        report = run(tmp_path, apply=True)
        assert report['dry_run'] is False

        # 2 canonicals (group A + B) + 1 singleton + 2 blocking = 5 survivors.
        pending_after = len(queue.get_pending())
        assert pending_after == report['expected_survivors']
        assert pending_after == 5

    def test_second_run_is_noop(self, tmp_path: Path) -> None:
        """Second apply run collapses nothing and preserves dedupe_count."""
        queue = EscalationQueue(tmp_path)
        state = self._seed_queue(queue)

        # First run.
        report1 = run(tmp_path, apply=True)
        canonical_a = min(state['group_a'], key=lambda e: e.timestamp)
        canonical_b = min(state['group_b'], key=lambda e: e.timestamp)

        # Second run.
        report2 = run(tmp_path, apply=True)
        assert report2['groups_collapsed'] == 0
        assert report2['to_dismiss'] == 0
        assert len(queue.get_pending()) == report1['expected_survivors']

        # Canonicals' dedupe_count not clobbered.
        ca = queue.get(canonical_a.id)
        cb = queue.get(canonical_b.id)
        assert ca is not None and ca.dedupe_count == 3
        assert cb is not None and cb.dedupe_count == 2

    def test_blocking_records_survive_apply(self, tmp_path: Path) -> None:
        """Blocking records remain pending after apply."""
        queue = EscalationQueue(tmp_path)
        state = self._seed_queue(queue)

        run(tmp_path, apply=True)

        for b in state['blocking']:
            reloaded = queue.get(b.id)
            assert reloaded is not None
            assert reloaded.status == 'pending'


# ---------------------------------------------------------------------------
# TestSafetyAndCli
# ---------------------------------------------------------------------------

class TestSafetyAndCli:
    """Safety: non-esc files untouched; archive on dismiss; CLI argparse wiring."""

    def test_non_escalation_files_untouched(self, tmp_path: Path) -> None:
        """Sentinel files are never touched; blocking records remain pending."""
        queue = EscalationQueue(tmp_path)

        # Sentinel files that must survive unchanged.
        sentinels = {
            'tasks.db-wal': b'wal data',
            'tasks.db-shm': b'shm data',
            'notes.txt': b'some notes',
        }
        for name, content in sentinels.items():
            (tmp_path / name).write_bytes(content)

        # Blocking records.
        blocking = Escalation(
            id='esc-rf-block-1',
            task_id='99',
            agent_role='reconciler',
            severity='blocking',
            category='recon_failure',
            summary='Recon failure.',
        )
        queue.submit(blocking)

        # Duplicate recon_integrity_issue records (the actual pile).
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(3):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            queue.submit(_esc(timestamp=ts, task_id=str(i)))

        run(tmp_path, apply=True)

        # Sentinel files are byte-for-byte identical.
        for name, content in sentinels.items():
            assert (tmp_path / name).read_bytes() == content, f'{name} was modified'

        # Blocking record still pending.
        reloaded = queue.get(blocking.id)
        assert reloaded is not None
        assert reloaded.status == 'pending'

    def test_dismissed_children_are_archived_not_deleted(self, tmp_path: Path) -> None:
        """Children are archived into the archive subtree, not deleted."""
        queue = EscalationQueue(tmp_path)
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        escs = []
        for i in range(3):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            e = _esc(timestamp=ts, task_id=str(i))
            queue.submit(e)
            escs.append(e)

        plan = build_plan(queue.get_pending())
        apply_plan(queue, plan)

        for c in plan.collapses:
            for cid in c.child_ids:
                # File is gone from root.
                assert not (tmp_path / f'{cid}.json').exists()
                # But retrievable (in archive).
                child = queue.get(cid)
                assert child is not None
                assert child.status == 'dismissed'

    def test_cli_main_dry_run_emits_json(self, tmp_path: Path, capsys) -> None:
        """main() with no --apply prints JSON report to stdout."""
        import sys as _sys  # noqa: PLC0415

        old_argv = _sys.argv
        try:
            _sys.argv = ['backfill_recon_escalations.py', '--queue-dir', str(tmp_path)]
            with contextlib.suppress(SystemExit):
                _mod.main()
        finally:
            _sys.argv = old_argv

        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report.get('dry_run') is True

    def test_cli_apply_flag(self, tmp_path: Path, capsys) -> None:
        """main() with --apply sets dry_run=False in the report."""
        import sys as _sys  # noqa: PLC0415

        queue = EscalationQueue(tmp_path)
        base_ts = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(2):
            ts = (base_ts + timedelta(minutes=i)).isoformat()
            queue.submit(_esc(timestamp=ts, task_id=str(i)))

        old_argv = _sys.argv
        try:
            _sys.argv = ['backfill_recon_escalations.py', '--queue-dir', str(tmp_path), '--apply']
            with contextlib.suppress(SystemExit):
                _mod.main()
        finally:
            _sys.argv = old_argv

        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report.get('dry_run') is False
