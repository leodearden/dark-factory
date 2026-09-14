"""Tests for scripts/derive_orphaned_recon_escalations.py (task 3052).

The operator half of the terminal-subject reaper.  The in-cycle Stage-1 sweep
can only FLAG (the A7b contract above
``reconciliation/harness.py::_RECON_DEDUP_CONFIG`` makes the port-8103 watcher
the sole closer), and the watcher reads ``get_pending_escalations()`` — never
``report.items_flagged`` — so a flag alone cannot clear the 124 records
already on the live queue.  This script re-derives the reap set LIVE and, with
``--apply``, closes it.

Loaded via ``importlib.util.spec_from_file_location`` (the ``_load_module``
helper copied from ``test_audit_duplicate_memories.py``) because ``scripts/``
is not on ``PYTHONPATH``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.models import RESOLUTION_CLASSES, Escalation
from escalation.queue import EscalationQueue

from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
from fused_memory.reconciliation.orphaned_recon_escalation_sweep import classify_orphan
from fused_memory.utils.target_store_preflight import TargetStoreMissing

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'derive_orphaned_recon_escalations.py'
)

GATE_BACKLOG = 'reconciliation_stale_gate_backlog'
HUMAN_OPERATOR = 'reconciliation_stale_human_operator'
DARK_ROOT = '/srv/dark-factory'
PROJECT_ROOTS = {'dark_factory': DARK_ROOT}


def _load_module() -> types.ModuleType:
    """Load the script from its file path, registered in ``sys.modules``.

    Registration is required so ``@dataclass`` and other reflection-based
    decorators resolve ``cls.__module__``.
    """
    import sys  # noqa: PLC0415

    mod_name = 'derive_orphaned_recon_escalations'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


def _detail(project_id: str, task_id: str) -> str:
    return '\n'.join([
        f'project_id: {project_id}',
        'run_id: 62e9b073-a070-47dc-b179-03608db93bef',
        f'task_id: {task_id}',
        'gate_escalated_at: 2026-08-18T21:34:32.310663+00:00',
        'age_hours_at_filing: 48.6',
    ])


def _submit(
    queue: EscalationQueue,
    task_id: str,
    *,
    category: str = GATE_BACKLOG,
    level: int = 1,
    project_id: str = 'dark_factory',
) -> Escalation:
    """Write one record through ``EscalationQueue.submit`` — never hand-rolled JSON."""
    esc = Escalation(
        id=f'esc-{task_id}-1',
        task_id=task_id,
        agent_role='reconciliation-stage1',
        severity='blocking',
        category=category,
        summary=f'Gate task {task_id} has awaited a human decision',
        detail=_detail(project_id, task_id),
        level=level,
    )
    queue.submit(esc)
    return esc


def _make_taskmaster(censuses_by_root):
    """A ``TaskBackendProtocol`` double: ``{project_root: {tag: {id: status}}}``."""

    async def _list_tags(project_root):
        return list(censuses_by_root.get(project_root, {}))

    async def _get_statuses_fresh(project_root, ids=None, tag=None):
        by_tag = censuses_by_root.get(project_root, {})
        if tag is None:
            merged: dict[str, str] = {}
            for per_tag in by_tag.values():
                merged.update(per_tag)
            return merged
        return dict(by_tag.get(tag, {}))

    taskmaster = MagicMock(spec=TaskBackendProtocol)
    taskmaster.list_tags = AsyncMock(side_effect=_list_tags)
    taskmaster.get_statuses_fresh = AsyncMock(side_effect=_get_statuses_fresh)
    return taskmaster


@pytest.fixture
def seeded_queue(tmp_path):
    """A queue dir carrying one record of every classification-relevant shape."""
    queue = EscalationQueue(tmp_path)
    records = {
        'done': _submit(queue, '650'),
        'blocked': _submit(queue, '651'),
        'norow': _submit(queue, '652'),
        'hor_done': _submit(queue, '653', category=HUMAN_OPERATOR),
        'integrity': _submit(queue, '654', category='recon_integrity_issue'),
        'unknown_project': _submit(queue, '655', project_id='pump_web_ui'),
    }
    # A resolved gate-backlog record: submitted then closed, so it leaves the
    # queue root and get_pending() must not see it.
    resolved = _submit(queue, '656')
    queue.resolve(resolved.id, 'closed by hand', resolved_by='test')
    records['resolved'] = resolved
    return queue, tmp_path, records


@pytest.fixture
def taskmaster():
    return _make_taskmaster({
        DARK_ROOT: {
            'master': {
                '650': 'done',
                '651': 'blocked',
                '653': 'cancelled',
                '654': 'done',
                '656': 'done',
            },
        },
    })


def _hash_dir(queue_dir: Path) -> dict[str, str]:
    """Content hash of every file under *queue_dir*, for byte-identity checks."""
    return {
        str(p.relative_to(queue_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(queue_dir.rglob('*'))
        if p.is_file()
    }


class TestDeriveOrphanedReconEscalations:
    """``run(queue_dir, project_roots, *, apply=...)`` derives, and optionally reaps."""

    @pytest.mark.asyncio
    async def test_dry_run_is_the_default_and_touches_nothing(
        self, seeded_queue, taskmaster,
    ):
        """THE LOAD-BEARING SAFETY TEST — every on-disk byte survives a dry run.

        ``--apply`` mutates production operational state, so the default has
        to be inert; hashing the whole queue dir before and after is the only
        check that proves it rather than assuming it.
        """
        _, queue_dir, _ = seeded_queue
        before = _hash_dir(queue_dir)

        report = await _mod.run(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS, taskmaster=taskmaster,
        )

        assert _hash_dir(queue_dir) == before, 'a dry run must not write anything'
        assert report['dry_run'] is True
        assert set(report['reapable_ids']) == {'esc-650-1', 'esc-652-1', 'esc-653-1'}

    @pytest.mark.asyncio
    async def test_classification_matches_the_shared_classifier(
        self, seeded_queue, taskmaster,
    ):
        """The rule has ONE owner, shared with the in-cycle sweep.

        A second copy of the derivation could drift, making the Stage-1 flag
        and the operator reap disagree about which records are safe to close.
        """
        _, queue_dir, records = seeded_queue
        census = {'650': 'done', '651': 'blocked', '653': 'cancelled'}

        report = await _mod.run(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS, taskmaster=taskmaster,
        )

        for key in ('done', 'blocked', 'norow', 'hor_done'):
            esc = records[key]
            expected = classify_orphan(esc, census)
            reaped = esc.id in report['reapable_ids']
            assert reaped is (expected in ('terminal', 'missing')), (
                f'{esc.id} classified {expected!r} by the shared helper but '
                f'{"" if reaped else "not "}selected by the script'
            )

    @pytest.mark.asyncio
    async def test_live_closed_foreign_and_off_category_records_are_excluded(
        self, seeded_queue, taskmaster,
    ):
        """Only terminal/missing pending L1 records of the two families qualify."""
        _, queue_dir, records = seeded_queue

        report = await _mod.run(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS, taskmaster=taskmaster,
        )

        for key in ('blocked', 'resolved', 'integrity', 'unknown_project'):
            assert records[key].id not in report['reapable_ids'], (
                f'{key} record must not be reapable'
            )
        assert report['live'] == 1
        assert report['unresolvable'] == 1, (
            'the unknown-project record is unresolvable, not missing'
        )
        assert report['missing'] == 1

    @pytest.mark.asyncio
    async def test_census_is_cross_tag_complete(self, tmp_path):
        """A subject blocked in a non-default tag is live and is NOT reaped.

        Same soundness requirement as the in-cycle sweep: ``get_statuses_fresh``
        defaults to a single tag, so a single read would report this subject
        as having no row and drive an irreversible close of a live record.
        """
        queue = EscalationQueue(tmp_path)
        _submit(queue, '777')
        taskmaster = _make_taskmaster({
            DARK_ROOT: {'master': {'650': 'done'}, 'feature-x': {'777': 'blocked'}},
        })

        report = await _mod.run(
            queue_dir=tmp_path, project_roots=PROJECT_ROOTS, taskmaster=taskmaster,
        )

        assert report['reapable_ids'] == []
        assert report['live'] == 1
        assert report['missing'] == 0

    @pytest.mark.asyncio
    async def test_a_fail_open_census_reaps_nothing_under_apply(self, seeded_queue):
        """THE IRREVERSIBLE PATH — an empty census must close nothing.

        ``backends/sqlite_task_backend.py::get_statuses_fresh`` "fails open to
        ``{}`` on any error" and never raises, so an unreadable store arrives
        here as a clean empty census.  Unguarded, ``classify_orphan`` renders
        every pending record ``'missing'`` and ``--apply`` closes the entire
        queue — irreversibly, against no evidence at all.  ``list_tags`` is
        ``SELECT DISTINCT tag FROM tasks``, so the tag it just returned has at
        least one row and this emptiness is provably a failed read.
        """
        _, queue_dir, _ = seeded_queue
        taskmaster = _make_taskmaster({DARK_ROOT: {'master': {}}})
        before_pending = {e.id for e in EscalationQueue(queue_dir).get_pending()}

        report = await _mod.run(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS,
            apply=True, taskmaster=taskmaster,
        )

        assert report['errors'] > 0
        assert report['reapable_ids'] == []
        assert report['reaped'] == 0
        assert report['missing'] == 0, (
            'a failed read is never evidence that a subject has no row'
        )
        assert {
            e.id for e in EscalationQueue(queue_dir).get_pending()
        } == before_pending, 'no record may leave pending on a failed census'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('apply_mode', [False, True])
    async def test_a_cross_tag_id_collision_is_never_reaped(self, tmp_path, apply_mode):
        """THE IRREVERSIBLE PATH — an unidentifiable subject must stay pending.

        Ids are per-tag (``PRIMARY KEY (tag, id)`` with a per-tag
        ``id_counters`` high-water mark in
        ``backends/sqlite_task_backend.py``), so the same id in two tags is
        the norm.  A record carries no tag, so subject 777 ``blocked`` in
        ``master`` alongside an unrelated 777 ``done`` in ``feature-x`` cannot
        be resolved — and a last-tag-wins merge would close the live record.
        """
        queue = EscalationQueue(tmp_path)
        _submit(queue, '777')
        taskmaster = _make_taskmaster({
            DARK_ROOT: {
                'master': {'777': 'blocked'},
                'feature-x': {'777': 'done'},
            },
        })

        report = await _mod.run(
            queue_dir=tmp_path, project_roots=PROJECT_ROOTS,
            apply=apply_mode, taskmaster=taskmaster,
        )

        assert report['ambiguous'] == 1
        assert report['reapable_ids'] == []
        assert report['terminal'] == 0
        assert report['reaped'] == 0
        assert {e.id for e in EscalationQueue(tmp_path).get_pending()} == {'esc-777-1'}

    @pytest.mark.asyncio
    async def test_apply_closes_exactly_the_reapable_records(
        self, seeded_queue, taskmaster,
    ):
        """``--apply`` resolves the reap set with the purpose-built stamp."""
        _, queue_dir, records = seeded_queue

        report = await _mod.run(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS,
            apply=True, taskmaster=taskmaster,
        )

        assert report['dry_run'] is False
        assert report['reaped'] == 3
        assert _mod.RESOLUTION_CLASS in RESOLUTION_CLASSES, (
            'resolution_class must be a member EscalationQueue.resolve accepts; '
            'an invalid value raises ValueError before anything is persisted'
        )
        assert _mod.RESOLUTION_CLASS == 'moot-terminal-subject'

        reread = EscalationQueue(queue_dir)
        still_pending = {e.id for e in reread.get_pending()}
        for key in ('done', 'norow', 'hor_done'):
            assert records[key].id not in still_pending
            closed = reread.get(records[key].id)
            assert closed is not None
            assert closed.status in ('resolved', 'dismissed')
            assert closed.resolution_class == _mod.RESOLUTION_CLASS
            assert closed.resolved_by == _mod.RESOLVED_BY
        for key in ('blocked', 'unknown_project'):
            assert records[key].id in still_pending, (
                f'{key} record must stay pending'
            )

    @pytest.mark.asyncio
    async def test_apply_is_idempotent(self, seeded_queue, taskmaster):
        """A second ``--apply`` reaps nothing — the re-derivation excludes closed records.

        This asserts the SCRIPT re-derives rather than leaning on
        ``EscalationQueue.resolve``'s own no-op-on-non-pending behaviour.
        """
        _, queue_dir, _ = seeded_queue
        kwargs = dict(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS,
            apply=True, taskmaster=taskmaster,
        )

        first = await _mod.run(**kwargs)
        second = await _mod.run(**kwargs)

        assert first['reaped'] == 3
        assert second['reaped'] == 0
        assert second['reapable_ids'] == []
        assert second['scanned'] == first['scanned'] - 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize('apply_mode', [False, True])
    async def test_report_is_json_serialisable_with_all_keys_in_both_modes(
        self, seeded_queue, taskmaster, apply_mode,
    ):
        """Every count key is present in BOTH modes — unlike backfill_recon_escalations.

        That script adds ``dismissed``/``updated``/``pending_after`` only under
        ``--apply``, forcing consumers into ``.get()``.  This one does not
        copy the wart: an always-present key set means a dry-run report and an
        apply report are read the same way.
        """
        _, queue_dir, _ = seeded_queue

        report = await _mod.run(
            queue_dir=queue_dir, project_roots=PROJECT_ROOTS,
            apply=apply_mode, taskmaster=taskmaster,
        )

        for key in (
            'dry_run', 'queue_dir', 'scanned', 'terminal', 'missing', 'live',
            'ambiguous', 'unresolvable', 'errors', 'reaped', 'reapable_ids',
        ):
            assert key in report, f'{key} must be present in both modes'
        assert report['queue_dir'] == str(queue_dir)
        json.dumps(report)


class TestMissingQueueDirRefusal:
    """The task-4319 preflight: a queue dir that does not exist is REFUSED.

    ``EscalationQueue.__init__`` mkdirs its ``queue_dir``, so without the guard
    a mis-targeted ``--queue-dir`` -- or the RELATIVE default run from anywhere
    but the project root -- manufactures an empty queue, reports
    ``"scanned": 0, "reaped": 0`` and exits 0: a false all-clear
    indistinguishable from a clean one.

    Every OTHER class in this file passes an EXISTING ``tmp_path``, so the
    guard is a verified no-op for them.  If one of them breaks, that is a
    signal the guard was placed wrongly -- not a licence to weaken it.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('apply_mode', [False, True])
    async def test_refuses_a_missing_queue_dir(self, tmp_path, taskmaster, apply_mode):
        """The DRY RUN refuses too: its report is exactly as false as an apply."""
        missing = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing):
            await _mod.run(
                queue_dir=missing, project_roots=PROJECT_ROOTS,
                apply=apply_mode, taskmaster=taskmaster,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('apply_mode', [False, True])
    async def test_refusal_leaves_no_litter(self, tmp_path, taskmaster, apply_mode):
        """The queue was never constructed, so the ``mkdir`` never happened."""
        missing = tmp_path / 'data' / 'reconciliation' / 'escalations'

        with pytest.raises(TargetStoreMissing):
            await _mod.run(
                queue_dir=missing, project_roots=PROJECT_ROOTS,
                apply=apply_mode, taskmaster=taskmaster,
            )

        assert not missing.exists()
        assert not (tmp_path / 'data').exists()

    @pytest.mark.asyncio
    async def test_refusal_precedes_the_backend_build(self, tmp_path):
        """``taskmaster=None`` would build a real SqliteTaskBackend from config.

        The guard sits ahead of that branch, so a refusal costs no backend and
        no census read -- which is why this call can pass ``taskmaster=None``
        without touching a live store.
        """
        with pytest.raises(TargetStoreMissing):
            await _mod.run(
                queue_dir=tmp_path / 'absent', project_roots=PROJECT_ROOTS,
                taskmaster=None,
            )

    @pytest.mark.asyncio
    async def test_an_existing_queue_dir_passes_the_guard(self, tmp_path, taskmaster):
        """The guard requires neither absoluteness nor a non-empty queue."""
        report = await _mod.run(
            queue_dir=tmp_path, project_roots=PROJECT_ROOTS, taskmaster=taskmaster,
        )

        assert report['scanned'] == 0
        assert report['reaped'] == 0

    def test_main_does_not_return_zero_for_a_missing_queue_dir(self, tmp_path):
        """``main()`` returns 0 UNCONDITIONALLY, with no error accounting.

        A refusal routed through the normal report path would therefore exit 0,
        reproducing the very defect the guard exists to fix.  It must raise.
        """
        import sys as _sys  # noqa: PLC0415

        missing = tmp_path / 'data' / 'reconciliation' / 'escalations'
        old_argv = _sys.argv
        try:
            _sys.argv = [
                'derive_orphaned_recon_escalations.py', '--queue-dir', str(missing),
            ]
            with pytest.raises(TargetStoreMissing):
                _mod.main()
        finally:
            _sys.argv = old_argv

        assert not missing.exists()
