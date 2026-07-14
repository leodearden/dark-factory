"""Tests for scripts/tag_cgl_eta_rehome_scope.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_prune_recon_cycle_summaries.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'tag_cgl_eta_rehome_scope.py'


def _load_module() -> types.ModuleType:
    """Load tag_cgl_eta_rehome_scope.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators (e.g. @dataclass) work correctly.
    """
    mod_name = 'tag_cgl_eta_rehome_scope'
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


def _rehome_record(
    id: str,
    src_project: str | None = 'reify',
    src_entity: str | None = 'task 948',
    dst_entity: str | None = None,
    data: str = 'Stage 2 should re-escalate task 948 after remediation.',
    created_at: str = '2026-07-11T21:18:00+00:00',
) -> dict:
    """Build a scroll_by_metadata()-shaped record for a cgl_eta_cross_target_rehome
    entry: ``{'id', 'created_at', 'metadata': {...}}`` where ``metadata`` is the
    full Qdrant payload and the content lives at ``metadata['data']``."""
    metadata = {'kind': 'cgl_eta_cross_target_rehome', 'data': data}
    if src_project is not None:
        metadata['src_project'] = src_project
    if src_entity is not None:
        metadata['src_entity'] = src_entity
    if dst_entity is not None:
        metadata['dst_entity'] = dst_entity
    return {'id': id, 'created_at': created_at, 'metadata': metadata}


# ===========================================================================
# Tests: classify_rehome_record (pure core)
# ===========================================================================

class TestClassifyRehomeRecord:
    def test_taggable_record_returns_tag_action_with_new_content(self):
        record = _rehome_record('m1')
        decision = _mod.classify_rehome_record(record)

        assert decision['id'] == 'm1'
        assert decision['action'] == 'tag'
        assert decision['new_content'] == (
            '[reify:task 948] Stage 2 should re-escalate task 948 after remediation.'
        )

    def test_already_tagged_record_returns_skip_already_tagged(self):
        record = _rehome_record(
            'm2', data='[reify:task 948] Stage 2 should re-escalate task 948.',
        )
        decision = _mod.classify_rehome_record(record)

        assert decision['action'] == 'skip:already_tagged'
        assert decision['new_content'] == record['metadata']['data']

    def test_missing_src_project_returns_skip_no_src_project(self):
        record = _rehome_record('m3', src_project=None)
        decision = _mod.classify_rehome_record(record)

        assert decision['action'] == 'skip:no_src_project'
        assert decision['new_content'] == record['metadata']['data']

    def test_uses_dst_entity_when_src_entity_absent(self):
        record = _rehome_record(
            'm4', src_entity=None, dst_entity='task 111',
            data='Some fact about task 111.',
        )
        decision = _mod.classify_rehome_record(record)

        assert decision['action'] == 'tag'
        assert decision['new_content'] == '[reify:task 111] Some fact about task 111.'

    def test_missing_data_treated_as_empty_content(self):
        record = _rehome_record('m5', data='')
        decision = _mod.classify_rehome_record(record)

        assert decision['action'] == 'tag'
        assert decision['new_content'] == '[reify:task 948] '


# ===========================================================================
# Tests: build_tag_report (pure core)
# ===========================================================================

class TestBuildTagReport:
    def _decisions_by_project(self) -> dict:
        return {
            'reify': [
                _mod.classify_rehome_record(_rehome_record('r1')),
                _mod.classify_rehome_record(_rehome_record(
                    'r2', data='[reify:task 1] already tagged',
                )),
                _mod.classify_rehome_record(_rehome_record('r3', src_project=None)),
            ],
            'dark_factory': [
                _mod.classify_rehome_record(_rehome_record(
                    'd1', src_project='dark_factory', src_entity='task 5',
                )),
            ],
        }

    def test_top_level_keys_present(self):
        report = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        for key in ('dry_run', 'generated_at', 'projects', 'totals', 'changes'):
            assert key in report, f'missing top-level key {key!r}'

    def test_per_project_counts(self):
        report = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        reify = report['projects']['reify']
        assert reify['scanned'] == 3
        assert reify['taggable'] == 1
        assert reify['tagged'] == 0  # dry run: nothing applied yet
        assert reify['skipped'] == 2

        dark_factory = report['projects']['dark_factory']
        assert dark_factory['scanned'] == 1
        assert dark_factory['taggable'] == 1

    def test_totals_aggregate_across_projects(self):
        report = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        assert report['totals']['scanned'] == 4
        assert report['totals']['taggable'] == 2
        assert report['totals']['skipped'] == 2

    def test_changes_only_lists_taggable_records_on_dry_run(self):
        report = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        change_ids = {c['id'] for c in report['changes']}
        assert change_ids == {'r1', 'd1'}
        assert all(c['applied'] is False for c in report['changes'])

    def test_changes_applied_flag_reflects_applied_ids(self):
        report = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids={'r1'}, dry_run=False,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        r1_change = next(c for c in report['changes'] if c['id'] == 'r1')
        assert r1_change['applied'] is True
        d1_change = next(c for c in report['changes'] if c['id'] == 'd1')
        assert d1_change['applied'] is False

        assert report['projects']['reify']['tagged'] == 1

    def test_changes_ordering_is_deterministic(self):
        report1 = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        report2 = _mod.build_tag_report(
            self._decisions_by_project(), applied_ids=set(), dry_run=True,
            generated_at='2026-07-14T00:00:00+00:00',
        )
        assert report1['changes'] == report2['changes']

    def test_empty_decisions_produces_zeroed_report(self):
        report = _mod.build_tag_report(
            {}, applied_ids=set(), dry_run=True, generated_at='2026-07-14T00:00:00+00:00',
        )
        assert report['projects'] == {}
        assert report['changes'] == []
        assert report['totals']['scanned'] == 0


# ===========================================================================
# Tests: run (async, end-to-end live shell)
# ===========================================================================

class TestRun:
    """Async end-to-end tests for run(args, *, memory, known_projects_map).

    Mirrors test_prune_recon_cycle_summaries.TestRun's mock-memory harness.
    """

    def _fixture_records(self) -> list[dict]:
        """One taggable + one already-tagged + one no-src_project record."""
        return [
            _rehome_record('m-taggable'),
            _rehome_record(
                'm-already-tagged',
                data='[reify:task 948] Stage 2 should re-escalate task 948.',
            ),
            _rehome_record('m-no-src-project', src_project=None),
        ]

    def _make_memory(self, records: list[dict] | None = None) -> MagicMock:
        """Mock memory whose mem0 backend exposes the scroll+count+update contract.

        ``scroll_by_metadata`` is assumed to already return only the
        server-side-filtered {'kind': CGL_ETA_REHOME_KIND} records (mirrors
        the real Qdrant payload-filtered scroll).
        """
        memory = MagicMock()
        mem0 = MagicMock()
        all_records = records if records is not None else []
        mem0.scroll_by_metadata = AsyncMock(return_value=all_records)
        mem0.count_by_metadata = AsyncMock(return_value=len(all_records))
        mem0.update = AsyncMock(return_value={'message': 'Memory updated successfully!'})
        memory.mem0 = mem0
        return memory

    def _args(self, apply=False, project_id=None, scan_limit=10000) -> argparse.Namespace:
        return argparse.Namespace(
            apply=apply,
            project_id=project_id,
            scan_limit=scan_limit,
        )

    def _known_map(self, pid='dark_factory') -> dict:
        return {pid: '/some/path'}

    @pytest.mark.asyncio
    async def test_dry_run_no_updates_and_report_shape(self):
        """Dry-run performs NO update calls; scroll_by_metadata is called
        with the CGL-eta rehome filter; report classifies the fixture
        correctly: 1 taggable, 2 skipped (already-tagged + no-src_project)."""
        memory = self._make_memory(self._fixture_records())
        args = self._args(apply=False, project_id='dark_factory', scan_limit=10000)

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.update.assert_not_awaited()
        assert report['dry_run'] is True

        memory.mem0.scroll_by_metadata.assert_awaited_once()
        call = memory.mem0.scroll_by_metadata.call_args
        assert call.args[0].project_id == 'dark_factory'
        assert call.args[1] == {'kind': _mod.CGL_ETA_REHOME_KIND}
        assert call.kwargs.get('limit') == 10000

        dark_factory = report['projects']['dark_factory']
        assert dark_factory['scanned'] == 3
        assert dark_factory['taggable'] == 1
        assert dark_factory['tagged'] == 0
        assert dark_factory['skipped'] == 2

    @pytest.mark.asyncio
    async def test_apply_updates_only_taggable_record_with_preserved_metadata(self):
        """--apply calls memory.mem0.update exactly once -- for the single
        'tag' record in the fixture -- passing (id, tagged_content, scope)
        positionally and the record's FULL original metadata payload as the
        metadata= kwarg, so provenance (src_project/kind/...) survives
        mem0's payload-overwriting _update_memory. The 'skip:already_tagged'
        and 'skip:no_src_project' fixture records must trigger no update at
        all -- proven here by the single-call assertion covering the whole
        fixture, not just the taggable record."""
        records = self._fixture_records()
        memory = self._make_memory(records)
        args = self._args(apply=True, project_id='dark_factory', scan_limit=10000)

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.update.assert_awaited_once()
        call = memory.mem0.update.call_args
        assert call.args[0] == 'm-taggable'
        assert call.args[1] == (
            '[reify:task 948] Stage 2 should re-escalate task 948 after remediation.'
        )
        assert call.args[2].project_id == 'dark_factory'

        taggable_record = next(r for r in records if r['id'] == 'm-taggable')
        assert call.kwargs.get('metadata') == taggable_record['metadata']
        assert call.kwargs['metadata']['src_project'] == 'reify'
        assert call.kwargs['metadata']['kind'] == _mod.CGL_ETA_REHOME_KIND

        assert report['dry_run'] is False
        tag_change = next(c for c in report['changes'] if c['id'] == 'm-taggable')
        assert tag_change['applied'] is True
        assert report['projects']['dark_factory']['tagged'] == 1

    @pytest.mark.asyncio
    async def test_apply_is_idempotent_on_rerun(self):
        """Feeding scroll_by_metadata records whose data already carries the
        scope tag (as if --apply had just run) back into run() with
        apply=True again must result in ZERO update calls -- the tag is
        already present, so classify_rehome_record marks every record
        skip:already_tagged and the apply loop has nothing to do."""
        already_tagged_records = [
            _rehome_record(
                'm-taggable',
                data='[reify:task 948] Stage 2 should re-escalate task 948 after remediation.',
            ),
        ]
        memory = self._make_memory(already_tagged_records)
        args = self._args(apply=True, project_id='dark_factory', scan_limit=10000)

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        memory.mem0.update.assert_not_awaited()
        assert report['dry_run'] is False
        assert report['totals']['taggable'] == 0
        assert report['totals']['tagged'] == 0
