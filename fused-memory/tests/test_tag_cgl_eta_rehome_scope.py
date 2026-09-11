"""Tests for scripts/tag_cgl_eta_rehome_scope.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution -- mirrors the pattern in
test_prune_recon_cycle_summaries.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
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


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``run(..., apply=True)`` runs a fail-closed capability preflight before it
    tags (task 4293). That probe touches the real filesystem, so without this
    fixture every ``--apply`` test would pass or fail according to whether the
    machine running pytest happens to be able to write mem0's history
    directory -- and it genuinely cannot inside an agent sandbox, which is the
    whole reason the guard exists. This suite is deliberately MOCK-unit (a
    MagicMock mem0 backend, no live Qdrant), so the environment must not be an
    input to it.

    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


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
# Tests: _MEM0_MANAGED_METADATA_KEYS resolves to its single home
# ===========================================================================

class TestMem0ManagedKeysImported:
    """The script must IMPORT the mem0-owned key set, not define its own copy.

    Task 3195 gave the constant a single home next to ``Mem0Backend.update``
    (PRD D12 / decision doc ``plans/mem0-in-place-update-decision.md`` §6),
    and task 3088's in-place ``update_memory`` tool is a second consumer that
    imports rather than re-extracting it. The script keeps the module-local
    private spelling as an ALIAS, and an imported name is still a module
    attribute, so the ``_mod._MEM0_MANAGED_METADATA_KEYS`` references
    elsewhere in this file keep resolving.
    """

    def test_is_the_same_object_as_the_backend_constant(self):
        from fused_memory.backends import mem0_client

        assert _mod._MEM0_MANAGED_METADATA_KEYS is mem0_client.MEM0_MANAGED_METADATA_KEYS, (
            'the script must import the constant from its single home in '
            'backends/mem0_client.py, not re-define it (INV-5)'
        )


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
        positionally and the record's CUSTOM-provenance metadata subset (mem0-
        owned keys like 'data' stripped per _MEM0_MANAGED_METADATA_KEYS) as the
        metadata= kwarg, so provenance (src_project/kind/...) survives mem0's
        payload-overwriting _update_memory without also forwarding a stale
        copy of the content mem0 already owns. The 'skip:already_tagged' and
        'skip:no_src_project' fixture records must trigger no update at all --
        proven here by the single-call assertion covering the whole fixture,
        not just the taggable record."""
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
        expected_metadata = {
            k: v for k, v in taggable_record['metadata'].items()
            if k not in _mod._MEM0_MANAGED_METADATA_KEYS
        }
        assert call.kwargs.get('metadata') == expected_metadata
        assert 'data' not in call.kwargs['metadata'], (
            'mem0-owned data key (stale pre-tag content) must not be forwarded'
        )
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


# ===========================================================================
# Tests: scroll-truncation surface (non-aborting, unlike prune)
# ===========================================================================

class TestScanTruncationSurface:
    """Unlike prune_recon_cycle_summaries (which hard-aborts on a truncated
    scan because its deletes are irreversible), this script's in-place tag
    is idempotent and non-destructive, so a truncated scan is surfaced as a
    report-level flag rather than an abort (see plan design_decisions #5)."""

    def _make_memory(self, records: list[dict]) -> MagicMock:
        memory = MagicMock()
        mem0 = MagicMock()
        mem0.scroll_by_metadata = AsyncMock(return_value=records)
        mem0.count_by_metadata = AsyncMock(return_value=len(records))
        mem0.update = AsyncMock(return_value={'message': 'Memory updated successfully!'})
        memory.mem0 = mem0
        return memory

    def _args(self, apply=False, project_id='dark_factory', scan_limit=2) -> argparse.Namespace:
        return argparse.Namespace(apply=apply, project_id=project_id, scan_limit=scan_limit)

    def _known_map(self, pid='dark_factory') -> dict:
        return {pid: '/some/path'}

    @pytest.mark.asyncio
    async def test_scroll_at_scan_limit_flags_truncation_but_still_applies(self):
        """A scroll returning exactly scan_limit records is the ambiguous
        at-the-limit case (the true pool may be larger) -- the report must
        carry a truthy possibly_truncated flag naming the affected project,
        so a bounded scan is never silently mistaken for full coverage.
        Unlike prune, this must NOT abort: classification proceeds over what
        was scanned, and --apply still updates the taggable record found in
        that (possibly partial) scan. A subsequent re-run over the
        now-tagged data -- still pinned at scan_limit -- keeps the flag
        (coverage still isn't proven) but performs zero further updates,
        i.e. idempotent re-run remains safe even in the truncated case."""
        records = [_rehome_record('m1'), _rehome_record('m2', src_project=None)]
        memory = self._make_memory(records)
        args = self._args(apply=True, scan_limit=2)

        report = await _mod.run(args, memory=memory, known_projects_map=self._known_map())

        assert not report.get('aborted')
        assert report.get('possibly_truncated') is True
        assert 'dark_factory' in (report.get('truncated_projects') or [])
        assert report['projects']['dark_factory']['scanned'] == 2
        memory.mem0.update.assert_awaited_once()
        assert report['projects']['dark_factory']['tagged'] == 1

        rerun_records = [
            _rehome_record(
                'm1',
                data='[reify:task 948] Stage 2 should re-escalate task 948 after remediation.',
            ),
            _rehome_record('m2', src_project=None),
        ]
        memory2 = self._make_memory(rerun_records)
        report2 = await _mod.run(
            self._args(apply=True, scan_limit=2), memory=memory2,
            known_projects_map=self._known_map(),
        )

        assert report2.get('possibly_truncated') is True
        memory2.mem0.update.assert_not_awaited()


# ===========================================================================
# Tests: build_parser() defaults
# ===========================================================================

class TestArgparse:
    """build_parser() constructs the CLI parser used by main()."""

    def test_defaults(self):
        args = _mod.build_parser().parse_args([])
        assert args.apply is False
        assert args.project_id is None
        assert isinstance(args.scan_limit, int)
        assert args.scan_limit > 0


# ===========================================================================
# Tests: --apply store-mutation preflight
# ===========================================================================

class TestRunApplyStoreMutationPreflight:
    """``--apply`` refuses to START when this process cannot write mem0's store.

    Ported from ``test_sweep_toolcall_xml_leak.TestRunApplyStoreMutationPreflight``
    (task 3686), which is the in-repo precedent for this contract.

    ``apply_tags`` mutates through a bare ``memory.mem0.update(...)`` on the
    RAW backend -- deliberately bypassing MemoryService, per this script's own
    docstring -- so a grep for the usual mutation spellings does not find it.
    That call sits in a sequential loop behind a best-effort
    ``except Exception`` that logs a warning per record and keeps going, and
    ``StoreMutationUnavailable`` subclasses ``RuntimeError``, so a probe inside
    the helper would be swallowed into N warnings while ``run`` still returned
    a normal report. Only a run-wide probe ahead of the tagging bounds that.
    """

    def _records(self) -> list[dict]:
        """TWO taggable records, so one-probe-per-RUN is distinguishable from
        one-probe-per-record."""
        return [
            _rehome_record('m-taggable-1'),
            _rehome_record('m-taggable-2', src_entity='task 949'),
        ]

    def _make_memory(self, records: list[dict] | None = None) -> MagicMock:
        """Mirror of ``TestRun._make_memory``."""
        memory = MagicMock()
        mem0 = MagicMock()
        all_records = self._records() if records is None else records
        mem0.scroll_by_metadata = AsyncMock(return_value=all_records)
        mem0.count_by_metadata = AsyncMock(return_value=len(all_records))
        mem0.update = AsyncMock(return_value={'message': 'Memory updated successfully!'})
        memory.mem0 = mem0
        return memory

    def _args(self, apply=True, project_id='dark_factory', scan_limit=10000):
        """Mirror of ``TestRun._args``."""
        return argparse.Namespace(
            apply=apply, project_id=project_id, scan_limit=scan_limit,
        )

    def _known_map(self, pid='dark_factory') -> dict:
        return {pid: '/some/path'}

    @staticmethod
    def _deny(monkeypatch):
        """Rig the preflight to refuse, as it would inside an agent sandbox."""
        def _raise(*_args, **_kwargs):
            raise _mod.StoreMutationUnavailable('SENTINEL-store-unwritable')

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _raise)

    @staticmethod
    def _fail_closed_records(caplog) -> list:
        """The guard site's OWN diagnosis.

        ``main`` has no handler at all here, so the refusal exits as an
        uncaught traceback and this ERROR record is the ONLY place the operator
        is told what was refused and what to do instead. Pinned on the
        fail-closed marker and the remedy noun ONLY, so every other word of the
        message stays free to reword.

        Asserting on message CONTENT is deliberate, and is the narrow exception
        to the repo's don't-pin-guard-message-prose norm (task 3799): the record
        this test is about is defined BY its content -- mere record-existence
        would still pass if the whole diagnosis were replaced by "boom",
        precisely the regression this exists to catch. Verified non-vacuous:
        mutating the marker in the script turns this assertion red (task 4127
        amendment).
        """
        return [
            rec for rec in caplog.records
            if rec.name == 'tag_cgl_eta_rehome_scope'
            and rec.levelname == 'ERROR'
            and 'NOT started (fail-closed)' in rec.getMessage()
            and 'MCP server' in rec.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_apply_performs_zero_mutations_when_the_store_is_unwritable(
        self, monkeypatch
    ):
        """The whole point: refuse to start rather than half-complete.

        The mutation asserted here is the raw ``memory.mem0.update`` a pattern
        sweep provably misses -- it is the ONLY write this script performs.
        """
        self._deny(monkeypatch)
        memory = self._make_memory()

        with pytest.raises(
            _mod.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            await _mod.run(
                self._args(apply=True),
                memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.mem0.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_guard_sits_before_every_backend_read(self, monkeypatch):
        """It aborts without a single store round-trip.

        The probe sits below the unknown---project-id abort (kept probe-free by
        the test further down) but ABOVE the ``scroll_by_metadata`` loop, so a
        run that was never going to be allowed to mutate does not first pay for
        a full multi-project scroll of up to ``--scan-limit`` records per
        project. Nothing is TAGGED either, which is the point of the guard.
        """
        self._deny(monkeypatch)
        memory = self._make_memory()

        with pytest.raises(_mod.StoreMutationUnavailable):
            await _mod.run(
                self._args(apply=True),
                memory=memory,
                known_projects_map=self._known_map(),
            )

        memory.mem0.scroll_by_metadata.assert_not_awaited()
        memory.mem0.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch):
        """A read-only run mutates nothing, so it must not require the ability
        to mutate -- the tag report stays obtainable from anywhere, with the
        deny still installed."""
        self._deny(monkeypatch)
        memory = self._make_memory()

        report = await _mod.run(
            self._args(apply=False),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['dry_run'] is True
        assert report['totals']['taggable'] == 2
        assert report['totals']['tagged'] == 0
        memory.mem0.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_is_unchanged_when_the_preflight_passes(self, monkeypatch):
        """Happy path: a writable environment tags exactly as before."""
        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)
        memory = self._make_memory()

        report = await _mod.run(
            self._args(apply=True),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['dry_run'] is False
        assert report['totals']['tagged'] == 2
        assert memory.mem0.update.await_count == 2

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_being_gated(self, monkeypatch):
        """The refusal has to be attributable in a log, so the operation string
        identifies this script and its mutating mode -- and with TWO taggable
        records in the scroll it is still probed once for the RUN."""
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        memory = self._make_memory()

        report = await _mod.run(
            self._args(apply=True),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['totals']['tagged'] == 2
        assert len(calls) == 1, 'probed ONCE per run, not once per record'
        assert 'tag_cgl_eta_rehome_scope' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']

    @pytest.mark.asyncio
    async def test_the_unknown_project_id_abort_never_probes(self, monkeypatch):
        """A run that aborts before reaching the tagging must not be gated on
        write capability -- it mutates nothing.

        This is the test that pins the probe BELOW this abort rather than at
        the very top of ``run``: hoisting it further would turn a clear
        "unknown --project-id" abort into an unrelated store-capability
        refusal for anyone diagnosing a typo'd flag from a sandbox. Together
        with ``test_the_guard_sits_before_every_backend_read`` above -- which
        pins it ABOVE the scroll -- the two bracket the placement from both
        sides.
        """
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        memory = self._make_memory()

        report = await _mod.run(
            self._args(apply=True, project_id='no-such-project'),
            memory=memory,
            known_projects_map=self._known_map(),
        )

        assert report['aborted'] is True
        assert calls == [], 'an abort that mutates nothing must not probe'
        memory.mem0.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_refusal_is_loud(self, monkeypatch, caplog):
        """The guard site logs its own fail-closed diagnosis before raising.

        Nothing downstream will: this script's ``main`` has no blanket handler,
        so without this record the operator sees a bare traceback naming an
        exception class and no remedy.
        """
        self._deny(monkeypatch)
        memory = self._make_memory()

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(_mod.StoreMutationUnavailable),
        ):
            await _mod.run(
                self._args(apply=True),
                memory=memory,
                known_projects_map=self._known_map(),
            )

        assert self._fail_closed_records(caplog), (
            'nothing else explains this traceback -- the guard site must log '
            'the fail-closed diagnosis before raising; got: '
            f'{[rec.getMessage() for rec in caplog.records]}'
        )
        memory.mem0.update.assert_not_awaited()
