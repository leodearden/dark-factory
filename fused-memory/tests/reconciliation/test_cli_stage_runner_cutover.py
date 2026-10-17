"""Tests for PRD γ cutover: cli_stage_runner schema migration and output_schema passthrough."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.reconciliation.cli_stage_runner import (
    FINDING_ITEM_SCHEMA,
    RAW_TEXT_FALLBACK_STAT,
    STAGE3_REPORT_SCHEMA,
    STAGE_REPORT_SCHEMA,
    _extract_report,
)
from fused_memory.utils.clip_disclosure import TRUNCATION_FACTS_KEY, elision_marker


class TestFindingItemSchemaShape:
    """FINDING_ITEM_SCHEMA must carry four typed citation arrays per PRD §9.3."""

    def test_has_four_citation_arrays(self):
        """cited_entities, cited_edges, cited_tasks, cited_memories all present."""
        props = FINDING_ITEM_SCHEMA['properties']
        for key in ('cited_entities', 'cited_edges', 'cited_tasks', 'cited_memories'):
            assert key in props, f'Expected {key!r} in FINDING_ITEM_SCHEMA.properties'

    def test_no_affected_ids(self):
        """affected_ids must be removed from FINDING_ITEM_SCHEMA."""
        props = FINDING_ITEM_SCHEMA['properties']
        assert 'affected_ids' not in props, 'affected_ids must be retired from FINDING_ITEM_SCHEMA'

    def test_cited_entities_item_shape(self):
        """cited_entities items carry entity_uuid and canonical_name."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_entities']['items']
        props = item_schema.get('properties', {})
        assert 'entity_uuid' in props, 'cited_entities items must have entity_uuid'
        assert 'canonical_name' in props, 'cited_entities items must have canonical_name'

    def test_cited_edges_item_shape(self):
        """cited_edges items carry edge_uuid and fact_text_snapshot."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_edges']['items']
        props = item_schema.get('properties', {})
        assert 'edge_uuid' in props, 'cited_edges items must have edge_uuid'
        assert 'fact_text_snapshot' in props, 'cited_edges items must have fact_text_snapshot'

    def test_cited_tasks_item_shape(self):
        """cited_tasks items carry project_id, task_id, and title."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_tasks']['items']
        props = item_schema.get('properties', {})
        assert 'project_id' in props, 'cited_tasks items must have project_id'
        assert 'task_id' in props, 'cited_tasks items must have task_id'
        assert 'title' in props, 'cited_tasks items must have title'

    def test_cited_memories_item_shape(self):
        """cited_memories items carry memory_id, store, and metadata_fingerprint."""
        item_schema = FINDING_ITEM_SCHEMA['properties']['cited_memories']['items']
        props = item_schema.get('properties', {})
        assert 'memory_id' in props, 'cited_memories items must have memory_id'
        assert 'store' in props, 'cited_memories items must have store'
        assert 'metadata_fingerprint' in props, 'cited_memories items must have metadata_fingerprint'

    def test_description_and_severity_still_required(self):
        """description and severity remain required fields."""
        required = FINDING_ITEM_SCHEMA.get('required', [])
        assert 'description' in required, 'description must stay required'
        assert 'severity' in required, 'severity must stay required'

    def test_optional_top_level_fields_present(self):
        """task_id, flag_type, and actionable are still defined as top-level properties."""
        props = FINDING_ITEM_SCHEMA['properties']
        for key in ('task_id', 'flag_type', 'actionable'):
            assert key in props, f'Expected {key!r} still in FINDING_ITEM_SCHEMA.properties'


class TestStageReportSchemaShape:
    """STAGE_REPORT_SCHEMA must expose flagged_items, stats, summary with no affected_ids leak."""

    def test_has_required_top_level_keys(self):
        """flagged_items, stats, and summary are all present."""
        props = STAGE_REPORT_SCHEMA['properties']
        for key in ('flagged_items', 'stats', 'summary'):
            assert key in props, f'Expected {key!r} in STAGE_REPORT_SCHEMA.properties'

    def test_no_affected_ids_at_top_level(self):
        """affected_ids must not appear at the top level of STAGE_REPORT_SCHEMA."""
        props = STAGE_REPORT_SCHEMA['properties']
        assert 'affected_ids' not in props, 'affected_ids must not be at STAGE_REPORT_SCHEMA top level'


class TestStage3ReportSchemaShape:
    """STAGE3_REPORT_SCHEMA must mirror STAGE_REPORT_SCHEMA with structured finding items."""

    def test_has_required_top_level_keys(self):
        """flagged_items, stats, and summary are all present."""
        props = STAGE3_REPORT_SCHEMA['properties']
        for key in ('flagged_items', 'stats', 'summary'):
            assert key in props, f'Expected {key!r} in STAGE3_REPORT_SCHEMA.properties'

    def test_no_affected_ids_at_top_level(self):
        """affected_ids must not appear at the top level of STAGE3_REPORT_SCHEMA."""
        props = STAGE3_REPORT_SCHEMA['properties']
        assert 'affected_ids' not in props, 'affected_ids must not be at STAGE3_REPORT_SCHEMA top level'

    def test_flagged_items_uses_finding_item_schema(self):
        """STAGE3_REPORT_SCHEMA.flagged_items.items should be FINDING_ITEM_SCHEMA."""
        items_schema = STAGE3_REPORT_SCHEMA['properties']['flagged_items']['items']
        # The items schema must have the four citation arrays (same as FINDING_ITEM_SCHEMA)
        props = items_schema.get('properties', {})
        for key in ('cited_entities', 'cited_edges', 'cited_tasks', 'cited_memories'):
            assert key in props, f'STAGE3_REPORT_SCHEMA.flagged_items.items missing {key!r}'


class TestOutputSchemaNonePassthrough:
    """run_stage_via_cli must pass output_schema=None straight through (PRD γ step-14).

    When recon_report_state is active, the harness passes output_schema=None to
    run_stage_via_cli so the CLI agent can terminate with a short text turn after
    recon_report.complete instead of re-emitting a JSON report.

    RED: currently run_stage_via_cli replaces None with STAGE_REPORT_SCHEMA
    (line 255: output_schema=output_schema if output_schema is not None else STAGE_REPORT_SCHEMA).
    Step-14 changes this so None is passed straight through.
    """

    @pytest.mark.asyncio
    async def test_none_schema_reaches_invoke_with_cap_retry(self):
        """Passing output_schema=None must send output_schema=None to invoke_with_cap_retry.

        RED: Currently, None is replaced by STAGE_REPORT_SCHEMA before invoke.
        """
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli

        captured_kwargs: dict = {}

        async def _capture_invoke(**kwargs):
            captured_kwargs.update(kwargs)
            # Return a minimal AgentResult
            result = MagicMock()
            result.structured_output = None
            result.output = None
            result.turns = 0
            result.cost_usd = 0.0
            result.success = True
            return result

        config = ReconciliationConfig()
        mcp_config = {'mcpServers': {}}

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=_capture_invoke,
        ):
            await run_stage_via_cli(
                system_prompt='test',
                payload='test payload',
                disallowed_tools=[],
                config=config,
                mcp_config=mcp_config,
                output_schema=None,
            )

        assert captured_kwargs.get('output_schema') is None, (
            'output_schema=None must be passed straight through to invoke_with_cap_retry, '
            'not replaced by STAGE_REPORT_SCHEMA'
        )

    def test_extract_report_returns_empty_dict_when_no_output(self):
        """_extract_report must return {} (not a placeholder) when structured_output and output are empty.

        RED: Currently _extract_report returns {'summary': 'No output from agent'} as a placeholder.
        After step-14, it returns {} so BaseStage.run gets an empty dict that triggers the
        recon_report fallback path.
        """
        from shared.cli_invoke import AgentResult

        result = AgentResult(
            output='',
            structured_output=None,
            turns=0,
            cost_usd=0.0,
            success=True,
        )
        report = _extract_report(result)
        assert report == {}, (
            f'_extract_report must return {{}} when output is empty, got {report!r}'
        )


def _agent_result(output: str, structured_output: object = None):
    from shared.cli_invoke import AgentResult

    return AgentResult(output=output, structured_output=structured_output, turns=0, cost_usd=0.0, success=True)


class TestExtractReportRawTextFallback:
    """A prose-only reply is reported as such, and a clipped summary says how much it lost.

    Scoped to _extract_report: on the recon_report path BaseStage.run replaces this dict by design.
    """

    def test_long_prose_is_elided_and_its_clip_recorded_in_stats(self):
        prose = 'The stage found nothing worth flagging. ' * 1000

        report = _extract_report(_agent_result(prose))

        [fact] = report['stats'][TRUNCATION_FACTS_KEY]
        cap = fact['cap']
        assert fact == {'field': 'summary', 'cap': cap, 'elided': len(prose) - cap}
        assert report['summary'] == prose[:cap] + elision_marker(len(prose) - cap)
        assert report['flagged_items'] == []
        assert report['stats'][RAW_TEXT_FALLBACK_STAT] is True

    def test_short_prose_is_unmarked_but_still_flagged_as_a_fallback(self):
        report = _extract_report(_agent_result('All consistent; no findings.'))

        assert report == {
            'summary': 'All consistent; no findings.',
            'flagged_items': [],
            'stats': {RAW_TEXT_FALLBACK_STAT: True},
        }

    def test_parsed_json_output_carries_no_fallback_stat(self):
        report = _extract_report(_agent_result('{"summary": "ok", "flagged_items": []}'))

        assert report == {'summary': 'ok', 'flagged_items': []}

    def test_structured_output_carries_no_fallback_stat(self):
        structured = {'summary': 'ok', 'flagged_items': [], 'stats': {'n': 1}}

        report = _extract_report(_agent_result('ignored prose', structured_output=structured))

        assert report == structured


def _make_capture_invoke(captured_kwargs: dict):
    """Build an async stand-in for invoke_with_cap_retry that records its kwargs."""

    async def _capture_invoke(**kwargs):
        captured_kwargs.update(kwargs)
        result = MagicMock()
        result.structured_output = None
        result.output = None
        result.turns = 0
        result.cost_usd = 0.0
        result.success = True
        return result

    return _capture_invoke


class TestSessionAndConfigDirForwarding:
    """run_stage_via_cli must forward session_id= and config_dir= into
    invoke_with_cap_retry (task 2744), defaulting both to None when omitted so
    existing cutover/sandbox call sites keep today's behavior."""

    @pytest.mark.asyncio
    async def test_session_id_and_config_dir_forwarded(self, tmp_path):
        from shared.config_dir import TaskConfigDir

        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import (
            recon_config_base_dir,
            run_stage_via_cli,
        )

        captured_kwargs: dict = {}
        config = ReconciliationConfig()
        mcp_config = {'mcpServers': {}}
        cfg_dir = TaskConfigDir(task_id='run-xyz', base_dir=recon_config_base_dir(tmp_path))

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=_make_capture_invoke(captured_kwargs),
        ):
            await run_stage_via_cli(
                system_prompt='test',
                payload='test payload',
                disallowed_tools=[],
                config=config,
                mcp_config=mcp_config,
                session_id='sess-abc',
                config_dir=cfg_dir,
            )

        assert captured_kwargs.get('session_id') == 'sess-abc'
        assert captured_kwargs.get('config_dir') is cfg_dir

    @pytest.mark.asyncio
    async def test_session_id_and_config_dir_default_none(self):
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli

        captured_kwargs: dict = {}
        config = ReconciliationConfig()
        mcp_config = {'mcpServers': {}}

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=_make_capture_invoke(captured_kwargs),
        ):
            await run_stage_via_cli(
                system_prompt='test',
                payload='test payload',
                disallowed_tools=[],
                config=config,
                mcp_config=mcp_config,
            )

        assert 'session_id' in captured_kwargs
        assert captured_kwargs['session_id'] is None
        assert 'config_dir' in captured_kwargs
        assert captured_kwargs['config_dir'] is None


class TestReconConfigDirHelpers:
    """recon_config_base_dir / gc_run_config_dir path determinism + GC (task 2744)."""

    def test_recon_config_base_dir(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import recon_config_base_dir

        assert recon_config_base_dir(tmp_path) == tmp_path / 'recon-config'

    def test_config_dir_path_identity(self, tmp_path):
        """Pins the creator↔GC coupling: TaskConfigDir builds exactly the path
        gc_run_config_dir rmtrees (config_dir.py's claude-config-{id} naming)."""
        from shared.config_dir import TaskConfigDir

        from fused_memory.reconciliation.cli_stage_runner import recon_config_base_dir

        run_id = 'run-abc123'
        base = recon_config_base_dir(tmp_path)
        cfg = TaskConfigDir(task_id=run_id, base_dir=base)
        assert cfg.path == base / f'claude-config-{run_id}'

    def test_gc_removes_existing_dir(self, tmp_path):
        from shared.config_dir import TaskConfigDir

        from fused_memory.reconciliation.cli_stage_runner import (
            gc_run_config_dir,
            recon_config_base_dir,
        )

        run_id = 'run-to-gc'
        cfg = TaskConfigDir(task_id=run_id, base_dir=recon_config_base_dir(tmp_path))
        assert cfg.path.exists()

        gc_run_config_dir(tmp_path, run_id, archive_root=None)
        assert not cfg.path.exists()

    def test_gc_is_noop_when_absent(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        # No dir created — GC must be a silent no-op (ignore_errors), never raise.
        gc_run_config_dir(tmp_path, 'never-created', archive_root=None)

    def test_run_config_dir_path_matches_task_config_dir_path(self, tmp_path):
        """The pure derivation names exactly the dir ``TaskConfigDir`` builds."""
        from shared.config_dir import TaskConfigDir

        from fused_memory.reconciliation.cli_stage_runner import (
            recon_config_base_dir,
            run_config_dir_path,
        )

        run_id = 'run-abc123'
        cfg = TaskConfigDir(task_id=run_id, base_dir=recon_config_base_dir(tmp_path))
        assert run_config_dir_path(tmp_path, run_id) == cfg.path

    def test_run_config_dir_path_does_not_create_anything(self, tmp_path):
        """Deriving the path is a read: a probe must not resurrect the dir."""
        from fused_memory.reconciliation.cli_stage_runner import (
            recon_config_base_dir,
            run_config_dir_path,
        )

        path = run_config_dir_path(tmp_path, 'never-created')

        assert not path.exists()
        assert not recon_config_base_dir(tmp_path).exists()
        assert list(tmp_path.iterdir()) == []

    def test_recon_config_base_dir_absolutizes_a_relative_data_dir(
        self, tmp_path, monkeypatch
    ):
        """A relative ``data_dir`` must still yield an ABSOLUTE config-dir root.

        The root returned here becomes ``TaskConfigDir.path``, which is BOTH the
        ``CLAUDE_CONFIG_DIR`` handed to the CLI child and the path the parent
        verifies for sandbox containment. A relative string is resolved against
        the child's cwd in the first role and the parent's in the second:
        verified path A, written path B — the containment check says PASS while
        the kernel denies every transcript write.

        The relative case is the standalone/systemd production configuration, not
        a hypothetical. Full mechanism and deployment story: ``fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py::recon_config_base_dir``.
        """
        from fused_memory.reconciliation.cli_stage_runner import recon_config_base_dir

        monkeypatch.chdir(tmp_path)
        base = recon_config_base_dir(Path('data/reconciliation'))

        assert base.is_absolute(), (
            f'Config-dir root must be absolute so the parent and the CLI child '
            f'name the same directory; got {base!r}'
        )
        assert base == tmp_path / 'data' / 'reconciliation' / 'recon-config', (
            f'A relative data_dir must anchor at the process cwd (the same '
            f'implicit anchor ReconciliationJournal.initialize already uses for '
            f'reconciliation.db); got {base!r}'
        )

    def test_recon_config_base_dir_is_cwd_independent_for_an_absolute_data_dir(
        self, tmp_path, monkeypatch
    ):
        """An already-absolute ``data_dir`` is untouched, from any cwd.

        The pre-existing ``test_recon_config_base_dir`` leaf pins the VALUE; this
        pins that the value does not depend on where the process happens to be
        standing. Together they hold the "absolutize a relative input, leave an
        absolute one byte-identical" contract — the reason the fix uses
        ``Path.cwd() / base`` rather than ``.resolve()``, which would additionally
        collapse symlink components of an already-absolute deployment path.
        """
        from fused_memory.reconciliation.cli_stage_runner import recon_config_base_dir

        here = tmp_path / 'here'
        there = tmp_path / 'there'
        here.mkdir()
        there.mkdir()

        monkeypatch.chdir(here)
        from_here = recon_config_base_dir(tmp_path)
        monkeypatch.chdir(there)
        from_there = recon_config_base_dir(tmp_path)

        assert from_here == from_there == tmp_path / 'recon-config', (
            f'An absolute data_dir must yield the same root from any cwd; got '
            f'{from_here!r} and {from_there!r}'
        )

    def test_gc_run_config_dir_removes_the_cwd_anchored_dir_for_a_relative_data_dir(
        self, tmp_path, monkeypatch
    ):
        """The GC must target the same absolute path the creator built.

        ``gc_run_config_dir`` derives its rmtree target from
        ``recon_config_base_dir``, so absolutizing there fixes the GC for free —
        this leaf pins that it actually did, rather than leaving the GC pointed
        at a cwd-relative path that silently rmtrees nothing (``ignore_errors=True``
        makes a wrong target indistinguishable from a clean no-op).
        """
        from fused_memory.reconciliation.cli_stage_runner import (
            gc_run_config_dir,
            recon_config_base_dir,
        )

        doomed = tmp_path / 'data' / 'reconciliation' / 'recon-config' / 'claude-config-r1'
        doomed.mkdir(parents=True)

        monkeypatch.chdir(tmp_path)
        gc_run_config_dir(Path('data/reconciliation'), 'r1', archive_root=None)

        assert not doomed.exists(), (
            f'GC must remove the cwd-anchored dir the creator built; {doomed} '
            f'still exists, so the GC and the creator disagree on the path'
        )
        # Control: the root the GC derives is the absolute one, not a relative
        # sibling accidentally created under the cwd by the rmtree call.
        assert recon_config_base_dir(Path('data/reconciliation')).is_absolute()

    @pytest.mark.parametrize('mock_cls', [MagicMock, AsyncMock])
    def test_a_mock_data_dir_touches_no_filesystem(
        self, tmp_path, monkeypatch, mock_cls
    ):
        """A mock journal's data_dir must be left strictly alone — not coerced, not called.

        Many stage tests in this suite drive BaseStage with an ``AsyncMock``
        journal (``test_base_stage_cutover.py``), so ``self.journal.data_dir`` is a
        mock rather than a Path. Two distinct hazards, which is why both mock
        classes are parametrized rather than only the convenient one:

        - COERCION. Both classes implement ``__fspath__``, so ``Path(mock)``
          silently becomes the RELATIVE path ``AsyncMock/mock.data_dir/<id>``,
          which absolutization would anchor at the cwd and
          ``TaskConfigDir.__init__`` would really ``mkdir(parents=True)`` —
          littering the repo with junk dirs on every suite run (and sweeping them
          into any subsequent ``git add``).
        - INVOCATION. Calling ``is_absolute()`` on an ``AsyncMock`` returns a
          COROUTINE, not a bool. It is truthy, so the branch is skipped and no
          directory appears — the filesystem assertion below passes either way —
          but the coroutine is never awaited and CPython emits ``RuntimeWarning:
          coroutine 'AsyncMockMixin._execute_mock_call' was never awaited``, which
          ``orchestrator/pyproject.toml`` already promotes to an error in its
          ``filterwarnings``. That is why the ``assert_not_called`` leaf below is
          the load-bearing one for the AsyncMock parameter: a filesystem-only
          assertion cannot see this hazard at all.

        ``recon_config_base_dir`` guards on ``isinstance(base, PurePath)``, so a
        non-Path duck type is neither wrapped nor called; the mock's own
        ``__truediv__`` returns another mock and nothing reaches the filesystem.
        Composes the chain exactly as ``stages/base.py::BaseStage.run`` does.
        """
        from shared.config_dir import TaskConfigDir

        from fused_memory.reconciliation.cli_stage_runner import recon_config_base_dir

        journal = mock_cls()
        monkeypatch.chdir(tmp_path)
        TaskConfigDir(
            task_id='run-x', base_dir=recon_config_base_dir(journal.data_dir),
        )

        assert list(tmp_path.iterdir()) == [], (
            f'A mock data_dir must not be coerced into a real path; the chain '
            f'created {[p.name for p in tmp_path.iterdir()]!r} under the cwd'
        )
        journal.data_dir.is_absolute.assert_not_called()

    def test_production_chain_yields_an_absolute_config_dir(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: the real chain survives the fail-closed guard tightening.

        Composes exactly what ``stages/base.py::BaseStage.run`` does —
        ``TaskConfigDir(task_id=run_id, base_dir=recon_config_base_dir(journal.data_dir))``
        — with the PARENT standing in one directory and the CLI child spawned in
        ANOTHER (``run_stage_via_cli`` passes ``cwd=config.explore_codebase_root``
        to both ``resolve_recon_sandbox_wrap`` and ``_run_subprocess``). That
        divergence is the whole defect: with a relative ``data_dir`` the two
        components resolve one string against two cwds.

        This is the leaf that proves the guard tightenings do NOT fail-close the
        real production chain: the config dir must be absolute, must exist where
        the parent created it, and must actually appear as a ``--writable`` grant
        in the wrapped argv the child will run.
        """
        from unittest.mock import patch as _patch

        from shared.config_dir import TaskConfigDir

        from fused_memory.reconciliation.cli_stage_runner import recon_config_base_dir
        from fused_memory.reconciliation.sandbox_guard import resolve_recon_sandbox_wrap

        parent_cwd = tmp_path / 'parent'
        child_root = tmp_path / 'child'
        parent_cwd.mkdir()
        child_root.mkdir()

        monkeypatch.chdir(parent_cwd)
        cfg = TaskConfigDir(
            task_id='run-x',
            base_dir=recon_config_base_dir(Path('data/reconciliation')),
        )

        assert cfg.path.is_absolute(), (
            f'The per-run CLAUDE_CONFIG_DIR must be absolute; got {cfg.path!r}'
        )
        expected = (
            parent_cwd / 'data' / 'reconciliation' / 'recon-config' / 'claude-config-run-x'
        )
        assert cfg.path == expected, f'Expected {expected!r}; got {cfg.path!r}'
        assert cfg.path.is_dir(), (
            f'{cfg.path} must exist where the parent created it — the CLI child '
            f'writes its session JSONL there'
        )

        with _patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            wrap = resolve_recon_sandbox_wrap(
                child_root, [str(cfg.path)], config_dir=cfg.path,
            )
            wrapped = wrap(['claude', '--print'])

        writable_vals = [
            wrapped[i + 1] for i, tok in enumerate(wrapped) if tok == '--writable'
        ]
        assert str(cfg.path) in writable_vals, (
            f'The absolute config dir must be granted in the argv the child runs; '
            f'got {writable_vals!r}'
        )


class TestReconTranscriptArchiveKey:
    """The archive key a recon run files its transcripts under (task 3972)."""

    RUN_ID = 'd6782083-9319-48fa-b72c-1730fd9b1a35'

    def test_key_is_run_id_namespaced(self):
        from fused_memory.reconciliation.cli_stage_runner import (
            recon_transcript_archive_key,
        )

        assert (
            recon_transcript_archive_key(self.RUN_ID)
            == 'recon-d6782083-9319-48fa-b72c-1730fd9b1a35'
        )

    def test_key_is_a_single_path_segment(self):
        """``scripts/legibility/inventory.py::_archive_enc`` reads the
        archive-root-relative ``parts[1]`` as ``<enc>``; a key with an extra
        segment would land the archives on disk but make them unenumerable."""
        import os
        from pathlib import PurePosixPath

        from fused_memory.reconciliation.cli_stage_runner import (
            recon_transcript_archive_key,
        )

        key = recon_transcript_archive_key(self.RUN_ID)

        assert len(PurePosixPath(key).parts) == 1
        assert '/' not in key
        assert os.sep not in key

    def test_key_does_not_collide_with_orchestrator_task_ids(self):
        """Orchestrator task-id dirs in the shared archive root are bare numbers."""
        from fused_memory.reconciliation.cli_stage_runner import (
            recon_transcript_archive_key,
        )

        assert not recon_transcript_archive_key('4578').isdigit()
        assert not recon_transcript_archive_key(self.RUN_ID).isdigit()


class TestReconArchiveRoot:
    """recon_archive_root: the one composition home for the recon archive root."""

    LOGGER = 'fused_memory.reconciliation.cli_stage_runner'

    def test_composes_against_explore_codebase_root(self, tmp_path):
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import recon_archive_root

        cfg = ReconciliationConfig(explore_codebase_root=str(tmp_path))

        assert recon_archive_root(cfg) == (
            tmp_path / 'data' / 'orchestrator' / 'agent-transcripts'
        )

    def test_disabled_returns_none_and_creates_nothing(self, tmp_path):
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import recon_archive_root

        cfg = ReconciliationConfig(
            explore_codebase_root=str(tmp_path), transcript_archive_enabled=False,
        )

        assert recon_archive_root(cfg) is None
        assert list(tmp_path.iterdir()) == []

    def test_malformed_config_returns_none_and_warns(self, tmp_path, caplog):
        """A config regression disables archival loudly rather than raising on
        the recon dispatch path. The plain assignment is the shape an in-place
        ``apply_reload`` mutation lands in: no validate_assignment."""
        import logging

        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import recon_archive_root

        cfg = ReconciliationConfig(explore_codebase_root=str(tmp_path))
        cfg.transcript_archive_root = None  # type: ignore[assignment]

        with caplog.at_level(logging.WARNING, logger=self.LOGGER):
            result = recon_archive_root(cfg)

        assert result is None
        warnings = [
            r for r in caplog.records
            if r.name == self.LOGGER and r.levelno == logging.WARNING
        ]
        assert len(warnings) == 1, [r.getMessage() for r in warnings]

    def test_returns_a_path_not_a_string(self, tmp_path):
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import recon_archive_root

        cfg = ReconciliationConfig(explore_codebase_root=str(tmp_path))

        assert isinstance(recon_archive_root(cfg), Path)

    def _announced(self, caplog, cfg):
        import logging

        from fused_memory.reconciliation.cli_stage_runner import (
            announce_recon_archive_root,
        )

        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            announce_recon_archive_root(cfg)
        return [
            (r.levelno, r.getMessage())
            for r in caplog.records if r.name == self.LOGGER
        ]

    def test_announce_logs_an_absolute_root_at_info(self, tmp_path, caplog):
        import logging

        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig(explore_codebase_root=str(tmp_path))
        root = tmp_path / 'data' / 'orchestrator' / 'agent-transcripts'

        assert self._announced(caplog, cfg) == [
            (logging.INFO, f'recon transcript archive root: {root}'),
        ]
        assert list(tmp_path.iterdir()) == []

    def test_announce_warns_with_the_cwd_resolution_of_a_relative_root(
        self, tmp_path, monkeypatch, caplog
    ):
        """The shipped ``${PROJECT_ROOT:.}`` with PROJECT_ROOT unset."""
        import logging

        from fused_memory.config.schema import ReconciliationConfig

        monkeypatch.chdir(tmp_path)
        cfg = ReconciliationConfig(explore_codebase_root='.')

        [(level, message)] = self._announced(caplog, cfg)

        assert level == logging.WARNING
        assert str(tmp_path / 'data' / 'orchestrator' / 'agent-transcripts') in message
        assert list(tmp_path.iterdir()) == []

    def test_announce_reports_disabled(self, tmp_path, caplog):
        import logging

        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig(
            explore_codebase_root=str(tmp_path), transcript_archive_enabled=False,
        )

        assert self._announced(caplog, cfg) == [
            (logging.INFO, 'recon transcript archival: disabled'),
        ]


class TestGcArchivesBeforeDelete:
    """gc_run_config_dir moves a run's transcripts into the archive before it
    destroys the per-run config dir (task 3972)."""

    RUN_ID = 'd6782083-9319-48fa-b72c-1730fd9b1a35'
    SID = '0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0'
    ENC = '-home-leo-src-dark-factory'
    TRANSCRIPT = b'{"type":"user","message":"hello"}\n{"type":"assistant"}\n'

    def _plant(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import run_config_dir_path

        data_dir = tmp_path / 'data'
        config_dir = run_config_dir_path(data_dir, self.RUN_ID)
        session_dir = config_dir / 'projects' / self.ENC
        session_dir.mkdir(parents=True)
        transcript = session_dir / f'{self.SID}.jsonl'
        transcript.write_bytes(self.TRANSCRIPT)
        (config_dir / '.credentials.json').write_text('{"token": "secret"}')
        return data_dir, config_dir, transcript

    def _archived(self, archive_root):
        return archive_root / f'recon-{self.RUN_ID}' / self.ENC / f'{self.SID}.jsonl'

    def test_transcript_is_archived_then_the_dir_is_destroyed(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        data_dir, config_dir, _ = self._plant(tmp_path)
        archive_root = tmp_path / 'archive'

        gc_run_config_dir(data_dir, self.RUN_ID, archive_root=archive_root)

        assert self._archived(archive_root).read_bytes() == self.TRANSCRIPT
        assert not config_dir.exists()

    def test_archive_layout_is_key_enc_sid(self, tmp_path):
        """Exactly ``<key>/<enc>/<sid>.jsonl``: an extra key segment would leave
        the archive unenumerable by ``scripts/legibility/inventory.py::_archive_enc``."""
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        data_dir, _, _ = self._plant(tmp_path)
        archive_root = tmp_path / 'archive'

        gc_run_config_dir(data_dir, self.RUN_ID, archive_root=archive_root)

        [archived] = list(archive_root.rglob('*.jsonl'))
        assert archived.relative_to(archive_root).parts == (
            f'recon-{self.RUN_ID}', self.ENC, f'{self.SID}.jsonl',
        )

    def test_only_jsonl_under_projects_is_archived(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        data_dir, _, _ = self._plant(tmp_path)
        archive_root = tmp_path / 'archive'

        gc_run_config_dir(data_dir, self.RUN_ID, archive_root=archive_root)

        assert list(archive_root.rglob('.credentials.json')) == []

    def test_no_archive_root_keeps_todays_rmtree_behaviour(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        data_dir, config_dir, _ = self._plant(tmp_path)
        archive_root = tmp_path / 'archive'

        gc_run_config_dir(data_dir, self.RUN_ID, archive_root=None)

        assert not config_dir.exists()
        assert not archive_root.exists()

    def test_a_call_that_omits_archive_root_destroys_nothing(self, tmp_path):
        """Omitting or positionally passing archive_root is refused before any rmtree."""
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        data_dir, config_dir, transcript = self._plant(tmp_path)

        with pytest.raises(TypeError):
            gc_run_config_dir(data_dir, self.RUN_ID)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            gc_run_config_dir(data_dir, self.RUN_ID, None)  # pyright: ignore[reportCallIssue]

        assert transcript.read_bytes() == self.TRANSCRIPT
        assert config_dir.is_dir()

    def test_is_still_a_noop_when_the_dir_is_absent(self, tmp_path):
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        archive_root = tmp_path / 'archive'

        gc_run_config_dir(tmp_path / 'data', 'never-created', archive_root=archive_root)

        assert not archive_root.exists()

    def test_never_raises_on_an_unwritable_archive_root(self, tmp_path):
        """The un-archivable transcript is HELD while the credential is still
        purged, and gc stays total so it cannot mask a run's terminal outcome."""
        import os

        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        if os.geteuid() == 0:
            pytest.skip('chmod does not bind root')

        data_dir, config_dir, transcript = self._plant(tmp_path)
        archive_root = tmp_path / 'archive'
        archive_root.mkdir()
        archive_root.chmod(0o500)
        try:
            gc_run_config_dir(data_dir, self.RUN_ID, archive_root=archive_root)
        finally:
            archive_root.chmod(0o700)

        assert transcript.read_bytes() == self.TRANSCRIPT
        assert not (config_dir / '.credentials.json').exists()

    @pytest.mark.parametrize('mock_cls', [MagicMock, AsyncMock])
    def test_a_mock_data_dir_with_archival_touches_no_filesystem(
        self, tmp_path, monkeypatch, mock_cls
    ):
        """Archival passes the mock through ``Path(...)``, whose ``__fspath__``
        coercion yields a relative path; nothing may land under the cwd."""
        from fused_memory.reconciliation.cli_stage_runner import gc_run_config_dir

        monkeypatch.chdir(tmp_path)

        gc_run_config_dir(
            mock_cls().data_dir, 'run-x', archive_root=tmp_path / 'archive',
        )

        assert list(tmp_path.iterdir()) == [], (
            f'archival with a mock data_dir created '
            f'{[p.name for p in tmp_path.iterdir()]!r} under the cwd'
        )


class TestResumeParamForwarding:
    """run_stage_via_cli forwards resume_session_id + resume_delivers_prompt into
    invoke_with_cap_retry (task 2717 σ resume seam); both default to None/False so
    every existing (non-resume) call site stays byte-identical."""

    @pytest.mark.asyncio
    async def test_resume_params_forwarded(self):
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli

        captured_kwargs: dict = {}
        config = ReconciliationConfig()
        mcp_config = {'mcpServers': {}}

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=_make_capture_invoke(captured_kwargs),
        ):
            await run_stage_via_cli(
                system_prompt='test',
                payload='recovery payload',
                disallowed_tools=[],
                config=config,
                mcp_config=mcp_config,
                resume_session_id='sess-resume',
                resume_delivers_prompt=True,
            )

        assert captured_kwargs.get('resume_session_id') == 'sess-resume'
        assert captured_kwargs.get('resume_delivers_prompt') is True

    @pytest.mark.asyncio
    async def test_resume_params_default_off(self):
        from fused_memory.config.schema import ReconciliationConfig
        from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli

        captured_kwargs: dict = {}
        config = ReconciliationConfig()
        mcp_config = {'mcpServers': {}}

        with patch(
            'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
            new=_make_capture_invoke(captured_kwargs),
        ):
            await run_stage_via_cli(
                system_prompt='test',
                payload='test payload',
                disallowed_tools=[],
                config=config,
                mcp_config=mcp_config,
            )

        # Present-but-inert by default (mirrors session_id/config_dir passthrough).
        assert captured_kwargs.get('resume_session_id') is None
        assert captured_kwargs.get('resume_delivers_prompt') is False
