"""Tests for scripts/clear_malformed_empty_memory.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors the pattern in test_consolidate_namespace_families.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'clear_malformed_empty_memory.py'


def _load_module() -> types.ModuleType:
    """Load clear_malformed_empty_memory.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'clear_malformed_empty_memory'
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


# ===========================================================================
# Helpers
# ===========================================================================

def _make_qdrant_mock(records: list | None = None) -> AsyncMock:
    """AsyncMock stand-in for an AsyncQdrantClient exposing retrieve/delete --
    the raw transport clear_malformed_empty_memory reaches via
    memory.mem0._get_async_qdrant()."""
    client = AsyncMock()
    client.retrieve = AsyncMock(return_value=records if records is not None else [])
    client.delete = AsyncMock(return_value=None)
    return client


def _make_record(payload: dict | None = None) -> MagicMock:
    """Build a Qdrant-retrieve-shaped record stand-in (payload only --
    id/vector are irrelevant to this script)."""
    record = MagicMock()
    record.payload = payload if payload is not None else {}
    return record


# ===========================================================================
# Tests: extract_content
# ===========================================================================

class TestExtractContent:
    """Tests for the pure function extract_content(payload) -> str."""

    def test_returns_data_key_when_present(self):
        """'data' is the first-tried key and wins when present."""
        payload = {'data': 'hello from data', 'memory': 'hello from memory', 'content': 'hello from content'}

        assert _mod.extract_content(payload) == 'hello from data'

    def test_falls_back_to_memory_when_data_absent(self):
        """'memory' is used when 'data' is absent."""
        payload = {'memory': 'hello from memory', 'content': 'hello from content'}

        assert _mod.extract_content(payload) == 'hello from memory'

    def test_falls_back_to_content_when_data_and_memory_absent(self):
        """'content' is used when both 'data' and 'memory' are absent."""
        payload = {'content': 'hello from content'}

        assert _mod.extract_content(payload) == 'hello from content'

    def test_returns_empty_string_when_no_keys_present(self):
        """No content key at all -> ''."""
        payload = {'category': 'observations_and_summaries'}

        assert _mod.extract_content(payload) == ''

    def test_returns_empty_string_when_all_values_empty(self):
        """Every content key present but empty -> ''."""
        payload = {'data': '', 'memory': '', 'content': ''}

        assert _mod.extract_content(payload) == ''

    def test_returns_empty_string_for_empty_payload(self):
        """An entirely empty payload dict -> ''."""
        assert _mod.extract_content({}) == ''

    def test_skips_empty_data_and_falls_back_to_memory(self):
        """An empty 'data' value is treated as absent -- falls through to 'memory'."""
        payload = {'data': '', 'memory': 'hello from memory'}

        assert _mod.extract_content(payload) == 'hello from memory'

    def test_skips_non_string_values(self):
        """A non-string value (e.g. None) at a higher-priority key is
        skipped, falling through to the next key."""
        payload = {'data': None, 'memory': 'hello from memory'}

        assert _mod.extract_content(payload) == 'hello from memory'


# ===========================================================================
# Tests: is_malformed_empty_payload
# ===========================================================================

class TestIsMalformedEmptyPayload:
    """Tests for the pure function is_malformed_empty_payload(payload) -> bool."""

    def test_true_when_content_category_and_agent_id_all_absent(self):
        """The exact malformed fingerprint: empty content, category=None,
        agent_id=None."""
        payload = {'data': '', 'category': None, 'agent_id': None}

        assert _mod.is_malformed_empty_payload(payload) is True

    def test_true_when_keys_entirely_missing(self):
        """Missing keys are treated the same as explicit None -- an empty
        dict is still malformed."""
        payload = {}

        assert _mod.is_malformed_empty_payload(payload) is True

    def test_false_when_content_present(self):
        """Non-empty content alone rules out malformed, even with
        category/agent_id both None."""
        payload = {'data': 'some real content', 'category': None, 'agent_id': None}

        assert _mod.is_malformed_empty_payload(payload) is False

    def test_false_when_category_present(self):
        """A set category alone rules out malformed, even with empty
        content and agent_id=None."""
        payload = {'data': '', 'category': 'observations_and_summaries', 'agent_id': None}

        assert _mod.is_malformed_empty_payload(payload) is False

    def test_false_when_agent_id_present(self):
        """A set agent_id alone rules out malformed, even with empty
        content and category=None."""
        payload = {'data': '', 'category': None, 'agent_id': 'claude-task-2691-implementer'}

        assert _mod.is_malformed_empty_payload(payload) is False

    def test_false_for_healthy_observations_and_summaries_payload(self):
        """A realistic healthy record -- non-empty content, category set,
        agent_id set -- must never be classified malformed."""
        payload = {
            'data': 'Task 1470 wired /audit into /review Phase-2 Architectural Coherence.',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-1470-implementer',
        }

        assert _mod.is_malformed_empty_payload(payload) is False


# ===========================================================================
# Tests: classify_payload
# ===========================================================================

class TestClassifyPayload:
    """Tests for the pure function classify_payload(payload_or_none) -> str."""

    def test_none_classifies_absent(self):
        """A None payload (record not found) classifies as 'absent'."""
        assert _mod.classify_payload(None) == 'absent'

    def test_malformed_fingerprint_classifies_malformed(self):
        """A payload matching the malformed fingerprint classifies as
        'malformed'."""
        payload = {'data': '', 'category': None, 'agent_id': None}

        assert _mod.classify_payload(payload) == 'malformed'

    def test_healthy_payload_classifies_healthy(self):
        """A payload NOT matching the malformed fingerprint classifies as
        'healthy'."""
        payload = {
            'data': 'Task 1470 wired /audit into /review Phase-2 Architectural Coherence.',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-1470-implementer',
        }

        assert _mod.classify_payload(payload) == 'healthy'

    def test_empty_dict_classifies_malformed_not_absent(self):
        """An empty (but present) payload dict is 'malformed', distinct
        from a None (absent) record."""
        assert _mod.classify_payload({}) == 'malformed'


# ===========================================================================
# Tests: retrieve_payload
# ===========================================================================

class TestRetrievePayload:
    """Tests for async retrieve_payload(client, collection, memory_id) -> dict | None."""

    @pytest.mark.asyncio
    async def test_calls_retrieve_with_expected_args(self):
        """retrieve is awaited with collection_name/ids/with_payload=True/
        with_vectors=False."""
        client = _make_qdrant_mock([_make_record({'data': 'hello'})])

        await _mod.retrieve_payload(client, 'fused_dark_factory', '028edb1f-299c-4755-9432-f5a9fde97677')

        client.retrieve.assert_awaited_once_with(
            collection_name='fused_dark_factory',
            ids=['028edb1f-299c-4755-9432-f5a9fde97677'],
            with_payload=True,
            with_vectors=False,
        )

    @pytest.mark.asyncio
    async def test_returns_payload_dict_when_record_present(self):
        """Returns the first returned record's payload as a dict."""
        client = _make_qdrant_mock([_make_record({'data': 'hello', 'category': None, 'agent_id': None})])

        result = await _mod.retrieve_payload(client, 'fused_dark_factory', 'some-id')

        assert result == {'data': 'hello', 'category': None, 'agent_id': None}

    @pytest.mark.asyncio
    async def test_returns_none_when_retrieve_returns_empty_list(self):
        """An empty retrieve result (record not found / already deleted)
        returns None, not an empty dict."""
        client = _make_qdrant_mock([])

        result = await _mod.retrieve_payload(client, 'fused_dark_factory', 'absent-id')

        assert result is None


# ===========================================================================
# Tests: delete_point
# ===========================================================================

class TestDeletePoint:
    """Tests for async delete_point(client, collection, memory_id)."""

    @pytest.mark.asyncio
    async def test_calls_delete_exactly_once_with_collection_and_points_selector(self):
        """delete is awaited exactly once, with collection_name=collection and
        a points_selector carrying memory_id."""
        from qdrant_client.http import models as qmodels

        client = _make_qdrant_mock()

        await _mod.delete_point(client, 'fused_dark_factory', '028edb1f-299c-4755-9432-f5a9fde97677')

        client.delete.assert_awaited_once()
        call_kwargs = client.delete.call_args.kwargs
        assert call_kwargs.get('collection_name') == 'fused_dark_factory'
        points_selector = call_kwargs.get('points_selector')
        assert isinstance(points_selector, qmodels.PointIdsList)
        assert points_selector.points == ['028edb1f-299c-4755-9432-f5a9fde97677']


# ===========================================================================
# Tests: build_report
# ===========================================================================

class TestBuildReport:
    """Tests for the pure function build_report(...) -> dict."""

    def test_report_shape(self):
        """Returned dict has exactly the six report keys."""
        report = _mod.build_report(
            memory_id='id1', collection='fused_dark_factory', classification='malformed',
            dry_run=True, deleted=False, payload={},
        )

        assert set(report.keys()) == {
            'memory_id', 'collection', 'classification', 'dry_run', 'deleted', 'payload',
        }

    def test_fields_passed_through_verbatim(self):
        """Every field is passed through verbatim, not recomputed."""
        payload = {'data': '', 'category': None, 'agent_id': None}

        report = _mod.build_report(
            memory_id='028edb1f-299c-4755-9432-f5a9fde97677', collection='fused_dark_factory',
            classification='malformed', dry_run=False, deleted=True, payload=payload,
        )

        assert report['memory_id'] == '028edb1f-299c-4755-9432-f5a9fde97677'
        assert report['collection'] == 'fused_dark_factory'
        assert report['classification'] == 'malformed'
        assert report['dry_run'] is False
        assert report['deleted'] is True
        assert report['payload'] == payload

    def test_payload_none_for_absent_record(self):
        """An absent record's payload (None) is preserved, not coerced to {}."""
        report = _mod.build_report(
            memory_id='absent-id', collection='fused_dark_factory', classification='absent',
            dry_run=True, deleted=False, payload=None,
        )

        assert report['payload'] is None


# ===========================================================================
# Tests: resolve_exit_code
# ===========================================================================

class TestResolveExitCode:
    """Tests for the pure function resolve_exit_code(report) -> int."""

    def test_healthy_and_applied_returns_1(self):
        """A healthy record with dry_run=False (an --apply run that refused
        to delete) is the loud-refusal case -> exit 1."""
        report = _mod.build_report(
            memory_id='id1', collection='c', classification='healthy',
            dry_run=False, deleted=False, payload={'data': 'real content'},
        )

        assert _mod.resolve_exit_code(report) == 1

    def test_healthy_and_dry_run_returns_0(self):
        """A healthy record under dry-run (no --apply) is just a report,
        never a failure -> exit 0."""
        report = _mod.build_report(
            memory_id='id1', collection='c', classification='healthy',
            dry_run=True, deleted=False, payload={'data': 'real content'},
        )

        assert _mod.resolve_exit_code(report) == 0

    def test_malformed_returns_0_regardless_of_dry_run(self):
        """A malformed record always exits 0, whether or not it was applied."""
        for dry_run, deleted in ((True, False), (False, True)):
            report = _mod.build_report(
                memory_id='id1', collection='c', classification='malformed',
                dry_run=dry_run, deleted=deleted, payload={},
            )

            assert _mod.resolve_exit_code(report) == 0

    def test_absent_returns_0_regardless_of_dry_run(self):
        """An absent record (idempotent success) always exits 0."""
        for dry_run in (True, False):
            report = _mod.build_report(
                memory_id='id1', collection='c', classification='absent',
                dry_run=dry_run, deleted=False, payload=None,
            )

            assert _mod.resolve_exit_code(report) == 0


# ===========================================================================
# Tests: run
# ===========================================================================

class TestRun:
    """Tests for async run(args, qdrant_client, collection_name) -> dict."""

    @pytest.mark.asyncio
    async def test_dry_run_malformed_no_delete(self):
        """(a) apply=False + malformed: report reflects the classification,
        dry_run True, deleted False, and delete is never awaited."""
        payload = {'data': '', 'category': None, 'agent_id': None}
        client = _make_qdrant_mock([_make_record(payload)])
        args = types.SimpleNamespace(memory_id='id1', apply=False)

        report = await _mod.run(args, client, 'fused_dark_factory')

        assert report['classification'] == 'malformed'
        assert report['dry_run'] is True
        assert report['deleted'] is False
        client.delete.assert_not_awaited()
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_apply_malformed_deletes(self):
        """(b) apply=True + malformed: delete is awaited once, report
        reflects deleted=True, exit code 0."""
        payload = {'data': '', 'category': None, 'agent_id': None}
        client = _make_qdrant_mock([_make_record(payload)])
        args = types.SimpleNamespace(memory_id='id1', apply=True)

        report = await _mod.run(args, client, 'fused_dark_factory')

        client.delete.assert_awaited_once()
        assert report['deleted'] is True
        assert report['classification'] == 'malformed'
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_apply_healthy_refuses_loudly(self, caplog):
        """(c) apply=True + healthy: delete is NEVER awaited (fail-safe),
        classification 'healthy', exit code 1 (loud refusal), and a WARNING
        is logged."""
        payload = {
            'data': 'Task 1470 wired /audit into /review Phase-2 Architectural Coherence.',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-1470-implementer',
        }
        client = _make_qdrant_mock([_make_record(payload)])
        args = types.SimpleNamespace(memory_id='id1', apply=True)

        with caplog.at_level('WARNING'):
            report = await _mod.run(args, client, 'fused_dark_factory')

        client.delete.assert_not_awaited()
        assert report['classification'] == 'healthy'
        assert report['deleted'] is False
        assert _mod.resolve_exit_code(report) == 1
        assert any(rec.levelname == 'WARNING' for rec in caplog.records), (
            f'Expected a WARNING refusing the delete, got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_apply_absent_idempotent_success(self):
        """(d) apply=True + absent (retrieve -> []): no delete attempted,
        classification 'absent', exit code 0 (idempotent success)."""
        client = _make_qdrant_mock([])
        args = types.SimpleNamespace(memory_id='already-gone', apply=True)

        report = await _mod.run(args, client, 'fused_dark_factory')

        client.delete.assert_not_awaited()
        assert report['classification'] == 'absent'
        assert report['deleted'] is False
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_dry_run_healthy_no_delete(self):
        """(e) apply=False + healthy: no delete attempted, exit code 0."""
        payload = {
            'data': 'Task 1470 wired /audit into /review Phase-2 Architectural Coherence.',
            'category': 'observations_and_summaries',
            'agent_id': 'claude-task-1470-implementer',
        }
        client = _make_qdrant_mock([_make_record(payload)])
        args = types.SimpleNamespace(memory_id='id1', apply=False)

        report = await _mod.run(args, client, 'fused_dark_factory')

        client.delete.assert_not_awaited()
        assert report['classification'] == 'healthy'
        assert report['deleted'] is False
        assert _mod.resolve_exit_code(report) == 0


# ===========================================================================
# Tests: _build_parser
# ===========================================================================

class TestBuildParser:
    """Tests for the CLI parser factory _build_parser()."""

    def test_no_args_raises_system_exit(self):
        """--memory-id is required -- parsing with no args raises SystemExit."""
        parser = _mod._build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults_project_id_and_apply(self):
        """Only --memory-id given: project_id defaults to 'dark_factory' and
        apply defaults to False."""
        parser = _mod._build_parser()

        args = parser.parse_args(['--memory-id', 'X'])

        assert args.memory_id == 'X'
        assert args.project_id == 'dark_factory'
        assert args.apply is False

    def test_project_id_and_apply_overridable(self):
        """--project-id and --apply both override their defaults."""
        parser = _mod._build_parser()

        args = parser.parse_args(['--memory-id', 'X', '--project-id', 'reify', '--apply'])

        assert args.memory_id == 'X'
        assert args.project_id == 'reify'
        assert args.apply is True
