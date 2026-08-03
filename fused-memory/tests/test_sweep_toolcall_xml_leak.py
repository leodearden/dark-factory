"""Tests for scripts/sweep_toolcall_xml_leak.py — the PURE core (task 3083).

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors test_clear_malformed_empty_memory.py, which is
also this script's structural template (pure core / thin async I/O /
orchestration / CLI, dry-run default, loud refusal under --apply).

The async orchestration tests live alongside these in a later step; everything
here is pure — no I/O, no mocks.

## What the classification has to get right

The sweep repairs stored memory text, so its fail-safe direction is the whole
design. Two shapes are recoverable WITHOUT losing a single character of real
content, and only those two are ever auto-repaired:

  * ``repairable_tail``    — the leak fragment runs to end-of-content and
    carries nothing after its own marker (the c759c53b shape). Dropping it
    restores exactly the text that was written.
  * ``repairable_duplicate`` — the text after the marker is a VERBATIM
    duplicate of the text before it (the 9f2d2ae6 shape, where the harness
    re-appended the truncated argument's remainder into the same field).
    Dropping it restores exactly the text that was written.

Anything else carrying a leak is ``manual_review`` and is NEVER mutated,
mirroring ``is_malformed_empty_payload``'s all-conditions-required fail-safe:
the predicate must be structurally incapable of authorizing the destruction of
real content.

## Sentinel-literal hazard (task 3083, pre-1)

Every tool-call sentinel below is spelled with the ``\\x3c`` escape for ``<``.
Writing one verbatim would make THIS file's own authoring tool call terminate
early — reproducing the bug under test. Byte-identical at runtime; leave them
escaped.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fused_memory.models import SourceStore

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'sweep_toolcall_xml_leak.py'


def _load_module() -> types.ModuleType:
    """Load sweep_toolcall_xml_leak.py from its file path."""
    mod_name = 'sweep_toolcall_xml_leak'
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

_BODY = 'The merge worker consumes the stash stack in project_root.'

# c759c53b shape: stray closing tag, real whitespace, bare closing invoke tag,
# nothing after it.
_TAIL_LEAK = _BODY + '\x3c/content>\n\x3c/invoke>'

# 9f2d2ae6 shape: stray closing tag, real whitespace, a serialized opening
# parameter tag, then a VERBATIM duplicate of the body.
_DUPLICATE_LEAK = _BODY + '\x3c/content>\n\x3cparameter name="content">' + _BODY

# Same marker, but the trailing text is NEITHER absent NOR a duplicate — a
# genuinely different second value. Auto-repairing this would destroy content.
_MANUAL_LEAK = (
    _BODY + '\x3c/content>\n\x3cparameter name="content">A different second value entirely.'
)


def _payload(content: str, key: str = 'data', **overrides) -> dict:
    """A REALISTIC raw Qdrant payload for one Mem0 memory.

    Deliberately not minimal. The delete/re-add repair rebuilds the point from
    scratch, so every key here is a key that can be LOST — and the caller
    metadata keys (``task_id``, ``kind``, any ``x_*``) are the sole retrieval
    axis for ``get_memories_by_metadata``/``count_memories_by_metadata``, which
    match payload keys by equality via ``MatchValue`` (mem0_client.py:315-359).
    A repair that drops them makes the memory invisible to every
    metadata-scoped consumer that could previously find it.

    The mem0-OWNED keys (``hash``/``created_at``/``updated_at``/``user_id``/
    ``agent_id``/``run_id``) are present too, so the tests can pin that they are
    NOT forged back into ``metadata=`` — mem0 re-derives them on write.
    """
    payload = {
        key: content,
        # mem0/Qdrant-owned or re-derived on write:
        'category': 'observations_and_summaries',
        'agent_id': 'claude-task-3083',
        'run_id': 'r-1',
        'user_id': 'dark_factory',
        'hash': 'e3b0c44298fc1c14',
        'created_at': '2026-07-27T00:00:00+00:00',
        'updated_at': None,
        # caller metadata — the only metadata-scoped retrieval axis:
        'task_id': '3083',
        'kind': 'observation',
        'x_stage': 'stage1',
    }
    payload.update(overrides)
    return payload


class TestClassifyRecord:
    """classify_record(payload) -> 'clean'|'repairable_tail'|'repairable_duplicate'|'manual_review'."""

    def test_clean_payload_is_clean(self):
        assert _mod.classify_record(_payload(_BODY)) == 'clean'

    def test_payload_with_no_content_is_clean(self):
        """An empty/absent content payload has nothing to repair."""
        assert _mod.classify_record({}) == 'clean'

    def test_tail_shape_is_repairable_tail(self):
        assert _mod.classify_record(_payload(_TAIL_LEAK)) == 'repairable_tail'

    def test_duplicate_shape_is_repairable_duplicate(self):
        assert _mod.classify_record(_payload(_DUPLICATE_LEAK)) == 'repairable_duplicate'

    def test_non_duplicate_trailing_text_is_manual_review(self):
        """The fail-safe case: unrecognised trailing text is never auto-dropped."""
        assert _mod.classify_record(_payload(_MANUAL_LEAK)) == 'manual_review'

    def test_leak_at_offset_zero_is_manual_review(self):
        """Removing the fragment would leave nothing, so there is no repair that
        preserves content — refuse rather than write an empty memory."""
        assert _mod.classify_record(_payload('\x3c/content>\n\x3c/invoke>')) == 'manual_review'

    def test_content_is_read_through_the_canonical_key_order(self):
        """'data' is the canonical Qdrant payload key, with 'memory'/'content'
        as fallbacks — judged identically to the existing dedup sweep."""
        for key in ('data', 'memory', 'content'):
            assert _mod.classify_record({key: _TAIL_LEAK}) == 'repairable_tail', key
        assert _mod._CONTENT_KEYS == ('data', 'memory', 'content')


class TestRepairContent:
    """repair_content(text) -> str | None — None means REFUSE, never mutate."""

    def test_tail_shape_is_repaired_to_the_original_body(self):
        assert _mod.repair_content(_TAIL_LEAK) == _BODY

    def test_duplicate_shape_is_repaired_to_a_single_copy_of_the_body(self):
        assert _mod.repair_content(_DUPLICATE_LEAK) == _BODY

    def test_manual_review_shape_returns_none(self):
        """THE safety assertion: a record the classifier cannot vouch for can
        never be auto-mutated, because there is no repaired string to write."""
        assert _mod.repair_content(_MANUAL_LEAK) is None

    def test_clean_text_is_returned_unchanged(self):
        """A no-op repair on clean text is what makes repair idempotent; None is
        reserved exclusively for 'refuse to touch this'."""
        assert _mod.repair_content(_BODY) == _BODY

    @pytest.mark.parametrize('text', [_TAIL_LEAK, _DUPLICATE_LEAK, _BODY])
    def test_repair_is_idempotent(self, text):
        once = _mod.repair_content(text)
        assert _mod.repair_content(once) == once

    @pytest.mark.parametrize('text', [_TAIL_LEAK, _DUPLICATE_LEAK, _BODY])
    def test_repair_never_empties_non_empty_input(self, text):
        """A repair that emptied a memory would be a worse corruption than the
        leak it fixed."""
        assert _mod.repair_content(text)


class TestBuildReport:
    """build_report — pure assembly of already-computed fields, no I/O."""

    def _records(self):
        return [
            {'id': 'a', 'classification': 'clean', 'repaired': False},
            {'id': 'b', 'classification': 'repairable_tail', 'repaired': True},
            {'id': 'c', 'classification': 'manual_review', 'repaired': False},
        ]

    def test_report_carries_scope_mode_and_counts(self):
        report = _mod.build_report(
            project_id='dark_factory',
            collection='fused_dark_factory',
            dry_run=True,
            exhaustive=False,
            scanned=3,
            truncated=False,
            limit=None,
            records=self._records(),
        )

        assert report['project_id'] == 'dark_factory'
        assert report['collection'] == 'fused_dark_factory'
        assert report['dry_run'] is True
        assert report['exhaustive'] is False
        assert report['scanned'] == 3
        assert report['truncated'] is False
        assert report['limit'] is None
        assert report['records'] == self._records()

    def test_counts_are_tallied_per_classification(self):
        report = _mod.build_report(
            project_id='dark_factory',
            collection='fused_dark_factory',
            dry_run=True,
            exhaustive=True,
            scanned=3,
            truncated=False,
            limit=None,
            records=self._records(),
        )

        assert report['counts']['clean'] == 1
        assert report['counts']['repairable_tail'] == 1
        assert report['counts']['repairable_duplicate'] == 0
        assert report['counts']['manual_review'] == 1
        assert report['repaired'] == 1


class TestResolveExitCode:
    """resolve_exit_code — the loud refusal, pure and sync."""

    def _report(self, **overrides):
        base = {
            'dry_run': True,
            'truncated': False,
            'counts': {
                'clean': 1,
                'repairable_tail': 0,
                'repairable_duplicate': 0,
                'manual_review': 0,
            },
        }
        base.update(overrides)
        return base

    def test_clean_corpus_exits_zero(self):
        assert _mod.resolve_exit_code(self._report(dry_run=False)) == 0

    def test_dry_run_always_exits_zero(self):
        """A dry run mutates nothing, so it is never a failure — even when it
        finds manual_review records. The printed report IS the investigation."""
        counts = {
            'clean': 0,
            'repairable_tail': 0,
            'repairable_duplicate': 0,
            'manual_review': 5,
        }
        assert _mod.resolve_exit_code(self._report(counts=counts, truncated=True)) == 0

    def test_apply_with_manual_review_exits_non_zero(self):
        """The confidently-classified records are still repaired, but the
        operator must not be able to mistake a partial sweep for a complete
        one."""
        counts = {
            'clean': 0,
            'repairable_tail': 2,
            'repairable_duplicate': 0,
            'manual_review': 1,
        }
        assert _mod.resolve_exit_code(self._report(dry_run=False, counts=counts)) != 0

    def test_apply_that_was_truncated_exits_non_zero(self):
        """Same rationale as manual_review: a --limit-capped or otherwise
        truncated apply covered an unknown fraction of the corpus, and reporting
        success would make a silently partial sweep look complete."""
        assert _mod.resolve_exit_code(self._report(dry_run=False, truncated=True)) != 0

    @pytest.mark.parametrize('flag', [
        'content_lost_in_flight',
        'skipped_not_mem0_routed',
        'skipped_metadata_would_be_rejected',
        'record_error',
    ])
    def test_every_per_record_failure_flag_exits_non_zero(self, flag):
        """Each of these leaves a record a human still has to adjudicate, so
        none of them may be reported as a clean sweep."""
        report = self._report(dry_run=False, records=[{'id': 'x', flag: True}])

        assert _mod.resolve_exit_code(report) != 0


class TestBuildParser:
    """_build_parser — the CLI surface, testable without any live I/O."""

    def test_dry_run_is_the_default(self):
        args = _mod._build_parser().parse_args([])
        assert args.apply is False
        assert args.project_id == 'dark_factory'
        assert args.exhaustive is False
        assert args.config is None
        assert args.limit is None

    def test_flags_parse(self):
        args = _mod._build_parser().parse_args(
            ['--apply', '--exhaustive', '--project-id', 'reify', '--limit', '50',
             '--config', '/tmp/c.yaml']
        )
        assert args.apply is True
        assert args.exhaustive is True
        assert args.project_id == 'reify'
        assert args.limit == 50
        assert args.config == '/tmp/c.yaml'

    @pytest.mark.parametrize('bad', ['0', '-1'])
    def test_a_non_positive_limit_is_refused_at_the_cli(self, bad, capsys):
        """``--limit 0`` would request 0 points per scroll page, so the walk
        returns ``{'matches': [], 'scanned': 0, 'truncated': False}`` — read by
        ``resolve_exit_code`` as a complete, CLEAN sweep even under --apply. A
        mistyped flag must not be able to certify a corpus it never looked at."""
        with pytest.raises(SystemExit):
            _mod._build_parser().parse_args(['--limit', bad])

        assert 'strictly positive' in capsys.readouterr().err

    def test_a_non_integer_limit_is_refused_at_the_cli(self, capsys):
        with pytest.raises(SystemExit):
            _mod._build_parser().parse_args(['--limit', 'lots'])

        assert 'must be an integer' in capsys.readouterr().err


# ===========================================================================
# Async orchestration — AsyncMock service, no live Qdrant (the repo-wide
# cleanup-script test scope; the live --apply run is the operator close-out).
# ===========================================================================

def _match(memory_id: str, content: str, *, payload: dict | None = None, **payload_extra) -> dict:
    """One scan_memory_content match. 'metadata' carries the RAW payload, which
    is where the full content lives (the 'excerpt' field is display-only and is
    deliberately never used for classification or repair).

    Shares ``_payload``'s realistic shape so the pure classification tests and
    the async repair tests judge the same payload, rather than the repair tests
    quietly exercising a stripped-down one that cannot expose metadata loss.
    """
    payload = _payload(content, **payload_extra) if payload is None else payload
    return {
        'id': memory_id,
        'created_at': '2026-07-27T00:00:00+00:00',
        'matched_fragments': [],
        'excerpt': content[:80],
        'metadata': payload,
    }


def _ok_response(**overrides) -> SimpleNamespace:
    """A REALISTIC successful ``MemoryService.add_memory`` response.

    ``stores_written`` is load-bearing, not decoration: ``add_memory`` returns
    NORMALLY when the Mem0 write fails (memory_service.py:2316-2317 catches
    ``Exception``, logs, and folds the failure into ``message`` as
    ``[mem0_error: ...]``), so the only evidence a mem0 copy actually landed is
    a non-empty ``memory_ids`` PLUS ``mem0`` in ``stores_written`` PLUS a
    message with no ``mem0_error``.
    """
    fields = {
        'memory_ids': ['new-id'],
        'stores_written': [SourceStore.mem0],
        'category': None,
        'message': "Memory queued for ['mem0']",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _service(
    matches, *, scanned=None, truncated=False, enforce=False, enforce_kind_registry=False
) -> AsyncMock:
    """An AsyncMock ``MemoryService`` for the sweep.

    ``config.memory_metadata`` is a REAL namespace with real bools, not a mock
    attribute, because the sweep's metadata pre-flight reads those flags off the
    live service (``resolve_metadata_enforcement``) and treats an unreadable
    config as "assume enforcing" — the fail-safe direction, but one that would
    silently turn every repair test into a skip. Both default to the SHIPPED
    defaults (warn-mode, kind registry not enforced), so the mock matches what a
    live sweep would actually see today.
    """
    service = AsyncMock()
    service.config = SimpleNamespace(
        memory_metadata=SimpleNamespace(
            enforce=enforce, enforce_kind_registry=enforce_kind_registry
        )
    )
    service.scan_memory_content = AsyncMock(
        return_value={
            'matches': matches,
            'scanned': len(matches) if scanned is None else scanned,
            'truncated': truncated,
        }
    )
    service.delete_memory = AsyncMock(return_value={'deleted': True})
    service.add_memory = AsyncMock(return_value=_ok_response())
    return service


def _args(**overrides):
    defaults = {
        'project_id': 'dark_factory',
        'apply': False,
        'exhaustive': False,
        'limit': None,
        'config': None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _record_for(report: dict, memory_id: str) -> dict:
    return next(r for r in report['records'] if r['id'] == memory_id)


class TestRunDiscovery:
    """Discovery goes through the literal text scan, never semantic search."""

    @pytest.mark.asyncio
    async def test_uses_scan_memory_content_and_never_search(self):
        service = _service([_match('a', _BODY)])

        await _mod.run(_args(), service)

        service.scan_memory_content.assert_awaited_once()
        service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_exhaustive_and_limit_are_threaded_to_the_scan(self):
        """--exhaustive is the mode whose answer depends on nothing but the
        shared Python detector, so it must actually reach the backend."""
        service = _service([])

        await _mod.run(_args(exhaustive=True, limit=25), service)

        kwargs = service.scan_memory_content.call_args[1]
        assert kwargs['exhaustive'] is True
        assert kwargs['limit'] == 25
        assert kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_empty_corpus_is_a_clean_success(self):
        """Distinguishable in the report from a failed scan: scanned is present
        and the run exits 0 rather than raising."""
        service = _service([], scanned=19321)

        report = await _mod.run(_args(apply=True), service)

        assert report['scanned'] == 19321
        assert report['records'] == []
        assert report['counts']['manual_review'] == 0
        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_scan_timeout_propagates(self):
        """A timed-out scan must never be reported as a clean corpus — that is
        the silent-wrong-answer class this whole task exists to kill."""
        service = _service([])
        service.scan_memory_content = AsyncMock(side_effect=TimeoutError('qdrant read'))

        with pytest.raises(TimeoutError):
            await _mod.run(_args(), service)


class TestRunDryRun:
    """Dry run is the default and performs ZERO mutations."""

    @pytest.mark.asyncio
    async def test_dry_run_mutates_nothing_but_classifies_everything(self):
        service = _service(
            [
                _match('a', _BODY),
                _match('b', _TAIL_LEAK),
                _match('c', _DUPLICATE_LEAK),
                _match('d', _MANUAL_LEAK),
            ]
        )

        report = await _mod.run(_args(), service)

        service.delete_memory.assert_not_awaited()
        service.add_memory.assert_not_awaited()
        assert report['dry_run'] is True
        assert _record_for(report, 'a')['classification'] == 'clean'
        assert _record_for(report, 'b')['classification'] == 'repairable_tail'
        assert _record_for(report, 'c')['classification'] == 'repairable_duplicate'
        assert _record_for(report, 'd')['classification'] == 'manual_review'
        assert report['repaired'] == 0


class TestRunApply:
    """--apply repairs only what can be repaired without losing content."""

    @pytest.mark.asyncio
    async def test_repair_is_delete_then_readd_in_that_order(self):
        """Delete + re-add, never an in-place payload SET: the repaired text has
        to be re-embedded, or the vector would still point at the corrupted
        string. Order matters — a re-add before the delete would leave both
        copies live if the delete then failed.
        """
        service = _service([_match('b', _TAIL_LEAK)])
        calls = []
        service.delete_memory.side_effect = lambda *a, **k: calls.append('delete') or {}
        service.add_memory.side_effect = (
            lambda *a, **k: calls.append('add') or _ok_response()
        )

        await _mod.run(_args(apply=True), service)

        assert calls == ['delete', 'add']

    @pytest.mark.asyncio
    async def test_readd_carries_repaired_content_and_original_attribution(self):
        service = _service([_match('b', _TAIL_LEAK)])

        await _mod.run(_args(apply=True), service)

        kwargs = service.add_memory.call_args[1]
        assert kwargs['content'] == _BODY
        assert kwargs['category'] == 'observations_and_summaries'
        assert kwargs['agent_id'] == 'claude-task-3083'
        assert kwargs['project_id'] == 'dark_factory'

    @pytest.mark.asyncio
    async def test_report_makes_the_old_to_new_mapping_auditable(self):
        """Exactly like Stage 1's 2c47b5cb->0a4f4848: both ids and both contents,
        so a human can verify after the fact that nothing was lost."""
        service = _service([_match('b', _TAIL_LEAK)])

        report = await _mod.run(_args(apply=True), service)

        record = _record_for(report, 'b')
        assert record['repaired'] is True
        assert record['new_id'] == 'new-id'
        assert record['content'] == _TAIL_LEAK
        assert record['repaired_content'] == _BODY
        assert report['repaired'] == 1

    @pytest.mark.asyncio
    async def test_manual_review_is_never_mutated_but_is_reported(self):
        service = _service([_match('d', _MANUAL_LEAK)])

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_not_awaited()
        service.add_memory.assert_not_awaited()
        record = _record_for(report, 'd')
        assert record['classification'] == 'manual_review'
        assert record['repaired'] is False
        assert record['content'] == _MANUAL_LEAK
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_clean_records_are_not_rewritten(self):
        service = _service([_match('a', _BODY)])

        await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_not_awaited()
        service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_succeeded_then_readd_raised_is_reported_loudly(self):
        """The one failure that actually destroys content. It must never be
        swallowed: the report has to carry both ids and the ORIGINAL content so
        the memory can be restored by hand, and the run must exit non-zero.
        """
        service = _service([_match('b', _TAIL_LEAK)])
        service.add_memory = AsyncMock(side_effect=RuntimeError('embedding backend down'))

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_awaited_once()
        record = _record_for(report, 'b')
        assert record['content_lost_in_flight'] is True
        assert record['id'] == 'b'
        assert record['content'] == _TAIL_LEAK
        assert record['repaired_content'] == _BODY
        assert record['repaired'] is False
        assert 'embedding backend down' in record['error']
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_a_truncated_apply_exits_non_zero(self):
        service = _service([_match('a', _BODY)], scanned=1000, truncated=True)

        report = await _mod.run(_args(apply=True, limit=1000), service)

        assert report['truncated'] is True
        assert _mod.resolve_exit_code(report) != 0


class TestReportSurvivesAnAbort:
    """The report is the sweep's central safety promise, so it must OUTLIVE any
    failure.

    For a record flagged ``content_lost_in_flight`` the point is already
    deleted, so the printed report holds the only surviving copy of the original
    text. A later record's transport error must therefore never take that report
    down with it — the runbook's "restore it from there" instruction would
    otherwise be pointing at output that was never emitted.
    """

    @pytest.mark.asyncio
    async def test_a_later_delete_failure_still_yields_an_earlier_records_lost_content(self):
        """The exact scenario the reviewer named: record 'b' loses its content
        in flight, then record 'c' blows up in ``delete_memory``. 'b's original
        text must still be in the returned report."""
        service = _service([_match('b', _TAIL_LEAK), _match('c', _DUPLICATE_LEAK)])
        service.add_memory = AsyncMock(side_effect=RuntimeError('embedding backend down'))
        service.delete_memory = AsyncMock(
            side_effect=[{'deleted': True}, TimeoutError('qdrant delete timed out')]
        )

        report = await _mod.run(_args(apply=True), service)

        lost = _record_for(report, 'b')
        assert lost['content_lost_in_flight'] is True
        assert lost['content'] == _TAIL_LEAK
        assert lost['repaired_content'] == _BODY

        failed = _record_for(report, 'c')
        assert failed['record_error'] is True
        assert failed['repaired'] is False
        assert 'qdrant delete timed out' in failed['error']
        assert failed['content'] == _DUPLICATE_LEAK

        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_one_bad_record_does_not_stop_later_records_being_repaired(self):
        """Continue-on-error, not abort-on-error: a single transport blip must
        not silently shrink the sweep's coverage."""
        service = _service([_match('b', _TAIL_LEAK), _match('c', _DUPLICATE_LEAK)])
        service.delete_memory = AsyncMock(
            side_effect=[TimeoutError('transient'), {'deleted': True}]
        )

        report = await _mod.run(_args(apply=True), service)

        assert _record_for(report, 'b')['record_error'] is True
        assert _record_for(report, 'c')['repaired'] is True
        assert report['repaired'] == 1

    @pytest.mark.asyncio
    async def test_progress_is_caller_owned_so_records_survive_an_escaping_error(self):
        """Belt to the per-record braces: even if something escapes ``run``
        entirely, the caller's container still holds every record processed so
        far."""
        progress = _mod.new_progress()
        service = _service([_match('b', _TAIL_LEAK), _match('c', _DUPLICATE_LEAK)])
        service.add_memory = AsyncMock(side_effect=RuntimeError('embedding backend down'))
        # Not an Exception subclass — models a genuinely escaping abort.
        service.delete_memory = AsyncMock(
            side_effect=[{'deleted': True}, KeyboardInterrupt()]
        )

        with pytest.raises(KeyboardInterrupt):
            await _mod.run(_args(apply=True), service, progress)

        report = _mod.report_from_progress(_args(apply=True), progress)
        lost = _record_for(report, 'b')
        assert lost['content_lost_in_flight'] is True
        assert lost['content'] == _TAIL_LEAK
        assert _mod.resolve_exit_code(report) != 0


class TestMainEmitsThePartialReport:
    """``main()`` must print whatever the sweep accumulated before it died."""

    def _run_main(self, monkeypatch, capsys, *, side_effect):
        """Drive ``main()`` to an abort without touching a live backend.

        ``main`` owns the progress container, so the test reaches it by pinning
        ``new_progress``; ``asyncio.run`` is replaced so ``_run_live``'s
        coroutine (which would construct a real MemoryService) never executes.
        """
        monkeypatch.setattr(sys, 'argv', ['sweep_toolcall_xml_leak.py', '--apply'])
        progress = _mod.new_progress()
        monkeypatch.setattr(_mod, 'new_progress', lambda: progress)

        def _abort(coro):
            coro.close()  # never construct the live MemoryService
            progress['collection'] = 'mem0_dark_factory'
            progress['scanned'] = 7
            progress['records'].append({
                'id': 'b',
                'classification': 'repairable_tail',
                'repaired': False,
                'content': _TAIL_LEAK,
                'repaired_content': _BODY,
                'content_lost_in_flight': True,
                'error': 'embedding backend down',
            })
            raise side_effect

        monkeypatch.setattr(_mod.asyncio, 'run', _abort)
        code = _mod.main()
        return code, capsys.readouterr().out

    def test_a_fatal_error_still_prints_the_lost_content(self, monkeypatch, capsys):
        code, out = self._run_main(
            monkeypatch, capsys, side_effect=RuntimeError('qdrant went away')
        )

        assert code == 2
        report = json.loads(out)
        assert report['aborted'] is True
        record = report['records'][0]
        assert record['content_lost_in_flight'] is True
        assert record['content'] == _TAIL_LEAK
        assert record['repaired_content'] == _BODY

    def test_the_partial_report_uses_the_same_shape_as_a_complete_one(
        self, monkeypatch, capsys
    ):
        """No second, lossier emergency format that could drift out of step."""
        _, out = self._run_main(monkeypatch, capsys, side_effect=TimeoutError('scan'))

        report = json.loads(out)
        for key in ('project_id', 'collection', 'dry_run', 'exhaustive',
                    'scanned', 'truncated', 'limit', 'counts', 'repaired', 'records'):
            assert key in report
        assert report['counts']['repairable_tail'] == 1


class TestReaddPersistedPredicate:
    """``readd_persisted(response) -> (bool, reason)`` — pure, no I/O.

    A non-raising ``add_memory`` is NOT evidence of a write. The service
    swallows a Mem0 failure (memory_service.py:2316-2317) and still returns a
    normal ``AddMemoryResponse``, so persistence has to be VERIFIED from the
    response rather than assumed from the absence of an exception.
    """

    def test_a_real_success_passes(self):
        ok, reason = _mod.readd_persisted(_ok_response())
        assert ok is True
        assert reason == ''

    def test_empty_memory_ids_fails(self):
        ok, reason = _mod.readd_persisted(_ok_response(memory_ids=[]))
        assert ok is False
        assert 'memory_ids' in reason

    def test_mem0_absent_from_stores_written_fails(self):
        """The delete removed the mem0 copy; a re-add that landed only in
        Graphiti has not restored it."""
        ok, reason = _mod.readd_persisted(
            _ok_response(stores_written=[SourceStore.graphiti])
        )
        assert ok is False
        assert 'mem0' in reason

    def test_mem0_error_in_the_message_fails(self):
        ok, reason = _mod.readd_persisted(
            _ok_response(message='Memory added [mem0_error: boom]')
        )
        assert ok is False
        assert 'mem0_error' in reason

    def test_a_plain_string_store_is_tolerated(self):
        """The predicate is compared via ``getattr(s, 'value', s)`` so a mock or
        a plain ``'mem0'`` str reads the same as the real StrEnum member."""
        assert _mod.readd_persisted(_ok_response(stores_written=['mem0']))[0] is True


class TestRunApplyVerifiesPersistence:
    """The failure that ACTUALLY happens: a non-raising add that did not persist.

    ``MemoryService.add_memory`` provably never raises on a Mem0 write failure —
    it catches ``Exception``, sets ``_mem0_error``, and returns a normal
    ``AddMemoryResponse`` whose ``message`` carries ``[mem0_error: ...]``. So the
    exception branch alone can never fire for the real-world outage, and treating
    "did not raise" as "persisted" reports permanent content loss as a clean,
    complete sweep.
    """

    @pytest.mark.asyncio
    async def test_mem0_error_in_message_is_content_lost_in_flight(self):
        service = _service([_match('b', _TAIL_LEAK)])
        service.add_memory = AsyncMock(
            return_value=SimpleNamespace(
                memory_ids=[],
                stores_written=[],
                category=None,
                message='Memory added [mem0_error: boom]',
            )
        )

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_awaited_once()
        record = _record_for(report, 'b')
        assert record['content_lost_in_flight'] is True
        assert record['repaired'] is False
        assert record['id'] == 'b'
        assert record['content'] == _TAIL_LEAK
        assert record['repaired_content'] == _BODY
        assert 'mem0_error' in record['error']
        assert 'boom' in record['error']
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_ids_returned_but_mem0_not_written_is_content_lost_in_flight(self):
        """The mem0 copy the delete removed was never restored, even though the
        response looks superficially successful."""
        service = _service([_match('b', _TAIL_LEAK)])
        service.add_memory = AsyncMock(
            return_value=_ok_response(
                memory_ids=['graphiti-only-id'],
                stores_written=[SourceStore.graphiti],
            )
        )

        report = await _mod.run(_args(apply=True), service)

        record = _record_for(report, 'b')
        assert record['content_lost_in_flight'] is True
        assert record['repaired'] is False
        assert record['content'] == _TAIL_LEAK
        assert 'mem0' in record['error']
        # Whatever ids DID come back are still recorded, so a human chasing the
        # loss has every handle the run saw.
        assert record['new_id'] == 'graphiti-only-id'
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_repaired_is_set_only_on_a_verified_persisted_readd(self):
        service = _service([_match('b', _TAIL_LEAK)])

        report = await _mod.run(_args(apply=True), service)

        record = _record_for(report, 'b')
        assert record['repaired'] is True
        assert record['new_id'] == 'new-id'
        assert 'content_lost_in_flight' not in record
        assert _mod.resolve_exit_code(report) == 0


class TestRunApplyPreflightCategoryGuard:
    """A repairable record that does not route to mem0 is left ENTIRELY alone.

    ``write_mem0 = resolved_category in MEM0_PRIMARY or dual_write``
    (memory_service.py:2212-2214), so a plain re-add of a Graphiti-primary
    category would send the repaired text to Graphiti only and the Qdrant copy
    the delete removed would be gone. ``dual_write=True`` is no better: it would
    duplicate the Graphiti copy that ``delete_memory(store='mem0')``
    deliberately left alive. Unsafe both ways, so the sweep does not choose —
    it refuses before deleting anything and hands the record to a human.
    """

    @pytest.mark.asyncio
    async def test_graphiti_primary_category_is_never_deleted(self):
        service = _service(
            [_match('g', _TAIL_LEAK, category='decisions_and_rationale')]
        )

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_not_awaited()
        service.add_memory.assert_not_awaited()
        record = _record_for(report, 'g')
        assert record['skipped_not_mem0_routed'] is True
        assert record['repaired'] is False
        assert record['classification'] == 'repairable_tail'
        assert record['content'] == _TAIL_LEAK
        assert 'decisions_and_rationale' in record['error']
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_missing_category_is_skipped_rather_than_guessed(self):
        match = _match('h', _TAIL_LEAK)
        match['metadata'].pop('category')
        service = _service([match])

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_not_awaited()
        assert _record_for(report, 'h')['skipped_not_mem0_routed'] is True
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_unrecognised_category_is_skipped_not_raised(self):
        """An unknown category string must not crash the sweep mid-corpus and
        must not be optimistically treated as mem0-routed."""
        service = _service([_match('i', _TAIL_LEAK, category='not_a_real_category')])

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_not_awaited()
        assert _record_for(report, 'i')['skipped_not_mem0_routed'] is True
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_a_mem0_primary_category_still_repairs(self):
        """The guard must not become a blanket refusal — the whole corpus this
        sweep targets is mem0-primary."""
        for category in (
            'observations_and_summaries',
            'procedural_knowledge',
            'preferences_and_norms',
        ):
            service = _service([_match('m', _TAIL_LEAK, category=category)])

            report = await _mod.run(_args(apply=True), service)

            service.delete_memory.assert_awaited_once()
            record = _record_for(report, 'm')
            assert record['repaired'] is True, category
            assert 'skipped_not_mem0_routed' not in record, category


class TestMetadataAcceptedPredicate:
    """``metadata_accepted(metadata, *, enforce, enforce_kind_registry)`` — pure.

    ``MemoryService.add_memory`` runs ``_apply_memory_metadata_validation``,
    which raises ``MemoryMetadataValidationError`` when ``enforce`` is on and
    any violation is fatal. Since the repair is delete-THEN-add, a rejection
    discovered on the add lands after the only copy is already gone. The corpus
    being swept is precisely the OLD, pre-vocabulary one, so this is a live
    possibility rather than a theoretical one — and it is a pure sync shape
    check, so there is no excuse for not running it first.
    """

    def test_warn_mode_accepts_even_fatal_shapes(self):
        """``enforce=False`` is the shipped default and does NOT reject the
        write, so blocking on it would refuse repairs the store would happily
        have accepted."""
        accepted, reason = _mod.metadata_accepted(
            {'canonical': 1}, enforce=False, enforce_kind_registry=True
        )

        assert accepted is True
        assert reason == ''

    def test_enforcing_a_fatal_shape_is_refused_with_the_rule_named(self):
        accepted, reason = _mod.metadata_accepted(
            {'canonical': 1}, enforce=True, enforce_kind_registry=False
        )

        assert accepted is False
        assert 'invalid_canonical_type' in reason

    def test_clean_metadata_is_accepted_under_enforcement(self):
        accepted, reason = _mod.metadata_accepted(
            {'task_id': '3083', 'x_stage': 'stage1'},
            enforce=True,
            enforce_kind_registry=True,
        )

        assert accepted is True
        assert reason == ''

    def test_unknown_kind_is_fatal_only_when_the_registry_is_enforced(self):
        """The carved-out flag is the whole reason ``enforce`` alone is not
        enough: 242 of 329 live kind values are singletons, so treating an
        unknown kind as fatal under plain ``enforce`` would skip most of the
        corpus."""
        meta = {'kind': 'observation'}

        assert _mod.metadata_accepted(
            meta, enforce=True, enforce_kind_registry=False
        )[0] is True
        assert _mod.metadata_accepted(
            meta, enforce=True, enforce_kind_registry=True
        )[0] is False

    def test_the_callers_metadata_is_not_mutated(self):
        """``validate_memory_metadata`` normalizes ``supersedes`` IN PLACE, so
        the predicate must hand it a copy — otherwise a pure pre-flight would
        silently rewrite the very dict the repair is about to send."""
        meta = {'supersedes': 'a' * 8}
        before = dict(meta)

        _mod.metadata_accepted(meta, enforce=True, enforce_kind_registry=False)

        assert meta == before


class TestResolveMetadataEnforcement:
    """The flags are READ off the live service, never assumed."""

    def test_real_config_flags_are_read_through(self):
        service = _service([], enforce=True, enforce_kind_registry=True)

        assert _mod.resolve_metadata_enforcement(service) == (True, True)

    def test_shipped_defaults_are_read_through(self):
        assert _mod.resolve_metadata_enforcement(_service([])) == (False, False)

    def test_an_unreadable_config_fails_safe_to_enforcing(self, caplog):
        """A stub service must not be read as "enforcement off" — that would
        re-open the post-delete rejection this pre-flight exists to close. Both
        ON means a questionable record is skipped rather than destroyed, and it
        is logged rather than silent."""
        with caplog.at_level('WARNING'):
            result = _mod.resolve_metadata_enforcement(SimpleNamespace())

        assert result == (True, True)
        assert 'could not read config.memory_metadata' in caplog.text


class TestRunApplyPreflightMetadataGuard:
    """A record whose carried metadata would be REJECTED is never deleted.

    Without the pre-flight the sequence is delete → add → raise, and the record
    ends up ``content_lost_in_flight`` with its text surviving only on stdout —
    caused entirely by a condition that was checkable, synchronously, before
    anything was destroyed.
    """

    @pytest.mark.asyncio
    async def test_metadata_that_would_be_rejected_is_skipped_before_the_delete(self):
        service = _service(
            [_match('r', _TAIL_LEAK, canonical=1)],
            enforce=True,
        )

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_not_awaited()
        service.add_memory.assert_not_awaited()
        record = _record_for(report, 'r')
        assert record['skipped_metadata_would_be_rejected'] is True
        assert record['repaired'] is False
        assert record['classification'] == 'repairable_tail'
        # The content is still in the report, so a human can act on it.
        assert record['content'] == _TAIL_LEAK
        assert 'invalid_canonical_type' in record['error']
        assert _mod.resolve_exit_code(report) != 0

    @pytest.mark.asyncio
    async def test_an_unknown_kind_is_skipped_only_when_the_registry_is_enforced(self):
        """``_payload``'s ``kind='observation'`` is not in KIND_REGISTRY — the
        realistic old-corpus shape this guard exists for."""
        enforced = _service(
            [_match('r', _TAIL_LEAK)], enforce=True, enforce_kind_registry=True
        )
        report = await _mod.run(_args(apply=True), enforced)
        enforced.delete_memory.assert_not_awaited()
        assert _record_for(report, 'r')['skipped_metadata_would_be_rejected'] is True

        warn_only = _service([_match('r', _TAIL_LEAK)], enforce=True)
        report = await _mod.run(_args(apply=True), warn_only)
        warn_only.delete_memory.assert_awaited_once()
        assert _record_for(report, 'r')['repaired'] is True

    @pytest.mark.asyncio
    async def test_warn_mode_still_repairs_so_the_guard_is_not_a_blanket_refusal(self):
        """Warn-mode is the SHIPPED default, so a guard that blocked there would
        make the whole sweep a no-op on the live fleet."""
        service = _service([_match('r', _TAIL_LEAK, canonical=1)])

        report = await _mod.run(_args(apply=True), service)

        service.delete_memory.assert_awaited_once()
        record = _record_for(report, 'r')
        assert record['repaired'] is True
        assert 'skipped_metadata_would_be_rejected' not in record

    @pytest.mark.asyncio
    async def test_a_dry_run_never_skips_or_mutates(self):
        """The pre-flight guards a delete; with no delete there is nothing to
        guard, and the dry run must still CLASSIFY the record."""
        service = _service([_match('r', _TAIL_LEAK, canonical=1)], enforce=True)

        report = await _mod.run(_args(), service)

        service.delete_memory.assert_not_awaited()
        record = _record_for(report, 'r')
        assert record['classification'] == 'repairable_tail'
        assert 'skipped_metadata_would_be_rejected' not in record
        assert _mod.resolve_exit_code(report) == 0


class TestCarriedMetadata:
    """``carried_metadata(payload, memory_id) -> (metadata, dropped)`` — pure.

    The carry-over rule is ``payload keys - _MEM0_OWNED_KEYS``, plus the
    ``x_repaired_from_memory_id`` provenance marker.
    """

    def test_caller_metadata_is_carried_through_unchanged(self):
        metadata, _dropped = _mod.carried_metadata(_payload(_BODY), 'old-id')

        assert metadata['task_id'] == '3083'
        assert metadata['kind'] == 'observation'
        assert metadata['x_stage'] == 'stage1'
        assert metadata['x_repaired_from_memory_id'] == 'old-id'

    def test_mem0_owned_keys_are_never_forged_into_metadata(self):
        """mem0 re-derives each of these on write, and ``category`` is
        overwritten by the service anyway (memory_service.py:2193)."""
        metadata, _dropped = _mod.carried_metadata(_payload(_BODY), 'old-id')

        for owned in (
            'id', 'hash', 'data', 'memory', 'content', 'created_at', 'updated_at',
            'user_id', 'agent_id', 'run_id', 'actor_id', 'role', 'category',
        ):
            assert owned not in metadata, owned

    def test_dropped_names_only_the_keys_carried_nowhere(self):
        """``data``/``category``/``agent_id``/``run_id``/``user_id`` are carried
        as ARGUMENTS, so reporting them as dropped would be a false alarm that
        buries the real ones.

        ``user_id`` is the subtle one: mem0 writes that payload key from
        ``Scope.mem0_user_id`` (mem0_client.py:203), which returns ``project_id``
        verbatim (models/scope.py:399-404) — and the sweep passes
        ``project_id=args.project_id``. So it round-trips, and naming it a loss
        would be a per-record false alarm on EVERY repaired record.
        """
        _metadata, dropped = _mod.carried_metadata(_payload(_BODY), 'old-id')

        assert dropped == ['created_at', 'hash', 'updated_at']

    def test_the_original_creation_time_is_carried_as_provenance(self):
        """The delete/re-add necessarily re-stamps ``created_at``, so without an
        ``x_``-namespaced copy a repaired 2026-07-27 memory reappears as brand
        new and the original recency ordering is unrecoverable from the store."""
        metadata, dropped = _mod.carried_metadata(_payload(_BODY), 'old-id')

        assert metadata['x_original_created_at'] == '2026-07-27T00:00:00+00:00'
        # The KEY is still genuinely re-derived by the new write, so it stays
        # named as dropped even though its VALUE survives.
        assert 'created_at' in dropped

    def test_a_payload_with_no_created_at_carries_no_empty_marker(self):
        """A null provenance marker would be worse than none: it reads as
        "the original had no creation time" rather than "nothing was carried"."""
        metadata, _dropped = _mod.carried_metadata(
            {'data': _BODY, 'category': 'observations_and_summaries'}, 'old-id'
        )

        assert 'x_original_created_at' not in metadata

    def test_owned_keys_are_derived_from_the_shared_mem0_constant(self):
        """INV-5: mem0's owned-key set has exactly one home. A hand-maintained
        copy here would silently fail to grow when mem0's payload shape does,
        and the repair would start forging a newly-owned key back into
        metadata."""
        from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS

        assert MEM0_MANAGED_METADATA_KEYS <= _mod._MEM0_OWNED_KEYS
        assert _mod._MEM0_OWNED_KEYS - MEM0_MANAGED_METADATA_KEYS == {
            'id', 'memory', 'content', 'category',
        }

    def test_a_bare_payload_drops_and_preserves_nothing(self):
        metadata, dropped = _mod.carried_metadata(
            {'data': _BODY, 'category': 'observations_and_summaries'}, 'old-id'
        )

        assert dropped == []
        assert metadata == {'x_repaired_from_memory_id': 'old-id'}

    def test_the_callers_payload_is_not_mutated(self):
        payload = _payload(_BODY)
        before = dict(payload)

        _mod.carried_metadata(payload, 'old-id')

        assert payload == before


class TestRunApplyPreservesMetadata:
    """The repaired memory must stay findable by everything that could find it.

    ``get_memories_by_metadata``/``count_memories_by_metadata`` match payload
    KEYS by equality via ``MatchValue`` (mem0_client.py:315-359), so the payload
    metadata is a repaired record's only metadata-scoped retrieval axis.
    Rebuilding the point with a provenance marker alone would silently make it
    invisible to every such consumer — a second, undisclosed mutation of stored
    state layered on top of the harness's first, and flatly at odds with calling
    the repair content-preserving.
    """

    @pytest.mark.asyncio
    async def test_caller_metadata_survives_the_delete_and_readd(self):
        service = _service([_match('b', _TAIL_LEAK)])

        await _mod.run(_args(apply=True), service)

        metadata = service.add_memory.call_args[1]['metadata']
        assert metadata['task_id'] == '3083'
        assert metadata['kind'] == 'observation'
        assert metadata['x_stage'] == 'stage1'
        assert metadata['x_repaired_from_memory_id'] == 'b'

    @pytest.mark.asyncio
    async def test_mem0_owned_keys_are_not_forged_into_metadata(self):
        service = _service([_match('b', _TAIL_LEAK)])

        await _mod.run(_args(apply=True), service)

        metadata = service.add_memory.call_args[1]['metadata']
        for owned in (
            'id', 'hash', 'data', 'memory', 'content', 'created_at', 'updated_at',
            'user_id', 'agent_id', 'run_id', 'category',
        ):
            assert owned not in metadata, owned

    @pytest.mark.asyncio
    async def test_scope_identities_are_threaded_as_arguments_not_metadata(self):
        """mem0 writes its ``run_id`` payload key from ``Scope.session_id``
        (mem0_client.py:124-126), so the original point's ``run_id`` has to go
        back in through ``session_id=`` rather than being forged as metadata —
        otherwise the repaired point lands with a null run_id."""
        service = _service([_match('b', _TAIL_LEAK)])

        await _mod.run(_args(apply=True), service)

        kwargs = service.add_memory.call_args[1]
        assert kwargs['agent_id'] == 'claude-task-3083'
        assert kwargs['session_id'] == 'r-1'

    @pytest.mark.asyncio
    async def test_the_report_makes_any_metadata_loss_auditable(self):
        """The printed report is the whole investigation artifact, so what was
        carried and what was not has to be readable from it alone."""
        service = _service([_match('b', _TAIL_LEAK)])

        report = await _mod.run(_args(apply=True), service)

        record = _record_for(report, 'b')
        assert record['metadata_preserved'] == ['kind', 'task_id', 'x_stage']
        assert record['metadata_dropped'] == ['created_at', 'hash', 'updated_at']

    @pytest.mark.asyncio
    async def test_a_bare_payload_still_repairs_with_empty_preserved_and_dropped(self):
        bare = {'data': _TAIL_LEAK, 'category': 'observations_and_summaries'}
        service = _service([_match('b', _TAIL_LEAK, payload=bare)])

        report = await _mod.run(_args(apply=True), service)

        record = _record_for(report, 'b')
        assert record['repaired'] is True
        assert record['metadata_preserved'] == []
        assert record['metadata_dropped'] == []
        assert service.add_memory.call_args[1]['metadata'] == {
            'x_repaired_from_memory_id': 'b'
        }

    @pytest.mark.asyncio
    async def test_metadata_audit_is_present_even_when_content_is_lost(self):
        """Precisely when a hand restore is needed, the operator must be able to
        rebuild the metadata too, not just the text."""
        service = _service([_match('b', _TAIL_LEAK)])
        service.add_memory = AsyncMock(side_effect=RuntimeError('embedding backend down'))

        report = await _mod.run(_args(apply=True), service)

        record = _record_for(report, 'b')
        assert record['content_lost_in_flight'] is True
        assert record['metadata_preserved'] == ['kind', 'task_id', 'x_stage']
