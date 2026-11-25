"""Tests for ``fused-memory/scripts/sweep_episode_toolcall_xml_leak.py`` (task 3233).

The Graphiti counterpart of the Mem0 leak sweep: it reports which episodes
carry a leaked tool-call fragment and how each one classifies, and under
``--apply`` redacts the repairable ones in place.

The service is a stub exposing only the two methods the script may call, and
its scan returns a REAL ``EpisodeToolcallLeakScan``, so the report is built
from the same objects production hands it.

SENTINEL-LITERAL HAZARD (task 3083): every tool-call sentinel here is spelled
with the ``\\x3c`` escape for ``<``.
"""
from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import load_script_module
from _store_mutation_preflight_contract import (
    SENTINEL,
    deny,
    fail_closed_records,
    neutralise_fixture,
)

from fused_memory.backends.episode_toolcall_leak_scan import (
    EpisodeToolcallLeak,
    EpisodeToolcallLeakScan,
)
from fused_memory.backends.paged_read import INCOMPLETE_SHORT_READ, PagedRead
from fused_memory.maintenance import _utils as maintenance_utils
from fused_memory.utils.store_mutation_preflight import StoreMutationUnavailable
from fused_memory.utils.toolcall_xml_leak import (
    LeakShape,
    find_toolcall_xml_leak,
    repair_leaked_text,
)

SCRIPT_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'sweep_episode_toolcall_xml_leak.py'
_mod = load_script_module(SCRIPT_PATH, mod_name='sweep_episode_toolcall_xml_leak')

_neutralise = neutralise_fixture(
    _mod,
    note="""``run(..., apply=True)`` runs the preflight before it scans (task 3233),
    because an --apply redaction writes the Graphiti content SET plus a write
    journal row. This suite is MOCK-unit (a stub service, no live FalkorDB).
    ``TestApplyStoreMutationPreflight`` re-rigs it to refuse, so the guard's own
    behaviour is still pinned.""",
)

PROJECT = 'dark_factory'

_CLOSE_CONTENT = '\x3c/content>'
_CLOSE_INVOKE = '\x3c/invoke>'


def _open_param(name: str) -> str:
    return '\x3cparameter name="' + name + '">'


_BODY = 'The Stage 2 judge treats an empty backlog as a halt candidate.'
TAIL_CONTENT = _BODY + _CLOSE_CONTENT + '\n' + _CLOSE_INVOKE
DUPLICATE_CONTENT = _BODY + _CLOSE_CONTENT + '\n' + _open_param('content') + _BODY
MANUAL_CONTENT = _BODY + _CLOSE_CONTENT + '\n' + _open_param('priority') + 'low'


def _leak(uuid: str, content: str) -> EpisodeToolcallLeak:
    hit = find_toolcall_xml_leak(content)[0]
    return EpisodeToolcallLeak(
        uuid=uuid,
        name=f'episode {uuid}',
        created_at='2026-07-27T00:00:00Z',
        content=content,
        fragment=hit.fragment,
        index=hit.index,
    )


def _scan(
    *leaks: EpisodeToolcallLeak,
    expected_rows: int | None = None,
    incomplete_kind: str | None = None,
    episodes_in_group: int | None = 14456,
) -> EpisodeToolcallLeakScan:
    rows = [[leak.uuid, leak.name, leak.created_at, leak.content] for leak in leaks]
    complete = incomplete_kind is None
    candidates = PagedRead(
        rows=rows,
        complete=complete,
        rows_seen=len(rows),
        expected_rows=len(rows) if expected_rows is None else expected_rows,
        reason=None if complete else 'enumeration is SHORT',
        incomplete_kind=incomplete_kind,
    )
    return EpisodeToolcallLeakScan(
        group_id=PROJECT,
        episodes_in_group=episodes_in_group,
        candidates=candidates,
        leaks=tuple(leaks),
    )


class StubService:
    """Exposes ONLY the scan and the redact; any other attribute is an AttributeError."""

    __slots__ = ('redact_episode_content', 'scan', 'scan_calls')

    def __init__(
        self,
        scan: EpisodeToolcallLeakScan | BaseException,
        *,
        redact: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.scan = scan
        self.scan_calls: list[dict[str, Any]] = []
        self.redact_episode_content = AsyncMock(side_effect=redact or self._redacted)

    async def scan_episode_toolcall_leaks(self, **kwargs: Any) -> EpisodeToolcallLeakScan:
        self.scan_calls.append(kwargs)
        if isinstance(self.scan, BaseException):
            raise self.scan
        return self.scan

    def _redacted(self, episode_uuid: str, **kwargs: Any) -> dict[str, Any]:
        assert not isinstance(self.scan, BaseException)
        leak = next(leak for leak in self.scan.leaks if leak.uuid == episode_uuid)
        return {
            'status': 'redacted',
            'store': 'graphiti',
            'uuid': episode_uuid,
            'old_content': leak.content,
            'new_content': kwargs['new_content'],
        }


def _args(*argv: str) -> Any:
    return _mod._build_parser().parse_args(list(argv))


def _record(report: dict, uuid: str) -> dict:
    return next(record for record in report['records'] if record['uuid'] == uuid)


def _three_shapes() -> EpisodeToolcallLeakScan:
    return _scan(
        _leak('tail', TAIL_CONTENT),
        _leak('dup', DUPLICATE_CONTENT),
        _leak('manual', MANUAL_CONTENT),
    )


class TestParserDefaults:
    def test_defaults_to_a_dry_run_over_dark_factory(self) -> None:
        args = _args()

        assert args.apply is False
        assert args.project_id == 'dark_factory'


class TestDryRunReport:
    @pytest.mark.asyncio
    async def test_reports_every_leak_with_its_shape_and_mutates_nothing(self) -> None:
        service = StubService(_three_shapes())

        report = await _mod.run(_args(), service)

        assert report['project_id'] == PROJECT
        assert report['dry_run'] is True
        assert report['episodes_in_group'] == 14456
        assert report['candidates_seen'] == 3
        assert report['candidates_expected'] == 3
        assert report['complete'] is True
        assert report['incomplete_kind'] is None
        assert report['counts'][LeakShape.REPAIRABLE_TAIL] == 1
        assert report['counts'][LeakShape.REPAIRABLE_DUPLICATE] == 1
        assert report['counts'][LeakShape.MANUAL_REVIEW] == 1
        assert report['redacted'] == 0
        service.redact_episode_content.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_each_record_carries_the_episode_and_its_leak(self) -> None:
        scan = _three_shapes()

        report = await _mod.run(_args(), StubService(scan))

        for leak in scan.leaks:
            record = _record(report, leak.uuid)
            assert record['name'] == leak.name
            assert record['created_at'] == leak.created_at
            assert record['fragment'] == leak.fragment
            assert record['index'] == leak.index
            assert record['content'] == leak.content
        assert _record(report, 'tail')['classification'] == LeakShape.REPAIRABLE_TAIL
        assert _record(report, 'dup')['classification'] == LeakShape.REPAIRABLE_DUPLICATE
        assert _record(report, 'manual')['classification'] == LeakShape.MANUAL_REVIEW

    @pytest.mark.asyncio
    async def test_only_repairable_records_carry_a_repaired_content(self) -> None:
        report = await _mod.run(_args(), StubService(_three_shapes()))

        assert _record(report, 'tail')['repaired_content'] == repair_leaked_text(TAIL_CONTENT)
        assert _record(report, 'dup')['repaired_content'] == repair_leaked_text(DUPLICATE_CONTENT)
        assert 'repaired_content' not in _record(report, 'manual')

    @pytest.mark.asyncio
    async def test_an_incomplete_scan_is_disclosed(self) -> None:
        scan = _scan(
            _leak('tail', TAIL_CONTENT), expected_rows=9, incomplete_kind=INCOMPLETE_SHORT_READ,
        )

        report = await _mod.run(_args(), StubService(scan))

        assert report['complete'] is False
        assert report['incomplete_kind'] == INCOMPLETE_SHORT_READ
        assert report['candidates_expected'] == 9

    @pytest.mark.asyncio
    async def test_the_report_is_json_serializable(self) -> None:
        report = await _mod.run(_args(), StubService(_three_shapes()))

        assert json.loads(json.dumps(report))['records'][0]['uuid'] == 'tail'


class TestScanCall:
    @pytest.mark.asyncio
    async def test_scans_the_requested_project(self) -> None:
        service = StubService(_scan())

        await _mod.run(_args('--project-id', 'reify'), service)

        assert service.scan_calls == [{'project_id': 'reify'}]

    @pytest.mark.asyncio
    async def test_a_scan_timeout_propagates(self) -> None:
        service = StubService(TimeoutError('Graphiti scan_episode_toolcall_leaks timed out'))

        with pytest.raises(TimeoutError):
            await _mod.run(_args(), service)


class TestDryRunExitCode:
    @pytest.mark.asyncio
    async def test_a_dry_run_exits_zero_even_with_review_and_an_incomplete_scan(self) -> None:
        scan = _scan(
            _leak('manual', MANUAL_CONTENT), expected_rows=5, incomplete_kind=INCOMPLETE_SHORT_READ,
        )

        report = await _mod.run(_args(), StubService(scan))

        assert report['counts'][LeakShape.MANUAL_REVIEW] == 1
        assert _mod.resolve_exit_code(report) == 0


class TestMainReportsAnAbort:
    def test_a_service_entry_failure_prints_an_aborted_report_and_exits_two(
        self, monkeypatch, capsys,
    ) -> None:
        @contextlib.asynccontextmanager
        async def _failing_service(*_args: Any, **_kwargs: Any):
            raise RuntimeError('FalkorDB unreachable')
            yield

        monkeypatch.setattr(maintenance_utils, 'maintenance_service', _failing_service)

        code = _mod.main([])

        assert code == 2
        report = json.loads(capsys.readouterr().out)
        assert report['aborted'] is True
        assert report['records'] == []


# ---------------------------------------------------------------------------
# --apply
# ---------------------------------------------------------------------------


def _redacted_uuids(service: StubService) -> list[str]:
    return [call.args[0] for call in service.redact_episode_content.await_args_list]


class TestApplyRedactsOnlyRepairableShapes:
    @pytest.mark.asyncio
    async def test_redacts_each_repairable_episode_with_its_repair(self) -> None:
        scan = _three_shapes()
        service = StubService(scan)

        report = await _mod.run(_args('--apply'), service)

        assert _redacted_uuids(service) == ['tail', 'dup']
        for call in service.redact_episode_content.await_args_list:
            leak = next(leak for leak in scan.leaks if leak.uuid == call.args[0])
            assert call.kwargs['new_content'] == repair_leaked_text(leak.content)
            assert call.kwargs['project_id'] == PROJECT
        assert _record(report, 'tail')['redacted'] is True
        assert _record(report, 'dup')['redacted'] is True
        assert not _record(report, 'manual').get('redacted')
        assert report['redacted'] == 2

    @pytest.mark.asyncio
    async def test_redacts_into_the_requested_project(self) -> None:
        service = StubService(_scan(_leak('tail', TAIL_CONTENT)))

        await _mod.run(_args('--apply', '--project-id', 'reify'), service)

        (call,) = service.redact_episode_content.await_args_list
        assert call.kwargs['project_id'] == 'reify'


class TestApplyExitCode:
    @pytest.mark.asyncio
    async def test_a_clean_complete_fully_redacted_apply_exits_zero(self) -> None:
        scan = _scan(_leak('tail', TAIL_CONTENT), _leak('dup', DUPLICATE_CONTENT))

        report = await _mod.run(_args('--apply'), StubService(scan))

        assert _mod.resolve_exit_code(report) == 0

    @pytest.mark.asyncio
    async def test_any_manual_review_exits_one(self) -> None:
        report = await _mod.run(_args('--apply'), StubService(_three_shapes()))

        assert _mod.resolve_exit_code(report) == 1

    @pytest.mark.asyncio
    async def test_an_incomplete_scan_exits_one(self) -> None:
        scan = _scan(
            _leak('tail', TAIL_CONTENT), expected_rows=4, incomplete_kind=INCOMPLETE_SHORT_READ,
        )

        report = await _mod.run(_args('--apply'), StubService(scan))

        assert _mod.resolve_exit_code(report) == 1

    @pytest.mark.parametrize('flag', _mod.HUMAN_ADJUDICATION_FLAGS)
    def test_any_human_adjudication_flag_exits_one(self, flag: str) -> None:
        report = _mod.build_report(
            project_id=PROJECT,
            dry_run=False,
            scan_summary=_mod.summarize_scan(_scan(_leak('tail', TAIL_CONTENT))),
            records=[{'uuid': 'tail', 'classification': LeakShape.REPAIRABLE_TAIL, flag: True}],
        )

        assert _mod.resolve_exit_code(report) == 1


class TestApplyRecordsPerEpisodeFailures:
    @pytest.mark.asyncio
    async def test_a_failed_redact_is_recorded_and_the_sweep_continues(self) -> None:
        scan = _scan(_leak('first', TAIL_CONTENT), _leak('second', DUPLICATE_CONTENT))

        def _redact(episode_uuid: str, **kwargs: Any) -> dict[str, Any]:
            if episode_uuid == 'first':
                raise RuntimeError('FalkorDB write timed out')
            return {'old_content': DUPLICATE_CONTENT, 'new_content': kwargs['new_content']}

        report = await _mod.run(_args('--apply'), StubService(scan, redact=_redact))

        first = _record(report, 'first')
        assert first[_mod.RECORD_ERROR] is True
        assert 'FalkorDB write timed out' in first['error']
        assert not first.get('redacted')
        assert _record(report, 'second')['redacted'] is True
        assert _mod.resolve_exit_code(report) == 1

    @pytest.mark.asyncio
    async def test_content_that_changed_since_the_scan_is_flagged(self, caplog) -> None:
        scan = _scan(_leak('tail', TAIL_CONTENT))

        def _redact(episode_uuid: str, **kwargs: Any) -> dict[str, Any]:
            return {'old_content': 'someone rewrote it', 'new_content': kwargs['new_content']}

        with caplog.at_level(logging.WARNING, logger='sweep_episode_toolcall_xml_leak'):
            report = await _mod.run(_args('--apply'), StubService(scan, redact=_redact))

        record = _record(report, 'tail')
        assert record[_mod.CONTENT_CHANGED_SINCE_SCAN] is True
        assert _mod.resolve_exit_code(report) == 1
        assert _mod.REPLACED_CONTENT == 'replaced_content'
        assert record[_mod.REPLACED_CONTENT] == 'someone rewrote it'
        assert record['content'] == TAIL_CONTENT
        printed = _record(json.loads(json.dumps(report, default=str)), 'tail')
        assert printed[_mod.REPLACED_CONTENT] == 'someone rewrote it'
        naming_the_field = [
            entry for entry in caplog.records
            if entry.name == 'sweep_episode_toolcall_xml_leak'
            and entry.levelno == logging.WARNING
            and 'tail' in entry.getMessage()
            and _mod.REPLACED_CONTENT in entry.getMessage()
        ]
        assert len(naming_the_field) == 1

    @pytest.mark.asyncio
    async def test_an_unchanged_redaction_carries_no_replaced_content(self) -> None:
        report = await _mod.run(_args('--apply'), StubService(_scan(_leak('tail', TAIL_CONTENT))))

        record = _record(report, 'tail')
        assert record['redacted'] is True
        assert _mod.REPLACED_CONTENT not in record
        assert _mod.CONTENT_CHANGED_SINCE_SCAN not in record
        assert _mod.resolve_exit_code(report) == 0


class TestApplyStoreMutationPreflight:
    @pytest.mark.asyncio
    async def test_a_denied_preflight_refuses_before_the_scan(self, monkeypatch, caplog) -> None:
        deny(_mod, monkeypatch)
        service = StubService(_three_shapes())

        with caplog.at_level(logging.ERROR), pytest.raises(StoreMutationUnavailable, match=SENTINEL):
            await _mod.run(_args('--apply'), service)

        assert service.scan_calls == []
        service.redact_episode_content.assert_not_awaited()
        (record,) = fail_closed_records(caplog, 'sweep_episode_toolcall_xml_leak')
        assert SENTINEL in record.getMessage()

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch) -> None:
        deny(_mod, monkeypatch)
        service = StubService(_three_shapes())

        report = await _mod.run(_args(), service)

        assert report['dry_run'] is True
        assert len(report['records']) == 3
