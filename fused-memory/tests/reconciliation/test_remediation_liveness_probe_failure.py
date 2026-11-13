"""A liveness probe that fails must not suppress a stranded-work escalation (S8-22).

In ``_run_remediation_pass``'s live-workflow gate a LIVE cited task SUPPRESSES
the persistent-finding escalation, so LIVE is the permissive answer there. A
probe that could not run — corroboration raised, the detector raised, or the
corroboration inputs could not be read — is UNKNOWN: it never suppresses.
The recon escalation it lets through names the CITED tasks whose liveness is
unknown, rather than claiming they were confirmed not live. A routed target's
unknown liveness is marked on the routed filing's log line; the routed
record's own detail does not carry it.

The fake detector models the lingering shape task 2963 exists for: a worktree
registration or orchestrator lock that keeps asserting liveness with no recent
commit, so only an explicit ``corroborated=False`` downgrades it.
"""

from __future__ import annotations

import json
import logging

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir

import fused_memory.reconciliation.harness as harness_module
from fused_memory.reconciliation.harness import LIVENESS_UNKNOWN_DETAIL_KEY
from reconciliation.harness_fixtures import (
    make_memory_service,
    open_event_buffer,
    open_journal,
)
from reconciliation.remediation_pass_fixtures import (
    finding_citing,
    in_progress_task,
    prepare_remediation_pass,
)

_HARNESS_LOGGER = 'fused_memory.reconciliation.harness'


@pytest_asyncio.fixture
async def journal(tmp_path):
    async with open_journal(tmp_path / 'liveness_journal') as j:
        yield j


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    async with open_event_buffer(tmp_path / 'liveness_eb.db') as b:
        yield b


@pytest.fixture
def memory_service():
    return make_memory_service()


@pytest.fixture
def drive(journal, event_buffer, memory_service, tmp_path, monkeypatch):
    """Run one remediation pass over *findings*; return its recon escalation queue.

    *before_run* receives the wired harness after ``prepare_remediation_pass``
    has pinned its inputs, so a test can override one of them. *project_root*
    is forwarded to ``prepare_remediation_pass``.
    """

    async def _drive(findings, cited_tasks, *, before_run=None, **pass_kwargs):
        harness, esc_queue, run_pass = await prepare_remediation_pass(
            journal=journal, event_buffer=event_buffer, memory_service=memory_service,
            tmp_path=tmp_path, monkeypatch=monkeypatch,
            findings=findings, cited_tasks=cited_tasks, **pass_kwargs,
        )
        if before_run is not None:
            before_run(harness)
        await run_pass()
        return esc_queue

    async def _no_worktree_index(_project_root):
        return {}

    monkeypatch.setattr(harness_module, 'worktree_index_kwargs', _no_worktree_index)
    return _drive


def _stub_corroboration(monkeypatch, *, raising_for: frozenset[str] = frozenset()):
    """Corroboration answers False, or raises for each id in *raising_for*."""

    def _corroboration(_task, tid, **_kw):
        if str(tid) in raising_for:
            raise RuntimeError(f'corroboration inputs unreadable for {tid}')
        return False

    monkeypatch.setattr(harness_module, 'corroboration_for_task', _corroboration)


def _stub_detector(
    monkeypatch, *,
    live: frozenset[str] = frozenset(),
    raising_for: frozenset[str] = frozenset(),
    lingering: frozenset[str] = frozenset(),
):
    """Live for *live*, raise for *raising_for*, lingering for *lingering*, else not live."""

    async def _is_live(tid, _project_root, **kw):
        tid = str(tid)
        if tid in raising_for:
            raise RuntimeError(f'git probe failed for {tid}')
        if tid in lingering:
            return kw.get('corroborated') is not False
        return tid in live

    monkeypatch.setattr(harness_module, 'is_workflow_live_for_task', _is_live)


def _stranded(esc_queue) -> list:
    return [
        e for e in esc_queue.get_pending()
        if e.category == 'recon_integrity_issue' and 'Persistently unresolved' in e.summary
    ]


def _warnings_naming(caplog, needle: str) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.name == _HARNESS_LOGGER and r.levelno == logging.WARNING
        and needle in r.getMessage()
    ]


class TestAFailedProbeCannotSuppress:

    @pytest.mark.asyncio
    async def test_a_raising_corroboration_escalates_with_the_task_marked_unknown(
        self, drive, monkeypatch, caplog,
    ):
        _stub_corroboration(monkeypatch, raising_for=frozenset({'901'}))
        _stub_detector(monkeypatch, lingering=frozenset({'901'}))

        with caplog.at_level(logging.WARNING, logger=_HARNESS_LOGGER):
            esc_queue = await drive([finding_citing('901')], [in_progress_task('901')])

        [esc] = _stranded(esc_queue)
        assert json.loads(esc.detail)[LIVENESS_UNKNOWN_DETAIL_KEY] == ['901']
        assert _warnings_naming(caplog, '901'), caplog.text

    @pytest.mark.asyncio
    async def test_a_raising_detector_escalates_with_the_task_marked_unknown(
        self, drive, monkeypatch,
    ):
        _stub_corroboration(monkeypatch)
        _stub_detector(monkeypatch, raising_for=frozenset({'901'}))

        esc_queue = await drive([finding_citing('901')], [in_progress_task('901')])

        [esc] = _stranded(esc_queue)
        assert json.loads(esc.detail)[LIVENESS_UNKNOWN_DETAIL_KEY] == ['901']

    @pytest.mark.asyncio
    async def test_an_unreadable_corroboration_input_marks_a_negative_unknown(
        self, drive, monkeypatch, caplog,
    ):
        _stub_corroboration(monkeypatch)
        _stub_detector(monkeypatch)

        def _unreadable(_root):
            raise OSError('scheduler state unreadable')

        def _break_scheduler_state(_harness):
            monkeypatch.setattr(harness_module, 'read_scheduler_state', _unreadable)

        with caplog.at_level(logging.WARNING, logger=_HARNESS_LOGGER):
            esc_queue = await drive(
                [finding_citing('901')], [in_progress_task('901')],
                before_run=_break_scheduler_state,
            )

        [esc] = _stranded(esc_queue)
        assert json.loads(esc.detail)[LIVENESS_UNKNOWN_DETAIL_KEY] == ['901']
        assert _warnings_naming(caplog, 'read_scheduler_state'), caplog.text


class TestAHealthyProbeIsUnchanged:

    @pytest.mark.asyncio
    async def test_a_confirmed_not_live_task_escalates_without_the_unknown_key(
        self, drive, monkeypatch,
    ):
        _stub_corroboration(monkeypatch)
        _stub_detector(monkeypatch)

        esc_queue = await drive([finding_citing('901')], [in_progress_task('901')])

        [esc] = _stranded(esc_queue)
        assert LIVENESS_UNKNOWN_DETAIL_KEY not in json.loads(esc.detail)

    @pytest.mark.asyncio
    async def test_a_confirmed_live_task_suppresses(self, drive, monkeypatch):
        _stub_corroboration(monkeypatch)
        _stub_detector(monkeypatch, live=frozenset({'901'}))

        esc_queue = await drive([finding_citing('901')], [in_progress_task('901')])

        assert _stranded(esc_queue) == []

    @pytest.mark.asyncio
    async def test_one_live_task_suppresses_despite_an_unknown_sibling(
        self, drive, monkeypatch,
    ):
        """902 is probed first and fails; 901 is then confirmed live."""
        _stub_corroboration(monkeypatch)
        _stub_detector(
            monkeypatch, live=frozenset({'901'}), raising_for=frozenset({'902'}),
        )

        esc_queue = await drive(
            [finding_citing('902', '901')],
            [in_progress_task('901'), in_progress_task('902')],
        )

        assert _stranded(esc_queue) == []


class TestTheRoutedTargetGateFailsClosedToo:
    """901 is confirmed not live; the finding is routed to 777, which lingers."""

    @staticmethod
    async def _drive_routed(drive, tmp_path, monkeypatch, caplog):
        orchestrator_root = tmp_path / 'orchestrator-root'
        monkeypatch.setattr(harness_module, 'is_orchestrator_live_for', lambda _root: True)
        with caplog.at_level(logging.INFO, logger=_HARNESS_LOGGER):
            await drive(
                [finding_citing('901', routed_to='777')],
                [in_progress_task('901'), in_progress_task('777')],
                project_root=str(orchestrator_root),
            )
        routed = EscalationQueue(escalation_queue_dir(orchestrator_root)).get_pending()
        filed = [
            r for r in caplog.records
            if r.name == _HARNESS_LOGGER
            and r.getMessage() == 'reconciliation.finding_task_escalation_filed'
        ]
        return routed, filed

    @pytest.mark.asyncio
    async def test_a_routed_target_whose_corroboration_raised_is_filed_marked_unknown(
        self, drive, tmp_path, monkeypatch, caplog,
    ):
        _stub_corroboration(monkeypatch, raising_for=frozenset({'777'}))
        _stub_detector(monkeypatch, lingering=frozenset({'777'}))

        routed, [filed] = await self._drive_routed(drive, tmp_path, monkeypatch, caplog)

        assert [e.task_id for e in routed] == ['777']
        assert (filed.task_id, filed.liveness_unknown) == ('777', True)
        assert filed.levelno == logging.WARNING

    @pytest.mark.asyncio
    async def test_a_confirmed_not_live_routed_target_is_filed_unmarked(
        self, drive, tmp_path, monkeypatch, caplog,
    ):
        _stub_corroboration(monkeypatch)
        _stub_detector(monkeypatch, lingering=frozenset({'777'}))

        routed, [filed] = await self._drive_routed(drive, tmp_path, monkeypatch, caplog)

        assert [e.task_id for e in routed] == ['777']
        assert (filed.task_id, filed.liveness_unknown) == ('777', False)
        assert filed.levelno == logging.INFO
