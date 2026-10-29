"""BacklogPolicy scans, closes and files its escalation records off the event
loop's thread (INV-8, task 6086)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
from _fm_helpers import LoopFreedomProbe
from escalation.queue import EscalationQueue
from shared.project_layout import escalation_queue_dir

from fused_memory.reconciliation import backlog_policy
from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


def _policy(event_buffer: EventBuffer, project_root: Path) -> BacklogPolicy:
    policy = BacklogPolicy(event_buffer, None, lambda _root: True)
    policy.register_project_root('proj', str(project_root))
    return policy


def _probe_the_queue_dir(monkeypatch: pytest.MonkeyPatch, probe: LoopFreedomProbe) -> None:
    real = backlog_policy.escalation_queue_dir

    class _ProbedPath(type(Path())):
        def is_dir(self, *a, **kw):
            probe.block()
            return super().is_dir(*a, **kw)

        def glob(self, *a, **kw):
            probe.block()
            return super().glob(*a, **kw)

    monkeypatch.setattr(
        backlog_policy, 'escalation_queue_dir', lambda root: _ProbedPath(real(root)),
    )


class TestUnhaltRunsOffTheEventLoop:
    @pytest.mark.asyncio
    async def test_scanning_and_closing_a_halt_leaves_the_event_loop_free(
        self, event_buffer, tmp_path, monkeypatch,
    ):
        project_root = tmp_path / 'proj_root'
        project_root.mkdir()
        policy = _policy(event_buffer, project_root)

        verdict = await policy.on_judge_halt('proj', reason='seed')
        assert verdict.escalation_path is not None
        halt_id = Path(verdict.escalation_path).stem

        probe = LoopFreedomProbe()
        _probe_the_queue_dir(monkeypatch, probe)

        assert await policy.on_judge_unhalt('proj') == [halt_id]
        probe.assert_loop_stayed_free()

        closed = EscalationQueue(escalation_queue_dir(project_root)).get(halt_id)
        assert closed is not None
        assert closed.status == 'resolved'


class TestFilingRunsOffTheEventLoop:
    @pytest.mark.asyncio
    async def test_filing_a_halt_escalation_leaves_the_event_loop_free(
        self, event_buffer, tmp_path, monkeypatch,
    ):
        policy = _policy(event_buffer, tmp_path / 'proj_root')

        probe = LoopFreedomProbe()
        real = backlog_policy.submit_or_dedupe

        def blocking(*a, **kw):
            probe.block()
            return real(*a, **kw)

        monkeypatch.setattr(backlog_policy, 'submit_or_dedupe', blocking)

        verdict = await policy.on_judge_halt('proj', reason='probe')

        probe.assert_loop_stayed_free()
        assert verdict.outcome == 'escalated'
        assert verdict.escalation_path is not None
        record = json.loads(Path(verdict.escalation_path).read_text(encoding='utf-8'))
        assert record['error_type'] == 'ReconciliationJudgeHalted'
