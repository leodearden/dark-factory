"""Tests for `escalation.queue.observed_submit_response` — the ladder's front door.

Task 3236 made this function report OBSERVED state rather than write intent, so
a filing swallowed by a concurrent resolver is no longer reported as 'queued'.
Task 5368 closes the hole that fix left open: when the post-write re-read itself
fails (returns nothing, or raises), the response was byte-identical to a
CONFIRMED persist — so a filer whose escalation never landed was told it had.

The contract these tests freeze: an UNCONFIRMED persist is never shaped like a
confirmed one.  `'queued'` means, and only means, "the re-read found the record
pending on disk"; `'accepted_unpersisted'` means durability is unconfirmed and
the caller must keep driving its blocked task.

`observed_submit_response` had no direct test before this file — it was
exercised only incidentally through the server — so both the function-level
contract (below) and the server-boundary envelope it feeds live here together.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue, observed_submit_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUEUE_LOGGER = 'escalation.queue'


def _submit_pending(queue: EscalationQueue, *, level: int = 1) -> str:
    """Write one genuinely pending escalation and return its id."""
    esc = Escalation(
        id=queue.make_id('task-5368'),
        task_id='task-5368',
        agent_role='implementer',
        severity='blocking',
        category='infra_issue',
        summary='a filing whose persistence we then interrogate',
        level=level,
    )
    return queue.submit(esc)


def _error_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# Function-level contract
# ---------------------------------------------------------------------------


class TestObservedSubmitResponseShapes:
    """The three response shapes must stay mutually distinguishable."""

    def test_confirmed_pending_is_queued_with_no_persist_check(self, tmp_path: Path):
        """(a) A re-read that FINDS the record pending is the one durable shape.

        `queued` carries exactly id/status/level and no `persist_check` — the
        absence of that key is itself the signal that nothing was degraded.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert result == {'id': esc_id, 'status': 'queued', 'level': 1}, (
            f'Confirmed-persist shape drifted: {result}'
        )
        assert 'persist_check' not in result, (
            f'A confirmed persist must not carry a persist_check verdict: {result}'
        )

    def test_absent_record_reports_accepted_unpersisted(self, tmp_path: Path):
        """(b) A re-read returning None → unpersisted, discriminated as 'absent'.

        `EscalationQueue.get` searches the queue root, the archive, a targeted
        archive re-probe and a TOCTOU re-locate retry before returning None, so
        None is strong evidence the record genuinely is not on disk.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert result['status'] == 'accepted_unpersisted', (
            f"Expected 'accepted_unpersisted' for an absent record, got: {result}"
        )
        assert result['persist_check'] == 'absent', (
            f"Expected persist_check='absent', got: {result}"
        )
        assert result['level'] == 1, f'Level echo lost on the absent branch: {result}'
        assert result['id'] == esc_id, f'Escalation id lost: {result}'

    def test_unreadable_record_reports_accepted_unpersisted(self, tmp_path: Path):
        """(c) A re-read that RAISES → unpersisted, discriminated as 'unreadable'.

        The read failed, so the record's state is simply unknown — materially
        different from 'absent' for an operator debugging the outage, and the
        exception must not propagate to the filer.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        def _boom(escalation_id: str):
            raise OSError('simulated unreadable escalation file')

        queue.get = _boom  # type: ignore[method-assign]

        result = observed_submit_response(queue, esc_id, fallback_level=1)

        assert result['status'] == 'accepted_unpersisted', (
            f"Expected 'accepted_unpersisted' for an unreadable record, got: {result}"
        )
        assert result['persist_check'] == 'unreadable', (
            f"Expected persist_check='unreadable', got: {result}"
        )
        assert result['level'] == 1, f'Level echo lost on the unreadable branch: {result}'
        assert result['id'] == esc_id, f'Escalation id lost: {result}'

    def test_unpersisted_response_differs_from_confirmed(self, tmp_path: Path):
        """(d) The defect asserted directly: the two shapes must not re-converge.

        Same id, same level — if a future edit makes an unconfirmed persist
        indistinguishable from a confirmed one again, this fails.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        confirmed = observed_submit_response(queue, esc_id, fallback_level=1)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]
        unpersisted = observed_submit_response(queue, esc_id, fallback_level=1)

        assert unpersisted != confirmed, (
            'An UNCONFIRMED persist is shaped identically to a confirmed one — '
            f'the S8-13 defect: {unpersisted!r}'
        )

    @pytest.mark.parametrize('break_get', ['absent', 'unreadable'])
    def test_no_fallback_level_fabricates_no_level_key(self, tmp_path: Path, break_get: str):
        """(e) A legacy caller passing no fallback_level gets no fabricated level."""
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        if break_get == 'absent':
            queue.get = lambda escalation_id: None  # type: ignore[method-assign]
        else:
            def _boom(escalation_id: str):
                raise OSError('simulated unreadable escalation file')

            queue.get = _boom  # type: ignore[method-assign]

        result = observed_submit_response(queue, esc_id, fallback_level=None)

        assert result['status'] == 'accepted_unpersisted', f'Unexpected status: {result}'
        assert 'level' not in result, (
            f'level must never be fabricated when the caller supplied none: {result}'
        )


class TestUnpersistedLogsAtError:
    """(f) A filing whose durability is unconfirmed is not a routine degrade."""

    def test_absent_branch_logs_error_naming_the_id(self, tmp_path: Path, caplog):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)
        queue.get = lambda escalation_id: None  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger=_QUEUE_LOGGER):
            observed_submit_response(queue, esc_id, fallback_level=1)

        matching = [r for r in _error_records(caplog) if esc_id in r.getMessage()]
        assert matching, (
            f'Expected an ERROR naming {esc_id}; got: '
            f'{[(r.levelname, r.getMessage()) for r in caplog.records]}'
        )

    def test_unreadable_branch_logs_error_naming_the_id(self, tmp_path: Path, caplog):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _submit_pending(queue, level=1)

        def _boom(escalation_id: str):
            raise OSError('simulated unreadable escalation file')

        queue.get = _boom  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger=_QUEUE_LOGGER):
            observed_submit_response(queue, esc_id, fallback_level=1)

        matching = [r for r in _error_records(caplog) if esc_id in r.getMessage()]
        assert matching, (
            f'Expected an ERROR naming {esc_id}; got: '
            f'{[(r.levelname, r.getMessage()) for r in caplog.records]}'
        )
