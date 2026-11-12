"""``ReconciliationHarness._run_judge`` reports every judge outcome by its kind (S8-23).

``Judge.review_run`` returns a verdict or a ``NoVerdict`` naming why there is
none. The harness logs the reason; a halted-project skip is routine, every
other no-verdict outcome is a WARNING, and none of them is a crash.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.models.reconciliation import JudgeVerdict, VerdictAction, VerdictSeverity
from fused_memory.reconciliation.judge import NoVerdict, NoVerdictReason
from reconciliation.harness_fixtures import open_harness

_HARNESS_LOGGER = 'fused_memory.reconciliation.harness'
_RUN_ID = 'run-judged'


async def _run_judge_returning(tmp_path, caplog, outcome) -> list[logging.LogRecord]:
    async with open_harness(tmp_path, taskmaster=None) as harness:
        harness.judge = MagicMock()
        harness.judge.review_run = AsyncMock(return_value=outcome)
        with caplog.at_level(logging.DEBUG, logger=_HARNESS_LOGGER):
            await harness._run_judge(_RUN_ID)
    return [r for r in caplog.records if r.name == _HARNESS_LOGGER]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('reason', 'level'),
    [
        (NoVerdictReason.JUDGE_INFRA_FAILURE, logging.WARNING),
        (NoVerdictReason.REVIEW_ERROR, logging.WARNING),
        (NoVerdictReason.RUN_NOT_FOUND, logging.WARNING),
        (NoVerdictReason.PROJECT_HALTED, logging.INFO),
    ],
)
async def test_a_no_verdict_is_logged_once_naming_its_reason(tmp_path, caplog, reason, level):
    records = await _run_judge_returning(
        tmp_path, caplog, NoVerdict(_RUN_ID, reason, 'some detail'),
    )

    naming = [
        r for r in records if _RUN_ID in r.getMessage() and reason.value in r.getMessage()
    ]
    assert [r.levelno for r in naming] == [level], [r.getMessage() for r in records]
    assert not [r for r in records if 'Judge task failed' in r.getMessage()]


@pytest.mark.asyncio
async def test_a_verdict_is_logged_with_its_severity(tmp_path, caplog):
    verdict = JudgeVerdict(
        run_id=_RUN_ID,
        reviewed_at=datetime.now(UTC),
        severity=VerdictSeverity.moderate,
        findings=[],
        action_taken=VerdictAction.rollback,
    )

    records = await _run_judge_returning(tmp_path, caplog, verdict)

    naming = [
        r for r in records if _RUN_ID in r.getMessage() and 'severity=moderate' in r.getMessage()
    ]
    assert [r.levelno for r in naming] == [logging.INFO], [r.getMessage() for r in records]
