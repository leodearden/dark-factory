"""The architect's already-done report forwards only declared check descriptors (task 6306).

``TaskWorkflow._handle_already_done_report`` hands the task's
``metadata.delivered_checks`` to ``validate_reported_landing`` as read by
``shared/src/shared/declared_checks.py::declared_checks``, so an entry that is
not a dict never reaches the validator as landing evidence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from test_workflow_already_done import _AD_DC_CHECK, _arm_already_done_fixture, _arm_queue

from orchestrator.merge_lane.landing_evidence import validate_reported_landing
from orchestrator.workflow import WorkflowOutcome

_VALIDATE_TARGET = 'orchestrator.workflow.validate_reported_landing'


@pytest.mark.asyncio
async def test_validator_receives_only_the_dict_entries(tmp_path: Path) -> None:
    f = _arm_already_done_fixture(
        tmp_path,
        metadata={'files': ['helpers/foo.py'], 'delivered_checks': ['junk', _AD_DC_CHECK]},
    )
    _arm_queue(f, tmp_path)
    validate = AsyncMock(wraps=validate_reported_landing)

    with patch(_VALIDATE_TARGET, validate):
        outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    assert validate.await_args is not None
    assert validate.await_args.kwargs['delivered_checks'] == [_AD_DC_CHECK]
