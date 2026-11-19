"""The architect's already-done report forwards only declared check descriptors (task 6306).

``TaskWorkflow._handle_already_done_report`` hands the task's
``metadata.delivered_checks`` to ``validate_reported_landing`` as read by
``shared/src/shared/declared_checks.py::declared_checks``, so an entry that is
not a dict never reaches the validator as landing evidence.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from test_workflow_already_done import _AD_DC_CHECK, _arm_already_done_fixture

from orchestrator.workflow import WorkflowOutcome

_VALIDATE_TARGET = 'orchestrator.workflow.validate_reported_landing'


@pytest.mark.asyncio
async def test_validator_receives_only_the_dict_entries(tmp_path: Path) -> None:
    f = _arm_already_done_fixture(
        tmp_path,
        metadata={'files': ['helpers/foo.py'], 'delivered_checks': ['junk', _AD_DC_CHECK]},
    )
    f.wf._hold_unattributed_already_done = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.BLOCKED,
    )
    validate = AsyncMock(return_value=SimpleNamespace(accepted=False, evidence_sha=None))

    with patch(_VALIDATE_TARGET, validate):
        outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    assert validate.await_args is not None
    assert validate.await_args.kwargs['delivered_checks'] == [_AD_DC_CHECK]
