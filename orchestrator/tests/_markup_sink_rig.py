"""What a suite needs to drive ``orchestrator.mcp.markup_sink.make_escalation_sink``.

Shared by every suite that builds the sink directly or reads who a server's
sink attributed a record to, so a change to the queue interface the sink
drives, to its loss-sink contract or to how a subject is seeded is one edit
here.
"""

from __future__ import annotations

from typing import Any

import pytest
from escalation.models import Escalation

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.markup_sink import LossCause


class LossRecorder:
    """A loss sink that keeps every record handed to it."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def __call__(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class RefusingQueue:
    """A queue that opens, but whose ``submit`` fails as a full disk would."""

    def make_id(self, anchor: str) -> str:
        return f'esc-{anchor}-1'

    def submit(self, esc: Any) -> str:
        raise OSError('No space left on device')


class RecordingQueue:
    """A queue that takes every record and keeps it for the row to read."""

    def __init__(self) -> None:
        self.submitted: list[Any] = []

    def make_id(self, anchor: str) -> str:
        return f'esc-{anchor}-1'

    def get_by_task(self, anchor: str, status: str = '') -> list[Any]:
        return []

    def submit(self, esc: Any) -> str:
        self.submitted.append(esc)
        return esc.id


#: Every way the queue can fail to take a record, as
#: ``(resolve_root, open_channel, cause)``: the sink's two seams, and the
#: :class:`LossCause` it must name for that failure.
QUEUE_LOSS_SHAPES = [
    pytest.param(
        lambda worktree: None, lambda root: None,
        LossCause.ROOT_UNRESOLVED, id='root-unresolved',
    ),
    pytest.param(
        lambda worktree: worktree, lambda root: None,
        LossCause.QUEUE_UNOPENED, id='queue-unopened',
    ),
    pytest.param(
        lambda worktree: worktree, lambda root: (Escalation, RefusingQueue()),
        LossCause.SUBMIT_FAILED, id='submit-raised',
    ),
]


def seed_plan(artifacts: TaskArtifacts, task_id: str) -> None:
    """Write the plan a record is attributed to.

    Both servers' ``_markup_subject_task_id`` read ``plan.json``'s own
    ``task_id``, NOT ``metadata.json``'s. ``TaskArtifacts.init`` writes only
    metadata, so a row that wants the real attribution answer rather than
    ``markup_sink.resolve_subject``'s worktree-name fallback seeds it here.
    """
    artifacts.write_plan({
        'task_id': task_id,
        'title': 'Test task',
        'analysis': 'A test',
        'prerequisites': [],
        'steps': [],
    })
