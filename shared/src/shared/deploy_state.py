"""DeployState — the shared-visible W3 metadata sub-model for deploy lifecycle state.

PRD ``plans/harness-supervision-prd.md`` §5.2 (DS-1..4, stream W10-ε). The
schema lives here — shared-visible — rather than in
``orchestrator/deploy_state.py``, because the fused-memory process calls
``shared.task_metadata.parse_metadata`` but cannot import ``orchestrator``:
both processes must import the SAME ``DeployState`` class to populate their
own per-process ``_SUBMODEL_REGISTRY`` (see :func:`register_metadata_submodel`
below), or ``parse_metadata`` warns ``unknown_key`` on every deploy write —
breaking W3's θ2 enforce-flip gate (task 2184). The orchestrator-domain
state machine (``_LEGAL`` transition table + ``enforce_transition``) stays in
``orchestrator/src/orchestrator/deploy_state.py`` per the §5.2 seam note;
that module re-exports the schema defined here.

Registration is a side-effect of importing *this* module — not of importing
``shared.task_metadata`` itself. Auto-registering at ``shared.task_metadata``
import time would break dep 2158's ``TestSubmodelRegistry``
(``shared/tests/test_task_metadata.py``): those tests register a stub
``DeployState`` under ``'deploy_state'`` and their autouse fixture
snapshots-but-does-not-clear the registry, so a pre-registered real
``DeployState`` would make their stub registration raise a conflict
``ValueError``. Each process instead registers the real model by importing
this module (the orchestrator via ``orchestrator.deploy_state``;
fused-memory via a side-effect import in
``fused_memory.middleware.task_interceptor``).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from shared.task_metadata import register_metadata_submodel

__all__ = [
    'DeployPhase',
    'DeployState',
    'VerifyBaseline',
]


class DeployPhase(enum.StrEnum):
    """Deploy-lifecycle phase vocabulary (PRD §5.2 chart).

    Mirrors the ``shared.task_statuses.TaskStatus`` convention: genuine
    ``str`` members so equality against a plain string (e.g. a JSON metadata
    blob's raw ``phase`` value) holds without an explicit ``.value``.
    """

    SCHEDULED = 'scheduled'
    RAN = 'ran'
    VERIFIED = 'verified'
    FAILED = 'failed'
    ESCALATED = 'escalated'
    DONE = 'done'


class VerifyBaseline(BaseModel):
    """``deploy_state.verify_baseline`` — the DS-3 persisted fresh-PID baseline.

    Snake_case persistence of ``systemd_inspect.inspect_systemd_unit``'s
    ``ActiveEnterTimestampMonotonic``/``MainPID`` fields (PRD §5.1
    FreshPidVerify naming).
    """

    model_config = ConfigDict(extra='allow')

    active_enter_timestamp_monotonic: int
    main_pid: int


class DeployState(BaseModel):
    """``metadata.deploy_state`` — the DS-1 typed deploy-lifecycle state slice.

    ``extra='allow'`` so a future field survives round-trip through the
    metadata JSON blob untouched (mirrors ``shared.task_metadata``'s other
    sub-models: ``BeforeDone``, ``DoneProvenance``, ``RetryLedger``).
    """

    model_config = ConfigDict(extra='allow')

    phase: DeployPhase
    verify_baseline: VerifyBaseline | None = None
    ran_at: str | None = None
    verified_at: str | None = None
    escalated_at: str | None = None

    @classmethod
    def from_metadata(cls, md: Mapping[str, object]) -> DeployState | None:
        """Extract and validate the ``deploy_state`` slice from a metadata blob.

        Returns ``None`` when the slice is absent (or not a mapping) —
        distinguishes "no deploy state yet" from a phase value, for θ1's
        TaskGroundTruth deterministic branch.
        """
        slice_ = md.get('deploy_state')
        if not isinstance(slice_, dict):
            return None
        return cls(**slice_)

    def to_metadata(self) -> dict:
        """The DS-1 ``update_task(metadata_mode='merge')`` fragment.

        Top-level ``merge`` is shallow, so this always emits the FULL
        accumulated state (no ``exclude_none``) — a partial dump would drop
        previously-written fields on the next merge.
        """
        return {'deploy_state': self.model_dump(mode='json')}


# The one sanctioned shared/ registration call (Open-Q Q2: shared-visible).
# Importing this module — from either process — registers this SAME
# DeployState class into shared.task_metadata's per-process
# _SUBMODEL_REGISTRY, so parse_metadata validates a `deploy_state` slice and
# never emits an `unknown_key` SchemaWarning for it (W3 θ2 gate, task 2184).
# Deliberately NOT done at shared.task_metadata import time — see this
# module's docstring for why (would break dep 2158's TestSubmodelRegistry).
register_metadata_submodel('deploy_state', DeployState)
