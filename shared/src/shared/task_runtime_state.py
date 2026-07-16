"""TaskRuntimeEntry / TaskRuntimeSnapshot — the shared-visible wire contract
for per-task runtime state.

PRD ``plans/dashboard-task-runtime-endpoint-prd.md`` task beta (Open-Q2).
This is the ONE machine-checked contract (INV-1 contracts-machine-checked)
shared by two independent consumers:

- ``escalation.server``'s ``get_task_runtime_state`` MCP tool (task beta,
  this task) projects ``orchestrator.task_runtime.TaskRuntimeState`` (task
  alpha, task 2634) into this model and serializes it over the wire.
- The dashboard join (task gamma) imports this SAME model to decode the
  wire payload — no structurally-mirrored second copy (INV-5
  no-lockstep-duplication). The dashboard cannot import ``orchestrator``
  directly, so this model lives in ``shared/`` rather than alongside the
  ``orchestrator.task_runtime`` dataclass it mirrors.

Honest-failure projection: ``orchestrator.task_runtime.TaskRuntimeState``
reports ``loops``/``attempts``/``phase``/``started`` as ``None`` (plus a
non-empty ``error``) on a per-task artifact READ FAILURE — never a
fabricated ``0`` (structured-facts-at-failure). ``TaskRuntimeEntry`` mirrors
this exactly: ``loops``/``attempts`` are nullable ints, ``phase`` and
``lane_state`` are ``Literal | None`` (machine-checked vocabularies, but
absence is a legal, distinct state from every member), and ``error`` is
carried through unmodified. Coercing ``None`` to ``0``/a fabricated phase
would violate the project's loud-over-silent-degradation norm; dropping the
task would silently lose the failure signal. On the success path,
``loops``/``attempts`` ARE non-negative ints and ``phase`` IS one of the
three enum members — honest-zero and read-failure stay distinguishable on
the wire.

Follows the ``shared/src/shared/deploy_state.py`` precedent: a pydantic
model living in ``shared/`` and imported directly by each consumer
(``from shared.task_runtime_state import ...``), deliberately NOT
re-exported via ``shared/__init__.py`` (outside the enumerated public-API
surface that ``__all__`` / ``test_public_api.py`` track).

Cross-package vocabulary coupling: ``phase``/``lane_state`` are
machine-checked ``Literal`` vocabularies here, but free-form ``str`` on the
``orchestrator.task_runtime.TaskRuntimeState`` source (today's
``_derive_phase``/``_LANE_STATE_MAP`` only ever emit values inside this
vocab, but that is not an enforced invariant). The producer
(``escalation.server.get_task_runtime_state``) is responsible for handling
a future out-of-vocab value gracefully — it degrades that one task to an
honest per-task error entry instead of letting the ``ValidationError``
fail the whole snapshot; see ``escalation.server._project_task_runtime_entry``.
This model itself still raises on construction with an out-of-vocab value
(INV-1) — it does not silently widen the type to catch drift.
"""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    'TaskRuntimeEntry',
    'TaskRuntimeSnapshot',
]


class TaskRuntimeEntry(BaseModel):
    """One task's runtime state, projected for the wire.

    Field-for-field mirror of ``orchestrator.task_runtime.TaskRuntimeState``
    (task alpha, task 2634), with machine-checked ``Literal`` vocabularies
    for ``phase``/``lane_state`` in place of alpha's free-form ``str``.
    """

    task_id: int
    has_worktree: bool
    loops: int | None
    attempts: int | None
    started: str | None
    lane: str | None
    phase: Literal['PLAN', 'EXECUTE', 'DONE'] | None
    lane_state: Literal['assigned', 'quarantined', 'released'] | None
    error: str | None = None


class TaskRuntimeSnapshot(BaseModel):
    """The ``get_task_runtime_state`` tool's full response envelope.

    ``offline`` is always ``False`` as emitted by the server itself — the
    dashboard (task gamma) synthesizes ``True`` client-side when the
    escalation server is unreachable.
    """

    offline: bool = False
    tasks: list[TaskRuntimeEntry] = Field(default_factory=list)
