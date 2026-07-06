"""Versioned cross-process TaskMetadata schema — the single parser for task metadata.

``shared.task_metadata`` is the one schema shared by the fused-memory backend
(writer/validator) and the orchestrator (reader/writer) for the ``metadata``
JSON blob carried on every task.  It replaces eight independent ad-hoc parsers
that had drifted out of lockstep (see ``plans/task-metadata-schema-prd.md``).

Only the model classes and :func:`parse_metadata` below are public; the
module is accessed as a submodule (``shared.task_metadata.X``) and is
deliberately **not** re-exported from ``shared/__init__.py`` (see the PRD's
resolved design decisions — this keeps ``shared/tests/test_public_api.py``'s
strict ``__all__`` union assertion untouched).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    'BeforeDone',
    'DoneProvenance',
    'ExternalDep',
    'MemoryHints',
    'RetryLedger',
    'TaskMetadata',
]


class BeforeDone(BaseModel):
    """``metadata.before_done`` — the deterministic-task pre-done action descriptor.

    Mirrors the structural checks in
    ``deterministic_task_guard._validate_before_done`` (fused-memory); the
    filesystem-level checks (path containment, executable bit) stay at the
    ``submit_task`` guard — this model enforces only the type/shape layer.
    """

    model_config = ConfigDict(extra='allow')

    script: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    timeout_secs: int = Field(gt=0)
    target_unit: str | None = None


class DoneProvenance(BaseModel):
    """``metadata.done_provenance`` — the single valid-kinds declaration (I2).

    ``kind`` is the *only* place the valid-kinds vocabulary is declared;
    fused-memory's ``_VALID_PROVENANCE_KINDS`` is retired in favour of
    importing this model (see PRD §5, I2).
    """

    model_config = ConfigDict(extra='allow')

    kind: Literal[
        'merged',
        'found_on_main',
        'deterministic-deploy',
        'deterministic-deploy-scheduled',
    ]
    commit: str | None = None
    note: str | None = None
    pid: int | None = None
    unit: str | None = None
    active_enter_timestamp: str | None = None

    @model_validator(mode='after')
    def _check_conditional_requirements(self) -> DoneProvenance:
        if self.kind in ('merged', 'found_on_main') and self.commit is None:
            raise ValueError(f'DoneProvenance: commit is required when kind={self.kind!r}.')
        if self.kind == 'found_on_main' and self.note is None:
            raise ValueError("DoneProvenance: note is required when kind='found_on_main'.")
        return self


class MemoryHints(BaseModel):
    """``metadata.memory_hints`` — canonical ``{entities, queries}`` shape.

    Legacy ``[{entity, query}, ...]`` blobs are upgraded to this shape by the
    registered v0->v1 migration (see :func:`apply_migrations`) before
    validation.
    """

    entities: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class ExternalDep(BaseModel):
    """A single ``metadata.external_deps`` entry — ``"project_id:task_id"``.

    Validate/normalise-only (PRD Open Q #4): ``task_id`` stays ``str`` and
    neither field is case- or dash-normalised, so ``parse(s).render() == s``
    always holds. The stricter numeric/positive/dotted-id/lower-and-dash
    normalisation rules stay at the fused-memory backend's
    ``add_dependency`` (``_parse_qualified_dep``); this model only mirrors
    the structural "exactly two non-empty stripped parts" check.
    """

    project_id: str
    task_id: str

    @classmethod
    def parse(cls, wire: str) -> ExternalDep:
        malformed = (
            f'ExternalDep.parse: malformed dependency {wire!r}; expected "project_id:task_id"'
        )
        parts = wire.strip().split(':')
        if len(parts) != 2:
            raise ValueError(malformed)
        project_id, task_id = parts[0].strip(), parts[1].strip()
        if not project_id or not task_id:
            raise ValueError(malformed)
        return cls(project_id=project_id, task_id=task_id)

    def render(self) -> str:
        return f'{self.project_id}:{self.task_id}'


class RetryLedger(BaseModel):
    """``metadata.retry_ledger`` — anti-thrash counters (PRD §5).

    ``extra='allow'`` so a future 9th counter survives round-trip without a
    schema bump.
    """

    model_config = ConfigDict(extra='allow')

    consecutive_no_plan_failures: int = 0
    total_no_plan_failures: int = 0
    last_no_plan_main_sha: str | None = None
    consecutive_infra_resume_failures: int = 0
    last_infra_resume_iteration_count: int = 0
    consecutive_merge_thrash: int = 0
    last_merge_outcome_signature: str | None = None
    merge_first_enqueued_at: str | None = None


class TaskMetadata(BaseModel):
    """The versioned ``metadata`` JSON blob carried on every task (PRD §5).

    ``extra='allow'`` is load-bearing for I1 (round-trip preservation): any
    key this schema does not yet know about — a newer writer's field, a
    caller-private ``x_``-namespaced value — survives untouched through
    :func:`parse_metadata` rather than being silently dropped.
    """

    model_config = ConfigDict(extra='allow')

    schema_version: int = 1
    task_kind: Literal['normal', 'deterministic'] = 'normal'
    always_escalates: bool = False
    before_done: BeforeDone | None = None
    done_provenance: DoneProvenance | None = None
    memory_hints: MemoryHints | None = None
    retry_ledger: RetryLedger | None = None
    # Canonical "project_id:task_id" wire strings — NOT list[ExternalDep].
    # Keeping this a list[str] defers typed-vs-string consumption to the
    # scheduler (PRD Open Q #4); ExternalDep is a parse/render convenience.
    external_deps: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
