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
    'MemoryHints',
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
