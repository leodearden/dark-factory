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

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    'BeforeDone',
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
