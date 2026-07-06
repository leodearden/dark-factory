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

import json
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    'BeforeDone',
    'DoneProvenance',
    'ExternalDep',
    'MemoryHints',
    'RetryLedger',
    'SchemaWarning',
    'TaskMetadata',
    'apply_migrations',
    'parse_metadata',
    'register_metadata_submodel',
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

    @model_validator(mode='after')
    def _deterministic_invariants(self) -> TaskMetadata:
        if (
            self.task_kind == 'deterministic'
            and self.before_done is None
            and not self.always_escalates
        ):
            raise ValueError('deterministic task requires before_done or always_escalates')
        if self.before_done is not None and self.task_kind != 'deterministic':
            raise ValueError('before_done is only valid on deterministic tasks')
        return self


# W10 extension point: lets a later task register its own typed metadata
# sub-model (e.g. a future "deploy_state" slice) without this module having
# to know about it in advance. Keyed by the top-level metadata field name.
_SUBMODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_metadata_submodel(key: str, model: type[BaseModel]) -> None:
    """Register ``model`` as the typed shape for ``metadata[key]``.

    Idempotent when re-registering the *same* model object under the same
    key (e.g. a module reloaded/imported twice). Raises ``ValueError`` when a
    *different* model is registered for a key that already has one — this is
    a loud, fail-fast conflict intended to surface at import time.
    """

    existing = _SUBMODEL_REGISTRY.get(key)
    if existing is not None and existing is not model:
        raise ValueError(f'metadata sub-model already registered for {key!r}')
    _SUBMODEL_REGISTRY[key] = model


def _normalize_legacy_memory_hints(value: object) -> object:
    """Coerce a legacy list-of-dicts ``memory_hints`` value to canonical dict shape.

    Ported verbatim from fused-memory's ``_normalize_legacy_memory_hints_value``
    (``sqlite_task_backend.py:1320``) so ~2100 historical tasks upgrade
    identically: ``entities``/``queries`` are deduped *independently*
    (first-seen order, not per-pair), non-dict items and entries with
    missing/empty/non-string ``entity``/``query`` are skipped, and any
    non-list input (already-canonical dict, scalar, None) passes through
    unchanged.
    """
    if not isinstance(value, list):
        return value
    entities: list[str] = []
    queries: list[str] = []
    seen_entities: set[str] = set()
    seen_queries: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        entity = item.get('entity')
        if isinstance(entity, str) and entity and entity not in seen_entities:
            entities.append(entity)
            seen_entities.add(entity)
        query = item.get('query')
        if isinstance(query, str) and query and query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)
    return {'entities': entities, 'queries': queries}


def _migrate_v0_to_v1(blob: dict) -> dict:
    """v0->v1: normalize a legacy list-shaped ``memory_hints`` and stamp ``schema_version=1``."""
    upgraded = dict(blob)
    if 'memory_hints' in upgraded:
        upgraded['memory_hints'] = _normalize_legacy_memory_hints(upgraded['memory_hints'])
    upgraded['schema_version'] = 1
    return upgraded


# Versioned migration registry, keyed by SOURCE schema_version. apply_migrations
# chains through this until the blob's schema_version has no registered migration
# (i.e. it is current).
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    0: _migrate_v0_to_v1,
}


def apply_migrations(blob: dict) -> dict:
    """Upgrade *blob* through the registered migration chain to the current schema version.

    Non-mutating: the caller's ``blob`` is never modified in place — each
    step operates on (and returns) its own shallow copy.
    """
    upgraded = dict(blob)
    current = upgraded.get('schema_version', 0)
    while current in _MIGRATIONS:
        upgraded = _MIGRATIONS[current](upgraded)
        new_version = upgraded.get('schema_version', current)
        if new_version == current:
            # Safety net: a misconfigured migration that forgets to bump
            # schema_version would otherwise loop forever.
            break
        current = new_version
    return upgraded


class SchemaWarning(BaseModel):
    """A single non-fatal :func:`parse_metadata` finding.

    ``parse_metadata`` does not know which task a blob belongs to — the
    fused-memory backend attaches ``task_id`` when it emits the
    ``task_metadata.schema_warning`` census line (PRD §1/§5).
    """

    field: str
    code: str
    message: str


# Sentinel `field` for a SchemaWarning that cannot be pinned to one top-level
# key: a whole-blob JSON parse failure, or a whole-model cross-field
# invariant violation (PRD §5 `_deterministic_invariants`, `loc == ()`).
_WHOLE_METADATA_FIELD = '<metadata>'


def parse_metadata(
    blob: dict | str | None,
    *,
    direction: Literal['read', 'write'],
    enforce: bool = False,
) -> tuple[TaskMetadata, list[SchemaWarning]]:
    """Parse a task's ``metadata`` JSON blob into a validated :class:`TaskMetadata`.

    The single parser for the ``metadata`` column (PRD §5), replacing eight
    ad-hoc parsers. ``direction``/``enforce`` govern the failure policy for
    malformed input — unparseable JSON, invalid typed sub-models, unknown
    top-level keys — which is layered on top of this happy-path core:

    * ``blob is None`` -> an empty, all-defaults ``TaskMetadata`` (benign-absent).
    * ``blob`` is a JSON string -> ``json.loads`` then handled as a dict.
    * ``blob`` is a dict -> migrated (:func:`apply_migrations`), any
      registered sub-model slice (:data:`_SUBMODEL_REGISTRY`) present in it
      is validated and swapped in as a typed instance, then the whole thing
      is validated as :class:`TaskMetadata`.

    Failure policy (never the old silent-``{}`` discard — I4): ``write`` with
    ``enforce=True`` raises (``ValueError``/``ValidationError``) on malformed
    input; every other case (``read``, or ``write`` with ``enforce=False``)
    emits a :class:`SchemaWarning` and returns a best-effort model that
    retains the raw offending value so round-trip preservation (I1) holds.
    Unknown top-level keys are never rejected — only warned (``x_``-prefixed
    keys are the silent forward-compat namespace).
    """
    if blob is None:
        return TaskMetadata(), []

    warnings: list[SchemaWarning] = []

    # A dedicated, non-Optional-and-non-str local: reassigning the `blob`
    # parameter itself (declared `dict | str | None`) would reset pyright's
    # narrowed type back to the full declared union on every assignment
    # (json.loads returns `Any`, and assigning `Any` to a declared-type
    # variable narrows to the *declared* type, not to `dict`).
    parsed: dict
    if isinstance(blob, str):
        try:
            parsed = json.loads(blob)
        except ValueError as exc:
            if direction == 'write' and enforce:
                raise
            warnings.append(
                SchemaWarning(
                    field=_WHOLE_METADATA_FIELD,
                    code='unparseable_json',
                    message=f'metadata is not valid JSON: {exc}',
                )
            )
            return TaskMetadata(), warnings
    else:
        parsed = blob

    parsed = apply_migrations(parsed)

    for key, submodel in _SUBMODEL_REGISTRY.items():
        if key not in parsed:
            continue
        try:
            parsed = {**parsed, key: submodel(**parsed[key])}
        except ValidationError as exc:
            if direction == 'write' and enforce:
                raise
            warnings.append(SchemaWarning(field=key, code='invalid_submodel', message=str(exc)))

    try:
        model = TaskMetadata(**parsed)
    except ValidationError as exc:
        if direction == 'write' and enforce:
            raise
        # loc[0] is only ever a top-level dict key (hence `str`) for a
        # per-field error on this model; a whole-model error (e.g. the
        # deterministic invariant) has an empty loc, filtered out by `if loc`.
        offending = {
            loc[0]
            for loc in (err['loc'] for err in exc.errors())
            if loc and isinstance(loc[0], str)
        }
        if offending:
            remainder = {k: v for k, v in parsed.items() if k not in offending}
            try:
                model = TaskMetadata(**remainder)
            except ValidationError:
                # Popping the offending keys wasn't sufficient (e.g. a second,
                # independent invariant still trips) — fall back to a fully
                # raw, unvalidated model rather than raising.
                model = TaskMetadata.model_construct(**parsed)
            else:
                # Reattach the raw values via model_extra so model_dump
                # re-emits them verbatim (I1), rather than losing them to
                # the popped-and-revalidated remainder.
                extra = model.__pydantic_extra__
                if extra is not None:
                    for key in offending:
                        extra[key] = parsed[key]
            for key in offending:
                warnings.append(SchemaWarning(field=key, code='invalid_field', message=str(exc)))
        else:
            # A whole-model error (e.g. the deterministic cross-field
            # invariant) has no single offending key to pop; skip validation
            # entirely but keep every given value.
            model = TaskMetadata.model_construct(**parsed)
            warnings.append(
                SchemaWarning(
                    field=_WHOLE_METADATA_FIELD, code='invalid_metadata', message=str(exc)
                )
            )

    known_fields = set(TaskMetadata.model_fields) | set(_SUBMODEL_REGISTRY)
    for key in parsed:
        if key in known_fields or key.startswith('x_'):
            continue
        warnings.append(
            SchemaWarning(
                field=key, code='unknown_key', message=f'unrecognised metadata key {key!r}'
            )
        )

    return model, warnings
