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

import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    'HUMAN_CURATOR_GATE_KEY',
    'KNOWN_ROLE_NAMES',
    'BeforeDone',
    'DoneProvenance',
    'ExternalDep',
    'MemoryHints',
    'MergeRetryPending',
    'Milestone',
    'RetryLedger',
    'RoutingDecisionMirror',
    'RoutingState',
    'SchemaWarning',
    'SubmodelCardinality',
    'TaskMetadata',
    'apply_migrations',
    'parse_metadata',
    'register_metadata_submodel',
    'validate_model_overrides',
]


# The shared mirror of the orchestrator's dispatchable role names (PRD
# adaptive-model-routing task ζ, decision 9). fused-memory cannot import
# orchestrator (layering: shared is the base package both depend on), but
# the fused-memory submit_task/update_task guard must still shape-validate
# metadata.model_overrides role-name keys — so this frozenset is the single
# hand-maintained authority both sides read. It is a superset of both
# orchestrator.agents.roles.ROLES (full dispatch role_name keys, e.g.
# 'reviewer_comprehensive') and config.ModelsConfig's collapsed keys (e.g.
# 'reviewer'), so every actually-dispatchable role is always pinnable.
# orchestrator/tests carries a drift-guard test asserting
# set(ROLES) <= KNOWN_ROLE_NAMES (and set(ModelsConfig.model_fields) <=
# KNOWN_ROLE_NAMES) so this mirror can't silently diverge from either real
# authority.
#
# CAVEAT -- collapsed ModelsConfig-only keys are accepted-but-INERT as a
# model_overrides key: resolve_route's layer-1 metadata_override reader
# keys strictly on `inputs.role_name`, which is always the full dispatch
# identity from ROLES (e.g. 'reviewer_comprehensive'), never the collapsed
# config key ('reviewer', 'triage', 'module_tagger' — present here only
# because they are ModelsConfig fields). An override authored under one of
# those three collapsed keys passes this shape guard but will silently
# never match at resolve time. Authors pinning a model override MUST use
# the full role_name (see orchestrator.agents.roles.ROLES), not the
# collapsed config key.
KNOWN_ROLE_NAMES: frozenset[str] = frozenset(
    {
        'architect',
        'implementer',
        'debugger',
        'reviewer',
        'reviewer_comprehensive',
        'merger',
        'steward',
        'triage',
        'module_tagger',
        'deep_reviewer',
        'judge',
        'simple_task',
    }
)


# The human-curator-gate marker (task 3341), read by
# TaskMetadata._deterministic_invariants below and blessed in
# _BLESSED_METADATA_KEYS. It is an ``extra='allow'`` metadata key, NOT a typed
# TaskMetadata field, and is deliberately kept that way (task 3369): a typed
# ``bool`` would COERCE the fail-closed string 'true' to True in pydantic's
# non-strict mode -- silently rewriting an author's value in the one direction
# that makes it look intentional, breaking I1 -- and a concrete default would
# add ``human_curator_gate: false`` noise to every task's model_dump(), the
# same objection already documented for merge_retry_pending.
#
# This is the key's SINGLE definition codebase-wide (exported via __all__):
# orchestrator.deterministic_runner — the only other module that acts on the
# key — imports it from here rather than restating the literal, so the write
# boundary's rejection and the runner's dispatch-time guard cannot drift apart.
HUMAN_CURATOR_GATE_KEY: str = 'human_curator_gate'


def validate_model_overrides(value: object) -> None:
    """Shape-validate a ``metadata.model_overrides`` value (PRD decision 9).

    SHAPE only: ``value`` must be a ``dict`` whose keys are all members of
    :data:`KNOWN_ROLE_NAMES` and whose values are all non-empty ``str``.
    Raises ``ValueError`` (never returns a falsy sentinel) describing the
    offending role/value on the first violation found.

    Deliberately does **not** validate model strings against any allowlist
    — that is the orchestrator resolver's fail-safe job at resolve time
    (an override naming a model outside ``routing.allowed_models`` is
    skipped and recorded in ``RoutingDecision.rejected``, never raised
    here). fused-memory, the sole caller of this shape check at the
    submit/update boundary, does not know the orchestrator's allowlist.
    """
    if not isinstance(value, dict):
        raise ValueError(
            f'model_overrides must be an object mapping role name to model, '
            f'got {type(value).__name__}'
        )
    for role, model in value.items():
        if role not in KNOWN_ROLE_NAMES:
            raise ValueError(
                f'model_overrides: unknown role {role!r}; valid roles are '
                f'{sorted(KNOWN_ROLE_NAMES)}'
            )
        if not isinstance(model, str) or not model:
            raise ValueError(
                f'model_overrides: value for role {role!r} must be a non-empty '
                f'string, got {model!r}'
            )


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
    kind: Literal['deploy', 'predicate'] = 'deploy'


class DoneProvenance(BaseModel):
    """``metadata.done_provenance`` — the single valid-kinds declaration (I2).

    ``kind`` is the *only* place the valid-kinds vocabulary is declared;
    fused-memory's ``_VALID_PROVENANCE_KINDS`` is retired in favour of
    importing this model (see PRD §5, I2).

    ``stamped_at`` (task 3576) is the dedicated stamp-*write* timestamp
    (ISO-8601 UTC) for ``kind='found_on_main'``. It is written SERVER-SIDE
    by fused-memory's ``_validate_done_provenance`` chokepoint and is never
    supplied by a caller — a caller-supplied value is discarded with a
    warning. It exists because ``updatedAt`` is bumped by *any* later write
    to the task (a re-tag, an audit annotation, a dependency edit), so it
    cannot answer "when was this attribution asserted?"; the found_on_main
    soak-gate predicate needs that question answered to distinguish a
    genuinely new spurious stamp from legacy backlog.

    It is deliberately OPTIONAL rather than conditionally-required for
    ``found_on_main`` (i.e. ``_check_conditional_requirements`` is
    deliberately NOT extended) for two reasons: ~193 historical stamps
    predate the field and must keep validating at read time, and — more
    importantly — its ABSENCE is itself load-bearing signal. Because every
    write after task 3576 lands populates it at the chokepoint, a blob
    lacking it provably predates that landing, which is exactly how
    ``fused-memory/scripts/check_found_on_main_spurious_rate.py`` separates
    legacy backlog from new stamps.
    """

    model_config = ConfigDict(extra='allow')

    kind: Literal[
        'merged',
        'found_on_main',
        'deterministic-deploy',
        'deterministic-deploy-scheduled',
        'deterministic-gate',
        'deterministic-milestone',
        'operational-verified',
    ]
    commit: str | None = None
    note: str | None = None
    pid: int | None = None
    unit: str | None = None
    active_enter_timestamp: str | None = None
    escalation_id: str | None = None
    # Server-written; see the class docstring. Optional by design — absence
    # means "predates task 3576", which the soak-gate predicate relies on.
    stamped_at: str | None = None

    @model_validator(mode='after')
    def _check_conditional_requirements(self) -> DoneProvenance:
        if self.kind in ('merged', 'found_on_main') and self.commit is None:
            raise ValueError(f'DoneProvenance: commit is required when kind={self.kind!r}.')
        if self.kind == 'found_on_main' and self.note is None:
            raise ValueError("DoneProvenance: note is required when kind='found_on_main'.")
        if self.kind == 'operational-verified':
            if self.escalation_id is None:
                raise ValueError(
                    "DoneProvenance: escalation_id is required when kind='operational-verified'."
                )
            if self.note is None:
                raise ValueError(
                    "DoneProvenance: note is required when kind='operational-verified'."
                )
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


# Regexes used by RetryLedger.normalize_cause_hint — compiled once at module
# level. Order of application: ANSI first (so coloured file:line refs are
# cleaned before the file:line pattern matches them), then file:line, then
# whitespace. Ported verbatim from the orchestrator (formerly
# orchestrator.workflow._ANSI_ESCAPE_RE et al.) so this model is the single
# signature-keying authority (PRD §5 / task 2172).
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')
_FILE_LINE_RE = re.compile(
    r'\b[\w./\\-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|cpp|c|h|sh|md|yaml|yml|json|toml)'
    r':\d+(:\d+)?\b'
)
_WHITESPACE_RE = re.compile(r'\s+')


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

    @staticmethod
    def normalize_cause_hint(hint: str | None) -> str:
        """Normalise a VerifyResult cause_hint for equality comparison.

        Strips ANSI colour escape sequences, removes file:line (and
        file:line:col) numeric tails, collapses contiguous whitespace to a
        single space, lowercases, and strips leading/trailing whitespace.

        Returns an empty string for empty or None input — never raises.

        Used by the verify-loop and merge-outcome anti-thrash guards to
        detect consecutive identical failures even when line numbers shift
        between retries. This is the single signature-keying authority;
        ``orchestrator.workflow._normalize_cause_hint`` is a thin delegator
        kept for backward-compatible imports.
        """
        if not hint:
            return ''
        # 1. Strip ANSI colour codes (e.g. \x1b[31m...\x1b[0m) first so that
        #    coloured file:line references like \x1b[31mfoo.py:42\x1b[0m
        #    become plain foo.py:42 before the file:line pattern runs.
        result = _ANSI_ESCAPE_RE.sub('', hint)
        # 2. Strip file:line and file:line:col numeric tails
        #    (e.g. "tests/test_x.py:42" or "foo.py:42:7").
        result = _FILE_LINE_RE.sub('', result)
        # 3. Collapse contiguous whitespace (spaces, tabs, newlines) to one space.
        result = _WHITESPACE_RE.sub(' ', result)
        # 4. Lowercase and strip.
        return result.lower().strip()

    @staticmethod
    def compute_merge_outcome_signature(
        category: str | None,
        cause_hint: str | None,
        fallback_reason: str = '',
    ) -> str:
        """Compute a 16-hex-char sha-independent outcome signature from explicit fields.

        Keys on (category, normalised cause_hint) when either field is set;
        falls back to sha256(normalised_reason) when both are empty — same
        logic as ``TaskWorkflow._merge_outcome_signature()``, which delegates
        here (via ``orchestrator.workflow._compute_merge_outcome_signature``)
        so the hash algorithm stays in one place.
        """
        cat = category or ''
        hint = cause_hint or ''
        if cat or hint:
            basis = (cat + '\x1f' + RetryLedger.normalize_cause_hint(hint)).encode('utf-8')
        else:
            basis = RetryLedger.normalize_cause_hint(fallback_reason or '').encode('utf-8')
        return hashlib.sha256(basis).hexdigest()[:16]


class Milestone(BaseModel):
    """``metadata.milestone`` — a dated or delayed milestone spec (PRD §6.1).

    ``mode`` discriminates between the two mutually exclusive shapes: a
    ``'dated'`` milestone fires at an explicit ISO-8601 timestamp (``at``);
    a ``'delayed'`` milestone fires ``after_secs`` seconds after some
    reference point. The two field sets are an *iff* — a ``'dated'``
    milestone must not also carry ``after_secs``, and a ``'delayed'``
    milestone must not also carry ``at``.
    """

    model_config = ConfigDict(extra='allow')

    mode: Literal['dated', 'delayed']
    at: str | None = None
    after_secs: int | None = None

    @model_validator(mode='after')
    def _check_mode_fields(self) -> Milestone:
        if self.mode == 'dated':
            if self.at is None:
                raise ValueError("Milestone: at is required when mode='dated'.")
            try:
                datetime.fromisoformat(self.at)
            except ValueError as exc:
                raise ValueError(
                    f'Milestone: at={self.at!r} is not a valid ISO-8601 datetime: {exc}'
                ) from exc
            if self.after_secs is not None:
                raise ValueError("Milestone: after_secs must not be set when mode='dated'.")
        else:
            if self.after_secs is None or self.after_secs <= 0:
                raise ValueError(
                    "Milestone: after_secs is required and must be > 0 when mode='delayed'."
                )
            if self.at is not None:
                raise ValueError("Milestone: at must not be set when mode='delayed'.")
        return self


class RoutingDecisionMirror(BaseModel):
    """One resolved routing decision for a single LLM invocation (PRD γ).

    Mirrors the fields of ``orchestrator.routing.RoutingDecision`` (task ε)
    so that dataclass can be swapped in as this model's source with no
    schema change (PRD invariant 7). ``extra='allow'`` so any of ε's
    additional fields survive round-trip before this model is updated to
    know about them by name.
    """

    model_config = ConfigDict(extra='allow')

    role: str
    model: str
    effort: str
    budget_usd: float
    max_turns: int
    source_layer: str
    rule_id: str | None = None
    rejected: list[str] = Field(default_factory=list)
    routing_tier: int = 0
    decided_at: str | None = None


# Bounded history length for RoutingState.with_decision (PRD Open-Q 3).
_ROUTING_HISTORY_MAX = 5


class RoutingState(BaseModel):
    """``metadata.routing`` — the LATEST routing decision + bounded history (PRD γ).

    ``latest`` mirrors the most recent per-invocation routing resolution;
    ``history`` retains up to :data:`_ROUTING_HISTORY_MAX` of the most
    recent decisions (oldest dropped first — see :meth:`with_decision`).
    ``routing_tier``/``simple_saturated`` are counter/flag storage stamped
    by later tasks (μ/ν) — this task only provides the typed slice, so
    their defaults (0 / False) are inert until then.
    """

    model_config = ConfigDict(extra='allow')

    latest: RoutingDecisionMirror | None = None
    history: list[RoutingDecisionMirror] = Field(default_factory=list)
    routing_tier: int = 0
    simple_saturated: bool = False

    def with_decision(
        self,
        decision: RoutingDecisionMirror,
        *,
        history_max: int = _ROUTING_HISTORY_MAX,
    ) -> RoutingState:
        """Return a new RoutingState with ``decision`` as latest, appended to history.

        Bounds ``history`` to the newest ``history_max`` entries (oldest
        dropped first, order preserved). ``routing_tier``, ``simple_saturated``,
        and any ``extra`` fields are preserved unchanged (``model_copy`` update
        semantics) — this is a pure, unit-testable transform with no implicit
        trimming of unrelated state.
        """
        return self.model_copy(
            update={
                'latest': decision,
                'history': [*self.history, decision][-history_max:],
            }
        )

    @classmethod
    def from_metadata(cls, metadata: dict | None) -> RoutingState:
        """Safely reconstruct a :class:`RoutingState` from ``metadata['routing']``.

        Mirrors ``orchestrator.workflow._build_retry_ledger``'s tolerance: a
        missing/None ``metadata``, a missing/non-dict ``routing`` key, or a
        dict that fails validation all degrade to a fresh default
        ``RoutingState()`` rather than raising — routing telemetry must
        never block or crash a caller.
        """
        if not isinstance(metadata, dict):
            return cls()
        raw = metadata.get('routing')
        if not isinstance(raw, dict):
            return cls()
        try:
            return cls(**raw)
        except (ValidationError, TypeError):
            return cls()


class MergeRetryPending(BaseModel):
    """``metadata.merge_retry_pending`` — a durable merge-phase resume obligation (task 2795).

    Stamped by the orchestrator when a merge-phase escalation is resolved via
    ``resume``: the in-place merge retry that ``_requeue`` performs is
    otherwise IN-RAM ONLY (the task stays ``in-progress`` and its merge-queue
    entry was already finalised), so a restart mid-retry silently loses the
    "this task owes a merge resubmission" obligation (Reify 5166). Persisting
    ``{branch_head, base_sha, resolved_at}`` lets the resume guard in
    ``_drive`` reconstruct that obligation on re-dispatch and jump straight
    back to the merge phase when the post-rebase worktree HEAD still equals
    ``branch_head``.

    ``extra='allow'`` matches the milestone/routing precedent so a newer
    writer's field survives round-trip untouched (I1).
    """

    model_config = ConfigDict(extra='allow')

    branch_head: str
    base_sha: str
    resolved_at: str


class TaskMetadata(BaseModel):
    """The versioned ``metadata`` JSON blob carried on every task (PRD §5).

    ``extra='allow'`` is load-bearing for I1 (round-trip preservation): any
    key this schema does not yet know about — a newer writer's field, a
    caller-private ``x_``-namespaced value — survives untouched through
    :func:`parse_metadata` rather than being silently dropped.

    :meth:`_deterministic_invariants`' curator-gate clause is the *write-time*
    counterpart to ``DeterministicRunner``'s dispatch WARNING (task 3341):
    that WARNING fires when a record carrying both markers has already reached
    the runner, whereas this rejects the record at the ``submit_task`` /
    ``update_task`` boundary so it never lands (task 3369). The runner's
    WARNING is deliberately retained as a defence-in-depth backstop for
    records that did not pass through that boundary.
    """

    model_config = ConfigDict(extra='allow')

    schema_version: int = 2
    task_kind: Literal['normal', 'deterministic'] = 'normal'
    always_escalates: bool = False
    # Orthogonal discriminator, NOT an enum split of execution_class (PRD
    # operational-ask-routing decision 1); meaningful only when
    # execution_class='operational' (consumed by task β), recorded-but-inert
    # otherwise. Mirrors execution_class's INTENT but, unlike execution_class
    # (validated only by a fused-memory guard conditional on recon-stage
    # caller identity — logic a pydantic field validator cannot express),
    # operational_mode's {gate,llm}-or-absent rule is caller-independent and
    # fully expressible as a plain typed Literal field here. The concrete
    # 'gate' default implements absent≡gate while avoiding the None-valued
    # model_dump noise a `| None = None` field would add to every task.
    operational_mode: Literal['gate', 'llm'] = 'gate'
    before_done: BeforeDone | None = None
    done_provenance: DoneProvenance | None = None
    memory_hints: MemoryHints | None = None
    retry_ledger: RetryLedger | None = None
    # Canonical "project_id:task_id" wire strings — NOT list[ExternalDep].
    # Keeping this a list[str] defers typed-vs-string consumption to the
    # scheduler (PRD Open Q #4); ExternalDep is a parse/render convenience.
    external_deps: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    # role_name -> model string (PRD adaptive-model-routing task ζ). A plain
    # typed field mirroring external_deps: list[str] -- role-name/value
    # shape validation is NOT duplicated here as a model_validator; it
    # stays single-sourced in validate_model_overrides / the fused-memory
    # submit_task/update_task guard that delegates to it (decision 9).
    model_overrides: dict[str, str] = Field(default_factory=dict)

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
        # Read the marker from model_extra, NOT getattr: it is an extra='allow'
        # key (see HUMAN_CURATOR_GATE_KEY), and pydantic v2 populates
        # __pydantic_extra__ before mode='after' validators run. `or {}` keeps
        # the clause total if model_extra is ever None.
        #
        # LAST clause deliberately: pydantic stops at the first raise, so a
        # blob that is also malformed more fundamentally (e.g. task_kind
        # 'normal' with a before_done) still reports that error instead of
        # this narrower one.
        #
        # Plain TRUTHINESS, not `is True` — same fail-CLOSED posture as
        # orchestrator.deterministic_runner._is_human_curator_gate, and for the
        # same reason: this is a SAFETY marker whose false NEGATIVE is the
        # expensive direction. A truthy-but-not-True value (the string 'true'
        # from a hand edit or a JSON round-trip) must still be rejected here,
        # because the runner's own guard structurally cannot catch it on the
        # act-then-ask path.
        # Pinned by test_truthy_but_not_true_curator_marker_still_rejected.
        if self.before_done is not None and (self.model_extra or {}).get(HUMAN_CURATOR_GATE_KEY):
            raise ValueError(
                'human_curator_gate is only valid on a pure gate: a curator gate '
                'declares that only a human content judgement closes this task, '
                'while before_done is a machine step that closes it. Drop '
                'before_done to make this a real curator gate, or drop the marker '
                'if the machine step is what closes the task.'
            )
        return self


# W10 extension point: lets a later task register its own typed metadata
# sub-model (e.g. a future "deploy_state" slice) without this module having
# to know about it in advance. Keyed by the top-level metadata field name.
_SUBMODEL_REGISTRY: dict[str, type[BaseModel]] = {}

# The declared SHAPE of a registered slice's value: a single mapping
# (``'dict'``) or a list of mappings (``'list'``).
SubmodelCardinality = Literal['dict', 'list']

# Cardinality lives in a PARALLEL dict rather than widening
# _SUBMODEL_REGISTRY's value to a (model, cardinality) pair: six assertions
# across four packages do `_SUBMODEL_REGISTRY[key] is SomeModel` identity
# checks, and widening the value type would break all of them for zero
# functional gain. The two dicts have exactly ONE writer
# (register_metadata_submodel) so they cannot drift in normal use, and they
# degrade in the SAFE direction if they ever do — see the read note in
# register_metadata_submodel's docstring.
_SUBMODEL_CARDINALITY: dict[str, SubmodelCardinality] = {}


def register_metadata_submodel(
    key: str,
    model: type[BaseModel],
    *,
    cardinality: SubmodelCardinality = 'dict',
) -> None:
    """Register ``model`` as the typed shape for ``metadata[key]``.

    ``cardinality`` declares the shape of the slice's VALUE: ``'dict'`` means
    it must be a single mapping, ``'list'`` a list of mappings. It is enforced
    by :func:`parse_metadata`, which emits a ``wrong_cardinality``
    :class:`SchemaWarning` (fatal under ``write``+``enforce``) for a value of
    the other shape.

    ``'dict'`` is the DEFAULT because it is fail-closed: it restores the
    behavior that held before parse_metadata grew its list branch, so an
    existing or future dict-shaped registrant needs no change and only a
    genuinely list-valued slice (currently just ``delivered_checks``) opts in.
    An undeclared key therefore gets the STRICT shape, whose failure mode is a
    spurious warning on a genuinely list-valued slice — loud and immediately
    fixed — rather than a silently-accepted malformed one (task 4142).

    A key present in ``_SUBMODEL_REGISTRY`` but missing from
    ``_SUBMODEL_CARDINALITY`` reads as ``'dict'``: the two dicts have one
    writer, so they only desync if something reaches past this function, and
    that desync degrades safely (parse_metadata iterates the registry, so a
    stale cardinality entry for an absent key is never read).

    Idempotent when re-registering under the same key, provided the repeat
    agrees on BOTH the model object and the cardinality (e.g. a module
    reloaded/imported twice). Raises ``ValueError`` when a *different* model,
    or a different cardinality, is registered for a key that already has one
    — a loud, fail-fast conflict intended to surface at import time. Both
    checks run BEFORE either dict is written, so a rejected call leaves the
    registry and the cardinality map untouched (a partial write is the one
    way the parallel-dict design could genuinely desync). Cardinality is
    immutable for the same reason the model is: registration is a per-process,
    import-order-driven side effect, so a silent last-writer-wins would make
    the enforced shape depend on which module imported first.

    Registry keys are OWNED by the module that registers them. Tests must
    register test-only keys (``<name>_stub``) — never a key a production
    module registers — or a cross-package pytest co-run pre-registers the real
    model and the conflict raise below fires spuriously (task 3352).
    ``shared/tests/test_task_metadata.py`` enforces that convention in its
    autouse fixture's teardown, which asserts every key a test added ends in
    ``_stub``.
    """

    # Read both dicts and run every check before either assignment: a raise
    # partway through would leave the registry and the cardinality map
    # desynced, which is precisely the failure the parallel-dict design has
    # to avoid.
    existing = _SUBMODEL_REGISTRY.get(key)
    existing_cardinality = _SUBMODEL_CARDINALITY.get(key)
    if existing is not None and existing is not model:
        raise ValueError(f'metadata sub-model already registered for {key!r}')
    if existing is not None and existing_cardinality != cardinality:
        raise ValueError(
            f'metadata sub-model for {key!r} is already registered with '
            f'cardinality {existing_cardinality!r}; cannot re-register it as '
            f'{cardinality!r} (a key\'s declared shape is immutable — '
            'registration is import-order-driven, so a silent overwrite would '
            'make the enforced shape depend on which module imported first)'
        )
    _SUBMODEL_REGISTRY[key] = model
    _SUBMODEL_CARDINALITY[key] = cardinality


# Milestone is the first real W10 registrant: registering at module-import
# time (rather than lazily) guarantees the 'milestone' slice is validated
# and typed before any of parse_metadata's many callers across packages run.
#
# cardinality='dict' is stated explicitly on all three registrations below
# even though it is the default: these are the load-bearing declarations the
# task-4142 shape gate exists to make legible, and an explicit call site is
# immune to a future flip of the default.
register_metadata_submodel('milestone', Milestone, cardinality='dict')

# routing (PRD γ, task 2533): registered the same way so 'routing' lands in
# known_fields (no unknown_key census warning) and every parse_metadata
# caller gets a validated, typed RoutingState slice.
register_metadata_submodel('routing', RoutingState, cardinality='dict')

# merge_retry_pending (task 2795): registered like milestone/routing so the
# orchestrator's durable merge-phase-resume stamp lands in known_fields (no
# unknown_key census warning) and is typed/validated at the fused-memory
# write boundary — while, as a registered sub-model rather than an optional
# `| None = None` field, staying absent from model_dump() when unset (no
# None-noise on every task).
register_metadata_submodel('merge_retry_pending', MergeRetryPending, cardinality='dict')


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


# RetryLedger fields that some legacy blobs (pre-2172 placement) still carry
# as TOP-LEVEL metadata keys instead of nested under metadata.retry_ledger.
_LEGACY_RETRY_LEDGER_COUNTER_KEYS = (
    'consecutive_infra_resume_failures',
    'last_infra_resume_iteration_count',
)


def _migrate_v1_to_v2(blob: dict) -> dict:
    """v1->v2: lift legacy top-level infra-resume counters into ``retry_ledger``.

    ``consecutive_infra_resume_failures`` / ``last_infra_resume_iteration_count``
    are :class:`RetryLedger` fields, but some legacy blobs still carry them
    as top-level metadata keys (the live orchestrator writer already nests
    them correctly; this migration only repairs old data at parse-time).

    If ``retry_ledger`` is absent, any present legacy top-level counters are
    popped and merged into a new ``retry_ledger`` dict. If ``retry_ledger``
    is already a dict, it is ALWAYS copied — never the same object as
    ``blob['retry_ledger']``, even when there are no top-level counters to
    lift — and any present counters are merged in, with an existing nested
    value winning on key conflict. If ``retry_ledger`` is present but not a
    dict (already-malformed data), nothing is lifted or copied — the
    top-level counters are left as-is and the existing ``invalid_field``
    warning path (in :func:`parse_metadata`) handles the malformed value.
    Always stamps ``schema_version=2``. Non-mutating overall, mirroring
    :func:`_migrate_v0_to_v1`: ``blob`` itself is never modified in place.
    """
    upgraded = dict(blob)
    present = {key: upgraded[key] for key in _LEGACY_RETRY_LEDGER_COUNTER_KEYS if key in upgraded}

    existing_ledger = upgraded.get('retry_ledger')
    if isinstance(existing_ledger, dict):
        # Always copy, even when `present` is empty: the returned ledger
        # must never be the same object as the caller's nested dict, so a
        # later in-place mutation of it can never reach back into `blob`.
        new_ledger = dict(existing_ledger)
        for key, value in present.items():
            new_ledger.setdefault(key, value)
            del upgraded[key]
        upgraded['retry_ledger'] = new_ledger
    elif present and existing_ledger is None:
        new_ledger = dict(present)
        for key in present:
            del upgraded[key]
        upgraded['retry_ledger'] = new_ledger
    # else: retry_ledger is present but not a dict (malformed) -- leave
    # everything untouched; no lift, no copy.

    upgraded['schema_version'] = 2
    return upgraded


# Versioned migration registry, keyed by SOURCE schema_version. apply_migrations
# chains through this until the blob's schema_version has no registered migration
# (i.e. it is current).
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    0: _migrate_v0_to_v1,
    1: _migrate_v1_to_v2,
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


# Tier-A: the 39 load-bearing conventional metadata keys that real writers
# (orchestrator, curator, DeterministicRunner, escalation flows) already
# depend on but that are not (yet) typed TaskMetadata fields. Skipped in
# parse_metadata's unknown-key scan below so a deliberate, documented
# convention doesn't manufacture unknown_key census noise — extra='allow'
# still preserves each value byte-for-value (I1). None of these collide with
# TaskMetadata.model_fields or _SUBMODEL_REGISTRY (only 'milestone' is
# currently registered there). gate_escalated_at / before_done_ran_at /
# before_done_verified_at / before_done_verified_pid are the
# DeterministicRunner's own stamps (CLAUDE.md "Deterministic task kind").
#
# Tier-B alias-drift keys (prd/prd_ref/prd_leaf, inv, related_task*) and
# Tier-C ad-hoc/timestamped one-off keys are deliberately NOT included here
# — they keep emitting unknown_key as a greppable drift signal; see
# CLAUDE.md "Task metadata vocabulary & census" for the documented
# consolidation convention.
_BLESSED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        'source',
        'modules',
        'spawn_context',
        'complexity',
        'force_full_path',
        'branch_base_sha',
        '_causation_id',
        'dry_run_proposals',
        'reblock_guard',
        'agent_id',
        'escalation_id',
        'suggestion_hash',
        'prd_path',
        'prd_task_label',
        'user_observable_signal',
        'consumer_ref',
        'substrate_confirmed',
        'human_decomposed',
        'grammar_confirmed',
        'invariants',
        'optimistic_path',
        'capability_manifest',
        # AUTOMATED task-curator combine flow (fused_memory.task_interceptor).
        # Distinct from the HUMAN content curator of the `human_curator_*` keys
        # below — two unrelated actors, deliberately not sharing a prefix.
        'curator_action',
        'curator_justification',
        'combined_at',
        'gate_escalated_at',
        'before_done_ran_at',
        'before_done_verified_at',
        'before_done_verified_pid',
        'files_tagged_at',
        'origin_finding_id',
        'spawned_from',
        'program',
        'program_stream',
        'stream',
        # Cross-repo deliverable marker (task 3004): set by the fused-memory submit
        # path when a task's metadata.files are ALL owned by one other registered
        # project, read by the orchestrator pre-merge narrowing gate (routes to
        # OutcomeKind.plan_files_cross_repo instead of flagging 'files not touched').
        'cross_repo',
        'cross_repo_project',
        # Human-curator gate contract (task 3341): `human_curator_gate` marks a
        # pure deterministic gate whose resolution requires human CONTENT
        # adjudication, not merely a closed escalation record;
        # `human_curator_adjudicated_at` is the ISO-8601 stamp that proves the
        # per-entry review happened. Both are read by DeterministicRunner's
        # pure-gate resume guard, which refuses to drive such a task to done
        # when the marker is set and the stamp is absent or not a non-empty
        # string (task 3181 is the incident: an auto-resolved gate escalation
        # was treated as proof the curator work had been done).  The stamp
        # carries the `human_curator_` prefix rather than the bare `curator_`
        # one above precisely so the human content curator is not conflated
        # with the automated task curator.
        HUMAN_CURATOR_GATE_KEY,
        'human_curator_adjudicated_at',
        # Orchestrator block-stamp (task 3697): written by workflow.py
        # `_mark_blocked` on every block, read by agents/briefing.py for the
        # stale-briefing check; 78 tasks carry it (census 2026-08-06). The
        # writer symbol is named because that is what a future reader greps
        # for when deciding whether the key is still machine-written.
        # Promoted rather than x_-renamed because it is machine-written
        # against a live reader — renaming it on one task would fork the
        # vocabulary and be re-added on the next block.
        'last_blocked_at',
    }
)


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

    if not isinstance(parsed, dict):
        if direction == 'write' and enforce:
            raise TypeError(f'metadata must be a JSON object, got {type(parsed).__name__}')
        warnings.append(
            SchemaWarning(
                field=_WHOLE_METADATA_FIELD,
                code='not_an_object',
                message=f'metadata is not a JSON object: got {type(parsed).__name__}',
            )
        )
        return TaskMetadata(), warnings

    parsed = apply_migrations(parsed)

    for key, submodel in _SUBMODEL_REGISTRY.items():
        if key not in parsed:
            continue
        raw = parsed[key]
        try:
            if isinstance(raw, list):
                # A registered slice may itself be list-valued (e.g. a
                # future metadata.delivered_checks) rather than a single
                # mapping — validate each element independently and swap in
                # the typed list. The comprehension raises on the first bad
                # element (TypeError for a non-mapping item, ValidationError
                # for a mapping that fails the model), which aborts before
                # `parsed` is reassigned below, so a malformed list is
                # retained wholesale — same as the dict path.
                parsed = {**parsed, key: [submodel(**item) for item in raw]}
            else:
                # `submodel(**raw)` raises TypeError (not ValidationError)
                # when the slice's value isn't a mapping at all (e.g. a
                # str) — caught alongside ValidationError so a non-mapping
                # slice is absorbed by the same warn-or-raise policy as any
                # other malformed sub-model, never escaping uncaught in
                # read or write+enforce=False.
                parsed = {**parsed, key: submodel(**raw)}
        except (ValidationError, TypeError) as exc:
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
        # deterministic invariant) has an empty loc, filtered out below.
        # Grouping messages by field also scopes each field's warning to
        # only its own errors, rather than the full multi-error exception
        # dump (which would otherwise repeat every offending field's text
        # in every one of that blob's warnings).
        errors_by_field: dict[str, list[str]] = {}
        for err in exc.errors():
            loc = err['loc']
            if loc and isinstance(loc[0], str):
                errors_by_field.setdefault(loc[0], []).append(str(err['msg']))
        offending = set(errors_by_field)
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
                warnings.append(
                    SchemaWarning(
                        field=key,
                        code='invalid_field',
                        message='; '.join(errors_by_field[key]),
                    )
                )
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
        if key in known_fields or key.startswith('x_') or key in _BLESSED_METADATA_KEYS:
            continue
        warnings.append(
            SchemaWarning(
                field=key, code='unknown_key', message=f'unrecognised metadata key {key!r}'
            )
        )

    return model, warnings
