"""Manages the .task/ directory structure in each worktree."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import stat
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from shared import safe_io
from shared.plan_deviation import DeviationRecord

from orchestrator import file_mutex
from orchestrator.config import TASK_META_DIRNAME
from orchestrator.review_issue_contract import REVIEW_ISSUE_OFF_CONTRACT_FACT, check_issues
from orchestrator.review_payload import NoteDisposition, Verdict, normalize_review
from orchestrator.scope_drift import InvalidScopeRequest, ScopeDriftReport, ScopeRequest

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1
PLAN_TASK_TEXT_FP_KEY = '_task_text_fp'
PLAN_BASE_COMMIT_KEY = '_base_commit'

# Must exceed the longest legitimate critical section: plan_tools._mark_step_done's two timeout=10 git probes.
PLAN_MUTEX_TIMEOUT_SECS = 60.0

ORCHESTRATOR_ITERATION_SOURCES: frozenset[str] = frozenset({'orchestrator', 'amendment'})
"""The only ``source`` values :meth:`TaskArtifacts.append_iteration_log` writes."""


def iteration_entry_written_by_orchestrator(entry: object) -> bool:
    """Whether an iterations.jsonl entry went through ``TaskArtifacts.append_iteration_log``.

    Agents also append to iterations.jsonl by hand from their own shell
    (task 5726); only entries passing this check are orchestrator-written.
    Total over arbitrary JSON: any non-dict entry or non-str source is False.
    """
    if not isinstance(entry, dict):
        return False
    source = entry.get('source')
    return isinstance(source, str) and source in ORCHESTRATOR_ITERATION_SOURCES


class ArtifactWriteError(OSError):
    """An atomic artifact write was REFUSED; the target was left UNTOUCHED.

    Names both the requested and the resolved path in its message so the
    failure is actionable without log-scraping.

    Reserved for the ONE refusal that has no natural ``OSError`` of its own:
    declining to write through a DANGLING symlink is a decision this module
    makes, not an error the kernel reported.  Ordinary write failures
    (EACCES, ENOSPC, …) keep propagating as their own types — ``_write_json``
    has ~13 callers whose observable contract is "whatever OSError the
    filesystem raised", and blanket-wrapping them would change that for every
    artifact writer in the orchestrator as a side effect of an atomicity fix.

    Subclasses ``OSError`` so existing ``except OSError`` handlers keep
    working, and is a DISTINCT type so ``_write_json``'s vanished-root
    ``except FileNotFoundError`` tolerance cannot accidentally swallow it.
    """


class ReviewsRootMissing(FileNotFoundError):
    """A review could not be written because the root holding reviews/ is gone.

    Built from a single message, so ``errno`` stays None and
    ``workflow_types.py::classify_failure`` blocks the task, as it does
    ``git_ops.py::WorktreeMissing``.
    """

    def __init__(self, reviewer_name: str, root: Path):
        self.reviewer_name = reviewer_name
        self.root = root
        super().__init__(
            f'review {reviewer_name!r} cannot reach the review gate: '
            f'its root {root} no longer exists'
        )


class PlanMutexNotHeld(RuntimeError):
    """A plan.json read-modify-write reached its write without holding ``plan_mutex()``.

    Raised before anything is written: the alternative is a silently lost update.
    """


# Matches one-or-more leading ``[COMMITTED <hex>]`` provenance tags (with any
# trailing whitespace) at the START of a step description.  Used by
# :meth:`TaskArtifacts.mark_step_committed` to strip a stale tag before
# prepending the current one, so re-marking a step against a DIFFERENT sha
# (e.g. an amended/squashed commit during re-planning) REPLACES the tag rather
# than stacking provenance.  The ``+`` also cleans up any tags that a prior
# (pre-strip) implementation may have already stacked.
_COMMITTED_TAG_RE = re.compile(r'^(?:\[COMMITTED [0-9a-fA-F]+\]\s*)+')


def _existing_mode(target: Path) -> int | None:
    """The permission bits *target* must carry AFTER an atomic replace.

    An existing target's own mode is preserved verbatim, matching what
    ``Path.write_text`` (which truncates rather than recreates) did before
    every artifact write became a tmp+rename.  Without this, the first write
    after that migration would silently re-mode every `.task-meta` artifact.

    ``None`` for a target that does not exist yet lets ``atomic_write_text``
    create the temp 0o666 and have the KERNEL apply the process umask —
    exactly what ``write_text`` produces for a new file, and free of the
    non-thread-safe ``os.umask(os.umask(0))`` read-back that plan-tools'
    superseded (now deleted) mode helper needed to compute the same answer.

    Swallows ONLY FileNotFoundError.  EACCES/ELOOP/ENAMETOOLONG on the stat
    are genuine failures and must surface, not be papered over as "new file".
    """
    try:
        return stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        return None


# Sidecar schema for ``agent_session.json`` (task 2771).  v1 carried only
# session_id/role/started_at/owner_pid; v2 adds the durable task_id binding,
# resume_count, and this discriminator.  Every write emits the current version.
AGENT_SESSION_SCHEMA_VERSION = 2


@dataclass
class AgentSession:
    """Typed payload of the ``agent_session.json`` sidecar (schema v2).

    The sidecar is written before ``_invoke`` spawns an agent subprocess and
    marks that "an invocation is in flight" (presence ⇔ in-flight).  v2 adds
    the durable ``task_id`` lane→task binding, a ``resume_count`` incremented
    on each adopted ``--resume``, and a ``schema_version`` discriminator so a
    reader can tell a pre-deploy v1 sidecar from a v2 one.  A v1 sidecar reads
    back with legacy defaults via :meth:`from_mapping`; every write emits v2.
    """

    session_id: str
    role: str
    started_at: str
    owner_pid: int
    task_id: str | None = None
    resume_count: int = 0
    schema_version: int = AGENT_SESSION_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, data: dict) -> AgentSession:
        """Build from a parsed sidecar dict, applying v1 legacy defaults.

        A v1 sidecar carries none of the v2 keys: ``task_id`` defaults to
        ``None``, ``resume_count`` to ``0``, and ``schema_version`` to ``1``
        (a v2 sidecar reports its own persisted version).
        """
        return cls(
            session_id=data.get('session_id', ''),
            role=data.get('role', ''),
            started_at=data.get('started_at', ''),
            owner_pid=data.get('owner_pid', 0),
            task_id=data.get('task_id'),
            resume_count=int(data.get('resume_count', 0) or 0),
            schema_version=int(data.get('schema_version', 1)),
        )

    def to_dict(self) -> dict:
        """Serialize to the on-disk JSON mapping."""
        return asdict(self)


# Keys the architect LLM sometimes uses instead of the required "steps".
_STEPS_ALIASES = ('tdd_steps', 'tdd_plan', 'implementation_steps')

# Key for deeply-nested phase structures (tdd_implementation_plan.phase_1_red, …).
_NESTED_PLAN_KEY = 'tdd_implementation_plan'


def _wrap_string_items(items: list, prefix: str) -> tuple[list, bool]:
    """Wrap any plain-string items in a list into canonical step dicts.

    Returns (new_list, was_modified).
    """
    modified = False
    out: list = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            out.append({
                'id': f'{prefix}-{i + 1}',
                'description': item,
                'status': 'pending',
                'commit': None,
            })
            modified = True
        else:
            out.append(item)
    return out, modified


def _flatten_phases(phases: dict) -> list[dict]:
    """Flatten a nested tdd_implementation_plan dict into a flat step list."""
    steps: list[dict] = []
    step_counter = 0
    for _phase_key in sorted(phases):
        value = phases[_phase_key]
        if isinstance(value, list):
            items = value
        elif isinstance(value, (dict, str)):
            items = [value]
        else:
            logger.warning('Skipping unrecognisable phase value type %s in %s',
                           type(value).__name__, _phase_key)
            continue
        for item in items:
            step_counter += 1
            if isinstance(item, dict):
                # Ensure it has an id
                if 'id' not in item:
                    item['id'] = f'step-{step_counter}'
                if 'status' not in item:
                    item['status'] = 'pending'
                if 'commit' not in item:
                    item['commit'] = None
                steps.append(item)
            elif isinstance(item, str):
                steps.append({
                    'id': f'step-{step_counter}',
                    'description': item,
                    'status': 'pending',
                    'commit': None,
                })
            else:
                logger.warning('Skipping non-dict/str item in phase %s', _phase_key)
    return steps


def _normalize_plan(plan: dict) -> tuple[dict, bool]:
    """Detect and fix common malformed plan shapes.

    Returns (plan, was_modified).  The caller should write back to disk
    when *was_modified* is True.
    """
    modified = False

    # Rule 1-2: rename known step-key aliases → steps
    if 'steps' not in plan:
        for alias in _STEPS_ALIASES:
            if alias in plan and isinstance(plan[alias], list):
                logger.debug('Normalizing plan: renaming %r → "steps"', alias)
                plan['steps'] = plan.pop(alias)
                modified = True
                break

    # Rule 3: flatten nested tdd_implementation_plan phases
    if 'steps' not in plan and _NESTED_PLAN_KEY in plan:
        nested = plan[_NESTED_PLAN_KEY]
        if isinstance(nested, dict):
            logger.debug(
                'Normalizing plan: flattening %s phases from %r',
                len(nested), _NESTED_PLAN_KEY,
            )
            plan['steps'] = _flatten_phases(nested)
            plan.pop(_NESTED_PLAN_KEY)
            modified = True

    # Rule 4 (removed): string prerequisites are now rejected by
    # validate_plan_prerequisites() rather than silently coerced.

    # Rule 5: wrap string steps
    steps = plan.get('steps')
    if isinstance(steps, list):
        new_steps, changed = _wrap_string_items(steps, 'step')
        if changed:
            logger.debug('Normalizing plan: wrapping %d string steps as dicts',
                         sum(1 for s in steps if isinstance(s, str)))
            plan['steps'] = new_steps
            modified = True

    return plan, modified


class PlanState(StrEnum):
    SHELL = 'shell'
    PARTIAL = 'partial'
    CONFIRMED = 'confirmed'
    STAMPED = 'stamped'


class PlanConfirmation(StrEnum):
    ARCHITECT = 'architect'
    INJECTED = 'injected'


_LIFECYCLE_KEYS: Mapping[str, str] = MappingProxyType({
    'session_id': '_session_id',
    'created_at': '_created_at',
    'confirmed_at': '_finalized_at',
    'confirmed_by': '_confirmed_by',
    'revalidated_at': '_revalidated_at',
    'revalidated_by_session': '_revalidated_by_session',
    'task_text_fp': PLAN_TASK_TEXT_FP_KEY,
    'base_commit': PLAN_BASE_COMMIT_KEY,
    'schema_version': '_schema_version',
})
_LIFECYCLE_DISK_KEYS = frozenset(_LIFECYCLE_KEYS.values())
_CONFIRMER_VALUES = frozenset(confirmer.value for confirmer in PlanConfirmation)
_LIFECYCLE_REQUIREMENTS: Mapping[str, str] = MappingProxyType({
    'confirmed_by': 'must be a PlanConfirmation or None',
    'schema_version': 'must be an int (not a bool) or None',
})
_STR_LIFECYCLE_REQUIREMENT = 'must be a non-empty str or None'


def _well_formed(name: str, value: object) -> bool:
    """Whether *value* may be held by the set PlanLifecycle field *name*: the one definition parse and construction share."""
    if name == 'confirmed_by':
        return isinstance(value, PlanConfirmation)
    if name == 'schema_version':
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str) and bool(value)


def _unpaired_confirmer(confirmed_by: object, confirmed_at: object) -> bool:
    return confirmed_by is not None and confirmed_at is None


def _malformed_lifecycle(name: str, value: object, requirement: str) -> ValueError:
    return ValueError(f'PlanLifecycle.{name} {requirement}; got {value!r}')


@dataclass(frozen=True)
class PlanLifecycle:
    """The typed lifecycle markers of a plan document; a field is None when absent on disk."""

    session_id: str | None = None
    created_at: str | None = None
    confirmed_at: str | None = None
    confirmed_by: PlanConfirmation | None = None
    revalidated_at: str | None = None
    revalidated_by_session: str | None = None
    task_text_fp: str | None = None
    base_commit: str | None = None
    schema_version: int | None = None

    def __post_init__(self) -> None:
        for name in _LIFECYCLE_KEYS:
            value = getattr(self, name)
            if value is not None and not _well_formed(name, value):
                raise _malformed_lifecycle(name, value, _LIFECYCLE_REQUIREMENTS.get(name, _STR_LIFECYCLE_REQUIREMENT))
        if _unpaired_confirmer(self.confirmed_by, self.confirmed_at):
            raise _malformed_lifecycle('confirmed_by', self.confirmed_by, 'is set without confirmed_at')

    @property
    def confirmation(self) -> PlanConfirmation | None:
        """Who confirmed the plan; a legacy marker with no confirmer reads as the architect."""
        if self.confirmed_at is None:
            return None
        return self.confirmed_by or PlanConfirmation.ARCHITECT

    def owned_by(self, session_id: str) -> bool:
        """Whether *session_id* planned this plan or last revalidated it."""
        return session_id in (self.session_id, self.revalidated_by_session)

    def markers_present(self) -> tuple[str, ...]:
        """The on-disk keys of the fields that are set, sorted."""
        return tuple(sorted(key for name, key in _LIFECYCLE_KEYS.items() if getattr(self, name) is not None))


def _decoded_marker(name: str, value: object) -> object:
    if name == 'confirmed_by' and isinstance(value, str) and value in _CONFIRMER_VALUES:
        return PlanConfirmation(value)
    return value


def _parse_lifecycle(data: Mapping[str, object]) -> tuple[PlanLifecycle, frozenset[str]]:
    """The typed lifecycle of *data*, and the lifecycle-named keys whose values are malformed."""
    fields: dict[str, Any] = {}
    malformed: set[str] = set()
    for name, key in _LIFECYCLE_KEYS.items():
        if key not in data:
            continue
        value = _decoded_marker(name, data[key])
        if _well_formed(name, value):
            fields[name] = value
        else:
            malformed.add(key)
    if _unpaired_confirmer(fields.get('confirmed_by'), fields.get('confirmed_at')):
        del fields['confirmed_by']
        malformed.add(_LIFECYCLE_KEYS['confirmed_by'])
    return PlanLifecycle(**fields), frozenset(malformed)


def plan_base_commit(plan: Mapping[str, object]) -> str | None:
    """The main SHA *plan* was last validated against (plans/task-meta-task-keying-prd.md §M2); None when unstamped or malformed."""
    return _parse_lifecycle(plan)[0].base_commit


def plan_task_text_fp(plan: Mapping[str, object]) -> str | None:
    """The task-text fingerprint *plan* was authored against; None when unstamped or malformed."""
    return _parse_lifecycle(plan)[0].task_text_fp


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    return copy.deepcopy(value)


def _thawed(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thawed(item) for item in value]
    return copy.deepcopy(value)


def _is_step_list(value: object) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(step, Mapping) for step in value)


def _freeze_steps(steps: list | tuple) -> tuple[Mapping[str, Any], ...]:
    return tuple(_frozen(step) for step in steps)


def _freeze_body(body: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _frozen(value) for key, value in body.items()})


class PlanNotConfirmed(ValueError):
    """A lifecycle move that needs a confirmed plan met a plan in another state."""

    def __init__(self, *, task_id: object, state: PlanState, markers: tuple[str, ...], action: str) -> None:
        self.task_id = task_id
        self.state = state
        self.markers = markers
        self.action = action
        super().__init__(
            f'cannot {action} the plan of task {task_id!r} in state {state.value!r}; '
            f'lifecycle markers present: {list(markers)}'
        )


@dataclass(frozen=True)
class Plan:
    """One plan.json document: the typed lifecycle, the steps, and every other key verbatim.

    Immutable all the way down: nested mappings are read-only proxies and
    lists are tuples. :meth:`to_json` hands back a fresh, editable document.
    """

    lifecycle: PlanLifecycle
    steps: tuple[Mapping[str, Any], ...]
    body: Mapping[str, Any]
    key_order: tuple[str, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        shadowed = sorted(set(self.lifecycle.markers_present()) & set(self.body))
        if shadowed:
            raise ValueError(f'Plan body repeats lifecycle keys its lifecycle sets: {shadowed}')
        if 'steps' in self.body and (self.steps or _is_step_list(self.body['steps'])):
            raise ValueError("Plan body holds 'steps' that belong in the steps tuple")

    @property
    def state(self) -> PlanState:
        if not self.steps:
            return PlanState.SHELL
        if self.lifecycle.confirmed_at is None:
            return PlanState.PARTIAL
        if self.lifecycle.session_id is None:
            return PlanState.CONFIRMED
        return PlanState.STAMPED

    @classmethod
    def from_json(cls, data: object) -> Plan:
        if not isinstance(data, Mapping):
            raise TypeError(f'a plan document must be a mapping, not {type(data).__name__}')
        lifecycle, malformed = _parse_lifecycle(data)
        typed_steps = _is_step_list(data.get('steps'))
        body = {
            key: value for key, value in data.items()
            if (key not in _LIFECYCLE_DISK_KEYS or key in malformed) and not (key == 'steps' and typed_steps)
        }
        return cls(
            lifecycle=lifecycle,
            steps=_freeze_steps(data['steps']) if typed_steps else (),
            body=_freeze_body(body),
            key_order=tuple(data),
        )

    def to_json(self) -> dict[str, Any]:
        fresh = {key: _thawed(value) for key, value in self.body.items()}
        if 'steps' not in fresh and (self.steps or 'steps' in self.key_order):
            fresh['steps'] = [_thawed(step) for step in self.steps]
        for name, key in _LIFECYCLE_KEYS.items():
            value = getattr(self.lifecycle, name)
            if value is not None:
                fresh[key] = value.value if isinstance(value, PlanConfirmation) else value
        document = {key: fresh[key] for key in self.key_order if key in fresh}
        document.update(fresh)
        return document

    def recreated(
        self, *, task_id: str, title: str, analysis: str, files: list[str], carried: Mapping[str, Any],
    ) -> Plan:
        """The fresh create_plan shell that replaces this plan wholesale, carrying *carried* keys.

        Nothing of this plan survives, its lifecycle included.
        """
        refused = sorted((_LIFECYCLE_DISK_KEYS | {'steps'}) & set(carried))
        if refused:
            raise ValueError(f'create_plan cannot carry lifecycle keys or steps: {refused}')
        return Plan.from_json({
            'task_id': task_id, 'title': title, 'analysis': analysis, 'files': files,
            'prerequisites': [], 'steps': [], 'design_decisions': [], 'reuse': [], 'dropped_files': [], **carried,
        })

    def restructured(self, **changes: Any) -> Plan:
        """A step-set edit: *changes* applied and confirmation cleared."""
        return self._with_body(changes)._with_markers(confirmed_at=None, confirmed_by=None)

    def confirmed(self, *, at: str, by: PlanConfirmation) -> Plan:
        """Confirmed at *at* by *by*, which dates the LATEST confirm, never the first.

        Also sets ``revalidated_at``, the fallback after ``created_at`` that
        ``workflow.py::TaskWorkflow._can_skip_revalidation`` ages the plan by.
        Only :meth:`restructured` and :meth:`recreated` clear a confirmation.
        """
        if not self.steps:
            raise self._refusal('confirm')
        return self._with_markers(confirmed_at=at, confirmed_by=by, revalidated_at=at)

    def stamped(self, *, session_id: str, at: str, task_text_fp: str | None, base_commit: str | None) -> Plan:
        if self.state not in (PlanState.CONFIRMED, PlanState.STAMPED):
            raise self._refusal('stamp')
        markers: dict[str, str] = {'session_id': session_id, 'created_at': at}
        if task_text_fp:
            markers['task_text_fp'] = task_text_fp
        if base_commit:
            markers['base_commit'] = base_commit
        return self._with_markers(**markers)

    def with_files(self, files: list[str], *, owner: str, at: str) -> Plan:
        """A files-only edit that keeps confirmation and makes *owner* own the plan."""
        edited = self._with_body({'files': files})
        if self.lifecycle.owned_by(owner):
            return edited
        return edited._with_markers(session_id=owner, created_at=self.lifecycle.created_at or at)

    def revalidated(self, *, at: str, by_session: str, base_commit: str | None) -> Plan:
        markers: dict[str, str] = {'revalidated_at': at, 'revalidated_by_session': by_session}
        if base_commit:
            markers['base_commit'] = base_commit
        return self._with_markers(**markers)

    def confirmed_since(self, pre: Plan) -> bool:
        """Whether this plan carries a confirmation *pre* did not."""
        confirmed_at = self.lifecycle.confirmed_at
        return confirmed_at is not None and confirmed_at != pre.lifecycle.confirmed_at

    def with_lifecycle(self, lifecycle: PlanLifecycle) -> Plan:
        return self._under(lifecycle, stripped=frozenset(lifecycle.markers_present()))

    def _with_markers(self, **fields: Any) -> Plan:
        return self._under(
            replace(self.lifecycle, **fields), stripped=frozenset(_LIFECYCLE_KEYS[name] for name in fields),
        )

    def _under(self, lifecycle: PlanLifecycle, *, stripped: frozenset[str]) -> Plan:
        body = MappingProxyType({key: value for key, value in self.body.items() if key not in stripped})
        return Plan(lifecycle=lifecycle, steps=self.steps, body=body, key_order=self.key_order)

    def _with_body(self, changes: Mapping[str, Any]) -> Plan:
        refused = sorted(_LIFECYCLE_DISK_KEYS & set(changes))
        if refused:
            raise ValueError(f'lifecycle keys cannot change through a body edit: {refused}')
        body = dict(self.body)
        steps = self.steps
        key_order = self.key_order
        if 'steps' in changes:
            body.pop('steps', None)
            if _is_step_list(changes['steps']):
                steps = _freeze_steps(changes['steps'])
                if 'steps' not in key_order:
                    key_order = (*key_order, 'steps')
            else:
                steps = ()
                body['steps'] = changes['steps']
        body.update((key, value) for key, value in changes.items() if key != 'steps')
        return Plan(lifecycle=self.lifecycle, steps=steps, body=_freeze_body(body), key_order=key_order)

    def _refusal(self, action: str) -> PlanNotConfirmed:
        return PlanNotConfirmed(
            task_id=self.body.get('task_id'), state=self.state,
            markers=self.lifecycle.markers_present(), action=action,
        )


def _at_current_schema(plan: Plan) -> Plan:
    return plan.with_lifecycle(replace(plan.lifecycle, schema_version=PLAN_SCHEMA_VERSION))


_VALID_VERDICT_ROLE_RE = re.compile(r'^[A-Za-z0-9_-]+$')


def _validate_verdict_role(role: str) -> None:
    """Guard ``verdicts/<role>.json`` against escaping the ``verdicts/`` dir.

    ``role`` is currently always the trusted ``--verdict-role`` CLI arg
    (see verdict_tools.py's ``_artifacts_from_args``), so a path separator
    or ``..`` can never reach here today. The guard exists so a future,
    less-trusted caller can't turn an unsanitized role into a
    write/read/clear outside ``verdicts/``.

    Raises:
        ValueError: if *role* contains anything other than letters, digits,
            underscores, or hyphens.
    """
    if not _VALID_VERDICT_ROLE_RE.fullmatch(role):
        raise ValueError(
            f'invalid verdict role {role!r}: must match '
            f'{_VALID_VERDICT_ROLE_RE.pattern!r}'
        )


_VALID_TASK_ID_RE = re.compile(r'^[A-Za-z0-9._-]+$')


def _validate_task_id(task_id: str) -> None:
    """Guard ``.task-meta/task-<id>`` against naming anything but one dir.

    The ``task-`` prefix means even ``.`` or ``..`` can only produce a literal
    ``task-..`` directory name, never a traversal; a separator, whitespace or
    a non-str id is refused outright.

    Raises:
        ValueError: if *task_id* is not a str of letters, digits, dots,
            underscores, or hyphens.
    """
    if not isinstance(task_id, str) or not _VALID_TASK_ID_RE.fullmatch(task_id):
        raise ValueError(
            f'invalid task id {task_id!r}: must be a str matching '
            f'{_VALID_TASK_ID_RE.pattern!r}'
        )


class ArtifactRoot(Enum):
    """Which `.task-meta` root an artifact lives under.

    LANE is per-dispatch (``TaskArtifacts.meta_root_for``); TASK follows the
    task across lanes (``TaskArtifacts.task_root_for``).
    """

    LANE = 'lane'
    TASK = 'task'


_REVIEW_ARCHIVE_FAMILY = 'reviews-<label>/'
_ZERO_OUTPUT_EVIDENCE_NAME_TEMPLATE = 'zero_output_evidence-iter{}.json'
_ZERO_OUTPUT_EVIDENCE_FAMILY = _ZERO_OUTPUT_EVIDENCE_NAME_TEMPLATE.format('<n>')

ARTIFACT_ROOTS: Mapping[str, ArtifactRoot] = MappingProxyType({
    'metadata.json': ArtifactRoot.LANE,
    'agent_session.json': ArtifactRoot.LANE,
    'interactive.json': ArtifactRoot.LANE,
    'blocking_dependency.json': ArtifactRoot.LANE,
    'already_done.json': ArtifactRoot.LANE,
    'ready_to_merge.json': ArtifactRoot.LANE,
    'false_premise.json': ArtifactRoot.LANE,
    'unactionable_task.json': ArtifactRoot.LANE,
    'scope_requests.json': ArtifactRoot.LANE,
    'scope_drift_brief.json': ArtifactRoot.TASK,
    'plan.json': ArtifactRoot.TASK,
    'plan.lock': ArtifactRoot.TASK,
    'iterations.jsonl': ArtifactRoot.TASK,
    'reviews': ArtifactRoot.TASK,
    'verdicts': ArtifactRoot.TASK,
    'review_state.json': ArtifactRoot.TASK,
    'reconcile_state.json': ArtifactRoot.TASK,
    _REVIEW_ARCHIVE_FAMILY: ArtifactRoot.TASK,
    _ZERO_OUTPUT_EVIDENCE_FAMILY: ArtifactRoot.TASK,
})
"""The single routing table: artifact name (or family key) -> owning root.

PRD ``plans/task-meta-task-keying-prd.md`` §Artifact routing, plus the
architect's ``ready_to_merge.json`` exit and the review-archive and
zero-output-evidence families. ``plan.json.mutex`` is deliberately absent: it
is derived from the resolved plan.json, so it always sits beside that plan.
"""

LANE_SYMLINKED_ARTIFACTS: tuple[str, ...] = ('plan.json', 'iterations.jsonl')
"""The task-root artifacts the lane ``.task/`` exposes to agents."""


@dataclass
class ReviewAggregation:
    """Aggregated review results from all reviewers."""

    has_blocking_issues: bool
    blocking_issues: list[dict]
    suggestions: list[dict]
    reviews: dict[str, dict]
    reviewer_errors: list[str] = field(default_factory=list)
    errored_reviews: dict[str, dict] = field(default_factory=dict)
    note_dispositions: tuple[NoteDisposition, ...] = ()

    @property
    def all_reviewers_errored(self) -> bool:
        """True when every reviewer returned ERROR (no usable reviews)."""
        return bool(self.reviewer_errors) and not self.reviews

    def format_for_replan(self) -> str:
        """Blocking issues only — the architect's input (TaskWorkflow._replan)."""
        return _format_issue_section('# Review Feedback — Blocking Issues\n', self.blocking_issues)

    def format_for_escalation(self) -> str:
        """Blocking issues then suggestions — the steward's review_issues detail (TaskWorkflow._escalate_review_issues)."""
        text = self.format_for_replan()
        if self.suggestions:
            text += '\n' + _format_issue_section('# Review Feedback — Suggestions\n', self.suggestions)
        return text


def _warn_off_contract_issues(reviewer_name: str, issues: tuple[dict, ...]) -> None:
    deviations = check_issues(issues).deviations
    if deviations:
        logger.warning(
            'review from %s carries %d deviation(s) from the reviewer issue contract',
            reviewer_name, len(deviations),
            extra={
                'event': REVIEW_ISSUE_OFF_CONTRACT_FACT,
                'reviewer': reviewer_name,
                'deviations': [d.describe() for d in deviations],
            },
        )


def _format_issue_section(heading: str, issues: list[dict]) -> str:
    lines = [heading]
    for issue in issues:
        reviewer = issue.get('reviewer', 'unknown')
        location = issue.get('location', '')
        category = issue.get('category', '')
        description = issue.get('description', '')
        fix = issue.get('suggested_fix', '')
        lines.append(f'## [{reviewer}] {category}')
        if location:
            lines.append(f'**Location:** {location}')
        lines.append(f'**Issue:** {description}')
        if fix:
            lines.append(f'**Suggested fix:** {fix}')
        lines.append('')
    return '\n'.join(lines)


def _replace_with_symlink(link: Path, target: Path) -> None:
    if link.is_dir() and not link.is_symlink():
        shutil.rmtree(link)
    else:
        link.unlink(missing_ok=True)
    os.symlink(target, link)


class TaskArtifacts:
    """A task's artifacts, each resolved to its root through ``ARTIFACT_ROOTS``."""

    def __init__(
        self,
        worktree: Path,
        meta_root: Path | None = None,
        task_root: Path | None = None,
    ):
        """
        Args:
            worktree: The task's git worktree.
            meta_root: The LANE root (see ``meta_root_for``). ``None`` means
                the legacy ``<worktree>/.task``.
            task_root: The TASK root (see ``task_root_for``). ``None`` means
                the lane root, so omitting it keeps one single root for
                every artifact. Production dual-root construction goes
                through ``for_task``.
        """
        self.worktree = worktree
        self._lane_root = meta_root if meta_root is not None else worktree / '.task'
        self._task_root = task_root if task_root is not None else self._lane_root

    @classmethod
    def for_task(cls, worktree: Path, task_id: str) -> TaskArtifacts:
        """Both roots of *task_id* dispatched into *worktree*.

        Raises:
            ValueError: *task_id* fails ``_validate_task_id``.
        """
        base = worktree.parent
        return cls(worktree, cls.meta_root_for(base, worktree.name), cls.task_root_for(base, task_id))

    @property
    def lane_root(self) -> Path:
        return self._lane_root

    @property
    def task_root(self) -> Path:
        return self._task_root

    def _root_for(self, key: str) -> Path:
        owner = ARTIFACT_ROOTS.get(key)
        if owner is None:
            raise ValueError(
                f'unknown task artifact {key!r}: known keys are {sorted(ARTIFACT_ROOTS)}'
            )
        return self._task_root if owner is ArtifactRoot.TASK else self._lane_root

    def artifact_path(self, name: str) -> Path:
        """Where artifact *name* (``'plan.json'``, ``'reviews/merge.json'``)
        lives, routed on its first path component.

        Raises:
            ValueError: that component is not a key of ``ARTIFACT_ROOTS``.
        """
        return self._root_for(Path(name).parts[0]) / name

    @property
    def review_archives_root(self) -> Path:
        """The directory holding every ``reviews-<label>`` archive."""
        return self._root_for(_REVIEW_ARCHIVE_FAMILY)

    def review_archive_dir(self, label: str) -> Path:
        """The archive of one review round, e.g. ``label='amend-1'``."""
        return self.review_archives_root / f'reviews-{label}'

    def zero_output_evidence_path(self, iteration: int) -> Path:
        return self._root_for(_ZERO_OUTPUT_EVIDENCE_FAMILY) / _ZERO_OUTPUT_EVIDENCE_NAME_TEMPLATE.format(iteration)

    def zero_output_evidence_glob(self) -> str:
        """Every iteration's ``zero_output_evidence_path``, as one glob, for operator-facing text."""
        return str(self._root_for(_ZERO_OUTPUT_EVIDENCE_FAMILY) / _ZERO_OUTPUT_EVIDENCE_NAME_TEMPLATE.format('*'))

    def read_title(self) -> str | None:
        """metadata.json's title, else plan.json's; None unless a non-blank str."""
        for name in ('metadata.json', 'plan.json'):
            data = self._read_json_object(name)
            title = data.get('title') if data is not None else None
            if isinstance(title, str) and title.strip():
                return title
        return None

    @staticmethod
    def meta_root_for(worktree_base: Path, worktree_name: str) -> Path:
        """Derive the LANE (per-dispatch) `.task-meta` root for a worktree —
        the single owner of this path shape. Its task-lifetime counterpart is
        ``task_root_for`` (PRD ``plans/task-meta-task-keying-prd.md``
        §Path-derivation contract).

        Returns ``<worktree_base>/.task-meta/<worktree_name>`` — a SIBLING
        of the worktree itself (``<worktree_base>/<worktree_name>``), not
        nested inside it like the legacy ``<worktree>/.task``. Callers must
        call this instead of joining ``'.task'`` (or ``'.task-meta'``) by
        hand.

        Because this path lives OUTSIDE the worktree, it survives a worktree
        reset/``git clean``; a pooled lane reused by a new task can therefore
        hold the PRIOR dispatch's lane-keyed exits until they are cleared on
        lane acquisition. See ``plans/worktree-lane-lifecycle-prd.md``.
        """
        return worktree_base / TASK_META_DIRNAME / worktree_name

    @staticmethod
    def task_root_for(worktree_base: Path, task_id: str) -> Path:
        """Derive the TASK-lifetime `.task-meta` root: the single owner of the
        ``<worktree_base>/.task-meta/task-<task_id>`` shape that holds the
        artifacts which follow a task across lanes (PRD
        ``plans/task-meta-task-keying-prd.md`` §Path-derivation contract).

        Raises:
            ValueError: *task_id* fails ``_validate_task_id``.
        """
        _validate_task_id(task_id)
        return worktree_base / TASK_META_DIRNAME / f'task-{task_id}'

    @staticmethod
    def read_lane_task_id(meta_root: Path) -> str | None:
        """The task id ``<meta_root>/metadata.json`` was dispatched for, else None.

        The lane->task key of ``GitOps``'s disk backstop. One root, no
        ``plan.json`` and no legacy fallback: PRD
        ``plans/task-meta-task-keying-prd.md`` decision 7. Never raises.
        """
        meta_path = meta_root / 'metadata.json'
        if not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text())
        except (ValueError, OSError) as exc:
            logger.warning('Unreadable metadata.json at %s: %s', meta_path, exc)
            return None
        if not isinstance(metadata, dict):
            logger.warning('Malformed metadata.json at %s: not an object', meta_path)
            return None
        task_id = metadata.get('task_id')
        if isinstance(task_id, str):
            return task_id or None
        if isinstance(task_id, int) and not isinstance(task_id, bool):
            return str(task_id)
        return None

    def init(
        self,
        task_id: str,
        task_title: str,
        task_description: str,
        base_commit: str | None = None,
    ) -> None:
        """Create both roots, the reviews/ and verdicts/ dirs, and metadata.json.

        IMPORTANT: This method is safe to call when the roots already exist
        (e.g. inherited from a contaminated main).
        """
        self._lane_root.mkdir(parents=True, exist_ok=True)
        self._task_root.mkdir(parents=True, exist_ok=True)
        self.artifact_path('reviews').mkdir(exist_ok=True)
        self.artifact_path('verdicts').mkdir(exist_ok=True)

        metadata = {
            'task_id': task_id,
            'title': task_title,
            'description': task_description,
            'created_at': datetime.now(UTC).isoformat(),
        }
        if base_commit:
            metadata['base_commit'] = base_commit
        self._write_json('metadata.json', metadata)

    def read_base_commit(self) -> str | None:
        """Read the base commit SHA stored at init time."""
        meta_path = self.artifact_path('metadata.json')
        if not meta_path.exists():
            return None
        metadata = json.loads(meta_path.read_text())
        return metadata.get('base_commit')

    def read_created_at(self) -> str | None:
        """Read the created_at timestamp stored at init time.

        Mirrors ``read_base_commit``'s shape, but is exception-tolerant for a
        corrupt/unreadable or malformed metadata.json (returns ``None``
        instead of raising) so a runtime "started" lookup never raises.
        """
        metadata = self._read_json_object('metadata.json')
        if metadata is None:
            return None
        return metadata.get('created_at')

    def update_base_commit(self, new_base: str) -> None:
        """Update the base_commit in metadata.json after a rebase."""
        meta_path = self.artifact_path('metadata.json')
        if not meta_path.exists():
            return
        metadata = json.loads(meta_path.read_text())
        metadata['base_commit'] = new_base
        self._write_json('metadata.json', metadata)

    @contextmanager
    def plan_mutex(self) -> Iterator[None]:
        """The cross-writer lock every plan.json read-modify-write holds.

        Reentrant per thread and honoured across processes
        (``orchestrator.file_mutex``). Keyed on the RESOLVED plan path, so the
        lane symlink view and the meta-root view share one lock. Distinct from
        the ownership lock ``lock_plan`` (plan.lock). A vanished root takes no
        lock: it holds no plan to lose an update of, and must not be
        re-materialized by locking.

        A WAIT BLOCKS THE CALLING THREAD, for up to PLAN_MUTEX_TIMEOUT_SECS. The
        orchestrator calls its plan methods synchronously on its event-loop
        thread, so while a plan-tools process holds the mutex, every task
        workflow in the orchestrator stalls. Keep critical sections short, and
        never hold one across a subprocess the plan does not depend on.
        """
        with self._artifact_mutex('plan.json'):
            yield

    @contextmanager
    def _artifact_mutex(self, name: str) -> Iterator[None]:
        """The cross-process lock for a read-modify-write of artifact *name*.

        The policy :meth:`plan_mutex` documents, for any artifact: keyed on
        the RESOLVED path, and no lock once the root is gone.
        """
        lock_path = self._artifact_mutex_path(name)
        if lock_path is None:
            yield
            return
        with file_mutex.exclusive(lock_path, timeout=PLAN_MUTEX_TIMEOUT_SECS):
            yield

    def _artifact_mutex_path(self, name: str) -> Path | None:
        lock_path = Path(os.path.realpath(self.artifact_path(name))).with_name(f'{name}.mutex')
        return lock_path if lock_path.parent.is_dir() else None

    def write_plan_model(self, plan: Plan, *, mutex_held: bool = False) -> Plan:
        """Write .task/plan.json, stamped with the current schema version: its only writer.

        Returns the Plan as written, which is *plan* at the current schema version.

        A bare overwrite holds ``plan_mutex()`` for the write alone. The write
        that ends a read-modify-write passes ``mutex_held=True``, a claim that
        is VERIFIED: if the calling thread does not hold the mutex it raises
        :class:`PlanMutexNotHeld` and writes nothing.
        """
        written = _at_current_schema(plan)
        with nullcontext() if mutex_held else self.plan_mutex():
            lock_path = self._artifact_mutex_path('plan.json')
            if lock_path is not None and not file_mutex.held(lock_path):
                raise PlanMutexNotHeld(
                    f'{self.artifact_path("plan.json")} reached its write without {lock_path} held: '
                    'run the whole read-modify-write inside plan_mutex()'
                )
            self._write_json('plan.json', written.to_json())
        return written

    def write_plan(self, plan: dict, *, mutex_held: bool = False) -> None:
        """The dict adapter over :meth:`write_plan_model`; also stamps the schema version on *plan* itself."""
        plan['_schema_version'] = PLAN_SCHEMA_VERSION
        self.write_plan_model(Plan.from_json(plan), mutex_held=mutex_held)

    def ensure_lane_symlinks(self) -> None:
        """Expose each ``LANE_SYMLINKED_ARTIFACTS`` name in the lane ``.task/``.

        Makes ``<worktree>/.task/<name>`` an ABSOLUTE symlink to
        ``artifact_path(<name>)``, so an agent reading or appending through the
        lane path reaches the durable file and the lane copy can never diverge
        from it (the esc-5205-9 stale trap: steps appended to a regular-FILE
        lane plan that ``mark_step_done`` then rejected).

        Idempotent and recreated each dispatch: replaces a regular file, a
        stale or broken symlink, or a stray directory. Skips a name whose
        durable path already IS the lane path (legacy single-root mode), where
        a self-referential symlink would be broken.
        """
        lane_task = self.worktree / '.task'
        lane_task.mkdir(parents=True, exist_ok=True)
        for name in LANE_SYMLINKED_ARTIFACTS:
            durable = self.artifact_path(name)
            if durable != lane_task / name:
                _replace_with_symlink(lane_task / name, durable)

    def write_blocking_dependency(
        self,
        depends_on_task_id: str,
        reason: str,
        main_sha_at_report: str,
    ) -> None:
        """Write .task/blocking_dependency.json — architect's report that
        this task cannot be planned until ``depends_on_task_id`` lands.

        The workflow reads this artifact after the architect returns and
        deterministically registers the Taskmaster dependency before
        requeueing.  ``main_sha_at_report`` lets the workflow detect a
        ``dep_id``-already-terminal race (main advanced after the report).
        """
        data = {
            'depends_on_task_id': depends_on_task_id,
            'reason': reason,
            'main_sha_at_report': main_sha_at_report,
            'reported_at': datetime.now(UTC).isoformat(),
        }
        self._write_json('blocking_dependency.json', data)

    def read_blocking_dependency(self) -> dict | None:
        """Return the parsed ``.task/blocking_dependency.json`` artifact if
        present, else ``None``.
        """
        path = self.artifact_path('blocking_dependency.json')
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear_blocking_dependency(self) -> None:
        """Remove ``.task/blocking_dependency.json`` if present.

        Called by the workflow after the dependency has been registered (or
        determined to be a no-op due to an advance-past-report race) so a
        subsequent architect invocation does not see a stale report.
        """
        self._clear_path('blocking_dependency.json')

    def write_scope_request(self, request: ScopeRequest) -> int:
        """Record a writer agent's request to add an existing file to its scope.

        One entry per path: a repeat replaces the earlier entry and moves to
        the end. Returns how many requests are now pending. The plan-tools
        server runs concurrent tool calls on threads, so the read-modify-write
        holds its :meth:`_artifact_mutex`. The workflow applies and removes
        them at each writer boundary
        (``workflow.py::TaskWorkflow._apply_scope_requests``).
        """
        with self._artifact_mutex('scope_requests.json'):
            kept = [r for r in self.read_scope_requests() if r.path != request.path]
            entries = [r.to_mapping() for r in (*kept, request)]
            self._write_json('scope_requests.json', {'requests': entries})
            return len(entries)

    def read_scope_requests(self) -> tuple[ScopeRequest, ...]:
        """The recorded scope requests in write order; each invalid entry is dropped with a warning."""
        data = self._read_json_object('scope_requests.json') or {}
        entries = data.get('requests', [])
        if not isinstance(entries, list):
            logger.warning('Malformed scope_requests.json: requests is %r', entries)
            return ()
        requests: list[ScopeRequest] = []
        for entry in entries:
            try:
                requests.append(ScopeRequest.from_mapping(entry))
            except InvalidScopeRequest as exc:
                logger.warning('Dropping scope request %r from scope_requests.json: %s', entry, exc)
        return tuple(requests)

    def clear_scope_requests(self, applied: Iterable[ScopeRequest]) -> None:
        """Remove the *applied* requests, keeping any filed since they were read.

        The artifact goes once nothing is left in it.
        """
        done = set(applied)
        with self._artifact_mutex('scope_requests.json'):
            remaining = [r for r in self.read_scope_requests() if r not in done]
            if remaining:
                self._write_json(
                    'scope_requests.json', {'requests': [r.to_mapping() for r in remaining]},
                )
            else:
                self._clear_path('scope_requests.json')

    def read_scope_drift_brief(self) -> ScopeDriftReport | None:
        """The scope-drift report no writer session has received yet, or None.

        TASK-rooted, so a brief outlives the dispatch that recorded it: a
        requeue or restart before the next writer session still briefs it.
        Only the workflow writes it, and reads it once per dispatch
        (``workflow.py::TaskWorkflow._hold_scope_drift_brief``). A malformed
        file reads as None with a warning.
        """
        data = self._read_json_object('scope_drift_brief.json')
        if data is None:
            return None
        try:
            return ScopeDriftReport.from_event_data(data)
        except ValueError as exc:
            logger.warning('Ignoring malformed scope_drift_brief.json: %s', exc)
            return None

    def write_scope_drift_brief(self, report: ScopeDriftReport) -> None:
        self._write_json('scope_drift_brief.json', report.event_data())

    def clear_scope_drift_brief(self) -> None:
        self._clear_path('scope_drift_brief.json')

    # ──────────────────────────────────────────────────────────────────
    # Architect task-rejection artifacts (paired with plan-tools MCP)
    #
    # These are structured exits the architect takes *instead of* writing a
    # plan, when it can determine the task should not proceed as written.
    # The workflow reads the artifact after the architect returns and takes
    # a deterministic action — see workflow._plan().
    # ──────────────────────────────────────────────────────────────────

    def write_already_done(self, commit: str, evidence: str) -> None:
        """Write ``.task/already_done.json`` — architect's claim that this
        task's work is already on main at ``commit``.

        The workflow validates the claim (commit reachable from main) and
        sets task status to ``done`` with provenance pointing at ``commit``.
        """
        data = {
            'commit': commit,
            'evidence': evidence,
            'reported_at': datetime.now(UTC).isoformat(),
        }
        self._write_json('already_done.json', data)

    def read_already_done(self) -> dict | None:
        """Return the parsed ``.task/already_done.json`` artifact if present,
        else ``None``.
        """
        path = self.artifact_path('already_done.json')
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear_already_done(self) -> None:
        """Remove ``.task/already_done.json`` if present."""
        self._clear_path('already_done.json')

    def write_ready_to_merge(self, commit: str, evidence: str) -> None:
        """Write ``.task/ready_to_merge.json`` — architect's claim that this
        task's work is complete on the branch at ``commit`` and only the
        physical merge to main is missing (a merge-landing desync).

        The workflow validates the desync predicate first-hand (clean
        fast-forward of main, verify PASSED on this tip, review PASS on this
        tree) and, if it holds, deterministically enqueues a merge request
        rather than filing a human escalation.  The exit never lands main
        itself — the merge worker's scoped re-verify remains the sole gate.
        """
        data = {
            'commit': commit,
            'evidence': evidence,
            'reported_at': datetime.now(UTC).isoformat(),
        }
        self._write_json('ready_to_merge.json', data)

    def read_ready_to_merge(self) -> dict | None:
        """Return the parsed ``.task/ready_to_merge.json`` artifact if
        present, else ``None``.
        """
        path = self.artifact_path('ready_to_merge.json')
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear_ready_to_merge(self) -> None:
        """Remove ``.task/ready_to_merge.json`` if present."""
        self._clear_path('ready_to_merge.json')

    def write_unactionable_task(self, reason: str, evidence: str) -> None:
        """Write ``.task/unactionable_task.json`` — architect's claim that
        the task spec itself is unworkable (false premise, contradiction,
        incompatible with main).

        The workflow short-circuits to a level-1 escalation, bypassing the
        steward — the auto-watcher triages; a human (L2) rewrites or cancels the task.
        """
        data = {
            'reason': reason,
            'evidence': evidence,
            'reported_at': datetime.now(UTC).isoformat(),
        }
        self._write_json('unactionable_task.json', data)

    def read_unactionable_task(self) -> dict | None:
        """Return the parsed ``.task/unactionable_task.json`` artifact if
        present, else ``None``.
        """
        path = self.artifact_path('unactionable_task.json')
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear_unactionable_task(self) -> None:
        """Remove ``.task/unactionable_task.json`` if present."""
        self._clear_path('unactionable_task.json')

    def write_false_premise(
        self,
        classification: str,
        premise: str,
        evidence: str,
        proposed_resolution: str,
    ) -> None:
        """Write ``.task/false_premise.json`` — architect's report that a
        RED-test premise is false, unreachable, or misattributed.

        The workflow short-circuits to a level-1 design_concern escalation,
        bypassing the steward — the auto-watcher triages; a human/curator (L2) re-specs the premise.
        """
        data = {
            'classification': classification,
            'premise': premise,
            'evidence': evidence,
            'proposed_resolution': proposed_resolution,
            'reported_at': datetime.now(UTC).isoformat(),
        }
        self._write_json('false_premise.json', data)

    def read_false_premise(self) -> dict | None:
        """Return the parsed ``.task/false_premise.json`` artifact if present,
        else ``None``.
        """
        path = self.artifact_path('false_premise.json')
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def clear_false_premise(self) -> None:
        """Remove ``.task/false_premise.json`` if present."""
        self._clear_path('false_premise.json')

    # ──────────────────────────────────────────────────────────────────
    # Review-state persistence (review_state.json)
    #
    # A single reseed-surviving artifact holding (a) the tree-hash-keyed
    # verdict cache — so a re-dispatch on an unchanged committed tree does
    # not re-mint a fresh reviewer nit — and (b) the task-lifetime
    # amendment/review counters, so ``max_amendment_rounds`` /
    # ``max_review_cycles`` bound the WHOLE task lifetime, not each
    # dispatch.  Fail-safe: a corrupt/absent file reads as defaults; the
    # cache is an optimization/churn guard, never load-bearing.
    # ──────────────────────────────────────────────────────────────────

    def read_review_state(self) -> dict:
        """Read ``review_state.json`` — the tree-hash verdict cache plus the
        task-lifetime amendment/review counters.

        Returns a fresh default state ``{'amendment_rounds_total': 0,
        'review_cycles_total': 0, 'verdicts': {}}`` when the file is absent,
        and returns those same defaults, logging a warning, when the file is
        corrupt, unreadable, not valid UTF-8, or not a JSON object — the
        fail-safe every ``_read_json_object`` reader shares.  A
        present-but-partial file is merged over the defaults so all canonical
        keys are always exposed.
        """
        default = {
            'amendment_rounds_total': 0,
            'review_cycles_total': 0,
            'verdicts': {},
        }
        data = self._read_json_object('review_state.json')
        if data is None:
            return default
        merged = {**default, **data}
        if not isinstance(merged.get('verdicts'), dict):
            merged['verdicts'] = {}
        return merged

    def record_review_verdict(
        self,
        tree_hash: str,
        verdict: str,
        suggestions_routed: bool,
        reviewer_fingerprint: str | None = None,
    ) -> None:
        """Record a non-blocking ``verdict`` for a committed ``tree_hash``.

        Read-modify-write of ``review_state.json`` that preserves the
        lifetime counters and any other recorded verdicts.  Only PASS /
        suggestions_only verdicts are ever recorded by the caller — a
        cache hit therefore unconditionally short-circuits REVIEW to DONE.

        ``reviewer_fingerprint`` (optional) captures the identity of the
        review inputs — the active reviewer roster and their resolved
        models — at the moment the verdict was minted.  The cache reader
        (``workflow._execute_verify_review_loop``) only honours a hit when
        this fingerprint still matches, so a reviewer-roster or per-role
        model-config change forces a fresh review even on a byte-identical
        committed tree (task 2749 amendment).  ``None`` is stored when the
        caller could not compute a fingerprint; such a record can never
        satisfy the reader's positive-match requirement, so it fails safe
        to a full review rather than a stale skip.
        """
        state = self.read_review_state()
        state['verdicts'][tree_hash] = {
            'verdict': verdict,
            'suggestions_routed': bool(suggestions_routed),
            'reviewer_fingerprint': reviewer_fingerprint,
            'ts': datetime.now(UTC).isoformat(),
        }
        self._write_json('review_state.json', state)

    def get_cached_verdict(self, tree_hash: str) -> dict | None:
        """Return the recorded verdict record for ``tree_hash``, else ``None``."""
        return self.read_review_state()['verdicts'].get(tree_hash)

    def get_amendment_rounds_total(self) -> int:
        """Return the persisted task-lifetime amendment-round count (0 default)."""
        try:
            return int(self.read_review_state().get('amendment_rounds_total', 0))
        except (TypeError, ValueError):
            return 0

    def get_review_cycles_total(self) -> int:
        """Return the persisted task-lifetime review-cycle count (0 default)."""
        try:
            return int(self.read_review_state().get('review_cycles_total', 0))
        except (TypeError, ValueError):
            return 0

    def set_review_counters(
        self,
        *,
        amendment_rounds_total: int | None = None,
        review_cycles_total: int | None = None,
    ) -> None:
        """Persist the task-lifetime review counters (read-modify-write).

        Updates only the provided keys, preserving the verdicts map and the
        other counter, so ``max_amendment_rounds`` / ``max_review_cycles``
        can bound the whole task lifetime rather than each dispatch.
        """
        state = self.read_review_state()
        if amendment_rounds_total is not None:
            state['amendment_rounds_total'] = int(amendment_rounds_total)
        if review_cycles_total is not None:
            state['review_cycles_total'] = int(review_cycles_total)
        self._write_json('review_state.json', state)

    def clear_review_counters(self) -> None:
        """Reset the task-lifetime amendment/review counters to zero.

        The operator/resume-path reset hook (task 2749 amendment).  The
        lifetime counters intentionally survive re-dispatches so restart
        churn / requeue / resume cannot re-grant a fresh
        ``max_amendment_rounds`` / ``max_review_cycles`` allowance.  That
        same persistence means a task that has EXHAUSTED its allowance and
        escalated keeps the exhausted counters across a human-driven
        re-pend: on the next blocking review the seeded local is already at
        the cap, so the loop re-escalates WITHOUT attempting a replan.

        Deliberately NOT auto-cleared on re-dispatch — the loop cannot tell
        churn (must keep the counters) from a legitimate operator re-pend
        (should reset them) without an out-of-band signal, so granting a
        fresh allowance is an EXPLICIT operator action, never inferred.
        When an operator resolves the review escalation and re-pends the
        task to make replan progress, call this hook (e.g. from the
        resume/re-pend path, or manually against the ``.task-meta/<name>/``
        store) to grant a fresh lifetime allowance.  The verdict cache is
        left intact — a genuine re-pend that changed the tree misses it
        anyway, and an unchanged tree with an unchanged config legitimately
        still skips.
        """
        state = self.read_review_state()
        state['amendment_rounds_total'] = 0
        state['review_cycles_total'] = 0
        self._write_json('review_state.json', state)

    def _read_json_object(self, name: str) -> dict | None:
        """The single fail-safe read seam for the whole-file JSON-object
        sidecars written by ``_write_json``.

        Returns ``None`` for an absent, unreadable, undecodable, malformed or
        non-object file, logging a warning for every case except absent.
        Decodes as UTF-8 to match ``shared.safe_io.atomic_write_text``.
        """
        path = self.artifact_path(name)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            logger.warning('Corrupt %s at %s: %s', name, path, exc)
            return None
        if not isinstance(data, dict):
            logger.warning('Malformed %s at %s: not an object', name, path)
            return None
        return data

    def _clear_path(self, name: str) -> None:
        """Remove artifact *name* if present."""
        self.artifact_path(name).unlink(missing_ok=True)

    def read_plan(self) -> dict:
        """Read current plan state, auto-normalizing malformed shapes."""
        plan = self.read_plan_model()
        return plan.to_json() if plan is not None else {}

    def peek_plan(self) -> dict:
        """Return the plan exactly as :meth:`read_plan` would, but never repair it on disk."""
        plan = self.peek_plan_model()
        return plan.to_json() if plan is not None else {}

    def read_plan_model(self) -> Plan | None:
        """The plan, None iff plan.json is absent; a malformed shape is repaired on disk and returned as written."""
        document, modified = self._read_normalized_plan()
        if modified:
            with self.plan_mutex():
                document, modified = self._read_normalized_plan()
                if modified and document is not None:
                    logger.warning(
                        'Plan normalization applied — writing corrected plan to %s',
                        self.artifact_path('plan.json'),
                    )
                    return self.write_plan_model(Plan.from_json(document), mutex_held=True)
        return Plan.from_json(document) if document is not None else None

    def peek_plan_model(self) -> Plan | None:
        """The plan exactly as :meth:`read_plan_model` would return it, but never repaired on disk.

        No ``plan_mutex`` is taken and nothing is written. For observers that
        poll a live task's plan (such as
        ``orchestrator/task_runtime.py::_read_task_entry``) and must not
        rewrite it under the agent that owns it.
        """
        document, needs_repair = self._read_normalized_plan()
        if document is None:
            return None
        plan = Plan.from_json(document)
        return _at_current_schema(plan) if needs_repair else plan

    def _read_normalized_plan(self) -> tuple[dict | None, bool]:
        plan_path = self.artifact_path('plan.json')
        if not plan_path.exists():
            return None, False
        return _normalize_plan(json.loads(plan_path.read_text()))

    def update_step_status(
        self, step_id: str, status: str, commit: str | None = None
    ) -> None:
        """Update ONLY the status and commit fields of one step in plan.json.

        Plan structure changes only through the architect-role revalidation
        tools (``orchestrator/mcp/plan_tools.py::_remove_plan_step`` /
        ``::_replace_plan_step``), and only on steps that are not done.
        Per-step deviation records go through :meth:`append_step_deviation`.
        """
        with self.plan_mutex():
            plan = self.read_plan()

            for collection in ('prerequisites', 'steps'):
                for item in plan.get(collection, []):
                    if not isinstance(item, dict):
                        continue
                    if item.get('id') == step_id:
                        item['status'] = status
                        if commit is not None:
                            item['commit'] = commit
                        self.write_plan_model(Plan.from_json(plan), mutex_held=True)
                        return

        logger.warning(f'Step {step_id} not found in plan')

    def append_step_deviation(
        self, step_id: str, record: DeviationRecord
    ) -> int | None:
        """Append *record* to the matching step's ``deviations`` list.

        Returns the list's new length, or ``None`` (with a warning, and no
        write) when no step or prerequisite has ``id == step_id``. Any step
        status is accepted, including ``done``.

        Append-only: this class never edits or removes a record. ``status``,
        ``commit``, ``type`` and ``description`` are never touched, so
        :meth:`update_step_status` stays "status and commit only".

        Markup-repair limits: ``plans/plan-deviation-recording-prd.md`` D4.
        """
        with self.plan_mutex():
            plan = self.read_plan()

            for collection in ('prerequisites', 'steps'):
                for item in plan.get(collection, []):
                    if not isinstance(item, dict):
                        continue
                    if item.get('id') == step_id:
                        existing = item.get('deviations')
                        item['deviations'] = [
                            *(existing if isinstance(existing, list) else []),
                            record.to_dict(),
                        ]
                        self.write_plan_model(Plan.from_json(plan), mutex_held=True)
                        return len(item['deviations'])

        logger.warning(f'Step {step_id} not found in plan')
        return None

    def mark_step_committed(self, step_id: str, sha: str) -> bool:
        """Pre-satisfy a plan step/prerequisite at AUTHORING time.

        Flips the matching item's ``status`` to ``'done'``, records the FULL
        *sha* in its ``commit`` field, and prepends a ``[COMMITTED <sha[:12]>]``
        provenance tag to its ``description``. Returns ``True`` on a match,
        ``False`` (with a warning mirroring :meth:`update_step_status`) when no
        step/prerequisite has ``id == step_id``.

        Unlike :meth:`update_step_status` — contractually "status and commit
        fields ONLY" — this DELIBERATELY mutates the description, which is why it
        is a dedicated method rather than an extension of that one. Idempotent
        per ``(step_id, sha)``: a repeated call keeps the item ``'done'`` and
        does NOT double-prepend the tag. A re-mark with a DIFFERENT sha REPLACES
        the tag (any stale leading ``[COMMITTED ...]`` tag is stripped first)
        rather than stacking provenance, so the description carries exactly the
        current commit's tag. Pure persistence, NO git — the
        corroborate-before-acting guard that *sha* actually resolves on-branch
        lives in ``plan_tools._sha_exists_on_branch``; this method does NOT
        prove the step's semantics (VERIFY remains the gate).
        """
        tag = f'[COMMITTED {sha[:12]}]'
        with self.plan_mutex():
            plan = self.read_plan()

            for collection in ('prerequisites', 'steps'):
                for item in plan.get(collection, []):
                    if not isinstance(item, dict):
                        continue
                    if item.get('id') == step_id:
                        item['status'] = 'done'
                        item['commit'] = sha
                        # Strip any existing leading [COMMITTED <sha>] tag(s) before
                        # prepending the current one: idempotent for a repeat of the
                        # same sha, and a clean replace (no stacking) for a re-mark
                        # with a different sha.
                        description = _COMMITTED_TAG_RE.sub('', item.get('description') or '')
                        item['description'] = f'{tag} {description}'
                        self.write_plan_model(Plan.from_json(plan), mutex_held=True)
                        return True

        logger.warning(f'Step {step_id} not found in plan')
        return False

    def get_pending_steps(self) -> list[dict]:
        """Return all steps with status 'pending', in order."""
        plan = self.read_plan()
        pending = []
        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if isinstance(item, dict) and item.get('status') == 'pending':
                    pending.append(item)
        return pending

    def get_completed_steps(self) -> list[dict]:
        """Return all steps with status 'done'."""
        plan = self.read_plan()
        completed = []
        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if isinstance(item, dict) and item.get('status') == 'done':
                    completed.append(item)
        return completed

    def append_iteration_log(self, entry: dict) -> None:
        """Append to iterations.jsonl — one JSON object per line.

        A missing ``source`` is stamped ``'orchestrator'``; any source outside
        :data:`ORCHESTRATOR_ITERATION_SOURCES` is refused before writing.
        """
        entry.setdefault('source', 'orchestrator')
        if not iteration_entry_written_by_orchestrator(entry):
            raise ValueError(
                f'iteration-log source must be one of '
                f'{sorted(ORCHESTRATOR_ITERATION_SOURCES)}, got {entry["source"]!r}'
            )
        entry['timestamp'] = datetime.now(UTC).isoformat()
        log_path = self.artifact_path('iterations.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def read_iteration_log(self) -> tuple[list[dict], list[str]]:
        """Read all iteration log entries, skipping corrupted lines.

        Returns (entries, corrupted) where corrupted contains raw lines that
        failed JSON parsing.
        """
        log_path = self.artifact_path('iterations.jsonl')
        if not log_path.exists():
            return [], []
        entries = []
        corrupted = []
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning('Skipping corrupted iteration log line: %s', line[:120])
                corrupted.append(line)
        return entries, corrupted

    def clear_iteration_log(self) -> None:
        """Remove ``iterations.jsonl`` if present.

        Called on re-dispatch onto a new fork point (Layer B, task 2372) so a
        prior dispatch's entries never masquerade as current-branch evidence
        — see ``_setup_worktree_and_artifacts``'s guarded wipe.
        """
        self._clear_path('iterations.jsonl')

    def write_review(self, reviewer_name: str, review: dict) -> None:
        """Write ``reviews/{reviewer_name}.json``.

        reviews/ is load-bearing: ``aggregate_reviews()`` is the review gate's
        only input. A skipped write would leave an aggregation with no
        blocking issues and no errors, which the gate reads as PASS, so a
        vanished root raises instead. The root is never recreated.

        Raises:
            ReviewsRootMissing: the root holding reviews/ no longer exists.
        """
        self._write_json(
            f'reviews/{reviewer_name}.json', review,
            on_root_gone=lambda root: ReviewsRootMissing(reviewer_name, root),
        )

    def read_reviews(self) -> dict[str, dict]:
        """Read all reviews from the reviews/ directory."""
        reviews: dict[str, dict] = {}
        reviews_dir = self.artifact_path('reviews')
        if reviews_dir.is_dir():
            for path in reviews_dir.glob('*.json'):
                reviews[path.stem] = json.loads(path.read_text())
        return reviews

    def clear_reviews(self) -> None:
        """Remove every review so the next aggregation holds only reviews written after this call."""
        reviews_dir = self.artifact_path('reviews')
        if not reviews_dir.is_dir():
            return
        removed = []
        for entry in reviews_dir.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink(missing_ok=True)
                removed.append(entry.name)
        if removed:
            logger.info('TaskArtifacts: cleared reviews %s from %s', sorted(removed), reviews_dir)

    def aggregate_reviews(self) -> ReviewAggregation:
        """Aggregate every review as ``orchestrator/review_payload.py::normalize_review`` reads it.

        ERROR reviews are kept out of ``reviews`` and returned verbatim in
        ``errored_reviews`` (``reviewer_errors`` is its names), so the review
        arm can read each errored reviewer's failure record. Issues outside the
        reviewer issue contract are flagged by a WARNING, never rewritten.
        """
        reviews = self.read_reviews()
        blocking = []
        suggestions = []
        errored_reviews = {}
        note_dispositions: list[NoteDisposition] = []

        for reviewer_name, review in reviews.items():
            payload = normalize_review(review)
            if payload.verdict is Verdict.ERROR:
                errored_reviews[reviewer_name] = review
            else:
                _warn_off_contract_issues(reviewer_name, payload.issues)
            note_dispositions.extend(payload.note_dispositions)
            blocking.extend(
                {**issue, 'reviewer': reviewer_name} for issue in payload.blocking_issues
            )
            suggestions.extend(
                {**issue, 'reviewer': reviewer_name} for issue in payload.suggestions
            )

        clean = {k: v for k, v in reviews.items() if k not in errored_reviews}
        return ReviewAggregation(
            has_blocking_issues=len(blocking) > 0,
            blocking_issues=blocking,
            suggestions=suggestions,
            reviews=clean,
            reviewer_errors=list(errored_reviews),
            errored_reviews=errored_reviews,
            note_dispositions=tuple(note_dispositions),
        )

    def validate_plan_prerequisites(self) -> None:
        """Validate that every prerequisite in plan.json is a dict with required keys.

        Required keys: id, description, status.

        Raises:
            ValueError: if any prerequisite is not a dict, or a dict is missing any
                of the required keys {id, description, status}.  The error message
                lists the offending indices and values so the operator can see exactly
                which prerequisites were malformed.
        """
        REQUIRED_KEYS = {'id', 'description', 'status'}
        plan = self.read_plan()
        prerequisites = plan.get('prerequisites', [])
        if not prerequisites:
            return None

        errors: list[str] = []
        for i, item in enumerate(prerequisites):
            if not isinstance(item, dict):
                errors.append(
                    f'  [index {i}] not a dict: {item!r}'
                )
            else:
                missing = REQUIRED_KEYS - item.keys()
                if missing:
                    errors.append(
                        f'  [index {i}] dict missing required keys {sorted(missing)}: {item!r}'
                    )

        if errors:
            raise ValueError(
                'plan.json prerequisites must be dicts with {id, description, status} keys. '
                'Found malformed prerequisites:\n' + '\n'.join(errors)
            )

        return None

    def bump_revalidation_stamp(
        self, session_id: str, base_commit: str | None = None,
    ) -> None:
        """Stamp ``_revalidated_at`` on plan.json and advance the plan's own
        ``_base_commit`` to *base_commit*. Mirrors the write semantics of the
        ``confirm_plan`` plan-tools MCP call but is invoked by orchestrator
        code (Lever B — overlap=0 revalidation skip) rather than by an
        architect tool call. metadata.json is never written here: its
        ``base_commit`` is this dispatch's worktree fork point, owned by
        ``init``/``update_base_commit`` alone (PRD task-meta-task-keying §M2).

        Args:
            session_id: Current orchestrator session id, written into
                ``_revalidated_by_session`` for telemetry / audit. The
                plan.json's existing ``_session_id`` (the original planner)
                is preserved.
            base_commit: The main SHA the plan has now been validated
                against. A falsy value leaves an existing stamp untouched.

        Raises:
            ValueError: if plan.json is missing or has no ``steps`` —
                callers should never invoke this on an empty plan.
        """
        with self.plan_mutex():
            plan = self.read_plan_model()
            if plan is None or not plan.steps:
                raise ValueError(
                    'bump_revalidation_stamp called before plan.json contains a '
                    'valid plan (missing or empty steps).'
                )
            revalidated = plan.revalidated(
                at=datetime.now(UTC).isoformat(), by_session=session_id, base_commit=base_commit,
            )
            self.write_plan_model(revalidated, mutex_held=True)

    def stamp_plan_provenance(
        self, session_id: str, *, task_text_fp: str | None = None,
        base_commit: str | None = None,
    ) -> Plan:
        """Stamp _session_id and _created_at into plan.json, and return the Plan written.

        A falsy ``task_text_fp`` leaves an existing fingerprint untouched (the
        per-iteration and recovered-plan stamps rely on this); a falsy
        ``base_commit`` likewise leaves an existing ``_base_commit`` untouched.

        Raises:
            ValueError: if plan.json is missing or does not contain a 'steps' list.
                This prevents silently creating a provenance-only stub that passes
                bool() checks but has no actual plan content.
        """
        now = datetime.now(UTC).isoformat()
        with self.plan_mutex():
            plan = self.read_plan_model()
            if plan is None or not plan.steps:
                raise ValueError(
                    'stamp_plan_provenance called before plan.json contains a valid plan '
                    '(missing or empty steps). '
                    'Ensure the architect has written a complete plan before stamping provenance.'
                )
            if plan.state is PlanState.PARTIAL:
                # Transitional until beta: plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md D3.
                logger.warning(
                    'stamp_plan_provenance: confirming the %s plan of task %r as %s before stamping '
                    '(lifecycle markers present: %s)',
                    plan.state.value, plan.body.get('task_id'), PlanConfirmation.INJECTED.value,
                    list(plan.lifecycle.markers_present()),
                )
                plan = plan.confirmed(at=now, by=PlanConfirmation.INJECTED)
            stamped = plan.stamped(
                session_id=session_id, at=now, task_text_fp=task_text_fp, base_commit=base_commit,
            )
            return self.write_plan_model(stamped, mutex_held=True)

    def validate_plan_owner(self, session_id: str) -> bool:
        """Return True if *session_id* owns plan.json.

        Two acceptance criteria:
        - ``_session_id == session_id`` — original planner / freshly stamped.
        - ``_revalidated_by_session == session_id`` — Lever B revalidation_skip
          preserves the original ``_session_id`` for audit but transfers
          ownership to the revalidating session via ``_revalidated_by_session``.

        Returns False on any read or parse error (JSONDecodeError, OSError, etc.)
        so that a corrupt or unreadable plan.json triggers the ownership-mismatch
        escalation path rather than a generic BLOCKED error.
        """
        try:
            plan_path = self.artifact_path('plan.json')
            data = json.loads(plan_path.read_text())
            return Plan.from_json(data).lifecycle.owned_by(session_id)
        except Exception as exc:
            logger.warning(
                'validate_plan_owner: failed to read/parse plan.json — treating as mismatch: %s',
                exc,
            )
            return False

    def set_plan_files(self, files: list[str], session_id: str) -> None:
        """Replace plan.json's ``files`` list in place — the orchestrator-side
        mechanical write used by ``TaskWorkflow._set_task_scope`` (the
        resume-path scope grant and the merge-gate narrowing pass) instead of
        re-invoking the architect for a known file set.

        Preserves every other plan field (steps, analysis, design_decisions,
        provenance) untouched — only ``files`` is replaced.

        Ownership: if *session_id* does not already own the plan (neither
        ``_session_id`` nor ``_revalidated_by_session`` matches), stamps
        ``_session_id`` = *session_id* and backfills ``_created_at`` (via
        setdefault, never clobbering an existing value) so that
        ``validate_plan_owner(session_id)`` is True afterward — this write
        must never trip ``_escalate_plan_overwrite``. When *session_id*
        already owns the plan via ``_revalidated_by_session``, provenance is
        left untouched: the original planner's ``_session_id`` is preserved
        for audit rather than being promoted.
        """
        with self.plan_mutex():
            plan = self.read_plan_model() or Plan.from_json({})
            edited = plan.with_files(files, owner=session_id, at=datetime.now(UTC).isoformat())
            self.write_plan_model(edited, mutex_held=True)

    # ──────────────────────────────────────────────────────────────────
    # Agent-session sidecar — marker for "an agent subprocess is in flight".
    #
    # Written before _invoke spawns the subprocess and deleted in the
    # invocation's finally block.  Presence after an orchestrator restart
    # signals that the prior orchestrator died mid-invocation; the harness
    # uses this to preserve the worktree and resume the same Claude session
    # via --resume rather than spawning a fresh agent.
    # ──────────────────────────────────────────────────────────────────

    def write_agent_session(
        self, session_id: str, role: str, started_at: str,
        task_id: str | None = None, resume_count: int = 0,
    ) -> None:
        """Write ``.task/agent_session.json`` with the in-flight session info.

        Always emits schema v2 (:data:`AGENT_SESSION_SCHEMA_VERSION`), so a
        pre-deploy v1 sidecar is rewritten as v2 on the next write (B11).
        ``task_id`` is the durable lane→task binding; ``resume_count`` is the
        cumulative adopted-resume count.
        """
        session = AgentSession(
            session_id=session_id,
            role=role,
            started_at=started_at,
            owner_pid=os.getpid(),
            task_id=task_id,
            resume_count=resume_count,
            schema_version=AGENT_SESSION_SCHEMA_VERSION,
        )
        self._write_json('agent_session.json', session.to_dict())

    def clear_agent_session(self) -> None:
        """Remove ``.task/agent_session.json`` if present (idempotent)."""
        self._clear_path('agent_session.json')

    def read_agent_session(self) -> AgentSession | None:
        """Return the parsed ``.task/agent_session.json`` as a typed
        :class:`AgentSession`, or ``None`` if missing/corrupt.

        Fail-safe: a missing file, corrupt/unreadable or non-UTF-8 JSON, a
        present-but-non-object payload, or a payload whose numeric fields
        (``resume_count`` / ``schema_version``) cannot be coerced to ``int``
        all return ``None`` (the caller reads absence as "no in-flight
        session").  A v1 sidecar (missing v2 keys) is tolerated via
        :meth:`AgentSession.from_mapping`, which supplies legacy defaults.
        """
        name = 'agent_session.json'
        data = self._read_json_object(name)
        if data is None:
            return None
        try:
            return AgentSession.from_mapping(data)
        except (ValueError, TypeError) as exc:
            # A non-numeric resume_count/schema_version (e.g. "abc" or null)
            # makes from_mapping's int() coercion raise; map that to the same
            # fail-safe None as any other corruption rather than propagating.
            logger.warning(
                'Malformed %s at %s: unparseable field (%s)',
                name, self.artifact_path(name), exc,
            )
            return None

    # ──────────────────────────────────────────────────────────────────
    # Reconcile-state sidecar — durable, cross-restart dedup for the
    # orphaned-done-step info escalations filed by
    # TaskWorkflow._reconcile_done_step_commits (task 2764).
    #
    # The workflow's in-memory (step_id, stale_commit) flood guard is
    # process-local, so an orchestrator restart re-files every prior pair.
    # These emitted keys are persisted here in the meta-root (which survives
    # worktree resets AND process restarts) and hydrated into the in-memory
    # set on the workflow's first reconcile pass, so a restarted orchestrator
    # files at most genuinely-new pairs.
    # ──────────────────────────────────────────────────────────────────

    _RECONCILE_STATE_NAME = 'reconcile_state.json'
    _EMITTED_STEP_ESCALATIONS_KEY = 'emitted_done_step_escalations'

    def read_emitted_step_escalations(self) -> set[tuple[str, str]]:
        """Return the set of ``(step_id, stale_commit)`` orphaned-done-step
        escalation keys already filed for this task, persisted across restarts.

        Stored at ``artifact_path('reconcile_state.json')`` (beside plan.json
        in the task root).  JSON has no set/tuple types, so the on-disk shape is
        ``{"emitted_done_step_escalations": [[step_id, commit], ...]}``; each
        2-element list of strings is re-tupled here; any other entry is
        skipped, with one warning counting the skipped entries.

        Fail-safe: a missing file, corrupt/unreadable or non-UTF-8 JSON, a
        non-object payload, or a non-list ``emitted_done_step_escalations``
        field all return an empty ``set``, so a corrupt sidecar re-files
        rather than silently suppressing.
        """
        name = self._RECONCILE_STATE_NAME
        key = self._EMITTED_STEP_ESCALATIONS_KEY
        data = self._read_json_object(name)
        if data is None:
            return set()
        entries = data.get(key, [])
        if not isinstance(entries, list):
            logger.warning(
                'Malformed %s at %s: %s is not a list',
                name, self.artifact_path(name), key,
            )
            return set()
        pairs = [
            (entry[0], entry[1])
            for entry in entries
            if isinstance(entry, list)
            and len(entry) == 2
            and all(isinstance(x, str) for x in entry)
        ]
        if len(pairs) < len(entries):
            logger.warning(
                'Malformed %s at %s: skipped %d malformed %s entries',
                name, self.artifact_path(name), len(entries) - len(pairs), key,
            )
        return set(pairs)

    def append_emitted_step_escalation(
        self, step_id: str, stale_commit: str
    ) -> None:
        """Durably record that a ``(step_id, stale_commit)`` orphaned-done-step
        escalation was filed, so a restarted workflow does not re-file it.

        Read-modify-write: reads the current set via
        :meth:`read_emitted_step_escalations`, adds the pair, and rewrites
        ``reconcile_state.json`` with the keys serialized as a sorted list of
        2-element lists (stable diffs; JSON has no set/tuple types).

        Idempotent *without* redundant disk work (task 2764 amend): if the pair
        is already persisted, return early and skip the rewrite entirely, so a
        repeat append — or a caller that forgets the workflow's in-memory flood
        guard — costs one read and no write.  This dedup-before-write preserves
        the escalate-first-then-persist ordering: the early return fires only
        when a prior persist already succeeded, so a genuinely-new pair (the
        only case the caller normally reaches) is never skipped.
        """
        keys = self.read_emitted_step_escalations()
        if (step_id, stale_commit) in keys:
            return
        keys.add((step_id, stale_commit))
        self._write_json(
            self._RECONCILE_STATE_NAME,
            {
                self._EMITTED_STEP_ESCALATIONS_KEY: sorted(
                    [k[0], k[1]] for k in keys
                )
            },
        )

    # ──────────────────────────────────────────────────────────────────
    # Verdict artifacts — written by the per-worktree verdict-tools MCP
    # server (task 2481). One file per role under verdicts/<role>.json;
    # the role-derived filename is authoritative (never an agent-supplied
    # field), so a reviewer cannot misname its artifact onto a sibling's.
    # ──────────────────────────────────────────────────────────────────

    def write_verdict(self, role: str, envelope: dict) -> None:
        """Write ``verdicts/{role}.json``.

        Best-effort through ``_write_json``'s default root-gone no-op: a
        vanished root (worktree gone out-of-band) skips the write and is never
        recreated. Skipping fails closed, because ``_run_reviewer`` treats a
        missing verdict as ERROR.

        Raises:
            ValueError: if *role* is invalid — see ``_validate_verdict_role``.
        """
        _validate_verdict_role(role)
        self._write_json(f'verdicts/{role}.json', envelope)

    def read_verdict(self, role: str) -> dict | None:
        """Return parsed ``.task/verdicts/{role}.json``, or ``None`` if the
        file is missing, corrupt, or not a JSON object.

        Raises:
            ValueError: if *role* is invalid — see ``_validate_verdict_role``.
        """
        _validate_verdict_role(role)
        return self._read_json_object(f'verdicts/{role}.json')

    def clear_verdict(self, role: str) -> None:
        """Remove ``.task/verdicts/{role}.json`` if present (idempotent).

        Called before spawning a role's agent so a stale verdict from a
        prior invocation never masquerades as this run's output.

        Raises:
            ValueError: if *role* is invalid — see ``_validate_verdict_role``.
        """
        _validate_verdict_role(role)
        self._clear_path(f'verdicts/{role}.json')

    def lock_plan(self, session_id: str, *, run_id: str | None = None) -> bool:
        """Atomically acquire the plan lock.

        Uses O_CREAT|O_EXCL for atomic exclusive creation (POSIX).
        Returns True if the lock was acquired, False if already locked.
        Raises ValueError if os.getpid() returns a non-int (unexpected env).

        *run_id* is the PROCESS-level run id (``Harness._run_id``, threaded
        through as ``TaskWorkflow._process_run_id``).  Recording it lets
        ``TaskGroundTruth._resolve_live_claimant`` compose a full
        ``shared.task_claimant.compose_claimant_run_id`` identity from the
        lock — byte-identical to the DB claimant stamp that this same
        incarnation writes, since ``owner_pid`` below is this same process's
        pid and the caller passes this same ``session_id`` (task 3563).

        The ``run_id`` key is written ONLY when it is known and non-blank, so
        its ABSENCE unambiguously means "unknown" — the state of every legacy
        lock already on disk and of harness-less (test/eval) workflows, whose
        ``_process_run_id`` is None.  TaskGroundTruth reads that as an unknown
        claimant identity (``Claimant.run_id is None``) rather than composing
        a partial one: a well-shaped-but-wrong identity would string-mismatch
        the DB-composed filing identity and be read downstream as "a DIFFERENT
        incarnation is live", which is the unsafe direction.
        """
        lock_path = self.artifact_path('plan.lock')
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        try:
            owner_pid = os.getpid()
            if not isinstance(owner_pid, int):
                raise ValueError(
                    f'plan.lock owner_pid must be a non-null int, got {owner_pid!r}'
                )
            payload = {
                'session_id': session_id,
                'locked_at': datetime.now(UTC).isoformat(),
                'owner_pid': owner_pid,
            }
            # Omit entirely when unknown — never write '' (see docstring).
            if isinstance(run_id, str) and run_id.strip():
                payload['run_id'] = run_id
            data = json.dumps(payload)
            os.write(fd, data.encode())
        except Exception:
            # Clean up the empty file created by O_CREAT|O_EXCL so a subsequent
            # lock_plan call is not falsely blocked by a poison-pill empty file.
            lock_path.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        return True

    def is_plan_locked(self) -> bool:
        """Return True if plan.lock exists."""
        return self.artifact_path('plan.lock').exists()

    def read_plan_lock(self) -> dict | None:
        """Read plan.lock contents. Returns dict or None if not locked."""
        lock_path = self.artifact_path('plan.lock')
        if not lock_path.exists():
            return None
        return json.loads(lock_path.read_text())

    def clear_stale_plan_lock(
        self, current_task_id: str, stale_threshold_secs: float = 600.0
    ) -> bool:
        """Remove plan.lock if it's stale. Returns True if lock was cleared.

        *stale_threshold_secs* is plan.lock's own eviction window; its default
        equals ``shared.task_claimant.DEFAULT_CLAIMANT_HEARTBEAT_TTL`` only
        coincidentally and is deliberately not derived from it.
        """
        lock_data = self.read_plan_lock()
        if lock_data is None:
            return False

        lock_session = lock_data.get('session_id', '')
        locked_at_str = lock_data.get('locked_at', '')

        # Self-lock from a crashed previous session of the same task — always clear.
        # Session IDs are formatted as '{task_id}-{uuid_hex[:8]}', so a lock
        # from any prior session of the same task shares the task_id prefix.
        if lock_session.startswith(current_task_id + '-'):
            logger.info(
                'Clearing self-owned plan.lock (session %s, task %s)',
                lock_session,
                current_task_id,
            )
            self.artifact_path('plan.lock').unlink(missing_ok=True)
            return True

        # Age-based stale detection
        try:
            locked_at = datetime.fromisoformat(locked_at_str)
            age_secs = (datetime.now(UTC) - locked_at).total_seconds()
            if age_secs > stale_threshold_secs:
                logger.warning(
                    'Clearing stale plan.lock (session %s, age %.0fs > %.0fs threshold)',
                    lock_session,
                    age_secs,
                    stale_threshold_secs,
                )
                self.artifact_path('plan.lock').unlink(missing_ok=True)
                return True
        except (ValueError, TypeError):
            # Unparseable timestamp — treat as stale
            logger.warning(
                'Clearing plan.lock with unparseable timestamp: %r', locked_at_str
            )
            self.artifact_path('plan.lock').unlink(missing_ok=True)
            return True

        return False

    def _write_json(
        self, name: str, data: dict,
        *, on_root_gone: Callable[[Path], FileNotFoundError] | None = None,
    ) -> None:
        """Write *data* to artifact *name* (routed by ``artifact_path``) as
        JSON, ATOMICALLY and DURABLY.

        The single write seam behind every whole-file `.task-meta` artifact
        (metadata, plan, review_state, reviews, verdicts, agent_session,
        reconcile_state, the architect's report artifacts), and therefore the
        single owner of their byte format and their durability guarantees.

        It is also the single owner of what a vanished root (the worktree
        deleted out-of-band) means for a write. A write never recreates its
        root: only ``init()`` creates roots, so the write creates just the
        artifact's own directory directly below the root. A root found gone,
        before or during the write, makes the write a logged no-op. A caller
        whose artifact must not be lost passes *on_root_gone*, and the write
        raises ``on_root_gone(root)`` instead.

        What a concurrent reader may rely on: it observes either the complete
        OLD contents or the complete NEW contents — never a truncated file,
        and never a missing one.  ``os.replace`` on a temp written in the
        destination's own directory is what buys that, by itself and for
        almost nothing.  (The superseded ``path.write_text`` was
        truncate-then-write, and a reader racing it could and did observe a
        half-written file — see ``TestWriteJsonIsAtomic``.)

        ``fsync=True`` is a SEPARATE guarantee, deliberately bought, not a
        free extension of that one — and it is the dominant cost of this
        method.  MEASURED on this repo's own filesystem (ext4 on NVMe, a
        5.5 KB plan.json, 200-write loops): ``write_text`` 0.18 ms/write,
        atomic WITHOUT fsync 0.27 ms, atomic WITH fsync 5.26 ms.  Atomicity
        costs ~0.1 ms; durability costs ~5 ms on top — ~20x more — and every
        caller of this seam pays it.  ``shared.safe_io`` defaults the flag to
        False for precisely that reason, so overriding its default here is a
        decision, and it rests on three things:

        * Passing ``fsync=False`` would be a REGRESSION rather than a neutral
          default.  The plan-tools plan.json writer this method absorbed
          fsynced its temp before every replace, so a non-fsyncing
          ``_write_json`` would silently weaken plan.json's existing
          guarantee on the very change that consolidated it.
          ``atomic_write_text``'s flag is both-or-neither (temp file AND
          parent directory), so the real choice was "keep it for every
          artifact" or "lose it for plan.json".
        * These artifacts ARE the crash-recovery inputs.  A resumed
          orchestrator reads plan.json / metadata.json / review_state.json /
          agent_session.json back to learn what a task already did; an
          artifact that is atomically written but not durable is exactly what
          a host crash turns into a silently rewound task.  The
          parent-directory half earns its keep for the same reason — without
          it a crash can leave fully synced bytes unreachable under a name
          that was never persisted.
        * The cost is bounded because nothing behind this seam is a hot loop.
          These are lifecycle writes — at most tens per task across all ~13
          artifacts — so ~5 ms apiece totals well under a second per task,
          once, against a task lifetime measured in minutes.

        A per-call-site durability flag was rejected: it would make an
        artifact's crash-safety depend on which method happened to write it,
        and no current site has a defensible reason to opt out.  Should a
        future caller genuinely write in a loop, give THAT caller the flag
        rather than flipping the default underneath the recovery artifacts.
        The ``fsync`` kwarg this contract turns on is pinned by
        ``TestWriteJsonDelegatesToSharedSafeIo``.

        An EXISTING target keeps its own permission bits across the swap (see
        :func:`_existing_mode`), which the superseded ``write_text`` got for
        free by truncating rather than recreating; a new one lands with the
        umask-derived mode ``write_text`` would have given it.

        A SYMLINKED target is written THROUGH, as ``write_text`` wrote through
        it: the path is resolved first and the real file is what gets replaced.
        This is load-bearing, not hygiene — ``ensure_lane_symlinks`` makes
        ``<worktree>/.task/plan.json`` a symlink onto the meta-root plan, and
        a bare replace onto the link path would swap it for a regular file,
        silently re-forking the lane and meta-root copies (the esc-5205-9
        stale-plan divergence).  Resolving also guarantees the temp shares a
        filesystem with the real file, so the replace is an intra-filesystem
        rename rather than EXDEV.  A DANGLING link is REFUSED with
        :class:`ArtifactWriteError` rather than materialising a stray regular
        file at the link path — that stray file would BE the divergence.

        The tmp+rename itself is DELEGATED to ``shared.safe_io`` rather than
        re-implemented here: that helper is the repo's single blessed home for
        the pattern (its unique-per-writer O_CREAT|O_EXCL temp, fchmod on the
        still-open fd, and BaseException-safe cleanup are all things a local
        copy would have to re-earn), and the consolidation is machine-enforced
        by ``TestNoRegrownAtomicWriters`` in
        ``tests/scripts/test_atomic_write_regrowth.py``.

        ``append_iteration_log`` is deliberately NOT routed through here: it
        is genuinely append-only, and tmp+rename would turn an O(1) append
        into an O(n) rewrite while destroying the partial-read tolerance that
        makes a truncated log still usable line-by-line.
        """
        path = self.artifact_path(name)
        root = self._root_for(Path(name).parts[0])
        try:
            # Resolve FIRST — write through a symlink, never onto it.  Order
            # inside this block is deliberate: the refusal must precede any
            # filesystem mutation, so dangling check → directory creation →
            # mode lookup → write.
            target = Path(os.path.realpath(path))
            if path.is_symlink() and not target.parent.is_dir():
                raise ArtifactWriteError(
                    f'atomic artifact write to {path} refused: its resolved '
                    f'target {target} has no existing parent directory (a '
                    'dangling symlink?) — refusing to materialise a stray '
                    'file at the link path'
                )
            if path.parent != root:
                path.parent.mkdir(exist_ok=True)
            # ``_existing_mode`` is inside the try so a NON-FileNotFoundError
            # stat failure (EACCES/ELOOP/ENAMETOOLONG) surfaces through the
            # same path as any other write failure.  It swallows
            # FileNotFoundError ITSELF and reports the target as new, so a
            # vanished target does NOT short-circuit here — it reaches
            # ``atomic_write_text``, whose temp-file create raises
            # FileNotFoundError when the root is gone.  The root-gone branch
            # below is therefore reached from the directory creation or the
            # WRITE, never from the mode lookup.
            safe_io.atomic_write_text(
                target,
                json.dumps(data, indent=2) + '\n',  # byte format UNCHANGED
                mode=_existing_mode(target),
                fsync=True,
            )
        except FileNotFoundError as exc:
            if root.is_dir():
                raise
            if on_root_gone is not None:
                raise on_root_gone(root) from exc
            logger.info(
                'TaskArtifacts: skipping _write_json to %s — root %s no longer exists',
                path, root,
            )
