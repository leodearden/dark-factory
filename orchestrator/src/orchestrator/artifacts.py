"""Manages the .task/ directory structure in each worktree."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.config import TASK_META_DIRNAME

logger = logging.getLogger(__name__)

PLAN_SCHEMA_VERSION = 1

# Matches one-or-more leading ``[COMMITTED <hex>]`` provenance tags (with any
# trailing whitespace) at the START of a step description.  Used by
# :meth:`TaskArtifacts.mark_step_committed` to strip a stale tag before
# prepending the current one, so re-marking a step against a DIFFERENT sha
# (e.g. an amended/squashed commit during re-planning) REPLACES the tag rather
# than stacking provenance.  The ``+`` also cleans up any tags that a prior
# (pre-strip) implementation may have already stacked.
_COMMITTED_TAG_RE = re.compile(r'^(?:\[COMMITTED [0-9a-fA-F]+\]\s*)+')

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
                logger.warning('Normalizing plan: renaming %r → "steps"', alias)
                plan['steps'] = plan.pop(alias)
                modified = True
                break

    # Rule 3: flatten nested tdd_implementation_plan phases
    if 'steps' not in plan and _NESTED_PLAN_KEY in plan:
        nested = plan[_NESTED_PLAN_KEY]
        if isinstance(nested, dict):
            logger.warning(
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
            logger.warning('Normalizing plan: wrapping %d string steps as dicts',
                           sum(1 for s in steps if isinstance(s, str)))
            plan['steps'] = new_steps
            modified = True

    return plan, modified


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


@dataclass
class ReviewAggregation:
    """Aggregated review results from all reviewers."""

    has_blocking_issues: bool
    blocking_issues: list[dict]
    suggestions: list[dict]
    reviews: dict[str, dict]
    reviewer_errors: list[str] = field(default_factory=list)

    @property
    def all_reviewers_errored(self) -> bool:
        """True when every reviewer returned ERROR (no usable reviews)."""
        return bool(self.reviewer_errors) and not self.reviews

    def format_for_replan(self) -> str:
        """Format blocking issues for the architect to address."""
        lines = ['# Review Feedback — Blocking Issues\n']
        for issue in self.blocking_issues:
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


class TaskArtifacts:
    """Manages .task/ directory in a worktree."""

    def __init__(self, worktree: Path, meta_root: Path | None = None):
        """
        Args:
            worktree: The task's git worktree.
            meta_root: Optional `.task-meta` root (see ``meta_root_for``).
                When omitted (``None``), ``self.root`` falls back to the
                legacy ``<worktree>/.task`` path and this class behaves
                byte-identically to before this argument existed. When
                supplied, all reads and writes target ``meta_root`` only —
                the legacy path is never consulted.
        """
        self.worktree = worktree
        self.root = meta_root if meta_root is not None else worktree / '.task'

    @staticmethod
    def meta_root_for(worktree_base: Path, worktree_name: str) -> Path:
        """Derive the `.task-meta` root for a worktree — the single owner of
        this path shape (PRD `.task-meta path-derivation contract`,
        worktree-lane-lifecycle W11-β).

        Returns ``<worktree_base>/.task-meta/<worktree_name>`` — a SIBLING
        of the worktree itself (``<worktree_base>/<worktree_name>``), not
        nested inside it like the legacy ``<worktree>/.task``.  Callers
        (γ/ε1/ε2/δ) must call this instead of joining ``'.task'`` (or
        ``'.task-meta'``) by hand.

        Lane-reuse note for whichever of γ/ε1/ε2/δ first constructs
        ``TaskArtifacts`` with a real ``meta_root``: because this path lives
        OUTSIDE the worktree, it survives a worktree reset/``git clean`` that
        would wipe a legacy ``<worktree>/.task``. With a pooled/reused lane
        (e.g. the ``_lane-0``-style names used in this module's tests), a new
        task assigned a previously-used lane name could otherwise see the
        PRIOR task's leftover ``.task-meta/<name>`` artifacts directly (a
        stale ``blocking_dependency``/``already_done``/``false_premise``
        report). Nothing in β triggers this today (no caller passes
        ``meta_root`` yet); the first caller that does so for a pooled lane
        is responsible for clearing/re-initializing ``.task-meta/<name>`` on
        lane acquisition. See ``plans/worktree-lane-lifecycle-prd.md``.
        """
        return worktree_base / TASK_META_DIRNAME / worktree_name

    def init(
        self,
        task_id: str,
        task_title: str,
        task_description: str,
        base_commit: str | None = None,
    ) -> None:
        """Create .task/ with initial metadata.json.

        IMPORTANT: This method is safe to call when .task/ already exists
        (e.g. inherited from a contaminated main).
        """
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'reviews').mkdir(exist_ok=True)
        (self.root / 'verdicts').mkdir(exist_ok=True)

        metadata = {
            'task_id': task_id,
            'title': task_title,
            'description': task_description,
            'created_at': datetime.now(UTC).isoformat(),
        }
        if base_commit:
            metadata['base_commit'] = base_commit
        self._write_json(self.root / 'metadata.json', metadata)

    def read_base_commit(self) -> str | None:
        """Read the base commit SHA stored at init time."""
        meta_path = self._read_path('metadata.json')
        if not meta_path.exists():
            return None
        metadata = json.loads(meta_path.read_text())
        return metadata.get('base_commit')

    def read_created_at(self) -> str | None:
        """Read the created_at timestamp stored at init time.

        Mirrors ``read_base_commit``'s shape, but is exception-tolerant for a
        corrupt/unreadable metadata.json (returns ``None`` instead of
        raising) so a runtime "started" lookup never raises.
        """
        meta_path = self._read_path('metadata.json')
        if not meta_path.exists():
            return None
        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Corrupt metadata.json at %s: %s', meta_path, exc)
            return None
        return metadata.get('created_at')

    def update_base_commit(self, new_base: str) -> None:
        """Update the base_commit in metadata.json after a rebase."""
        meta_path = self._read_path('metadata.json')
        if not meta_path.exists():
            return
        metadata = json.loads(meta_path.read_text())
        metadata['base_commit'] = new_base
        self._write_json(self.root / 'metadata.json', metadata)

    def write_plan(self, plan: dict) -> None:
        """Write .task/plan.json — the structured plan."""
        plan['_schema_version'] = PLAN_SCHEMA_VERSION
        self._write_json(self.root / 'plan.json', plan)

    def ensure_lane_plan_symlink(self) -> None:
        """Single-source plan.json onto the durable meta-root copy.

        Makes the lane copy ``<worktree>/.task/plan.json`` an ABSOLUTE symlink
        to ``self.root / 'plan.json'`` (the durable meta-root plan). Any
        residual reader of the lane path then transparently resolves to the
        meta-root file, so the lane copy can never diverge from the meta-root —
        the esc-5205-9 stale trap (a remediation agent appended steps to a
        regular-FILE lane copy that ``mark_step_done`` then rejected) becomes
        impossible because the lane path IS the meta-root file.

        Idempotent and recreated each dispatch: replaces a pre-existing regular
        file, a stale/broken symlink, or a pathological stray directory. No-op
        in legacy ``meta_root=None`` mode, where the lane path already IS the
        durable copy (``self.root == <worktree>/.task``) so a self-referential
        symlink would be broken/nonsensical.
        """
        if self.root == self.worktree / '.task':
            # Legacy meta_root=None mode: the lane path already IS the durable
            # copy — a self-referential symlink would be broken/nonsensical.
            return
        lane_task = self.worktree / '.task'
        # meta_root mode's init() never creates <worktree>/.task, so make it.
        lane_task.mkdir(parents=True, exist_ok=True)
        lane_plan = lane_task / 'plan.json'
        # Clear any occupant: a regular file (the esc-5205-9 stale trap), a
        # stale/broken symlink, or the pathological stray directory.
        if lane_plan.is_symlink() or lane_plan.exists():
            if lane_plan.is_dir() and not lane_plan.is_symlink():
                shutil.rmtree(lane_plan)
            else:
                lane_plan.unlink(missing_ok=True)
        os.symlink(self.root / 'plan.json', lane_plan)

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
        self._write_json(self.root / 'blocking_dependency.json', data)

    def read_blocking_dependency(self) -> dict | None:
        """Return the parsed ``.task/blocking_dependency.json`` artifact if
        present, else ``None``.
        """
        path = self._read_path('blocking_dependency.json')
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
        self._write_json(self.root / 'already_done.json', data)

    def read_already_done(self) -> dict | None:
        """Return the parsed ``.task/already_done.json`` artifact if present,
        else ``None``.
        """
        path = self._read_path('already_done.json')
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
        self._write_json(self.root / 'ready_to_merge.json', data)

    def read_ready_to_merge(self) -> dict | None:
        """Return the parsed ``.task/ready_to_merge.json`` artifact if
        present, else ``None``.
        """
        path = self._read_path('ready_to_merge.json')
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
        self._write_json(self.root / 'unactionable_task.json', data)

    def read_unactionable_task(self) -> dict | None:
        """Return the parsed ``.task/unactionable_task.json`` artifact if
        present, else ``None``.
        """
        path = self._read_path('unactionable_task.json')
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
        self._write_json(self.root / 'false_premise.json', data)

    def read_false_premise(self) -> dict | None:
        """Return the parsed ``.task/false_premise.json`` artifact if present,
        else ``None``.
        """
        path = self._read_path('false_premise.json')
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
        and — mirroring ``read_created_at``'s fail-safe (:264-279) — logs a
        warning and returns those same defaults on a corrupt/unreadable or
        malformed file rather than raising.  A present-but-partial file is
        merged over the defaults so all canonical keys are always exposed.
        """
        default = {
            'amendment_rounds_total': 0,
            'review_cycles_total': 0,
            'verdicts': {},
        }
        path = self._read_path('review_state.json')
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Corrupt review_state.json at %s: %s', path, exc)
            return default
        if not isinstance(data, dict):
            logger.warning(
                'Malformed review_state.json at %s: not an object', path
            )
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
        self._write_json(self.root / 'review_state.json', state)

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
        self._write_json(self.root / 'review_state.json', state)

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
        self._write_json(self.root / 'review_state.json', state)

    def _read_path(self, name: str) -> Path:
        """Resolve *name* under ``self.root``."""
        return self.root / name

    def _clear_path(self, name: str) -> None:
        """Remove *name* from ``self.root`` if present."""
        (self.root / name).unlink(missing_ok=True)

    def read_plan(self) -> dict:
        """Read current plan state, auto-normalizing malformed shapes."""
        plan_path = self._read_path('plan.json')
        if not plan_path.exists():
            return {}
        plan = json.loads(plan_path.read_text())
        plan, modified = _normalize_plan(plan)
        if modified:
            logger.warning(
                'Plan normalization applied — writing corrected plan to %s',
                self.root / 'plan.json',
            )
            self._write_json(self.root / 'plan.json', plan)
        return plan

    def update_step_status(
        self, step_id: str, status: str, commit: str | None = None
    ) -> None:
        """Update ONLY status and commit fields in plan.json. Structure is immutable."""
        plan = self.read_plan()

        for collection in ('prerequisites', 'steps'):
            for item in plan.get(collection, []):
                if not isinstance(item, dict):
                    continue
                if item.get('id') == step_id:
                    item['status'] = status
                    if commit is not None:
                        item['commit'] = commit
                    self._write_json(self.root / 'plan.json', plan)
                    return

        logger.warning(f'Step {step_id} not found in plan')

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
        plan = self.read_plan()
        tag = f'[COMMITTED {sha[:12]}]'

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
                    self.write_plan(plan)
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
        """Append to .task/iterations.jsonl — one JSON object per line."""
        entry['timestamp'] = datetime.now(UTC).isoformat()
        log_path = self.root / 'iterations.jsonl'
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def read_iteration_log(self) -> tuple[list[dict], list[str]]:
        """Read all iteration log entries, skipping corrupted lines.

        Returns (entries, corrupted) where corrupted contains raw lines that
        failed JSON parsing.
        """
        log_path = self._read_path('iterations.jsonl')
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
        """Write .task/reviews/{name}.json.

        Best-effort: if the artifacts root has been removed (worktree gone
        out-of-band), skip the write rather than raising — the review JSON
        is a debugging aid, not load-bearing for workflow correctness. This
        method assumes ``init()`` has already created ``self.root``:
        reviews are a post-implementation workflow phase, so init() has
        always run by the time a reviewer's output is written.
        """
        if not self.root.is_dir():
            logger.info(
                'TaskArtifacts: skipping write_review %r — root %s no longer exists',
                reviewer_name, self.root,
            )
            return
        review_path = self.root / 'reviews' / f'{reviewer_name}.json'
        self._write_json(review_path, review)

    def read_reviews(self) -> dict[str, dict]:
        """Read all reviews from the reviews/ directory."""
        reviews: dict[str, dict] = {}
        reviews_dir = self.root / 'reviews'
        if reviews_dir.is_dir():
            for path in reviews_dir.glob('*.json'):
                reviews[path.stem] = json.loads(path.read_text())
        return reviews

    def aggregate_reviews(self) -> ReviewAggregation:
        """Parse all reviews, separate blocking from suggestions.

        Reviews with verdict ``ERROR`` are filtered into a separate list
        so the workflow can detect incomplete reviewer coverage.
        """
        reviews = self.read_reviews()
        blocking = []
        suggestions = []
        errors = []

        for reviewer_name, review in reviews.items():
            if review.get('verdict') == 'ERROR':
                errors.append(reviewer_name)
                continue
            for issue in review.get('issues', []):
                enriched = {**issue, 'reviewer': reviewer_name}
                if issue.get('severity') == 'blocking':
                    blocking.append(enriched)
                else:
                    suggestions.append(enriched)

        clean = {k: v for k, v in reviews.items() if k not in errors}
        return ReviewAggregation(
            has_blocking_issues=len(blocking) > 0,
            blocking_issues=blocking,
            suggestions=suggestions,
            reviews=clean,
            reviewer_errors=errors,
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
        """Stamp ``_revalidated_at`` on plan.json and update metadata.json's
        ``base_commit`` to *base_commit*. Mirrors the write semantics of the
        ``confirm_plan`` plan-tools MCP call but is invoked by orchestrator
        code (Lever B — overlap=0 revalidation skip) rather than by an
        architect tool call.

        Args:
            session_id: Current orchestrator session id, written into
                ``_revalidated_by_session`` for telemetry / audit. The
                plan.json's existing ``_session_id`` (the original planner)
                is preserved.
            base_commit: New main SHA the plan is now tracked against. When
                ``None``, metadata.json is left untouched (typical for tests
                that exercise only the plan.json side-effect).

        Raises:
            ValueError: if plan.json is missing or has no ``steps`` —
                callers should never invoke this on an empty plan.
        """
        plan = self.read_plan()
        if not plan.get('steps'):
            raise ValueError(
                'bump_revalidation_stamp called before plan.json contains a '
                'valid plan (missing or empty steps).'
            )
        plan['_revalidated_at'] = datetime.now(UTC).isoformat()
        plan['_revalidated_by_session'] = session_id
        self._write_json(self.root / 'plan.json', plan)
        if base_commit is not None:
            self.update_base_commit(base_commit)

    def stamp_plan_provenance(self, session_id: str) -> None:
        """Stamp _session_id and _created_at into plan.json.

        Reads current plan.json, adds provenance fields, writes back.

        Raises:
            ValueError: if plan.json is missing or does not contain a 'steps' list.
                This prevents silently creating a provenance-only stub that passes
                bool() checks but has no actual plan content.
        """
        plan = self.read_plan()
        if not plan.get('steps'):
            raise ValueError(
                'stamp_plan_provenance called before plan.json contains a valid plan '
                '(missing or empty steps). '
                'Ensure the architect has written a complete plan before stamping provenance.'
            )
        plan['_session_id'] = session_id
        plan['_created_at'] = datetime.now(UTC).isoformat()
        # Invariant: a provenance-stamped plan is a complete plan, so it must
        # carry the completeness marker.  The architect's confirm_plan normally
        # sets this already (idempotent setdefault); paths that inject a
        # ready-made plan (eval initial_plan, SIMPLE_TASK) rely on this so the
        # plan is not mistaken for an interrupted partial on a later requeue.
        plan.setdefault('_finalized_at', plan['_created_at'])
        self._write_json(self.root / 'plan.json', plan)

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
            plan_path = self._read_path('plan.json')
            data = json.loads(plan_path.read_text())
            if data.get('_session_id') == session_id:
                return True
            return data.get('_revalidated_by_session') == session_id
        except Exception as exc:
            logger.warning(
                'validate_plan_owner: failed to read/parse plan.json — treating as mismatch: %s',
                exc,
            )
            return False

    def set_plan_files(self, files: list[str], session_id: str) -> None:
        """Widen plan.json's ``files`` list in place — the orchestrator-side
        mechanical union used by the resume-path scope-grant consumer
        (``TaskWorkflow._set_task_scope``) instead of re-invoking the
        architect for a known file-set append.

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
        plan = self.read_plan()
        plan['files'] = files
        already_owner = (
            plan.get('_session_id') == session_id
            or plan.get('_revalidated_by_session') == session_id
        )
        if not already_owner:
            plan['_session_id'] = session_id
            plan.setdefault('_created_at', datetime.now(UTC).isoformat())
        self._write_json(self.root / 'plan.json', plan)

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
        self._write_json(self.root / 'agent_session.json', session.to_dict())

    def clear_agent_session(self) -> None:
        """Remove ``.task/agent_session.json`` if present (idempotent)."""
        self._clear_path('agent_session.json')

    def read_agent_session(self) -> AgentSession | None:
        """Return the parsed ``.task/agent_session.json`` as a typed
        :class:`AgentSession`, or ``None`` if missing/corrupt.

        Fail-safe: a missing file, corrupt/unreadable JSON, a present-but-
        non-object payload, or a payload whose numeric fields
        (``resume_count`` / ``schema_version``) cannot be coerced to ``int``
        all return ``None`` (the caller reads absence as "no in-flight
        session").  A v1 sidecar (missing v2 keys) is tolerated via
        :meth:`AgentSession.from_mapping`, which supplies legacy defaults.
        """
        path = self._read_path('agent_session.json')
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Corrupt agent_session.json at %s: %s', path, exc)
            return None
        if not isinstance(data, dict):
            logger.warning(
                'Malformed agent_session.json at %s: not an object', path
            )
            return None
        try:
            return AgentSession.from_mapping(data)
        except (ValueError, TypeError) as exc:
            # A non-numeric resume_count/schema_version (e.g. "abc" or null)
            # makes from_mapping's int() coercion raise; map that to the same
            # fail-safe None as any other corruption rather than propagating.
            logger.warning(
                'Malformed agent_session.json at %s: unparseable field (%s)',
                path, exc,
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

    def read_emitted_step_escalations(self) -> set[tuple[str, str]]:
        """Return the set of ``(step_id, stale_commit)`` orphaned-done-step
        escalation keys already filed for this task, persisted across restarts.

        Stored at ``self.root/reconcile_state.json`` (beside plan.json in the
        meta-root).  JSON has no set/tuple types, so the on-disk shape is
        ``{"emitted_done_step_escalations": [[step_id, commit], ...]}``; each
        2-element list is re-tupled here and malformed entries are skipped.

        Fail-safe: a missing file, corrupt/unreadable JSON, or a non-object
        payload all return an empty ``set`` (mirrors ``read_agent_session``),
        so a corrupt sidecar re-files rather than silently suppressing.
        """
        path = self._read_path('reconcile_state.json')
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Corrupt reconcile_state.json at %s: %s', path, exc)
            return set()
        if not isinstance(data, dict):
            logger.warning(
                'Malformed reconcile_state.json at %s: not an object', path
            )
            return set()
        result: set[tuple[str, str]] = set()
        for entry in data.get('emitted_done_step_escalations', []):
            # Each entry must be a 2-element [step_id, commit] list of strings;
            # skip anything malformed rather than raising (fail-safe read).
            if (
                isinstance(entry, list)
                and len(entry) == 2
                and all(isinstance(x, str) for x in entry)
            ):
                result.add((entry[0], entry[1]))
        return result

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
            self.root / 'reconcile_state.json',
            {
                'emitted_done_step_escalations': sorted(
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
        """Write ``.task/verdicts/{role}.json``.

        Best-effort: if the artifacts root has been removed (worktree gone
        out-of-band), skip the write rather than raising — mirrors
        ``write_review``'s root-gone guard. This method assumes ``init()``
        has already created ``self.root``.

        Raises:
            ValueError: if *role* is invalid — see ``_validate_verdict_role``.
        """
        _validate_verdict_role(role)
        if not self.root.is_dir():
            logger.info(
                'TaskArtifacts: skipping write_verdict %r — root %s no longer exists',
                role, self.root,
            )
            return
        verdict_path = self.root / 'verdicts' / f'{role}.json'
        self._write_json(verdict_path, envelope)

    def read_verdict(self, role: str) -> dict | None:
        """Return parsed ``.task/verdicts/{role}.json``, or ``None`` if
        missing/corrupt.

        Raises:
            ValueError: if *role* is invalid — see ``_validate_verdict_role``.
        """
        _validate_verdict_role(role)
        path = self._read_path(f'verdicts/{role}.json')
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Corrupt verdicts/%s.json at %s: %s', role, path, exc)
            return None

    def clear_verdict(self, role: str) -> None:
        """Remove ``.task/verdicts/{role}.json`` if present (idempotent).

        Called before spawning a role's agent so a stale verdict from a
        prior invocation never masquerades as this run's output.

        Raises:
            ValueError: if *role* is invalid — see ``_validate_verdict_role``.
        """
        _validate_verdict_role(role)
        self._clear_path(f'verdicts/{role}.json')

    def lock_plan(self, session_id: str) -> bool:
        """Atomically acquire the plan lock.

        Uses O_CREAT|O_EXCL for atomic exclusive creation (POSIX).
        Returns True if the lock was acquired, False if already locked.
        Raises ValueError if os.getpid() returns a non-int (unexpected env).
        """
        lock_path = self.root / 'plan.lock'
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
            data = json.dumps({
                'session_id': session_id,
                'locked_at': datetime.now(UTC).isoformat(),
                'owner_pid': owner_pid,
            })
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
        return (self.root / 'plan.lock').exists()

    def read_plan_lock(self) -> dict | None:
        """Read plan.lock contents. Returns dict or None if not locked."""
        lock_path = self.root / 'plan.lock'
        if not lock_path.exists():
            return None
        return json.loads(lock_path.read_text())

    def clear_stale_plan_lock(
        self, current_task_id: str, stale_threshold_secs: float = 600.0
    ) -> bool:
        """Remove plan.lock if it's stale. Returns True if lock was cleared."""
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
            (self.root / 'plan.lock').unlink(missing_ok=True)
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
                (self.root / 'plan.lock').unlink(missing_ok=True)
                return True
        except (ValueError, TypeError):
            # Unparseable timestamp — treat as stale
            logger.warning(
                'Clearing plan.lock with unparseable timestamp: %r', locked_at_str
            )
            (self.root / 'plan.lock').unlink(missing_ok=True)
            return True

        return False

    def _write_json(self, path: Path, data: dict) -> None:
        # Tolerate a vanished worktree.  Callers that care about durability
        # (init, plan, base_commit, etc.) check ``self.root`` themselves; the
        # opportunistic writes (reviews, iteration log) just want a no-op
        # when the worktree has been deleted out-of-band.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + '\n')
        except FileNotFoundError:
            if not self.root.is_dir():
                logger.info(
                    'TaskArtifacts: skipping _write_json to %s — root %s no longer exists',
                    path, self.root,
                )
                return
            raise
