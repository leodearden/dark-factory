"""Detect whether a live workflow (orchestrator worktree) is active for a given task.

Provides a per-task/branch liveness signal that complements the project-level
``is_orchestrator_live_for`` check.  Three OR-ed signals bias toward 'live' so
recon does not race a working pipeline:

1. **worktree_registered** — ``git worktree list --porcelain`` lists a worktree
   whose branch is ``task/<task_id>``.
2. **recent_commit** — the tip of ``task/<task_id>`` is newer than
   *max_commit_age_hours* (default: 6 h), detected via ``git log -1 --format=%cI``.
3. **orchestrator_live** — ``is_orchestrator_live_for(project_root)`` returns True
   (the project-level orchestrator-lock signal).

``is_live = worktree_registered OR recent_commit OR orchestrator_live``

Fail-safe contract: a missing branch, subprocess error, or unparseable timestamp
makes *that individual signal* False without raising.  OR-aggregation means a
transient git glitch never flips a genuinely live task to 'not live'.

**Important — orchestrator_live is project-wide, not per-task.**
``is_orchestrator_live_for(project_root)`` holds one PID lock for the entire
project.  When it is True, *every* task in the project will have ``is_live=True``
regardless of its worktree or commit recency.  This is intentional: while any
task in the project is being actively dispatched, recon cannot distinguish which
specific task the orchestrator is acting on without deeper orchestrator coupling,
so it conservatively treats all tasks as owned.  Consequence: while the project
orchestrator is alive, stranded-work escalations are suppressed project-wide,
not just for the task the orchestrator is currently processing.

**Status scoping.** Callers that know the task's current status may pass it via
the ``status`` keyword.  When ``status`` is one of ``ORCH_LIVE_INELIGIBLE_STATUSES``
(statuses that are never actively dispatched: ``deferred``, ``done``,
``cancelled``), the project-wide ``orchestrator_live`` signal is forced to
``False`` for that call — the orchestrator being alive elsewhere is not evidence
that *this* task is live.  The per-task ``worktree_registered`` and
``recent_commit`` signals are unaffected by ``status``, so genuine per-branch
evidence still marks the task live.  ``status=None`` (the default) preserves the
prior, status-blind behavior.

**Blocked deterministic tasks.** Callers may also pass ``task_kind`` (the task's
``metadata.task_kind``).  Deliberately, ``'blocked'`` is NOT added to
``ORCH_LIVE_INELIGIBLE_STATUSES`` wholesale — a normal blocked task may
auto-unblock and be legitimately mid-pipeline, so the project-wide orchestrator
lock remains real per-task evidence for it (task 2031).  However, a
**deterministic** task (``task_kind == 'deterministic'``) never acquires a
worktree/branch — it is routed to ``DeterministicRunner`` instead — so for a
*blocked deterministic* task the bare orchestrator lock is not task-specific
evidence either.  ``detect_live_workflow`` therefore also forces
``orchestrator_live`` to ``False`` when ``status == 'blocked' AND task_kind ==
'deterministic'`` (see :func:`_orchestrator_signal_ineligible`).  This is a
compound condition, not a status addition: non-deterministic blocked tasks
(``task_kind`` absent or not ``'deterministic'``) keep the signal.  The
per-task ``worktree_registered``/``recent_commit`` signals remain unaffected,
so a blocked deterministic task that somehow did acquire a worktree is still
live.

**Blocked normal tasks (task 2409).** The same bare-orchestrator-lock problem
also affects **normal** (``task_kind`` absent or ``'normal'``) blocked tasks:
a blocked, normal-kind task with satisfied dependencies and no genuine live
pipeline can show only the project-wide ``orchestrator`` signal and be treated
as owned/live indefinitely, so reconciliation/redispatch never fires — the
repeated re-deferral loop observed for tasks 2335/2196.  ``detect_live_workflow``
therefore also forces ``orchestrator_live`` to ``False`` when ``status ==
'blocked' AND task_kind in (None, 'normal') AND NOT worktree_registered AND NOT
recent_commit``.  Unlike the deterministic rule, this one is guarded on the
per-task git signals: a normal blocked task *can* legitimately hold a
registered worktree or a recent commit while auto-unblocking mid-pipeline
(task 2031's original concern), so the guard only suppresses the bare case —
whenever real per-task evidence exists, ``orchestrator_live`` is reported
honestly and the task remains live via that evidence (``is_live`` is
unaffected either way, since it already ORs in ``worktree_registered``/
``recent_commit``).

The legitimate stranded case (orchestrator down, no worktree, no recent commits)
has all three signals False, so recon still escalates it.

**Pending deterministic pure gates (task 3751).** A third bare-orchestrator-lock
false positive affects **pending** deterministic tasks of the PURE-GATE shape —
``always_escalates`` truthy AND ``before_done`` absent (see
:func:`is_pure_gate_metadata`).  ``DeterministicRunner`` dispatches such a task
through section 3 only: file one born-at-L2 escalation (deduped), stamp
``metadata.gate_escalated_at``, set status ``blocked``.  No script, no systemd,
no ``git_ops`` — and, like every deterministic task, no worktree/branch ever.
So while it is pending the bare project-wide lock is never task-specific
evidence for it.  ``detect_live_workflow`` therefore forces ``orchestrator_live``
to ``False`` when ``status == 'pending' AND task_kind == 'deterministic' AND
pure_gate`` (rule 5 of :func:`_orchestrator_signal_ineligible`), with
``pure_gate`` passed in by the callers that hold the task's metadata.

This is deliberately NOT a wholesale addition of ``'pending'`` to
``ORCH_LIVE_INELIGIBLE_STATUSES``.  Beyond breaking every ordinary pending task
(which is dispatch-eligible), it would race deterministic tasks that DO carry a
``before_done``: ``Harness._run_deterministic_slot`` never flips a deterministic
task to ``'in-progress'`` before invoking the runner, so such a task stays
``'pending'`` for the whole duration of a blocking deploy/predicate script plus
systemd inspection and fresh-PID verification — minutes of real, side-effecting
work with zero git evidence to reveal it.  Restricting the exemption to the
pure-gate subclass keeps it exactly as wide as the class of tasks that provably
cannot be mid-side-effect.  Confirmed incident: dark_factory task **3845** (a
pending ``always_escalates`` deterministic gate with no ``before_done``, verified
by direct ``get_task`` read) stalled 3+ consecutive reconciliation cycles (runs
0e98f0ac, f112de8), listed in Stage 2's Live-Workflow Signals with ONLY the bare
``orchestrator`` signal, which permanently blocked its recon disposition; tasks
3741/3749 were earlier observed instances of the same shape.

**Reaped worktrees and bare branches (task 2767, reify#5245).** A worktree
entry in ``git worktree list --porcelain`` marked ``prunable`` (its directory
was removed/reaped, but the registration itself was not yet pruned) does NOT
count toward ``worktree_registered`` — only a LIVE (non-prunable) worktree
does. Likewise, a branch with zero commits of its own beyond ``base_branch``
(checked via ``git rev-list --count <base_branch>..<branch>``; "bare" — e.g.
a branch whose reflog holds only a "Created from main" entry) has its tip
timestamp stripped from ``recent_commit``, since that timestamp is just the
base-branch commit, not evidence of task work. When BOTH hold — the branch is
bare AND no live worktree is registered AND there is no recent commit —
:func:`_orchestrator_signal_ineligible`'s rule 4 additionally drops the
project-wide ``orchestrator_live`` signal, **regardless of status or
task_kind** (unlike rules 1-3): this is the exact reify#5245 shape, where a
task with no live work of its own kept showing worktree+orchestrator signals
solely because a *different*, genuinely live task shared the same project-wide
orchestrator lock. Both gates are fail-safe TOWARD live: a missing branch, a
subprocess error, or an unparseable ``rev-list`` count never marks a worktree
prunable or a branch bare — only positive evidence does.

**In-progress corroboration gate (task 2963).** The three rules above force
the *project-wide* ``orchestrator_live`` signal False for statuses/task_kinds
where the bare lock is not per-task evidence. A separate, orthogonal failure
mode remains for **in-progress** tasks after a fleet redeploy: the redeploy
KILLS the running workflow, yet its git worktree registration lingers
(``worktree_registered`` stays True until pruned) and the freshly-restarted
orchestrator re-acquires the project-wide PID lock (``orchestrator_live`` stays
True), so an in-progress task whose only signals are those two — no
``recent_commit`` — is falsely reported live. ``detect_live_workflow`` accepts
an opt-in ``corroborated: bool | None`` verdict: when ``status ==
'in-progress'`` AND not ``recent_commit`` AND (``worktree_registered`` OR
``orchestrator_live``) AND ``corroborated is False`` (the caller found no fresh
per-task corroborating signal — see :func:`corroboration_for_task`), ``is_live``
is forced False and the result is flagged ``indeterminate=True`` (distinct from
an all-signals-false genuine not-live result). ``corroborated is None`` (the
default, used by every existing caller and by the two consumers that lack the
task dict) or ``True`` leaves the detector byte-for-byte unchanged. This lets
the stranded-remediation path proceed for a killed-but-lingering in-progress
task while never racing a genuinely live pipeline (any one fresh signal keeps
it live). ``recent_commit`` is deliberately exempt — it is genuine per-task
work evidence.

Branch convention: ``task/<task_id>`` (matches the orchestrator's worktree naming).
Injectable ``now`` for deterministic tests.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from shared.task_claimant import has_live_claimant
from shared.task_metadata import RoutingState

from fused_memory.services.orchestrator_detector import is_orchestrator_live_for

logger = logging.getLogger(__name__)

# Default age threshold for the recent-commit signal.  A commit newer than this
# value (relative to ``now``) counts as evidence of an active workflow.
DEFAULT_MAX_COMMIT_AGE_HOURS: float = 6.0

# Orchestrator branch naming convention: ``task/<id>``.
DEFAULT_BRANCH_PREFIX: str = 'task/'

# Default base branch a task branch is created from.  Used to compute a
# branch's own commit count via ``git rev-list --count <base>..<branch>`` —
# see `_branch_own_commit_count` and the `base_branch` param on
# `detect_live_workflow`.
DEFAULT_BASE_BRANCH: str = 'main'

# Timeout for each individual git subprocess call (seconds).
_GIT_TIMEOUT: int = 10

# Statuses that are never actively dispatched by the orchestrator.  For a task
# in one of these statuses, the project-wide orchestrator PID-lock signal is not
# evidence of a live pipeline for *that* task, so it is dropped (see `status`
# param on detect_live_workflow).  Deliberately excludes 'blocked' (may
# auto-unblock), 'review'/'merge-deferred' (in the verify/review/merge
# pipeline), and 'pending'/'in-progress' (dispatch-eligible or active) — for
# those statuses a live orchestrator legitimately means "don't race the
# pipeline."  A blocked *deterministic* task is handled separately by
# `_orchestrator_signal_ineligible` below — see its docstring — rather than by
# adding 'blocked' here, since that would wrongly suppress normal blocked tasks.
# The one 'pending' exception is likewise handled there, by rule 5 (task 3751):
# a pending deterministic PURE GATE (`always_escalates` truthy, `before_done`
# absent) is exempt, because its whole DeterministicRunner run is
# file-escalation-then-block. It is scoped as a compound rule rather than a
# status addition here because a pending deterministic task WITH `before_done`
# can be mid-deploy inside the runner (its status is never flipped to
# 'in-progress'), and every ordinary pending task is dispatch-eligible.
ORCH_LIVE_INELIGIBLE_STATUSES: frozenset[str] = frozenset({'deferred', 'done', 'cancelled'})

# task_kind value used by the orchestrator/scheduler for deterministic tasks
# (DeterministicRunner-routed: no worktree/branch is ever acquired). Mirrors
# Scheduler._is_deterministic's metadata.task_kind == 'deterministic' check.
DETERMINISTIC_TASK_KIND: str = 'deterministic'

# task_kind value (and documented default when metadata.task_kind is unset) for
# ordinary LLM-agent-pipeline tasks. Used by rule 3 of
# _orchestrator_signal_ineligible (task 2409) to scope the blocked-normal
# bare-orchestrator suppression.
NORMAL_TASK_KIND: str = 'normal'

# Heartbeat staleness threshold for the fresh-claimant corroboration signal
# (task 2963). A killed workflow keeps its durable ``claimant_run_id`` but stops
# advancing ``heartbeat_at``, so FRESHNESS (not presence) is the discriminator:
# `has_live_claimant(task, now, ttl)` reports live only while the heartbeat is
# newer than ``now - ttl``. Mirrors the orchestrator's validated
# ``_RECONCILE_HEARTBEAT_TTL`` (harness.py:189) / ``_DEFAULT_HEARTBEAT_TTL``
# (task_ground_truth.py:146) constant — the coordination point with sibling
# task 2931's orphan-reaper liveness predicate (reuse this shared TTL, do not
# invent a parallel heartbeat check).
DEFAULT_HEARTBEAT_TTL: timedelta = timedelta(minutes=10)


@dataclass(frozen=True)
class WorkflowLiveness:
    """Result of :func:`detect_live_workflow`.

    Attributes:
        is_live: True when any of the three signals indicates an active workflow.
        worktree_registered: True when a LIVE (non-prunable) git worktree is
            registered for ``branch``. A worktree entry marked ``prunable`` in
            ``git worktree list --porcelain`` (its directory has been removed
            or reaped, but the registration itself has not yet been pruned)
            does NOT count — a reaped worktree is not a live workspace.
        recent_commit: True when the tip of ``branch`` is newer than the threshold.
        orchestrator_live: True when the project-level orchestrator lock is live.
        branch: The branch name inspected (e.g. ``task/4321``).
        last_commit_at: Parsed commit timestamp when ``recent_commit`` was evaluated.
            ``None`` when the branch was absent or the timestamp was unparseable.
        indeterminate: True when the per-task corroboration gate (task 2963)
            downgraded ``is_live`` to False for an in-progress task whose only
            liveness signals were the project-wide ``orchestrator_live`` and/or
            a lingering ``worktree_registered`` (no ``recent_commit``) and for
            which the caller supplied ``corroborated=False`` (no fresh per-task
            signal). This is SEMANTICALLY DISTINCT from an all-signals-false
            genuine not-live result (``indeterminate`` False): the former had a
            worktree/orchestrator signal but no corroboration — stranded but was
            dispatched — while the latter never had any signal. Both carry
            ``is_live=False`` so ``if is_live`` consumers stop treating the task
            as owned, permitting the stranded-remediation path. Always False
            unless the gate fired.
    """

    is_live: bool
    worktree_registered: bool
    recent_commit: bool
    orchestrator_live: bool
    branch: str
    last_commit_at: datetime | None
    indeterminate: bool = False


def detect_live_workflow(
    task_id: str,
    project_root: str | Path,
    *,
    now: datetime | None = None,
    max_commit_age_hours: float = DEFAULT_MAX_COMMIT_AGE_HOURS,
    branch_prefix: str = DEFAULT_BRANCH_PREFIX,
    base_branch: str = DEFAULT_BASE_BRANCH,
    status: str | None = None,
    task_kind: str | None = None,
    pure_gate: bool | None = None,
    corroborated: bool | None = None,
    _orchestrator_live: bool | None = None,
) -> WorkflowLiveness:
    """Detect whether a live workflow is active for *task_id*.

    Args:
        task_id: Task identifier (numeric string, e.g. ``"4321"``).
        project_root: Absolute path to the git repository root.
        now: Reference time for the recent-commit age check.  Defaults to
            ``datetime.now(UTC)`` when ``None``.  Pass a fixed datetime in
            tests for determinism.
        max_commit_age_hours: Commits newer than this many hours count as recent.
        branch_prefix: Branch name prefix; combined with *task_id* to form the
            branch name (e.g. ``"task/4321"``).
        base_branch: The branch *branch* is created from.  Used to compute the
            branch's own commit count via ``git rev-list --count
            <base_branch>..<branch>`` — a count of ``0`` means the branch is
            "bare" (no commits beyond *base_branch*, e.g. only a
            `git worktree add`/branch-creation reflog entry), which forces
            ``recent_commit`` to ``False`` even if the branch tip's timestamp
            would otherwise be within *max_commit_age_hours* (the tip is just
            the base-branch commit, not task work — reify#5245's shape). A
            branch missing entirely, or any rev-list error, yields an unknown
            count and does NOT count as bare (fail-safe toward live).
        status: The task's current status, when known.  When this is a member
            of :data:`ORCH_LIVE_INELIGIBLE_STATUSES` (statuses never actively
            dispatched: ``deferred``, ``done``, ``cancelled``), the project-wide
            ``orchestrator_live`` signal is forced to ``False`` for this call,
            short-circuiting before ``_orchestrator_live``/
            ``is_orchestrator_live_for`` are consulted.  ``None`` (default)
            leaves the orchestrator_live computation unaffected — fully
            backward compatible.
        task_kind: The task's ``metadata.task_kind``, when known.  When
            ``status == 'blocked'`` and ``task_kind == DETERMINISTIC_TASK_KIND``,
            the project-wide ``orchestrator_live`` signal is also forced to
            ``False`` — a deterministic task never acquires a worktree/branch,
            so the bare project lock is not task-specific evidence for it while
            blocked.  Likewise (task 2409), when ``status == 'blocked'`` and
            ``task_kind`` is ``None`` or ``NORMAL_TASK_KIND`` and there is
            neither a registered worktree nor a recent commit for the task's
            own branch, the signal is forced ``False`` too — see
            :func:`_orchestrator_signal_ineligible` for both rules.  For a
            blocked task with genuine per-task evidence (a registered worktree
            or a recent commit), or for any non-blocked status, the
            orchestrator_live computation is unaffected by ``task_kind``.
            ``task_kind`` also scopes rule 5 via ``pure_gate`` (below).
        pure_gate: Whether the task's metadata has the ``DeterministicRunner``
            PURE-GATE shape (``always_escalates`` truthy AND ``before_done``
            absent), as classified by :func:`is_pure_gate_metadata` — passed in
            by the callers that hold the task's metadata dict. When ``status ==
            'pending'`` AND ``task_kind == DETERMINISTIC_TASK_KIND`` AND this is
            truthy, the project-wide ``orchestrator_live`` signal is forced
            ``False`` (rule 5, task 3751): such a task's entire run is "file one
            escalation, stamp ``gate_escalated_at``, set blocked", so it can
            never be mid-side-effect and the bare project lock is not evidence
            for it. ``None`` (the default) and ``False`` are equivalent and
            leave the rule inert, so the detector is byte-for-byte unchanged for
            every caller that does not pass this kwarg — including a pending
            deterministic task WITH a ``before_done``, which may be mid-deploy
            inside ``DeterministicRunner`` and must keep the signal. The
            per-task ``worktree_registered``/``recent_commit`` signals are
            unaffected, so a pure gate that somehow did acquire a worktree is
            still ``is_live``.
        corroborated: Per-task corroboration verdict for the in-progress
            liveness gate (task 2963). Only an explicit ``False`` downgrades:
            when ``status == 'in-progress'`` AND there is no ``recent_commit``
            AND at least one of ``worktree_registered``/``orchestrator_live``
            is set AND ``corroborated is False`` (the caller found no fresh
            per-task corroborating signal — live claimant/heartbeat, scheduler
            holder/park, or a post-restart routing decision), ``is_live`` is
            forced ``False`` and ``indeterminate`` is set ``True`` on the
            result. This targets the fleet-redeploy false positive: a KILLED
            in-progress workflow whose lingering worktree registration and the
            freshly-restarted project-wide orchestrator lock both still assert
            liveness. ``recent_commit`` is EXEMPT (genuine per-task work
            evidence). ``corroborated is None`` (the default — used by every
            existing caller, and by ``recon_write_policy``/integrity-escalation
            which lack the task dict) and ``corroborated is True`` both leave
            the gate inert, so the detector's behavior is byte-for-byte
            unchanged for all existing callers. The verdict is computed by
            :func:`corroboration_for_task` and passed in only by
            :func:`_render_live_workflow_section`, which HAS the task dict.
        _orchestrator_live: Pre-computed project-level orchestrator-lock result.
            When provided, skips the ``is_orchestrator_live_for(project_root)``
            call — use this to hoist the constant project-level check out of
            per-task loops (e.g. in :func:`_render_live_workflow_section`).
            ``None`` (default) triggers a fresh ``is_orchestrator_live_for``
            call.  Tests monkeypatch the module attribute directly; this
            parameter is only for performance hoisting, not test isolation.
            Ignored when :func:`_orchestrator_signal_ineligible` returns True
            for the given *status*/*task_kind* pair.

    Returns:
        A :class:`WorkflowLiveness` dataclass with all signals populated.
    """
    branch = f'{branch_prefix}{task_id}'
    root = str(project_root)

    worktree_present, worktree_prunable = _check_worktree_registered(root, branch)
    worktree_registered = worktree_present and not worktree_prunable
    last_commit_at, recent_commit = _check_recent_commit(
        root, branch, now=now, max_commit_age_hours=max_commit_age_hours
    )
    # branch_bare: branch carries zero commits of its own beyond base_branch
    # (its tip is just the base-branch commit — reify#5245's shape). An
    # unknown count (missing branch, rev-list error) is NOT bare (fail-safe
    # toward live). A bare branch's recent-looking tip timestamp is not
    # evidence of task work, so it is stripped from recent_commit here.
    #
    # Skip the rev-list subprocess call when its result cannot change any
    # output field: if a LIVE worktree is already registered, worktree_registered
    # is already True, so rule 4 below (which requires `not worktree_registered`)
    # can never fire regardless of branch_bare; and if recent_commit is already
    # False, `recent_commit and not branch_bare` stays False regardless of
    # branch_bare. When BOTH hold, branch_bare cannot affect anything it feeds
    # into, so computing it is skipped — saving a git subprocess call on this
    # common already-live path, which matters when this detector is fanned out
    # across many tasks in a recon sweep.
    if worktree_registered and not recent_commit:
        branch_bare = False
    else:
        own_commit_count = _branch_own_commit_count(root, base_branch, branch)
        branch_bare = own_commit_count == 0
    recent_commit = recent_commit and not branch_bare
    # orchestrator_live is the project-level lock signal (True when the
    # orchestrator process holds an active lock for this project_root, regardless
    # of which task it is currently dispatching).  Pre-computed callers may pass
    # it via _orchestrator_live to avoid redundant per-task subprocess calls.
    # Ineligible status/task_kind combinations (see
    # _orchestrator_signal_ineligible) force this signal False — the
    # project-wide lock is not evidence of liveness for a task that will never
    # be dispatched (or, for blocked deterministic tasks, never acquires a
    # worktree/branch of its own; or, for blocked normal tasks with no
    # per-task git evidence, task 2409; or, for pending deterministic PURE
    # GATES whose whole run is file-escalation-then-block, task 3751; or,
    # status-agnostically, for a task whose branch is provably bare with its
    # worktree reaped, task 2767).  The FINAL worktree_registered/recent_commit
    # (post-prunable/post-bare, already computed above) and branch_bare are
    # threaded through so rules 3 and 4 can evaluate the same evidence the
    # caller sees.  pure_gate is normalized to a bool here so the default None
    # (every caller that does not classify metadata) leaves rule 5 inert.
    if _orchestrator_signal_ineligible(
        status, task_kind, worktree_registered, recent_commit, branch_bare, bool(pure_gate)
    ):
        orchestrator_live = False
    else:
        orchestrator_live = (
            _orchestrator_live
            if _orchestrator_live is not None
            else is_orchestrator_live_for(project_root)
        )

    # Per-task corroboration gate (task 2963). Composes with — and runs after —
    # the _orchestrator_signal_ineligible block above (which may already have
    # zeroed orchestrator_live). For an IN-PROGRESS task whose liveness rests
    # SOLELY on the project-wide orchestrator lock and/or a lingering worktree
    # registration (no recent_commit), a fleet redeploy that KILLED the
    # workflow leaves those two signals falsely asserting liveness. An explicit
    # corroborated=False (the caller found no fresh per-task signal) downgrades
    # is_live to False and flags the result indeterminate — distinct from an
    # all-signals-false genuine not-live result (indeterminate stays False when
    # there was no worktree/orchestrator signal to downgrade). corroborated is
    # None (default, every existing caller) or True leaves behavior unchanged.
    # recent_commit is exempt: it is genuine per-task work evidence.
    gate_fires = (
        status == 'in-progress'
        and not recent_commit
        and (worktree_registered or orchestrator_live)
        and corroborated is False
    )
    if gate_fires:
        is_live = False
        indeterminate = True
    else:
        is_live = worktree_registered or recent_commit or orchestrator_live
        indeterminate = False

    return WorkflowLiveness(
        is_live=is_live,
        worktree_registered=worktree_registered,
        recent_commit=recent_commit,
        orchestrator_live=orchestrator_live,
        branch=branch,
        last_commit_at=last_commit_at,
        indeterminate=indeterminate,
    )


def is_workflow_live_for_task(
    task_id: str,
    project_root: str | Path,
    **kwargs,
) -> bool:
    """Convenience wrapper — returns :attr:`WorkflowLiveness.is_live`.

    Accepts the same keyword arguments as :func:`detect_live_workflow`,
    including the ``_orchestrator_live`` performance hint.
    """
    return detect_live_workflow(task_id, project_root, **kwargs).is_live


# ---------------------------------------------------------------------------
# Per-task corroboration (task 2963)
# ---------------------------------------------------------------------------
#
# For an IN-PROGRESS task whose liveness rests SOLELY on the project-wide
# ``orchestrator_live`` and/or the (lingering) ``worktree_registered`` signal —
# i.e. no ``recent_commit`` — a fleet redeploy that KILLED the workflow leaves
# both those signals asserting stale liveness. These helpers require at least
# one FRESH per-task corroborating signal before the caller treats such a task
# as live; when none is present the caller downgrades ``is_live`` to False and
# marks the result ``indeterminate`` (see the gate in :func:`detect_live_workflow`).
#
# Three corroborating signals, ANY sufficient:
#   1. Fresh claimant/heartbeat — ``has_live_claimant(task, now, ttl)``.
#   2. Scheduler holder/park — task_id in ``parks`` keys OR ``current_holders``
#      values (both reset by an orchestrator restart).
#   3. routing.latest.decided_at newer than the orchestrator's start time (a
#      dispatch predating the last restart cannot have a live workflow).


def has_live_workflow_corroboration(
    task_id: str | int,
    *,
    claimant_live: bool,
    scheduler_state: Mapping | None,
    routing_latest_decided_at: str | None,
    orchestrator_started_at: datetime | None,
) -> bool:
    """Return True when at least one fresh per-task corroborating signal exists.

    ORs the three corroboration signals (see module section above). A True
    ``claimant_live`` short-circuits before the scheduler/routing checks are
    consulted. All inputs are pre-extracted so this predicate stays pure and
    unit-testable; :func:`corroboration_for_task` is the assembler that derives
    them from a task dict.
    """
    if claimant_live:
        return True
    if _task_in_scheduler_holders_or_parks(task_id, scheduler_state):
        return True
    return _routing_decided_after_restart(routing_latest_decided_at, orchestrator_started_at)


def corroboration_for_task(
    task: Mapping,
    task_id: str | int,
    *,
    now: datetime,
    heartbeat_ttl: timedelta = DEFAULT_HEARTBEAT_TTL,
    scheduler_state: Mapping | None = None,
    orchestrator_started_at: datetime | None = None,
) -> bool:
    """Assemble the three corroboration signals from a *task* dict.

    Derives:
      - ``claimant_live`` via ``has_live_claimant(task, now, heartbeat_ttl)``
        (freshness-checking — a killed workflow's stale heartbeat reports NOT
        live even though its durable ``claimant_run_id`` persists);
      - ``routing_latest_decided_at`` via
        ``RoutingState.from_metadata(task['metadata']).latest.decided_at``
        (degrades to ``None`` on missing/invalid metadata — never raises);

    then delegates to :func:`has_live_workflow_corroboration`. The *scheduler_state*
    and *orchestrator_started_at* inputs are hoisted once per render by the
    caller (see :func:`_render_live_workflow_section`) and threaded through.

    Returns True when any signal corroborates a live workflow for *task_id*.
    """
    claimant_live = has_live_claimant(task, now, heartbeat_ttl)
    latest = RoutingState.from_metadata(task.get('metadata')).latest
    routing_latest_decided_at = latest.decided_at if latest is not None else None
    return has_live_workflow_corroboration(
        task_id,
        claimant_live=claimant_live,
        scheduler_state=scheduler_state,
        routing_latest_decided_at=routing_latest_decided_at,
        orchestrator_started_at=orchestrator_started_at,
    )


def is_pure_gate_metadata(metadata: Mapping | object | None) -> bool:
    """Return True when *metadata* has the ``DeterministicRunner`` PURE-GATE shape.

    The pure-gate shape is ``always_escalates`` truthy AND ``before_done``
    absent/falsy — the same two metadata fields ``DeterministicRunner`` itself
    branches on (orchestrator/src/orchestrator/deterministic_runner.py section
    3), and the same pair ``shared.task_metadata.TaskMetadata.
    _deterministic_invariants`` validates. Such a task's ENTIRE execution is
    "file one born-at-L2 escalation (deduped), stamp ``metadata.
    gate_escalated_at``, set status ``blocked``": no script, no systemd, no
    ``git_ops`` (the runner is constructed without ``git_ops`` for this path,
    precisely to prove it). It therefore can never be mid-side-effect, which is
    what makes it safe to drop the project-wide orchestrator lock for it while
    it is still ``pending`` (see rule 5 of
    :func:`_orchestrator_signal_ineligible`, task 3751).

    A truthy ``before_done`` deliberately DISQUALIFIES the shape: that path runs
    a blocking deploy/predicate script plus systemd inspection and fresh-PID
    verification — minutes of real, side-effecting work — while the task's
    status is still ``'pending'``, because ``Harness._run_deterministic_slot``
    never flips a deterministic task to ``'in-progress'`` before invoking the
    runner. Such a task has no git evidence to reveal that it is running, so the
    orchestrator lock is the only signal protecting it from a recon race.

    Fail-safe toward live: only POSITIVE evidence of the shape returns True. Any
    non-``Mapping`` input (``None``, a string, an int, a list) returns False, so
    an absent, malformed, or unparseable metadata blob leaves the task live
    rather than suppressing its signal. Uses truthiness rather than ``is True``
    because task metadata round-trips through JSON.
    """
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get('always_escalates')) and not metadata.get('before_done')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _task_in_scheduler_holders_or_parks(
    task_id: str | int, scheduler_state: Mapping | None
) -> bool:
    """Return True when *task_id* is a scheduler park key or a current-holder value.

    ``scheduler_state`` is the ``read_scheduler_state`` snapshot: ``parks`` is
    keyed by task_id, ``current_holders`` is ``{module: task_id}`` (so the task
    appears as a VALUE there). Both are reset by an orchestrator restart, so a
    stranded post-redeploy task is absent from both. Coerces to ``str`` for the
    comparison since the on-disk JSON stores task_ids as strings. Tolerates a
    ``None`` state or missing/ non-dict ``parks``/``current_holders`` keys
    (returns False, never raises).
    """
    if not isinstance(scheduler_state, Mapping):
        return False
    tid = str(task_id)
    parks = scheduler_state.get('parks')
    if isinstance(parks, Mapping) and tid in {str(k) for k in parks}:
        return True
    holders = scheduler_state.get('current_holders')
    return isinstance(holders, Mapping) and tid in {str(v) for v in holders.values()}


def _routing_decided_after_restart(
    decided_at: str | None, orchestrator_started_at: datetime | None
) -> bool:
    """Return True when *decided_at* is strictly newer than *orchestrator_started_at*.

    A routing decision stamped after the current orchestrator's start time is
    evidence of a post-restart dispatch (a live workflow); one predating the
    last restart cannot correspond to a live workflow. Returns False when
    either side is ``None`` (no restart boundary → cannot prove post-restart)
    or ``decided_at`` is unparseable (fail-safe: absence of positive evidence
    is not corroboration). Both operands are UTC-aware for the comparison.
    """
    if decided_at is None or orchestrator_started_at is None:
        return False
    parsed = _parse_iso_timestamp(decided_at)
    if parsed is None:
        return False
    return parsed > orchestrator_started_at


def _orchestrator_signal_ineligible(
    status: str | None,
    task_kind: str | None,
    worktree_registered: bool = False,
    recent_commit: bool = False,
    branch_bare: bool = False,
    pure_gate: bool = False,
) -> bool:
    """Return True when the project-wide ``orchestrator_live`` signal must be
    forced False for this *status*/*task_kind*/*worktree_registered*/
    *recent_commit*/*branch_bare*/*pure_gate* combination.

    Five independent rules are centralized here:

    1. ``status`` is a member of :data:`ORCH_LIVE_INELIGIBLE_STATUSES`
       (``deferred``, ``done``, ``cancelled``) — statuses that are never
       actively dispatched, so the project-wide lock is not evidence for this
       task (task 2031).
    2. ``status == 'blocked' and task_kind == DETERMINISTIC_TASK_KIND`` — a
       blocked deterministic task, which never acquires a worktree/branch (it
       is routed to ``DeterministicRunner`` instead), so the bare project lock
       is not task-specific evidence for it either (task 2067). Unconditional:
       a deterministic task never has git evidence to lose.
    3. ``status == 'blocked' and task_kind in (None, NORMAL_TASK_KIND) and not
       worktree_registered and not recent_commit`` — a blocked normal (or
       task_kind-absent) task with NO per-task git evidence, which is the
       bare-orchestrator false positive that caused tasks 2335/2196 to loop
       through repeated re-deferral (task 2409). Guarded on the git signals
       (unlike rule 2): a normal blocked task can legitimately hold a
       worktree/recent commit while auto-unblocking mid-pipeline (task 2031's
       concern), so this rule only suppresses the bare case, leaving
       ``is_live`` (which already ORs in ``worktree_registered``/
       ``recent_commit``) unaffected whenever real per-task evidence exists.
    4. ``branch_bare and not worktree_registered and not recent_commit`` — the
       task's branch is provably bare (zero commits beyond ``base_branch``)
       AND no LIVE worktree is registered (a registered-but-prunable entry
       does not count — see ``worktree_registered``'s meaning) AND there is
       no recent commit, i.e. POSITIVE evidence the branch has no work and
       its worktree was reaped, so a running project orchestrator is
       demonstrably not on THIS task (task 2767, reify#5245). Deliberately
       **status/task_kind-agnostic** — unlike rules 1-3, this rule is
       evaluated regardless of ``status``/``task_kind`` (including for
       ``pending``/``in-progress``/``review``/``merge-deferred``, and
       independent of rules 2/3's blocked-only scoping), because
       ``recon_write_policy`` Gate 2 calls the detector without ``task_kind``
       and with whatever status the task currently holds — a status-gated
       rule would not fire there. Not a race risk: an absent branch yields an
       unknown (not ``0``) commit count (see
       :func:`_branch_own_commit_count`), so a not-yet-dispatched task is
       unaffected; a just-started dispatch keeps a LIVE (non-prunable)
       worktree, so ``worktree_registered`` is True and this rule stays
       inert.
    5. ``status == 'pending' and task_kind == DETERMINISTIC_TASK_KIND and
       pure_gate`` — a PENDING deterministic task of the PURE-GATE shape
       (``always_escalates`` truthy AND ``before_done`` absent, classified by
       :func:`is_pure_gate_metadata`), whose entire ``DeterministicRunner`` run
       is "file one born-at-L2 escalation, stamp ``gate_escalated_at``, set
       status blocked" — no script, no systemd, no ``git_ops``, and (like every
       deterministic task) no worktree/branch — so it can never be
       mid-side-effect and the bare project lock is not evidence for it (task
       3751; confirmed incident: task 3845 stalled 3+ recon cycles showing only
       the bare ``orchestrator`` signal). Unconditional on the git signals, like
       rule 2. ``pure_gate`` defaults False, so this rule is inert for every
       caller that does not classify the task's metadata. It is evaluated
       before rule 4 in the body below purely for readability — the rules form
       a disjunction of independent early returns, so their order is immaterial.

    ``'blocked'`` is deliberately NOT added to ``ORCH_LIVE_INELIGIBLE_STATUSES``
    wholesale: a normal blocked task (``task_kind`` absent or not
    ``'deterministic'``) may auto-unblock and be legitimately mid-pipeline, so
    the project-wide orchestrator lock remains real per-task evidence for it
    *when there is other evidence to corroborate it*. Hence the compound
    (status AND task_kind [AND NOT git-evidence]) conditions for rules 2 and 3,
    rather than an unconditional status addition. Rule 4 is the sole exception
    to the "scoped by status" pattern, by design (see above).

    ``'pending'`` is likewise NOT added to ``ORCH_LIVE_INELIGIBLE_STATUSES``:
    every ordinary pending task is dispatch-eligible, and a pending
    *deterministic* task carrying a ``before_done`` may be mid-deploy inside
    ``DeterministicRunner`` (``Harness._run_deterministic_slot`` never flips it
    to ``'in-progress'``) with no git evidence to reveal it. Hence rule 5's
    compound (status AND task_kind AND pure_gate) condition.
    """
    if status is not None and status in ORCH_LIVE_INELIGIBLE_STATUSES:
        return True
    if status == 'blocked':
        if task_kind == DETERMINISTIC_TASK_KIND:
            return True
        if (
            task_kind in (None, NORMAL_TASK_KIND)
            and not worktree_registered
            and not recent_commit
        ):
            return True
    if status == 'pending' and task_kind == DETERMINISTIC_TASK_KIND and pure_gate:
        return True
    return branch_bare and not worktree_registered and not recent_commit


def _check_worktree_registered(project_root: str, branch: str) -> tuple[bool, bool]:
    """Return ``(registered, prunable)`` for the git worktree tracking *branch*.

    Parses ``git -C <root> worktree list --porcelain`` output into
    blank-line-delimited stanzas (one per registered worktree). ``registered``
    is True iff some stanza contains a line equal to ``branch refs/heads/<branch>``;
    ``prunable`` is True iff that SAME stanza also contains a line starting with
    ``prunable`` — git's marker for a worktree whose directory has been removed
    or reaped, but whose registration has not yet been pruned (reify#5245's
    shape: a stale worktree entry survives after the directory itself is gone).

    Any subprocess error or unexpected output silently returns ``(False, False)``
    (fail-safe).

    Note: git only started emitting the ``prunable`` porcelain annotation in
    git 2.36 (2022). On an older git binary, a reaped worktree's directory can
    be gone yet no ``prunable`` line is ever produced, so ``prunable`` silently
    stays False and a reaped worktree keeps counting as registered — the same
    silent-degradation shape reify#5245 hardened against, just one layer down
    in the toolchain. If a stale/reaped-worktree false positive resists this
    fix, check ``git --version`` on the host running this detector first.
    """
    try:
        result = subprocess.run(
            ['git', '-C', project_root, 'worktree', 'list', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug('live_workflow_detector: worktree list failed: %s', exc)
        return False, False

    if result.returncode != 0:
        logger.debug(
            'live_workflow_detector: worktree list returned %d: %s',
            result.returncode, result.stderr.strip(),
        )
        return False, False

    target = f'branch refs/heads/{branch}'
    for stanza in result.stdout.split('\n\n'):
        lines = [line.strip() for line in stanza.splitlines()]
        if target in lines:
            prunable = any(line.startswith('prunable') for line in lines)
            return True, prunable
    return False, False


def _check_recent_commit(
    project_root: str,
    branch: str,
    *,
    now: datetime | None,
    max_commit_age_hours: float,
) -> tuple[datetime | None, bool]:
    """Return (last_commit_at, recent_commit) for *branch*.

    ``last_commit_at`` is the parsed commit timestamp, or ``None`` if unavailable.
    ``recent_commit`` is True when the commit is within *max_commit_age_hours*
    of *now* (or ``datetime.now(UTC)`` when *now* is ``None``).
    """
    try:
        result = subprocess.run(
            ['git', '-C', project_root, 'log', '-1', '--format=%cI', branch],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug('live_workflow_detector: git log failed for %s: %s', branch, exc)
        return None, False

    if result.returncode != 0:
        logger.debug(
            'live_workflow_detector: git log returned %d for branch %s (branch may not exist)',
            result.returncode, branch,
        )
        return None, False

    ts_str = result.stdout.strip()
    if not ts_str:
        return None, False

    last_commit_at = _parse_iso_timestamp(ts_str)
    if last_commit_at is None:
        logger.debug(
            'live_workflow_detector: cannot parse commit timestamp %r for branch %s',
            ts_str, branch,
        )
        return None, False

    reference = now if now is not None else datetime.now(UTC)
    age = reference - last_commit_at
    recent_commit = age <= timedelta(hours=max_commit_age_hours)
    return last_commit_at, recent_commit


def _branch_own_commit_count(project_root: str, base_branch: str, branch: str) -> int | None:
    """Return the number of commits *branch* carries beyond *base_branch*.

    Runs ``git -C <root> rev-list --count <base_branch>..<branch>``.  A result
    of ``0`` means *branch* is "bare" — created from *base_branch* but with no
    commits of its own (e.g. only a ``git worktree add``/branch-creation
    reflog entry — reify#5245's shape).  Any subprocess error, non-zero
    returncode (e.g. *base_branch* or *branch* missing), or unparseable output
    silently returns ``None`` (fail-safe: an unknown count is never treated as
    ``0``/bare by callers).
    """
    try:
        result = subprocess.run(
            ['git', '-C', project_root, 'rev-list', '--count', f'{base_branch}..{branch}'],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug(
            'live_workflow_detector: rev-list failed for %s..%s: %s',
            base_branch, branch, exc,
        )
        return None

    if result.returncode != 0:
        logger.debug(
            'live_workflow_detector: rev-list returned %d for %s..%s: %s',
            result.returncode, base_branch, branch, result.stderr.strip(),
        )
        return None

    stdout = result.stdout.strip()
    try:
        return int(stdout)
    except (TypeError, ValueError):
        logger.debug(
            'live_workflow_detector: cannot parse rev-list count %r for %s..%s',
            stdout, base_branch, branch,
        )
        return None


def _parse_iso_timestamp(ts_str: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string from ``git log --format=%cI``.

    Returns ``None`` on any parse error.
    """
    try:
        dt = datetime.fromisoformat(ts_str)
        # Ensure UTC-aware for consistent comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None
