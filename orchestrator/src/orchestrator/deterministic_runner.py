"""DeterministicRunner — orchestrator-side runner for deterministic gate tasks (β/γ/ε).

A *deterministic* task (``metadata.task_kind == 'deterministic'``) is routed
here by ``Harness._run_slot`` instead of ``TaskWorkflow``.  The runner holds
only ``scheduler`` + ``escalation_queue`` (no git_ops) — structurally proving
that no worktree, branch, agent, or steward is created for a gate (I4/B2).

``DeterministicRunner.run`` dispatches in three sections:

1. Idempotency / quiescence — hold or resume a task that already escalated
   or already ran its ``before_done`` action.
2. before_done execution — the stop-instruction guard and the refused-record
   check, then the ε self-restart, γ cross-unit, target_unit-less and
   predicate sub-paths.
3. Pure gate — file one born-at-L2 escalation and block; resume only on the
   two-rung proof ladder.

Full dispatch rationale: docs/deterministic-runner.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from escalation.models import Escalation, ReviewDueState
from shared.before_done_paths import (
    BeforeDonePaths,
    BeforeDonePathsRefused,
    FactoryRootUnresolved,
    resolve_before_done_paths,
)
from shared.milestone_rearm import rearmed_milestone
from shared.repo_paths import DARK_FACTORY_ROOT_ENV, resolve_dark_factory_root
from shared.task_metadata import (
    HUMAN_CURATOR_ADJUDICATED_AT_KEY,
    HUMAN_CURATOR_GATE_KEY,
    DoneProvenance,
    TriggerChain,
)
from shared.task_read import TaskRead, TaskReadFailed

from orchestrator import systemd_inspect
from orchestrator.before_done_script import (
    REAP_GRACE_SECS,
    ProcessTeardown,
    ScriptTimeout,
    run_script,
)
from orchestrator.config import PREDICATE_RECHECK_STALL_CAP
from orchestrator.dependency_verdict import task_dependency_ids
from orchestrator.deploy_state import (
    SCHEDULED_DEPLOY_PROVENANCE_KIND,
    DeployPhase,
    DeployState,
    IllegalDeployTransition,
    VerifyBaseline,
    enforce_transition,
)
from orchestrator.event_store import EventType
from orchestrator.fm_retry import FM_RESTART_RETRY_WINDOW_SECS, fm_retry_backoffs
from orchestrator.gate_preconditions import (
    gate_held_since_escalation,
    gate_preconditions_stamp,
)
from orchestrator.predicate_output import predicate_verdict, summarize_predicate_output
from orchestrator.predicate_recheck import (
    PredicateRecheck,
    RearmedRecheck,
    rearmed,
    recheck_declaration,
    render_verdict,
    reset,
    stall_detail,
    verdict_stamps,
)
from orchestrator.proc_supervision import (
    EscalationSpec,
    FreshPidVerify,
    RestartDisposition,
    RestartPlan,
)
from orchestrator.review_due import (
    REVIEW_DUE_CATEGORY,
    CompletionGateReading,
    HumanGateFacts,
    InvalidTriggerChain,
    ReviewDueWithheld,
    build_review_due,
    completion_gate_reading,
    first_completion_gate,
    human_gate_chain,
)
from orchestrator.stop_instruction import detect_stop_instruction
from orchestrator.systemd_inspect import (
    _deterministic_deploy_health_verdict,
    _wedged_unit_sentinel,
    inspect_systemd_unit,
)
from orchestrator.workflow import WorkflowOutcome

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

# Task 2066: cross-unit before_done writeback resilience.
#
# The cross-unit blocking deploy may restart the very service backing the
# orchestrator's own fused-memory/MCP connection (e.g. task 2059), severing
# it for the duration of a `--drain` restart.  `shared.mcp_session.McpSession.call`
# opens a fresh httpx.AsyncClient and re-inits the session
# on any transient failure, so the NEXT call auto-reconnects — a successful
# write IS the reconnection proof.  The post-deploy verify+writeback retry
# budget (``DeterministicRunner._writeback_backoffs``) sizes off the shared
# ``orchestrator.fm_retry.fm_retry_backoffs()`` schedule (task 2706) instead
# of an independently-guessed budget, so it patiently outlasts such a
# restart — see orchestrator/src/orchestrator/fm_retry.py.

# Task 2090: outer wall-clock backstop around the cross-unit `run_fn` call in
# run().  This is a pure safety margin ON TOP OF before_done['timeout_secs'] —
# it must never fire before the inner script-runner timeout in the normal
# case, so 30s is deliberately generous (covers process-group teardown +
# reap_grace_secs with room to spare) rather than tight.
#
# Task 4065: because this guard is a strict superset, the INNER timeout is
# what actually fires for any real, SIGKILL-able subprocess — and for the
# DEFAULT runner that event is now a distinct signal (`ScriptTimeout`, in
# orchestrator/src/orchestrator/before_done_script.py).
# This guard's remaining job is the case it was always for: a seam that never
# returns at all (a detached/unkillable child, or a custom script_runner that
# simply hangs).
_RUN_TIMEOUT_GRACE_SECS: float = 30.0


def _deploy_timeout_fact_lines(exc: ScriptTimeout, *, script: Path) -> list[str]:
    """The FACT sentences both DEPLOY arms print for a ``ScriptTimeout``.

    Shared so the two deploy branches cannot come to say different things
    about the same event (task 4252).  The predicate arm shares only
    ``ScriptTimeout.budget_line``: its CONSEQUENCE differs (no milestone
    VERDICT), and the predicate infra arms all file one category, so their
    wording is the only thing telling a human which guard fired.

    Args:
        script: the absolute path that ran (``_before_done_paths``), the
            survivor sentence's search key.
    """
    lines = [exc.budget_line('Deploy script')]
    if exc.exit_code is None:
        if exc.teardown is ProcessTeardown.NOT_SIGNALLED:
            # This cell is reachable only as a signal death, and the plain
            # "had not exited" spelling below would be false in it:
            # NOT_SIGNALLED means the script was ALREADY reaped when the
            # teardown ran, and a reaped script with no exit code is one that
            # died of a signal (a negative returncode is not an exit code).
            why_no_code = (
                'the script had already died of a SIGNAL rather than exiting, and '
                'a signal death produces no exit code'
            )
        else:
            why_no_code = 'the script had not exited when the timeout fired'
        lines.append(
            f'No exit code was produced — {why_no_code}. No output was captured '
            f'either: the merged stdout/stderr read was still in flight. The '
            f'script DID run, so it may have applied PART of its effect: inspect '
            f'out-of-band before resolving.'
        )
    elif exc.teardown is ProcessTeardown.GROUP_KILLED:
        lines.append(
            f'The script ITSELF had already exited with code {exc.exit_code}: '
            f'the timeout fired because a process it spawned outlived it, '
            f'still holding the merged stdout/stderr pipe open, and that '
            f'survivor was killed with the group. So the script ran to '
            f'completion — but its exit code never reached the deploy '
            f'classifier, so this deploy is NOT recorded as successful even '
            f'when that code is 0, and no output was captured because the '
            f'read was still in flight. Check out-of-band what the surviving '
            f'child was and whether killing it left the effect half-applied.'
        )
    else:
        lines.append(
            f'The script ITSELF had already exited with code {exc.exit_code}: '
            f'the timeout fired because a process it spawned outlived it, '
            f'still holding the merged stdout/stderr pipe open. So the script '
            f'ran to completion — but its exit code never reached the deploy '
            f'classifier, so this deploy is NOT recorded as successful even '
            f'when that code is 0, and no output was captured because the read '
            f'was still in flight.'
        )
    survivor = exc.survivor_line(script)
    if survivor is not None:
        lines.append(survivor)
    return lines


# Task 2091 / 2119: bound `_default_inspect_unit`'s `systemctl --user show`
# call — a parallel latent-hang gap to task 2090, which only wraps the
# before_done run_fn subprocess and not this inspect call (esc-2090-11).
# This runs on both the baseline inspect and the post-deploy verify inspect;
# an unbounded `communicate()` here strands the runner exactly like task 2087
# (before_done_ran_at stamped, before_done_verified_at never stamped, no
# escalation filed). The hardening itself (and its 10s default timeout) now
# lives in ``systemd_inspect.py`` — see ``_default_inspect_unit`` below,
# which is a thin delegate to ``systemd_inspect.inspect_systemd_unit``.
_INSPECT_TIMEOUT_SECS: float = systemd_inspect._INSPECT_TIMEOUT_SECS

# Task 2120: sentinel agent_role the runner stamps on its own escalations and
# scopes all its get_by_task queries to — an unrelated escalation with the
# same task_id (e.g. a starvation-watchdog filing) must never alias as this
# runner's own dedup/quiescence/resolution-proof signal.
DETERMINISTIC_AGENT_ROLE: str = 'orchestrator-deterministic'

# The category the scheduled self-restart transient's RP-4 wrapper files a
# fire-time failure under; the scheduled-deploy settler reads it as failure
# evidence (orchestrator/src/orchestrator/scheduled_deploy_settle.py).
SELF_RESTART_FAILURE_CATEGORY: str = 'infra_issue'

# The actor named in this runner's own filings and their log lines.
_RUNNER_ACTOR = 'DeterministicRunner'

# Task 2803 (γ, PRD plans/operational-ask-routing-prd.md): stable, fixed
# token emitted on a pure-gate born-at-L2 escalation's summary/detail when
# the gate originated from an execution_class='operational' +
# operational_mode='llm' submission — makes it machine-distinguishable from
# a plain deterministic gate (a future LLM-operational-lane PRD consumes
# this as its trigger; INV-2 requires a structured token, not log-scraped
# prose).
OPERATIONAL_LLM_NEEDS_LANE_TOKEN: str = 'operational_llm_needs_lane'

# Task 2803 (γ): mirrors β's producer key
# (fused_memory.middleware.operational_routing_guard.OPERATIONAL_LLM_GATE_MARKER_KEY,
# task 2802) as a bare orchestrator-side literal — orchestrator does not
# depend on the fused_memory package (see DETERMINISTIC_AGENT_ROLE-adjacent
# metadata-key-as-literal convention used for task_kind/before_done/
# gate_escalated_at elsewhere in this module). β stamps this key True ONLY
# for execution_class='operational' + operational_mode='llm' — a
# decision+operational_mode='llm' gate does NOT get it, so this (not the raw
# operational_mode) is the precise signal to key on.
OPERATIONAL_LLM_GATE_MARKER_KEY: str = 'x_operational_llm_gate'

# Task 3341: the human-curator-gate contract (marker and stamp keys, their
# single source, the naming) — docs/deterministic-runner.md, section
# "Human-curator gate contract (task 3341)".

# The born-at-L2 escalation categories this runner files — EVERY one of which
# `escalation.authority.L2_AUTO_CLOSE_DENY_CATEGORIES` must refuse to auto-close.
#
# DUPLICATED — not imported — in that denylist (task 3181).  escalation is the
# lower fleet-wide package and must not import orchestrator, so each pair of
# literals is pinned in lockstep by the cross-layer TEST imports in
# `escalation/tests/test_authority.py`, mirroring the `_WATCHER_AUTO_IDENTITY`
# convention documented in authority.py's module docstring.  If any of these
# strings changes, update both sides together.
#
# All four are named here rather than just the curator one (reviewer
# amendment): they are siblings with identical exposure, so pinning one and
# leaving the others as bare literals would leave a reader of the denylist an
# unexplained asymmetry — and would leave the same silent-rename hole open on
# the unpinned members.  `curator_adjudication_missing` is the re-ask raised
# when a `human_curator_gate` task resumes with no `human_curator_adjudicated_at`
# stamp; `milestone_gate` is the gate itself; `milestone_check_failed` is a
# failed predicate check; `milestone_check_stalled` is a re-arming predicate
# still "not yet" at `deterministic.predicate_recheck_stall_cap` (PRD
# plans/completion-driven-triggers-prd.md C1).  The fifth, `review_due` (C4),
# is defined with the rest of its contract in `orchestrator.review_due`.
CURATOR_ADJUDICATION_MISSING_CATEGORY: str = 'curator_adjudication_missing'
MILESTONE_GATE_CATEGORY: str = 'milestone_gate'
MILESTONE_CHECK_FAILED_CATEGORY: str = 'milestone_check_failed'
MILESTONE_CHECK_STALLED_CATEGORY: str = 'milestone_check_stalled'

# `metadata.recurrence` — one link of a recurring deterministic chain (task
# 4676 / r1, modelled by `shared.task_metadata.Recurrence`).  Read here by
# `_is_recurrence_carrier` only; this module never constructs or validates the
# payload, so the key is named rather than imported from `shared` (which
# exports the MODEL, not a key constant).
RECURRENCE_KEY: str = 'recurrence'

# Rendered in place of the configured project_root when the runner's duck-typed
# `scheduler` collaborator cannot supply a usable one.  A VISIBLE placeholder,
# not an omission: it keeps the runbook's update_task call syntactically valid
# and still NAMES the required parameter.
_PROJECT_ROOT_PLACEHOLDER: str = '<project_root>'


def _configured_project_root(scheduler) -> Path | None:
    """Return the scheduler's configured project_root, or None if unusable.

    ``scheduler`` is duck-typed here — ``Harness._run_deterministic_slot``
    builds the runner "with only the minimal dependencies needed" — so the
    attribute chain must never raise.  That matters most to
    ``_file_curator_adjudication_missing_and_block``, whose contract is that a
    durable on-disk safety escalation is filed no matter what: the chain is
    read while formatting the detail string, i.e. BEFORE
    ``escalation_queue.submit()``, so an ``AttributeError`` there would not
    just skip the BLOCK — it would lose the escalation entirely and propagate.

    Accepts the value ONLY when it is a non-empty ``str``/``Path`` that is not
    the literal ``'None'``, so a test double's Mock attribute reaches neither
    an operator (as ``<MagicMock id=...>``) nor a filesystem path.  The
    ``'None'`` rejection mirrors the scheduler's own project_root guard, which defends
    against a value that bypassed pydantic validation.

    Deliberately no broad ``except Exception``: the two ``getattr`` defaults
    plus the isinstance/non-empty check are the entire failure surface.
    """
    root = getattr(getattr(scheduler, 'config', None), 'project_root', None)
    if isinstance(root, (str, Path)):
        text = str(root).strip()
        if text and text != 'None':
            return Path(text)
    return None


def _snippet_project_root(scheduler) -> str:
    """The configured project_root for the runbook snippet, else a visible placeholder."""
    root = _configured_project_root(scheduler)
    return str(root) if root is not None else _PROJECT_ROOT_PLACEHOLDER


def _configured_project_id(scheduler) -> str | None:
    """The scheduler's configured fused-memory project_id, or None if unusable.

    Never raises, for the reason ``_configured_project_root`` gives.
    """
    fused_memory = getattr(getattr(scheduler, 'config', None), 'fused_memory', None)
    project_id = getattr(fused_memory, 'project_id', None)
    if isinstance(project_id, str) and project_id.strip():
        return project_id.strip()
    return None

# Length bound applied to the externally-supplied `human_curator_adjudicated_at`
# value when it is interpolated into the curator-gate `done_provenance.note`.
#
# Task 3286 established that `done_provenance.note` is NOT a private field:
# fused-memory reconciliation's `_format_outcome_echo` appends it to the
# "Task '<title>' completed." Mem0 write under a 500-char `max_note_chars` cap.
# `_curator_adjudication_confirmed` deliberately proves only "a human stamped
# something" — it is NOT a length validator, because rejecting a long-but-
# genuine stamp would re-open the very phantom-done failure mode task 3341
# exists to close.  So the bound lives here, at the note-construction site,
# where an over-length stamp degrades the audit string but never the safety
# decision.
_CURATOR_STAMP_NOTE_MAX_CHARS: int = 64

# Task 2238 (W10-δ): a guaranteed-non-self placeholder `own_unit` fed to
# proc_supervision.RestartPlan.execute() on the cross-unit blocking deploy
# path when the runner's own ORCH_UNIT-derived own_unit is falsy (the
# fail-open case, ORCH_UNIT unset).  RestartPlan.execute()'s RP-1 fail-closed
# guard refuses a blocking restart whenever own_unit is falsy — but the
# runner only reaches the cross-unit branch AFTER its own `self_target` check
# above has already ruled out a same-unit restart, so forcing a truthy,
# provably-non-target own_unit here routes execute() into RP-2 (cross-unit
# blocking + verify) instead of RP-1 (refuse), preserving the existing
# fail-open-to-cross-unit behaviour (see test_env_unset_takes_cross_unit_path).
# Not a valid systemd unit name (unit names never contain angle brackets), so
# it can never collide with a real target_unit.
_CROSS_UNIT_OWN_UNIT_SENTINEL: str = '<no-self-target-known>'

# The closing line of every escalation filed after before_done_ran_at is stamped.
_I1_NOT_RERUN_LINE: str = 'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.'


def _build_done_provenance(kind: str, **fields: object) -> dict:
    """Build a ``done_provenance`` dict via the shared ``DoneProvenance`` model.

    THE single seam every runner ``done_provenance`` construction routes
    through (task 2167 — W3-δ SEAM B), sharing ONE valid-kinds enum with the
    fused-memory validator (I2) instead of six independent inline dict
    literals.  An unknown/typo *kind* raises ``pydantic.ValidationError``
    here, at build time, on the orchestrator side — structurally preventing
    the 1902/1976/1982 permanently-blocked self-restart failure mode (a
    ``kind`` fused-memory silently rejects).

    ``exclude_none=True`` keeps the emitted wire dict compatible with the
    hand-written literals this replaces for every key that carries a
    non-``None`` value (they never carried explicit ``None`` values either).
    One path diverges: the crash-resume call at ``before_done_verified_pid``
    used to always emit a ``pid`` key (``None`` when that metadata field was
    absent), whereas ``exclude_none`` now omits the key entirely in that
    case.  All known consumers (``task_interceptor.py`` and this module's
    tests) read it via ``.get('pid')``, so a missing key and an explicit
    ``None`` are observationally identical there. ``transient_unit`` and
    ``fire_delay_secs`` are declared, typed fields of ``DoneProvenance``, so
    a wrongly-typed value raises here too.

    ``kind``/``**fields`` are intentionally loosely typed (``str``/``object``):
    the six call sites forward heterogeneous field subsets, so pyright's
    dataclass-transform-synthesized ``DoneProvenance.__init__`` (each
    parameter narrowly typed, e.g. ``kind: Literal[...]``, ``pid: int | None``)
    cannot be satisfied by a generic pass-through signature. The real
    validation happens at runtime, in the model itself.
    """
    return DoneProvenance(kind=kind, **fields).model_dump(exclude_none=True)  # type: ignore[arg-type]


def _cite_gate_escalation_id(own_records: list, gate_category: str) -> str | None:
    """Pick the escalation a `deterministic-gate` close should CITE.

    Cite the gate record of this gate's own *gate_category*
    (:func:`gate_escalation_category`), NOT merely the newest own-role one
    (reviewer amendment, task 3341).  The standard curator remediation round-trip
    leaves TWO own-role records behind:
      (A) the original `milestone_gate`, resolved WITHOUT a stamp — the
          record that proves rung one, and
      (B) the `curator_adjudication_missing` re-ask the guard filed,
          resolved once the human finally stamped.
    ``escalation_id`` is rung-ONE evidence, and both docs/deterministic-runner.md
    and docs/task-authoring.md promise it names the gate record.  Taking the
    newest own record would cite (B), the re-ask — the one record that proves
    nothing about the gate — quietly mis-aiming the audit trail this citation
    exists to sharpen.  So filter to the gate category first and take the
    newest of those; fall back to the newest own record only when no
    gate-category record survives (e.g. a human gate filed as
    ``milestone_gate`` before ``review_due`` existed, or with its
    ``review_due`` withheld for an unusable config), and return ``None``
    for an empty list (an empty list must
    never be indexed; ``_build_done_provenance``'s ``exclude_none=True`` then
    simply drops the key).

    BOTH gate arms call this — the curator close and, since task 5241, the
    pure-gate close.  The rule must not exist twice: its whole value is being
    uniform, and a second copy would strand this rationale on one of them.
    Neither arm can actually reach the ``None`` return: both run only after
    the zero-record strand branch has re-filed the gate and returned BLOCKED.

    Pinned by test_curator_gate_remediation_round_trip_cites_the_gate_record,
    test_pure_gate_resume_cites_the_gate_record_not_the_newest and
    test_human_gate_cites_its_review_due_over_a_newer_milestone_gate.
    """
    gate_records = [e for e in own_records if e.category == gate_category]
    citable = gate_records or own_records
    if not citable:
        return None
    return sorted(citable, key=lambda e: e.timestamp)[-1].id


def _is_operational_llm_gate(metadata: dict) -> bool:
    """Return True iff *metadata* carries β's operational-llm-gate marker.

    Task 2803 (γ). Keys on ``OPERATIONAL_LLM_GATE_MARKER_KEY`` (an explicit
    ``is True`` check, not mere truthiness) rather than the raw
    ``operational_mode`` field — a ``decision``+``operational_mode='llm'``
    pure gate preserves ``operational_mode='llm'`` too but never gets this
    marker, so reading the raw mode would false-positive that case.
    """
    return metadata.get(OPERATIONAL_LLM_GATE_MARKER_KEY) is True


def _is_human_curator_gate(metadata: dict) -> bool:
    """Return True iff *metadata* marks a gate needing human CONTENT adjudication.

    Task 3341.  Keys on ``HUMAN_CURATOR_GATE_KEY`` via plain TRUTHINESS —
    a DELIBERATE divergence from ``_is_operational_llm_gate`` directly above,
    which uses a strict ``is True``.  The two markers have opposite failure
    costs, so they get opposite postures:

    * ``_is_operational_llm_gate`` is a ROUTING hint. A false positive
      misroutes a human-handling lane, so failing open (``is True``) is right.
    * This is a SAFETY gate. A false NEGATIVE silently reproduces the task-3181
      phantom-done — the exact failure this predicate exists to prevent. So a
      truthy-but-not-``True`` value (e.g. the string ``'true'`` from a hand
      edit or a JSON round-trip) must fail CLOSED: the guard still applies.

    Pinned by ``test_string_truthy_curator_gate_marker_still_trips_the_guard``.
    """
    return bool(metadata.get(HUMAN_CURATOR_GATE_KEY))


def _curator_adjudication_confirmed(metadata: dict) -> bool:
    """Return True iff *metadata* carries positive proof the content review happened.

    Task 3341.  Proof is ``HUMAN_CURATOR_ADJUDICATED_AT_KEY`` holding a non-empty,
    non-whitespace ``str`` (an ISO-8601 stamp).  Every other value — ``bool``,
    ``int``, ``None``, an empty or blank string — is NOT proof and fails CLOSED.

    Note ``bool`` is excluded explicitly rather than incidentally: ``True`` is
    the single most likely wrong value a hand edit produces here (it reads as
    "yes, adjudicated"), and it must not be accepted as a stamp.

    This checks TYPE and EMPTINESS only, never length or ISO shape — see
    ``_CURATOR_STAMP_NOTE_MAX_CHARS`` for why bounding belongs at the
    note-construction site instead. Pinned by
    ``test_blank_adjudication_stamp_is_not_proof``.
    """
    stamp = metadata.get(HUMAN_CURATOR_ADJUDICATED_AT_KEY)
    return isinstance(stamp, str) and bool(stamp.strip())


def _is_recurrence_carrier(metadata: dict) -> bool:
    """Return True iff *metadata* carries a recurring-chain link (γ-predicate).

    Task 4678 (r3) / PRD ``docs/prds/recurring-deterministic-tasks.md``
    decision R-D6, contract C-5.  A task carrying ``metadata.recurrence`` is
    one link of a recurring deterministic chain, and every failure leg of such
    a run must file ``milestone_check_failed`` — the deny-listed, discriminable
    category — instead of dropping into the crowded ``infra_issue`` bucket
    where a recurring job's failures become indistinguishable from fleet noise.

    PRESENCE of a non-empty ``recurrence`` dict is the WHOLE test.  The other
    two conditions of the PRD's C-1 carrier contract — ``task_kind
    ='deterministic'`` and ``before_done['kind'] == 'predicate'`` — hold BY
    CONSTRUCTION at the only call site: ``_run_predicate`` is reachable only
    from inside ``run()``'s ``before_done.get('kind') == 'predicate'`` branch
    of a deterministic dispatch, so re-checking them here would re-derive what
    the call path already proves.

    It deliberately does NOT re-validate the ``Recurrence`` shape or the dated
    milestone, even though ``docs/task-authoring.md`` §6.1 warns that the
    carrier contract is enforced at SUBMIT time only and that "any consumer
    acting on a chain link (the mint above all) should re-verify the carrier".
    That instruction is aimed at r2's MINT, which creates state: minting off a
    malformed link writes a bad successor.  This output selects an escalation
    LABEL, and the failure costs are asymmetric —

    * mislabelling a hand-edited malformed carrier is INERT: both categories
      are born-at-L2, ``orchestrator-deterministic``, severity ``critical``,
      task ``blocked``, and both are un-auto-closable (``milestone_check_failed``
      by category, ``orchestrator-deterministic`` by role);
    * failing CLOSED would silently drop a real recurring job's failure into
      the largest, most crowded category — the exact degradation r3 exists to
      remove.

    So it fails OPEN on validity and CLOSED on presence: a non-``dict`` value
    (a stray ``recurrence: 'daily'`` hand edit) and the empty dict are both
    rejected, so nothing scalar or vacant can be read as a chain link.

    Deliberately PRIVATE to this module, following its three sibling metadata
    predicates above.  r2's mint (fused-memory) and r4's gauge (dashboard)
    are NOT prospective sharers as-is: the mint must re-verify the full carrier
    contract before writing a successor, which is strictly more than presence.
    A genuine second consumer needing THESE semantics is the trigger to hoist
    it into ``shared.task_metadata`` — not before.

    Pinned by ``TestIsRecurrenceCarrier`` in
    orchestrator/tests/test_deterministic_runner.py.
    """
    recurrence = metadata.get(RECURRENCE_KEY)
    return isinstance(recurrence, dict) and bool(recurrence)


def _is_scheduled_self_deploy_complete(task: dict | None) -> bool:
    """Return True iff *task* is an already-completed scheduled self-deploy.

    Task 2983 fix (a).  A deterministic self-restart deploy that took the ε
    scheduled path completes by stamping ``before_done_scheduled_at`` and
    setting the task ``done`` with
    ``done_provenance.kind == 'deterministic-deploy-scheduled'`` (done =
    *scheduled*, not *verified*; verification is out-of-band).  When a STALE
    eligibility snapshot re-selects such a task after its first dispatch has
    completed, ``run()`` must recognize the FRESH task shape as deploy-complete
    rather than tripping the crash-window detector.

    True iff a fresh read of *task* shows EITHER ``metadata['done_provenance']``
    is a dict with ``kind == 'deterministic-deploy-scheduled'`` OR
    ``metadata['before_done_scheduled_at']`` is truthy.

    Fail-safe (mirrors ``harness._is_terminal_merged``'s shape): a ``None``
    *task* (``get_task`` failure or absence) or a missing/non-dict ``metadata``
    is treated as non-matching rather than raising — the caller then falls
    through to the unchanged crash-window re-escalation path, so a genuine
    crash-window is never silently dismissed.

    This detects the scheduled-self-deploy SHAPE only; it deliberately does
    NOT read ``always_escalates`` (the before_done_scheduled_at stamp is
    written on BOTH the always_escalates=False done path AND the act-then-ask
    always_escalates=True gate path).  A True result therefore means "drive
    to DONE" ONLY for always_escalates=False — the caller MUST apply the
    always_escalates policy split (re-file the milestone gate + block when
    always_escalates=True), exactly as the (b-self) branch does; short-
    circuiting to DONE on a bare True would bypass a still-open act-then-ask
    gate (reviewer_comprehensive amendment, task 2983).
    """
    if not isinstance(task, dict):
        return False
    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        return False
    provenance = metadata.get('done_provenance')
    if (
        isinstance(provenance, dict)
        and provenance.get('kind') == SCHEDULED_DEPLOY_PROVENANCE_KIND
    ):
        return True
    return bool(metadata.get('before_done_scheduled_at'))


def file_illegal_deploy_transition(
    escalation_queue: EscalationQueue,
    task_id: str,
    old: DeployPhase,
    new: DeployPhase,
    *,
    actor: str,
) -> None:
    """DS-2 sink: file a born-at-L2 escalation for an illegal deploy-phase edge *actor* attempted.

    Wired into ``enforce_transition``'s ``escalation_sink`` — files BEFORE
    ``enforce_transition`` raises ``IllegalDeployTransition``, so an illegal
    edge is never silently swallowed (D2).  Mirrors the other
    escalation-filing helpers' construction (sentinel role keeps level=2 past
    the server downgrade gate) but is unconditional — no dedup guard — since
    an illegal transition is itself a bug signal that must never be
    suppressed by an unrelated pending escalation.
    """
    esc = Escalation(
        id=escalation_queue.make_id(task_id),
        task_id=task_id,
        agent_role=DETERMINISTIC_AGENT_ROLE,
        severity='critical',
        category='illegal_deploy_transition',
        summary=f'Illegal deploy-phase transition {old} -> {new}'[:200],
        detail=(
            f'{actor} attempted an illegal deploy-phase '
            f'transition {old!r} -> {new!r} for task {task_id}. This '
            f'indicates a bug in {actor} or corrupted deploy_state metadata — '
            'investigate before resuming.'
        ),
        level=2,
    )
    escalation_queue.submit(esc)
    logger.error(
        '%s: filed L2 illegal_deploy_transition escalation '
        '%s for task %s (%s -> %s)',
        actor, esc.id, task_id, old, new,
    )


def file_deterministic_escalation(
    escalation_queue: EscalationQueue,
    task_id: str,
    *,
    actor: str,
    summary: str,
    detail: str,
    category: str = 'infra_issue',
) -> str | None:
    """File a born-at-L2 escalation unless one of ours is pending; its id, or None when skipped.

    *actor* names the filer in the log lines.  Dedup rationale:
    ``DeterministicRunner._file_infra_issue_and_block``.
    """
    existing_pending = escalation_queue.get_by_task(
        task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
    )
    if existing_pending:
        logger.info(
            '%s: task %s already has %d pending escalation(s) — '
            'skipping re-file (%s dedup guard)',
            actor, task_id, len(existing_pending), category,
        )
        return None
    esc = Escalation(
        id=escalation_queue.make_id(task_id),
        task_id=task_id,
        agent_role=DETERMINISTIC_AGENT_ROLE,
        severity='critical',
        category=category,
        summary=summary[:200],
        detail=detail,
        level=2,
    )
    escalation_queue.submit(esc)
    logger.info(
        '%s: filed L2 %s escalation %s for task %s',
        actor, category, esc.id, task_id,
    )
    return esc.id


@dataclass(frozen=True)
class GateEscalationFields:
    """The fields of a pure gate's born-at-L2 escalation that depend on the task."""

    category: str
    summary: str
    detail: str
    options: tuple[str, ...]
    review_due: ReviewDueState | None


def gate_escalation_category(metadata: dict) -> str:
    """The category a pure gate's done provenance cites: ``review_due`` for a valid human-gate chain."""
    if isinstance(human_gate_chain(metadata), TriggerChain):
        return REVIEW_DUE_CATEGORY
    return MILESTONE_GATE_CATEGORY


def build_gate_escalation_fields(
    task: dict | None,
    metadata: dict,
    human_gate: HumanGateFacts | ReviewDueWithheld | None,
    *,
    escalation_id: str,
    fired_at: str,
) -> GateEscalationFields:
    """The one builder both pure-gate producers file through.

    ``DeterministicRunner._file_milestone_gate_and_block`` and
    ``Harness._recover_stranded_deterministic_gate`` both call it with
    *human_gate* from :func:`read_human_gate_facts`, so a recovered gate
    matches the runner's filing. ``HumanGateFacts`` files ``review_due``
    (plans/completion-driven-triggers-prd.md C4); anything else files
    ``milestone_gate``, a ``ReviewDueWithheld`` with a detail line saying why.
    The operational-LLM token prefix applies to either.
    """
    task = task or {}
    dep_ids = task_dependency_ids(task)
    detail_parts = [task.get('description', '')]
    if dep_ids:
        detail_parts.append(f'\nLanded dependencies: {", ".join(dep_ids)}')
    category = MILESTONE_GATE_CATEGORY
    summary = task.get('title', '')
    payload: ReviewDueState | None = None
    if isinstance(human_gate, ReviewDueWithheld):
        detail_parts.append(human_gate.detail_line())
    elif human_gate is not None:
        due = build_review_due(
            human_gate, human_gate_task_id=str(task.get('id')),
            escalation_id=escalation_id, fired_at=fired_at,
        )
        category = REVIEW_DUE_CATEGORY
        summary = due.summary
        payload = due.payload
        detail_parts.extend(due.detail_lines)
    detail = '\n'.join(detail_parts)
    if _is_operational_llm_gate(metadata):
        # Task 2803 (γ): token-first so the token survives the summary's
        # [:200] slice and any tail-only log scrape.
        detail = (
            f'[{OPERATIONAL_LLM_NEEDS_LANE_TOKEN}] This operational ask needs '
            'LLM-operational handling; no automated lane exists yet — resolve '
            'by hand.\n\n'
        ) + detail
        summary = f'{OPERATIONAL_LLM_NEEDS_LANE_TOKEN}: {summary}'
    return GateEscalationFields(
        category=category,
        summary=summary[:200],
        detail=detail,
        options=tuple(metadata.get('gate_options') or ()),
        review_due=payload,
    )


async def read_human_gate_facts(
    scheduler, task: dict | None, metadata: dict,
) -> HumanGateFacts | ReviewDueWithheld | None:
    """A human gate's chain and facts, or why it withholds ``review_due``; ``None`` for any other gate.

    Both producers call it before :func:`build_gate_escalation_fields`, so
    they cannot read the chain, the project or the completion gate
    differently. Nothing is read for a gate that is not a human gate.
    """
    chain = human_gate_chain(metadata)
    if chain is None:
        return None
    if isinstance(chain, InvalidTriggerChain):
        return ReviewDueWithheld(f'trigger_chain is invalid: {chain.reason}')
    project_id = _configured_project_id(scheduler)
    project_root = _configured_project_root(scheduler)
    if project_id is None or project_root is None:
        unusable = ' and '.join(
            name for name, value in (
                ('fused_memory.project_id', project_id), ('project_root', project_root),
            ) if value is None
        )
        return ReviewDueWithheld(f'the orchestrator config has no usable {unusable}')
    return HumanGateFacts(
        chain=chain,
        project_id=project_id,
        project_root=str(project_root),
        completion_gate=await _read_completion_gate(scheduler, task or {}),
    )


async def _read_completion_gate(scheduler, task: dict) -> CompletionGateReading:
    """Every dependency, read concurrently, then :func:`first_completion_gate` in dependency order."""
    dep_ids = task_dependency_ids(task)
    reads = await asyncio.gather(
        *(_read_dependency(scheduler, task.get('id'), dep_id) for dep_id in dep_ids),
    )
    return first_completion_gate(
        completion_gate_reading(dep_id, read) for dep_id, read in zip(dep_ids, reads, strict=True)
    )


async def _read_dependency(scheduler, task_id: object, dep_id: str) -> TaskRead:
    try:
        return await scheduler.read_task(dep_id)
    except Exception as exc:
        logger.warning(
            'DeterministicRunner: task %s dependency %s read raised %s: %s',
            task_id, dep_id, type(exc).__name__, exc,
        )
        return TaskReadFailed(f'{type(exc).__name__}: {exc}')


class DeterministicRunner:
    """Per-slot runner for deterministic gate tasks.

    Constructed by ``Harness._run_deterministic_slot`` with only the minimal
    dependencies needed (no git_ops — provably no worktree creation).

    Args:
        scheduler: The orchestrator Scheduler instance.
        escalation_queue: The in-process EscalationQueue for filing L2 gates.
        unit_inspector: Optional callable ``(unit: str) -> dict`` returning
            ``{MainPID, ActiveState, ActiveEnterTimestamp,
            ActiveEnterTimestampMonotonic}`` for the given systemd unit.
            Defaults to ``_default_inspect_unit`` (systemctl --user show).
            Injected in tests to avoid touching real systemd.
        script_runner: Optional callable ``(before_done: dict) -> (rc, tail)``
            that runs the deploy script to completion.  Defaults to
            ``_default_run_script`` (awaited create_subprocess_exec).
            Injected in tests to avoid spawning real processes.
            Seam contract (task 4065): an injected runner returns a plain
            ``(rc, tail)`` and never raises ``ScriptTimeout`` — only the
            default implementation does — so on the γ-predicate path an
            injected runner's non-zero rc is always a milestone VERDICT,
            whatever its tail says.  A custom runner wanting the infra-fault
            classification for its own timeout must raise ``ScriptTimeout``
            explicitly; see that class's docstring.
        writeback_backoffs: Bound + pacing for the post-deploy verify+writeback
            retry loop (task 2066), as a list of between-attempt sleep
            seconds (``attempts = len(writeback_backoffs) + 1``).  Defaults
            to ``None``, which resolves to the shared
            ``orchestrator.fm_retry.fm_retry_backoffs()`` schedule (task
            2706) once, at construction time.
        sleeper: Optional async callable ``(seconds: float) -> None`` used to
            pace writeback retries.  Defaults to ``asyncio.sleep``.  Injected
            in tests so retries don't actually sleep.
        reap_grace_secs: Bound (seconds) on reaping a subprocess after a
            whole-process-group SIGKILL on timeout (task 2090).  Defaults to
            ``before_done_script.REAP_GRACE_SECS``.  Injected in tests so an unkillable-process
            test doesn't actually wait.
        run_timeout_grace_secs: Extra margin (seconds), added to
            ``before_done['timeout_secs']``, for the outer wall-clock guard
            around the cross-unit ``run_fn`` call in ``run()`` (task 2090).
            Defaults to ``_RUN_TIMEOUT_GRACE_SECS``.  Injected in tests so a
            stuck-run_fn test doesn't actually wait.
        inspect_timeout_secs: Bound (seconds) on the default unit-inspector's
            ``systemctl --user show`` ``communicate()`` call (task 2091).
            Defaults to ``_INSPECT_TIMEOUT_SECS``.  Injected in tests so a
            wedged-inspect test doesn't actually wait.
        factory_root_resolver: Optional callable ``() -> Path | None`` locating
            the dark-factory root a ``before_done.script_root='factory'``
            script resolves under.  Defaults to
            ``shared.repo_paths.resolve_dark_factory_root``; consulted only for
            such scripts.  ``None`` raises
            ``shared.before_done_paths.FactoryRootUnresolved``.
        predicate_recheck_stall_cap: How many consecutive exit-75 "not yet"
            verdicts a re-arming predicate may return before the runner files
            ``milestone_check_stalled`` (PRD completion-driven-triggers C1).
            ``Harness._run_deterministic_slot`` passes the live
            ``deterministic.predicate_recheck_stall_cap``; the default is
            ``orchestrator.config.PREDICATE_RECHECK_STALL_CAP``.
        clock: Optional callable ``() -> datetime`` (aware, UTC) timing a
            predicate check (the re-arm, its streak record and the verdict
            filer's ``gate_escalated_at``) and a pure gate's filing (its
            ``gate_escalated_at`` and ``review_due.fired_at``).  Defaults to
            ``datetime.now(UTC)``.
            Injected in tests for exact ``milestone.at`` assertions.
    """

    def __init__(
        self,
        scheduler,
        escalation_queue: EscalationQueue,
        unit_inspector=None,
        script_runner=None,
        own_unit_resolver=None,
        restart_scheduler=None,
        writeback_backoffs: list[float] | None = None,
        sleeper=None,
        reap_grace_secs=REAP_GRACE_SECS,
        run_timeout_grace_secs=_RUN_TIMEOUT_GRACE_SECS,
        inspect_timeout_secs=_INSPECT_TIMEOUT_SECS,
        factory_root_resolver: Callable[[], Path | None] | None = None,
        predicate_recheck_stall_cap: int = PREDICATE_RECHECK_STALL_CAP,
        clock: Callable[[], datetime] | None = None,
    ):
        self.scheduler = scheduler
        self.escalation_queue = escalation_queue
        self._unit_inspector = unit_inspector
        self._script_runner = script_runner
        self._own_unit_resolver = own_unit_resolver
        self._restart_scheduler = restart_scheduler
        self._writeback_backoffs = (
            writeback_backoffs if writeback_backoffs is not None
            else fm_retry_backoffs(FM_RESTART_RETRY_WINDOW_SECS)
        )
        self._sleeper = sleeper or asyncio.sleep
        self._reap_grace_secs = reap_grace_secs
        self._run_timeout_grace_secs = run_timeout_grace_secs
        self._inspect_timeout_secs = inspect_timeout_secs
        self._factory_root_resolver = factory_root_resolver or resolve_dark_factory_root
        self._predicate_recheck_stall_cap = predicate_recheck_stall_cap
        self._clock = clock or (lambda: datetime.now(UTC))

    # ------------------------------------------------------------------
    # Default injectable seam implementations
    # ------------------------------------------------------------------

    def _default_resolve_own_unit(self) -> str:
        """Return the orchestrator's own systemd unit name from ORCH_UNIT env var.

        Returns an empty string if ORCH_UNIT is not set (fail-open to cross-unit
        path so existing CI tests with ORCH_UNIT unset stay on the cross-unit path).
        Operators set ORCH_UNIT in the [Service] Environment of the orchestrator unit.
        """
        return os.environ.get('ORCH_UNIT', '')

    def _before_done_paths(self, before_done: dict) -> BeforeDonePaths:
        root = _configured_project_root(self.scheduler)
        if root is None:
            root = Path.cwd()
            logger.warning(
                'DeterministicRunner: scheduler carries no configured project_root — '
                'resolving before_done paths against the process cwd %s',
                root,
            )
        return resolve_before_done_paths(
            before_done, root.resolve(), self._factory_root_resolver,
        )

    async def _default_schedule_detached_restart(
        self,
        before_done: dict,
        *,
        transient_unit: str,
        on_active_secs: int,
        task_id: str,
        summary: str = '',
    ) -> tuple[int, str]:
        """Schedule a detached systemd-run transient unit for a self-restart.

        Thin ``proc_supervision.RestartPlan`` caller (task 2238/δ) — mirrors
        ``service_restart.schedule_detached_systemd_restart``'s conversion
        (task 2237/γ). Builds a same-unit, DETACHED ``RestartPlan``
        (``target_unit == own_unit == transient_unit``, ``verify=None``,
        ``transient_unit`` set) carrying an ``EscalationSpec`` for the RP-4
        on-failure wrapper, and delegates the actual systemd-run registration
        to ``plan.execute()``. ``RestartPlan.execute()``'s detached path
        (``_execute_detached_systemd_run``) builds a SINGLE ``--on-active``
        transient unit whose payload is a ``/bin/sh -c`` wrapper that runs the
        restart script and, *only if it exits non-zero*, fires δ's
        escalation-submit CLI before re-raising the original exit code (so
        journald records the unit as failed):

            <script> <args>
            __rc=$?
            if [ "$__rc" -ne 0 ]; then <escalation submit …>; __esc=$?
              if [ "$__esc" -ne 0 ]; then echo "RP-4: …" >&2; exit 97; fi
            fi
            exit "$__rc"

        A FAILING ``escalation submit`` is no longer swallowed (task 3404): it
        exits ``proc_supervision.RP4_ESCALATION_SUBMIT_FAILED_RC`` with an
        ``RP-4: on-failure escalation submit failed rc=… (payload rc=…)`` line
        on stderr, so an unfiled L2 is both visible in journald and
        machine-distinguishable.  When the submit succeeds the wrapper still
        exits the payload's own code, so journald keeps the real restart cause.

        Why not a separate ``OnFailure=`` handler unit?  ``systemd-run`` has no
        register-without-start mode — registering a companion handler transient
        *service* would activate it immediately at scheduling time, filing a
        spurious born-at-L2 on EVERY successful self-deploy rather than only on a
        fire-time failure (the bug this method previously had).  ``--on-active``
        defers the whole unit, so NOTHING runs at registration; the escalation
        is reached only through the shell's failure branch when the restart
        actually fails at fire time.  This preserves the intended semantics —
        "run δ's escalation-submit CLI iff the restart fires and fails" — with a
        single deferred unit and no eager execution.

        systemd-run returns immediately after registering the transient unit;
        the orchestrator is NEVER blocked or killed — the payload fires later
        under the user systemd manager.

        Returns:
            (rc, tail) — rc=0 on successful registration; rc≠0 if registration
            fails (tail carries the error detail).
        """
        target_unit = before_done.get('target_unit', 'unknown')
        args = before_done.get('args') or []
        # docs/task-authoring.md §5: paths resolve against the configured
        # project_root (shared/src/shared/before_done_paths.py).
        paths = self._before_done_paths(before_done)

        esc_summary = summary or (
            f'Self-restart fire-time failure: {target_unit}'
        )

        # δ's escalation-submit CLI, fired ONLY when the restart fails at fire
        # time — built here as data and handed to RestartPlan/EscalationSpec,
        # which owns the RP-4 on-failure ``/bin/sh -c`` wrapper + the
        # ``python -m escalation submit`` argv construction (byte-identical to
        # this method's prior inline argv — see EscalationSpec.to_submit_argv).
        #
        # Deployment assumption: the `escalation` package must be importable
        # from sys.executable's interpreter.  That importability rests on TWO
        # independent legs (task 3453, both MEASURED):
        #
        #  1. Interpreter identity — `sys.executable` IS the workspace venv
        #     interpreter (`uv run --frozen --project orchestrator` resolves
        #     ExecStart to `<repo>/.venv/bin/python3`, which carries an editable
        #     install of `escalation`).  This leg is load-bearing and cannot be
        #     substituted: `python -S -m escalation submit` with the src roots on
        #     PYTHONPATH still dies at `shared.async_sqlite_base` ->
        #     `import aiosqlite`, a site-packages-only wheel.  So PYTHONPATH
        #     alone is NOT sufficient.
        #  2. Workspace src roots — supplied explicitly by the
        #     `--setenv=PYTHONPATH=` token `RestartPlan`'s detached argv now
        #     carries (proc_supervision._submit_child_pythonpath).  This is the
        #     only leg systemd-run can break: it propagates NONE of the caller's
        #     environment to a transient unit.
        #
        # NOT via "a PYTHONPATH side-channel from the orchestrator service
        # unit", as this comment previously claimed — MEASURED FALSE:
        # orchestrator-dark-factory.service sets only Environment=PATH=/LANG=/
        # ORCH_UNIT= and no PYTHONPATH, and systemd-run would not propagate it
        # anyway.
        #
        # If `escalation` is nonetheless unimportable, the OnFailure branch
        # itself fails and no L2 is filed — the task is already marked
        # done=scheduled at this point, so nothing downstream would notice.
        # That loss is reported twice over, at both ends of the deferral:
        #
        #  - At REGISTRATION time, `RestartPlan._execute_detached_systemd_run`
        #    WARNs as soon as it can prove the import fails in-process (the
        #    `escalation.submit` canary, task 3453) — while a human is still
        #    watching the deploy.
        #  - At FIRE time, the loss is no longer SILENT either (task 3404): the
        #    fired unit exits the reserved
        #    proc_supervision.RP4_ESCALATION_SUBMIT_FAILED_RC and prints an
        #    `RP-4: on-failure escalation submit failed rc=… (payload rc=…)`
        #    line, so journald carries both the fact and the cause.
        #
        # Operators can still verify the same thing directly, ahead of a deploy:
        #   <sys.executable> -c "import escalation"
        # A marker-file fallback is intentionally not implemented here to keep
        # the failure path auditable via journald.
        #
        # End-to-end coverage of this branch lives in
        # test_deterministic_runner.py's TestDefaultScheduleDetachedRestart,
        # which fires the deferred wrapper for real.  Those tests now preflight
        # interpreter importability via `_assert_submit_cli_invokable` and hand
        # the child the repo src roots on PYTHONPATH — a fresh interpreter
        # inherits none of conftest.py's in-process sys.path injection, so
        # without that the branch silently filed nothing in a venv lacking the
        # `escalation` editable install (task 3404).
        escalation_spec = EscalationSpec(
            queue_dir=str(self.escalation_queue.queue_dir),
            task_id=task_id,
            summary=esc_summary,
            detail=(
                f'Transient unit {transient_unit} fired and failed (task {task_id}). '
                f'Check journald for restart output: '
                f'journalctl --user -u {transient_unit}'
            ),
            severity='critical',
            category=SELF_RESTART_FAILURE_CATEGORY,
            agent_role=DETERMINISTIC_AGENT_ROLE,
        )

        plan = RestartPlan(
            script=paths.script,
            args=list(args),
            cwd=paths.cwd,
            target_unit=transient_unit,
            own_unit=transient_unit,
            on_failure_escalation=escalation_spec,
            verify=None,
            transient_unit=transient_unit,
            on_active_secs=on_active_secs,
            env=dict(before_done.get('env') or {}),
        )
        outcome = await plan.execute()
        if outcome.disposition == RestartDisposition.REGISTRATION_FAILED:
            logger.warning(
                'DeterministicRunner: failed to register restart transient unit %s '
                'for task %s: %s',
                transient_unit, task_id, outcome.detail,
            )
            return 1, outcome.detail
        return 0, ''

    async def _default_inspect_unit(
        self, unit: str, *, degradation_sink: dict | None = None,
    ) -> dict:
        """Query systemctl for unit state fields needed for fresh-PID verify.

        Task 2119: thin delegate to the hoisted, hardened
        ``systemd_inspect.inspect_systemd_unit`` (task 2091's timeout/kill/
        reap hardening now lives there exactly once). Returns a dict with at
        minimum: MainPID (int), ActiveState (str), ActiveEnterTimestamp
        (str), ActiveEnterTimestampMonotonic (int). Integers default to 0 on
        parse failure (sentinel-safe).

        ``degradation_sink`` (task 4157 reviewer amendment) is forwarded
        verbatim: an optional out-dict into which ``inspect_systemd_unit``
        records WHY it degraded to the sentinel, so the escalation an operator
        reads can name the cause instead of stranding it in the journal.
        """
        return await inspect_systemd_unit(
            unit,
            timeout_secs=self._inspect_timeout_secs,
            reap_grace_secs=self._reap_grace_secs,
            degradation_sink=degradation_sink,
        )

    async def _inspect_unit_guarded(
        self, inspect_fn, unit: str,
    ) -> tuple[dict, str | None]:
        """Await the unit inspector, degrading an ``OSError`` to the wedged sentinel.

        Task 4157. Used by ALL THREE ``run()`` call sites that await a unit
        inspector — the crash-window re-verify, the pre-deploy baseline
        capture, and (reviewer amendment) the post-deploy verify re-inspect
        inside ``_capturing_inspector`` — so the guard exists once rather than
        three times.

        WHY this exists at the call site even though
        ``systemd_inspect.inspect_systemd_unit`` is itself fail-closed:
        ``inspect_fn`` resolves to ``self._unit_inspector or
        self._default_inspect_unit``, and ``_unit_inspector`` is a
        constructor-injectable seam. The default's contract does not bind an
        injected inspector, so hardening only the source would leave ``run()``'s
        documented "always returns BLOCKED, never a raw exception" contract
        (stated in ``deterministic_runner.py::DeterministicRunner._writeback_deploy_success``'s
        task-2240 stamp-only fallback, immediately above its ``except
        IllegalDeployTransition``) false for any caller that uses the seam —
        and a guarantee that holds only when a documented seam is unused is not
        a guarantee.

        Catches ``OSError`` ONLY, never a bare ``Exception``: ``run()``
        deliberately raises ``ValueError`` (the sole entry in its documented
        ``Raises:`` section) and ``NotImplementedError`` (the "gate resolved
        but before_done_ran_at is not set" operator guard), and swallowing
        either into a silent BLOCKED escalation would convert a loud
        authoring/state defect into a silent one. Note ``TimeoutError`` IS an
        ``OSError`` subclass and so is covered on purpose: an inspector that
        times out is degraded, not broken, and belongs on the same fail-closed
        path.

        Returns ``(state, cause)``:

        * ``state`` — the inspector's dict unchanged on success; on a
          degradation, :func:`~orchestrator.systemd_inspect._wedged_unit_sentinel`,
          whose four fields every existing fail-closed consumer already rejects
          (the verify leg's ``pid > 0`` check, the baseline leg's task-2091
          ``ActiveState`` gate), so no new downstream branch is needed.
        * ``cause`` — ``None`` when the reading is trustworthy, else a short
          string naming the real failure (errno/exception repr, or the timeout
          budget). The sentinel deliberately renders identically for EMFILE, a
          missing ``systemctl`` and a hung systemd, so WITHOUT this the three
          would be indistinguishable in the escalation an operator reads while
          calling for three different fixes — a fact the emitter held in a
          variable and dropped, which is what INV-2
          ``structured-facts-at-failure`` forbids. Callers append it to their
          own escalation detail; it never enters the sentinel dict, so
          classification stays byte-identical (reviewer amendment).
        """
        # The sink is only meaningful for the DEFAULT delegate, which swallows
        # its own OSError/timeout inside `inspect_systemd_unit` and would
        # otherwise have no way to report which mode fired. An INJECTED
        # inspector is called verbatim, exactly as before — the seam's contract
        # is `inspect_fn(unit)` and must not silently grow a keyword argument.
        sink: dict = {}
        try:
            if self._unit_inspector is None:
                state = await self._default_inspect_unit(unit, degradation_sink=sink)
            else:
                state = await inspect_fn(unit)
        except OSError as exc:
            logger.warning(
                'DeterministicRunner: unit inspect of %s failed to run (%r) — '
                'degrading to the MainPID=0 sentinel so the failure routes '
                'through the existing fail-closed path instead of escaping run()',
                unit, exc,
            )
            return _wedged_unit_sentinel(), f'unit inspect raised {exc!r}'
        return state, sink.get('cause')

    async def _default_run_script(self, before_done: dict) -> tuple[int, str]:
        """Run *before_done*'s script to completion through ``run_script``.

        Resolves the script and cwd against the configured project_root
        (``_before_done_paths``) and hands ``run_script`` the typed values.
        Raises ``ScriptTimeout`` when the script overruns
        ``before_done['timeout_secs']``; see
        ``orchestrator/src/orchestrator/before_done_script.py::ScriptTimeout``.
        """
        paths = self._before_done_paths(before_done)
        return await run_script(
            paths.script,
            before_done.get('args') or [],
            cwd=paths.cwd,
            env_overrides=before_done.get('env'),
            timeout_secs=before_done.get('timeout_secs', 60),
            reap_grace_secs=self._reap_grace_secs,
        )

    async def _file_infra_issue_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
        *,
        metadata: dict | None = None,
        category: str = 'infra_issue',
    ) -> WorkflowOutcome:
        """File a born-at-L2 infra_issue escalation and set the task to blocked.

        Reuses β's escalation construction pattern (sentinel role keeps level=2
        past the server downgrade gate).  Includes a dedup guard — if a pending
        escalation already exists (e.g. prior crash-safe re-dispatch), filing is
        skipped to avoid duplicate L2 escalations.

        The trailing ``set_task_status(..., 'blocked')`` write is best-effort
        (task 2066 amendment): callers — including
        ``_writeback_deploy_success`` on writeback-budget exhaustion — may
        reach this method with the scheduler's own connection persistently
        severed, i.e. the same failure mode that triggered the escalation in
        the first place.  The escalation itself is already durable at this
        point (``EscalationQueue.submit`` writes to local disk, independent of
        fused-memory), so a failure to *also* persist the 'blocked' status
        must not propagate — doing so would defeat this method's "always
        returns BLOCKED, never a raw exception" contract in exactly the
        scenario it exists to cover.

        ``category`` (task 4678 / r3, PRD ``docs/prds/
        recurring-deterministic-tasks.md`` decision R-D6) overrides the filed
        category.  The ONLY non-default caller is ``_run_predicate``'s three
        no-verdict arms when the task is a RECURRENCE CARRIER, which file
        ``MILESTONE_CHECK_FAILED_CATEGORY`` instead so every failure leg of a
        recurring chain sits in one deny-listed, discriminable category
        (contract C-5).  Nothing else changes: the dedup guard, the deploy-only
        phase advance, the best-effort blocked write and the "always returns
        BLOCKED, never a raw exception" contract are all category-agnostic, and
        the three log lines below interpolate the category so a carrier's logs
        read ``milestone_check_failed`` rather than a lie.

        THIS method — not ``_file_predicate_escalation_and_block`` — is the
        right home for that override precisely BECAUSE it does not stamp
        ``gate_escalated_at``.  The verdict helper does, and task 4065
        established that a no-verdict leg must NOT (the check is re-attempted
        on the next dispatch instead of being latched into section-1's
        resolve-to-done path).  r3 is a CATEGORY change only, so routing the
        carrier arms through the stamping helper would smuggle in a semantic
        change nobody asked for.  Widened here rather than copied into a new
        predicate-specific helper for the reason task 2632's amendment gave
        when it extracted ``_deploy_outer_timeout``: a second copy of the dedup
        guard and the best-effort blocked write could silently drift apart, on
        a path whose whole contract is that it never raises.

        ``metadata``, when passed by a DEPLOY-path caller (``before_done``
        set — every ``run()``-internal call site qualifies; ``_run_predicate``
        never passes it), best-effort advances ``deploy_state.phase`` to
        ``ESCALATED`` too (ζ DS-1) — same tolerate-a-severed-connection
        posture as the trailing blocked-status write below, since the
        escalation above is already durable regardless. Skipped entirely
        (reviewer amendment, task 2240) when the CURRENT phase is already
        ``ESCALATED`` or ``DONE``: on the rare crash-resume edge where the
        deploy already reached one of those phases but resolution could not
        be proven (e.g. the runner's own escalation record is missing or
        expired), a bare re-advance would attempt a pinned-illegal
        self-loop, filing a spurious ``illegal_deploy_transition`` L2 on top
        of the ``infra_issue`` one filed above — the skip avoids that noise
        without weakening DS-2 loudness, since the phase is already at (or
        past) the target.

        Consequence of that phase advance failing (reviewer amendment, task
        2240): if it raises transiently — e.g. the SAME severed connection
        that triggered this escalation in the first place — the disk-backed
        escalation above is still filed, but ``deploy_state.phase`` stays at
        its PRE-escalation value (typically ``RAN``) instead of advancing to
        ``ESCALATED``. Once a human resolves that escalation, the runner's
        resume path reads ``phase != ESCALATED`` and treats it as a
        crash-window rather than a resolved gate — re-filing a fresh
        ``infra_issue`` and blocking again instead of driving to done, i.e.
        one spurious extra human round-trip. This self-heals: the re-filed
        escalation's OWN best-effort advance retries the SAME
        ``RAN -> ESCALATED`` edge, and once the connection recovers it lands,
        after which the next resolution correctly proves ``phase ==
        ESCALATED`` and resumes to done. No state is corrupted either way —
        only convergence is delayed by one extra escalate/resolve cycle.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        file_deterministic_escalation(
            self.escalation_queue, task_id,
            actor=_RUNNER_ACTOR, summary=summary, detail=detail, category=category,
        )

        if metadata is not None and metadata.get('before_done') is not None:
            _current_deploy_state = DeployState.from_metadata(metadata)
            if _current_deploy_state is not None and _current_deploy_state.phase in (
                DeployPhase.ESCALATED, DeployPhase.DONE,
            ):
                # Reviewer amendment (task 2240): already at (or past)
                # ESCALATED — e.g. this is the rare crash-resume edge where
                # phase==ESCALATED but resolution_proven is false (the
                # runner's own escalation record is missing/expired), so
                # execution reaches this unknown-crash infra_issue path
                # again. A bare re-advance would attempt an illegal
                # ESCALATED->ESCALATED (or DONE->ESCALATED) self-loop —
                # neither edge is in _LEGAL — filing a SPURIOUS born-at-L2
                # illegal_deploy_transition escalation on top of the
                # infra_issue one just filed above. Skip the redundant
                # advance; the infra_issue escalation is already the loud
                # signal for this crash.
                logger.debug(
                    'DeterministicRunner: task %s deploy_state already at '
                    'phase=%s — skipping redundant ESCALATED advance',
                    task_id, _current_deploy_state.phase,
                )
            else:
                try:
                    await self._advance_deploy_phase(
                        task_id, metadata, DeployPhase.ESCALATED,
                        phase_timestamp=datetime.now(UTC).isoformat(),
                    )
                except Exception as exc:
                    logger.warning(
                        'DeterministicRunner: task %s deploy_state phase-escalated '
                        'advance failed (%s: %s) — the infra_issue escalation above '
                        'is already durable regardless',
                        task_id, type(exc).__name__, exc,
                    )

        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info('DeterministicRunner: task %s blocked — %s', task_id, category)
        except Exception as exc:
            # Do NOT let a still-severed connection turn this into a
            # propagated exception — it would bubble past run() into the
            # harness's generic handler and produce a SILENT blocked report,
            # defeating the point of filing a durable escalation above.
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the infra_issue '
                'escalation is already durable on local disk, so returning '
                'BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )
        return WorkflowOutcome.BLOCKED

    async def _file_stop_instruction_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
    ) -> WorkflowOutcome:
        """File a born-at-L2 stop_instruction escalation and set the task to blocked.

        Task 2509 (reconciliation finding 0aac21b4): mirrors
        ``_file_infra_issue_and_block``'s dedup guard, escalation construction,
        and best-effort blocked-status writeback, but uses ``category=
        'stop_instruction'`` so the halt is distinguishable from an infra
        fault — and, critically, does NOT stamp ``metadata.gate_escalated_at``.

        Unlike ``_file_infra_issue_and_block``'s dedup guard, this one is
        scoped to ``category == 'stop_instruction'`` (review amendment): a
        pre-existing, unrelated pending escalation on the same task (e.g. an
        ``infra_issue`` filed by a prior crash) must not silently suppress
        filing THIS one — the stop-instruction halt is a distinct, higher-
        authority signal that a human resolving the other escalation could
        otherwise miss entirely.
        This is a re-checkable HALT, not a resolve-to-done gate: leaving
        ``gate_escalated_at`` unset means the next dispatch re-evaluates the
        stop-instruction guard from scratch — section 1's
        ``if gate_escalated_at:`` resume-to-done branch is never taken —
        rather than silently driving to done with no proof the deploy ran.

        That re-evaluation only happens once the task is back in
        ``pending``, though: a task in ``blocked`` holding this open pending
        escalation is NOT re-dispatched automatically (reviewer amendment —
        see task 2509 review). So the full recovery is two steps, not one:
        (1) edit the task description to remove the stop instruction — if
        left in place, the guard simply re-fires and re-escalates on the
        very next dispatch, since it re-reads the description fresh every
        time — and (2) resolve this escalation (e.g. ``resolve_issue``
        action ``resume`` or ``restart``, either of which flips the task's
        status back to ``pending`` per ``escalation.server.resolve_issue``'s
        Table B — this runner has no live agent to resume, so both actions
        have the same practical effect here: a fresh dispatch). Editing the
        description alone, with the escalation left pending, does NOT
        re-enable dispatch. A human may instead cancel the task outright.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        # category-scoped (review amendment): get_by_task has no server-side
        # category filter, so filter client-side — an unrelated pending
        # escalation (e.g. infra_issue) on the same task must not suppress
        # filing this one. See the docstring note above.
        existing_pending = [
            e for e in self.escalation_queue.get_by_task(
                task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
            )
            if e.category == 'stop_instruction'
        ]
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending '
                'stop_instruction escalation(s) — skipping re-file '
                '(stop_instruction dedup guard)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category='stop_instruction',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 stop_instruction escalation %s for task %s',
                esc.id, task_id,
            )

        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info('DeterministicRunner: task %s blocked — stop_instruction', task_id)
        except Exception as exc:
            # Same rationale as _file_infra_issue_and_block: a still-severed
            # connection must not turn this into a propagated exception — the
            # escalation is already durable on local disk, so return BLOCKED
            # regardless.
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the stop_instruction '
                'escalation is already durable on local disk, so returning '
                'BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )
        return WorkflowOutcome.BLOCKED

    async def _file_curator_adjudication_missing_and_block(
        self, task_id: str, task: dict,
    ) -> WorkflowOutcome:
        """File a born-at-L2 ``curator_adjudication_missing`` escalation and block.

        Task 3341.  Reached when a ``human_curator_gate`` pure gate resumes with
        its own escalation record closed but NO ``human_curator_adjudicated_at``
        stamp — i.e. the record says "someone closed the gate", nothing says
        "the human reviewed the content".

        Modelled on ``_file_stop_instruction_and_block``: a born-at-L2
        ``Escalation`` that survives the server's downgrade gate, and a
        best-effort blocked writeback wrapped so a severed fused-memory
        connection cannot turn a durable on-disk escalation into a propagated
        exception.

        NO dedup guard (reviewer amendment).  ``_file_stop_instruction_and_block``
        carries one because its caller is not gated on the pending set; this
        method's sole call site sits inside section 1's ``else:`` of
        ``if pending:``, where ``pending`` is the *identical*
        ``get_by_task(task_id, status='pending', agent_role=...)`` query — so a
        dedup filter here is provably always empty, i.e. dead code asserting a
        justification that does not transfer.  A future second call site NOT
        gated that way must re-introduce a category-scoped filter.

        Two deliberate omissions:

        * It does NOT re-stamp ``gate_escalated_at`` — already stamped; this is
          a re-ask on an established gate, not a new one.
        * It does NOT re-file a ``milestone_gate``. A DISTINCT category keeps
          the census legible, stops the re-ask presenting to a resolver as
          "the same gate again" (which would invite the identical auto-
          resolution that caused task 3181), gives the harness ``reblock_guard``
          its own signature — it keys on ``category:summary``, so INV-4's
          storm-escape is satisfied without building a second counter (INV-5) —
          and lets a future watcher policy special-case "never auto-resolve
          this category".

        Returns:
            WorkflowOutcome.BLOCKED
        """
        title = task.get('title') or task_id
        summary = f'Human curator gate {task_id} resumed without content adjudication'
        # The configured per-project root — the SAME value the scheduler puts on
        # its own real update_task MCP calls. Deliberately NOT os.getcwd(): every
        # fleet orchestrator unit pins WorkingDirectory to the dark-factory
        # checkout (so `uv run --project orchestrator` resolves) while selecting
        # its actual target project via --config, so a cwd-derived value would
        # silently aim most operators at the wrong project.
        project_root = _snippet_project_root(self.scheduler)
        detail = (
            f"Task {task_id} ({title!r}) is a deterministic pure gate marked "
            f"`metadata.{HUMAN_CURATOR_GATE_KEY}`, meaning it closes only when a "
            f"human has adjudicated its CONTENT — not merely when its escalation "
            f"record is closed.\n\n"
            f"An escalation record for this gate exists and is no longer pending, "
            f"so the task-2954 proof check passed. But "
            f"`metadata.{HUMAN_CURATOR_ADJUDICATED_AT_KEY}` is absent or is not a "
            f"non-empty string, so there is NO evidence the per-entry review "
            f"actually happened. Driving to done here would repeat task 3181, "
            f"where esc-3181-1 was resolved by the automated escalation-watcher "
            f"whose own resolution text said the curator work was "
            f"'deliberately NOT executed here' — and the task was nonetheless "
            f"closed with the generic note 'pure gate resolved'.\n\n"
            f"REMEDIATION — one of:\n"
            f"  (a) Perform the per-entry content review this gate asks for, then "
            f"stamp the proof and resolve this escalation:\n"
            f"      update_task(id={task_id!r}, project_root={project_root!r}, "
            f"metadata={{'{HUMAN_CURATOR_ADJUDICATED_AT_KEY}': "
            f"'<ISO-8601 timestamp>'}}, metadata_mode='merge')\n"
            f"      The stamp must be a non-empty string; a bare `true` is NOT "
            f"accepted (it asserts the conclusion without recording when the "
            f"review happened).\n"
            f"  (b) Cancel the task, if the content work is no longer wanted.\n\n"
            f"Resolving THIS escalation without doing (a) or (b) will simply "
            f"re-file it on the next dispatch."
        )

        # No dedup filter — see the docstring: the sole call site already ran
        # the identical pending query and only reaches here when it came back
        # empty, so any filter written here would be unreachable by
        # construction.
        esc = Escalation(
            id=self.escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role=DETERMINISTIC_AGENT_ROLE,
            severity='critical',
            category=CURATOR_ADJUDICATION_MISSING_CATEGORY,
            summary=summary[:200],
            detail=detail,
            level=2,
        )
        self.escalation_queue.submit(esc)
        logger.info(
            'DeterministicRunner: filed L2 curator_adjudication_missing '
            'escalation %s for task %s',
            esc.id, task_id,
        )

        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info(
                'DeterministicRunner: task %s blocked — curator adjudication missing',
                task_id,
            )
        except Exception as exc:
            # Same rationale as _file_stop_instruction_and_block: the escalation
            # is already durable on local disk, so return BLOCKED regardless.
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the '
                'curator_adjudication_missing escalation is already durable on '
                'local disk, so returning BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )
        return WorkflowOutcome.BLOCKED

    def _deploy_transition_escalation_sink(
        self, task_id: str, old: DeployPhase, new: DeployPhase,
    ) -> None:
        """DS-2 sink for every ``_advance_deploy_phase``: see :func:`file_illegal_deploy_transition`."""
        file_illegal_deploy_transition(self.escalation_queue, task_id, old, new, actor=_RUNNER_ACTOR)

    def _compute_deploy_phase_advance(
        self,
        task_id: str,
        metadata: dict,
        new_phase: DeployPhase,
        *,
        verify_baseline: VerifyBaseline | dict | None = None,
        phase_timestamp: str | None = None,
    ) -> DeployState:
        """DS-2: enforce the transition and build the advanced ``DeployState``.

        Pure computation, no I/O — reads the CURRENT phase from *metadata*
        (the in-memory dict threaded through ``run()``), enforces the
        transition (files a loud escalation then raises on an illegal edge
        via ``_deploy_transition_escalation_sink``), and returns the new
        ``DeployState`` with the OLD ``ran_at``/``verified_at``/
        ``escalated_at``/``verify_baseline`` evidence carried forward.

        Shared by ``_advance_deploy_phase`` (the single-write path) and
        ``_writeback_deploy_success`` (which folds the resulting
        ``to_metadata()`` into its own retry-loop write rather than issuing a
        second ``update_task`` call — the retry loop needs the RAW boolean
        ``update_task`` return to detect a transient failure, which
        ``_advance_deploy_phase`` does not expose).

        ``verify_baseline``, when omitted, carries forward whatever baseline
        was already recorded — a phase advance must never drop previously
        persisted DS-3 evidence.

        ``phase_timestamp`` (reviewer amendment, task 2240): when given, is
        written into whichever ONE of ``ran_at`` / ``verified_at`` /
        ``escalated_at`` matches *new_phase* (RAN / VERIFIED / ESCALATED
        respectively — a SCHEDULED advance, or any other, has no matching
        field and leaves all three untouched). Callers pass the SAME ISO
        timestamp already being written to the corresponding top-level
        evidence stamp (e.g. ``before_done_ran_at``), so ``DeployState``
        stays self-describing instead of emitting these fields as ``null``
        on every write forever. Omitted (``None``) leaves the OLD
        carried-forward value in place, exactly as before.
        """
        old_state = DeployState.from_metadata(metadata)
        old_phase = old_state.phase if old_state is not None else None
        enforce_transition(
            old_phase, new_phase, task_id=task_id,
            escalation_sink=self._deploy_transition_escalation_sink,
        )

        if verify_baseline is not None and not isinstance(verify_baseline, VerifyBaseline):
            verify_baseline = VerifyBaseline(**verify_baseline)
        carried_baseline = (
            verify_baseline if verify_baseline is not None
            else (old_state.verify_baseline if old_state is not None else None)
        )
        ran_at = old_state.ran_at if old_state is not None else None
        verified_at = old_state.verified_at if old_state is not None else None
        escalated_at = old_state.escalated_at if old_state is not None else None
        if phase_timestamp is not None:
            if new_phase == DeployPhase.RAN:
                ran_at = phase_timestamp
            elif new_phase == DeployPhase.VERIFIED:
                verified_at = phase_timestamp
            elif new_phase == DeployPhase.ESCALATED:
                escalated_at = phase_timestamp
        return DeployState(
            phase=new_phase,
            verify_baseline=carried_baseline,
            ran_at=ran_at,
            verified_at=verified_at,
            escalated_at=escalated_at,
        )

    async def _advance_deploy_phase(
        self,
        task_id: str,
        metadata: dict,
        new_phase: DeployPhase,
        *,
        evidence: dict | None = None,
        verify_baseline: VerifyBaseline | dict | None = None,
        phase_timestamp: str | None = None,
    ) -> DeployState:
        """DS-1/DS-2: atomically advance ``metadata.deploy_state.phase`` + evidence.

        Persists ``{**new DeployState.to_metadata(), **evidence}`` in ONE
        ``update_task(metadata_mode='merge')`` call, and refreshes
        ``metadata['deploy_state']`` in place so a LATER call within the SAME
        ``run()`` invocation observes the just-written phase as ``old``. See
        ``_compute_deploy_phase_advance`` for the transition-enforcement +
        state-construction logic this wraps, including what
        ``phase_timestamp`` does.
        """
        new_state = self._compute_deploy_phase_advance(
            task_id, metadata, new_phase,
            verify_baseline=verify_baseline, phase_timestamp=phase_timestamp,
        )
        payload = {**new_state.to_metadata(), **(evidence or {})}
        await self.scheduler.update_task(task_id, payload, metadata_mode='merge')
        metadata['deploy_state'] = new_state.to_metadata()['deploy_state']
        return new_state

    async def _enrich_deploy_state_baseline(
        self, task_id: str, metadata: dict, baseline: dict,
    ) -> None:
        """DS-3: persist ``verify_baseline`` WITHOUT a phase transition.

        Bypasses ``enforce_transition`` entirely — this is a same-phase
        enrichment (advancing RAN -> RAN would hit the pinned-illegal
        self-loop edge). Best-effort: this is a freshness UPGRADE, never a
        new failure mode — a crash (or a still-severed connection) before
        this write lands leaves ``phase==RAN`` with no baseline, which the
        strand detector still correctly classifies as a RAN-strand and the
        freshness verdict's no-baseline fallback still handles (see ζ design
        decision on verify_baseline persistence timing).
        """
        old_state = DeployState.from_metadata(metadata)
        if old_state is None:
            # Defensive: the shared before_done_ran_at/phase=ran write always
            # precedes baseline capture — nothing to enrich onto if absent.
            return
        verify_baseline = VerifyBaseline(
            active_enter_timestamp_monotonic=baseline.get('ActiveEnterTimestampMonotonic', 0),
            main_pid=baseline.get('MainPID', 0),
        )
        new_state = DeployState(
            phase=old_state.phase,
            verify_baseline=verify_baseline,
            ran_at=old_state.ran_at,
            verified_at=old_state.verified_at,
            escalated_at=old_state.escalated_at,
        )
        try:
            await self.scheduler.update_task(
                task_id, new_state.to_metadata(), metadata_mode='merge',
            )
            metadata['deploy_state'] = new_state.to_metadata()['deploy_state']
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s verify_baseline enrichment failed '
                '(%s: %s) — freshness will fall back to liveness-only; not a new '
                'failure mode (phase==ran is still correctly detected as a strand)',
                task_id, type(exc).__name__, exc,
            )

    async def _writeback_deploy_success(
        self,
        task_id: str,
        metadata: dict,
        new_state: dict,
        target_unit: str,
        description: str,
    ) -> WorkflowOutcome:
        """Patiently persist a successful cross-unit deploy's verify+done writeback.

        Task 2066: the cross-unit deploy script may restart the very service
        backing the orchestrator's own fused-memory/MCP connection (e.g. task
        2059), severing it for the duration of a ``--drain`` restart.  Because
        ``shared.mcp_session.McpSession.call`` auto-reconnects on the next call after any
        transient failure, a plain patient retry — not an explicit reconnect —
        is sufficient: a successful write IS the reconnection proof.

        Retries the ``before_done_verified_at``/``before_done_verified_pid``
        stamp (treating a ``False`` return from ``update_task`` as transient)
        until it lands, then retries the ``done`` write (catching a transient/
        ``RuntimeError`` from ``set_task_status``'s own exhausted retry) —
        both within the SAME bounded, paced ``_writeback_backoffs`` budget
        (task 2706, sized off the shared ``orchestrator.fm_retry`` schedule),
        without re-stamping and without re-running the deploy script (I1
        once-only — this helper only persists an already-completed deploy's
        outcome).

        ζ DS-1: the verified-stamp write also atomically carries
        ``deploy_state.phase=='verified'`` — folded into the SAME write
        computed ONCE up front (the pre-write phase does not change across
        retries within this call), so the shared retry budget stays exactly
        two writes.  The done-write below intentionally does NOT also advance
        phase to ``done`` — ``verified -> done`` is a real DS-2-legal edge,
        but leaving the PERSISTED phase at ``verified`` avoids a third write
        (verified already proves success; done tasks are never swept). If
        computing that advance raises ``IllegalDeployTransition`` (reviewer
        amendment, task 2240 — corrupted/unexpected ``deploy_state``), the
        writeback falls back to a stamp-only payload (no ``deploy_state``
        key) so the already-succeeded deploy still converges to done instead
        of the raise propagating out of ``run()``.

        On budget exhaustion (fused-memory never recovers in-window), files a
        durable local ``infra_issue`` escalation (disk-backed, connection-
        independent) and returns BLOCKED — so the task is never silently
        stranded with an empty escalation queue (the EVIDENCE failure on task
        2059).

        Returns:
            WorkflowOutcome.DONE — verified stamp + done write both persisted.
            WorkflowOutcome.BLOCKED — writeback budget exhausted; infra_issue filed.
        """
        verified_iso = datetime.now(UTC).isoformat()
        pid = new_state.get('MainPID', 0)
        active_enter_timestamp = new_state.get('ActiveEnterTimestamp', '')

        # Reviewer amendment (task 2240): _compute_deploy_phase_advance is
        # pure computation (no I/O) but still enforces DS-2 — an illegal
        # source phase (metadata.deploy_state corrupted or otherwise
        # unexpected) makes enforce_transition file a loud L2 escalation and
        # then raise. This call sits OUTSIDE the retry loop below and the
        # deploy has ALREADY succeeded by this point, so letting the raise
        # propagate out of run() would violate the "run() always returns
        # BLOCKED, never a raw exception" contract at exactly the moment the
        # task would otherwise land neither done nor cleanly blocked. Fall
        # back to a stamp-only write (no deploy_state payload) so the
        # verified deploy still converges to done — the escalation sink
        # already made the anomaly loud (file-before-raise).
        try:
            verified_deploy_state = self._compute_deploy_phase_advance(
                task_id, metadata, DeployPhase.VERIFIED,
                phase_timestamp=verified_iso,
            )
            deploy_state_payload = verified_deploy_state.to_metadata()
        except IllegalDeployTransition as exc:
            logger.warning(
                'DeterministicRunner: task %s deploy_state phase-advance to '
                'VERIFIED failed (%s) — an L2 illegal_deploy_transition '
                'escalation was filed; falling back to a stamp-only '
                'writeback so the verified deploy still converges to done',
                task_id, exc,
            )
            deploy_state_payload = {}

        backoffs = self._writeback_backoffs
        attempts = len(backoffs) + 1
        stamped = False
        for attempt in range(attempts):
            if not stamped:
                stamped = bool(await self.scheduler.update_task(
                    task_id,
                    {
                        'before_done_verified_at': verified_iso,
                        'before_done_verified_pid': pid,
                        **deploy_state_payload,
                    },
                    metadata_mode='merge',
                ))
                if stamped and 'deploy_state' in deploy_state_payload:
                    metadata['deploy_state'] = deploy_state_payload['deploy_state']
                if not stamped:
                    logger.warning(
                        'DeterministicRunner: task %s verified-stamp writeback failed '
                        '(attempt %d/%d) — connection may be severed by the deploy; '
                        'retrying',
                        task_id, attempt + 1, attempts,
                    )
                    if attempt < len(backoffs):
                        await self._sleeper(backoffs[attempt])
                    continue

            # Shared-budget design (task 2066 amendment): the verified-stamp
            # write above and the done-write below intentionally draw from
            # the SAME `_writeback_backoffs`-derived budget and `attempt`
            # counter rather than each getting its own sub-budget.  If the
            # stamp only lands on the final iteration, the done-write gets
            # exactly one remaining try before the loop falls through to the
            # budget-exhausted escalation path below — there is no dedicated
            # done-write retry allowance.  This keeps the overall ceiling
            # simple (the shared orchestrator.fm_retry window bounds BOTH
            # writes together, task 2706) at the cost of a possibly-single
            # done-write attempt in the worst case; see
            # test_stamp_lands_on_final_attempt_then_done_write_fails_still_escalates
            # for the regression guard covering this exact edge.
            logger.info(
                'DeterministicRunner: task %s before_done deploy verified — '
                'pid=%s unit=%s — setting done',
                task_id, pid, target_unit,
            )
            try:
                await self.scheduler.set_task_status(
                    task_id,
                    'done',
                    done_provenance=_build_done_provenance(
                        'deterministic-deploy',
                        pid=pid,
                        active_enter_timestamp=active_enter_timestamp,
                        unit=target_unit,
                    ),
                )
                return WorkflowOutcome.DONE
            except Exception as exc:
                # set_task_status already exhausted its own transient retry
                # budget (the shared orchestrator.fm_retry schedule, task
                # 2706) before raising — the connection may still be
                # severed.  Retry the done-write within the SAME writeback
                # budget rather than letting this propagate out of run()
                # (stamped stays True — no re-stamp).
                logger.warning(
                    'DeterministicRunner: task %s done-write failed (attempt %d/%d): '
                    '%s: %s — connection may still be severed; retrying',
                    task_id, attempt + 1, attempts,
                    type(exc).__name__, exc,
                )
                if attempt < len(backoffs):
                    await self._sleeper(backoffs[attempt])
                continue

        # Budget exhausted — fused-memory never recovered in-window.  File a
        # durable, connection-independent local escalation (EscalationQueue.submit
        # writes to disk) so the task is never silently stranded — the exact
        # EVIDENCE failure on task 2059 (empty queue + blocked forever).
        # before_done_ran_at is already stamped (I1 — the deploy is NOT re-run
        # here).  This files an L2 escalation and blocks the task; the
        # quiescence guard (section 1, before_done_ran_at+pending sub-case)
        # prevents re-dispatch while that escalation stays open, so resume is
        # NOT automatic while blocked.  If the verified stamp landed before
        # the budget ran out, a human resolving this escalation — and the
        # task then being re-dispatched — is a second safety net: crash-resume
        # sub-case (a) will drive it to done at that point, without re-running
        # the deploy.
        detail = '\n'.join([
            description,
            f'Target unit: {target_unit}',
            'The cross-unit deploy ran and verified successfully, but the '
            "orchestrator's own fused-memory/MCP connection was severed by "
            'the deploy (e.g. the deploy restarted the service backing that '
            'connection) and the verify+writeback could not be persisted '
            f'within the reconnect budget ({attempts} '
            'attempts). before_done_ran_at is already stamped (I1 — the '
            'deploy is NOT re-run). Resolve this escalation to unblock the '
            'task: if the verified stamp landed before the connection was '
            'lost, resolving it and letting the task re-dispatch will then '
            'resume it to done automatically (no re-run); otherwise inspect '
            f'{target_unit} and the task metadata to determine whether the '
            'deploy needs to be retried manually.',
        ])
        return await self._file_infra_issue_and_block(
            task_id,
            summary=f'Deploy verify/writeback stranded (connection severed): {target_unit}',
            detail=detail,
            metadata=metadata,
        )

    async def _file_milestone_gate_and_block(
        self, task_id: str, task: dict, metadata: dict
    ) -> WorkflowOutcome:
        """File the gate's born-at-L2 escalation and block the task.

        The record comes from :func:`build_gate_escalation_fields`:
        ``milestone_gate``, or ``review_due`` for a ``trigger_chain`` human gate.

        Encapsulates section 3's gate-filing logic for reuse by:
        - The pure-gate path (section 3) in ``run()``.
        - The (b-self) crash-resume path when ``always_escalates=True``
          (crash between ``before_done_scheduled_at`` stamp and gate filing).

        Includes a dedup guard — if a pending escalation already exists
        (e.g. prior crash-safe re-dispatch), filing is skipped to avoid
        duplicate L2 escalations.  Stamps ``gate_escalated_at`` so the
        next resume routes through section-1 quiescence.

        Task 4048 (recovered task-2240 review suggestion): on the DEPLOY
        path (``before_done`` set), the ``gate_escalated_at`` stamp normally
        rides along with a ``deploy_state.phase`` advance to ``ESCALATED``
        (DS-1, ζ) — but that advance is SKIPPED when the CURRENT phase is
        already ``ESCALATED`` or ``DONE``, e.g. the rare crash-resume edge
        where a prior ``_file_infra_issue_and_block`` advanced phase to
        ``ESCALATED`` without stamping ``gate_escalated_at`` and a later
        ``always_escalates=True`` dispatch reaches this gate again. Both
        ``ESCALATED->ESCALATED`` and ``DONE->ESCALATED`` are pinned-illegal
        edges, so a bare re-advance would file a spurious born-at-L2
        ``illegal_deploy_transition`` escalation on top of the
        ``milestone_gate`` one just filed above. Unlike
        ``_file_infra_issue_and_block``'s equivalent guard, the skip here
        does NOT drop the write entirely — ``gate_escalated_at`` is still
        stamped (stamp-only, no ``deploy_state`` payload), because that
        stamp is what routes the next dispatch through section-1
        quiescence/resolve-to-done; dropping it would permanently deny the
        task that fork rather than merely delay it. This mirrors the
        stamp-only fallback precedent in
        ``deterministic_runner.py::DeterministicRunner._writeback_deploy_success``.

        Consequence of that deploy-path write failing (task 4048, mirroring
        ``_file_infra_issue_and_block``'s equivalent paragraph): both arms
        above (the stamp-only skip and the legal-edge full advance) are
        best-effort — a transient failure (e.g. the same severed connection
        that might be the reason this gate is being filed at all) is caught
        and logged rather than propagated. The ``milestone_gate`` escalation
        filed above is already durable on local disk regardless, so a failed
        write here simply means ``gate_escalated_at`` stays unset — which is
        exactly the crash-safe file-before-stamp ordering already documented
        above: the next dispatch finds no stamp, re-files the gate (the
        dedup guard skips re-submitting since the escalation is still
        pending) and retries the same write, rather than the exception
        escaping ``run()`` and leaving the task neither done nor cleanly
        blocked. Swallowing the exception here converts an escaped raise
        into this already-designed recovery rather than adding a new one.

        Returns:
            ``WorkflowOutcome.BLOCKED``
        """
        # One instant: the review_due payload's fired_at and the
        # gate_escalated_at stamp, which strand recovery re-files with.
        now_iso = self._clock().isoformat()

        # File the born-at-L2 escalation FIRST (crash-safe ordering: a stamp
        # failure on the following update_task re-files the gate on next dispatch
        # rather than silently skipping it).
        existing_pending = self.escalation_queue.get_by_task(
            task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
        )
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending escalation(s) — '
                'skipping re-file (gate_escalated_at stamp must have failed on prior dispatch)',
                task_id, len(existing_pending),
            )
        else:
            # The shared builder, so strand recovery re-files the same gate.
            human_gate = await read_human_gate_facts(self.scheduler, task, metadata)
            esc_id = self.escalation_queue.make_id(task_id)
            fields = build_gate_escalation_fields(
                task, metadata, human_gate, escalation_id=esc_id, fired_at=now_iso,
            )
            esc = Escalation(
                id=esc_id,
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category=fields.category,
                summary=fields.summary,
                detail=fields.detail,
                options=list(fields.options),
                level=2,
                review_due=fields.review_due,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 %s gate escalation %s for task %s',
                fields.category, esc.id, task_id,
            )

        # Stamp gate_escalated_at AFTER successful escalation submit.  ζ: on the
        # DEPLOY path (before_done set — this helper is also reached from the
        # self-restart and cross-unit act-then-ask fallthroughs, and the
        # self-restart crash-resume re-file), this atomically advances
        # deploy_state.phase to ESCALATED too (DS-1). A pure gate
        # (before_done=None) is not a deploy and gets no deploy_state.
        if metadata.get('before_done') is not None:
            # Task 4048 amendment: pre-bind so both names are defined on every
            # path into `except` below, including one raised by the
            # DeployState.from_metadata read itself (see next comment).
            _already_at_target = False
            try:
                # Task 4048 amendment: the read lives inside this try (not
                # before it) so a corrupted/hand-edited deploy_state slice
                # (pydantic ValidationError from DeployState.from_metadata)
                # is caught by the except below instead of escaping — the
                # write is simply skipped entirely in that case.
                _current_deploy_state = DeployState.from_metadata(metadata)
                _already_at_target = (
                    _current_deploy_state is not None
                    and _current_deploy_state.phase in (DeployPhase.ESCALATED, DeployPhase.DONE)
                )
                if _already_at_target:
                    assert _current_deploy_state is not None  # implied by _already_at_target (short-circuit `and` above)
                    # Task 4048: stamp-only skip of a redundant ESCALATED
                    # advance — see docstring above for the full rationale.
                    logger.debug(
                        'DeterministicRunner: task %s deploy_state already at '
                        'phase=%s — skipping redundant ESCALATED advance, '
                        'stamping gate_escalated_at only',
                        task_id, _current_deploy_state.phase,
                    )
                    await self.scheduler.update_task(
                        task_id, {'gate_escalated_at': now_iso}, metadata_mode='merge',
                    )
                else:
                    await self._advance_deploy_phase(
                        task_id, metadata, DeployPhase.ESCALATED,
                        evidence={'gate_escalated_at': now_iso},
                        phase_timestamp=now_iso,
                    )
            except Exception as exc:
                if _already_at_target:
                    logger.warning(
                        'DeterministicRunner: task %s milestone_gate '
                        'gate_escalated_at stamp-only write failed (deploy_state '
                        'already at target phase) (%s: %s) — the milestone_gate '
                        'escalation above is already durable regardless',
                        task_id, type(exc).__name__, exc,
                    )
                else:
                    logger.warning(
                        'DeterministicRunner: task %s milestone_gate deploy_state '
                        'ESCALATED advance failed (%s: %s) — the milestone_gate '
                        'escalation above is already durable regardless',
                        task_id, type(exc).__name__, exc,
                    )
        else:
            await self.scheduler.update_task(
                task_id,
                {'gate_escalated_at': now_iso, **gate_preconditions_stamp(task)},
                metadata_mode='merge',
            )

        # Set status to blocked — gate awaits human decision.  Best-effort, same
        # as _file_infra_issue_and_block (reviewer amendment): the escalation +
        # gate_escalated_at stamp above are already durable, so a severed
        # connection on this trailing write must not propagate and mask them
        # behind a raw exception.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info(
                'DeterministicRunner: task %s blocked at deterministic gate', task_id,
            )
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the milestone_gate '
                'escalation is already durable on local disk, so returning '
                'BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )

        return WorkflowOutcome.BLOCKED

    async def _file_predicate_escalation_and_block(
        self,
        task_id: str,
        *,
        category: str,
        summary: str,
        detail: str,
        stamps: dict,
        now: datetime,
    ) -> WorkflowOutcome:
        """File a born-at-L2 predicate VERDICT escalation and block (γ-predicate).

        Serves every verdict a human must weigh: ``milestone_check_failed``
        ("the invariant does not hold") and ``milestone_check_stalled`` (a
        re-arming predicate still "not yet" at its stall cap).  A verdict is
        semantically distinct from an infra fault, so this mirrors
        ``_file_infra_issue_and_block``'s dedup/file/block shape but
        additionally stamps ``gate_escalated_at`` (mirroring
        ``_file_milestone_gate_and_block``) so the next dispatch routes through
        section-1's existing quiescence/resolve-to-done fork — a human
        resolving the escalation re-runs the check with no extra wiring.

        *stamps* (``predicate_verdict`` / ``predicate_recheck``, PRD
        plans/completion-driven-triggers-prd.md C1) ride in the SAME single
        ``update_task`` as ``gate_escalated_at``, so a verdict exit is still
        exactly one metadata write.  ``gate_escalated_at`` is *now*, the
        instant the check ran, so it agrees with the streak record's
        ``last_checked_at``.

        Includes a dedup guard — if a pending escalation already exists (e.g.
        prior crash-safe re-dispatch), filing is skipped to avoid duplicate L2
        escalations.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        file_deterministic_escalation(
            self.escalation_queue, task_id,
            actor=_RUNNER_ACTOR, summary=summary, detail=detail, category=category,
        )

        # Stamp gate_escalated_at AFTER successful escalation submit (file-before-stamp
        # ordering, same as _file_milestone_gate_and_block) so a stamp failure re-files
        # on the next dispatch rather than silently skipping it.  A predicate NEVER
        # stamps before_done_ran_at — there is no I1 side effect to guard here.
        await self.scheduler.update_task(
            task_id,
            {'gate_escalated_at': now.isoformat(), **stamps},
            metadata_mode='merge',
        )

        # Best-effort, same as _file_infra_issue_and_block (reviewer amendment):
        # the escalation + gate_escalated_at stamp above are already durable, so
        # a severed connection on this trailing write must not propagate and
        # mask them behind a raw exception.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info('DeterministicRunner: task %s blocked — %s', task_id, category)
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the %s escalation '
                'is already durable on local disk, so returning BLOCKED regardless',
                task_id, type(exc).__name__, exc, category,
            )

        return WorkflowOutcome.BLOCKED

    async def _run_predicate(
        self, task_id: str, before_done: dict, description: str, metadata: dict,
    ) -> WorkflowOutcome:
        """Run a read-only predicate check and map its exit code to a verdict (γ-predicate).

        Unlike the ``before_done`` deploy path, a predicate is a READ-ONLY
        exit-code verdict check — NOT a systemd deploy.  There is no unit to
        inspect, no baseline/fresh-PID verify, and no ``before_done_ran_at``
        I1 stamp (re-running a read-only check on crash-resume is harmless;
        see docs/deterministic-runner.md's γ-predicate subsection).

        Called from two sites: the first dispatch (section 2), and section 1's
        resume branch once a ``milestone_check_failed`` escalation is
        resolved — resume re-invokes this method to re-verify the invariant
        rather than trusting the resolution blindly (a human may resolve the
        escalation without the underlying check now passing).

        ``metadata.always_escalates`` is never consulted here — see
        docs/deterministic-runner.md's γ-predicate subsection.

        Maps the outcome:
        - ``rc == 0`` -> ``WorkflowOutcome.DONE``, with
          ``done_provenance.kind='deterministic-milestone'`` and a BOUNDED,
          STRUCTURED ``note`` from ``summarize_predicate_output`` — never the
          raw stdout tail (task 3286, specimen 2902).  The note is not a
          private field: fused-memory's ``_format_outcome_echo`` appends it to
          a Mem0 completion-summary write, so raw subprocess output landing
          there is ingested into memory.  The full raw output is logged at
          INFO immediately before summarizing, so nothing is silently lost.
        - ``rc == 75`` (``os.EX_TEMPFAIL``) on a gate declaring
          ``before_done.recheck_secs`` with a dated milestone -> "not yet"
          (PRD ``plans/completion-driven-triggers-prd.md`` C1): ONE merge
          write advances ``milestone.at`` by ``recheck_secs``, keeping the
          link's scheduled fire time (``shared.milestone_rearm``), and
          stamps the streak record and verdict, then a best-effort ``pending`` ->
          ``WorkflowOutcome.REQUEUED``.  A refused write files the
          no-verdict category and blocks instead.  The 75 that brings the
          streak to ``deterministic.predicate_recheck_stall_cap`` does not
          re-arm: it files born-at-L2 ``milestone_check_stalled`` (with the
          ``gate_escalated_at`` stamp, like a failure) and blocks.
        - ``rc != 0`` -> a milestone VERDICT failure ("the invariant does not
          hold"): born-at-L2 ``milestone_check_failed`` escalation +
          ``gate_escalated_at`` stamp + blocked (routes through section-1's
          resume/quiescence machinery on the next dispatch).  Any non-zero rc
          returned by the seam is a verdict — including one from an injected
          ``script_runner``, whatever its tail text says.  An UNDECLARED 75
          lands here, its detail naming what the declaration lacks.
        - Every verdict exit (0, 75, other) stamps ``predicate_verdict`` by
          ``orchestrator.predicate_recheck.verdict_stamps``, and every exit
          but a re-arm resets a ``predicate_recheck`` streak.
        - ``ScriptTimeout`` (task 4065) -> an INFRA fault, not a verdict: the
          DEFAULT runner's own inner per-script timeout fired, so no exit code
          exists.  born-at-L2 escalation + blocked, with NO
          ``gate_escalated_at`` stamp, so the read-only check is re-attempted
          on the next dispatch instead of being latched into the
          resolve-to-done path.  See the ``ScriptTimeout`` docstring.  This
          arm's DETAIL is the one piece of wording the carrier split changes
          (next bullet but one): its non-carrier text asserts "deliberately not
          milestone_check_failed", which would contradict a carrier's own
          category.
        - Outer-guard timeout /
          ``shared.before_done_paths.FactoryRootUnresolved`` (task 6386) /
          unexpected error -> likewise an INFRA fault (no verdict was
          produced): born-at-L2 escalation + blocked (re-attempted on the next
          dispatch, no ``gate_escalated_at`` stamp).  The outer
          ``asyncio.wait_for`` guard (``timeout_secs + run_timeout_grace_secs``)
          stays the backstop for a seam that never returns at all.  Every infra
          arm shares one category, so each carries deliberately distinct
          summary/detail wording — that text is the only thing telling a human
          which guard fired.
        - Any other ``shared.before_done_paths.BeforeDonePathsRefused`` (a
          record that arrived through ``update_task`` with a non-string cwd or
          a ``script_root`` it may not use, task 9116) -> likewise an INFRA
          fault: the check was never run.  Summary "Predicate before_done
          refused (check not run)"; the detail names the refused field.
        - CARRIER SPLIT (task 4678 / r3, PRD R-D6 + C-5): for a task carrying
          ``metadata.recurrence`` — one link of a recurring chain — EVERY
          no-verdict arm above files ``milestone_check_failed`` instead of
          ``infra_issue`` (one ``no_verdict_category``, resolved once from
          ``_is_recurrence_carrier``), so a recurring job's failures never
          disappear into the crowded ``infra_issue`` bucket.  Every OTHER
          predicate — and every deploy path — is unchanged.  Two things do NOT
          move with the label: the STAMP (still no ``gate_escalated_at``,
          because these are still no-verdict legs) and the SUMMARIES
          (byte-identical, so the arm discriminator keeps working for carriers
          too).  The single wording change is the ``ScriptTimeout`` arm's
          detail sentence, which on the carrier path names R-D6 as the reason
          for the category instead of asserting the opposite of it.

        Returns:
            WorkflowOutcome.DONE, WorkflowOutcome.BLOCKED, or
            WorkflowOutcome.REQUEUED after a re-arm.
        """
        run_fn = self._script_runner or self._default_run_script
        outer_timeout = before_done.get('timeout_secs', 60) + self._run_timeout_grace_secs

        # Task 4678 (r3): resolved ONCE, so the no-verdict arms below
        # cannot drift into different categories for the same task.
        no_verdict_category = (
            MILESTONE_CHECK_FAILED_CATEGORY
            if _is_recurrence_carrier(metadata)
            else 'infra_issue'
        )

        async def _invoke_run_fn():
            # See run()'s identical inner wrapper: translate a seam-internal
            # TimeoutError into a distinct exception type here so `except
            # TimeoutError` below can only ever mean "the outer wall-clock
            # guard itself fired" — never a misattributed application error.
            #
            # Task 4065: do NOT broaden this except clause — ScriptTimeout must
            # propagate UNTOUCHED to the dedicated arm below (see ScriptTimeout).
            try:
                return await run_fn(before_done)
            except TimeoutError as exc:
                raise RuntimeError(
                    f'run_fn raised TimeoutError internally (not the '
                    f'outer guard): {exc!r}'
                ) from exc

        try:
            rc, out = await asyncio.wait_for(_invoke_run_fn(), timeout=outer_timeout)
        except TimeoutError:
            # A hung/unresponsive seam produced NO exit code, so there is no
            # verdict to report — this is an INFRA fault (parity with the
            # deploy path's outer-guard handling), not milestone_check_failed.
            # infra_issue does NOT stamp gate_escalated_at, so the check is
            # re-attempted on the next dispatch rather than latched into the
            # resolve-to-done path.
            timeout_detail = '\n'.join([
                description,
                f'Predicate check run_fn exceeded the outer guard timeout ({outer_timeout}s = '
                f"before_done['timeout_secs'] + run_timeout_grace_secs).",
                'The subprocess may be detached/unkillable — check out-of-band '
                '(e.g. ps) before taking further action.',
                'This is a read-only predicate — safe to re-run on the next '
                'dispatch (no before_done_ran_at stamp to worry about).',
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate check timed out (subprocess hung)',
                detail=timeout_detail,
                category=no_verdict_category,
            )
        except ScriptTimeout as exc:
            # Task 4065: the DEFAULT runner's own per-script timeout fired — no
            # exit code, so no verdict (see ScriptTimeout).  Wording is
            # deliberately distinct from the sibling arms: they all file
            # the SAME category as each other (`no_verdict_category`), so the
            # text is the only thing telling a human WHICH guard fired — which
            # is why task 4678's carrier split moves the category and leaves
            # every summary byte-identical.
            # Task 4678: the non-carrier sentence below asserts a CATEGORY
            # ("deliberately not milestone_check_failed"), so for a carrier —
            # which IS filed under that category — it would tell a human
            # reading the escalation the opposite of what the record says.
            # Both spellings carry the same two FACTS (no exit code -> no
            # verdict; no gate_escalated_at stamp -> simply re-attempted);
            # only the category claim differs.
            # Task 4252: the opening clause asserted "No exit code was
            # produced" unconditionally — an affirmative falsehood in the same
            # shape the deploy arms had, whenever the script DID exit and only
            # its surviving child held the pipe open.  The conclusion is
            # unchanged either way (that code never reached the classifier, so
            # there is still no verdict); only the claim about what happened
            # is now measured.  The exit_code-None spelling is byte-identical
            # to before, which its pins require.
            if exc.exit_code is None:
                _no_verdict_opening = 'No exit code was produced, so there is NO verdict'
            else:
                _no_verdict_opening = (
                    f'The script itself exited with code {exc.exit_code}, but that '
                    f'code never reached the classifier, so there is still NO verdict'
                )
            if no_verdict_category == MILESTONE_CHECK_FAILED_CATEGORY:
                _no_verdict_sentence = (
                    f'{_no_verdict_opening} and NO '
                    'gate_escalated_at stamp is written — this read-only check is '
                    'simply re-attempted on the next dispatch rather than latched '
                    'into the resolve-to-done path. It is filed under '
                    'milestone_check_failed because this task carries a '
                    'metadata.recurrence chain link: PRD '
                    'docs/prds/recurring-deterministic-tasks.md R-D6 / C-5 puts '
                    'EVERY failure leg of a recurring chain in one deny-listed, '
                    'discriminable category rather than in the crowded '
                    'infra_issue bucket.'
                )
            else:
                _no_verdict_sentence = (
                    f'{_no_verdict_opening} — this is an '
                    'INFRA fault, deliberately not milestone_check_failed ("the '
                    'invariant does not hold").\n'
                    'No gate_escalated_at stamp is written: this read-only check is '
                    'simply re-attempted on the next dispatch rather than latched '
                    'into the resolve-to-done path.'
                )
            inner_timeout_detail = '\n'.join([
                description,
                exc.budget_line('Predicate check script'),
                _no_verdict_sentence,
                "Either the check is genuinely too slow for its configured "
                "before_done['timeout_secs'] budget (raise it), or whatever it "
                'probes is itself wedged.',
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate check script timed out (no verdict produced)',
                detail=inner_timeout_detail,
                category=no_verdict_category,
            )
        except FactoryRootUnresolved as exc:
            # A configuration fault, not a verdict: the script was never located.
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate script root unresolvable',
                detail='\n'.join([
                    description,
                    str(exc),
                    f'Set {DARK_FACTORY_ROOT_ENV} for the orchestrator process to the '
                    'dark-factory checkout.',
                ]),
                category=no_verdict_category,
            )
        except BeforeDonePathsRefused as exc:
            # The record names a script or cwd this process cannot resolve, so
            # the check never ran. After the FactoryRootUnresolved arm: that
            # subclass has its own.
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate before_done refused (check not run)',
                detail='\n'.join([description, str(exc)]),
                category=no_verdict_category,
            )
        except Exception as exc:
            # Likewise an infra fault, not a verdict — an unexpected error
            # means the check never ran to completion.
            error_detail = '\n'.join([
                description,
                f'Predicate check run_fn raised an unexpected error: {exc!r}',
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate check run_fn failed (unexpected error)',
                detail=error_detail,
                category=no_verdict_category,
            )

        verdict = predicate_verdict(out)
        prev_recheck = PredicateRecheck.from_metadata(metadata)
        now = self._clock()
        declaration_gap: str | None = None
        if rc == os.EX_TEMPFAIL:
            declaration = recheck_declaration(before_done, metadata.get('milestone'))
            if declaration.recheck_secs is not None:
                return await self._rearm_or_stall(
                    task_id,
                    description=description,
                    metadata=metadata,
                    out=out,
                    verdict=verdict,
                    prev=prev_recheck,
                    now=now,
                    recheck_secs=declaration.recheck_secs,
                    no_verdict_category=no_verdict_category,
                )
            declaration_gap = declaration.gap
        settled_stamps = verdict_stamps(
            metadata, verdict, reset(prev_recheck, now) if prev_recheck else None,
        )

        if rc != 0:
            # A non-zero exit is a milestone VERDICT ("invariant does not
            # hold"), not an infra fault — file milestone_check_failed (NOT
            # infra_issue) and stamp gate_escalated_at so a human resolving
            # the escalation drives the task to done on the next dispatch.
            #
            # Task 4065: the DEFAULT runner's inner per-script timeout can no
            # longer reach this branch (it raises ScriptTimeout, handled above
            # as an infra fault).  A non-zero rc from an INJECTED script_runner
            # IS still a verdict by contract, whatever its tail says —
            # classification keys on the exception type, never on the text.
            #
            # The RAW `out` below is deliberate and must stay raw (task 3286):
            # unlike done_provenance.note, an escalation detail is read by a
            # human diagnosing the failing check and is never ingested into
            # memory, so full subprocess output is exactly what is wanted here.
            fail_detail = '\n'.join([
                description,
                f'Predicate check exit code: rc={rc}',
                *([declaration_gap] if declaration_gap else []),
                f'Output:\n{out}',
            ])
            return await self._file_predicate_escalation_and_block(
                task_id,
                category=MILESTONE_CHECK_FAILED_CATEGORY,
                summary=f'Milestone predicate check failed (rc={rc})',
                detail=fail_detail,
                stamps=settled_stamps,
                now=now,
            )

        # Log the RAW output before summarizing it.  `summarize_predicate_output`
        # deliberately discards everything it does not recognize, and dropping
        # data with no trace would be a silent degradation — so the full text
        # lands here first, in the orchestrator log, which (unlike the note) is
        # never ingested into Mem0/Graphiti.
        logger.info(
            'DeterministicRunner: task %s predicate raw output (%d chars, '
            'summarized into done_provenance.note): %s',
            task_id, len(out) if isinstance(out, str) else 0, out,
        )
        logger.info(
            'DeterministicRunner: task %s predicate check passed (rc=0) — setting done',
            task_id,
        )
        # Written before done, so no metadata write lands on a terminal row.  An
        # observability record, never a gate: a refused write still drives to done.
        if settled_stamps and not await self.scheduler.update_task(
            task_id, settled_stamps, metadata_mode='merge',
        ):
            logger.warning(
                'DeterministicRunner: task %s predicate verdict stamp write refused '
                '(keys %s) — driving to done regardless',
                task_id, sorted(settled_stamps),
            )
        await self.scheduler.set_task_status(
            task_id,
            'done',
            done_provenance=_build_done_provenance(
                'deterministic-milestone',
                note=summarize_predicate_output(out, rc=rc),
            ),
        )
        return WorkflowOutcome.DONE

    async def _rearm_or_stall(
        self,
        task_id: str,
        *,
        description: str,
        metadata: dict,
        out: object,
        verdict: dict | None,
        prev: PredicateRecheck | None,
        now: datetime,
        recheck_secs: int,
        no_verdict_category: str,
    ) -> WorkflowOutcome:
        """Answer a declared exit-75 "not yet": re-arm, or stall at the cap (PRD C1)."""
        logger.info(
            'DeterministicRunner: task %s predicate raw output (%d chars, '
            'summarized into predicate_verdict): %s',
            task_id, len(out) if isinstance(out, str) else 0, out,
        )
        record = rearmed(prev, now, recheck_secs)
        if record.consecutive >= self._predicate_recheck_stall_cap:
            return await self._stall_predicate(
                task_id,
                description=description,
                metadata=metadata,
                verdict=verdict,
                prev=prev,
                now=now,
                streak=record.consecutive,
                recheck_secs=recheck_secs,
            )
        return await self._rearm_predicate(
            task_id,
            metadata=metadata,
            verdict=verdict,
            record=record,
            recheck_secs=recheck_secs,
            no_verdict_category=no_verdict_category,
        )

    async def _rearm_predicate(
        self,
        task_id: str,
        *,
        metadata: dict,
        verdict: dict | None,
        record: RearmedRecheck,
        recheck_secs: int,
        no_verdict_category: str,
    ) -> WorkflowOutcome:
        """Advance ``milestone.at`` to *record*'s next check, keep the fire time, re-pend.

        ONE merge write carries the advanced milestone, the streak record and
        the verdict.  It comes first, so a lost trailing status write leaves a
        task whose advanced ``at`` already withholds re-dispatch.  A REFUSED
        write is escalated instead: with ``at`` not advanced, re-pending would
        re-dispatch at once into the same 75, a loop with no escape.  So is a
        milestone with no readable scheduled fire time, which
        ``rearmed_milestone`` refuses before anything is written.
        """
        next_check_at = record.next_check_at.isoformat()
        try:
            milestone = rearmed_milestone(metadata['milestone'], next_check_at)
        except ValueError as exc:
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate re-arm refused (rc=75)',
                detail='\n'.join([
                    f'The predicate answered "not yet" (exit 75) and should have re-armed '
                    f'to next_check_at={next_check_at}, but the milestone was refused: {exc}',
                    'Blocked rather than re-pended: nothing was written, so a re-dispatch '
                    'would re-run the check at once into the same verdict.',
                    'Repair milestone.scheduled_fire_at (or remove it so milestone.at is '
                    'the fire time), then resume to re-run the check.',
                ]),
                category=no_verdict_category,
            )
        stamps = {
            'milestone': milestone,
            **verdict_stamps(metadata, verdict, record, always_stamp_verdict=True),
        }
        if not await self.scheduler.update_task(task_id, stamps, metadata_mode='merge'):
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate re-arm write failed (rc=75)',
                detail='\n'.join([
                    f'The predicate answered "not yet" (exit 75) and should have re-armed '
                    f'to next_check_at={next_check_at}, but the metadata write advancing '
                    'milestone.at was refused, so the gate was not re-armed.',
                    'Blocked rather than re-pended: with milestone.at unchanged, a '
                    're-dispatch would re-run the check at once into the same verdict.',
                    'Resume to re-run the check.',
                ]),
                category=no_verdict_category,
            )
        self._emit_predicate_event(
            EventType.predicate_rearmed, task_id,
            consecutive=record.consecutive, next_check_at=record.next_check_at,
            recheck_secs=recheck_secs, verdict=verdict,
        )
        logger.info(
            'DeterministicRunner: task %s predicate re-armed (rc=75, consecutive=%d/%d, '
            'next_check_at=%s): %s',
            task_id, record.consecutive, self._predicate_recheck_stall_cap, next_check_at,
            render_verdict(verdict),
        )
        try:
            await self.scheduler.set_task_status(task_id, 'pending')
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s pending-status writeback failed after the '
                're-arm (%s: %s) — milestone.at is already advanced, so the gate stays '
                'withheld until next_check_at regardless',
                task_id, type(exc).__name__, exc,
            )
        return WorkflowOutcome.REQUEUED

    async def _stall_predicate(
        self,
        task_id: str,
        *,
        description: str,
        metadata: dict,
        verdict: dict | None,
        prev: PredicateRecheck | None,
        now: datetime,
        streak: int,
        recheck_secs: int,
    ) -> WorkflowOutcome:
        """File ``milestone_check_stalled`` instead of re-arming: the streak hit the cap.

        ``milestone.at`` is left where it is and the streak resets, keeping
        ``first_rearmed_at``.  The filer's ``gate_escalated_at`` stamp routes
        the next dispatch through section 1: quiescent while the escalation is
        pending, a re-run check (and a fresh streak) once it is resolved.
        """
        cap = self._predicate_recheck_stall_cap
        stored = reset(prev, now)
        outcome = await self._file_predicate_escalation_and_block(
            task_id,
            category=MILESTONE_CHECK_STALLED_CATEGORY,
            summary=f'Milestone predicate still not satisfied after {cap} consecutive checks',
            detail=stall_detail(
                description,
                streak=streak,
                stall_cap=cap,
                first_rearmed_at=stored.first_rearmed_at,
                now=now,
                recheck_secs=recheck_secs,
                verdict=verdict,
            ),
            stamps=verdict_stamps(metadata, verdict, stored, always_stamp_verdict=True),
            now=now,
        )
        self._emit_predicate_event(
            EventType.predicate_recheck_stalled, task_id,
            consecutive=streak, next_check_at=None,
            recheck_secs=recheck_secs, verdict=verdict,
        )
        logger.warning(
            'DeterministicRunner: task %s predicate stalled (rc=75, consecutive=%d/%d) — '
            'filed %s instead of re-arming: %s',
            task_id, streak, cap, MILESTONE_CHECK_STALLED_CATEGORY, render_verdict(verdict),
        )
        return outcome

    def _emit_predicate_event(
        self,
        event_type: EventType,
        task_id: str,
        *,
        consecutive: int,
        next_check_at: datetime | None,
        recheck_secs: int,
        verdict: dict | None,
    ) -> None:
        event_store = getattr(self.scheduler, 'event_store', None)
        if event_store is None:
            return
        event_store.emit(
            event_type,
            task_id=task_id,
            data={
                'rc': os.EX_TEMPFAIL,
                'consecutive': consecutive,
                'stall_cap': self._predicate_recheck_stall_cap,
                'recheck_secs': recheck_secs,
                'next_check_at': next_check_at.isoformat() if next_check_at else None,
                'verdict': verdict,
            },
        )

    # ------------------------------------------------------------------
    # before_done deploy script execution — shared by run()'s named-target
    # and target_unit-less branches (task 2632 review amendment: these two
    # branches previously each carried their own copy of the run_fn
    # TimeoutError-translation wrapper and outer_timeout computation, which
    # could silently drift apart — e.g. one path's default timeout or
    # exception classification changing without the other).
    # ------------------------------------------------------------------

    def _deploy_outer_timeout(self, before_done: dict) -> float:
        """The Layer-B outer wall-clock guard budget for a ``before_done``
        deploy run (or run+verify, for a named ``target_unit``):
        ``before_done['timeout_secs']`` plus the runner's configured grace
        period.  Shared by both of ``run()``'s ``before_done`` deploy
        branches so a future default-timeout change can't apply to one
        path and not the other."""
        return before_done.get('timeout_secs', 60) + self._run_timeout_grace_secs

    async def _invoke_run_fn_translating_timeout(self, run_fn, before_done: dict):
        """Await ``run_fn(before_done)``, translating a seam-internal
        ``TimeoutError`` into ``RuntimeError`` first.

        A custom/injected ``run_fn`` seam could itself raise a
        ``TimeoutError`` internally (e.g. its own inner
        ``asyncio.wait_for``) BEFORE an outer
        ``asyncio.wait_for(..., timeout=outer_timeout)`` elapses —
        ``asyncio.wait_for`` cannot tell that apart from its OWN
        outer-guard timeout; both surface as the same ``TimeoutError``
        type at the call site.  Translating here means a caller's
        ``except TimeoutError`` around the outer ``wait_for`` can only
        ever mean "the outer wall-clock guard itself fired" — never a
        misattributed application error.

        Shared by both of ``run()``'s ``before_done`` deploy branches (the
        named-target ``RestartPlan`` shim runner, and the target_unit-less
        direct invocation below) so the translation logic cannot drift
        between the two copies.

        Task 4252: this wrapper must NOT catch the default runner's
        ``ScriptTimeout``.  It used to convert it back into the legacy
        ``(1, '<script timed out after Ns>')`` pair, which handed both deploy
        classifiers an exit code the script never produced.  It now
        propagates untouched (deliberately not a ``TimeoutError`` subclass,
        so the translation below cannot swallow it) to a dedicated arm in
        EACH deploy branch — the arm has to live where ``task_id`` /
        ``description`` / ``metadata`` are in scope to file the escalation,
        which is not here.  The anti-drift property this helper exists for is
        preserved for the timeout case by the shared
        ``_deploy_timeout_fact_lines``; both branches are pinned by
        ``TestDefaultRunnerInnerTimeoutDeployHonesty``.
        """
        try:
            return await run_fn(before_done)
        except TimeoutError as exc:
            raise RuntimeError(
                f'run_fn raised TimeoutError internally (not the '
                f'outer guard): {exc!r}'
            ) from exc

    async def _run_deploy_script_guarded(
        self,
        task_id: str,
        before_done: dict,
        description: str,
        note: str,
        *,
        metadata: dict | None,
    ) -> tuple[int, str] | WorkflowOutcome:
        """Run a ``before_done`` deploy script under the Layer-B outer
        wall-clock guard, mapping an outer-guard timeout / the default
        runner's own inner per-script timeout (``ScriptTimeout``) /
        an unexpected error / a non-zero exit to an already-filed born-at-L2
        ``infra_issue`` + ``BLOCKED``.

        Returns ``(rc, tail)`` ONLY on a successful (``rc == 0``) run — a
        caller never needs to re-check ``rc``.  Any other outcome returns
        ``WorkflowOutcome.BLOCKED`` directly, having already filed the
        escalation; callers distinguish the two with
        ``isinstance(result, WorkflowOutcome)``.

        Used by the target_unit-less branch below, which has no
        baseline/verify machinery to run afterwards — unlike the
        named-target branch, whose single outer guard must instead bound
        ``RestartPlan.execute()``'s COMBINED run+verify (see the comment
        there), so that branch cannot use this helper without splitting
        the run and verify legs onto two separate timeout budgets, a
        behaviour change out of scope for this amendment.
        """
        run_fn = self._script_runner or self._default_run_script
        outer_timeout = self._deploy_outer_timeout(before_done)

        try:
            rc, tail = await asyncio.wait_for(
                self._invoke_run_fn_translating_timeout(run_fn, before_done),
                timeout=outer_timeout,
            )
        except TimeoutError:
            timeout_detail = '\n'.join([
                description,
                note,
                f'Deploy run exceeded the outer guard timeout ({outer_timeout}s = '
                f"before_done['timeout_secs'] + run_timeout_grace_secs).",
                _I1_NOT_RERUN_LINE,
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Deploy run exceeded outer guard (no target_unit)',
                detail=timeout_detail,
                metadata=metadata,
            )
        except ScriptTimeout as exc:
            # The default runner's own per-script guard fired (see the
            # ScriptTimeout docstring for the classification).  MUST precede
            # the catch-all below, which would otherwise mislabel it as an
            # 'unexpected error' — same category, materially worse diagnostics.
            inner_timeout_detail = '\n'.join([
                description,
                note,
                *_deploy_timeout_fact_lines(
                    exc, script=self._before_done_paths(before_done).script,
                ),
                _I1_NOT_RERUN_LINE,
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary=(
                    f'Deploy script timed out '
                    f'({exc.summary_phrase}, no target_unit)'
                ),
                detail=inner_timeout_detail,
                metadata=metadata,
            )
        except Exception as exc:
            error_detail = '\n'.join([
                description,
                note,
                f'Deploy run_fn raised an unexpected error: {exc!r}',
                _I1_NOT_RERUN_LINE,
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Deploy run_fn failed (unexpected error, no target_unit)',
                detail=error_detail,
                metadata=metadata,
            )

        if rc != 0:
            fail_detail = '\n'.join([
                description,
                note,
                f'Deploy script exit code: rc={rc}',
                f'Output:\n{tail}',
                _I1_NOT_RERUN_LINE,
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary=f'Deploy failed (no target_unit) (rc={rc})',
                detail=fail_detail,
                metadata=metadata,
            )

        return rc, tail

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    async def run(self, assignment) -> WorkflowOutcome:
        """Execute the deterministic gate logic for *assignment*.

        Returns:
            WorkflowOutcome.DONE  — gate resolved, task driven to done.
            WorkflowOutcome.BLOCKED — gate filed, open escalation, or deploy failure.

        A unit-inspector failure — a spawn error or a timeout — is NOT raised
        on ANY of the three legs that inspect a unit: the pre-deploy baseline
        capture, the post-deploy verify re-inspect, and the crash-window
        re-verify. Task 4157 degrades it to the MainPID=0 sentinel via
        ``_inspect_unit_guarded``, which routes it into the existing
        fail-closed escalation for that leg ('Baseline inspect failed before
        deploy', 'Deploy verify failed', and the crash-window re-escalation
        respectively) with the real cause named in the detail. This discharges
        the "always returns BLOCKED, never a raw exception" contract stated in
        ``DeterministicRunner._writeback_deploy_success``'s task-2240
        stamp-only fallback on the recovery paths too, and keeps an inspect
        failure from being reported as a deploy-script failure.

        Raises:
            ValueError — if ``always_escalates`` is False with ``before_done=None``
                (unsupported misconfiguration in β).
        """
        task_id = str(assignment.task_id)
        task = assignment.task
        metadata = task.get('metadata') or {}
        description = task.get('description', '')
        before_done = metadata.get('before_done')
        always_escalates = metadata.get('always_escalates', False)
        gate_escalated_at = metadata.get('gate_escalated_at')

        # Task 3341 (reviewer amendment) — a curator gate is a PURE gate by
        # construction: `human_curator_gate` declares "only a human's CONTENT
        # judgement closes this", while a `before_done` action is a machine step
        # that closes it.  The two are contradictory, so the rung-two guard below
        # lives only on the pure-gate resume path.  But the marker is LLM-authored
        # (reconciliation Stage 2), so a misauthored task CAN carry both — and
        # would then take the act-then-ask path with the marker never read.  Say
        # so LOUDLY rather than degrading silently (repo norm), on every dispatch
        # of such a task and before any branch consumes it.
        #
        # Deliberately a WARNING and not a block: the defect is in task
        # AUTHORING, and hard-failing here would strand a deploy that may have
        # no curator semantics at all — trading a silent fail-open for a silent-
        # to-the-author fail-closed.
        #
        # Task 3369 landed the durable fix: `shared.task_metadata` now REJECTS
        # the marker alongside a non-null before_done at write time, so this
        # combination can no longer reach us through the fused-memory
        # submit_task/update_task boundary.  This WARNING is deliberately KEPT
        # at WARNING level as defence-in-depth, because three routes still reach
        # the runner without passing that boundary: `task_metadata.enforce` is a
        # RED-TIER restart-only flag (a restart into warn-mode leaves this as the
        # only remaining signal), records written before the validator existed,
        # and any writer that does not go through SqliteTaskBackend (operator
        # hand-edit, direct sqlite write, a future backend).  Downgrading it
        # would delete the signal in exactly the configurations where it is the
        # last one standing.
        # Pinned by test_curator_marker_on_a_non_pure_gate_is_loud.
        if before_done is not None and _is_human_curator_gate(metadata):
            logger.warning(
                'DeterministicRunner: task %s carries metadata.%s but ALSO a '
                'before_done action. A curator gate is a pure gate by '
                'construction, so the content-adjudication guard (task 3341) '
                'does NOT apply on the act-then-ask path — the marker is being '
                'IGNORED. Fix the task metadata: drop before_done to make this a '
                'real curator gate, or drop the marker if the machine step is '
                'what closes the task.',
                task_id, HUMAN_CURATOR_GATE_KEY,
            )

        # ── 1. Idempotency / quiescence ──────────────────────────────────────
        # If the gate escalation was already filed in a prior dispatch, check
        # whether it is still open or has been resolved.
        if gate_escalated_at:
            pending = self.escalation_queue.get_by_task(
                task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
            )
            if pending:
                # Escalation still open — quiescence (B3): return BLOCKED without
                # re-escalating.  The existing L2 is still awaiting human action.
                logger.debug(
                    'DeterministicRunner: task %s quiescent — %d pending escalation(s)',
                    task_id, len(pending),
                )
                return WorkflowOutcome.BLOCKED
            else:
                # Escalation resolved — drive to done (I2/B4/B11).
                # γ-predicate: a read-only predicate NEVER stamps before_done_ran_at
                # (no I1 side effect to prove), so it must short-circuit here, BEFORE
                # the before_done_ran_at proof-check below — otherwise a resolved
                # milestone_check_failed escalation would wrongly raise
                # NotImplementedError.
                #
                # Unlike a deploy (where the human's out-of-band fix IS the
                # resolution, and re-running risks a double-apply), a predicate
                # is a cheap, idempotent, READ-ONLY exit-code check — resolving
                # the escalation alone is NOT proof the invariant now holds (a
                # human may resolve it prematurely or in error).  So resume
                # RE-RUNS the check via _run_predicate rather than trusting the
                # resolution blindly (reviewer amendment): rc==0 drives to done
                # (deterministic-milestone, never 'deterministic-deploy' — no
                # deploy happened); rc!=0 re-files milestone_check_failed (the
                # dedup guard sees no pending escalation, since it was just
                # resolved) and stays blocked; a timeout/error routes to
                # infra_issue — identical to the first-dispatch path.
                if before_done is not None and before_done.get('kind') == 'predicate':
                    logger.info(
                        'DeterministicRunner: task %s predicate resume — gate '
                        'resolved — re-running predicate check (read-only, '
                        'safe to repeat) before trusting the resolution',
                        task_id,
                    )
                    return await self._run_predicate(task_id, before_done, description, metadata)

                # Task 3341: bound here rather than only inside the pure-gate
                # branch below, so the symmetric `if before_done is not None:`
                # at the done write has a definite assignment to read.  Stays
                # empty on the act-then-ask path, which never consults it.
                own_records: list = []

                # γ: if before_done is set, the action must have already ran (I1) for us
                # to safely drive to done here.  Check before_done_ran_at as proof.
                if before_done is not None:
                    before_done_ran_at_check = metadata.get('before_done_ran_at')
                    if not before_done_ran_at_check:
                        # Gate resolved but before_done never ran — unexpected state.
                        # Conservatively raise so the operator can investigate; this path
                        # should not occur in normal operation once γ ships.
                        raise NotImplementedError(
                            f'DeterministicRunner: gate resolved but before_done_ran_at '
                            f'is not set — cannot safely drive to done without proof the '
                            f'action ran.  Task id={task_id}.'
                        )
                    # before_done already ran (I1) — proceed to done (act-then-ask resume)
                    logger.info(
                        'DeterministicRunner: task %s act-then-ask resume — '
                        'before_done_ran_at=%s, gate resolved — setting done',
                        task_id, before_done_ran_at_check,
                    )
                else:
                    # Pure-gate resume (before_done is None): an empty PENDING
                    # queue is NOT proof a human resolved the gate.  Require the
                    # SAME archive-inclusive positive proof the deploy path
                    # demands below (section 2's own_escalation_resolved) before
                    # driving to done — otherwise a LOST born-at-L2 gate
                    # escalation (task 2954 strand) would let an operator's
                    # re-pend of the stuck task silently BYPASS the human gate.
                    # status=None scans the archive too, so a resolved/dismissed
                    # record counts as proof a human was in the loop; agent_role
                    # scopes it to the runner's OWN gate escalations.  When NO
                    # record ever landed, re-establish the gate via
                    # _file_milestone_gate_and_block and stay BLOCKED rather than
                    # phantom-completing.
                    # Task 3341: keep the RECORDS, not just the boolean — the
                    # curator-gate done write below cites the newest one as
                    # rung-one evidence via done_provenance.escalation_id.  The
                    # `bool(...)` semantics below are byte-identical to task
                    # 2954's original expression.
                    own_records = await asyncio.to_thread(
                        self.escalation_queue.get_by_task,
                        task_id, agent_role=DETERMINISTIC_AGENT_ROLE,
                    )
                    own_escalation_resolved = bool(own_records)
                    if not own_escalation_resolved:
                        logger.warning(
                            'DeterministicRunner: task %s pure-gate resume — '
                            'gate_escalated_at stamped but NO escalation record '
                            'exists (pending or archived); the gate was never '
                            'proven resolved (likely lost across a restart) — '
                            're-filing the born-at-L2 gate instead of driving to '
                            'done (task 2954)',
                            task_id,
                        )
                        return await self._file_milestone_gate_and_block(
                            task_id, task, metadata,
                        )
                    # Re-held behind a new precondition: docs/deterministic-runner.md §3.
                    if gate_held_since_escalation(task):
                        logger.warning(
                            'DeterministicRunner: task %s pure-gate resume — the task '
                            'now waits on a dependency or milestone its gate '
                            'escalation (gate_escalated_at=%s) did not; the '
                            'resolution held the gate, it did not approve it — '
                            're-filing the born-at-L2 gate instead of driving to '
                            'done (task 9520)',
                            task_id, gate_escalated_at,
                        )
                        return await self._file_milestone_gate_and_block(
                            task_id, task, metadata,
                        )
                    # ORDERING INVARIANT (task 3341) — these two branches are
                    # NOT interchangeable, and the curator guard below MUST stay
                    # second:
                    #   no record at all  ⇒ the gate was never established
                    #                     ⇒ re-file the ORIGINAL milestone_gate
                    #                       (above); asking for an adjudication
                    #                       stamp on a gate nobody has yet seen
                    #                       would be incoherent.
                    #   a record exists,
                    #   content unproven  ⇒ ask for the adjudication stamp
                    #                       (below).
                    # Pinned by
                    # test_curator_gate_with_zero_records_still_refiles_milestone_gate.
                    #
                    # Second rung of the pure-gate proof ladder (task 3341): an
                    # existing, no-longer-pending record proves a gate was
                    # established and closed — it does NOT prove a human did the
                    # CONTENT work a curator gate asks for. Task 3181 is where
                    # the two diverged. Fail closed.
                    if (
                        _is_human_curator_gate(metadata)
                        and not _curator_adjudication_confirmed(metadata)
                    ):
                        logger.warning(
                            'DeterministicRunner: task %s pure-gate resume — '
                            'metadata.%s is set but metadata.%s carries no '
                            'non-empty string, so the human CONTENT adjudication '
                            'is unproven; a closed escalation record is not '
                            'evidence the review happened (task 3181 precedent: '
                            'esc-3181-1 was auto-resolved by escalation-watcher '
                            'and the curator work was explicitly not done) — '
                            'filing curator_adjudication_missing and staying '
                            'BLOCKED instead of driving to done',
                            task_id, HUMAN_CURATOR_GATE_KEY,
                            HUMAN_CURATOR_ADJUDICATED_AT_KEY,
                        )
                        return await self._file_curator_adjudication_missing_and_block(
                            task_id, task,
                        )
                logger.info(
                    'DeterministicRunner: task %s gate resolved — setting done',
                    task_id,
                )
                if before_done is not None:
                    # Act-then-ask resume: include deploy provenance so the audit trail
                    # matches the B6 / resume paths and passes require_done_provenance.
                    _ata_unit = before_done.get('target_unit', '')
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-deploy',
                            unit=_ata_unit,
                            note='resumed after gate resolution',
                        ),
                    )
                elif _is_human_curator_gate(metadata):
                    # Curator gate closing legitimately (task 3341).  Reaching
                    # here means BOTH rungs of the proof ladder held: a closed
                    # own-role escalation record exists (rung one, task 2954)
                    # AND `human_curator_adjudicated_at` carries a non-empty string
                    # (rung two, checked above) — otherwise the guard returned
                    # BLOCKED and never got here.
                    #
                    # The note deliberately differs from the generic
                    # 'pure gate resolved' below: that string is what made task
                    # 3181's phantom closure indistinguishable from a genuine
                    # one in the audit trail.  A curator gate that closes must
                    # name the evidence it closed on — the adjudication stamp
                    # here, plus the gate record via `escalation_id`.
                    #
                    # BOUND THE STAMP (task 3286's finding): this note is
                    # memory-ingested — fused-memory's `_format_outcome_echo`
                    # appends it to the "Task '<title>' completed." Mem0 write
                    # under a 500-char cap — and the stamp is an externally-
                    # supplied metadata value that `_curator_adjudication_confirmed`
                    # validates for TYPE only, never length.  So truncate here,
                    # at construction, where over-length degrades the audit
                    # string but never the safety decision.  `summarize_predicate_output`
                    # is deliberately NOT reused: it strips log noise from raw
                    # subprocess output, a different threat model, and borrowing
                    # it would imply a sanitization guarantee this single
                    # structured field does not need.
                    _raw_stamp = str(metadata.get(HUMAN_CURATOR_ADJUDICATED_AT_KEY, ''))
                    _stamp_for_note = (
                        _raw_stamp
                        if len(_raw_stamp) <= _CURATOR_STAMP_NOTE_MAX_CHARS
                        else _raw_stamp[:_CURATOR_STAMP_NOTE_MAX_CHARS] + '…'
                    )
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-gate',
                            note=(
                                'human curator gate: per-entry content '
                                f'adjudication confirmed at {_stamp_for_note}'
                            ),
                            escalation_id=_cite_gate_escalation_id(
                                own_records, gate_escalation_category(metadata),
                            ),
                        ),
                    )
                else:
                    # Pure-gate resume: no before_done action ran, so there is no
                    # deploy/unit/pid evidence — stamp a dedicated gate-kind
                    # provenance so the done write passes require_done_provenance
                    # without lying about a deploy having happened (task 2331).
                    #
                    # The gate record is cited through the SAME rule the curator
                    # arm uses (task 5241): `escalation_id` is now REQUIRED for
                    # this kind server-side, and this arm is reached only with a
                    # non-empty `own_records` — the zero-record case re-files the
                    # gate and returns BLOCKED above, never arriving here.
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-gate',
                            note='pure gate resolved',
                            escalation_id=_cite_gate_escalation_id(
                                own_records, gate_escalation_category(metadata),
                            ),
                        ),
                    )
                return WorkflowOutcome.DONE

        # ── 2. before_done execution (γ) ────────────────────────────────────
        # Cross-unit blocking deploy: stamp → baseline → run script → verify → done.
        # Self-target detection + detached systemd-run is deferred to ε.
        # A falsy target_unit (cross-unit, no named unit) skips the
        # baseline/verify legs entirely and drives on rc alone — see
        # docs/deterministic-runner.md's target_unit-less sub-path and the
        # `if not target_unit:` branch below (task 2632 / esc-2585-1).
        if before_done is not None:
            # γ-predicate: a read-only exit-code verdict check — NOT a deploy.
            # Dispatched FIRST, above target_unit/I1/baseline, so a predicate
            # never touches systemd and never stamps before_done_ran_at.  This
            # is also BEFORE `always_escalates` is read below — a predicate
            # task's always_escalates is intentionally never consulted (see
            # docs/deterministic-runner.md's γ-predicate subsection); a
            # kind='predicate' + always_escalates=True task is not rejected
            # here and simply behaves as a plain predicate.
            if before_done.get('kind') == 'predicate':
                return await self._run_predicate(task_id, before_done, description, metadata)

            target_unit: str = before_done.get('target_unit', '')
            before_done_ran_at = metadata.get('before_done_ran_at')

            # I1 once-only idempotency guard (parallel to β's gate_escalated_at branch):
            # if the deploy already ran, check whether its escalation is still open.
            if before_done_ran_at:
                pending = self.escalation_queue.get_by_task(
                    task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
                )
                if pending:
                    # Pending infra_issue → quiescent BLOCKED (B7 reaper / I1 no-rerun)
                    logger.debug(
                        'DeterministicRunner: task %s before_done already ran, '
                        'pending escalation — quiescent BLOCKED (B7/I1)',
                        task_id,
                    )
                    return WorkflowOutcome.BLOCKED

                # No pending escalation.  Before driving to done we MUST have
                # POSITIVE proof the deploy reached a terminal decision —
                # otherwise a crash in the window between stamping
                # ``before_done_ran_at`` (above) and recording a terminal outcome
                # (verify-success stamp, or the failure escalation) lands us here
                # with neither verification nor a human in the loop, producing a
                # phantom-done.  The bare "stamp set + empty queue → done" rule
                # conflated 'crashed mid-deploy' with 'human resolved escalation'.
                # Three sub-states are distinguished:
                #   (a) before_done_verified_at set → the fresh-PID check passed
                #       (crash between the verify stamp and the done write) →
                #       safe to drive to done.
                #   (b) an escalation was filed for this task and later resolved →
                #       a human acted on the failure (act-then-ask resume) →
                #       drive to done, no re-run (I1).
                #   (c) neither → crash mid-deploy before any terminal decision →
                #       re-escalate.  NEVER phantom-done; NEVER re-run (I1).
                before_done_verified_at = metadata.get('before_done_verified_at')
                # status=None scans the archive too → detects a resolved/dismissed
                # escalation, i.e. proof a human was in the loop on a prior failure.
                # agent_role scopes this to the runner's OWN escalations — an
                # unrelated escalation sharing this task_id (e.g. a starvation-
                # watchdog filing) must never alias as proof a human resolved
                # THIS runner's failure (task 2120).
                own_escalation_resolved = bool(await asyncio.to_thread(
                    self.escalation_queue.get_by_task,
                    task_id, agent_role=DETERMINISTIC_AGENT_ROLE,
                ))
                # ζ D3 (finding 4.0): bare escalation existence is not proof
                # THIS deploy's own gate/failure is what got resolved — even
                # role-scoped, a stale or unrelated same-role record could
                # alias.  When ζ has been tracking this deploy (deploy_state
                # present), require the RECORDED fact that the runner itself
                # transitioned to `escalated` when it filed that failure/gate
                # escalation, in addition to the resolved-record check above.
                # A task with no deploy_state at all began its deploy before ζ
                # activated (no phase was ever recorded) — fall back to the
                # pre-ζ bare-existence check so in-flight legacy deploys don't
                # spuriously re-escalate the moment ζ ships (backward-compat
                # migration shim).
                deploy_state = DeployState.from_metadata(metadata)
                if deploy_state is not None:
                    resolution_proven = (
                        deploy_state.phase == DeployPhase.ESCALATED
                        and own_escalation_resolved
                    )
                else:
                    resolution_proven = own_escalation_resolved

                if before_done_verified_at:
                    # (a) Deploy verified OK; crash before the done write.
                    logger.info(
                        'DeterministicRunner: task %s before_done verified (%s) — '
                        'resume after crash-before-done-write, setting done',
                        task_id, before_done_verified_at,
                    )
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-deploy',
                            pid=metadata.get('before_done_verified_pid'),
                            note='resumed after verified deploy (crash before done write)',
                            unit=target_unit,
                        ),
                    )
                    return WorkflowOutcome.DONE

                before_done_scheduled_at_stamp = metadata.get('before_done_scheduled_at')
                if before_done_scheduled_at_stamp:
                    # (b-self) Self-restart was successfully scheduled
                    # (before_done_scheduled_at set) but the done write or gate
                    # filing did not complete (crash in the window after stamp).
                    _sched = before_done_scheduled_at_stamp if isinstance(before_done_scheduled_at_stamp, dict) else {}
                    logger.info(
                        'DeterministicRunner: task %s self-restart scheduled (%s) + '
                        'crash before done/gate write — resume path '
                        '(transient_unit=%s, always_escalates=%s)',
                        task_id, _sched.get('at', ''), _sched.get('transient_unit', ''),
                        always_escalates,
                    )
                    if not always_escalates:
                        # always_escalates=False: the transient unit is registered and WILL
                        # fire — driving to done here is safe and avoids a spurious
                        # crash-window L2.
                        await self.scheduler.set_task_status(
                            task_id,
                            'done',
                            done_provenance=_build_done_provenance(
                                SCHEDULED_DEPLOY_PROVENANCE_KIND,
                                unit=target_unit,
                                transient_unit=_sched.get('transient_unit', ''),
                                fire_delay_secs=_sched.get('fire_delay_secs', 0),
                                note='resumed after self-restart scheduled (crash before done write)',
                            ),
                        )
                        return WorkflowOutcome.DONE
                    # always_escalates=True (act-then-ask): the milestone gate must NOT
                    # be bypassed.  Re-file the gate and block; gate_escalated_at is
                    # stamped so the next resume routes through section-1 quiescence
                    # rather than entering this (b-self) branch again.
                    logger.info(
                        'DeterministicRunner: task %s scheduled-resume with '
                        'always_escalates=True — re-filing milestone gate (gate not bypassed)',
                        task_id,
                    )
                    return await self._file_milestone_gate_and_block(task_id, task, metadata)

                # (b-self-stale) Task 2983 fix (a): the b-self branch above only
                # fires when before_done_scheduled_at is present in the IN-HAND
                # snapshot.  The reported incident (task 2912 / γ3 self-unit
                # restart) re-selected this deterministic self-deploy off a
                # STALE eligibility snapshot that carried ONLY
                # before_done_ran_at — the scheduled stamp and the done-writeback
                # had not yet landed when the snapshot was read — so b-self was
                # skipped and the second dispatch fell through to the
                # crash-window detector below, filing a born-at-L2 infra_issue
                # false positive.  Re-read the CURRENT task (a fresh get_task,
                # taken at execution time ~seconds later, after the first
                # dispatch's writes landed): if it is an already-completed
                # scheduled self-deploy, the deploy is done (scheduled) — return
                # DONE with NO escalation and NO status write (the task is
                # already terminal; a status write would trip the terminal-exit
                # gate that produced the observed TerminalExitRejection).  A
                # fresh read that is NOT scheduled-complete falls through to the
                # unchanged re-verify/re-escalate handling below, so a genuine
                # crash-window still re-escalates exactly as today.
                current_task = await self.scheduler.get_task(task_id)
                if _is_scheduled_self_deploy_complete(current_task):
                    # Amendment (reviewer_comprehensive): the scheduled stamp is
                    # written on BOTH the always_escalates=False path (task set
                    # 'done' via 'deterministic-deploy-scheduled') AND the
                    # act-then-ask always_escalates=True path (b-self above,
                    # which RE-FILES the milestone gate and BLOCKS).  Mirror
                    # b-self's policy split here so the DONE short-circuit
                    # applies ONLY to the always_escalates=False shape: for an
                    # always_escalates=True self-deploy whose fresh get_task now
                    # carries before_done_scheduled_at, re-file the gate and
                    # block instead of silently bypassing the still-open
                    # act-then-ask gate with a phantom-done.
                    if always_escalates:
                        logger.info(
                            'DeterministicRunner: task %s stale-snapshot '
                            'double-dispatch of a scheduled self-deploy with '
                            'always_escalates=True — re-filing milestone gate '
                            '(gate not bypassed, task 2983)',
                            task_id,
                        )
                        return await self._file_milestone_gate_and_block(
                            task_id, task, metadata,
                        )
                    logger.info(
                        'DeterministicRunner: task %s double-dispatch of a '
                        'completed scheduled self-deploy detected via fresh '
                        'get_task — returning DONE with no escalation and no '
                        'status write (crash-window false positive avoided, '
                        'task 2983)',
                        task_id,
                    )
                    return WorkflowOutcome.DONE

                if resolution_proven:
                    # (b) A failure escalation was filed and resolved by a human.
                    logger.info(
                        'DeterministicRunner: task %s before_done ran + escalation '
                        'resolved — resume-after-resolution, setting done (no re-run)',
                        task_id,
                    )
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-deploy',
                            note='resumed after human resolution',
                            unit=target_unit,
                        ),
                    )
                    return WorkflowOutcome.DONE

                # (c) Crash window: stamped but never verified and never escalated.
                # Task 2618: before re-escalating, if a persisted freshness
                # baseline is available, RE-RUN the read-only verify inspect and
                # reuse the SAME health classifier the harness recon-sweep
                # already applies to a stranded deploy
                # (_deterministic_deploy_health_verdict) — this recovers a
                # deploy that actually succeeded but crashed inside
                # _writeback_deploy_success before before_done_verified_at could
                # be stamped (task 2584/esc-2584-*: an infinite reblock loop on
                # every orchestrator restart). Gated on BOTH deploy_state.phase
                # == RAN (the genuine crash-window strand: ran but never
                # verified/escalated) AND a PERSISTED verify_baseline —
                # without the baseline, the classifier's no-baseline fallback is
                # a near-constant 'healthy' liveness check for an always-on
                # unit, which would phantom-done and regress finding-4.0's D3
                # guard (see this module's design-decision history / task 2618
                # plan); without the RAN-phase check, the rare corrupted state
                # of phase==ESCALATED with every escalation record for this
                # task+role deleted (own_escalation_resolved/resolution_proven
                # both False) would also reach here and could re-verify/
                # phantom-done over a real prior failure instead of raising a
                # fresh crash-window escalation. This also scopes auto-recovery
                # to be best-effort
                # across an ORCHESTRATOR restart only: verify_baseline's
                # monotonic clock is CLOCK_MONOTONIC and resets on a MACHINE
                # reboot, so a post-reboot re-verify conservatively falls
                # through to 'unconfirmed' (safe — never phantom-done; a
                # reboot strand still resolves via a human, as today).
                #
                # Direct-completion note (reviewer_comprehensive, task 2618
                # amendment): _deterministic_deploy_health_verdict's own
                # docstring states its safety property rests on callers
                # RE-FILING/resolving an escalation rather than flipping task
                # status directly, so a wrong verdict surfaces via the
                # normal escalation/watcher machinery instead of silently
                # corrupting state. This call site is a deliberate, narrowly
                # -scoped THIRD exception to that pattern: on 'healthy' it
                # flips straight to done via _writeback_deploy_success
                # instead of escalating for a human/watcher to confirm.
                # Accepted risk, NOT closed by the RAN-phase/baseline gate
                # above: a deploy that genuinely FAILED (script rc!=0, or
                # never finished exec'ing) whose unit is SEPARATELY
                # restarted to a fresh active state before this resume runs
                # — an unrelated operator action, or systemd's own
                # Restart=on-failure cycling it back up — would also read
                # 'healthy' here and phantom-done, because none of
                # phase==RAN, a persisted baseline, or a strict monotonic
                # advance is evidence that THIS script's own run exited 0.
                # Closing that gap needs a positive, independently-persisted
                # "this run exited 0" signal (e.g. an rc stamped by
                # proc_supervision.RestartPlan.execute() before its verify
                # leg) — proc_supervision.py is outside this task's locked
                # module scope, so that is intentionally deferred as a
                # follow-up rather than bolted on here. Accepted for now:
                # the alternative is task 2584's proven-in-production
                # failure mode — an infinite reblock loop re-escalating on
                # EVERY orchestrator restart with zero auto-recovery —
                # against a narrow, comparatively rare false-positive
                # window.
                #
                # reverify_note, when set, records that a live re-verify was
                # attempted and came back non-healthy — enriches the
                # escalation detail below so the operator sees the runner
                # already re-checked (distinct from the no-baseline case,
                # which never attempts a re-verify and keeps the original
                # generic detail).  The summary is deliberately left
                # untouched in both cases — it is the dedup/reblock_guard
                # signature and must stay stable.
                reverify_note: str | None = None
                if (
                    deploy_state is not None
                    and deploy_state.phase == DeployPhase.RAN
                    and deploy_state.verify_baseline is not None
                ):
                    inspect_fn = self._unit_inspector or self._default_inspect_unit
                    # Task 4157: guarded — an inspector OSError here would
                    # otherwise escape run() and bypass this very escalation.
                    # The sentinel it degrades to classifies as 'unconfirmed'
                    # below under BOTH baseline modes, so control falls
                    # through to the crash-window escalation with the
                    # reverify_note enrichment already in place.
                    fresh_state, reverify_cause = await self._inspect_unit_guarded(
                        inspect_fn, target_unit,
                    )
                    verdict = _deterministic_deploy_health_verdict(
                        fresh_state, verify_baseline=deploy_state.verify_baseline,
                    )
                    if verdict == 'healthy':
                        logger.info(
                            'DeterministicRunner: task %s crash-window re-verify '
                            'against persisted baseline came back healthy — '
                            'recovering to done without re-running the deploy '
                            '(I1): %s',
                            task_id, target_unit,
                        )
                        return await self._writeback_deploy_success(
                            task_id, metadata, fresh_state, target_unit, description,
                        )
                    logger.warning(
                        'DeterministicRunner: task %s crash-window re-verify came '
                        'back %s (state=%s) — falling through to the crash-window '
                        'escalation',
                        task_id, verdict, fresh_state,
                    )
                    reverify_note = (
                        f'A live re-verify was attempted against the persisted '
                        f'verify_baseline and came back {verdict!r} (observed '
                        f'unit state: {fresh_state}) — not confirmed fresh enough '
                        f'to recover automatically.'
                    )
                    if reverify_cause is not None:
                        # The observed state above is the MainPID=0 sentinel,
                        # which renders identically for every degradation mode
                        # — name the real one rather than making the operator
                        # scrape the journal for it (INV-2).
                        reverify_note += (
                            f'\nCause: the re-verify inspect did not return a '
                            f'trustworthy reading — {reverify_cause}'
                        )

                # Re-escalate instead of phantom-completing; the deploy is NOT
                # re-run (I1 once-only) — a human must verify the unit state.
                logger.warning(
                    'DeterministicRunner: task %s before_done_ran_at set but neither '
                    'verified nor escalated — crash-window detected; re-escalating '
                    'instead of phantom-done',
                    task_id,
                )
                crash_detail_lines = [
                    description,
                    f'Target unit: {target_unit}',
                    'before_done_ran_at is stamped but the deploy recorded neither a '
                    'verification (before_done_verified_at unset) nor a failure '
                    'escalation — the orchestrator crashed mid-deploy between '
                    'stamping and completing.  The deploy is NOT re-run (I1 '
                    'once-only); a human must inspect the unit and resolve.',
                ]
                if reverify_note is not None:
                    crash_detail_lines.append(reverify_note)
                return await self._file_infra_issue_and_block(
                    task_id,
                    summary=f'Deploy state unknown after crash: {target_unit}',
                    detail='\n'.join(crash_detail_lines),
                    metadata=metadata,
                )

            # ── Stop-instruction guard (task 2509) ───────────────────────────
            # First dispatch of a non-predicate before_done task (reached here
            # only when before_done_ran_at was falsy above — a resume/idempotency
            # dispatch never reaches this point).  Scan the task description for
            # an explicit stop instruction BEFORE running the deploy and BEFORE
            # stamping before_done_ran_at — a stop instruction is the
            # highest-authority human directive and must never be self-
            # authorized around (task 2407's autonomous /unblock session did
            # exactly that; task 2273's SIGTERM-killed sibling session is the
            # "what right looks like" precedent this guard makes a self-halt
            # rather than relying on an external kill).
            stop_phrase = detect_stop_instruction(description)
            if stop_phrase:
                stop_detail = '\n'.join([
                    description,
                    f'Target unit: {target_unit}',
                    f'Explicit stop instruction detected: {stop_phrase!r}.',
                    'The deploy was NOT run and before_done_ran_at was NOT '
                    'stamped.  Recovery needs BOTH steps: (1) edit the task '
                    'description to remove the stop instruction — otherwise '
                    'the guard re-fires and re-escalates on the very next '
                    "dispatch — and (2) resolve this escalation (e.g. "
                    "resolve_issue action='resume' or 'restart') to move the "
                    'task back to pending; a blocked task is not '
                    're-dispatched automatically.  Alternatively, cancel the '
                    'task outright.',
                ])
                return await self._file_stop_instruction_and_block(
                    task_id,
                    summary=f'Explicit stop instruction — deploy withheld: {target_unit}',
                    detail=stop_detail,
                )

            # ── Refused before_done record (task 9116) ───────────────────────
            # A refusal depends only on the stored record, so it is filed
            # BEFORE the stamp below: once the record is fixed, the next
            # dispatch runs the deploy instead of resuming to done unrun (I1).
            # No metadata: there is no deploy_state phase to advance yet.
            try:
                paths = self._before_done_paths(before_done)
            except BeforeDonePathsRefused as exc:
                return await self._file_infra_issue_and_block(
                    task_id,
                    summary=(
                        f'Deploy refused before_done: {target_unit}' if target_unit
                        else 'Deploy refused before_done (no target_unit)'
                    ),
                    detail='\n'.join([
                        description,
                        f'Target unit: {target_unit or "(none)"}',
                        f'before_done was refused, so the deploy was NOT run: {exc}',
                        'before_done_ran_at was NOT stamped.  Fix the record with '
                        'update_task, then resolve this escalation (e.g. '
                        "resolve_issue action='resume' or 'restart') to move the "
                        'task back to pending; the next dispatch runs the deploy.',
                    ]),
                )

            # Stamp before_done_ran_at FIRST (crash-safe I1: stamp-before-run means a
            # crash mid-deploy leaves the stamp set → re-dispatch does NOT re-run).
            # This stamp is SHARED for both self-target and cross-unit paths so I1
            # holds for both (ε design decision 5). ζ: this is also the initial
            # deploy_state.phase write (None -> RAN), atomically merged in the
            # SAME update_task call (DS-1).
            now_iso = datetime.now(UTC).isoformat()
            await self._advance_deploy_phase(
                task_id, metadata, DeployPhase.RAN,
                evidence={'before_done_ran_at': now_iso},
                phase_timestamp=now_iso,
            )

            # ── ε: self-target detection ─────────────────────────────────────
            # If target_unit IS the orchestrator's own unit, running the blocking
            # cross-unit deploy would kill this runner mid-execution (self-kill risk).
            # Instead, schedule a detached transient unit via systemd-run so the
            # restart fires AFTER run() returns, and set the task done immediately
            # with kind='deterministic-deploy-scheduled' (done = scheduled, not
            # verified).  See PRD §3, §4 decisions 8/9.
            own_unit = (self._own_unit_resolver or self._default_resolve_own_unit)()
            self_target = bool(own_unit) and (target_unit == own_unit)
            if self_target:
                transient_unit = f'orch-redeploy-restart-{task_id}.service'
                # Clamp to a sane minimum (5 s) so a task that sets
                # on_active_delay_secs=0 (or a non-positive value) cannot
                # produce --on-active=0, which would make the transient unit fire
                # effectively immediately — re-introducing the self-kill window
                # this detached-deferral design exists to prevent.
                on_active_secs = max(int(before_done.get('on_active_delay_secs', 60)), 5)

                async def _completion_failure(exc: Exception) -> WorkflowOutcome:
                    # Defense-in-depth (task 2004): ANY exception in the restart-fn ->
                    # stamp -> scheduled-done-write window (including a terminal write
                    # rejected by the provenance validator) must surface LOUDLY instead
                    # of bubbling past Harness._run_slot's generic `except Exception`
                    # handler, which returns a silent BLOCKED TaskReport with no
                    # escalation and no way to self-heal — the exact signature observed
                    # on tasks 1976/1982.  before_done_ran_at is already stamped (I1
                    # once-only), so it is always safe to resolve via a born-at-L2
                    # infra_issue rather than re-running the restart.
                    #
                    # Deliberately NOT used to guard the rc!=0 branch's own
                    # _file_infra_issue_and_block call below (task 2004 amendment): if
                    # that filing itself raises, letting it propagate avoids a
                    # misleading re-file under a "completion failed" summary — the
                    # dedup guard in _file_infra_issue_and_block makes a second filing
                    # harmless, but the mismatched summary/detail is not.
                    completion_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Transient unit: {transient_unit}',
                        f'Self-restart completion failed: {exc!r}',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Self-restart completion failed: {target_unit}',
                        detail=completion_detail,
                        metadata=metadata,
                    )

                try:
                    restart_fn = self._restart_scheduler or self._default_schedule_detached_restart
                    rc, tail = await restart_fn(
                        before_done,
                        transient_unit=transient_unit,
                        on_active_secs=on_active_secs,
                        task_id=task_id,
                        summary=f'Self-restart scheduling failed: {target_unit}',
                    )
                except Exception as exc:
                    return await _completion_failure(exc)

                if rc != 0:
                    detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Transient unit: {transient_unit}',
                        f'systemd-run exit code: rc={rc}',
                        f'Output:\n{tail}',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Self-restart scheduling failed: {target_unit}',
                        detail=detail,
                        metadata=metadata,
                    )

                try:
                    # Stamp before_done_scheduled_at — positive proof the transient unit
                    # was successfully registered.  If the orchestrator crashes between
                    # this stamp and the done write, the resume path (sub-case b-self
                    # above) drives to done with scheduled provenance instead of
                    # re-escalating as a generic crash-window.  ζ: this is also the
                    # ran->scheduled phase advance, atomically merged in the SAME
                    # update_task call (DS-1).
                    await self._advance_deploy_phase(
                        task_id, metadata, DeployPhase.SCHEDULED,
                        evidence={'before_done_scheduled_at': {
                            'at': datetime.now(UTC).isoformat(),
                            'transient_unit': transient_unit,
                            'fire_delay_secs': on_active_secs,
                        }},
                    )

                    if not always_escalates:
                        logger.info(
                            'DeterministicRunner: task %s self-restart scheduled — '
                            'transient_unit=%s on_active_secs=%d — setting done',
                            task_id, transient_unit, on_active_secs,
                        )
                        await self.scheduler.set_task_status(
                            task_id,
                            'done',
                            done_provenance=_build_done_provenance(
                                SCHEDULED_DEPLOY_PROVENANCE_KIND,
                                unit=target_unit,
                                transient_unit=transient_unit,
                                fire_delay_secs=on_active_secs,
                            ),
                        )
                        return WorkflowOutcome.DONE
                except Exception as exc:
                    return await _completion_failure(exc)
                # always_escalates=True (act-then-ask, non-exemplar): fall through
                # to the gate (section 3) WITHOUT running the blocking cross-unit
                # deploy.  The `if not self_target:` guard below ensures the
                # cross-unit baseline→run→verify block is SKIPPED entirely so the
                # orchestrator is not self-killed and the restart is not double-deployed.
                logger.info(
                    'DeterministicRunner: task %s self-restart scheduled with '
                    'always_escalates=True — falling through to gate (no cross-unit deploy)',
                    task_id,
                )

            # ── end ε self-target branch ─────────────────────────────────────
            # Cross-unit blocking deploy: guarded by `if not self_target:` so a
            # self-target task NEVER runs the blocking deploy against its own unit.
            # Self-target always_escalates=False already returned DONE above.
            # Self-target always_escalates=True falls through directly to the gate
            # (section 3) WITHOUT running the blocking deploy — this prevents:
            #   (a) self-kill: the blocking deploy would kill this runner mid-run;
            #   (b) double-deploy: the detached transient restart was already scheduled.
            if not self_target:
                if not target_unit:
                    # ── target_unit-less cross-unit deploy (task 2632 / esc-2585-1) ──
                    # A falsy target_unit (an explicit None, or the key
                    # omitted entirely -> before_done.get('target_unit', '')
                    # == '') names no specific systemd unit, so there is
                    # nothing to baseline-inspect or fresh-PID-verify against
                    # — inspecting the empty unit name returns a degenerate
                    # ActiveState-less dict that the baseline gate below
                    # misreads as a wedge (esc-2585-1's exact escalation).
                    # Skip the baseline gate AND the FreshPidVerify/
                    # RestartPlan leg entirely (deliberately never reached)
                    # and drive the outcome on the script's exit code alone,
                    # via the shared _run_deploy_script_guarded helper (task
                    # 2632 review amendment: this branch used to hand-inline
                    # a byte-for-byte copy of that helper's run + timeout-
                    # translate + escalate ladder, which could silently drift
                    # from the helper over time — routing through the helper
                    # makes that impossible).
                    no_target_note = (
                        'No target_unit is set — the deploy is driven on the '
                        'script exit code alone (task 2632 / esc-2585-1).'
                    )
                    result = await self._run_deploy_script_guarded(
                        task_id, before_done, description, no_target_note,
                        metadata=metadata,
                    )
                    if isinstance(result, WorkflowOutcome):
                        return result
                    rc, tail = result

                    if not always_escalates:
                        return await self._writeback_deploy_success(
                            task_id, metadata, {}, target_unit or '', description,
                        )

                    # always_escalates=True (act-then-ask, no target_unit):
                    # the script already ran — fall through to the gate
                    # rather than the named-target path's textual fallthrough
                    # (this branch always RETURNs, so it calls the gate
                    # helper directly instead of falling out of the `if not
                    # target_unit:` block into the shared cross-unit code).
                    logger.info(
                        'DeterministicRunner: task %s target_unit-less deploy ran with '
                        'always_escalates=True — falling through to gate',
                        task_id,
                    )
                    return await self._file_milestone_gate_and_block(task_id, task, metadata)

                # Capture baseline unit state before the deploy fires
                inspect_fn = self._unit_inspector or self._default_inspect_unit
                baseline, baseline_cause = await self._inspect_unit_guarded(
                    inspect_fn, target_unit,
                )

                # Task 2091 (baseline-leg hardening): a wedged/failed baseline
                # inspect returns the same MainPID=0/ActiveState='' sentinel
                # dict used on the verify leg (see _default_inspect_unit's
                # TimeoutError branch).
                #
                # Task 4157: that sentinel now also stands in for a SPAWN
                # failure (missing `systemctl`, or a fork failure under
                # resource pressure), not only a timeout. Both modes arrive
                # here via `_inspect_unit_guarded` above — which is what keeps
                # the gate below reachable at all, since an unguarded OSError
                # would escape run() entirely and file no escalation.
                #
                # On the VERIFY leg that sentinel is
                # already caught by the `pid > 0` half of the freshness check
                # below. On the BASELINE leg it is NOT: baseline_monotonic
                # would silently become 0, and `new_monotonic >
                # baseline_monotonic` is then trivially true for any active
                # unit — a wedged baseline would be swallowed and a deploy
                # falsely reported verified even though freshness was never
                # actually established. ActiveState is the signal: a real
                # `systemctl show` always populates it (even 'inactive' for a
                # nonexistent unit); only the timeout sentinel leaves it ''.
                # Fail closed exactly like the other pre-deploy failure paths
                # below — before_done_ran_at is already stamped (I1
                # once-only), so the deploy is NOT attempted on an untrusted
                # baseline.
                if not baseline.get('ActiveState'):
                    baseline_lines = [
                        description,
                        f'Target unit: {target_unit}',
                        f'Baseline inspect failed/wedged before deploy: {baseline!r}',
                    ]
                    if baseline_cause is not None:
                        # That sentinel repr is identical for EMFILE, a missing
                        # `systemctl` and a hung systemd — three failures with
                        # three different fixes (host, install, unit). Name the
                        # one that actually fired instead of leaving it in the
                        # journal only (INV-2 structured-facts-at-failure).
                        baseline_lines.append(f'Cause: {baseline_cause}')
                    baseline_lines.append(
                        'Cannot establish a trustworthy pre-deploy baseline — '
                        'the deploy was NOT attempted (before_done_ran_at is '
                        'already stamped; I1 once-only — a human must inspect '
                        'the unit and resolve).',
                    )
                    baseline_detail = '\n'.join(baseline_lines)
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Baseline inspect failed before deploy: {target_unit}',
                        detail=baseline_detail,
                        metadata=metadata,
                    )

                # ζ DS-3: persist the trustworthy baseline into deploy_state as a
                # same-phase enrichment (phase stays RAN — see
                # _enrich_deploy_state_baseline). Best-effort; never blocks the
                # deploy itself.
                await self._enrich_deploy_state_baseline(task_id, metadata, baseline)

                # Run the deploy script + fresh-PID verify by delegating to
                # proc_supervision.RestartPlan.execute() (task 2238/δ, RP-2/
                # RP-5) instead of an inline run-then-reinspect block.  The
                # runner still owns: the pre-flight baseline above, the outer
                # wall-clock guard below (task 2090 Layer B — now bounding the
                # WHOLE execute() call, run + verify, rather than only the
                # run), and the disposition -> outcome mapping.
                # proc_supervision owns the RP-2 blocking-run/RP-5 fresh-PID-
                # verify mechanics.
                #
                # The subprocess RUN itself is routed through the runner's own
                # task-2090-hardened script_runner/_default_run_script seam
                # via a create_subprocess_exec-compatible SHIM below — NOT
                # through execute()'s own bare (untimed, non-process-group-
                # aware) spawn — so Layer A (process-group kill of leaked
                # grandchildren) and the (before_done)->(rc, tail) seam
                # contract are both preserved.
                run_fn = self._script_runner or self._default_run_script
                outer_timeout = self._deploy_outer_timeout(before_done)

                class _RunFnProcShim:
                    """Adapts a (rc, tail) pair from run_fn into a
                    create_subprocess_exec-compatible object for
                    RestartPlan._execute_cross_unit_blocking's injectable
                    `runner` seam (exposes .returncode + async communicate())."""

                    def __init__(self, rc: int, tail: str) -> None:
                        self.returncode = rc
                        self._tail = tail

                    async def communicate(self) -> tuple[bytes, None]:
                        return self._tail.encode(errors='replace'), None

                async def _shim_runner(*_args, **_kwargs):
                    # Route through the shared translating-timeout wrapper
                    # (see _invoke_run_fn_translating_timeout's docstring for
                    # why the seam-internal TimeoutError must be translated
                    # here) instead of a local copy, so this branch and the
                    # target_unit-less branch's _run_deploy_script_guarded
                    # cannot drift apart (task 2632 review amendment).
                    #
                    # Task 4252: a ScriptTimeout from run_fn deliberately is
                    # NOT translated here — it propagates out of this shim,
                    # through plan.execute() (which wraps nothing), to run()'s
                    # own dedicated arm below.  That is why the arm must
                    # precede that try's `except Exception` catch-all.
                    rc, tail = await self._invoke_run_fn_translating_timeout(run_fn, before_done)
                    return _RunFnProcShim(rc, tail)

                # RestartOutcome carries only disposition/escalated/detail —
                # not the verified MainPID/timestamp _writeback_deploy_success
                # needs for done_provenance.  This wrapper stashes execute()'s
                # single post-deploy re-inspect (via the SAME inspect_fn used
                # for the baseline above) so the fresh unit state survives
                # past plan.execute() without a second, wasteful/racy inspect.
                captured: dict = {}

                async def _capturing_inspector(unit: str, **_kwargs) -> dict:
                    # Task 4157 (reviewer amendment): guarded like the other
                    # two inspect legs. An inspector OSError here does NOT
                    # violate run()'s contract — plan.execute()'s broad
                    # `except Exception` below already catches it — but it is
                    # MISATTRIBUTED there as 'Deploy run_fn failed (unexpected
                    # error)', blaming the deploy script for a systemd-inspect
                    # failure (INV-2 block-report-misattribution). Degrading to
                    # the sentinel instead routes it into the verify leg's own
                    # `pid > 0` / _empty_baseline_fresh rejection, so it lands
                    # as VERIFY_FAILED -> 'Deploy verify failed: <unit>' — the
                    # same disposition the DEFAULT inspector already produces
                    # for this failure, since systemd_inspect swallows the
                    # OSError into the sentinel before it can be raised. The
                    # cause is stashed for the escalation detail below.
                    state, cause = await self._inspect_unit_guarded(inspect_fn, unit)
                    if cause is not None:
                        captured['inspect_cause'] = cause
                    captured['new_state'] = state
                    return state

                verify = FreshPidVerify(
                    baseline_active_enter_monotonic=baseline.get(
                        'ActiveEnterTimestampMonotonic', 0,
                    ),
                    baseline_main_pid=baseline.get('MainPID', 0),
                    inspect_timeout_secs=self._inspect_timeout_secs,
                )
                try:
                    plan = RestartPlan(
                        script=paths.script,
                        args=list(before_done.get('args') or []),
                        cwd=paths.cwd,
                        target_unit=target_unit,
                        # own_unit must be truthy and provably non-self here (this
                        # branch is only reached when the runner's OWN self_target
                        # check above already ruled out self-target), so
                        # RestartPlan.execute() takes the RP-2 cross-unit-blocking
                        # path — never RP-1's fail-closed refuse — even when
                        # ORCH_UNIT is unset (own_unit == ''), preserving the
                        # existing fail-open-to-cross-unit behaviour.
                        own_unit=own_unit or _CROSS_UNIT_OWN_UNIT_SENTINEL,
                        on_failure_escalation=None,
                        verify=verify,
                        transient_unit=None,
                        env=dict(before_done.get('env') or {}),
                    )
                except ValueError as exc:
                    refusal_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'The deploy plan refused before_done, so the deploy script '
                        f'was NOT run: {exc}',
                        _I1_NOT_RERUN_LINE,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy plan refused before_done: {target_unit}',
                        detail=refusal_detail,
                        metadata=metadata,
                    )

                try:
                    outcome = await asyncio.wait_for(
                        plan.execute(runner=_shim_runner, inspector=_capturing_inspector),
                        timeout=outer_timeout,
                    )
                except TimeoutError:
                    # This guard now bounds the delegated plan.execute() call,
                    # which runs BOTH the deploy subprocess and the post-deploy
                    # verify re-inspect (_capturing_inspector -> inspect_fn) —
                    # asyncio.wait_for cannot tell which of the two was still
                    # pending when the timeout fired, so the message must not
                    # pin the blame solely on "the subprocess".
                    timeout_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Deploy run+verify exceeded the outer guard timeout ({outer_timeout}s = '
                        f"before_done['timeout_secs'] + run_timeout_grace_secs).",
                        'The hang may be in the deploy subprocess itself or in the post-deploy '
                        'unit inspect (verify) — the outer guard wraps both and cannot '
                        'distinguish which was pending. Check the unit out-of-band (e.g. '
                        'systemctl --user status, ps) before taking further action.',
                        _I1_NOT_RERUN_LINE,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy run+verify exceeded outer guard: {target_unit}',
                        detail=timeout_detail,
                        metadata=metadata,
                    )
                except ScriptTimeout as exc:
                    # The default runner's own per-script guard fired inside
                    # _shim_runner and propagated out of plan.execute()
                    # untouched (see the ScriptTimeout docstring for the
                    # classification).  MUST precede the catch-all below,
                    # which would otherwise mislabel it as an 'unexpected
                    # error' — same category, materially worse diagnostics.
                    inner_timeout_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        *_deploy_timeout_fact_lines(exc, script=paths.script),
                        'The post-deploy fresh-PID verify never ran, so the unit state '
                        'after the timeout is unobserved — check it out-of-band (e.g. '
                        'systemctl --user status) before resolving.',
                        _I1_NOT_RERUN_LINE,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=(
                            f'Deploy script timed out '
                            f'({exc.summary_phrase}): {target_unit}'
                        ),
                        detail=inner_timeout_detail,
                        metadata=metadata,
                    )
                except Exception as exc:
                    error_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Deploy run_fn raised an unexpected error: {exc!r}',
                        _I1_NOT_RERUN_LINE,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy run_fn failed (unexpected error): {target_unit}',
                        detail=error_detail,
                        metadata=metadata,
                    )

                if outcome.disposition == RestartDisposition.RESTART_FAILED:
                    # B7a: script failed — file infra_issue escalation, set blocked (B7a)
                    deploy_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        outcome.detail,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy failed: {target_unit}',
                        detail=deploy_detail,
                        metadata=metadata,
                    )

                if outcome.disposition == RestartDisposition.VERIFY_FAILED:
                    # B7b: verify failed — file infra_issue escalation, set blocked
                    verify_lines = [
                        description,
                        f'Target unit: {target_unit}',
                        outcome.detail,
                    ]
                    inspect_cause = captured.get('inspect_cause')
                    if inspect_cause is not None:
                        # The verify DID fail, but on a degraded reading rather
                        # than on evidence the deploy didn't take — the operator
                        # needs to know which (task 4157 reviewer amendment).
                        verify_lines.append(
                            f'Cause: the post-deploy verify inspect did not return '
                            f'a trustworthy reading — {inspect_cause}. The unit '
                            f'state above is the MainPID=0 degradation sentinel, '
                            f'not an observation of the unit.'
                        )
                    verify_detail = '\n'.join(verify_lines)
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy verify failed: {target_unit}',
                        detail=verify_detail,
                        metadata=metadata,
                    )

                if outcome.disposition != RestartDisposition.DEPLOYED_AND_VERIFIED:
                    # Defensive: REFUSED is structurally unreachable from this
                    # call site (own_unit is always forced truthy+non-self
                    # above, so RP-1 never refuses here) — never silently
                    # swallow an unexpected disposition rather than surfacing
                    # it as an infra_issue.
                    refused_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Unexpected RestartPlan disposition: {outcome.disposition!r}',
                        outcome.detail,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy refused unexpectedly: {target_unit}',
                        detail=refused_detail,
                        metadata=metadata,
                    )

                new_state = captured.get('new_state', {})

                if not always_escalates:
                    # Pure cross-unit deploy (B6): verified → set done with provenance.
                    # Task 2066: the deploy may have severed the orchestrator's own
                    # fused-memory/MCP connection (e.g. it restarted the service
                    # backing that connection) — delegate to the connection-resilient
                    # writeback helper rather than a single unguarded write pair.
                    return await self._writeback_deploy_success(
                        task_id, metadata, new_state, target_unit, description,
                    )

                # always_escalates=True with before_done (cross-unit act-then-ask):
                # action already ran — fall through to the gate below.
                logger.info(
                    'DeterministicRunner: task %s before_done ran with always_escalates=True '
                    '— falling through to gate',
                    task_id,
                )

        # ── 3. Pure gate ─────────────────────────────────────────────────────
        # Assertion: always_escalates must be True for a pure gate task.
        # (A non-escalating deterministic task with no before_done would be a
        # no-op — guard against misconfiguration loudly.)
        if not always_escalates:
            raise ValueError(
                f'DeterministicRunner: task {task_id} has before_done=None and '
                'always_escalates=False — this combination is not supported in β.'
            )

        return await self._file_milestone_gate_and_block(task_id, task, metadata)
