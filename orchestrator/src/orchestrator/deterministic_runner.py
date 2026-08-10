"""DeterministicRunner — orchestrator-side runner for deterministic gate tasks (β/γ/ε).

A *deterministic* task (``metadata.task_kind == 'deterministic'``) is routed
here by ``Harness._run_slot`` instead of ``TaskWorkflow``.  The runner holds
only ``scheduler`` + ``escalation_queue`` (no git_ops) — structurally proving
that no worktree, branch, agent, or steward is created for a gate (I4/B2).

Phase β delivers the **pure-gate** pattern
(``before_done=None``, ``always_escalates=True``).

Phase γ adds the **before_done blocking cross-unit deploy** path
(``before_done`` is a dict, ``always_escalates=False``):

1. **Idempotency / quiescence** (checked first):
   If ``metadata.gate_escalated_at`` is already set:
   - If a pending escalation still exists for the task → return BLOCKED (B3:
     no second escalation on quiescence).
   - Else (escalation resolved) → drive the task to ``done`` and return DONE
     (I2/B4/B11: resume path).

   If ``metadata.before_done_ran_at`` is already set (γ):
   - If a pending infra_issue escalation exists → return BLOCKED (B7: reaper
     no-rerun, I1 once-only).
   - Else (no pending escalation) → require POSITIVE proof of a terminal
     outcome before completing (phantom-done guard for the crash window
     between stamping ``before_done_ran_at`` and recording a terminal result):
     * ``before_done_verified_at`` set → verify passed (crash before the done
       write) → drive to ``done``.
     * an escalation was filed and resolved → human acted (act-then-ask) →
       drive to ``done``, no re-run.
     * neither → crash mid-deploy before any terminal decision.  Task 2618:
       if ``deploy_state.phase`` is ``RAN`` (the genuine crash-window strand)
       AND a PERSISTED ``deploy_state.verify_baseline`` is available, RE-RUN
       the read-only verify inspect against ``target_unit`` and re-classify
       via ``_deterministic_deploy_health_verdict`` (the same classifier the
       harness recon-sweep uses) — a positive ``'healthy'`` verdict means the
       deploy actually succeeded but crashed inside
       ``_writeback_deploy_success`` before ``before_done_verified_at`` could
       be stamped (task 2584's shape), so it drives straight to ``done`` via
       ``_writeback_deploy_success`` instead of re-escalating.  Any other
       verdict (including no persisted baseline at all) falls through to
       re-escalate (infra_issue, blocked); never phantom-done, never re-run
       (I1).  Auto-recovery is best-effort across an ORCHESTRATOR restart
       only — ``verify_baseline``'s monotonic clock resets on a MACHINE
       reboot, so a post-reboot re-verify conservatively yields
       ``'unconfirmed'`` and still escalates to a human.  This is a
       deliberate, narrowly-scoped exception to
       ``_deterministic_deploy_health_verdict``'s documented "callers only
       re-file/resolve an escalation, never flip status directly" contract
       — see the sub-case (c) inline comment below for the accepted-risk
       rationale.

2. **before_done execution** (γ: ``before_done`` is not None):
   **Stop-instruction guard** (task 2509, reconciliation finding 0aac21b4):
   on the FIRST dispatch of a non-predicate ``before_done`` task — after the
   predicate dispatch below and after the ``before_done_ran_at`` idempotency
   block above, but BEFORE the ``before_done_ran_at`` stamp is written —
   ``orchestrator.stop_instruction.detect_stop_instruction`` scans the task
   description for an explicit stop instruction (e.g. "do not apply").  If
   found: file a born-at-L2 ``stop_instruction`` escalation and block WITHOUT
   running the deploy and WITHOUT stamping ``before_done_ran_at``.  Unlike
   the other escalations in this section, ``gate_escalated_at`` is
   deliberately NOT stamped — this is a re-checkable HALT (mirrors task
   2273's SIGTERM-kill on its human-rehearsal mandate as a self-halt rather
   than an external kill), not a resolve-to-done gate: it guarantees the
   NEXT dispatch re-evaluates this guard from scratch instead of taking
   section 1's ``if gate_escalated_at:`` resume-to-done branch (which would
   either raise ``NotImplementedError`` — no ``before_done_ran_at`` proof —
   or wrongly drive to done without the action ever running).  That
   re-evaluation only happens once the task is back in ``pending``, though:
   a ``blocked`` task holding this open pending escalation is NOT
   re-dispatched automatically, so removing the stop instruction from the
   description is necessary but not sufficient — full recovery needs BOTH
   (1) editing the description (else the guard re-fires and re-escalates on
   the very next dispatch) AND (2) resolving the ``stop_instruction``
   escalation itself (e.g. ``resolve_issue`` action ``resume``/``restart``,
   which is what actually flips the task's status back to ``pending`` — see
   ``escalation.server.resolve_issue`` Table B), or cancelling the task.
   Excludes predicates (read-only, apply nothing — see the predicate
   sub-path below) and excludes every resume/idempotency branch above (those
   already have ``before_done_ran_at`` or ``gate_escalated_at`` set, so they
   never reach this first-dispatch check).

   The ``before_done_ran_at`` stamp is written FIRST (crash-safe I1) and is
   SHARED between the self-target (ε) and cross-unit sub-paths below.

   Phase ε **self-restart** sub-path (``before_done.target_unit`` == own unit):
   - Detects self-target by comparing ``before_done.target_unit`` to the
     orchestrator's own systemd unit name, resolved from the ``ORCH_UNIT``
     environment variable via ``_default_resolve_own_unit()``.  An empty or
     unresolved ``ORCH_UNIT`` fails-open to the cross-unit path so existing
     CI runs (where ``ORCH_UNIT`` is unset) are unaffected.
   - **Operator requirement**: set ``ORCH_UNIT=<unit-name>`` in the
     ``[Service] Environment`` of the orchestrator's own systemd unit.
   - Instead of running the blocking cross-unit deploy (which would kill this
     runner mid-execution), schedules a detached ``systemd-run`` transient
     unit that fires *after* ``run()`` returns.
   - The transient unit's payload is a ``/bin/sh -c`` wrapper that runs the
     restart and, *only if it exits non-zero*, fires δ's ``escalation submit``
     CLI (file-backed, no MCP server needed in the detached unit) to file a
     born-at-L2 ``infra_issue`` escalation.  Because the whole unit is deferred
     via ``--on-active``, nothing runs at scheduling time — the escalation is
     reached only through the shell failure branch at fire time, never on the
     success path.
   - If scheduling succeeds (``rc == 0``): stamp ``before_done_scheduled_at``
     (a dict carrying the transient unit name and fire delay) as a crash-safe
     marker.  If the runner crashes between this stamp and the done write or
     gate filing, the resume path (b-self) below handles recovery.
   - If scheduling succeeds (``rc == 0``) and ``always_escalates=False``:
     set task ``done`` with
     ``done_provenance.kind='deterministic-deploy-scheduled'`` carrying the
     transient unit name and fire delay; return DONE (done = *scheduled*, not
     *verified*).  A crash between the stamp and done write is recovered by
     (b-self) driving to done with scheduled provenance.
   - If scheduling succeeds (``rc == 0``) and ``always_escalates=True``
     (act-then-ask): fall through directly to section 3 (the gate) WITHOUT
     running the blocking cross-unit deploy.  The detached restart is already
     scheduled; the gate is filed as a ``milestone_gate`` escalation; the task
     is set to ``blocked``.  Neither ``unit_inspector`` nor ``script_runner``
     are invoked on this path (self-kill and double-deploy prevention).  A
     crash between the stamp and the gate filing is recovered by (b-self)
     re-filing the gate — the human-approval gate is NEVER bypassed.
   - If scheduling fails (``rc != 0``): file born-at-L2 infra_issue, block
     (parallel to γ's rc≠0/verify-fail handling); ``before_done_ran_at``
     already stamped (not re-run, I1).

   Phase γ **cross-unit** sub-path (``target_unit`` ≠ own unit or ``ORCH_UNIT`` unset):
   - Capture baseline unit state (``unit_inspector``).
   - Run the deploy script to completion (``script_runner``, blocking).
   - If ``rc != 0``: file born-at-L2 ``infra_issue`` escalation, set blocked
     (B7a).
   - Re-inspect and verify freshness (B7b), delegated to
     ``proc_supervision.RestartPlan.execute()``'s ``FreshPidVerify`` check
     (task 2238/δ): when the pre-deploy baseline had a persistent MainPID
     (``baseline_main_pid > 0``), a fresh non-sentinel ``MainPID`` (>0,
     different from the baseline) and a strictly-later
     ``ActiveEnterTimestampMonotonic`` are both required. When the baseline
     was EMPTY (``baseline_main_pid == 0`` — a ``.timer`` unit or a
     ``Type=oneshot`` service, neither of which ever reports a live MainPID,
     even once genuinely active — task 2611/esc-2584-1), the ``MainPID>0``
     requirement is dropped: freshness instead defers to the shared
     ``systemd_inspect._empty_baseline_fresh`` predicate, requiring the
     ``ActiveEnterTimestampMonotonic`` to have strictly advanced past the
     baseline AND ``ActiveState`` to have settled into ``'active'`` or
     ``'inactive'`` (an allowlist — a transient/mid-transition state, a
     wedged/missing ``ActiveState``, and ``'failed'`` are all rejected).
   - If ``always_escalates=False``: hand off to ``_writeback_deploy_success``
     (task 2066), which stamps ``before_done_verified_at`` (the positive proof
     the resume path requires) and then sets the task to ``done`` with
     ``done_provenance.kind='deterministic-deploy'`` carrying the fresh PID
     and timestamp (B6).  Both writes are patiently retried — bounded and
     paced — because the deploy script may itself have severed the
     orchestrator's own fused-memory/MCP connection (e.g. it restarted the
     service backing that connection); a plain retry recovers once the
     connection auto-reconnects.  If the writeback budget is exhausted
     first, files a durable local ``infra_issue`` escalation and returns
     BLOCKED instead of silently stranding the task with an empty queue.
   - If ``always_escalates=True``: fall through to gate (act-then-ask).

   Phase γ **cross-unit, target_unit-less** sub-path (``target_unit`` is
   falsy — an explicit ``None``, or the key omitted entirely so
   ``before_done.get('target_unit', '')`` is ``''``; the documented
   "``target_unit=None`` → cross-unit, no named unit" configuration): there
   is no specific systemd unit to baseline-inspect or fresh-PID-verify
   against — inspecting the empty unit name returns a degenerate
   ``{'MainPID': 0, 'ActiveEnterTimestampMonotonic': 0}`` dict with no
   ``ActiveState``, which the baseline gate above misreads as a wedge (task
   2632 / esc-2585-1's exact escalation — the 2-key baseline repr
   byte-matches an inspect of ``''``, not a real wedge). Both the baseline
   gate and the FreshPidVerify/RestartPlan leg are skipped entirely; the
   deploy is driven on the script's exit code alone:
   - ``rc == 0``: hand off to ``_writeback_deploy_success`` (empty
     ``new_state``, ``pid=0`` — the same helper the named-target path uses)
     unless ``always_escalates=True``, in which case fall through to the
     gate (act-then-ask) instead, since the script already ran.
   - ``rc != 0``, an outer wall-clock guard timeout, or an unexpected
     ``run_fn`` error: file born-at-L2 ``infra_issue``, return BLOCKED
     (parallel to B7a); ``before_done_ran_at`` is already stamped (I1), so
     the deploy is NOT re-run.
   Named-target genuine-wedge detection (the baseline/verify logic above)
   is entirely unchanged — this sub-path is reached only when
   ``target_unit`` itself is falsy.

   Phase γ **predicate** sub-path (``before_done['kind'] == 'predicate'``):
   - A read-only exit-code VERDICT check — NOT a deploy.  Dispatched at the
     very top of section 2, above the deploy-only ``target_unit``/I1/baseline
     machinery, so a predicate NEVER touches systemd (no ``unit_inspector``,
     no fresh-PID verify) and NEVER stamps ``before_done_ran_at`` (re-running
     a read-only check on crash-resume is harmless — there is no I1 side
     effect to double-apply).  This dispatch also happens BEFORE
     ``always_escalates`` is ever consulted, so a predicate task's
     ``always_escalates`` value is simply ignored — it is meaningful only on
     the deploy/gate paths' act-then-ask semantics.  A ``kind='predicate'``
     task with ``always_escalates=True`` set still goes straight to
     ``done``/``milestone_check_failed`` from the check's exit code alone,
     with no gate ever filed.  Rejecting (or normalizing) that combination,
     if ever desired, belongs in fused-memory's ``submit_task``/
     ``deterministic_task_guard`` validation, not here (out of this runner's
     scope).
   - Runs the check script under the same outer wall-clock guard pattern as
     the deploy path (``script_runner``/``_default_run_script`` +
     ``asyncio.wait_for(timeout_secs + run_timeout_grace_secs)``).
   - ``rc == 0`` (invariant holds): set task ``done`` with
     ``done_provenance.kind='deterministic-milestone'`` (never
     ``'deterministic-deploy'`` — no deploy happened).  The ``note`` carries
     a BOUNDED STRUCTURED VERDICT from ``_summarize_predicate_output``, not
     the raw stdout tail (task 3286): the note is appended to a Mem0
     completion summary downstream, so raw subprocess output landing there is
     ingested into memory (task 2902's specimen).  The raw output is logged
     at INFO first, so nothing is silently discarded.
   - ``rc != 0`` (invariant violated — a VERDICT, not an infra fault): file a
     born-at-L2 ``milestone_check_failed`` escalation, stamp
     ``gate_escalated_at`` (reusing the gate resume machinery so a human
     resolving the escalation on the next dispatch routes through section 1's
     quiescence/resolve-to-done fork), and block.
   - Timeout or an unexpected ``run_fn`` error (no verdict was produced — an
     infra fault, not a check failure): file born-at-L2 ``infra_issue`` (the
     existing timeout→escalate path) and block — NO ``gate_escalated_at``
     stamp, so the check is simply re-attempted on the next dispatch.
   - Section-1 resume: when ``gate_escalated_at`` is set and the
     ``milestone_check_failed`` escalation is resolved, RE-RUNS the predicate
     check (delegating back to ``_run_predicate`` — read-only, so repeating it
     is harmless) rather than trusting the resolution blindly: resolving an
     escalation is not proof the invariant now holds (a human may resolve it
     prematurely or in error).  ``rc == 0`` drives to ``done`` with
     ``deterministic-milestone`` provenance; ``rc != 0`` re-files
     ``milestone_check_failed`` (the dedup guard sees no pending escalation,
     since it was just resolved) and stays blocked; a timeout/error routes to
     ``infra_issue``, identical to the first-dispatch path.  This
     short-circuits BEFORE the ``before_done_ran_at`` proof-check (a
     predicate never stamps it).

3. **Pure gate** (``before_done=None``, ``always_escalates=True``):
   - File one born-at-L2 escalation (I3: in-process submit, sentinel role
     ``orchestrator-deterministic`` keeps level=2 past the server downgrade gate).
     Dedup: if a pending escalation already exists for the task (e.g. from a prior
     crash-safe re-dispatch), filing is skipped to avoid duplicate L2 escalations.
   - Stamp ``metadata.gate_escalated_at`` (crash-safe: file-before-stamp means a
     stamp failure re-files rather than silently skipping the gate).
   - Set task status to ``blocked``.
   - Return BLOCKED (B2).

   **Resuming a pure gate** — the proof ladder has TWO rungs, not one:

   (i) An archive-inclusive, role-scoped escalation record must exist (task
   2954). This proves a gate was established and is no longer pending. An
   empty PENDING queue alone is not proof: a LOST escalation would otherwise
   let an operator's re-pend of a stuck task silently bypass the gate.

   (ii) If ``metadata.human_curator_gate`` is truthy, then
   ``metadata.human_curator_adjudicated_at`` must carry a non-empty string (task
   3341). This proves the human-judgment CONTENT work actually happened.

   Rung (i) passing is NOT evidence rung (ii) holds. Task 3181 was a
   ``human_curator_gate`` pure gate; ``esc-3181-1`` was resolved by
   ``escalation-watcher`` with ``action='resume'`` on 2026-07-30T19:41Z, and
   that resolution's own text says the ~13-entry Mem0 corpus edit was
   "curator action, deliberately NOT executed here". The resume path
   nonetheless drove the task to ``done`` with the generic
   ``note='pure gate resolved'``, contradicting the task's own
   ``consolidation_note``. Closing an escalation record and performing the
   curator's content review are different propositions.

   Rung (ii) is a PURE-gate rung only: ``human_curator_gate`` and a
   ``before_done`` action are contradictory (one says only a human closes
   this, the other is a machine step that does), so a task carrying both
   takes the act-then-ask path with the marker unread. That combination is a
   task-authoring defect and is logged as a WARNING on every dispatch rather
   than being silently ignored. It is ALSO rejected at write time by
   ``shared.task_metadata``'s cross-field validator (task 3369), so it can no
   longer land through the ``submit_task``/``update_task`` boundary; this
   WARNING is retained as the defence-in-depth backstop for records that did
   not pass through it.

   Both rung-(ii) checks fail CLOSED: a truthy-but-not-``True`` marker still
   trips the guard, and a non-``str`` or blank stamp is not proof. An
   unproven curator gate files a born-at-L2 ``curator_adjudication_missing``
   escalation — a DISTINCT category, so the re-ask is legible in the census
   and does not present to a resolver as "the same gate again" — and stays
   BLOCKED. The block → resolve → re-dispatch → block loop is bounded by the
   harness ``reblock_guard`` (``Harness._check_reblock_guard``), which keys
   on ``category:summary``, so the distinct category gets its own counter
   (INV-4's storm-escape, with no second counter built here — INV-5).

   Ordering matters and is pinned by test: rung (i) runs FIRST. Zero records
   anywhere means the gate was never established, so re-filing the ORIGINAL
   ``milestone_gate`` is the correct recovery; asking for an adjudication
   stamp on a gate nobody has yet seen would be incoherent.

   A curator gate that DOES close carries a specific ``done_provenance.note``
   naming the adjudication stamp (plus the gate record via
   ``escalation_id``), never the generic string — that genericness is what
   made 3181's phantom closure indistinguishable from a genuine one. The
   interpolated stamp is length-capped because this note is memory-ingested
   downstream (``_format_outcome_echo``, task 3286) — the same constraint
   section 2's predicate subsection documents, so both note-writing sites in
   this module read consistently.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from shared.task_metadata import HUMAN_CURATOR_GATE_KEY, DoneProvenance

from orchestrator import systemd_inspect
from orchestrator.deploy_state import (
    DeployPhase,
    DeployState,
    IllegalDeployTransition,
    VerifyBaseline,
    enforce_transition,
)
from orchestrator.fm_retry import FM_RESTART_RETRY_WINDOW_SECS, fm_retry_backoffs
from orchestrator.proc_supervision import (
    EscalationSpec,
    FreshPidVerify,
    RestartDisposition,
    RestartPlan,
)
from orchestrator.stop_instruction import detect_stop_instruction
from orchestrator.systemd_inspect import (
    _deterministic_deploy_health_verdict,
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
# it for the duration of a `--drain` restart.  `McpSession._raw_call`
# (mcp_lifecycle.py) opens a fresh httpx.AsyncClient and re-inits the session
# on any transient failure, so the NEXT call auto-reconnects — a successful
# write IS the reconnection proof.  The post-deploy verify+writeback retry
# budget (``DeterministicRunner._writeback_backoffs``) sizes off the shared
# ``orchestrator.fm_retry.fm_retry_backoffs()`` schedule (task 2706) instead
# of an independently-guessed budget, so it patiently outlasts such a
# restart — see orchestrator/src/orchestrator/fm_retry.py.

# Task 2090: bound the reap after a whole-process-group SIGKILL on a
# before_done subprocess timeout.  A process stuck in an uninterruptible
# state (or one that otherwise ignores SIGKILL) must never be allowed to hang
# `_terminate_process_tree` — and therefore `_default_run_script` — forever.
# 5s comfortably covers a normal kernel-mediated reap while staying bounded.
_REAP_GRACE_SECS: float = 5.0

# Task 2090: outer wall-clock backstop around the cross-unit `run_fn` call in
# run().  This is a pure safety margin ON TOP OF before_done['timeout_secs'] —
# it must never fire before the inner script-runner timeout in the normal
# case, so 30s is deliberately generous (covers process-group teardown +
# reap_grace_secs with room to spare) rather than tight.
_RUN_TIMEOUT_GRACE_SECS: float = 30.0

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

# Task 3341: the human-curator-gate contract — a marker key plus a stamp key.
#
# `human_curator_gate` marks a pure deterministic gate whose resolution requires
# human CONTENT adjudication, not merely a closed escalation record.  It is
# written today by reconciliation Stage 2 (LLM-authored; no code produces it).
# `human_curator_adjudicated_at` is the ISO-8601 stamp that PROVES the per-entry
# review actually happened — stamped on the task via `update_task` by whoever
# performed the review.  Both are blessed Tier-A keys
# (shared.task_metadata._BLESSED_METADATA_KEYS) because this module reads them.
#
# The two are deliberately separate: the marker says "a human must judge the
# CONTENT", the stamp says "a human did".  Task 3181 is the incident where the
# runner had the first and inferred the second from a closed escalation record.
#
# SINGLE SOURCE (task 3369): the MARKER's spelling is imported from
# `shared.task_metadata` (see the import block above) rather than restated as a
# local literal.  That module's `_deterministic_invariants` validator now reads
# the key too — to reject the marker alongside a non-null `before_done` at the
# write boundary — so a second definition would put two spellings of one key in
# the two modules that both act on it, which is exactly the drift a named
# constant exists to prevent.  The STAMP keeps a bare literal because `shared`
# offers no constant to import: it names the stamp only inside
# `_BLESSED_METADATA_KEYS`, with no validator reading it.
#
# NAMING (reviewer amendment): the stamp is `human_curator_adjudicated_at`, NOT
# `curator_adjudicated_at`.  The bare `curator_*` metadata namespace is already
# owned by a different subsystem — `curator_action` / `curator_justification` /
# `combined_at` are written by the AUTOMATED task curator's combine flow
# (fused_memory.task_interceptor).  Squatting that prefix for the HUMAN content
# curator would invite a reader of the Tier-A list, or a census consumer
# grouping by prefix, to conflate two unrelated actors.  The `human_curator_`
# prefix pairs the stamp unambiguously with its marker.
HUMAN_CURATOR_ADJUDICATED_AT_KEY: str = 'human_curator_adjudicated_at'

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
    ``None`` are observationally identical there; extra fields such as
    ``transient_unit`` / ``fire_delay_secs`` survive via ``DoneProvenance``'s
    ``extra='allow'``.

    ``kind``/``**fields`` are intentionally loosely typed (``str``/``object``):
    the six call sites forward heterogeneous field subsets, so pyright's
    dataclass-transform-synthesized ``DoneProvenance.__init__`` (each
    parameter narrowly typed, e.g. ``kind: Literal[...]``, ``pid: int | None``)
    cannot be satisfied by a generic pass-through signature. The real
    validation happens at runtime, in the model itself.
    """
    return DoneProvenance(kind=kind, **fields).model_dump(exclude_none=True)  # type: ignore[arg-type]


# The payload budget.  Sits safely inside the 500-char `max_note_chars` cap
# that fused-memory's `_format_outcome_echo` applies downstream (tasks
# 2049/2054/2080), so a surviving note is not re-truncated there.
_PREDICATE_NOTE_MAX_PAYLOAD_CHARS = 400

# Log shape: a leading ISO/logging timestamp, OR a standalone level token
# anywhere in the line.  Used only to REJECT a tier-2 candidate line.
_LOG_LINE_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}'
    r'|\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b'
)


def _summarize_predicate_output(out: object, *, rc: int) -> str:
    """Summarize a predicate check's raw output into a bounded provenance note.

    Anything the extractor does not recognize is dropped, leaving the bare
    verdict prefix — an unrecognized shape therefore yields a less
    informative note, never a corrupted one.

    Motivating specimen — task 2902.  ``_default_run_script`` merges stderr
    into stdout and returns ``decode()[-2000:]``, so a chatty predicate
    script's server-log noise reached ``done_provenance.note`` verbatim: a
    1999-char blob starting mid-token, carrying FalkorDB identity-scan
    WARNINGs for an unrelated project and ``httpx`` request lines.

    Why this matters beyond tidiness: ``note`` is not a private field.
    fused-memory's reconciliation ``_format_outcome_echo`` reads
    ``done_provenance['note']`` and appends it to the ``"Task '<title>'
    completed."`` Mem0 write, so whatever lands here is INGESTED INTO MEMORY.
    Raw subprocess output is dropped at this seam for that reason alone; it
    stays recoverable in the orchestrator log, which is not memory-ingested.

    Extraction runs two tiers with DIFFERENT guarantees — do not read the
    whole of it as an allowlist.  Tier 1 (the script's own trailing JSON
    block) IS a true allowlist: only a parseable structured payload survives,
    and every preceding log line is excluded STRUCTURALLY rather than by
    pattern-matching.  Tier 2 (a single clean final line) is a best-effort
    heuristic guarded by a log-shape DENYLIST (``_LOG_LINE_RE``), and it
    inherits a denylist's weakness: a log line under a formatter nobody
    anticipated still reaches the note.  Concretely, a ``%(name)s
    %(message)s`` line such as ``httpx HTTP Request: GET http://...`` carries
    neither timestamp nor level token and is kept verbatim.

    Tier 2 cannot be tightened into a grammar-based allowlist without
    discarding the very verdicts it exists to preserve: the real in-repo
    predicate outputs ``check ok: 0 flakes`` and
    ``check_merge_flakiness: ... -- invariant holds`` are prose-shaped and
    structurally indistinguishable from a formatter-less log line.  The
    residual exposure is therefore bounded rather than closed: at most ONE
    line, at most ``_PREDICATE_NOTE_MAX_PAYLOAD_CHARS`` of it — never the
    multi-KB blob task 2902 stamped.  ``scripts/scan_provenance_note_log_leaks.py``
    shares this blind spot (its discriminator also requires a timestamp), so
    a predicate script that wants its verdict preserved intact should emit
    trailing JSON — tier 1 is the only tier with a real guarantee.

    The denylist rejection is otherwise deliberately CONSERVATIVE and will
    drop an otherwise clean final line that merely contains a standalone
    ``INFO``/``ERROR`` token.  Losing a payload is the safe failure
    direction: the verdict prefix always survives, and ``_run_predicate``
    logs the raw output before calling this, so nothing is silently
    discarded.

    An oversized payload is replaced WHOLESALE with a marker naming the
    dropped size — never sliced.  Task 2054 found a raw ``note[:N]`` slice
    downstream cutting mid-token (garbling ``8679,8680`` into ``8679,868``);
    a sliced JSON object is worse still, unparseable yet still
    structured-looking to a reader.

    *rc* is rendered as both a word and a number, and the word is DERIVED
    from the number (``passed`` iff ``rc == 0``).  Today the only caller is
    ``_run_predicate``'s ``rc == 0`` branch, but a hardcoded ``passed`` would
    let a future non-zero caller stamp the self-contradictory note
    ``predicate check passed (rc=2)``; the text cannot contradict the value.
    """
    verdict = f'predicate check {"passed" if rc == 0 else "failed"} (rc={rc})'
    payload = _extract_predicate_payload(out)
    if payload is None:
        return verdict
    if len(payload) > _PREDICATE_NOTE_MAX_PAYLOAD_CHARS:
        payload = (
            f'<verdict payload elided: {len(payload)} chars exceeds '
            f'{_PREDICATE_NOTE_MAX_PAYLOAD_CHARS}-char cap '
            f'— see the orchestrator log>'
        )
    return f'{verdict}: {payload}'


def _extract_predicate_payload(out: object) -> str | None:
    """Return the recognized structured payload in *out*, or None.

    Tier 1 — a trailing JSON block: walk backwards to the last line whose
    strip opens a ``{``/``[``, and try to parse from there to end of output.
    Re-dumped with compact separators so the note stays a single line.

    Tier 2 — the last non-blank line, iff it carries no log shape
    (``_LOG_LINE_RE``).  This is what preserves the real in-repo predicate
    verdicts (``check ok: 0 flakes``, ``-- invariant holds``,
    ``measured_median_ms=…``).

    Otherwise None: an unrecognized shape yields no payload at all.
    """
    lines = out.splitlines() if isinstance(out, str) else []
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip().startswith(('{', '[')):
            continue
        try:
            parsed = json.loads('\n'.join(lines[index:]))
        except ValueError:
            continue
        return json.dumps(parsed, separators=(',', ':'))

    for line in reversed(lines):
        candidate = line.strip()
        if not candidate:
            continue
        return None if _LOG_LINE_RE.search(candidate) else candidate
    return None


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
        and provenance.get('kind') == 'deterministic-deploy-scheduled'
    ):
        return True
    return bool(metadata.get('before_done_scheduled_at'))


def build_milestone_gate_escalation_fields(
    task: dict | None, metadata: dict,
) -> tuple[str, str, list]:
    """Build ``(summary, detail, options)`` for a born-at-L2 ``milestone_gate``.

    THE single seam both born-at-L2 pure-gate producers route through:
      - the runner's own gate filing
        (``DeterministicRunner._file_milestone_gate_and_block``), and
      - the harness recon-sweep's strand recovery
        (``Harness._recover_stranded_deterministic_gate``, task 2954).

    Extracted (task 2954 amendment) so a re-filed gate is byte-identical to
    what the runner would have filed.  Critically this includes the
    operational-LLM token prefix (task 2803 γ): a naive re-build in the sweep
    would DROP the ``OPERATIONAL_LLM_NEEDS_LANE_TOKEN`` on summary+detail and
    misroute the human handling lane, so centralizing the construction keeps
    the two producers in lockstep.

    ``detail`` = description + a "Landed dependencies:" line when the task has
    dependencies; ``summary`` = ``title[:200]``.  For an operational-LLM gate
    (``_is_operational_llm_gate``) both are token-prefixed (token-first so the
    token survives the summary's ``[:200]`` truncation and any tail-only log
    scrape; ``detail`` itself is not truncated).  Plain gates (marker absent)
    are byte-unchanged.  ``options`` is the task's ``gate_options`` (empty list
    when absent).  None/empty *task* is tolerated (defensive; the sweep's task
    dict is read back from the scheduler).
    """
    task = task or {}
    title = task.get('title', '')
    description = task.get('description', '')
    deps = task.get('dependencies', []) or []
    dep_ids = [
        str(d.get('id', d) if isinstance(d, dict) else d) for d in deps
    ]
    detail_parts = [description]
    if dep_ids:
        detail_parts.append(f'\nLanded dependencies: {", ".join(dep_ids)}')
    detail = '\n'.join(detail_parts)
    summary = title[:200]
    if _is_operational_llm_gate(metadata):
        # Task 2803 (γ): token-first prefix/build so the token survives
        # downstream truncation (the summary's [:200] slice) and any
        # tail-only log scrape; detail itself is not truncated. Plain
        # gates (marker absent) are byte-unchanged.
        detail = (
            f'[{OPERATIONAL_LLM_NEEDS_LANE_TOKEN}] This operational ask needs '
            'LLM-operational handling; no automated lane exists yet — resolve '
            'by hand.\n\n'
        ) + detail
        summary = f'{OPERATIONAL_LLM_NEEDS_LANE_TOKEN}: {title}'[:200]
    options = list(metadata.get('gate_options') or [])
    return summary, detail, options


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
            ``_REAP_GRACE_SECS``.  Injected in tests so an unkillable-process
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
        reap_grace_secs=_REAP_GRACE_SECS,
        run_timeout_grace_secs=_RUN_TIMEOUT_GRACE_SECS,
        inspect_timeout_secs=_INSPECT_TIMEOUT_SECS,
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
        script = before_done['script']
        args = before_done.get('args') or []

        # The transient unit runs under the systemd --user manager, which does
        # NOT inherit the orchestrator's own working directory — it defaults to
        # $HOME.  A relative deploy `script` would therefore fail to be found
        # (exit 127) once the unit fires.  Resolve an explicit cwd from
        # before_done['cwd'] when the caller supplied one; otherwise fall back
        # to this process's own os.getcwd(), which is project_root because the
        # orchestrator's own systemd unit pins WorkingDirectory=project_root.
        # RestartPlan.__post_init__ absolutizes a relative `script` against
        # this `cwd` (RP-3), byte-identical to this method's prior inline
        # absolutization.
        cwd = before_done.get('cwd') or os.getcwd()

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
            category='infra_issue',
            agent_role=DETERMINISTIC_AGENT_ROLE,
        )

        plan = RestartPlan(
            script=Path(script),
            args=list(args),
            cwd=Path(cwd),
            target_unit=transient_unit,
            own_unit=transient_unit,
            on_failure_escalation=escalation_spec,
            verify=None,
            transient_unit=transient_unit,
            on_active_secs=on_active_secs,
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

    async def _default_inspect_unit(self, unit: str) -> dict:
        """Query systemctl for unit state fields needed for fresh-PID verify.

        Task 2119: thin delegate to the hoisted, hardened
        ``systemd_inspect.inspect_systemd_unit`` (task 2091's timeout/kill/
        reap hardening now lives there exactly once). Returns a dict with at
        minimum: MainPID (int), ActiveState (str), ActiveEnterTimestamp
        (str), ActiveEnterTimestampMonotonic (int). Integers default to 0 on
        parse failure (sentinel-safe).
        """
        return await inspect_systemd_unit(
            unit,
            timeout_secs=self._inspect_timeout_secs,
            reap_grace_secs=self._reap_grace_secs,
        )

    async def _default_run_script(self, before_done: dict) -> tuple[int, str]:
        """Run the deploy script to completion under a timeout.

        Adapts ``service_restart.py``'s spawn pattern but AWAITS completion
        (blocking cross-unit deploy — no self-kill risk on this path).

        Returns:
            (rc, output_tail) — rc is the process return code; output_tail is
            the last 2000 chars of combined stdout/stderr.
        """
        script = before_done['script']
        args = before_done.get('args') or []
        # Merge over os.environ so the child sees a full environment (PATH, HOME,
        # XDG_RUNTIME_DIR …).  An empty / absent env dict means full inherit.
        env = {**os.environ, **before_done['env']} if before_done.get('env') else None
        cwd = before_done.get('cwd') or None
        timeout_secs = before_done.get('timeout_secs', 60)

        proc = await asyncio.create_subprocess_exec(
            script, *args,
            env=env,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_secs)
            tail = (stdout or b'').decode(errors='replace')[-2000:]
            return proc.returncode or 0, tail
        except TimeoutError:
            # Task 2090: the deploy script may spawn grandchildren (e.g.
            # restart-fused-memory.sh --drain forking systemctl/curl/journalctl/
            # sleep/the restarted daemon) that inherit the write end of the
            # merged stdout pipe above.  Killing only this direct child leaves
            # them alive as orphans — still holding that pipe open and still
            # running indefinitely.  start_new_session=True (above) makes this
            # process its own session/group leader so the WHOLE tree can be
            # torn down together instead of leaking orphaned processes.
            await self._terminate_process_tree(proc)
            return 1, f'<script timed out after {timeout_secs}s>'

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Kill *proc*'s entire process group and bound the reap (task 2090).

        ``proc`` must have been spawned with ``start_new_session=True`` so its
        process group is its own — NOT the orchestrator's — otherwise
        ``os.killpg`` would SIGKILL the orchestrator itself.

        Tries a whole-group SIGKILL first, which reaches any grandchildren
        (e.g. a backgrounded ``systemctl``/``curl``/``sleep``) that would
        otherwise survive the direct child's death as leaked orphans —
        continuing to run (and keep the inherited stdout pipe's write end
        open) indefinitely.  Note this is NOT needed to unblock ``proc.wait()``
        below: asyncio's ``Process.wait()`` resolves on the child's own exit
        notification (via the event loop's child watcher), not on pipe EOF —
        only ``communicate()`` waits on that.  The process-group kill's value
        is purely in not leaking the grandchildren themselves.  If the process
        has already exited (or the group lookup otherwise fails), falls back
        to a direct ``proc.kill()``.  Both kill attempts tolerate an
        already-dead process — this helper must NEVER raise, so a timeout
        branch can always return cleanly even if the process is already gone.

        The reap itself is bounded by ``self._reap_grace_secs`` so a process
        stuck in an uninterruptible state cannot hang this helper (and
        therefore ``_default_run_script``) forever.

        Note (residual race): ``os.killpg(os.getpgid(proc.pid), ...)`` targets
        ``proc.pid`` by number.  If asyncio's child watcher has already reaped
        the zombie in the background before this call runs, the OS could in
        principle recycle that PID before ``os.getpgid``/``os.killpg`` execute,
        making the SIGKILL target an unrelated process group.  This is a
        low-probability race shared with any PID-based kill (not specific to
        this helper); the ``ProcessLookupError``/``OSError`` suppression above
        is the existing mitigation, not a full fix for the underlying race.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            logger.debug(
                'DeterministicRunner: killpg(%s) failed (%s: %s) — falling back '
                'to direct kill()',
                proc.pid, type(exc).__name__, exc,
            )
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()

        try:
            await asyncio.wait_for(proc.wait(), timeout=self._reap_grace_secs)
        except TimeoutError:
            logger.warning(
                'DeterministicRunner: process %s did not exit within '
                'reap_grace_secs=%s after termination signal — abandoning reap '
                '(process may be unkillable)',
                proc.pid, self._reap_grace_secs,
            )

    async def _file_infra_issue_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
        *,
        metadata: dict | None = None,
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
        from escalation.models import Escalation

        existing_pending = self.escalation_queue.get_by_task(
            task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
        )
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending escalation(s) — '
                'skipping re-file (infra_issue dedup guard)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category='infra_issue',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 infra_issue escalation %s for task %s',
                esc.id, task_id,
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
            logger.info('DeterministicRunner: task %s blocked — infra_issue', task_id)
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
        from escalation.models import Escalation

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
        from escalation.models import Escalation

        title = task.get('title') or task_id
        summary = f'Human curator gate {task_id} resumed without content adjudication'
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
            f"      update_task({task_id}, metadata={{'{HUMAN_CURATOR_ADJUDICATED_AT_KEY}': "
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
            category='curator_adjudication_missing',
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
        """DS-2 sink: file a born-at-L2 escalation for an illegal deploy-phase edge.

        Wired into every ``_advance_deploy_phase`` call via
        ``enforce_transition``'s ``escalation_sink`` — files BEFORE
        ``enforce_transition`` raises ``IllegalDeployTransition``, so an
        illegal edge is never silently swallowed (D2).  Mirrors the other
        escalation-filing helpers' construction (sentinel role keeps level=2
        past the server downgrade gate) but is unconditional — no dedup
        guard — since an illegal transition is itself a bug signal that must
        never be suppressed by an unrelated pending escalation.
        """
        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role=DETERMINISTIC_AGENT_ROLE,
            severity='critical',
            category='illegal_deploy_transition',
            summary=f'Illegal deploy-phase transition {old} -> {new}'[:200],
            detail=(
                f'DeterministicRunner attempted an illegal deploy-phase '
                f'transition {old!r} -> {new!r} for task {task_id}. This '
                'indicates a runner bug or corrupted deploy_state metadata — '
                'investigate before resuming.'
            ),
            level=2,
        )
        self.escalation_queue.submit(esc)
        logger.error(
            'DeterministicRunner: filed L2 illegal_deploy_transition escalation '
            '%s for task %s (%s -> %s)',
            esc.id, task_id, old, new,
        )

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
        ``McpSession._raw_call`` auto-reconnects on the next call after any
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
        """File a born-at-L2 ``milestone_gate`` escalation and block the task.

        Encapsulates section 3's gate-filing logic for reuse by:
        - The pure-gate path (section 3) in ``run()``.
        - The (b-self) crash-resume path when ``always_escalates=True``
          (crash between ``before_done_scheduled_at`` stamp and gate filing).

        Includes a dedup guard — if a pending escalation already exists
        (e.g. prior crash-safe re-dispatch), filing is skipped to avoid
        duplicate L2 escalations.  Stamps ``gate_escalated_at`` so the
        next resume routes through section-1 quiescence.

        Returns:
            ``WorkflowOutcome.BLOCKED``
        """
        from escalation.models import Escalation

        # Task 2954 amendment: summary/detail/options built via the shared
        # `build_milestone_gate_escalation_fields` seam so the harness recon-
        # sweep's strand recovery re-files a byte-identical gate (including the
        # operational-LLM token prefix) and the two producers stay in lockstep.
        summary, detail, gate_options = build_milestone_gate_escalation_fields(
            task, metadata,
        )

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
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category='milestone_gate',
                summary=summary,
                detail=detail,
                options=list(gate_options),
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 milestone gate escalation %s for task %s',
                esc.id, task_id,
            )

        # Stamp gate_escalated_at AFTER successful escalation submit.  ζ: on the
        # DEPLOY path (before_done set — this helper is also reached from the
        # self-restart and cross-unit act-then-ask fallthroughs, and the
        # self-restart crash-resume re-file), this atomically advances
        # deploy_state.phase to ESCALATED too (DS-1). A pure gate
        # (before_done=None) is not a deploy and gets no deploy_state.
        now_iso = datetime.now(UTC).isoformat()
        if metadata.get('before_done') is not None:
            await self._advance_deploy_phase(
                task_id, metadata, DeployPhase.ESCALATED,
                evidence={'gate_escalated_at': now_iso},
                phase_timestamp=now_iso,
            )
        else:
            await self.scheduler.update_task(
                task_id,
                {'gate_escalated_at': now_iso},
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

    async def _file_milestone_check_failed_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
    ) -> WorkflowOutcome:
        """File a born-at-L2 ``milestone_check_failed`` escalation and block (γ-predicate).

        A non-zero predicate exit code is a milestone VERDICT ("the invariant
        does not hold") — semantically distinct from an infra fault, so this
        mirrors ``_file_infra_issue_and_block``'s dedup/file/block shape but
        with a dedicated category AND additionally stamps ``gate_escalated_at``
        (mirroring ``_file_milestone_gate_and_block``) so the next dispatch
        routes through section-1's existing quiescence/resolve-to-done fork —
        a human resolving the escalation drives the task to done with no
        extra wiring.

        Includes a dedup guard — if a pending escalation already exists (e.g.
        prior crash-safe re-dispatch), filing is skipped to avoid duplicate L2
        escalations.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        from escalation.models import Escalation

        existing_pending = self.escalation_queue.get_by_task(
            task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
        )
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending escalation(s) — '
                'skipping re-file (milestone_check_failed dedup guard)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category='milestone_check_failed',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 milestone_check_failed escalation %s for task %s',
                esc.id, task_id,
            )

        # Stamp gate_escalated_at AFTER successful escalation submit (file-before-stamp
        # ordering, same as _file_milestone_gate_and_block) so a stamp failure re-files
        # on the next dispatch rather than silently skipping it.  A predicate NEVER
        # stamps before_done_ran_at — there is no I1 side effect to guard here.
        now_iso = datetime.now(UTC).isoformat()
        await self.scheduler.update_task(
            task_id,
            {'gate_escalated_at': now_iso},
            metadata_mode='merge',
        )

        # Best-effort, same as _file_infra_issue_and_block (reviewer amendment):
        # the escalation + gate_escalated_at stamp above are already durable, so
        # a severed connection on this trailing write must not propagate and
        # mask them behind a raw exception.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info(
                'DeterministicRunner: task %s blocked — milestone_check_failed', task_id,
            )
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the '
                'milestone_check_failed escalation is already durable on local '
                'disk, so returning BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )

        return WorkflowOutcome.BLOCKED

    async def _run_predicate(
        self, task_id: str, before_done: dict, description: str,
    ) -> WorkflowOutcome:
        """Run a read-only predicate check and map its exit code to a verdict (γ-predicate).

        Unlike the ``before_done`` deploy path, a predicate is a READ-ONLY
        exit-code verdict check — NOT a systemd deploy.  There is no unit to
        inspect, no baseline/fresh-PID verify, and no ``before_done_ran_at``
        I1 stamp (re-running a read-only check on crash-resume is harmless;
        see the module docstring's γ-predicate subsection).

        Called from two sites: the first dispatch (section 2), and section 1's
        resume branch once a ``milestone_check_failed`` escalation is
        resolved — resume re-invokes this method to re-verify the invariant
        rather than trusting the resolution blindly (a human may resolve the
        escalation without the underlying check now passing).

        ``metadata.always_escalates`` is never consulted here — see the
        module docstring's γ-predicate subsection.

        Maps the outcome:
        - ``rc == 0`` -> ``WorkflowOutcome.DONE``, with
          ``done_provenance.kind='deterministic-milestone'`` and a BOUNDED,
          STRUCTURED ``note`` from ``_summarize_predicate_output`` — never the
          raw stdout tail (task 3286, specimen 2902).  The note is not a
          private field: fused-memory's ``_format_outcome_echo`` appends it to
          a Mem0 completion-summary write, so raw subprocess output landing
          there is ingested into memory.  The full raw output is logged at
          INFO immediately before summarizing, so nothing is silently lost.
        - ``rc != 0`` -> a milestone VERDICT failure: born-at-L2
          ``milestone_check_failed`` escalation + ``gate_escalated_at`` stamp
          + blocked (routes through section-1's resume/quiescence machinery
          on the next dispatch).
        - Timeout / unexpected error -> an INFRA fault (no verdict was
          produced): born-at-L2 ``infra_issue`` escalation + blocked
          (re-attempted on the next dispatch, no ``gate_escalated_at`` stamp).

        Returns:
            WorkflowOutcome.DONE or WorkflowOutcome.BLOCKED.
        """
        run_fn = self._script_runner or self._default_run_script
        outer_timeout = before_done.get('timeout_secs', 60) + self._run_timeout_grace_secs

        async def _invoke_run_fn():
            # See run()'s identical inner wrapper: translate a seam-internal
            # TimeoutError into a distinct exception type here so `except
            # TimeoutError` below can only ever mean "the outer wall-clock
            # guard itself fired" — never a misattributed application error.
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
            )

        if rc != 0:
            # A non-zero exit is a milestone VERDICT ("invariant does not
            # hold"), not an infra fault — file milestone_check_failed (NOT
            # infra_issue) and stamp gate_escalated_at so a human resolving
            # the escalation drives the task to done on the next dispatch.
            #
            # The RAW `out` below is deliberate and must stay raw (task 3286):
            # unlike done_provenance.note, an escalation detail is read by a
            # human diagnosing the failing check and is never ingested into
            # memory, so full subprocess output is exactly what is wanted here.
            fail_detail = '\n'.join([
                description,
                f'Predicate check exit code: rc={rc}',
                f'Output:\n{out}',
            ])
            return await self._file_milestone_check_failed_and_block(
                task_id,
                summary=f'Milestone predicate check failed (rc={rc})',
                detail=fail_detail,
            )

        # Log the RAW output before summarizing it.  `_summarize_predicate_output`
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
        await self.scheduler.set_task_status(
            task_id,
            'done',
            done_provenance=_build_done_provenance(
                'deterministic-milestone',
                note=_summarize_predicate_output(out, rc=rc),
            ),
        )
        return WorkflowOutcome.DONE

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
        wall-clock guard, mapping a timeout / unexpected error / non-zero
        exit to an already-filed born-at-L2 ``infra_issue`` + ``BLOCKED``.

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
                'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Deploy run exceeded outer guard (no target_unit)',
                detail=timeout_detail,
                metadata=metadata,
            )
        except Exception as exc:
            error_detail = '\n'.join([
                description,
                note,
                f'Deploy run_fn raised an unexpected error: {exc!r}',
                'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
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
                'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
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
                    return await self._run_predicate(task_id, before_done, description)

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
                    own_records = self.escalation_queue.get_by_task(
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
                    # string but never the safety decision.  `_summarize_predicate_output`
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
                    # Cite the GATE record — NOT merely the newest own-role one
                    # (reviewer amendment).  The standard remediation round-trip
                    # this very guard creates leaves TWO own-role records behind:
                    #   (A) the original `milestone_gate`, resolved WITHOUT a
                    #       stamp — the record that proves rung one, and
                    #   (B) the `curator_adjudication_missing` re-ask this guard
                    #       filed, resolved once the human finally stamped.
                    # `escalation_id` is rung-ONE evidence, and both the module
                    # docstring and docs/task-authoring.md promise it names the
                    # gate record.  Taking the newest own record would cite (B),
                    # the re-ask — the one record that proves nothing about the
                    # gate — quietly mis-aiming the audit trail this whole change
                    # exists to sharpen.  So filter to the gate category first
                    # and take the newest of those; fall back to the newest own
                    # record only when no gate-category record survives (e.g. a
                    # gate filed under an older category), and to omitting the
                    # key when `own_records` is empty (unreachable — the strand
                    # branch already returned — but an empty list must never be
                    # indexed; `exclude_none=True` then simply drops the key).
                    # Pinned by test_curator_gate_remediation_round_trip_cites_the_gate_record.
                    _gate_records = [
                        e for e in own_records if e.category == 'milestone_gate'
                    ]
                    _citable = _gate_records or own_records
                    _gate_esc_id = (
                        sorted(_citable, key=lambda e: e.timestamp)[-1].id
                        if _citable else None
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
                            escalation_id=_gate_esc_id,
                        ),
                    )
                else:
                    # Pure-gate resume: no before_done action ran, so there is no
                    # deploy/unit/pid evidence — stamp a dedicated gate-kind
                    # provenance so the done write passes require_done_provenance
                    # without lying about a deploy having happened (task 2331).
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-gate',
                            note='pure gate resolved',
                        ),
                    )
                return WorkflowOutcome.DONE

        # ── 2. before_done execution (γ) ────────────────────────────────────
        # Cross-unit blocking deploy: stamp → baseline → run script → verify → done.
        # Self-target detection + detached systemd-run is deferred to ε.
        # A falsy target_unit (cross-unit, no named unit) skips the
        # baseline/verify legs entirely and drives on rc alone — see the
        # module docstring's target_unit-less sub-path and the
        # `if not target_unit:` branch below (task 2632 / esc-2585-1).
        if before_done is not None:
            # γ-predicate: a read-only exit-code verdict check — NOT a deploy.
            # Dispatched FIRST, above target_unit/I1/baseline, so a predicate
            # never touches systemd and never stamps before_done_ran_at.  This
            # is also BEFORE `always_escalates` is read below — a predicate
            # task's always_escalates is intentionally never consulted (see
            # module docstring's γ-predicate subsection); a
            # kind='predicate' + always_escalates=True task is not rejected
            # here and simply behaves as a plain predicate.
            if before_done.get('kind') == 'predicate':
                return await self._run_predicate(task_id, before_done, description)

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
                own_escalation_resolved = bool(self.escalation_queue.get_by_task(
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
                                'deterministic-deploy-scheduled',
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
                    fresh_state = await inspect_fn(target_unit)
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
                                'deterministic-deploy-scheduled',
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
                baseline = await inspect_fn(target_unit)

                # Task 2091 (baseline-leg hardening): a wedged/failed baseline
                # inspect returns the same MainPID=0/ActiveState='' sentinel
                # dict used on the verify leg (see _default_inspect_unit's
                # TimeoutError branch). On the VERIFY leg that sentinel is
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
                    baseline_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Baseline inspect failed/wedged before deploy: {baseline!r}',
                        'Cannot establish a trustworthy pre-deploy baseline — '
                        'the deploy was NOT attempted (before_done_ran_at is '
                        'already stamped; I1 once-only — a human must inspect '
                        'the unit and resolve).',
                    ])
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
                    state = await inspect_fn(unit)
                    captured['new_state'] = state
                    return state

                verify = FreshPidVerify(
                    baseline_active_enter_monotonic=baseline.get(
                        'ActiveEnterTimestampMonotonic', 0,
                    ),
                    baseline_main_pid=baseline.get('MainPID', 0),
                    inspect_timeout_secs=self._inspect_timeout_secs,
                )
                plan = RestartPlan(
                    script=Path(before_done['script']),
                    args=list(before_done.get('args') or []),
                    cwd=Path(before_done.get('cwd') or os.getcwd()).resolve(),
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
                        'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy run+verify exceeded outer guard: {target_unit}',
                        detail=timeout_detail,
                        metadata=metadata,
                    )
                except Exception as exc:
                    error_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Deploy run_fn raised an unexpected error: {exc!r}',
                        'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
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
                    verify_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        outcome.detail,
                    ])
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
