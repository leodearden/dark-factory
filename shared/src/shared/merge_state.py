"""Merge-state wire vocabularies — the POLL and SUBMIT contracts.

See PRD ``plans/merge-status-durable-non-landed-prd.md`` (contract D1,
decision D3, behaviour B9).  ``MergeState`` is the vocabulary of
``merge_status`` / ``merge_cancel``'s ``state`` field; ``MergeSubmitStatus``
is the vocabulary of ``merge_request``'s ``status`` field.  They are TWO
vocabularies, not one — see ``MergeSubmitStatus``'s docstring.

This module supersedes the hand-maintained enumerations previously duplicated
across:

- ``escalation/src/escalation/server.py`` (``merge_cancel``'s and
  ``merge_status``'s docstring state lists)
- ``skills/merge-queue/SKILL.md``
- ``skills/unblock/SKILL.md``
- ``skills/unblock-low-risk/SKILL.md``
- ``skills/escalation-watcher/SKILL.md``

Those prose sites keep their value lists INLINE — a runbook an agent executes
at runtime must show the values, not a pointer to a Python file — but each
list now sits inside a ``merge-state-vocab`` marked span naming the partition
it must equal, and ``scripts/tests/test_merge_state_vocabulary_consistency.py``
asserts set equality between every span and its partition.  Same convention as
``CONTRIBUTING.md``'s ``lint-command-mirror`` block.

This module is intentionally PURE: it defines the two vocabularies and their
derived partitions only.  It holds no mapping between them and no I/O — the
raw-status -> ``MergeState`` collapse belongs to
``escalation/src/escalation/server.py::_map_terminal_state``, and duplicating
it here would create the second home this module exists to remove.

It is also intentionally STANDALONE-LOADABLE: stdlib imports only, no
intra-package imports.  ``scripts/tests/test_merge_state_vocabulary_consistency.py``
loads this file by absolute path via ``importlib.util.spec_from_file_location``
rather than importing ``shared.merge_state``, so that the guard reads the
vocabulary from the SAME tree as the SKILL.md files it checks.  A plain import
would resolve through whatever editable install is on ``sys.path``, which in a
task worktree is typically the MAIN checkout (see ``CLAUDE.md``, "Locating
installed code").  ``shared/tests/test_merge_state.py::TestStandaloneLoadability``
pins the property.

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (``from shared.merge_state
import MergeState, ...``), consistent with the ``task_statuses``/
``mcp_envelope``/``neutral_cwd``/``config_dir`` sub-module convention.
"""

from __future__ import annotations

import enum

__all__ = [
    'MergeState',
    'LIVE_STATES',
    'TERMINAL_STATES',
    'OUTCOME_STATES',
    'EPISTEMIC_STATES',
    'POLL_STOP_STATES',
    'CANCEL_STATES',
    'MergeSubmitStatus',
    'SUBMIT_NON_TERMINAL',
    'SUBMIT_TERMINAL',
]


class MergeState(enum.StrEnum):
    """Closed POLL vocabulary — ``merge_status`` / ``merge_cancel``'s ``state``.

    Members are genuine ``str`` instances (``enum.StrEnum``), so the server's
    switchover onto this enum is wire-compatible: responses serialise to the
    same JSON strings and existing ``resp['state'] == 'done'`` comparisons
    keep holding.

    Member NAMES are the wire values (unlike
    ``shared/src/shared/task_statuses.py``'s UPPER_CASE names), matching the
    PRD's D1 contract block, whose member names ARE the values.

    ``already_merged`` is deliberately NOT a member: it is a SUBMIT-only value
    that ``escalation/src/escalation/server.py::_map_terminal_state`` collapses
    to ``done``.  It lives on ``MergeSubmitStatus``.
    """

    # ── outcome states (PRD D1): what actually happened to the merge ──
    # live
    queued = 'queued'
    verifying = 'verifying'
    gate = 'gate'
    finalizing = 'finalizing'
    # terminal
    done = 'done'
    conflict = 'conflict'
    blocked = 'blocked'
    abandoned = 'abandoned'
    superseded = 'superseded'

    # ── epistemic states (PRD D1): what the SERVER KNOWS about the merge ──
    # Landed here in alpha; nothing emits the first three until beta ships
    # their emitters.  ``unknown`` is emitted today (Tier 4, and
    # ``merge_cancel``'s no-live-waiter miss).
    no_record = 'no_record'
    stale_record = 'stale_record'
    journaled = 'journaled'
    unknown = 'unknown'


# ---------------------------------------------------------------------------
# Poll partitions.
#
# LIVE_STATES and TERMINAL_STATES are the real primitive split and are
# enumerated.  Everything else is DERIVED, so the exhaustiveness assertions in
# shared/tests/test_merge_state.py are true by construction and a new member
# cannot be added without being assigned a partition.  That assignment is what
# reddens the prose spans pinned to that partition — the B9 drift mechanism.
# ---------------------------------------------------------------------------

#: In-flight states — the merge has not reached an outcome yet.
#: escalation/src/escalation/server.py::_map_live_state emits exactly these.
LIVE_STATES = frozenset(
    {
        MergeState.queued,
        MergeState.verifying,
        MergeState.gate,
        MergeState.finalizing,
    }
)

#: Settled states — the merge reached an outcome.
#: escalation/src/escalation/server.py::_map_terminal_state emits exactly these.
TERMINAL_STATES = frozenset(
    {
        MergeState.done,
        MergeState.conflict,
        MergeState.blocked,
        MergeState.abandoned,
        MergeState.superseded,
    }
)

#: What happened to the merge (PRD D1's 9 outcome states).
OUTCOME_STATES = LIVE_STATES | TERMINAL_STATES

#: What the server KNOWS about the merge.  Derived as the COMPLEMENT of
#: OUTCOME_STATES (mirroring task_statuses.ACTIVE) rather than re-listed, so a
#: member added to MergeState without an outcome assignment lands here and the
#: exhaustiveness assertion stays true by construction.
EPISTEMIC_STATES = frozenset(MergeState) - OUTCOME_STATES

#: What a REQUEST_ID/TASK_ID-scoped ``merge_status`` poll loop may stop on.
POLL_STOP_STATES = TERMINAL_STATES | {MergeState.unknown}

#: Exactly what ``merge_cancel``'s ``state`` field can be.
#:
#: Its equality with POLL_STOP_STATES today is INCIDENTAL, not a shared
#: definition — these are two facts with two homes, and they answer different
#: questions ("what can merge_cancel report" vs "what may a poll loop stop
#: on").  They are expected to DIVERGE: PRD D1 narrows ``unknown`` to
#: probe-failure, so beta moves merge_cancel's no-record miss path to
#: ``no_record`` while the poll-loop stop set is unaffected.  Do not collapse
#: them into one constant — that would make the beta change silently mutate an
#: unrelated pinned prose span.
CANCEL_STATES = TERMINAL_STATES | {MergeState.unknown}


class MergeSubmitStatus(enum.StrEnum):
    """Closed SUBMIT vocabulary — ``merge_request``'s ``status`` field.

    A SEPARATE vocabulary from ``MergeState``, not a superset of it.  Five
    spellings appear in both (``queued``, ``done``, ``conflict``, ``blocked``,
    ``superseded``), but they are distinct members of distinct enums that
    happen to share a wire spelling.  Nine values are submit-only and eight are
    poll-only; ``shared/tests/test_merge_state.py::TestVocabulariesAreDistinct``
    pins the split.

    WHY 14 MEMBERS AND NOT THE PRD CONTRACT BLOCK'S ILLUSTRATIVE 6.
    ``escalation/src/escalation/server.py`` returns ``'status': outcome.status``
    VERBATIM, so the wire vocabulary IS
    ``orchestrator/src/orchestrator/merge_types.py::MergeOutcome.status`` — a
    12-member ``Literal`` — plus the two non-terminal response shapes.  The
    PRD's 6-member block names only the values its surrounding prose was
    discussing; PRD Open Question 3 explicitly delegates full membership to
    this task ("Decide in alpha").  Shipping the 6-member sketch would make the
    drift guard fire falsely against ``skills/merge-queue/SKILL.md``'s submit
    list, which already names ``done``, ``conflict`` and ``blocked``.

    THE MAPPING TO ``MergeState``, stated as a POINTER not as data.  When a
    submit status is later served through the poll wire,
    ``escalation/src/escalation/server.py::_map_terminal_state`` collapses it:
    ``already_merged`` and ``done_wip_recovery`` -> ``MergeState.done``;
    ``wip_halted``, ``wip_recovery_no_advance``, ``unmerged_state``,
    ``stash_failed``, ``unknown_branch`` and ``error`` -> ``MergeState.blocked``.
    That mapping is deliberately NOT encoded here — it belongs to the server,
    and a copy of it would be exactly the second home this module exists to
    remove.

    Two values the runbooks have historically invented are NOT members, and
    both absences are asserted in the test file: ``'failed'`` (which appears
    nowhere in ``server.py`` — the real value is ``error``) and
    ``'needs_rebase'`` (which exists only in
    ``orchestrator/src/orchestrator/suffix_graph.py`` as an internal structured
    bounce log line, never as a response status).
    """

    # ── non-terminal response shapes ──
    #: wait_secs=0 dispatched, or a wait_secs>0 bounded wait that timed out.
    queued = 'queued'
    #: coalesced onto an already in-flight request.
    attached = 'attached'

    # ── terminal: MergeOutcome.status's 12-member Literal, verbatim ──
    done = 'done'
    conflict = 'conflict'
    blocked = 'blocked'
    already_merged = 'already_merged'
    wip_halted = 'wip_halted'
    done_wip_recovery = 'done_wip_recovery'
    wip_recovery_no_advance = 'wip_recovery_no_advance'
    unmerged_state = 'unmerged_state'
    stash_failed = 'stash_failed'
    unknown_branch = 'unknown_branch'
    superseded = 'superseded'
    error = 'error'


#: The two shapes that mean "not settled yet" — the caller must keep polling
#: ``merge_status``.  Enumerated (it is the small, stable half).
SUBMIT_NON_TERMINAL = frozenset({MergeSubmitStatus.queued, MergeSubmitStatus.attached})

#: Every settled ``merge_request`` status == MergeOutcome.status's Literal.
#: DERIVED by complement, deliberately: a member added to MergeSubmitStatus
#: lands in the terminal partition by DEFAULT, which reddens the prose spans
#: pinned to SUBMIT_TERMINAL rather than letting the new value slip in
#: undocumented.
SUBMIT_TERMINAL = frozenset(MergeSubmitStatus) - SUBMIT_NON_TERMINAL
