"""Tests for shared.merge_state — the two merge-state wire vocabularies.

PRD ``plans/merge-status-durable-non-landed-prd.md`` (contract D1, decision D3,
behaviour B9).  ``MergeState`` is the POLL vocabulary (``merge_status`` /
``merge_cancel``'s ``state`` field); ``MergeSubmitStatus`` is the SUBMIT
vocabulary (``merge_request``'s ``status`` field).  They are two distinct
vocabularies, not one — see ``TestVocabulariesAreDistinct``.

Modelled on ``shared/tests/test_task_statuses.py``, including its central
constraint: ``shared/tests/conftest.py`` puts only ``shared/src`` on
``sys.path``, so neither ``orchestrator`` nor ``escalation`` is importable
from this test environment.  The legacy literal sets are therefore INLINED
below with a ``path::symbol`` citation rather than imported, exactly as
``test_task_statuses.py``'s ``_LEGACY_*`` frozensets are.

TDD pair 1: MergeState + poll partitions (GREEN on impl step-2).
TDD pair 2: MergeSubmitStatus + submit partitions (GREEN on impl step-4).
"""
from __future__ import annotations

import ast
import enum
import importlib.util
import sys
from pathlib import Path

from shared.merge_state import (
    CANCEL_STATES,
    EPISTEMIC_STATES,
    LIVE_STATES,
    OUTCOME_STATES,
    POLL_STOP_STATES,
    TERMINAL_STATES,
    MergeState,
)

# Same src-root expression as shared/tests/conftest.py and
# test_pure_stdlib_leaves.py — read the LOCAL tree, never an installed copy.
_SRC = Path(__file__).resolve().parent.parent / 'src'
_MERGE_STATE_SOURCE = _SRC / 'shared' / 'merge_state.py'

# ---------------------------------------------------------------------------
# Legacy literal sets — read off the live producers in this worktree, base
# c84286375c.  These are what the server can emit TODAY; the assertions below
# pin the new vocabulary against them so a divergence is visible here.
# ---------------------------------------------------------------------------

# escalation/src/escalation/server.py::_map_live_state — every value it returns
# for a raw it recognises ('queued' | 'merging'/'awaiting_verify'/'verifying' |
# 'gate_reverify' | 'finalizing').  Its trailing `return raw` passthrough for
# UNRECOGNISED worker states is deliberate fail-open and is not a vocabulary
# member — see the plan's design decision on _map_live_state.
_LEGACY_LIVE = frozenset({'queued', 'verifying', 'gate', 'finalizing'})

# escalation/src/escalation/server.py::_map_terminal_state — every value it
# returns.  NOTE 'already_merged' is deliberately NOT a MergeState member:
# _map_terminal_state collapses it to 'done' (it is a SUBMIT-only value, i.e. a
# MergeSubmitStatus member).  Same for 'done_wip_recovery' -> 'done' and
# 'wip_halted'/'wip_recovery_no_advance'/'unmerged_state'/'unknown_branch'/
# 'error' -> 'blocked'.
_LEGACY_TERMINAL = frozenset({'done', 'conflict', 'blocked', 'abandoned', 'superseded'})

# escalation/src/escalation/server.py::_found_on_main_response — the Tier-3.5
# git-authority response state.
_LEGACY_FOUND_ON_MAIN = 'done'

# escalation/src/escalation/server.py — the Tier-4 "honest unknown" return, and
# merge_cancel's no-live-waiter miss path.
_LEGACY_TIER4 = 'unknown'


# ---------------------------------------------------------------------------
# Pair 1 — MergeState enum (step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestMergeStateEnum:
    def test_is_str_enum(self):
        assert issubclass(MergeState, enum.StrEnum)

    def test_members_are_str_equal(self):
        # StrEnum members are genuine str, so the switchover in
        # escalation/src/escalation/server.py is wire-compatible: existing
        # `resp['state'] == 'done'` assertions keep passing.
        assert MergeState.done == 'done'
        assert isinstance(MergeState.queued, str)

    def test_member_names_are_the_wire_values(self):
        # Unlike task_statuses.TaskStatus (UPPER_CASE names), MergeState's
        # member NAMES are the wire values, matching the PRD's D1 block.
        for member in MergeState:
            assert member.name == member.value, (
                f'MergeState.{member.name} has wire value {member.value!r}; '
                'member names must BE the wire values (PRD D1)'
            )

    def test_exact_vocabulary(self):
        # The 9 outcome states of PRD D1 plus its 4 epistemic states.
        assert {s.value for s in MergeState} == {
            'queued',
            'verifying',
            'gate',
            'finalizing',
            'done',
            'conflict',
            'blocked',
            'abandoned',
            'superseded',
            'no_record',
            'stale_record',
            'journaled',
            'unknown',
        }
        assert len(MergeState) == 13


class TestPollPartitions:
    def test_all_partitions_are_frozensets_of_members(self):
        for name, partition in (
            ('LIVE_STATES', LIVE_STATES),
            ('TERMINAL_STATES', TERMINAL_STATES),
            ('OUTCOME_STATES', OUTCOME_STATES),
            ('EPISTEMIC_STATES', EPISTEMIC_STATES),
            ('POLL_STOP_STATES', POLL_STOP_STATES),
            ('CANCEL_STATES', CANCEL_STATES),
        ):
            assert isinstance(partition, frozenset), f'{name} must be a frozenset'
            assert partition, f'{name} must not be empty'
            for value in partition:
                assert isinstance(value, MergeState), (
                    f'{name} contains {value!r}, which is not a MergeState member'
                )

    def test_live_states(self):
        assert LIVE_STATES == {
            MergeState.queued,
            MergeState.verifying,
            MergeState.gate,
            MergeState.finalizing,
        }

    def test_terminal_states(self):
        assert TERMINAL_STATES == {
            MergeState.done,
            MergeState.conflict,
            MergeState.blocked,
            MergeState.abandoned,
            MergeState.superseded,
        }

    def test_outcome_states(self):
        assert OUTCOME_STATES == LIVE_STATES | TERMINAL_STATES

    def test_epistemic_states(self):
        assert EPISTEMIC_STATES == {
            MergeState.no_record,
            MergeState.stale_record,
            MergeState.journaled,
            MergeState.unknown,
        }

    def test_poll_stop_states(self):
        assert POLL_STOP_STATES == TERMINAL_STATES | {MergeState.unknown}

    def test_cancel_states(self):
        assert CANCEL_STATES == TERMINAL_STATES | {MergeState.unknown}


class TestPartitionsAreExhaustiveAndDisjoint:
    """The B9 drift mechanism.

    A member added to ``MergeState`` without a partition assignment must fail
    HERE, and — because every pinned prose span in the four SKILL.md files is
    checked against a partition by
    ``scripts/tests/test_merge_state_vocabulary_consistency.py`` — assigning it
    to a partition then reddens every span pinned to that partition.  That
    chain is what makes "add a member and the guard fails" true without any
    hand-copied master list.
    """

    def test_outcome_epistemic_exhaustive(self):
        assert OUTCOME_STATES | EPISTEMIC_STATES == frozenset(MergeState), (
            'OUTCOME_STATES | EPISTEMIC_STATES must cover every MergeState member; '
            'unassigned: '
            f'{sorted(frozenset(MergeState) - (OUTCOME_STATES | EPISTEMIC_STATES))}'
        )

    def test_outcome_epistemic_disjoint(self):
        assert OUTCOME_STATES & EPISTEMIC_STATES == frozenset(), (
            'OUTCOME_STATES and EPISTEMIC_STATES must be disjoint; both claim: '
            f'{sorted(OUTCOME_STATES & EPISTEMIC_STATES)}'
        )

    def test_live_terminal_exhaustive_over_outcome(self):
        assert LIVE_STATES | TERMINAL_STATES == OUTCOME_STATES, (
            'LIVE_STATES | TERMINAL_STATES must cover every OUTCOME_STATES member; '
            'unassigned: '
            f'{sorted(OUTCOME_STATES - (LIVE_STATES | TERMINAL_STATES))}'
        )

    def test_live_terminal_disjoint(self):
        assert LIVE_STATES & TERMINAL_STATES == frozenset(), (
            'LIVE_STATES and TERMINAL_STATES must be disjoint; both claim: '
            f'{sorted(LIVE_STATES & TERMINAL_STATES)}'
        )


class TestParityWithTodaysServer:
    """Pin the new vocabulary against what the server emits in this worktree.

    Literals inlined (not imported) because ``escalation`` is not importable
    from this test environment — same constraint and same technique as
    ``test_task_statuses.py``'s ``_LEGACY_*`` sets.
    """

    def test_live_states_match_map_live_state(self):
        # escalation/src/escalation/server.py::_map_live_state
        assert {s.value for s in LIVE_STATES} == _LEGACY_LIVE

    def test_terminal_states_match_map_terminal_state(self):
        # escalation/src/escalation/server.py::_map_terminal_state
        assert {s.value for s in TERMINAL_STATES} == _LEGACY_TERMINAL

    def test_found_on_main_is_done(self):
        # escalation/src/escalation/server.py::_found_on_main_response
        assert MergeState.done == _LEGACY_FOUND_ON_MAIN

    def test_tier4_is_unknown(self):
        # escalation/src/escalation/server.py — Tier 4 "honest unknown"
        assert MergeState.unknown == _LEGACY_TIER4

    def test_already_merged_is_not_a_poll_state(self):
        # _map_terminal_state collapses 'already_merged' to 'done', so it is a
        # SUBMIT-only value.  Asserted explicitly because three runbooks
        # currently name it in a POLL terminal set (that is DEFECT 2).
        assert 'already_merged' not in {s.value for s in MergeState}

    def test_epistemic_members_are_not_emitted_yet(self):
        # PRD D1's three new epistemic members land in the enum HERE (alpha)
        # but nothing emits them until beta.  Pinning that they are absent
        # from today's server output keeps this file honest about which
        # members are contract-only.
        not_yet_emitted = {MergeState.no_record, MergeState.stale_record, MergeState.journaled}
        assert not_yet_emitted <= EPISTEMIC_STATES
        assert {s.value for s in not_yet_emitted} & (_LEGACY_LIVE | _LEGACY_TERMINAL) == set()


class TestStandaloneLoadability:
    """The contract ``scripts/tests/`` depends on.

    ``scripts/tests/test_merge_state_vocabulary_consistency.py`` loads this
    module BY ABSOLUTE FILE PATH (``importlib.util.spec_from_file_location``),
    never via ``import shared.merge_state`` — see that guard's docstring for
    both reasons.  A file-path load executes the module OUTSIDE its package, so
    any intra-package or third-party import would break the guard.  These tests
    pin that property so it cannot regress silently.
    """

    def test_source_file_exists(self):
        assert _MERGE_STATE_SOURCE.is_file(), f'missing: {_MERGE_STATE_SOURCE}'

    def test_loads_by_file_path_outside_its_package(self):
        spec = importlib.util.spec_from_file_location(
            '_merge_state_standalone_probe', _MERGE_STATE_SOURCE
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name in (
            'MergeState',
            'LIVE_STATES',
            'TERMINAL_STATES',
            'OUTCOME_STATES',
            'EPISTEMIC_STATES',
            'POLL_STOP_STATES',
            'CANCEL_STATES',
        ):
            assert hasattr(module, name), (
                f'{name} does not resolve on a file-path load of {_MERGE_STATE_SOURCE}'
            )
        assert {s.value for s in module.MergeState} == {s.value for s in MergeState}

    def test_top_level_imports_are_stdlib_only(self):
        tree = ast.parse(_MERGE_STATE_SOURCE.read_text())
        offenders: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # a relative import cannot survive a file-path load
                    offenders.append('.' * node.level + (node.module or ''))
                    continue
                roots = [(node.module or '').split('.')[0]]
            else:
                continue
            offenders.extend(r for r in roots if r and r not in sys.stdlib_module_names)

        assert offenders == [], (
            f'{_MERGE_STATE_SOURCE.name} must import only stdlib at module level so the '
            'scripts/tests/ drift guard can load it by file path; found: '
            f'{sorted(set(offenders))}'
        )
