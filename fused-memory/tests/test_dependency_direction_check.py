"""Post-write sanity check for Graphiti dependency-direction extraction (task 3770).

Graphiti's extraction LLM does not hallucinate task numbers here — every task
it names is real, and adjacent to the ones it is genuinely about. What it gets
wrong is the DIRECTION: it FLATTENS sibling/parallel relations into sequential
ones ("A waits behind B" for two tasks that are merely both prerequisites of C)
and INVERTS transitive chains ("A waits behind B" where B in fact reaches A).
Because the numbers are plausible and adjacent, a planning read accepts the
resulting fact without friction — which makes this strictly more dangerous than
random hallucination, and is why the check exists.

The decision core is pure: the ground-truth graph is passed IN as data, so
every classification below is exercised against a plain dict with no Taskmaster,
no Graphiti and no I/O.
"""

from __future__ import annotations

import dataclasses

import pytest

from fused_memory.middleware.dependency_direction_check import (
    DependencyAssertion,
    extract_dependency_assertions,
)

# ── The frozen ground-truth fixture ────────────────────────────────────────
#
# PROVENANCE. Read READ-ONLY from `/home/leo/src/dark-factory/.taskmaster/
# tasks/tasks.db` (table `dependencies(tag, task_id, depends_on)`, tag
# 'master') during task-3770 planning. These are measured values, NOT invented
# ones. Note the DB path: `.taskmaster/tasks.db` is a 0-byte DECOY and is not
# the live database — the live one is `.taskmaster/tasks/tasks.db`.
#
# WHY IT IS FROZEN, and deliberately NOT refreshed from live. The constant
# reproduces the graph AS OF the bad extraction (2026-08-05). The LIVE graph has
# since DRIFTED, in two ways that both matter:
#
#   1. Task 3578 has gained the dependencies {3983, 4005}.
#   2. Task 5020 now depends on BOTH 3730 AND 3733 — so today those two share a
#      DEPENDENT, which they did NOT at extraction time.
#
# Drift (2) is the load-bearing one. In this frozen shape 3730/3733 share only
# DEPENDENCIES, which is the only reason the shared-dependency arm of the
# sibling predicate gets exercised at all. Refreshing against live would let
# that arm go untested, and the predicate could silently narrow back to the
# escalation's literal "two tasks that share a dependent" one-liner with no test
# failing. Independently, a live read would make the leaf signal
# non-reproducible: this suite must fail before the change and pass after, and a
# fixture that mutates under operator activity guarantees neither.
#
# A future reader who diffs this constant against live data will find the
# discrepancy already explained here rather than "correcting" it.
#
# DERIVED CLOSURES (recorded so expectations can be re-verified without
# re-deriving them by hand):
#   closure(3578) = {3256, 3619, 3727, 3618}
#   closure(3730) = {3578, 3727, 3728, 3256, 3619, 3618}
#   closure(3733) = {3578, 3728, 3256, 3727, 3619, 3618}
#   closure(3619) = {3256, 3618}
#   closure(3618) = {}  (a leaf: 3618 depends on nothing)
LIVE_SHAPE_EDGES: dict[int, list[int]] = {
    3256: [],
    3618: [],
    3619: [3256, 3618],
    3727: [3256],
    3728: [3256, 3727],
    3578: [3256, 3619, 3727],
    3730: [3578, 3727, 3728],
    3733: [3578, 3728],
}


# ── extract_dependency_assertions — the scope gate plus parser ─────────────


class TestExtractDependencyAssertions:
    """The SCOPE GATE. Only compact multi-task dependency shorthand is in scope;
    everything else must return `[]` without the caller ever touching Taskmaster.
    """

    def test_forward_phrase_binds_dependent_then_dependency(self):
        assertions = extract_dependency_assertions(
            'Task 3727 waits behind task 3619'
        )
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]
        assert 'waits behind' in assertions[0].phrase.lower()

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 3727 needs task 3619',
            'Task 3727 depends on task 3619',
            'Task 3727 is blocked by task 3619',
            'Task 3727 is blocked on task 3619',
            'Task 3727 is waiting on task 3619',
        ],
    )
    def test_other_forward_phrasings_bind_the_same_direction(self, fact):
        assertions = extract_dependency_assertions(fact)
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 3619 blocks task 3727',
            'Task 3619 gates task 3727',
            'Task 3619 unblocks task 3727',
            'Task 3619 is a dependency of task 3727',
        ],
    )
    def test_inverse_phrase_yields_the_swapped_pair(self, fact):
        """Direction comes from the PHRASE, never from word order."""
        assertions = extract_dependency_assertions(fact)
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]

    def test_multi_clause_chain_yields_both_pairs(self):
        assertions = extract_dependency_assertions(
            'task 3727 waits behind task 3619, task 3730 needs task 3733'
        )
        assert [(a.dependent, a.dependency) for a in assertions] == [
            (3727, 3619),
            (3730, 3733),
        ]

    @pytest.mark.parametrize(
        'fact',
        [
            # No direction-bearing phrase at all.
            'Task 3727 and task 3619 were both filed today',
            # Only one task reference.
            'Task 3727 depends on the merge queue draining',
            # A dependency phrase but no task references.
            'The parser depends on the extraction stage running first',
            # The phrase and the two refs sit in DIFFERENT clauses: no
            # cross-clause binding.
            'Task 3727 waits behind. Task 3619 was filed',
            'Task 3727 waits behind; task 3619 landed',
        ],
    )
    def test_out_of_scope_returns_empty(self, fact):
        assert extract_dependency_assertions(fact) == []

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 3727 needs task 3619',
            'task #3727 needs task #3619',
            'task/3727 needs task/3619',
            '#3727 needs #3619',
        ],
    )
    def test_accepts_the_inherited_task_ref_spellings(self, fact):
        """The gate must not be narrower than TASK_REF_RE's canonical vocabulary."""
        assertions = extract_dependency_assertions(fact)
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]

    def test_assertion_record_is_frozen(self):
        assertion = extract_dependency_assertions(
            'Task 3727 waits behind task 3619'
        )[0]
        assert isinstance(assertion, DependencyAssertion)
        with pytest.raises(dataclasses.FrozenInstanceError):
            assertion.dependent = 1  # type: ignore[misc]

    @pytest.mark.parametrize('fact', [None, '', '   ', 12345, object()])
    def test_never_raises_on_a_non_string_or_empty_fact(self, fact):
        assert extract_dependency_assertions(fact) == []
