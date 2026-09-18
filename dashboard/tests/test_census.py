"""Tests for dashboard.data.census — the view vocabulary and the task census.

Pins the contract declared in ``plans/dashboard-one-datum-one-path-prd.md``
("The task census", decision 3). Every expectation is DERIVED from
``shared.task_statuses`` rather than restated here — a second hand-written
copy of the vocabulary is exactly the drift this module exists to remove. The
sole exceptions are ``REVIEW`` and ``INFRA_HOLD``, named explicitly below
because their in-flight membership is the PRD's decision-3 amendment and
therefore the claim a future reader is most likely to doubt.
"""

from __future__ import annotations

import enum

import pytest
from shared.task_statuses import TERMINAL, TaskStatus

from dashboard.data.census import SUB_VIEWS, TONES, VIEWS, TaskView, build_census


def test_task_view_is_a_str_enum():
    """View members are genuine strings, so they key the wire without conversion."""
    assert issubclass(TaskView, enum.StrEnum)


def test_task_view_vocabulary_is_exactly_the_contract_four():
    """Three partition views plus one sub-view; a fifth would change the contract."""
    assert {member.value for member in TaskView} == {
        'in_flight',
        'backlog',
        'terminal',
        'running',
    }


def test_views_and_sub_views_split_the_view_vocabulary():
    """Every TaskView is keyed by exactly one of the two constants."""
    assert set(VIEWS).isdisjoint(SUB_VIEWS)
    assert set(VIEWS) | set(SUB_VIEWS) == set(TaskView)


def test_views_partition_the_task_statuses():
    """The three views are pairwise disjoint and cover every TaskStatus member."""
    members = list(VIEWS.values())
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            assert left.isdisjoint(right)
    assert set().union(*members) == set(TaskStatus)


def test_terminal_view_reuses_the_shared_partition():
    """`terminal` is bound to shared's TERMINAL by reference, not restated."""
    assert VIEWS[TaskView.TERMINAL] == TERMINAL


def test_running_is_a_strict_subset_of_in_flight():
    """`running` narrows in_flight; it is a sub-view, never a fourth partition cell."""
    running = SUB_VIEWS[TaskView.RUNNING]
    assert running == {TaskStatus.IN_PROGRESS}
    assert running < VIEWS[TaskView.IN_FLIGHT]


def test_review_and_infra_hold_are_in_flight():
    """The PRD's decision-3 amendment: both are dispatched work, not backlog."""
    in_flight = VIEWS[TaskView.IN_FLIGHT]
    assert TaskStatus.REVIEW in in_flight
    assert TaskStatus.INFRA_HOLD in in_flight


def test_tones_cover_every_status_exactly_once():
    """Every status draws in some tone; a new member with no tone fails here."""
    assert set(TONES) == set(TaskStatus)
    assert all(TONES[member] for member in TaskStatus)


@pytest.mark.parametrize('constant', [VIEWS, SUB_VIEWS, TONES], ids=['VIEWS', 'SUB_VIEWS', 'TONES'])
def test_vocabulary_constants_reject_mutation(constant):
    """These are imported by beta, the generator and the parity test — SPOT."""
    with pytest.raises(TypeError):
        constant['whatever'] = 'anything'  # type: ignore[index]


# ---------------------------------------------------------------------------
# build_census over a nine-member fixture (the PRD's boundary sketch #5 —
# synthetic, because no live `review`/`infra-hold` rows exist today). Keys are
# ints and values plain strings, matching what tasks.py::fetch_statuses hands
# back.
# ---------------------------------------------------------------------------

NINE_MEMBER_MAP = {index: member.value for index, member in enumerate(TaskStatus)}


def assert_views_sum_to_total(census):
    """The three views partition the statuses, so they must sum to the total."""
    assert sum(census.views.values()) == census.total
    assert sum(census.counts.values()) == census.total
    assert census.sub_views[TaskView.RUNNING] <= census.views[TaskView.IN_FLIGHT]


def test_build_census_counts_each_member_of_a_nine_member_map_once():
    """One id per status: every count is 1 and the total is nine."""
    census = build_census(NINE_MEMBER_MAP)
    assert set(census.counts) == set(TaskStatus)
    assert all(count == 1 for count in census.counts.values())
    assert census.total == 9 == sum(census.counts.values())


def test_build_census_views_over_a_nine_member_map():
    """Five statuses are in flight, two are backlog, two are terminal."""
    census = build_census(NINE_MEMBER_MAP)
    assert dict(census.views) == {
        TaskView.IN_FLIGHT: 5,
        TaskView.BACKLOG: 2,
        TaskView.TERMINAL: 2,
    }
    assert dict(census.sub_views) == {TaskView.RUNNING: 1}
    assert_views_sum_to_total(census)


def test_build_census_over_an_unbalanced_map():
    """Several ids sharing a status tally into that status, and the sums still hold."""
    status_map = {
        1: TaskStatus.IN_PROGRESS.value,
        2: TaskStatus.IN_PROGRESS.value,
        3: TaskStatus.IN_PROGRESS.value,
        4: TaskStatus.PENDING.value,
        5: TaskStatus.DONE.value,
        6: TaskStatus.DONE.value,
    }
    census = build_census(status_map)
    assert census.counts[TaskStatus.IN_PROGRESS] == 3
    assert census.counts[TaskStatus.DONE] == 2
    assert census.counts[TaskStatus.BLOCKED] == 0
    assert census.total == 6
    assert dict(census.views) == {
        TaskView.IN_FLIGHT: 3,
        TaskView.BACKLOG: 1,
        TaskView.TERMINAL: 2,
    }
    assert_views_sum_to_total(census)


def test_build_census_over_an_empty_map_reports_zeroes_not_absences():
    """All nine keys are present at zero — a missing key would read as a gap."""
    census = build_census({})
    assert set(census.counts) == set(TaskStatus)
    assert all(count == 0 for count in census.counts.values())
    assert census.total == 0
    assert all(count == 0 for count in census.views.values())
    assert all(count == 0 for count in census.sub_views.values())
    assert_views_sum_to_total(census)


def test_build_census_keeps_an_absent_member_at_zero():
    """A status nobody is in is still a key, so a consumer never sees a KeyError."""
    census = build_census({1: TaskStatus.PENDING.value})
    assert census.counts[TaskStatus.INFRA_HOLD] == 0
    assert census.counts[TaskStatus.PENDING] == 1
    assert census.total == 1
    assert_views_sum_to_total(census)
