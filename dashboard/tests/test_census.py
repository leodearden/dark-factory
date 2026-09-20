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
import json
from types import MappingProxyType

import pytest
from shared.task_statuses import TERMINAL, TaskStatus

from dashboard.data.census import (
    SUB_VIEWS,
    TONES,
    VIEWS,
    CensusVocabularyError,
    TaskView,
    build_census,
)


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


# ---------------------------------------------------------------------------
# The vocabulary boundary. An off-vocabulary status means the vocabulary
# drifted, so build_census refuses loudly rather than dropping the row (which
# would under-report `total`) or opening a tenth bucket (which would break the
# nine-key contract every consumer is written against).
# ---------------------------------------------------------------------------


def test_build_census_rejects_an_off_vocabulary_status():
    """The message names the offending value AND the id that carried it."""
    with pytest.raises(CensusVocabularyError) as excinfo:
        build_census({7: TaskStatus.IN_PROGRESS.value, 9: 'archived'})
    message = str(excinfo.value)
    assert repr('archived') in message
    assert repr(9) in message


def test_build_census_vocabulary_error_names_the_legal_members():
    """An operator sees what WAS legal without going to read the enum."""
    with pytest.raises(CensusVocabularyError) as excinfo:
        build_census({1: 'archived'})
    message = str(excinfo.value)
    assert all(repr(member.value) in message for member in TaskStatus)


def test_build_census_reports_every_offending_value_at_once():
    """One raise names every distinct offender, so a fix is not found one rerun at a time."""
    with pytest.raises(CensusVocabularyError) as excinfo:
        build_census({1: 'archived', 2: 'retired', 3: TaskStatus.DONE.value})
    message = str(excinfo.value)
    assert repr('archived') in message
    assert repr('retired') in message


# The message crosses the wire verbatim as a Datum's `reason` on every poll of
# every project, so its SIZE is part of the contract. Both drifts below make
# EVERY row an offender at once, which is what the realistic ones do: a tenth
# status shipping upstream, or fetch_statuses changing the shape of its values.
# A per-row enumeration of either fixture would run past 100_000 characters.
MESSAGE_CEILING = 2000


def test_build_census_vocabulary_error_stays_bounded_when_one_value_drifts():
    """A thousand rows sharing one unknown status carry one bit of information."""
    with pytest.raises(CensusVocabularyError) as excinfo:
        build_census(dict.fromkeys(range(5000), 'archived'))
    message = str(excinfo.value)
    assert repr('archived') in message
    assert '5000' in message
    assert len(message) < MESSAGE_CEILING


def test_build_census_vocabulary_error_stays_bounded_when_the_value_shape_drifts():
    """Every row a DISTINCT offender — the enumeration is capped, the count is not."""
    drifted = {index: f'unknown-{index}' for index in range(5000)}
    with pytest.raises(CensusVocabularyError) as excinfo:
        build_census(drifted)
    message = str(excinfo.value)
    assert '5000' in message
    assert len(message) < MESSAGE_CEILING


def test_build_census_vocabulary_error_survives_an_unhashable_value():
    """The shape drift it must report is exactly the one that resists grouping."""
    with pytest.raises(CensusVocabularyError) as excinfo:
        build_census({1: {'status': 'done'}})  # type: ignore[dict-item]
    assert repr('status') in str(excinfo.value)


def test_build_census_does_not_mutate_the_mapping_it_was_handed():
    """Purity: the caller's map is an input, never scratch space."""
    status_map = dict(NINE_MEMBER_MAP)
    before = dict(status_map)
    build_census(status_map)
    assert status_map == before


def test_build_census_accepts_an_immutable_mapping():
    """A MappingProxyType input is read, never written, so it is accepted."""
    census = build_census(MappingProxyType(dict(NINE_MEMBER_MAP)))
    assert census.total == 9


# ---------------------------------------------------------------------------
# The census wire shape. Plain strings throughout, so a consumer reads the
# payload without the Python enums and json.dumps needs no custom encoder.
# ---------------------------------------------------------------------------


def wire_section(wire, key) -> dict[str, int]:
    """The *key* section of a census wire dict, narrowed for indexing.

    ``to_wire()`` is honestly typed ``dict[str, object]`` — its values are
    heterogeneous — so a test that reaches into a section narrows it here
    rather than the module weakening its own annotation.
    """
    section = wire[key]
    assert isinstance(section, dict)
    return section


def test_to_wire_emits_exactly_the_four_contract_keys():
    """The wire shape is closed: counts, total, views, sub_views."""
    wire = build_census(NINE_MEMBER_MAP).to_wire()
    assert set(wire) == {'counts', 'total', 'views', 'sub_views'}


def test_to_wire_keys_counts_by_the_plain_status_strings():
    """`'in-progress'`/`'infra-hold'` as written, not enum objects."""
    wire = build_census(NINE_MEMBER_MAP).to_wire()
    counts = wire_section(wire, 'counts')
    assert set(counts) == {member.value for member in TaskStatus}
    assert all(type(key) is str for key in counts)


def test_to_wire_keys_views_by_the_plain_view_strings():
    """The view keys cross the wire as the strings the SPA reads."""
    wire = build_census(NINE_MEMBER_MAP).to_wire()
    views = wire_section(wire, 'views')
    assert set(views) == {'in_flight', 'backlog', 'terminal'}
    assert set(wire_section(wire, 'sub_views')) == {'running'}
    assert all(type(key) is str for key in views)


def test_to_wire_is_json_serialisable_without_a_custom_encoder():
    """A round-trip through json pins that nothing enum-shaped survives to the wire."""
    wire = build_census(NINE_MEMBER_MAP).to_wire()
    assert json.loads(json.dumps(wire)) == wire


def test_to_wire_parts_sum_to_the_whole():
    """The PRD's boundary sketch #4, asserted on the wire the SPA actually reads."""
    wire = build_census(NINE_MEMBER_MAP).to_wire()
    counts_total = sum(wire_section(wire, 'counts').values())
    assert counts_total == wire['total'] == sum(wire_section(wire, 'views').values())


def test_to_wire_never_puts_running_in_the_partition():
    """`running` is a sub-view; a fourth `views` entry would break the sum above."""
    wire = build_census(NINE_MEMBER_MAP).to_wire()
    assert 'running' not in wire_section(wire, 'views')
    assert wire_section(wire, 'sub_views')['running'] == 1
