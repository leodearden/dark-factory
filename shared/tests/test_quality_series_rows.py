"""shared.quality_series.rows: one LandingRow per landing, attributed over (previous landing, landing]."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from quality_series_fixtures import LandingScenario, at_hour, build_landing_scenario

from shared.quality_series import LandingRow, landings, rows_for, unit_heads
from shared.quality_series.rows import SourceEpochs

POST_MERGE = 'post_merge_verify'


def _rows(scenario: LandingScenario) -> list[LandingRow]:
    batch = landings(scenario.db.conn, scenario.repo.root, since_cursor=None)
    return rows_for(batch.landings, scenario.db.conn, scenario.repo.root, git_budget_s=60)


def _by_key(rows: list[LandingRow]) -> dict[tuple[str, str], LandingRow]:
    return {(row.landing.task_id, row.landing.merge_sha): row for row in rows}


@pytest.fixture(scope='module')
def outcomes_scenario(tmp_path_factory: pytest.TempPathFactory) -> LandingScenario:
    """The step-11 history plus outcome events; read-only for every test that uses it."""
    s = build_landing_scenario(tmp_path_factory.mktemp('rows') / 'proj')
    db = s.db
    db.merge_blocked(at_hour(10.0), '101', POST_MERGE, 'Post-merge verification failed: a')
    db.merge_blocked(at_hour(10.9), '102', POST_MERGE, 'Post-merge verification failed: b')
    db.merge_blocked(at_hour(10.95), '102', 'merge_ff_failed', 'Post-merge verification failed: c')
    db.merge_blocked(at_hour(11.5), '102', POST_MERGE, 'Post-merge verification failed: late')
    db.merge_blocked(at_hour(11.8), '104', POST_MERGE, 'Post-merge verification failed: d')

    db.merge_verify(at_hour(9.0), '101', s.m1, passed=True)
    db.merge_verify(at_hour(11.2), '103', 'f' * 40, passed=True)
    db.merge_verify(at_hour(11.7), '105', s.m3, passed=False)
    db.merge_verify(at_hour(11.9), '105', s.m3, passed=True)
    db.merge_verify(at_hour(12.0), '101', s.m6, passed=True)

    db.flake(at_hour(11.6), '104', 'passes_in_isolation')
    db.flake(at_hour(12.5), '101', 'fails_in_isolation')
    db.flake(at_hour(13.5), '107', 'unconfirmable')

    db.review_completed(at_hour(10.8), '102', cycle=1, has_blocking=True)
    db.review_completed(at_hour(10.85), '102', cycle=2, has_blocking=False)
    db.review_completed(at_hour(11.5), '104', cycle=1, has_blocking=False)
    return s


@pytest.fixture(scope='module')
def outcome_rows(outcomes_scenario: LandingScenario) -> dict[tuple[str, str], LandingRow]:
    return _by_key(_rows(outcomes_scenario))


class TestRejections:
    def test_post_merge_category_in_the_window_counts(self, outcomes_scenario, outcome_rows):
        f2 = outcome_rows[('102', outcomes_scenario.f2)].outcomes
        assert (f2.rejected, f2.rejection_events) == (True, 1)

    def test_events_at_the_previous_landing_belong_to_it(self, outcomes_scenario, outcome_rows):
        s = outcomes_scenario
        assert outcome_rows[('101', s.m1)].outcomes.rejection_events == 1
        m6 = outcome_rows[('101', s.m6)].outcomes
        assert (m6.rejected, m6.rejection_events) == (False, 0)

    def test_a_train_rejection_is_the_units_on_every_member(self, outcomes_scenario, outcome_rows):
        for task in ('103', '104', '105'):
            member = outcome_rows[(task, outcomes_scenario.m3)].outcomes
            assert (member.rejected, member.rejection_events) == (True, 1)

    def test_a_landing_before_the_first_rejection_event_is_not_started(
        self, tmp_path: Path
    ) -> None:
        s = build_landing_scenario(tmp_path / 'proj')
        s.db.merge_blocked(at_hour(10.5), '102', POST_MERGE, 'Post-merge verification failed: e')
        m1 = _by_key(_rows(s))[('101', s.m1)].outcomes
        assert (m1.rejected, m1.rejection_events) == (None, None)


class TestVerifies:
    def test_a_trains_verify_rows_combine_across_members(self, outcomes_scenario, outcome_rows):
        for task in ('103', '104', '105'):
            member = outcome_rows[(task, outcomes_scenario.m3)].outcomes
            assert (member.reverify_count, member.verify_failed_rows) == (2, 1)

    def test_a_single_verify_is_no_reverify(self, outcomes_scenario, outcome_rows):
        m6 = outcome_rows[('101', outcomes_scenario.m6)].outcomes
        assert (m6.reverify_count, m6.verify_failed_rows) == (0, 0)


class TestFlakes:
    def test_a_confirmed_flake_of_one_member_marks_the_unit(self, outcomes_scenario, outcome_rows):
        for task in ('103', '104', '105'):
            member = outcome_rows[(task, outcomes_scenario.m3)].outcomes
            assert member.flake_confirmed is True
            assert member.flake_rows == 1

    def test_a_real_red_is_not_a_flake(self, outcomes_scenario, outcome_rows):
        m6 = outcome_rows[('101', outcomes_scenario.m6)].outcomes
        assert (m6.flake_real_red, m6.flake_confirmed) == (True, False)

    def test_an_unconfirmable_row_counts_but_flags_nothing(self, outcomes_scenario, outcome_rows):
        m7 = outcome_rows[('107', outcomes_scenario.m7)].outcomes
        assert (m7.flake_confirmed, m7.flake_real_red, m7.flake_rows) == (False, False, 1)

    def test_landings_before_the_first_flake_row_are_not_started(
        self, outcomes_scenario, outcome_rows
    ):
        m1 = outcome_rows[('101', outcomes_scenario.m1)].outcomes
        assert (m1.flake_confirmed, m1.flake_real_red, m1.flake_rows) == (None, None, None)


class TestReviews:
    def test_a_blocking_review_in_the_window(self, outcomes_scenario, outcome_rows):
        f2 = outcome_rows[('102', outcomes_scenario.f2)].outcomes
        assert (f2.reviewed, f2.review_blocking) == (True, True)

    def test_only_clean_reviews_are_not_blocking(self, outcomes_scenario, outcome_rows):
        member = outcome_rows[('104', outcomes_scenario.m3)].outcomes
        assert (member.reviewed, member.review_blocking) == (True, False)

    def test_reviews_stay_per_landing_inside_a_train(self, outcomes_scenario, outcome_rows):
        member = outcome_rows[('103', outcomes_scenario.m3)].outcomes
        assert (member.reviewed, member.review_blocking) == (False, None)

    def test_after_the_epoch_without_a_review_is_unreviewed(self, outcomes_scenario, outcome_rows):
        m6 = outcome_rows[('101', outcomes_scenario.m6)].outcomes
        assert (m6.reviewed, m6.review_blocking) == (False, None)

    def test_before_the_epoch_is_not_started(self, outcomes_scenario, outcome_rows):
        m1 = outcome_rows[('101', outcomes_scenario.m1)].outcomes
        assert (m1.reviewed, m1.review_blocking) == (None, None)


def test_rule_vintage_is_the_first_row_of_each_source(outcomes_scenario, outcome_rows):
    expected = SourceEpochs(
        rejections=at_hour(10.0).isoformat(),
        merge_verify=at_hour(9.0).isoformat(),
        review_completed=at_hour(10.8).isoformat(),
        flake_occurrence=at_hour(11.6).isoformat(),
        invocation_turns=None,
        invocations=None,
    )
    assert {row.rule_vintage for row in outcome_rows.values()} == {expected}


def test_unit_heads_keep_one_row_per_unit_at_its_lowest_ordinal(outcomes_scenario):
    rows = _rows(outcomes_scenario)
    heads = unit_heads(rows)
    assert len(heads) == len({row.landing.unit_id for row in rows})
    train = [head for head in heads if head.landing.unit_id == outcomes_scenario.m3]
    assert [head.landing.task_id for head in train] == ['103']


def test_a_landing_row_is_frozen(outcome_rows):
    row = next(iter(outcome_rows.values()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.outcomes = row.outcomes  # type: ignore[misc]
