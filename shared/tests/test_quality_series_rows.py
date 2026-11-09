"""shared.quality_series.rows: one LandingRow per landing, attributed over (previous landing, landing]."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from quality_series_fixtures import (
    LandingScenario,
    LateClaimScenario,
    at_hour,
    build_landing_scenario,
    build_late_claim_scenario,
)

from shared.cost_store import CapReason
from shared.quality_series import Landing, LandingRow, landings, rows_for, unit_heads
from shared.quality_series.git_lines import landed_lines
from shared.quality_series.rows import (
    TURNS_CAP_REASON,
    CharterClass,
    CharterClassSource,
    QualityRowInvariantError,
    ReviewRoundsSource,
    SourceEpochs,
    UnknownCharterClass,
    module_allocation,
)

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

    def test_a_store_without_a_rejection_reads_zero_after_its_first_categorised_block(
        self, tmp_path: Path
    ) -> None:
        s = build_landing_scenario(tmp_path / 'proj')
        s.db.merge_blocked(at_hour(9.0), '101', 'merge_ff_failed', 'fast-forward failed')
        m1 = _by_key(_rows(s))[('101', s.m1)]
        assert (m1.outcomes.rejected, m1.outcomes.rejection_events) == (False, 0)
        assert m1.rule_vintage.rejections == at_hour(9.0)


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
        rejections=at_hour(10.0),
        merge_verify=at_hour(9.0),
        review_completed=at_hour(10.8),
        flake_occurrence=at_hour(11.6),
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


@pytest.fixture(scope='module')
def cost_scenario(tmp_path_factory: pytest.TempPathFactory) -> LandingScenario:
    """The step-11 history plus invocations, invocation_end turns and task_results rows."""
    s = build_landing_scenario(tmp_path_factory.mktemp('cost') / 'proj')
    db = s.db
    db.invocation('101', 'architect', 1.0, at_hour(8.0))
    db.invocation('101', 'implementer', 2.0, at_hour(8.5))
    db.invocation('101', 'implementer', 2.0, at_hour(9.0))
    db.invocation('101', 'reviewer_comprehensive', 0.5, at_hour(9.5))
    db.invocation('101', 'implementer', 3.0, at_hour(10.5))
    db.invocation('102', 'simple_task', 1.5, at_hour(10.2))
    db.invocation('107', 'architect', 0.7, at_hour(13.2))

    db.invocation_end(at_hour(8.6), '101', 'implementer', 30)
    db.invocation_end(at_hour(9.1), '101', 'implementer', 50, capped_reason='turns')
    db.invocation_end(at_hour(9.6), '101', 'reviewer_comprehensive', 99)
    db.invocation_end(at_hour(10.25), '102', 'simple_task', 12)

    db.task_result('101', 'run-a', 'requeued', at_hour(9.2), execute_iterations=1)
    db.task_result('101', 'run-b', 'blocked', at_hour(9.8), execute_iterations=2)
    db.task_result('101', 'run-c', 'done', at_hour(10.4), execute_iterations=3)
    db.task_result('101', 'run-d', 'cancelled', '', execute_iterations=5)
    db.task_result('101', 'run-e', 'done', at_hour(13.05), execute_iterations=4)
    db.task_result('102', 'run-a', 'merge-deferred', at_hour(10.9), execute_iterations=2)
    db.task_result('102', 'run-b', 'requeued', at_hour(11.2), execute_iterations=7)
    return s


@pytest.fixture(scope='module')
def cost_rows(cost_scenario: LandingScenario) -> dict[tuple[str, str], LandingRow]:
    return _by_key(_rows(cost_scenario))


class TestInvocationCost:
    def test_all_roles_by_role_and_implementer_invocations(self, cost_scenario, cost_rows):
        cost = cost_rows[('101', cost_scenario.m1)].cost
        assert cost.all_role_usd == pytest.approx(5.5)
        assert cost.by_role == (
            ('architect', 1.0), ('implementer', 4.0), ('reviewer_comprehensive', 0.5),
        )
        assert cost.implementer_invocations == 2

    def test_a_second_landing_counts_only_what_started_after_the_first(
        self, cost_scenario, cost_rows
    ):
        cost = cost_rows[('101', cost_scenario.m6)].cost
        assert cost.all_role_usd == pytest.approx(3.0)
        assert cost.by_role == (('implementer', 3.0),)
        assert cost.implementer_invocations == 1

    def test_simple_task_is_implementer_class(self, cost_scenario, cost_rows):
        assert cost_rows[('102', cost_scenario.f2)].cost.implementer_invocations == 1

    def test_only_an_architect_invocation_is_zero_implementer_invocations(
        self, cost_scenario, cost_rows
    ):
        cost = cost_rows[('107', cost_scenario.m7)].cost
        assert (cost.all_role_usd, cost.implementer_invocations) == (pytest.approx(0.7), 0)

    def test_no_invocation_at_all_is_unknown_cost(self, cost_scenario, cost_rows):
        cost = cost_rows[('108', cost_scenario.m8)].cost
        assert (cost.all_role_usd, cost.implementer_invocations, cost.by_role) == (None, None, ())


class TestTurns:
    def test_implementer_turns_sum_and_a_cap_is_flagged(self, cost_scenario, cost_rows):
        cost = cost_rows[('101', cost_scenario.m1)].cost
        assert (cost.turns, cost.turns_capped) == (80, True)

    def test_simple_task_turns_count(self, cost_scenario, cost_rows):
        cost = cost_rows[('102', cost_scenario.f2)].cost
        assert (cost.turns, cost.turns_capped) == (12, False)

    def test_no_implementer_invocation_end_is_unknown_turns(self, cost_scenario, cost_rows):
        cost = cost_rows[('107', cost_scenario.m7)].cost
        assert (cost.turns, cost.turns_capped) == (None, False)

    def test_before_the_first_turns_bearing_row_turns_are_unknown_not_zero(
        self, tmp_path: Path
    ) -> None:
        s = build_landing_scenario(tmp_path / 'proj')
        s.db.invocation_end(at_hour(9.0), '101', 'implementer', None)
        s.db.invocation_end(at_hour(12.5), '101', 'implementer', 7)
        rows = _by_key(_rows(s))
        assert rows[('101', s.m1)].cost.turns is None
        assert rows[('101', s.m6)].cost.turns == 7

    def test_the_cap_reason_is_the_cost_store_vocabulary(self) -> None:
        assert TURNS_CAP_REASON == CapReason.TURNS


class TestExecuteIterations:
    def test_the_window_rows_and_the_landing_dispatchs_own_row_written_after_it(
        self, cost_scenario, cost_rows
    ):
        assert cost_rows[('101', cost_scenario.m1)].cost.execute_iterations == 1 + 2 + 3

    def test_the_first_landings_dispatch_row_is_not_charged_to_the_second(
        self, cost_scenario, cost_rows
    ):
        assert cost_rows[('101', cost_scenario.m6)].cost.execute_iterations == 4

    def test_a_row_after_the_landing_counts_only_when_it_reports_done(
        self, cost_scenario, cost_rows
    ):
        assert cost_rows[('102', cost_scenario.f2)].cost.execute_iterations == 2

    def test_no_rows_is_unknown(self, cost_scenario, cost_rows):
        assert cost_rows[('107', cost_scenario.m7)].cost.execute_iterations is None


def _review_state(root: Path, dirname: str, content: str | bytes) -> None:
    meta = root / '.worktrees' / '.task-meta' / dirname
    meta.mkdir(parents=True, exist_ok=True)
    state = meta / 'review_state.json'
    if isinstance(content, bytes):
        state.write_bytes(content)
    else:
        state.write_text(content)


UNACTIONABLE = 'Architect reported task unactionable: x'
FALSE_PREMISE = 'Architect reported false RED-test premise: y'
ALREADY_DONE = 'Architect reported task already done at abc123 but the claim could not be accepted'
PLAN_FILES = 'Plan files not touched by branch: a.py'


@pytest.fixture(scope='module')
def review_scenario(tmp_path_factory: pytest.TempPathFactory) -> LandingScenario:
    """Review events start at 13:30, after every landing but m7/m8; plus charter block rows."""
    s = build_landing_scenario(tmp_path_factory.mktemp('review') / 'proj')
    db, root = s.db, s.repo.root
    db.review_completed(at_hour(13.5), '107', cycle=1, has_blocking=True)
    db.review_completed(at_hour(13.8), '107', cycle=2, has_blocking=False)
    _review_state(root, '101', json.dumps({'review_cycles_total': 2}))
    _review_state(root, 'task-102', json.dumps({'review_cycles_total': 0}))
    _review_state(root, '103', b'{"review_cycles_total": \xff}')
    _review_state(root, '104', '{not json')
    _review_state(root, '105', json.dumps({'amendment_rounds_total': 1}))
    _review_state(root, '106', json.dumps({'review_cycles_total': 'two'}))
    _review_state(root, '108', json.dumps({'review_cycles_total': 4}))

    db.task_result('101', 'run-a', 'blocked', at_hour(9.0), block_reason=UNACTIONABLE)
    db.task_result('101', 'run-b', 'blocked', at_hour(9.5), block_reason=ALREADY_DONE)
    db.task_result('102', 'run-a', 'blocked', at_hour(10.5), block_reason=UNACTIONABLE)
    db.task_result('103', 'run-a', 'blocked', at_hour(11.5), block_reason=FALSE_PREMISE)
    db.task_result('105', 'run-a', 'blocked', at_hour(11.5), block_reason=PLAN_FILES)
    db.task_result('106', 'run-a', 'blocked', at_hour(12.2),
                   block_reason='Post-merge verification failed: z')
    return s


@pytest.fixture(scope='module')
def review_rows(review_scenario: LandingScenario) -> dict[tuple[str, str], LandingRow]:
    return _by_key(_rows(review_scenario))


class TestReviewRounds:
    def test_the_highest_review_cycle_in_the_window(self, review_scenario, review_rows):
        cost = review_rows[('107', review_scenario.m7)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (
            2, ReviewRoundsSource.REVIEW_COMPLETED,
        )

    def test_before_the_epoch_the_disk_counter_reads_lifetime_rounds(
        self, review_scenario, review_rows
    ):
        cost = review_rows[('101', review_scenario.m1)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (
            3, ReviewRoundsSource.DISK_COUNTER,
        )

    def test_the_task_prefixed_meta_directory_is_read(self, review_scenario, review_rows):
        cost = review_rows[('102', review_scenario.f2)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (
            1, ReviewRoundsSource.DISK_COUNTER,
        )

    @pytest.mark.parametrize('task', ['103', '104', '105'])
    def test_no_usable_counter_before_the_epoch_is_unknown(
        self, review_scenario, review_rows, task
    ):
        cost = review_rows[(task, review_scenario.m3)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (None, None)

    def test_a_non_integer_counter_is_unknown(self, review_scenario, review_rows):
        cost = review_rows[('106', review_scenario.side)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (None, None)

    def test_after_the_epoch_the_disk_counter_is_never_read(self, review_scenario, review_rows):
        cost = review_rows[('108', review_scenario.m8)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (None, None)

    def test_the_task_keyed_directory_wins_over_the_legacy_one(self, tmp_path: Path) -> None:
        s = build_landing_scenario(tmp_path / 'proj')
        s.db.review_completed(at_hour(13.5), '107', cycle=1, has_blocking=False)
        _review_state(s.repo.root, '102', json.dumps({'review_cycles_total': 5}))
        _review_state(s.repo.root, 'task-102', json.dumps({'review_cycles_total': 1}))
        cost = _by_key(_rows(s))[('102', s.f2)].cost
        assert (cost.review_rounds, cost.review_rounds_source) == (
            2, ReviewRoundsSource.DISK_COUNTER,
        )


class TestCharterClassBackfill:
    @pytest.mark.parametrize(
        ('task', 'sha', 'expected'),
        [
            ('102', 'f2', CharterClass.UNDER_SPECIFIED),
            ('103', 'm3', CharterClass.UNDER_SPECIFIED),
            ('105', 'm3', CharterClass.PLAN_FILES_UNTOUCHED),
            ('101', 'm1', CharterClass.STALE_CHARTER),
        ],
    )
    def test_block_reason_prefixes_map_to_classes(
        self, review_scenario, review_rows, task, sha, expected
    ):
        cost = review_rows[(task, getattr(review_scenario, sha))].cost
        assert (cost.charter_class, cost.charter_class_source) == (
            expected, CharterClassSource.REGEX_BACKFILL,
        )

    def test_other_block_reasons_are_no_charter_class(self, review_scenario, review_rows):
        cost = review_rows[('106', review_scenario.side)].cost
        assert (cost.charter_class, cost.charter_class_source) == (None, None)

    def test_the_vocabulary(self):
        assert {c.value for c in CharterClass} == {
            'under_specified', 'stale_charter', 'plan_files_untouched',
        }


class TestCharterClassColumn:
    def test_the_column_wins_over_a_matching_block_reason(self, tmp_path: Path) -> None:
        s = build_landing_scenario(tmp_path / 'proj', with_charter_class=True)
        s.db.task_result('102', 'run-a', 'blocked', at_hour(10.5),
                         block_reason=UNACTIONABLE, charter_class='stale_charter')
        cost = _by_key(_rows(s))[('102', s.f2)].cost
        assert (cost.charter_class, cost.charter_class_source) == (
            CharterClass.STALE_CHARTER, CharterClassSource.CHARTER_CLASS,
        )

    def test_a_null_column_falls_back_to_the_prefix_backfill_per_row(
        self, tmp_path: Path
    ) -> None:
        s = build_landing_scenario(tmp_path / 'proj', with_charter_class=True)
        s.db.task_result('102', 'run-a', 'blocked', at_hour(10.5), block_reason=UNACTIONABLE)
        s.db.task_result('103', 'run-a', 'blocked', at_hour(11.5),
                         block_reason=UNACTIONABLE, charter_class='stale_charter')
        rows = _by_key(_rows(s))
        f2, m3 = rows[('102', s.f2)].cost, rows[('103', s.m3)].cost
        assert (f2.charter_class, f2.charter_class_source) == (
            CharterClass.UNDER_SPECIFIED, CharterClassSource.REGEX_BACKFILL,
        )
        assert (m3.charter_class, m3.charter_class_source) == (
            CharterClass.STALE_CHARTER, CharterClassSource.CHARTER_CLASS,
        )

    def test_an_unknown_column_value_is_refused(self, tmp_path: Path) -> None:
        s = build_landing_scenario(tmp_path / 'proj', with_charter_class=True)
        s.db.task_result('102', 'run-a', 'blocked', at_hour(10.5), charter_class='bogus')
        with pytest.raises(UnknownCharterClass) as excinfo:
            _rows(s)
        assert 'bogus' in str(excinfo.value)
        assert '102' in str(excinfo.value)


class TestLines:
    def _expect(self, row: LandingRow, repo: Path, base: str, sha: str) -> None:
        lines = landed_lines(repo, base, sha, timeout_s=60)
        assert (row.cost.code_lines, row.cost.src_lines, row.cost.test_lines) == (
            lines.code_lines, lines.src_lines, lines.test_lines,
        )
        assert row.modules == lines.modules

    def test_a_merge_is_measured_against_its_first_parent(self, outcomes_scenario, outcome_rows):
        s = outcomes_scenario
        self._expect(outcome_rows[('101', s.m1)], s.repo.root, s.c0, s.m1)

    def test_train_members_carry_the_merge_commits_lines(self, outcomes_scenario, outcome_rows):
        s = outcomes_scenario
        for task in ('103', '104', '105'):
            self._expect(outcome_rows[(task, s.m3)], s.repo.root, s.f2, s.m3)

    def test_a_fast_forward_landing_is_measured_against_the_previous_landing(
        self, outcomes_scenario, outcome_rows
    ):
        s = outcomes_scenario
        self._expect(outcome_rows[('102', s.f2)], s.repo.root, s.m1, s.f2)

    def test_an_off_first_parent_landing_is_unmeasurable(self, outcomes_scenario, outcome_rows):
        row = outcome_rows[('106', outcomes_scenario.side)]
        assert (row.cost.code_lines, row.cost.src_lines, row.cost.test_lines) == (None, None, None)
        assert row.modules == ()


class TestGitBudget:
    def test_a_spent_budget_measures_the_first_unit_whole_then_stops_before_git(
        self, outcomes_scenario
    ):
        s = outcomes_scenario
        batch = landings(s.db.conn, s.repo.root, since_cursor=None).landings
        rows = rows_for(batch[2:], s.db.conn, s.repo.root, git_budget_s=0)
        assert [(r.landing.task_id, r.landing.merge_sha) for r in rows] == [
            ('103', s.m3), ('104', s.m3), ('105', s.m3), ('106', s.side),
        ]
        assert all(r.cost.code_lines is not None for r in rows[:3])

    def test_resuming_with_the_rest_continues_from_the_next_unit(
        self, outcomes_scenario, outcome_rows
    ):
        s = outcomes_scenario
        remaining = landings(s.db.conn, s.repo.root, since_cursor=None).landings
        measured: list[LandingRow] = []
        while remaining:
            rows = rows_for(remaining, s.db.conn, s.repo.root, git_budget_s=0)
            assert rows, 'every call makes progress'
            assert rows[0].landing == remaining[0]
            measured.extend(rows)
            remaining = remaining[len(rows):]
        assert _by_key(measured) == outcome_rows


class TestModuleAllocation:
    def test_shares_follow_src_lines_and_sum_to_one(self, outcomes_scenario, outcome_rows):
        row = outcome_rows[('105', outcomes_scenario.m3)]
        allocation = module_allocation(row)
        assert [module for module, _ in allocation] == [m.module for m in row.modules]
        assert sum(share for _, share in allocation) == pytest.approx(1.0)
        assert len(allocation) == 2

    def test_no_src_lines_allocates_nothing(self, outcomes_scenario, outcome_rows):
        assert module_allocation(outcome_rows[('106', outcomes_scenario.side)]) == ()


@pytest.fixture(scope='module')
def late_claim_scenario(tmp_path_factory: pytest.TempPathFactory) -> LateClaimScenario:
    """The late-claim history plus outcome events; read-only for every test that uses it."""
    s = build_late_claim_scenario(tmp_path_factory.mktemp('late-claim-rows') / 'proj')
    s.db.merge_blocked(at_hour(9.5), '201', POST_MERGE, 'Post-merge verification failed: a')
    s.db.merge_blocked(at_hour(10.8), '203', POST_MERGE, 'Post-merge verification failed: b')
    s.db.merge_verify(at_hour(9.8), '201', s.x, passed=True)
    s.db.merge_verify(at_hour(10.9), '203', s.r203, passed=True)
    return s


def _late_claim_batch(s: LateClaimScenario) -> tuple[Landing, ...]:
    return landings(s.db.conn, s.repo.root, since_cursor=None).landings


def _keys(rows: Sequence[LandingRow]) -> list[tuple[str, str]]:
    return [(row.landing.task_id, row.landing.merge_sha) for row in rows]


class TestLateClaimRows:
    def test_rows_come_back_in_input_order(self, late_claim_scenario):
        s = late_claim_scenario
        batch = _late_claim_batch(s)
        rows = rows_for(batch, s.db.conn, s.repo.root, git_budget_s=60)
        assert [row.landing for row in rows] == list(batch)

    def test_the_first_claim_and_the_late_claim_are_separate_units(self, late_claim_scenario):
        s = late_claim_scenario
        rows = _by_key(rows_for(_late_claim_batch(s), s.db.conn, s.repo.root, git_budget_s=60))
        assert rows[('201', s.x)].outcomes.rejection_events == 1
        late = rows[('203', s.x)]
        assert late.outcomes.rejection_events == 1
        assert late.outcomes.reverify_count == 0

    def test_the_late_claim_has_no_lines_and_the_first_claim_keeps_its_own(
        self, late_claim_scenario
    ):
        s = late_claim_scenario
        rows = _by_key(rows_for(_late_claim_batch(s), s.db.conn, s.repo.root, git_budget_s=60))
        late = rows[('203', s.x)]
        assert (late.cost.code_lines, late.cost.src_lines, late.cost.test_lines) == (
            None, None, None,
        )
        assert late.modules == ()
        first = rows[('201', s.x)]
        lines = landed_lines(s.repo.root, s.r203, s.x, timeout_s=60)
        assert (first.cost.code_lines, first.cost.src_lines, first.cost.test_lines) == (
            lines.code_lines, lines.src_lines, lines.test_lines,
        )
        assert first.modules == lines.modules

    def test_every_landing_heads_its_own_unit(self, late_claim_scenario):
        s = late_claim_scenario
        rows = rows_for(_late_claim_batch(s), s.db.conn, s.repo.root, git_budget_s=60)
        assert len(unit_heads(rows)) == 4

    def test_a_budget_limited_resume_covers_each_landing_once_in_order(
        self, late_claim_scenario
    ):
        s = late_claim_scenario
        batch = _late_claim_batch(s)
        remaining = batch
        measured: list[LandingRow] = []
        while remaining:
            rows = rows_for(remaining, s.db.conn, s.repo.root, git_budget_s=0)
            assert rows, 'every call makes progress'
            assert rows[0].landing == remaining[0]
            measured.extend(rows)
            remaining = remaining[len(rows):]
        assert _keys(measured) == [(landing.task_id, landing.merge_sha) for landing in batch]


class TestUnitShape:
    def _batch(self, s: LandingScenario) -> dict[str, Landing]:
        rows = landings(s.db.conn, s.repo.root, since_cursor=None).landings
        return {landing.task_id: landing for landing in rows if landing.merge_sha == s.m3
                or landing.task_id == '106'}

    def test_a_unit_missing_a_member_is_refused(self, outcomes_scenario):
        s = outcomes_scenario
        by_task = self._batch(s)
        with pytest.raises(QualityRowInvariantError, match='unit_size'):
            rows_for([by_task['103'], by_task['104']], s.db.conn, s.repo.root, git_budget_s=60)

    def test_an_interleaved_unit_is_refused(self, outcomes_scenario):
        s = outcomes_scenario
        by_task = self._batch(s)
        split = [by_task['103'], by_task['106'], by_task['104'], by_task['105']]
        with pytest.raises(QualityRowInvariantError):
            rows_for(split, s.db.conn, s.repo.root, git_budget_s=60)

    def test_a_unit_id_starting_a_second_run_is_refused(self, late_claim_scenario):
        s = late_claim_scenario
        batch = list(_late_claim_batch(s))
        index = next(i for i, landing in enumerate(batch) if landing.task_id == '203')
        batch[index] = dataclasses.replace(batch[index], late_claim=False, unit_id=s.x)
        with pytest.raises(QualityRowInvariantError, match=s.x):
            rows_for(batch, s.db.conn, s.repo.root, git_budget_s=60)
