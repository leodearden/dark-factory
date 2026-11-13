"""shared.quality_series.landings: the landing datum from runs.db, cross-checked against git."""

from __future__ import annotations

import dataclasses
from datetime import timedelta, timezone
from pathlib import Path

import pytest
from quality_series_fixtures import (
    GitRepoBuilder,
    LandingScenario,
    LateClaimScenario,
    RunsDbBuilder,
    at_hour,
    build_landing_scenario,
    build_late_claim_scenario,
)

from shared.project_id import project_id_for_root
from shared.quality_series import CursorNotOnMain, Landing, UnrecognisedTrainEvent, landings
from shared.quality_series.git_lines import GitUnavailable
from shared.quality_series.landings import LandingSource, LinesBase, recorded_landing_keys
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


@pytest.fixture(scope='module')
def scenario(tmp_path_factory: pytest.TempPathFactory) -> LandingScenario:
    """Read-only for every test here, so one build serves the module."""
    return build_landing_scenario(tmp_path_factory.mktemp('landings') / 'proj')


def _key(landing: Landing) -> tuple[str, str]:
    return landing.task_id, landing.merge_sha


def _by_key(rows: tuple[Landing, ...]) -> dict[tuple[str, str], Landing]:
    return {_key(landing): landing for landing in rows}


class TestFullHistory:
    def test_every_landing_once_in_landed_order(self, scenario: LandingScenario) -> None:
        s = scenario
        rows = landings(s.db.conn, s.repo.root, since_cursor=None).landings
        late = sorted([('107', s.m7), ('108', s.m8)], key=lambda pair: pair[1])
        assert [_key(row) for row in rows] == [
            ('101', s.m1), ('102', s.f2), ('103', s.m3), ('104', s.m3), ('105', s.m3),
            ('106', s.side), ('101', s.m6), *late,
        ]
        assert [row.ordinal for row in rows] == list(range(len(rows)))

    def test_project_id_is_derived_from_the_repo_root(self, scenario: LandingScenario) -> None:
        rows = landings(scenario.db.conn, scenario.repo.root, since_cursor=None).landings
        assert {row.project_id for row in rows} == {project_id_for_root(scenario.repo.root)}

    def test_an_explicit_project_id_overrides(self, scenario: LandingScenario) -> None:
        rows = landings(scenario.db.conn, scenario.repo.root, None, project_id='p').landings
        assert {row.project_id for row in rows} == {'p'}

    def test_train_members_share_one_unit(self, scenario: LandingScenario) -> None:
        s = scenario
        rows = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)
        members = [rows[(task, s.m3)] for task in ('103', '104', '105')]
        for member in members:
            assert member.unit_id == member.merge_sha == s.m3
            assert member.unit_size == 3
            assert member.source is LandingSource.TRAIN_MERGED
            assert member.landed_at == at_hour(LandingScenario.LANDED_HOURS['m3'])

    def test_plain_landings_are_units_of_one(self, scenario: LandingScenario) -> None:
        s = scenario
        rows = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)
        for key in [('101', s.m1), ('102', s.f2), ('106', s.side), ('101', s.m6)]:
            assert rows[key].unit_id == rows[key].merge_sha
            assert rows[key].unit_size == 1
            assert rows[key].source is LandingSource.MERGE_FINALIZED

    def test_first_parent_membership_and_lines_base(self, scenario: LandingScenario) -> None:
        s = scenario
        rows = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)
        m1, f2, side = rows[('101', s.m1)], rows[('102', s.f2)], rows[('106', s.side)]
        for key in [('101', s.m1), ('102', s.f2), ('105', s.m3), ('101', s.m6)]:
            assert rows[key].git_first_parent is True
        assert side.git_first_parent is False
        assert (m1.lines_base, m1.base_sha) == (LinesBase.FIRST_PARENT, s.c0)
        assert (f2.lines_base, f2.base_sha) == (LinesBase.PREVIOUS_LANDING, s.m1)
        assert (side.lines_base, side.base_sha) == (LinesBase.UNMEASURABLE, None)
        assert rows[('101', s.m6)].base_sha == s.m4

    def test_prev_landed_at_is_the_same_tasks_previous_landing(
        self, scenario: LandingScenario
    ) -> None:
        s = scenario
        rows = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)
        assert rows[('101', s.m6)].prev_landed_at == rows[('101', s.m1)].landed_at
        assert rows[('101', s.m1)].prev_landed_at is None
        assert rows[('104', s.m3)].prev_landed_at is None

    def test_git_cross_check_counts_both_disagreements(self, scenario: LandingScenario) -> None:
        check = landings(scenario.db.conn, scenario.repo.root, since_cursor=None).git_check
        assert check.off_first_parent == 1
        assert check.merges_without_landing == 1
        assert check.late_claims == 0
        assert check.disagreement == 2


class TestCursor:
    def test_resumes_after_the_cursor_with_stable_ordinals(self, scenario: LandingScenario) -> None:
        s = scenario
        full = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)
        resumed = landings(s.db.conn, s.repo.root, since_cursor=s.m1).landings
        assert ('101', s.m1) not in {_key(row) for row in resumed}
        for row in resumed:
            assert row.ordinal == full[_key(row)].ordinal
        assert ('106', s.side) in {_key(row) for row in resumed}
        assert len(resumed) == len(full) - 1

    def test_an_off_first_parent_landing_before_the_cursor_is_not_re_emitted(
        self, scenario: LandingScenario
    ) -> None:
        s = scenario
        resumed = landings(s.db.conn, s.repo.root, since_cursor=s.m6).landings
        assert [_key(row) for row in resumed] == sorted(
            [('107', s.m7), ('108', s.m8)], key=lambda pair: pair[1]
        )

    def test_next_cursor_is_the_newest_emitted_first_claim(self, scenario: LandingScenario) -> None:
        s = scenario
        assert landings(s.db.conn, s.repo.root, since_cursor=None).next_cursor == s.m8
        assert landings(s.db.conn, s.repo.root, since_cursor=s.m1).next_cursor == s.m8

    def test_with_nothing_emitted_next_cursor_is_the_cursor_in_full(
        self, scenario: LandingScenario
    ) -> None:
        s = scenario
        batch = landings(s.db.conn, s.repo.root, since_cursor=s.m8[:12])
        assert (batch.landings, batch.next_cursor) == ((), s.m8)

    def test_next_cursor_never_passes_a_merge_whose_event_is_not_written_yet(
        self, tmp_path: Path
    ) -> None:
        s = build_landing_scenario(tmp_path / 'proj')
        m9 = s.repo.branch_merge('109', {'pkg/src/pkg/f/u.py': 'u = 1\n'})
        before = landings(s.db.conn, s.repo.root, since_cursor=None)
        assert before.next_cursor == s.m8 != s.repo.head()
        assert before.git_check.merges_without_landing == 2
        s.db.merge_done(at_hour(15.0), '109', m9)
        after = landings(s.db.conn, s.repo.root, since_cursor=before.next_cursor)
        assert [_key(row) for row in after.landings] == [('109', m9)]
        assert after.next_cursor == m9

    def test_a_cursor_off_main_is_refused(self, scenario: LandingScenario) -> None:
        with pytest.raises(CursorNotOnMain):
            landings(scenario.db.conn, scenario.repo.root, since_cursor=scenario.side)

    def test_a_directory_outside_any_repository_is_git_unavailable(
        self, scenario: LandingScenario, tmp_path: Path
    ) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        with pytest.raises(GitUnavailable):
            landings(scenario.db.conn, plain, since_cursor=None)


def test_a_unit_has_at_least_one_member(scenario: LandingScenario) -> None:
    row = landings(scenario.db.conn, scenario.repo.root, since_cursor=None).landings[0]
    with pytest.raises(ValueError, match='unit_size'):
        dataclasses.replace(row, unit_size=0)


def test_landing_times_are_timezone_aware(scenario: LandingScenario) -> None:
    row = landings(scenario.db.conn, scenario.repo.root, since_cursor=None).landings[0]
    assert row.landed_at.utcoffset() is not None
    with pytest.raises(ValueError, match='landed_at'):
        dataclasses.replace(row, landed_at=row.landed_at.replace(tzinfo=None))


@pytest.fixture(scope='module')
def late_claim(tmp_path_factory: pytest.TempPathFactory) -> LateClaimScenario:
    """Read-only for every test that uses it."""
    return build_late_claim_scenario(tmp_path_factory.mktemp('late-claim') / 'proj')


def _keys(rows: tuple[Landing, ...]) -> list[tuple[str, str]]:
    return [_key(row) for row in rows]


class TestLateClaim:
    """A done row naming a sha that an earlier event already claimed for another task."""

    def test_full_history_keeps_the_late_claim_in_landed_order(
        self, late_claim: LateClaimScenario
    ) -> None:
        s = late_claim
        rows = landings(s.db.conn, s.repo.root, since_cursor=None).landings
        assert _keys(rows) == [('201', s.x), ('202', s.y), ('203', s.x), ('204', s.z)]
        assert [row.ordinal for row in rows] == [0, 1, 2, 3]

    def test_the_first_claim_keeps_the_sha_as_its_unit(self, late_claim: LateClaimScenario) -> None:
        s = late_claim
        first = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)[('201', s.x)]
        assert first.late_claim is False
        assert first.unit_id == s.x
        assert first.unit_size == 1
        assert (first.lines_base, first.base_sha) == (LinesBase.FIRST_PARENT, s.r203)

    def test_the_late_claim_is_its_own_unit_with_unmeasurable_lines(
        self, late_claim: LateClaimScenario
    ) -> None:
        s = late_claim
        rows = landings(s.db.conn, s.repo.root, since_cursor=None).landings
        late = _by_key(rows)[('203', s.x)]
        assert late.late_claim is True
        assert late.unit_size == 1
        assert late.source is LandingSource.MERGE_FINALIZED
        assert late.git_first_parent is True
        assert (late.lines_base, late.base_sha) == (LinesBase.UNMEASURABLE, None)
        assert late.unit_id != s.x
        unit_ids = [row.unit_id for row in rows]
        assert len(set(unit_ids)) == len(unit_ids)

    def test_git_check_counts_the_late_claim(self, late_claim: LateClaimScenario) -> None:
        s = late_claim
        check = landings(s.db.conn, s.repo.root, since_cursor=None).git_check
        assert check.late_claims == 1
        assert check.off_first_parent == 0
        assert check.merges_without_landing == 0
        assert check.disagreement == 1

    def test_a_range_reaching_the_late_claimers_own_merge_counts_it(
        self, late_claim: LateClaimScenario
    ) -> None:
        s = late_claim
        check = landings(s.db.conn, s.repo.root, since_cursor=s.c0).git_check
        assert check.late_claims == 1
        assert check.merges_without_landing == 1
        assert check.off_first_parent == 0
        assert check.disagreement == 2

    @pytest.mark.parametrize(
        ('cursor', 'expected'),
        [
            ('x', [('202', 'y'), ('203', 'x'), ('204', 'z')]),
            ('y', [('203', 'x'), ('204', 'z')]),
            ('z', []),
        ],
    )
    def test_a_cursor_emits_the_late_claim_until_a_landing_after_it(
        self, late_claim: LateClaimScenario, cursor: str, expected: list[tuple[str, str]]
    ) -> None:
        s = late_claim
        full = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)
        resumed = landings(s.db.conn, s.repo.root, since_cursor=getattr(s, cursor)).landings
        assert _keys(resumed) == [(task, getattr(s, name)) for task, name in expected]
        for row in resumed:
            assert row.ordinal == full[_key(row)].ordinal

    def test_a_late_claim_refuses_measurable_lines(self, late_claim: LateClaimScenario) -> None:
        s = late_claim
        late = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)[('203', s.x)]
        with pytest.raises(ValueError, match='late_claim'):
            dataclasses.replace(late, lines_base=LinesBase.FIRST_PARENT, base_sha=s.r203)

    def test_a_first_claim_refuses_a_unit_id_other_than_its_sha(
        self, late_claim: LateClaimScenario
    ) -> None:
        s = late_claim
        first = _by_key(landings(s.db.conn, s.repo.root, since_cursor=None).landings)[('201', s.x)]
        with pytest.raises(ValueError, match='unit_id'):
            dataclasses.replace(first, unit_id='other')


class TestTrainAttribution:
    """A train member that passed solo after the train's union verify went red."""

    @pytest.fixture
    def solo(self, tmp_path: Path) -> tuple[GitRepoBuilder, RunsDbBuilder, str, str]:
        repo = GitRepoBuilder(tmp_path / 'proj')
        repo.commit({'README.md': 'solo\n'}, 'root')
        a = repo.branch_merge('301', {'pkg/src/pkg/a.py': 'a = 1\n'})
        b = repo.branch_merge('302', {'pkg/src/pkg/b.py': 'b = 1\n'})
        db = RunsDbBuilder(repo.root / 'data' / 'orchestrator' / 'runs.db')
        db.train_attributed(at_hour(10.0), '301', 'train-x', a)
        db.merge_done(at_hour(11.0), '302', b)
        return repo, db, a, b

    def test_a_solo_passer_is_a_landing_and_a_unit_of_one(self, solo) -> None:
        repo, db, a, b = solo
        rows = landings(db.conn, repo.root, since_cursor=None).landings
        assert _keys(rows) == [('301', a), ('302', b)]
        passer = rows[0]
        assert passer.source is LandingSource.TRAIN_ATTRIBUTED
        assert (passer.unit_id, passer.unit_size) == (a, 1)
        assert passer.late_claim is False
        assert passer.git_first_parent is True
        assert passer.lines_base is LinesBase.FIRST_PARENT

    def test_an_unrecognised_train_payload_is_a_typed_refusal(self, solo) -> None:
        repo, db, _, _ = solo
        db.event(at_hour(12.0), 'train_merged', '303', {'train_id': 'train-y', 'verdict': 'other'})
        with pytest.raises(UnrecognisedTrainEvent) as excinfo:
            landings(db.conn, repo.root, since_cursor=None)
        assert len(excinfo.value.event_ids) == 1


class TestRecordedLandingKeys:
    """The landings runs.db records, read without git so a request path may count them."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> RunsDbBuilder:
        """A runs.db outside any git repository; the reader takes the connection and no repo."""
        return RunsDbBuilder(tmp_path / 'not-a-repo' / 'runs.db')

    def test_a_done_merge_counts_once_and_any_other_state_not_at_all(
        self, db: RunsDbBuilder
    ) -> None:
        db.merge_done(at_hour(1.0), '401', 'sha-a')
        db.merge_done(at_hour(2.0), '401', 'sha-a')
        db.merge_state(at_hour(3.0), '402', 'blocked')
        db.merge_state(at_hour(4.0), '403', 'superseded', superseded_by='404')

        assert recorded_landing_keys(db.conn, recorded_after=None) == frozenset({('sha-a', '401')})

    def test_a_coalesced_train_records_each_member_under_its_merge_commit(
        self, db: RunsDbBuilder
    ) -> None:
        db.train_merged(at_hour(1.0), '503', ['501', '502', '503'], 'train-sha', 'base-sha')

        assert recorded_landing_keys(db.conn, recorded_after=None) == frozenset({
            ('train-sha', '501'), ('train-sha', '502'), ('train-sha', '503'),
        })

    def test_an_attributed_solo_passer_records_its_own_task(self, db: RunsDbBuilder) -> None:
        db.train_attributed(at_hour(1.0), '601', 'train-x', 'solo-sha')

        assert recorded_landing_keys(db.conn, recorded_after=None) == frozenset({
            ('solo-sha', '601'),
        })

    def test_only_landings_recorded_strictly_after_the_instant_count(
        self, db: RunsDbBuilder
    ) -> None:
        plus_one = timezone(timedelta(hours=1))
        db.merge_done(at_hour(9.0), '701', 'before')
        db.merge_done(at_hour(10.0).astimezone(plus_one), '702', 'at-the-instant')
        db.merge_done(at_hour(10.5).astimezone(plus_one), '703', 'after')
        db.train_merged(at_hour(11.0), '705', ['704', '705'], 'train-after', 'base')

        assert recorded_landing_keys(db.conn, recorded_after=at_hour(10.0)) == frozenset({
            ('after', '703'), ('train-after', '704'), ('train-after', '705'),
        })

    def test_an_unrecognised_train_payload_is_the_same_typed_refusal(
        self, db: RunsDbBuilder
    ) -> None:
        db.event(at_hour(1.0), 'train_merged', '801', {'train_id': 'train-z', 'verdict': 'other'})

        with pytest.raises(UnrecognisedTrainEvent):
            recorded_landing_keys(db.conn, recorded_after=None)
