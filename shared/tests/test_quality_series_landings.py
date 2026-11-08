"""shared.quality_series.landings: the landing datum from runs.db, cross-checked against git."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from quality_series_fixtures import LandingScenario, at_hour, build_landing_scenario
from shared.quality_series.landings import LandingSource, LinesBase

from shared.project_id import project_id_for_root
from shared.quality_series import CursorNotOnMain, Landing, landings
from shared.quality_series.git_lines import GitUnavailable


@pytest.fixture
def scenario(tmp_path: Path) -> LandingScenario:
    return build_landing_scenario(tmp_path / 'proj')


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
            assert member.landed_at == at_hour(LandingScenario.LANDED_HOURS['m3']).isoformat()

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
