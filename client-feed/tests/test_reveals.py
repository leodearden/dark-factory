"""Reveals through the publisher: the engagement read, each request answered, each revealed file followed (PRD B11-B17)."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _git_repo import Repo, make_repo
from _in_dir import write_engagement
from _stores import RunsDb, TasksDb

from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    FEED_FILENAME,
    STATE_FILENAME,
    Degradation,
    FeedLine,
    Heartbeat,
    Source,
)
from client_feed.engagement import EngagementProjectMismatch
from client_feed.feed_io import CURSOR_FILENAME, OutDir, read_feed_lines
from client_feed.policy import Policy
from client_feed.publisher import PassReport, run_pass
from client_feed.validate import validate_out_dir

PROJECT = 'acme'
BASE = datetime(2026, 10, 22, 10, 0, tzinfo=UTC)
NOW = BASE + timedelta(hours=1)
UNHELD_SHA = '9b1e4c7a2d5f8e0b3a6c9d2e5f8a1b4c7d0e3f6a'


def at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


@dataclass(frozen=True)
class Project:
    """A root with empty stores and a committed main, its policy, and the portal's in_dir beside out_dir."""

    repo: Repo
    policy: Policy

    @property
    def in_dir(self) -> Path:
        return self.policy.in_dir

    @property
    def out_dir(self) -> Path:
        return self.policy.out_dir

    def open(self) -> OutDir:
        return OutDir.open(self.out_dir, project_id=PROJECT)

    def run(self, *, minutes: float = 60) -> PassReport:
        return run_pass(self.policy, self.open(), now=at(minutes))

    def feed(self) -> tuple[FeedLine, ...]:
        return read_feed_lines(self.out_dir)


def make_project(
    tmp_path: Path, files: Mapping[str, str | bytes], **policy_fields: Any
) -> Project:
    repo = make_repo(tmp_path / 'repo')
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    for path, content in files.items():
        repo.write(path, content)
    repo.commit('c1', when=at(0))
    TasksDb.create(repo.root)
    RunsDb.create(repo.root)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    policy = Policy.model_validate(
        {
            'project_id': PROJECT,
            'project_root': str(repo.root),
            'out_dir': str(out_dir),
            'in_dir': str(tmp_path / 'in'),
            'visible_globs': ['src/**', 'tests/**'],
            **policy_fields,
        }
    )
    return Project(repo=repo, policy=policy)


def heartbeats(project: Project) -> list[Heartbeat]:
    return [line for line in project.feed() if isinstance(line, Heartbeat)]


def engagement_degradation(report: PassReport) -> Degradation:
    (degradation,) = report.degraded
    assert degradation.source is Source.ENGAGEMENT
    return degradation


@pytest.fixture
def small(tmp_path: Path) -> Project:
    return make_project(tmp_path, {'src/a.py': 'a = 1\n', 'tests/test_a.py': 'def test_a():\n    pass\n'})


def test_every_heartbeat_echoes_the_engagement_version_it_applied(small: Project) -> None:
    write_engagement(small.in_dir, project_id=PROJECT, version=1)
    small.run(minutes=60)
    small.run(minutes=61)
    write_engagement(small.in_dir, project_id=PROJECT, version=2)
    small.run(minutes=62)

    assert [beat.engagement_version for beat in heartbeats(small)] == [1, 1, 2]


def _publishes_every_other_source(project: Project, report: PassReport) -> None:
    (beat,) = heartbeats(project)
    assert beat.engagement_version is None
    assert beat.degraded == report.degraded
    assert {'factory_state', 'branch_snapshot', 'tree_snapshot', 'heartbeat'} <= {
        line.type for line in project.feed()
    }
    assert validate_out_dir(project.out_dir, expected_project_id=PROJECT).ok


def test_a_missing_engagement_is_degraded_and_every_other_source_still_publishes(
    small: Project,
) -> None:
    report = small.run()

    assert 'missing' in engagement_degradation(report).reason
    _publishes_every_other_source(small, report)


def test_an_unparsable_engagement_is_degraded_as_invalid(small: Project) -> None:
    small.in_dir.mkdir()
    (small.in_dir / ENGAGEMENT_FILENAME).write_text('{"engagement_version": ')

    report = small.run()

    assert 'invalid' in engagement_degradation(report).reason
    _publishes_every_other_source(small, report)


def test_a_delivered_commit_the_repository_does_not_hold_is_degraded_naming_it(
    small: Project,
) -> None:
    write_engagement(small.in_dir, project_id=PROJECT, delivered=UNHELD_SHA)

    report = small.run()

    assert UNHELD_SHA in engagement_degradation(report).reason
    _publishes_every_other_source(small, report)


def _out_bytes(project: Project) -> dict[str, bytes]:
    names = (FEED_FILENAME, STATE_FILENAME, CURSOR_FILENAME)
    return {name: (project.out_dir / name).read_bytes() for name in names}


def test_an_engagement_of_another_project_stops_the_pass_before_anything_is_appended(
    small: Project,
) -> None:
    write_engagement(small.in_dir, project_id=PROJECT)
    small.run(minutes=60)
    before = _out_bytes(small)
    write_engagement(small.in_dir, project_id='other')

    with pytest.raises(EngagementProjectMismatch):
        small.run(minutes=61)

    assert _out_bytes(small) == before
