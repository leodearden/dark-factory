"""in/engagement.json read as this project's engagement projection (PRD §4.14, §7.5)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _git_repo import make_repo
from _in_dir import write_engagement

from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    EngagementProjection,
    FeedError,
    parse_engagement,
)
from client_feed.engagement import (
    EngagementProblem,
    EngagementProjectMismatch,
    EngagementUnusable,
    read_engagement,
)

PROJECT = 'acme'
T0 = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)
IN_V1 = Path(__file__).parent / 'fixtures' / 'in-v1'
IN_V1_DELIVERED = '9b1e4c7a2d5f8e0b3a6c9d2e5f8a1b4c7d0e3f6a'


@dataclass(frozen=True)
class _Project:
    root: Path
    in_dir: Path
    c1: str


@pytest.fixture
def project(tmp_path: Path) -> _Project:
    repo = make_repo(tmp_path / 'repo')
    repo.write('src/a.py', 'a = 1\n')
    return _Project(root=repo.root, in_dir=tmp_path / 'in', c1=repo.commit('c1', when=T0))


def _unusable(project: _Project, *, project_id: str = PROJECT) -> EngagementUnusable:
    with pytest.raises(EngagementUnusable) as caught:
        read_engagement(project.root, project.in_dir, project_id=project_id)
    return caught.value


@pytest.mark.parametrize('delivered', ['none', 'c1'])
def test_a_valid_projection_is_returned_as_written(project: _Project, delivered: str) -> None:
    commit = None if delivered == 'none' else project.c1
    write_engagement(project.in_dir, project_id=PROJECT, version=3, delivered=commit)

    assert read_engagement(project.root, project.in_dir, project_id=PROJECT) == EngagementProjection(
        engagement_version=3, project_id=PROJECT, stage='building', delivered_commit=commit
    )


def test_no_engagement_file_is_missing(project: _Project) -> None:
    project.in_dir.mkdir()

    unusable = _unusable(project)

    assert unusable.problem is EngagementProblem.MISSING
    assert str(project.in_dir) not in str(unusable)
    assert 'reveals are disabled' in str(unusable)


@pytest.mark.parametrize(
    ('text', 'field'),
    [
        ('{"engagement_version": 1, ', 'not JSON'),
        ('{"engagement_version": 1, "project_id": "acme", "delivered_commit": null}', 'stage'),
        (
            '{"engagement_version": 1, "project_id": "acme", "stage": "building",'
            ' "delivered_commit": "HEAD"}',
            'delivered_commit',
        ),
    ],
    ids=['not-json', 'missing-field', 'not-a-sha'],
)
def test_an_engagement_that_is_not_a_projection_is_invalid(
    project: _Project, text: str, field: str
) -> None:
    project.in_dir.mkdir()
    (project.in_dir / ENGAGEMENT_FILENAME).write_text(text)

    unusable = _unusable(project)

    assert unusable.problem is EngagementProblem.INVALID
    assert field in unusable.detail


def test_a_symlinked_engagement_is_unreadable_and_its_target_never_read(
    project: _Project, tmp_path: Path
) -> None:
    target = tmp_path / 'elsewhere.json'
    target.write_text('CANARY-ENG')
    project.in_dir.mkdir()
    (project.in_dir / ENGAGEMENT_FILENAME).symlink_to(target)

    unusable = _unusable(project)

    assert unusable.problem is EngagementProblem.UNREADABLE
    assert 'CANARY-ENG' not in f'{unusable} {unusable.detail}'
    assert str(tmp_path) not in str(unusable)


def test_an_engagement_that_is_not_utf8_is_unreadable(project: _Project) -> None:
    project.in_dir.mkdir()
    (project.in_dir / ENGAGEMENT_FILENAME).write_bytes(b'{"engagement_version": \xff}')

    assert _unusable(project).problem is EngagementProblem.UNREADABLE


def test_a_delivered_commit_the_repository_does_not_hold_is_unknown(project: _Project) -> None:
    write_engagement(project.in_dir, project_id=PROJECT, delivered=IN_V1_DELIVERED)

    unusable = _unusable(project)

    assert unusable.problem is EngagementProblem.UNKNOWN_DELIVERED_COMMIT
    assert IN_V1_DELIVERED in str(unusable)


def test_a_projection_of_another_project_is_a_mismatch_refusal(project: _Project) -> None:
    write_engagement(project.in_dir, project_id='other')

    with pytest.raises(EngagementProjectMismatch) as caught:
        read_engagement(project.root, project.in_dir, project_id=PROJECT)

    assert isinstance(caught.value, FeedError)
    assert 'other' in str(caught.value)
    assert PROJECT in str(caught.value)
    assert (caught.value.found, caught.value.expected) == ('other', PROJECT)


def test_the_vendored_in_v1_projection_is_read_with_every_field(project: _Project) -> None:
    vendored = parse_engagement((IN_V1 / ENGAGEMENT_FILENAME).read_bytes().decode('utf-8'))
    rebased = vendored.model_copy(update={'delivered_commit': None})
    project.in_dir.mkdir()
    (project.in_dir / ENGAGEMENT_FILENAME).write_text(rebased.model_dump_json())

    assert vendored.delivered_commit == IN_V1_DELIVERED
    assert read_engagement(project.root, project.in_dir, project_id='demo-client') == rebased
