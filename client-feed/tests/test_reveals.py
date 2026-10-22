"""Reveals through the publisher: the engagement read, each request answered, each revealed file followed (PRD B11-B17)."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _git_repo import Repo, make_repo
from _in_dir import request_reveals, write_engagement
from _stores import RunsDb, TasksDb

from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    FEED_FILENAME,
    INBOX_FILENAME,
    STATE_FILENAME,
    Degradation,
    FeedLine,
    FeedState,
    FileRevealed,
    Heartbeat,
    RefusalReason,
    RevealBudget,
    RevealRefused,
    Source,
    TreeSnapshot,
    parse_state,
)
from client_feed.engagement import EngagementProjectMismatch
from client_feed.feed_io import CURSOR_FILENAME, Cursor, OutDir, read_feed_lines
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
    tmp_path: Path,
    files: Mapping[str, str | bytes],
    *,
    links: Callable[[Repo], None] = lambda repo: None,
    **policy_fields: Any,
) -> Project:
    """main's first commit c1 holds `files` and whatever `links` stages; the stores are made after it, untracked."""
    repo = make_repo(tmp_path / 'repo')
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    for path, content in files.items():
        repo.write(path, content)
    links(repo)
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


def numbered(name: str, count: int) -> str:
    return ''.join(f'{name}{index} = {index}\n' for index in range(count))


B_FILES: Mapping[str, str | bytes] = {
    'src/a.py': numbered('a', 200),
    'src/b.py': numbered('b', 210),
    'src/c.py': numbered('c', 100),
    'src/d.py': numbered('d', 10),
    'src/e.py': numbered('e', 480),
    'tests/test_a.py': 'def test_a():\n    assert True\n',
    '.mcp.json': '{"mcpServers": {"CANARY-MCP": {}}}\n',
    'dark-factory-orchestrator.yaml': 'project_root: CANARY-ORCH\n',
    'src/secret/k.py': 'KEY = "CANARY-SECRET"\n',
    'src/blob.bin': b'\xff\xfe\x00\x01 CANARY-BLOB\n',
}
VISIBLE_REGULAR_FILES = [
    'src/a.py', 'src/b.py', 'src/blob.bin', 'src/c.py', 'src/d.py', 'src/e.py', 'src/secret/k.py',
    'tests/test_a.py',
]


@pytest.fixture
def hostile(tmp_path: Path) -> Project:
    """B11-B15's tree: 1,000 counted lines in src/a-e.py, beside invisible, never-reveal, binary, linked and untracked paths."""
    outside = tmp_path / 'outside' / 'secret.txt'
    outside.parent.mkdir()
    outside.write_text('CANARY-LINK\n')
    (tmp_path / 'x').write_text('CANARY-DOTDOT\n')

    def links(repo: Repo) -> None:
        repo.symlink('src/link.py', str(outside))
        repo.gitlink('src/sub', 'b' * 40)

    project = make_project(
        tmp_path,
        B_FILES,
        links=links,
        reveal_proportion=0.25,
        reveal_floor_lines=50,
        never_reveal_globs=['src/secret/**'],
        generated_globs=['src/*.bin'],
    )
    project.repo.write('src/untracked.py', 'CANARY-UNTRACKED\n')
    write_engagement(project.in_dir, project_id=PROJECT)
    return project


def answers(project: Project) -> list[FileRevealed | RevealRefused]:
    return [line for line in project.feed() if isinstance(line, FileRevealed | RevealRefused)]


def refusals(project: Project) -> list[tuple[str, RefusalReason]]:
    return [(line.path, line.reason) for line in answers(project) if isinstance(line, RevealRefused)]


def newest_tree(project: Project) -> TreeSnapshot:
    return [line for line in project.feed() if isinstance(line, TreeSnapshot)][-1]


def state_of(project: Project) -> FeedState:
    return parse_state((project.out_dir / STATE_FILENAME).read_text())


def request(project: Project, *paths: str, minutes: float = 30) -> None:
    request_reveals(project.in_dir, *paths, at=at(minutes))


def test_b11_the_tree_lists_only_visible_regular_files_and_an_invisible_path_is_not_visible(
    hostile: Project,
) -> None:
    request(hostile, '.mcp.json')

    hostile.run()

    assert [file.path for file in newest_tree(hostile).files] == VISIBLE_REGULAR_FILES
    assert refusals(hostile) == [('.mcp.json', RefusalReason.NOT_VISIBLE)]
    assert len(answers(hostile)) == 1


def test_b12_a_granted_request_reveals_the_file_at_head_and_spends_its_lines(
    hostile: Project,
) -> None:
    request(hostile, 'src/a.py')

    hostile.run()

    (revealed,) = answers(hostile)
    assert isinstance(revealed, FileRevealed)
    assert (
        revealed.path, revealed.content, revealed.lines, revealed.commit, revealed.deleted,
        revealed.renamed_from,
    ) == ('src/a.py', B_FILES['src/a.py'], 200, hostile.repo.rev('main'), False, None)
    assert [file.path for file in state_of(hostile).revealed] == ['src/a.py']
    assert newest_tree(hostile).budget == RevealBudget(
        proportion=0.25, counted_lines=1000, budget_lines=250, spent_lines=200, floor_lines=50
    )
    assert validate_out_dir(hostile.out_dir, expected_project_id=PROJECT).ok


def test_a_request_landing_exactly_on_the_budget_is_granted(hostile: Project) -> None:
    request(hostile, 'src/a.py')
    hostile.run(minutes=60)
    request(hostile, 'src/d.py', minutes=61)

    hostile.run(minutes=62)

    assert [(line.type, line.path) for line in answers(hostile)] == [
        ('file_revealed', 'src/a.py'),
        ('file_revealed', 'src/d.py'),
    ]
    assert newest_tree(hostile).budget.spent_lines == 250
    assert validate_out_dir(hostile.out_dir, expected_project_id=PROJECT).ok


def test_b13_requests_are_answered_in_inbox_order_and_past_the_budget_refused(
    hostile: Project,
) -> None:
    request(hostile, 'src/b.py', 'src/c.py', 'src/d.py')

    hostile.run()

    assert [(line.type, line.path) for line in answers(hostile)] == [
        ('file_revealed', 'src/b.py'),
        ('reveal_refused', 'src/c.py'),
        ('reveal_refused', 'src/d.py'),
    ]
    assert refusals(hostile) == [
        ('src/c.py', RefusalReason.OVER_BUDGET),
        ('src/d.py', RefusalReason.OVER_BUDGET),
    ]
    assert newest_tree(hostile).budget.spent_lines == 210


def test_b14_a_free_file_is_granted_without_spending(hostile: Project) -> None:
    hostile.run(minutes=60)
    request(hostile, 'tests/test_a.py', minutes=61)

    hostile.run(minutes=62)

    (revealed,) = answers(hostile)
    assert (revealed.type, revealed.path) == ('file_revealed', 'tests/test_a.py')
    assert newest_tree(hostile).budget.spent_lines == 0


HOSTILE_REQUESTS = [
    ('../x', RefusalReason.NOT_IN_TREE),
    ('/etc/passwd', RefusalReason.NOT_IN_TREE),
    ('src/link.py', RefusalReason.SYMLINK),
    ('src/untracked.py', RefusalReason.NOT_IN_TREE),
    ('src/sub', RefusalReason.SYMLINK),
    ('src/secret/k.py', RefusalReason.NOT_REVEALABLE),
    ('src/blob.bin', RefusalReason.NOT_REVEALABLE),
]


def test_b15_a_hostile_path_is_refused_by_tree_lookup_and_nothing_outside_the_tree_is_read(
    hostile: Project,
) -> None:
    request(hostile, *(path for path, _ in HOSTILE_REQUESTS))

    hostile.run()

    assert refusals(hostile) == HOSTILE_REQUESTS
    assert len(answers(hostile)) == len(HOSTILE_REQUESTS)
    for path in sorted(hostile.out_dir.iterdir()):
        assert 'CANARY-' not in path.read_text(errors='replace'), path.name


def test_the_inbox_offset_moves_past_answered_requests_so_none_is_answered_twice(
    hostile: Project,
) -> None:
    request(hostile, 'src/a.py', '../x')
    hostile.run(minutes=60)

    hostile.run(minutes=61)

    assert len(answers(hostile)) == 2
    cursor = Cursor.model_validate_json((hostile.out_dir / CURSOR_FILENAME).read_text())
    assert cursor.inbox_offset == (hostile.in_dir / INBOX_FILENAME).stat().st_size


def test_a_symlinked_inbox_is_degraded_as_inbox_and_its_offset_held(
    hostile: Project, tmp_path: Path
) -> None:
    elsewhere = tmp_path / 'elsewhere'
    request_reveals(elsewhere, 'src/a.py', at=at(30))
    (hostile.in_dir / INBOX_FILENAME).symlink_to(elsewhere / INBOX_FILENAME)

    report = hostile.run()

    assert [degradation.source for degradation in report.degraded] == [Source.INBOX]
    assert answers(hostile) == []
    cursor = Cursor.model_validate_json((hostile.out_dir / CURSOR_FILENAME).read_text())
    assert cursor.inbox_offset == 0
    _publishes_every_other_source_but(hostile, Source.INBOX)


def _publishes_every_other_source_but(project: Project, source: Source) -> None:
    beat = heartbeats(project)[-1]
    assert {degradation.source for degradation in beat.degraded} == {source}
    assert {'factory_state', 'branch_snapshot', 'tree_snapshot', 'heartbeat'} <= {
        line.type for line in project.feed()
    }
    assert validate_out_dir(project.out_dir, expected_project_id=PROJECT).ok


def test_a_malformed_inbox_line_is_named_and_the_valid_line_after_it_still_answered(
    hostile: Project,
) -> None:
    with (hostile.in_dir / INBOX_FILENAME).open('a') as inbox:
        inbox.write('{"not": "a request"}\n')
    request(hostile, 'src/a.py')

    report = hostile.run()

    (degradation,) = report.degraded
    assert degradation.source is Source.INBOX
    assert degradation.reason.startswith('rejected 1 ')
    assert [(line.type, line.path) for line in answers(hostile)] == [('file_revealed', 'src/a.py')]


def _delivered_project(tmp_path: Path, *, proportion: float) -> Project:
    """B17: c1 is delivered; c2 appends 60 lines to src/a.py and adds src/n.py of 30."""
    project = make_project(
        tmp_path,
        {name: content for name, content in B_FILES.items() if name.startswith('src/') and name.endswith('.py')},
        reveal_proportion=proportion,
        reveal_floor_lines=50,
        never_reveal_globs=['src/secret/**'],
    )
    repo = project.repo
    c1 = repo.rev('main')
    repo.write('src/a.py', numbered('a', 200) + numbered('grown', 60))
    repo.write('src/n.py', numbered('n', 30))
    repo.commit('c2', when=at(10))
    write_engagement(project.in_dir, project_id=PROJECT, delivered=c1)
    return project


def test_b17_only_lines_added_or_changed_since_delivery_count_and_an_unchanged_file_is_free(
    tmp_path: Path,
) -> None:
    project = _delivered_project(tmp_path, proportion=0.25)
    request(project, 'src/b.py')

    project.run()

    assert newest_tree(project).budget.counted_lines == 90
    assert [(line.type, line.path) for line in answers(project)] == [('file_revealed', 'src/b.py')]
    assert newest_tree(project).budget.spent_lines == 0


def test_b17_a_changed_file_is_charged_its_lines_since_delivery_not_its_lines_at_head(
    tmp_path: Path,
) -> None:
    project = _delivered_project(tmp_path, proportion=1.0)
    request(project, 'src/a.py')

    project.run()

    (revealed,) = answers(project)
    assert isinstance(revealed, FileRevealed) and revealed.lines == 260
    assert newest_tree(project).budget.spent_lines == 60
