"""What a reveal costs and whether a request is granted (PRD §4.12, §4.13; B12-B14, B17)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _git_repo import Repo, make_repo

from client_feed.budget import AddedLineCounts, Granted, counted_set, judge, lines_since
from client_feed.contract import FileKind, RefusalReason, RevealBudget, TreeFile
from client_feed.policy import Policy
from client_feed.sources.git import EntryKind, TreeEntry
from client_feed.sources.tree import visible_tree

PROJECT = 'acme'
T0 = datetime(2026, 10, 22, 9, 0, tzinfo=UTC)


def policy_of(**fields: Any) -> Policy:
    return Policy.model_validate(
        {
            'project_id': PROJECT,
            'project_root': '/srv/acme',
            'out_dir': '/srv/feed/out',
            'in_dir': '/srv/feed/in',
            'visible_globs': ['src/**', 'tests/**'],
            'reveal_proportion': 0.25,
            'reveal_floor_lines': 50,
            **fields,
        }
    )


def tree_file(path: str, kind: FileKind, lines: int) -> TreeFile:
    return TreeFile(path=path, kind=kind, lines=lines, bytes=lines * 10, last_changed_at=T0)


POLICY = policy_of()
B12_FILES = (
    tree_file('src/a.py', FileKind.PRODUCTION, 200),
    tree_file('src/b.py', FileKind.PRODUCTION, 210),
    tree_file('src/c.py', FileKind.PRODUCTION, 100),
    tree_file('src/d.py', FileKind.PRODUCTION, 10),
    tree_file('src/e.py', FileKind.PRODUCTION, 480),
)
FREE_FILES = (
    tree_file('tests/test_a.py', FileKind.TEST, 300),
    tree_file('src/guide.md', FileKind.DOCS, 400),
    tree_file('src/gen.py', FileKind.GENERATED, 500),
    tree_file('src/empty.py', FileKind.PRODUCTION, 0),
)


def test_only_production_lines_outside_free_and_never_reveal_globs_are_counted() -> None:
    policy = policy_of(
        visible_globs=['**'],
        reveal_proportion=0.3,
        reveal_floor_lines=40,
        free_globs=['core/src/free/**'],
        never_reveal_globs=['core/src/secret/**'],
    )
    files = (
        tree_file('core/src/a.py', FileKind.PRODUCTION, 100),
        tree_file('core/src/b.py', FileKind.PRODUCTION, 33),
        tree_file('core/src/free/c.py', FileKind.PRODUCTION, 1000),
        tree_file('core/src/secret/d.py', FileKind.PRODUCTION, 2000),
        tree_file('core/tests/test_a.py', FileKind.TEST, 400),
        tree_file('docs/a.md', FileKind.DOCS, 500),
        tree_file('core/src/gen.py', FileKind.GENERATED, 600),
    )

    counted = counted_set(policy, files)

    assert counted.lines == {'core/src/a.py': 100, 'core/src/b.py': 33}
    assert counted.budget(()) == RevealBudget(
        proportion=0.3, counted_lines=133, budget_lines=39, spent_lines=0, floor_lines=40
    )


def test_b12_a_reveal_costs_its_lines_and_at_least_the_floor() -> None:
    counted = counted_set(POLICY, B12_FILES)

    assert counted.counted_lines == 1000
    assert counted.budget_lines == 250
    assert counted.cost('src/a.py') == 200
    assert counted.cost('src/d.py') == 50
    assert counted.budget({'src/a.py'}) == RevealBudget(
        proportion=0.25, counted_lines=1000, budget_lines=250, spent_lines=200, floor_lines=50
    )


def test_b13_a_grant_needs_spent_plus_cost_within_the_budget() -> None:
    counted = counted_set(POLICY, B12_FILES)

    assert counted.spent({'src/b.py'}) == 210
    assert not counted.grantable('src/c.py', {'src/b.py'})
    assert not counted.grantable('src/d.py', {'src/b.py'})
    assert counted.spent({'src/a.py'}) == 200
    assert counted.grantable('src/d.py', {'src/a.py'})
    assert counted.spent({'src/a.py', 'src/d.py'}) == 250


@pytest.mark.parametrize('free', [file.path for file in FREE_FILES])
def test_b14_a_file_outside_the_counted_set_costs_nothing(free: str) -> None:
    counted = counted_set(POLICY, (*B12_FILES, *FREE_FILES))

    assert counted.cost(free) == 0
    assert counted.spent({'src/a.py', free}) == 200
    assert counted.spent({'src/a.py', 'src/d.py'}) == counted.budget_lines
    assert counted.grantable(free, {'src/a.py', 'src/d.py'})
    assert counted.counted_lines == 1000


def test_a_revealed_path_stays_grantable_after_its_growth_overspends_the_budget() -> None:
    grown = (tree_file('src/a.py', FileKind.PRODUCTION, 400), *B12_FILES[1:])
    counted = counted_set(POLICY, grown)
    revealed = {'src/a.py', 'src/b.py'}

    assert counted.spent(revealed) > counted.budget_lines
    assert counted.grantable('src/a.py', revealed)
    assert counted.grantable('src/b.py', revealed)
    assert not counted.grantable('src/d.py', revealed)


@pytest.mark.parametrize('free', [file.path for file in FREE_FILES])
def test_b14_a_free_file_stays_grantable_once_spent_exceeds_the_budget(free: str) -> None:
    grown = (tree_file('src/a.py', FileKind.PRODUCTION, 400), *B12_FILES[1:], *FREE_FILES)
    counted = counted_set(POLICY, grown)
    revealed = {'src/a.py', 'src/b.py'}

    assert counted.spent(revealed) > counted.budget_lines
    assert counted.grantable(free, revealed)
    assert counted.spent(revealed | {free}) == counted.spent(revealed)
    assert not counted.grantable('src/d.py', revealed)


@pytest.fixture
def delivered_repo(tmp_path: Path) -> tuple[Repo, Policy, str, str]:
    """c1 is delivered; c2 appends a blank line, appends a long line, adds a file and renames one unchanged."""
    repo = make_repo(tmp_path / 'repo')
    repo.write('src/same.py', 'one = 1\ntwo = 2\n')
    repo.write('src/blank.py', 'b = 1\n')
    repo.write('src/long.py', 'l = 1\n')
    repo.write('src/old.py', 'old = "a line long enough to be its own file"\nmore = 2\n')
    c1 = repo.commit('delivered', when=T0)
    repo.write('src/blank.py', 'b = 1\n\n')
    repo.write('src/long.py', 'l = 1\n' + 'x' * 100 + '\n')
    repo.write('src/new.py', ''.join(f'n{index} = {index}\n' for index in range(7)))
    repo.write('tests/test_new.py', 'def test_x():\n    pass\n')
    repo.git('mv', 'src/old.py', 'src/moved.py')
    c2 = repo.commit('since delivery', when=T0 + timedelta(hours=1))
    policy = policy_of(
        project_root=str(repo.root),
        out_dir=str(tmp_path / 'out'),
        in_dir=str(tmp_path / 'in'),
        max_line_bytes=40,
    )
    return repo, policy, c1, c2


def test_b17_only_lines_added_or_changed_since_delivery_are_counted(
    delivered_repo: tuple[Repo, Policy, str, str],
) -> None:
    repo, policy, c1, c2 = delivered_repo
    files = visible_tree(repo.root, policy, c2, previous=None, line_counts=None).files

    since = lines_since(repo.root, policy, files, delivered=c1, head=c2, cached=None).by_path
    counted = counted_set(policy, files, since=since)

    assert since['src/long.py'] == 3
    assert since['src/new.py'] == 7
    assert since.get('src/blank.py', 0) == 0
    assert since.get('src/moved.py', 0) == 0
    assert 'src/same.py' not in since
    assert counted.lines == {'src/long.py': 3, 'src/new.py': 7}
    assert counted.counted_lines == 10
    for outside in ('src/same.py', 'src/blank.py', 'src/moved.py', 'tests/test_new.py'):
        assert counted.cost(outside) == 0
    assert counted.cost('src/long.py') == 50


def _long_pair(repo: Repo, c1: str, c2: str) -> tuple[str, str]:
    return repo.rev(f'{c1}:src/long.py'), repo.rev(f'{c2}:src/long.py')


def test_lines_since_carries_the_count_of_each_blob_pair_it_diffed(
    delivered_repo: tuple[Repo, Policy, str, str],
) -> None:
    repo, policy, c1, c2 = delivered_repo
    files = visible_tree(repo.root, policy, c2, previous=None, line_counts=None).files
    old, new = _long_pair(repo, c1, c2)

    counts = lines_since(repo.root, policy, files, delivered=c1, head=c2, cached=None).counts

    assert counts.max_line_bytes == 40
    assert counts.by_blob_pair[old] == {new: 3}


def test_a_blob_pair_counted_under_the_same_max_line_bytes_is_not_diffed_again(
    delivered_repo: tuple[Repo, Policy, str, str],
) -> None:
    repo, policy, c1, c2 = delivered_repo
    files = visible_tree(repo.root, policy, c2, previous=None, line_counts=None).files
    old, new = _long_pair(repo, c1, c2)
    cached = AddedLineCounts(max_line_bytes=40, by_blob_pair={old: {new: 99}, 'f' * 40: {'e' * 40: 5}})

    since = lines_since(repo.root, policy, files, delivered=c1, head=c2, cached=cached)

    assert since.by_path['src/long.py'] == 99
    assert since.counts.by_blob_pair[old] == {new: 99}
    assert 'f' * 40 not in since.counts.by_blob_pair


def test_a_blob_pair_counted_under_another_max_line_bytes_is_counted_again(
    delivered_repo: tuple[Repo, Policy, str, str],
) -> None:
    repo, policy, c1, c2 = delivered_repo
    files = visible_tree(repo.root, policy, c2, previous=None, line_counts=None).files
    old, new = _long_pair(repo, c1, c2)
    cached = AddedLineCounts(max_line_bytes=400, by_blob_pair={old: {new: 99}})

    since = lines_since(repo.root, policy, files, delivered=c1, head=c2, cached=cached)

    assert since.by_path['src/long.py'] == 3


JUDGE_POLICY = policy_of(never_reveal_globs=['src/secret/**'])
TEXT_A = 'a = 1\n\nb = 2\n'
NOT_UTF8 = b'\xff\xfe\x00binary'


def _entry(path: str, kind: EntryKind, sha: str) -> TreeEntry:
    return TreeEntry(path=path, kind=kind, object_sha=sha, size=None if kind is EntryKind.SUBMODULE else 1)


HEAD_ENTRIES = {
    entry.path: entry
    for entry in (
        _entry('src/a.py', EntryKind.FILE, 'a' * 40),
        _entry('src/b.py', EntryKind.FILE, 'b' * 40),
        _entry('src/c.py', EntryKind.FILE, 'c' * 40),
        _entry('src/d.py', EntryKind.FILE, 'd' * 40),
        _entry('src/e.py', EntryKind.FILE, 'e' * 40),
        _entry('tests/test_a.py', EntryKind.FILE, '1' * 40),
        _entry('.mcp.json', EntryKind.FILE, '2' * 40),
        _entry('src/link.py', EntryKind.SYMLINK, '3' * 40),
        _entry('private/link.py', EntryKind.SYMLINK, '4' * 40),
        _entry('src/sub', EntryKind.SUBMODULE, '5' * 40),
        _entry('src/secret/k.py', EntryKind.FILE, '6' * 40),
        _entry('src/blob.bin', EntryKind.FILE, '7' * 40),
    )
}
BLOBS = {
    'a' * 40: TEXT_A.encode(),
    'b' * 40: b'b = 1\n',
    'c' * 40: b'c = 1\n',
    'd' * 40: b'd = 1\n',
    'e' * 40: b'e = 1\n',
    '1' * 40: b'def test_a():\n    pass\n',
    '2' * 40: b'{"secret": true}\n',
    '3' * 40: b'/etc/passwd',
    '4' * 40: b'/etc/passwd',
    '6' * 40: b'KEY = "k"\n',
    '7' * 40: NOT_UTF8,
}


class RecordingReader:
    """A blob reader over BLOBS that records every sha it is asked for."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, sha: str) -> bytes:
        self.asked.append(sha)
        return BLOBS[sha]


def judged(path: str, *, revealed: frozenset[str] = frozenset()) -> tuple[Granted | RefusalReason, list[str]]:
    reader = RecordingReader()
    verdict = judge(
        path,
        entries=HEAD_ENTRIES,
        policy=JUDGE_POLICY,
        counted=counted_set(JUDGE_POLICY, B12_FILES),
        revealed=revealed,
        read=reader,
    )
    return verdict, reader.asked


@pytest.mark.parametrize(
    'path', ['../x', '/etc/passwd', 'src/untracked.py', 'src', 'src//a.py', './src/a.py']
)
def test_b15_a_path_that_is_not_an_entry_of_heads_tree_is_not_in_tree(path: str) -> None:
    assert judged(path) == (RefusalReason.NOT_IN_TREE, [])


def test_b11_a_tracked_file_outside_the_visible_set_is_not_visible() -> None:
    assert judged('.mcp.json') == (RefusalReason.NOT_VISIBLE, [])


@pytest.mark.parametrize('path', ['src/link.py', 'src/sub'])
def test_a_visible_symlink_or_submodule_is_refused_as_symlink(path: str) -> None:
    assert judged(path) == (RefusalReason.SYMLINK, [])


def test_an_invisible_symlink_is_not_visible_so_its_entry_type_never_leaks() -> None:
    assert judged('private/link.py') == (RefusalReason.NOT_VISIBLE, [])


def test_a_never_reveal_path_is_not_revealable_and_its_blob_is_never_read() -> None:
    assert judged('src/secret/k.py') == (RefusalReason.NOT_REVEALABLE, [])


def test_a_blob_that_is_not_utf8_is_not_revealable() -> None:
    assert judged('src/blob.bin') == (RefusalReason.NOT_REVEALABLE, ['7' * 40])


@pytest.mark.parametrize('path', ['src/c.py', 'src/d.py'])
def test_b13_a_request_past_the_budget_is_over_budget(path: str) -> None:
    verdict, asked = judged(path, revealed=frozenset({'src/b.py'}))

    assert verdict is RefusalReason.OVER_BUDGET
    assert asked == [HEAD_ENTRIES[path].object_sha]


def test_a_grant_carries_the_decoded_text_and_its_counted_lines() -> None:
    assert judged('src/a.py') == (
        Granted(text=TEXT_A, lines=JUDGE_POLICY.counted_lines(TEXT_A.encode())),
        ['a' * 40],
    )
    assert JUDGE_POLICY.counted_lines(TEXT_A.encode()) == 2


def test_a_free_file_is_granted_reading_its_blob_once() -> None:
    verdict, asked = judged('tests/test_a.py', revealed=frozenset({'src/b.py'}))

    assert isinstance(verdict, Granted)
    assert asked == ['1' * 40]
