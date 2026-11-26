"""The isolation gate, plans/client-feed-prd.md §8: B1 confinement, B2 the two-project canary, B15 hostile paths, B22 end to end.

B1 and B15 record every effect of a pass with the audit hook. B2 and B22 walk what a pass wrote for anything that
is not the client's to see. client_feed.cli is imported here, at module level, so every first-party module is
loaded before a recorded span opens.
"""

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest
import yaml
from _audit_hook import AuditRecord, interpreter_roots, recording
from _call_watch import watching
from _git_repo import Repo, clone_of_checkout
from _in_dir import request_reveals, write_engagement
from _live_slice import copy_live_slice, live_titles
from _two_projects import CANARY_A_DESC, Project, TwoProjects, build_two_projects
from shared.project_layout import (
    KNOWN_PROJECT_ROOTS_ENV,
    default_project_root,
    main_checkout_of,
    runs_db,
    tasks_db,
    verify_logs_dir,
)

from client_feed.cli import main
from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    FEED_FILENAME,
    INBOX_FILENAME,
    STATE_FILENAME,
    FileRevealed,
    RefusalReason,
    RevealRefused,
    TaskUpserted,
    TreeSnapshot,
    VerifyTestResults,
    parse_state,
)
from client_feed.feed_io import read_feed_lines
from client_feed.fold import fold
from client_feed.policy import matches_any
from client_feed.publisher import PassReport
from client_feed.redact import redact_paths

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_KEYS: Final = frozenset({'description', 'details', 'test_strategy', 'metadata'})
REQUESTED_AT: Final = datetime(2026, 10, 21, 11, 0, tzinfo=UTC)
CLIENT_FEED_PRD: Final = 'plans/client-feed-prd.md'
SNAPSHOT_PROJECT: Final = 'dark-factory'
SNAPSHOT_VISIBLE_GLOBS: Final = ('*/src/**', '*/tests/**')
SNAPSHOT_FIXTURE_MAX_CASES: Final = 20
SNAPSHOT_REVEALED: Final = 'client-feed/src/client_feed/envelope.py'
FIXTURE_DIR: Final = Path(__file__).parent / 'fixtures' / 'snapshot-dark-factory'
FIXTURE_FILES: Final = (FEED_FILENAME, STATE_FILENAME)
REGENERATE_FIXTURE_ENV: Final = 'CLIENT_FEED_REGENERATE_SNAPSHOT_FIXTURE'


@pytest.fixture
def projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TwoProjects:
    """A and B, with B as the cwd and the only known project root: traps for any path not taken from the policy."""
    built = build_two_projects(tmp_path)
    monkeypatch.chdir(built.b.root)
    monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(built.b.root))
    return built


def _snapshot(policy: Path, capsys: pytest.CaptureFixture[str]) -> PassReport:
    code = main(['snapshot', '--config', str(policy)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return PassReport.model_validate_json(captured.out)


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    """Whether `path`, both as named and with every symlink followed, lies under one of `roots`."""
    anchors = {*roots, *(root.resolve() for root in roots)}
    return all(any(form.is_relative_to(anchor) for anchor in anchors) for form in (path, path.resolve()))


def _breaches(record: AuditRecord, scope: Project) -> list[str]:
    """Every recorded effect of a pass over `scope` that lies outside it, one readable line each."""
    readable = (scope.root, scope.out_dir, scope.in_dir, scope.policy)
    argument_roots = (scope.root, scope.out_dir, scope.in_dir)
    exempt = interpreter_roots()
    opens = record.opens_outside(exempt)
    found = [f'open {opened.path}' for opened in opens if not _inside(opened.path, readable)]
    found += [
        f'write-open {opened.path}'
        for opened in opens
        if opened.writes and not _inside(opened.path, (scope.out_dir,))
    ]
    found += [
        f'{mutated.event} {path}'
        for mutated in record.mutations_outside(exempt)
        for path in mutated.paths
        if not _inside(path, (scope.out_dir,))
    ]
    found += [
        f'sqlite3.connect {database.path}'
        for database in record.database_opens
        if database.path is not None and not _inside(database.path, (scope.root,))
    ]
    found += [f'socket.connect {connected.address}' for connected in record.connects]
    for spawn in record.spawns:
        if spawn.cwd is None or not _inside(spawn.cwd, (scope.root,)):
            found.append(f'{spawn.event} {spawn.target} in cwd {spawn.cwd}')
        found += [
            f'{spawn.event} {spawn.target} argument {argument}'
            for argument in spawn.args
            if argument.startswith('/') and not _inside(Path(argument), argument_roots)
        ]
    return found


def test_b1_a_snapshot_touches_only_its_own_root_out_and_in_dirs_and_policy(
    projects: TwoProjects, capsys: pytest.CaptureFixture[str]
) -> None:
    a = projects.a

    with recording() as record, watching(default_project_root) as calls:
        report = _snapshot(a.policy, capsys)

    assert report.degraded == ()
    databases = {database.path for database in record.database_opens}
    assert {tasks_db(a.root), runs_db(a.root)} <= databases
    opened = {opened.path for opened in record.opens}
    assert any(
        path.is_relative_to(verify_logs_dir(a.root)) and '.junit-' in path.name for path in opened
    )
    assert {a.in_dir / ENGAGEMENT_FILENAME, a.in_dir / INBOX_FILENAME} <= opened
    assert a.root in {spawn.cwd for spawn in record.spawns if spawn.event == 'subprocess.Popen'}
    assert any(opened.writes and opened.path.is_relative_to(a.out_dir) for opened in record.opens)
    assert _breaches(record, a) == []
    assert calls.count == 0


def test_no_client_feed_source_names_default_project_root() -> None:
    found = subprocess.run(
        ['git', 'grep', '-n', 'default_project_root', '--', 'client-feed/src'],
        cwd=CHECKOUT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (found.returncode, found.stdout) == (1, ''), found.stdout + found.stderr


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _json_documents(path: Path, text: str) -> list[Any]:
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)] if path.suffix == '.json' else []


def _leaks(
    out_dir: Path, *, needles: tuple[str, ...], forbidden_keys: frozenset[str] = FORBIDDEN_KEYS
) -> list[tuple[str, str]]:
    """(file, finding) for each needle in a file's name or text, and each forbidden key in its JSON documents.

    Keys are walked, never grepped for: a tree path may well contain the word 'description'.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(item for item in out_dir.rglob('*') if item.is_file()):
        name = path.relative_to(out_dir).as_posix()
        text = path.read_bytes().decode('utf-8', errors='replace')
        found += [(name, f'needle {needle!r}') for needle in needles if needle in name or needle in text]
        found += [
            (name, f'key {key!r}')
            for document in _json_documents(path, text)
            for key in sorted(forbidden_keys & set(_keys(document)))
        ]
    return found


def test_b2_a_snapshot_of_a_carries_nothing_of_b_and_nothing_a_hides(
    projects: TwoProjects, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = projects.a, projects.b

    assert _snapshot(a.policy, capsys).degraded == ()

    assert _leaks(a.out_dir, needles=(projects.canary_b, 'CANARY-B', CANARY_A_DESC)) == []
    assert _snapshot(b.policy, capsys).degraded == ()
    assert projects.canary_b in (b.out_dir / FEED_FILENAME).read_text()


def test_b15_hostile_reveal_requests_are_refused_and_never_opened(
    projects: TwoProjects, capsys: pytest.CaptureFixture[str]
) -> None:
    a = projects.a
    assert a.tracked_symlink is not None and a.untracked is not None
    decoy = a.root.parent / 'x'
    decoy.write_text('decoy\n')
    hostile = ('../x', '/etc/passwd', a.tracked_symlink, a.untracked)
    request_reveals(a.in_dir, *hostile, at=REQUESTED_AT)

    with recording() as record:
        _snapshot(a.policy, capsys)

    lines = read_feed_lines(a.out_dir)
    refused = [(line.path, line.reason) for line in lines if isinstance(line, RevealRefused)]
    assert refused == list(zip(hostile, (
        RefusalReason.NOT_IN_TREE, RefusalReason.NOT_IN_TREE, RefusalReason.SYMLINK, RefusalReason.NOT_IN_TREE,
    ), strict=True))
    assert not [line for line in lines if isinstance(line, FileRevealed) and line.path in hostile]
    assert _breaches(record, a) == []
    never_opened = {decoy, Path('/etc/passwd'), (a.root / a.tracked_symlink).resolve(), a.root / a.untracked}
    opened = {form for opened in record.opens for form in (opened.path, opened.path.resolve())}
    assert never_opened.isdisjoint(opened)


def _prd_title(repo: Repo) -> str:
    markdown = repo.git('show', f'main:{CLIENT_FEED_PRD}')
    return next(line.removeprefix('# ').strip() for line in markdown.splitlines() if line.startswith('# '))


def test_b22_a_snapshot_of_this_checkout_over_its_live_slice_is_valid_and_leaks_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    live = main_checkout_of(CHECKOUT_ROOT)
    if not tasks_db(live).is_file():
        pytest.skip(f'no live task store at {tasks_db(live)}')
    repo = clone_of_checkout(CHECKOUT_ROOT, tmp_path / 'clone')
    ids = copy_live_slice(live, repo.root, prd_path=CLIENT_FEED_PRD, marker=CANARY_A_DESC)
    in_dir, out_dir, policy = tmp_path / 'in', tmp_path / 'out', tmp_path / 'policy.yaml'
    write_engagement(in_dir, project_id=SNAPSHOT_PROJECT, delivered=None)
    request_reveals(in_dir, SNAPSHOT_REVEALED, '../x', at=REQUESTED_AT)
    out_dir.mkdir()
    policy.write_text(yaml.safe_dump({
        'project_id': SNAPSHOT_PROJECT,
        'project_root': str(repo.root),
        'out_dir': str(out_dir),
        'in_dir': str(in_dir),
        'visible_globs': list(SNAPSHOT_VISIBLE_GLOBS),
        'max_cases_per_run': SNAPSHOT_FIXTURE_MAX_CASES,
    }))

    _snapshot(policy, capsys)

    validated = main(['validate-feed', '--out-dir', str(out_dir), '--project-id', SNAPSHOT_PROJECT])
    assert validated == 0, capsys.readouterr().out
    lines = read_feed_lines(out_dir)
    state = parse_state((out_dir / STATE_FILENAME).read_text())
    assert state == fold(lines, project_id=SNAPSHOT_PROJECT)
    assert state.prds
    assert state.prds[CLIENT_FEED_PRD].title == _prd_title(repo)
    upserts = [line for line in lines if isinstance(line, TaskUpserted)]
    assert upserts
    assert {line.task_id for line in upserts} <= set(ids)
    titles = live_titles(live, ids)
    assert [(line.task_id, line.title) for line in upserts] == [
        (line.task_id, redact_paths(titles[line.task_id])) for line in upserts
    ]
    assert [line.path for line in lines if isinstance(line, FileRevealed)] == [SNAPSHOT_REVEALED]
    refused = [(line.path, line.reason) for line in lines if isinstance(line, RevealRefused)]
    assert refused == [('../x', RefusalReason.NOT_IN_TREE)]
    trees = [line for line in lines if isinstance(line, TreeSnapshot)]
    assert trees and trees[-1].files
    assert all(matches_any(SNAPSHOT_VISIBLE_GLOBS, file.path) for tree in trees for file in tree.files)
    assert _leaks(out_dir, needles=(CANARY_A_DESC, '/home/', str(tmp_path))) == []
    if os.environ.get(REGENERATE_FIXTURE_ENV) == '1':
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for name in FIXTURE_FILES:
            shutil.copyfile(out_dir / name, FIXTURE_DIR / name)


def test_the_committed_dark_factory_snapshot_is_a_valid_bounded_leak_free_feed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert {path.name for path in FIXTURE_DIR.iterdir()} == {*FIXTURE_FILES, 'README.md'}
    validated = main(['validate-feed', '--out-dir', str(FIXTURE_DIR), '--project-id', SNAPSHOT_PROJECT])
    assert validated == 0, capsys.readouterr().out
    assert _leaks(FIXTURE_DIR, needles=('CANARY-', '/home/', '/tmp/', 'pytest-of-')) == []
    assert parse_state((FIXTURE_DIR / STATE_FILENAME).read_text()).prds
    lines = read_feed_lines(FIXTURE_DIR)
    for kind in (TaskUpserted, TreeSnapshot, FileRevealed):
        assert any(isinstance(line, kind) for line in lines), kind.__name__
    results = [line for line in lines if isinstance(line, VerifyTestResults)]
    assert all(len(line.cases) <= SNAPSHOT_FIXTURE_MAX_CASES for line in results)
