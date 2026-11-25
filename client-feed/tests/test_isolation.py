"""The isolation gate, plans/client-feed-prd.md §8: B1 confinement, B2 the two-project canary, B15 hostile paths, B22 end to end.

B1 and B15 record every effect of a pass with the audit hook. B2 and B22 walk what a pass wrote for anything that
is not the client's to see. client_feed.cli is imported here, at module level, so every first-party module is
loaded before a recorded span opens.
"""

import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pytest
from _audit_hook import AuditRecord, interpreter_roots, recording
from _call_watch import watching
from _in_dir import request_reveals
from _two_projects import CANARY_A_DESC, Project, TwoProjects, build_two_projects
from shared.project_layout import (
    KNOWN_PROJECT_ROOTS_ENV,
    default_project_root,
    runs_db,
    tasks_db,
    verify_logs_dir,
)

from client_feed.cli import main
from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    FEED_FILENAME,
    INBOX_FILENAME,
    FileRevealed,
    RefusalReason,
    RevealRefused,
)
from client_feed.feed_io import read_feed_lines
from client_feed.publisher import PassReport

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_KEYS: Final = frozenset({'description', 'details', 'test_strategy', 'metadata'})
REQUESTED_AT: Final = datetime(2026, 10, 21, 11, 0, tzinfo=UTC)


@pytest.fixture
def projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TwoProjects:
    """A and B, with B as the cwd and the only known project root: traps for any path not taken from the policy."""
    built = build_two_projects(tmp_path)
    monkeypatch.chdir(built.b.root)
    monkeypatch.setenv(KNOWN_PROJECT_ROOTS_ENV, str(built.b.root))
    return built


def _snapshot(project: Project, capsys: pytest.CaptureFixture[str]) -> PassReport:
    code = main(['snapshot', '--config', str(project.policy)])
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
        report = _snapshot(a, capsys)

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

    assert _snapshot(a, capsys).degraded == ()

    assert _leaks(a.out_dir, needles=(projects.canary_b, 'CANARY-B', CANARY_A_DESC)) == []
    assert _snapshot(b, capsys).degraded == ()
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
        _snapshot(a, capsys)

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
