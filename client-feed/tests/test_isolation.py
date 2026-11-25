"""The isolation gate, plans/client-feed-prd.md §8: B1 confinement, B2 the two-project canary, B15 hostile paths, B22 end to end.

B1 and B15 record every effect of a pass with the audit hook. B2 and B22 walk what a pass wrote for anything that
is not the client's to see. client_feed.cli is imported here, at module level, so every first-party module is
loaded before a recorded span opens.
"""

import subprocess
from pathlib import Path

import pytest
from _audit_hook import AuditRecord, interpreter_roots, recording
from _call_watch import watching
from _two_projects import Project, TwoProjects, build_two_projects
from shared.project_layout import (
    KNOWN_PROJECT_ROOTS_ENV,
    default_project_root,
    runs_db,
    tasks_db,
    verify_logs_dir,
)

from client_feed.cli import main
from client_feed.contract import ENGAGEMENT_FILENAME, INBOX_FILENAME
from client_feed.publisher import PassReport

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


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
