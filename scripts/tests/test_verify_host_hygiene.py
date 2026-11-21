"""Tests for scripts/verify_host_hygiene.py, driven as a subprocess.

The `mirror` fixture is a real git repo whose `.worktrees/` holds one
directory per verdict the tool can reach, made live or stale with real
flocks, real pgid records and real child processes.

Every run puts a fake `systemctl` first on PATH. It logs its argv and, by
default, lists no units, so no test can ever reach the real user manager.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from committed_git_repo import committed_repo
from git_repo_probe import git

SCRIPT = Path(__file__).resolve().parent.parent / "verify_host_hygiene.py"
CLI_TIMEOUT = cli_timeout_from_env("VERIFY_HOST_HYGIENE_CLI_TIMEOUT")
OLD_SECONDS = 48 * 3600
KINDS = {"merge-worktree", "task-worktree", "verify-scope"}
WORKTREE_CLASSES = ("--class", "merge-worktree", "--class", "task-worktree")

FAKE_SYSTEMCTL = (
    '#!/usr/bin/env bash\n'
    'echo "$*" >>"$FAKE_STATE/systemctl.log"\n'
    'case "$*" in\n'
    '  *list-units*) [ -f "$FAKE_STATE/units" ] && cat "$FAKE_STATE/units" ;;\n'
    'esac\n'
    'exit 0\n'
)

LOCK_HOLDER = (
    "import fcntl, os, sys, time\n"
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
    "fcntl.flock(fd, fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "time.sleep(600)\n"
)


@dataclass
class Mirror:
    repo: Path
    bin: Path
    state: Path
    lock_holder: subprocess.Popen
    children: list[subprocess.Popen] = field(default_factory=list)

    @property
    def base(self) -> Path:
        return self.repo / ".worktrees"

    @property
    def pgid_dir(self) -> Path:
        return self.base / ".merge_verify_pgids"

    def wt(self, name: str) -> Path:
        return self.base / name


def _age(path: Path) -> None:
    then = time.time() - OLD_SECONDS
    for entry in [path, *path.iterdir()]:
        os.utime(entry, (then, then), follow_symlinks=False)


def _spawn(argv: list[str], cwd: Path, **kwargs) -> subprocess.Popen:
    return subprocess.Popen(argv, cwd=cwd, start_new_session=True, **kwargs)


def _plain_dir(path: Path) -> None:
    path.mkdir()
    (path / "leftover.txt").write_text("x" * 8192)


def build_mirror(tmp_path: Path) -> Mirror:
    repo = Path(committed_repo(tmp_path / "mirror", {"README.md": "seed\n"}))
    base = repo / ".worktrees"
    base.mkdir()
    registered = ["_merge-aaaaaaaa", "_merge-bbbbbbbb", "_merge-cccccccc",
                  "_merge-dddddddd", "_merge-eeeeeeee", "_merge-99999999",
                  "_merge-verify", "_offline-deep", "123", "124"]
    for name in registered:
        git(repo, "worktree", "add", "--detach", str(base / name))
    git(repo, "worktree", "lock", str(base / "_merge-99999999"))
    (base / "124" / "scratch.txt").write_text("uncommitted\n")
    _plain_dir(base / "_merge-ffffffff")
    _plain_dir(base / "158")

    lock_holder = _spawn(
        [sys.executable, "-c", LOCK_HOLDER, str(base / "_merge-cccccccc.lock")],
        cwd=tmp_path, stdout=subprocess.PIPE, text=True,
    )
    assert lock_holder.stdout.readline().strip() == "locked"
    mirror = Mirror(repo=repo, bin=tmp_path / "bin", state=tmp_path / "state",
                    lock_holder=lock_holder, children=[lock_holder])

    pgid_owner = _spawn(["sleep", "600"], cwd=tmp_path)
    cwd_child = _spawn(["sleep", "600"], cwd=base / "_merge-eeeeeeee")
    dead = _spawn(["true"], cwd=tmp_path)
    dead.wait()
    mirror.children += [pgid_owner, cwd_child]
    pgids = mirror.pgid_dir
    pgids.mkdir()
    (pgids / "req1").write_text(str(pgid_owner.pid))
    (pgids / "req1.worktree").write_text(str(base / "_merge-dddddddd"))
    (pgids / "req2").write_text(str(dead.pid))
    (pgids / "req2.worktree").write_text(str(base / "_merge-aaaaaaaa"))

    for entry in base.iterdir():
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != "_merge-bbbbbbbb":
            _age(entry)

    mirror.bin.mkdir()
    mirror.state.mkdir()
    shim = mirror.bin / "systemctl"
    shim.write_text(FAKE_SYSTEMCTL)
    shim.chmod(0o755)
    return mirror


@pytest.fixture
def mirror(tmp_path):
    built = build_mirror(tmp_path)
    yield built
    for child in built.children:
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait()


def run(mirror: Mirror, *args: str, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PATH": f"{mirror.bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_STATE": str(mirror.state),
        **env_overrides,
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], env=env, cwd=mirror.repo.parent,
        capture_output=True, text=True, timeout=CLI_TIMEOUT,
    )


def verdicts(stdout: str) -> dict[str, tuple[str, str, str, int]]:
    """Target basename -> (kind, action, reason, bytes) for each verdict line."""
    table = {}
    for line in stdout.splitlines():
        fields = line.split("\t")
        if fields[0] in KINDS and len(fields) == 5:
            kind, action, reason, size, target = fields
            table[Path(target).name] = (kind, action, reason, int(size))
    return table


def action_reason(table: dict, name: str) -> tuple[str, str, str]:
    kind, action, reason, _ = table[name]
    return kind, action, reason


def systemctl_calls(mirror: Mirror) -> list[str]:
    log = mirror.state / "systemctl.log"
    return log.read_text().splitlines() if log.exists() else []


def pgid_records(mirror: Mirror) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(mirror.pgid_dir.iterdir())}


def lock_files(mirror: Mirror) -> set[str]:
    return {p.name for p in mirror.base.glob("*.lock")}


EXPECTED_REPORT = {
    "_merge-aaaaaaaa": ("merge-worktree", "reclaim", "stale"),
    "_merge-ffffffff": ("merge-worktree", "reclaim", "stale"),
    "_merge-bbbbbbbb": ("merge-worktree", "keep", "too-young"),
    "_merge-cccccccc": ("merge-worktree", "keep", "lock-held"),
    "_merge-dddddddd": ("merge-worktree", "keep", "pgid-live"),
    "_merge-eeeeeeee": ("merge-worktree", "keep", "process-cwd"),
    "_merge-99999999": ("merge-worktree", "keep", "git-locked"),
    "123": ("task-worktree", "reclaim", "stale"),
    "124": ("task-worktree", "keep", "dirty"),
    "158": ("task-worktree", "keep", "unregistered"),
}


def test_report_mode_classifies_and_changes_nothing(mirror):
    dirs_before = sorted(p.name for p in mirror.base.iterdir())
    listing_before = git(mirror.repo, "worktree", "list", "--porcelain")
    records_before = pgid_records(mirror)
    locks_before = lock_files(mirror)

    result = run(mirror, "--repo", str(mirror.repo), *WORKTREE_CLASSES)

    assert result.returncode == 0, result.stderr
    table = verdicts(result.stdout)
    assert {name: action_reason(table, name) for name in table} == EXPECTED_REPORT
    assert table["_merge-aaaaaaaa"][3] > 0
    assert table["_merge-ffffffff"][3] > 0
    for line in result.stdout.splitlines():
        assert "_merge-verify" not in line
        assert "_offline-deep" not in line
        assert ".merge_verify_pgids" not in line
    summary = [line for line in result.stdout.splitlines() if line.startswith("pgid-records")]
    assert len(summary) == 1
    assert "total=2" in summary[0]
    assert "dead=1" in summary[0]
    assert "left for task 4196" in summary[0]

    assert sorted(p.name for p in mirror.base.iterdir()) == dirs_before
    assert git(mirror.repo, "worktree", "list", "--porcelain") == listing_before
    assert lock_files(mirror) == locks_before
    assert pgid_records(mirror) == records_before
    assert systemctl_calls(mirror) == []


def test_repo_that_is_not_a_git_toplevel_is_a_usage_error(mirror, tmp_path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    result = run(mirror, "--repo", str(not_a_repo))

    assert result.returncode == 2
    assert str(not_a_repo) in result.stderr
