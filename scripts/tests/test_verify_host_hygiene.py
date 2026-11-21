"""Tests for scripts/verify_host_hygiene.py, driven as a subprocess.

The `mirror` fixture is a real git repo whose `.worktrees/` holds one
directory per verdict the tool can reach, made live or stale with real
flocks, real pgid records and real child processes.

Every run puts a fake `systemctl` first on PATH. It logs its argv and, by
default, lists no units, so no test can ever reach the real user manager.
"""
from __future__ import annotations

import fcntl
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
from verify_host_hygiene import ScopeMember, scope_is_orphaned

SCRIPT = Path(__file__).resolve().parent.parent / "verify_host_hygiene.py"
CLI_TIMEOUT = cli_timeout_from_env("VERIFY_HOST_HYGIENE_CLI_TIMEOUT")
OLD_SECONDS = 48 * 3600
KINDS = {"merge-worktree", "task-worktree", "verify-scope"}
WORKTREE_CLASSES = ("--class", "merge-worktree", "--class", "task-worktree")

FAKE_SYSTEMCTL = (
    '#!/usr/bin/env bash\n'
    'echo "$*" >>"$FAKE_STATE/systemctl.log"\n'
    'unit="${@: -1}"\n'
    'case "$*" in\n'
    '  *list-units*) [ -f "$FAKE_STATE/units" ] && cat "$FAKE_STATE/units" ;;\n'
    '  "--user show -p ControlGroup --value "*) cat "$FAKE_STATE/cgroup.$unit" ;;\n'
    '  "--user is-active "*)\n'
    '    state=$(cat "$FAKE_STATE/state.$unit" 2>/dev/null || echo inactive)\n'
    '    echo "$state"; [ "$state" = active ] || exit 3 ;;\n'
    '  "--user stop "*)\n'
    '    [ -f "$FAKE_STATE/sticky.$unit" ] || echo inactive >"$FAKE_STATE/state.$unit" ;;\n'
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


APPLY_KEEPS = {name: verdict for name, verdict in EXPECTED_REPORT.items()
               if verdict[1] == "keep"}
APPLY_RECLAIMS = {"_merge-aaaaaaaa", "_merge-ffffffff", "123"}


def lock_is_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


def free_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("free\t")]


def test_apply_reclaims_only_stale_worktrees(mirror):
    records_before = pgid_records(mirror)

    result = run(mirror, "--repo", str(mirror.repo), "--apply", *WORKTREE_CLASSES)

    assert result.returncode == 0, result.stderr
    table = verdicts(result.stdout)
    for name in APPLY_RECLAIMS:
        assert table[name][1:3] == ("reclaimed", "stale"), name
        assert not mirror.wt(name).exists(), name
    listing = git(mirror.repo, "worktree", "list", "--porcelain")
    assert "_merge-aaaaaaaa" not in listing
    assert f"{mirror.base}/123\n" not in listing
    assert git(mirror.repo, "worktree", "prune", "--dry-run", "--verbose") == ""
    assert not (mirror.base / "_merge-aaaaaaaa.lock").exists()
    for name, verdict in APPLY_KEEPS.items():
        assert action_reason(table, name) == verdict, name
        assert mirror.wt(name).is_dir(), name
    assert mirror.wt("_merge-verify").is_dir()
    assert (mirror.base / "_merge-cccccccc.lock").exists()
    assert mirror.lock_holder.poll() is None
    assert not lock_is_free(mirror.base / "_merge-cccccccc.lock")
    assert pgid_records(mirror) == records_before
    free = free_lines(result.stdout)
    assert free
    assert all("before=" in line and "after=" in line for line in free)


@pytest.fixture
def unremovable(mirror):
    victim = mirror.wt("_merge-77777777")
    sealed = victim / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "pinned.txt").write_text("cannot unlink me\n")
    sealed.chmod(0o500)
    _age(victim)
    yield victim
    sealed.chmod(0o700)


def test_apply_failure_exits_nonzero_and_reports_failed(mirror, unremovable):
    result = run(mirror, "--repo", str(mirror.repo), "--apply", *WORKTREE_CLASSES)

    assert result.returncode == 1
    table = verdicts(result.stdout)
    assert table[unremovable.name][1] == "failed"
    assert str(unremovable) in result.stderr
    for name in APPLY_RECLAIMS:
        assert table[name][1:3] == ("reclaimed", "stale"), name
        assert not mirror.wt(name).exists(), name


@pytest.mark.parametrize("members", [
    (),
    (ScopeMember(pid=100, ppid=1, parent_comm="systemd"),
     ScopeMember(pid=101, ppid=0, parent_comm=None)),
    (ScopeMember(pid=100, ppid=50, parent_comm="systemd"),
     ScopeMember(pid=101, ppid=100, parent_comm="bash")),
], ids=["empty", "parents-are-pid-0-or-1", "parents-are-systemd-or-members"])
def test_scope_without_a_live_controller_is_orphaned(members):
    assert scope_is_orphaned(members)


@pytest.mark.parametrize("members", [
    (ScopeMember(pid=100, ppid=4242, parent_comm="python3"),),
    (ScopeMember(pid=100, ppid=4242, parent_comm=None),),
    (ScopeMember(pid=100, ppid=50, parent_comm="systemd"),
     ScopeMember(pid=101, ppid=4242, parent_comm="python3")),
], ids=["python-parent", "unreadable-parent", "one-of-two-roots-has-a-controller"])
def test_scope_with_a_live_or_unknown_controller_is_not_orphaned(members):
    assert not scope_is_orphaned(members)


CONTROLLER = (
    "import subprocess\n"
    "child = subprocess.Popen(['sleep', '600'])\n"
    "print(child.pid, flush=True)\n"
    "child.wait()\n"
)
SCOPE_UNITS = {
    "df-verify-x-live.scope": "active",
    "df-verify-x-dead.scope": "active",
    "df-verify-x-failed.scope": "failed",
    "df-verify-x-unreadable.scope": "active",
}
SCOPE_CLASS = ("--class", "verify-scope")


@pytest.fixture
def scopes(mirror, tmp_path):
    """Four df-verify units in the fake systemctl, over a fake cgroup tree."""
    controller = _spawn([sys.executable, "-c", CONTROLLER], cwd=tmp_path,
                        stdout=subprocess.PIPE, text=True)
    mirror.children.append(controller)
    live_pid = int(controller.stdout.readline())
    reaped = _spawn(["true"], cwd=tmp_path)
    reaped.wait()
    procs = {"df-verify-x-live.scope": f"{live_pid}\n",
             "df-verify-x-dead.scope": f"{reaped.pid}\n"}
    cgroup_root = tmp_path / "cgroup"
    listing = []
    for unit, state in SCOPE_UNITS.items():
        sub = "running" if state == "active" else state
        listing.append(f"{unit} loaded {state} {sub} verify scope")
        cgroup = f"user.slice/app.slice/{unit}"
        (mirror.state / f"cgroup.{unit}").write_text(f"/{cgroup}\n")
        (mirror.state / f"state.{unit}").write_text(state)
        (cgroup_root / cgroup).mkdir(parents=True)
        if unit in procs:
            (cgroup_root / cgroup / "cgroup.procs").write_text(procs[unit])
    (mirror.state / "units").write_text("\n".join(listing) + "\n")
    return cgroup_root


def acting_calls(mirror: Mirror) -> list[str]:
    verbs = ("--user kill ", "--user stop ", "--user reset-failed ")
    return [call for call in systemctl_calls(mirror) if call.startswith(verbs)]


def test_scope_class_reclaims_only_orphaned_units(mirror, scopes):
    report = run(mirror, *SCOPE_CLASS, "--cgroup-root", str(scopes))

    assert report.returncode == 0, report.stderr
    assert {name: action_reason(verdicts(report.stdout), name) for name in SCOPE_UNITS} == {
        "df-verify-x-live.scope": ("verify-scope", "keep", "controller-alive"),
        "df-verify-x-dead.scope": ("verify-scope", "reclaim", "orphaned"),
        "df-verify-x-failed.scope": ("verify-scope", "reclaim", "unit-failed"),
        "df-verify-x-unreadable.scope": ("verify-scope", "keep", "cgroup-unreadable"),
    }
    assert acting_calls(mirror) == []

    applied = run(mirror, *SCOPE_CLASS, "--cgroup-root", str(scopes), "--apply")

    assert applied.returncode == 0, applied.stderr
    assert sorted(acting_calls(mirror)) == sorted([
        "--user kill --signal=SIGKILL df-verify-x-dead.scope",
        "--user stop df-verify-x-dead.scope",
        "--user reset-failed df-verify-x-failed.scope",
    ])
    table = verdicts(applied.stdout)
    assert table["df-verify-x-dead.scope"][1:3] == ("reclaimed", "orphaned")
    assert table["df-verify-x-failed.scope"][1:3] == ("reclaimed", "unit-failed")
    assert table["df-verify-x-live.scope"][1:3] == ("keep", "controller-alive")
    assert table["df-verify-x-unreadable.scope"][1:3] == ("keep", "cgroup-unreadable")


def test_scope_that_survives_the_reap_is_failed(mirror, scopes):
    (mirror.state / "sticky.df-verify-x-dead.scope").write_text("")

    result = run(mirror, *SCOPE_CLASS, "--cgroup-root", str(scopes), "--apply")

    assert result.returncode == 1
    assert verdicts(result.stdout)["df-verify-x-dead.scope"][1:3] == ("failed", "orphaned")
