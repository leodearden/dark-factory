"""Tests for scripts/verify_host_hygiene.py, driven as a subprocess.

The `mirror` fixture is a real git repo whose `.worktrees/` holds one
directory per verdict the tool can reach, made live or stale with real
flocks, real pgid records and real child processes.

Every run puts a fake `systemctl` first on PATH. It logs its argv and, by
default, lists no units, so no test can ever reach the real user manager.
"""
from __future__ import annotations

import errno
import fcntl
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import verify_host_hygiene as hygiene
from cli_subprocess_timeout import cli_timeout_from_env
from committed_git_repo import committed_repo
from dashboard.project_dbs import TASKS_DB_REL_PATH
from git_repo_probe import git
from orchestrator.git_ops import PROTECTED_PREFIXES, classify_worktree_entry
from verify_host_hygiene import SERVICE_MANAGER_COMM, Kind, ScopeMember, scope_is_orphaned

from orchestrator import verify_cancel

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
    repo = Path(committed_repo(tmp_path / "mirror", {"seed.txt": "seed\n"}))
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
    assert lock_holder.stdout is not None
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
    locks_before = lock_files(mirror)

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
    assert lock_files(mirror) == locks_before
    assert pgid_records(mirror) == records_before
    free = free_lines(result.stdout)
    assert free
    assert all("before=" in line and "after=" in line for line in free)


def with_unreadable_signals(mirror: Mirror) -> tuple[Path, Path]:
    """A merge dir whose `.lock` cannot be opened, and a task worktree git cannot read."""
    looped = mirror.wt("_merge-12121212")
    _plain_dir(looped)
    lock = mirror.base / f"{looped.name}.lock"
    lock.symlink_to(lock.name)
    broken = mirror.wt("125")
    git(mirror.repo, "worktree", "add", "--detach", str(broken))
    (broken / ".git").write_text("not a gitfile\n")
    _age(looped)
    _age(broken)
    return looped, broken


@pytest.mark.parametrize("apply", [False, True], ids=["report", "apply"])
def test_unreadable_signal_keeps_the_dir_and_says_why(mirror, apply):
    looped, broken = with_unreadable_signals(mirror)

    result = run(mirror, "--repo", str(mirror.repo), *WORKTREE_CLASSES,
                 *(["--apply"] if apply else []))

    assert result.returncode == 0, result.stderr
    table = verdicts(result.stdout)
    for path in (looped, broken):
        assert table[path.name][1:3] == ("keep", "unreadable"), path.name
        assert path.is_dir(), path.name
    stderr = result.stderr.splitlines()
    looped_line = next(line for line in stderr if str(looped) in line)
    assert f"{looped}.lock" in looped_line
    assert os.strerror(errno.ELOOP) in looped_line
    assert any(str(broken) in line for line in stderr)
    assert (mirror.base / f"{looped.name}.lock").is_symlink()


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
    (ScopeMember(pid=100, comm="bash", ppid=1, parent_comm=SERVICE_MANAGER_COMM),
     ScopeMember(pid=101, comm="sleep", ppid=0, parent_comm=None)),
    (ScopeMember(pid=100, comm="bash", ppid=50, parent_comm=SERVICE_MANAGER_COMM),
     ScopeMember(pid=101, comm="sleep", ppid=100, parent_comm="bash")),
], ids=["empty", "parents-are-pid-0-or-1", "parents-are-the-service-manager-or-members"])
def test_scope_without_a_live_controller_is_orphaned(members):
    assert scope_is_orphaned(members)


@pytest.mark.parametrize("members", [
    (ScopeMember(pid=100, comm="pytest", ppid=4242, parent_comm="python3"),),
    (ScopeMember(pid=100, comm="pytest", ppid=4242, parent_comm=None),),
    (ScopeMember(pid=100, comm="bash", ppid=50, parent_comm=SERVICE_MANAGER_COMM),
     ScopeMember(pid=101, comm="pytest", ppid=4242, parent_comm="python3")),
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
    "df-verify-x-deactivating.scope": "deactivating",
    "df-verify-x-inactive.scope": "inactive",
    "df-verify-x-odd-state.scope": "no-such-state",
}
SCOPE_CLASS = ("--class", "verify-scope")


def install_scope_units(mirror: Mirror, cgroup_root: Path,
                        units: dict[str, tuple[str, list[int] | None]]) -> None:
    """List *units* (unit -> state, cgroup pids or None for no cgroup.procs) in the fake systemctl."""
    listing = []
    for unit, (state, pids) in units.items():
        sub = "running" if state == "active" else state
        listing.append(f"{unit} loaded {state} {sub} verify scope")
        cgroup = f"user.slice/app.slice/{unit}"
        (mirror.state / f"cgroup.{unit}").write_text(f"/{cgroup}\n")
        (mirror.state / f"state.{unit}").write_text(state)
        (cgroup_root / cgroup).mkdir(parents=True)
        if pids is not None:
            (cgroup_root / cgroup / "cgroup.procs").write_text("".join(f"{p}\n" for p in pids))
    (mirror.state / "units").write_text("\n".join(listing) + "\n")


@pytest.fixture
def scopes(mirror, tmp_path):
    """The SCOPE_UNITS df-verify units in the fake systemctl, over a fake cgroup tree."""
    controller = _spawn([sys.executable, "-c", CONTROLLER], cwd=tmp_path,
                        stdout=subprocess.PIPE, text=True)
    mirror.children.append(controller)
    assert controller.stdout is not None
    live_pid = int(controller.stdout.readline())
    reaped = _spawn(["true"], cwd=tmp_path)
    reaped.wait()
    procs = {"df-verify-x-live.scope": [live_pid], "df-verify-x-dead.scope": [reaped.pid]}
    cgroup_root = tmp_path / "cgroup"
    install_scope_units(mirror, cgroup_root, {
        unit: (state, procs.get(unit)) for unit, state in SCOPE_UNITS.items()})
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
        "df-verify-x-deactivating.scope": ("verify-scope", "keep", "unit-transitioning"),
        "df-verify-x-inactive.scope": ("verify-scope", "keep", "unit-inactive"),
        "df-verify-x-odd-state.scope": ("verify-scope", "keep", "unit-state-unknown"),
    }
    assert scope_member_lines(report.stdout) == []
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


def scope_member_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("scope-member\t")]


PR_SET_NAME, PR_SET_CHILD_SUBREAPER = 15, 36
SUBREAPER = (
    "import ctypes, os, subprocess, sys\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    f"assert libc.prctl({PR_SET_CHILD_SUBREAPER}, 1, 0, 0, 0) == 0\n"
    f"assert libc.prctl({PR_SET_NAME}, {SERVICE_MANAGER_COMM.encode()!r}, 0, 0, 0) == 0\n"
    "controller = subprocess.Popen([sys.executable, '-c', sys.argv[1], *sys.argv[2:]],\n"
    "                              stdout=subprocess.PIPE, text=True)\n"
    "print(controller.pid, controller.stdout.readline().strip(), flush=True)\n"
    "while True:\n"
    "    os.wait()\n"
)
CONTROLLER_OF = (
    "import subprocess, sys\n"
    "children = [subprocess.Popen([program, '600']) for program in sys.argv[1:]]\n"
    "print(*(child.pid for child in children), flush=True)\n"
    "for child in children:\n"
    "    child.wait()\n"
)
REPARENTED_UNITS = {"df-verify-x-odd.scope": "a b) c", "df-verify-x-daemon.scope": "sccache"}


@dataclass(frozen=True)
class Reparenting:
    cgroup_root: Path
    controller: int
    members: dict[str, int]


@pytest.fixture
def reparenting(mirror, tmp_path) -> Reparenting:
    """One `sleep` copy per REPARENTED_UNITS comm, each alone in its unit's cgroup.

    Their controller runs under a subreaper named like the service manager, so
    killing the controller reparents them to it as systemd --user would.
    """
    sleep = shutil.which("sleep")
    assert sleep is not None
    programs = tmp_path / "programs"
    programs.mkdir()
    for comm in REPARENTED_UNITS.values():
        shutil.copy(sleep, programs / comm)
    helper = _spawn([sys.executable, "-c", SUBREAPER, CONTROLLER_OF,
                     *(str(programs / comm) for comm in REPARENTED_UNITS.values())],
                    cwd=tmp_path, stdout=subprocess.PIPE, text=True)
    mirror.children.append(helper)
    assert helper.stdout is not None
    controller, *member_pids = map(int, helper.stdout.readline().split())
    members = dict(zip(REPARENTED_UNITS, member_pids, strict=True))
    cgroup_root = tmp_path / "cgroup"
    install_scope_units(mirror, cgroup_root, {
        unit: ("active", [pid]) for unit, pid in members.items()})
    return Reparenting(cgroup_root=cgroup_root, controller=controller, members=members)


def wait_until_gone(pid: int) -> None:
    deadline = time.monotonic() + 10
    while Path(f"/proc/{pid}").exists():
        assert time.monotonic() < deadline, f"pid {pid} still present"
        time.sleep(0.05)


def test_reparented_scope_members_are_read_by_comm(mirror, reparenting):
    scope_args = (*SCOPE_CLASS, "--cgroup-root", str(reparenting.cgroup_root))
    alive = run(mirror, *scope_args)

    assert alive.returncode == 0, alive.stderr
    assert {unit: action_reason(verdicts(alive.stdout), unit) for unit in REPARENTED_UNITS} == {
        unit: ("verify-scope", "keep", "controller-alive") for unit in REPARENTED_UNITS}
    assert scope_member_lines(alive.stdout) == []

    os.kill(reparenting.controller, signal.SIGKILL)
    wait_until_gone(reparenting.controller)
    orphaned = run(mirror, *scope_args)

    assert orphaned.returncode == 0, orphaned.stderr
    assert {unit: action_reason(verdicts(orphaned.stdout), unit) for unit in REPARENTED_UNITS} == {
        "df-verify-x-odd.scope": ("verify-scope", "reclaim", "orphaned"),
        "df-verify-x-daemon.scope": ("verify-scope", "keep", "shared-daemon"),
    }
    assert sorted(scope_member_lines(orphaned.stdout)) == sorted(
        f"scope-member\t{unit}\t{reparenting.members[unit]}\t{comm}"
        for unit, comm in REPARENTED_UNITS.items())

    applied = run(mirror, *scope_args, "--apply")

    assert applied.returncode == 0, applied.stderr
    assert acting_calls(mirror) == ["--user kill --signal=SIGKILL df-verify-x-odd.scope",
                                    "--user stop df-verify-x-odd.scope"]


FAKE_SSH = (
    '#!/usr/bin/env bash\n'
    'printf "%s\\n" "$@" >"$FAKE_STATE/ssh.argv"\n'
    'shift\n'
    'exec bash -c "$*"\n'
)
REFUSING_SSH = (
    '#!/usr/bin/env bash\n'
    'echo "ssh: connect to host $1 port 22: Connection refused" >&2\n'
    'exit 255\n'
)


def install_shim(mirror: Mirror, name: str, body: str) -> Path:
    shim = mirror.bin / name
    shim.write_text(body)
    shim.chmod(0o755)
    return shim


def verdict_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.split("\t")[0] in KINDS]


def disk_state(mirror: Mirror) -> tuple:
    return (sorted(p.name for p in mirror.base.iterdir()),
            git(mirror.repo, "worktree", "list", "--porcelain"),
            lock_files(mirror), pgid_records(mirror))


def with_ignored_task_state(mirror: Mirror) -> Path:
    """Give the stale `123` an ignored `.task/` that the dirty guard cannot see."""
    exclude = mirror.repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a") as handle:
        handle.write(".task/\n")
    plan = mirror.wt("123") / ".task" / "plan.json"
    plan.parent.mkdir()
    plan.write_text("{}")
    _age(mirror.wt("123"))
    assert git(mirror.wt("123"), "status", "--porcelain") == ""
    return plan


def test_host_mode_runs_the_same_report_over_ssh(mirror):
    ssh = install_shim(mirror, "fakessh", FAKE_SSH)
    args = ("--repo", str(mirror.repo), *WORKTREE_CLASSES, "--min-age-hours", "24")
    local = run(mirror, *args)
    before = disk_state(mirror)

    remote = run(mirror, "--host", "fakehost", *args, SSH=str(ssh))

    assert remote.returncode == 0, remote.stderr
    assert verdict_lines(remote.stdout) == verdict_lines(local.stdout)
    assert verdict_lines(remote.stdout)
    argv = (mirror.state / "ssh.argv").read_text().splitlines()
    assert argv[:4] == ["fakehost", "python3", "-I", "-"]
    forwarded = argv[4:]
    assert "--host" not in forwarded
    assert forwarded[forwarded.index("--repo") + 1] == str(mirror.repo)
    classes = [forwarded[i + 1] for i, word in enumerate(forwarded) if word == "--class"]
    assert sorted(classes) == ["merge-worktree", "task-worktree"]
    assert float(forwarded[forwarded.index("--min-age-hours") + 1]) == 24
    assert "--apply" not in forwarded
    assert disk_state(mirror) == before


def test_host_mode_forwards_apply(mirror):
    ssh = install_shim(mirror, "fakessh", FAKE_SSH)

    remote = run(mirror, "--host", "fakehost", "--repo", str(mirror.repo), *WORKTREE_CLASSES,
                 "--apply", SSH=str(ssh))

    assert remote.returncode == 0, remote.stderr
    assert "--apply" in (mirror.state / "ssh.argv").read_text().splitlines()[4:]
    table = verdicts(remote.stdout)
    for name in APPLY_RECLAIMS:
        assert table[name][1:3] == ("reclaimed", "stale"), name
        assert not mirror.wt(name).exists(), name


def test_host_mode_relays_ssh_failure(mirror):
    ssh = install_shim(mirror, "refusingssh", REFUSING_SSH)

    result = run(mirror, "--host", "fakehost", "--repo", str(mirror.repo), SSH=str(ssh))

    assert result.returncode == 255
    assert "fakehost" in result.stderr


@pytest.mark.parametrize("apply", [False, True], ids=["report", "apply"])
@pytest.mark.parametrize("class_args", [
    [], ["--class", "task-worktree"], ["--class", "merge-worktree"],
], ids=["default-classes", "task-worktree", "merge-worktree"])
def test_orchestrator_project_root_refuses_the_worktree_classes(mirror, class_args, apply):
    store = mirror.repo / TASKS_DB_REL_PATH
    store.parent.mkdir(parents=True)
    store.touch()
    plan = with_ignored_task_state(mirror)
    before = disk_state(mirror)

    result = run(mirror, "--repo", str(mirror.repo), *class_args,
                 *(["--apply"] if apply else []))

    assert result.returncode == 2, result.stdout
    assert str(mirror.repo) in result.stderr
    assert str(store) in result.stderr
    assert verdict_lines(result.stdout) == []
    assert disk_state(mirror) == before
    assert plan.exists()
    assert systemctl_calls(mirror) == []


def test_verify_scope_class_alone_ignores_the_task_store(mirror):
    store = mirror.repo / TASKS_DB_REL_PATH
    store.parent.mkdir(parents=True)
    store.touch()

    result = run(mirror, "--repo", str(mirror.repo), *SCOPE_CLASS)

    assert result.returncode == 0, result.stderr


GIT_PROGRAM = "git"


def store_on_status_git(real_git: str) -> str:
    """A `git` that creates $STORE_ON_STATUS on its first `status`, then runs the real one."""
    return (
        '#!/usr/bin/env bash\n'
        'for word in "$@"; do\n'
        '  if [ "$word" = status ]; then\n'
        '    mkdir -p "$(dirname "$STORE_ON_STATUS")"\n'
        '    touch "$STORE_ON_STATUS"\n'
        '    break\n'
        '  fi\n'
        'done\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )


def test_task_store_appearing_mid_run_blocks_the_removal(mirror):
    real_git = shutil.which(GIT_PROGRAM)
    assert real_git is not None
    plan = with_ignored_task_state(mirror)
    install_shim(mirror, GIT_PROGRAM, store_on_status_git(real_git))
    store = mirror.repo / TASKS_DB_REL_PATH

    result = run(mirror, "--repo", str(mirror.repo), "--class", "task-worktree", "--apply",
                 STORE_ON_STATUS=str(store))

    assert result.returncode == 1, result.stderr
    assert verdicts(result.stdout)["123"][1:3] == ("failed", "stale")
    assert mirror.wt("123").is_dir()
    assert plan.exists()
    assert str(mirror.wt("123")) in result.stderr
    assert str(store) in result.stderr


def test_mirrored_orchestrator_facts_match_their_homes(tmp_path):
    base = tmp_path / ".worktrees"
    pgid_file = verify_cancel.pgid_file(base, "req1")
    lane = base / "_merge-abcdef12"

    assert hygiene.TASK_STORE_REL_PATH == TASKS_DB_REL_PATH
    assert verify_cancel.pgid_dir(base) == base / hygiene.PGID_DIR_NAME
    assert hygiene.worktree_record_path(pgid_file) == verify_cancel.worktree_path_file(pgid_file)
    assert hygiene.lane_lock_path(lane) == verify_cancel.lane_lock_path(lane)


@pytest.mark.parametrize("name", [
    "_merge-abcdef12", "_merge-verify", "123", "7269", "158",
    *(f"{prefix}x" for prefix in PROTECTED_PREFIXES),
])
def test_worktree_kind_refines_the_git_ops_band(name):
    band = classify_worktree_entry(name)
    kind = hygiene.worktree_kind(name)

    assert (kind is Kind.TASK_WORKTREE) == (band == "task")
    if kind is Kind.MERGE_WORKTREE:
        assert band == "merge"
