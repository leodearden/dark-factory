"""Tests for scripts/shared_repo_git_maintenance.py — the nightly incremental
repack + commit-graph of every orchestrator-managed shared repo (task 8076).

Every test drives ``main(argv)`` in-process against REAL tmp git repos: the risk
this job carries is git behaviour (what repack deletes, whether refs move), so a
fake git would test nothing that matters. The one fake is a ``nice`` recorder
that execs the real binary. Each run names its roots, an empty units dir and a
tmp lock file, so the host's orchestrator units and lock are never read.

``import shared_repo_git_maintenance`` resolves via scripts/tests/conftest.py's
sys.path insertion.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import shared_repo_git_maintenance as maintenance
from committed_git_repo import committed_repo
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

NICED_GIT = ["-n", "19", "ionice", "-c3", "git"]


def _git(repo: Path, *args: str) -> str:
    """``git -C <repo> <args>`` stdout; raises on a non-zero exit."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout


def _repo_with_loose_objects(root: Path) -> Path:
    """A real repo whose history is several commits of LOOSE reachable objects."""
    committed_repo(root, {"README.md": "seed\n"})
    for key, value in (("user.email", "t@e"), ("user.name", "t"), ("commit.gpgsign", "false")):
        _git(root, "config", key, value)
    for n in range(3):
        _commit_file(root, f"f{n}.txt", f"{n}\n")
    return root


def _commit_file(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "-m", name)
    return _git(repo, "rev-parse", "HEAD").strip()


def _loose_count(repo: Path) -> int:
    for line in _git(repo, "count-objects", "-v").splitlines():
        key, _, value = line.partition(":")
        if key == "count":
            return int(value)
    raise AssertionError(f"count-objects -v printed no count for {repo}")


def _commit_graph(repo: Path) -> Path:
    return Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / "objects" / "info" / "commit-graph"


def _object_exists(repo: Path, oid: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", oid], capture_output=True, timeout=60
    ).returncode == 0


def _hermetic_argv(tmp_path: Path, *roots: Path) -> list[str]:
    """Explicit roots, an empty units dir and a tmp lock: nothing of the host's is read."""
    units = tmp_path / "no-units"
    units.mkdir(exist_ok=True)
    argv = [arg for root in roots for arg in ("--project-root", str(root))]
    return [*argv, "--units-dir", str(units), "--lock-file", str(tmp_path / "maint.lock")]


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, list[dict], str]:
    rc = maintenance.main(argv)
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
    return rc, records, captured.err


def test_maintains_a_repo_packing_its_loose_objects_and_writing_the_commit_graph(tmp_path, capsys):
    repo = _repo_with_loose_objects(tmp_path / "repo")
    assert _loose_count(repo) > 0
    assert not _commit_graph(repo).exists()

    rc, records, _err = _run(capsys, _hermetic_argv(tmp_path, repo))

    assert rc == 0
    assert _loose_count(repo) == 0
    assert _commit_graph(repo).exists()
    [record] = records
    assert record["root"] == str(repo.resolve())
    assert record["outcome"] == "maintained"
    assert record["repack_rc"] == 0
    assert record["commit_graph_rc"] == 0
    assert record["loose_before"] > record["loose_after"]


def test_never_deletes_unreachable_objects_packed_or_loose(tmp_path, capsys):
    """`repack -a -d` drops a packed unreachable commit (measured, git 2.43.0);
    plain `repack -d` keeps it. This fails if the job is ever 'upgraded' to -a."""
    repo = _repo_with_loose_objects(tmp_path / "repo")
    _git(repo, "checkout", "-q", "-b", "side")
    side_commit = _commit_file(repo, "side.txt", "side\n")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "repack", "-a", "-d", "-q")
    _git(repo, "branch", "-q", "-D", "side")
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    blob_source = tmp_path / "unreachable.txt"
    blob_source.write_text("unreachable\n", encoding="utf-8")
    loose_blob = _git(repo, "hash-object", "-w", str(blob_source)).strip()
    _commit_file(repo, "after.txt", "after\n")

    _run(capsys, _hermetic_argv(tmp_path, repo))

    assert _object_exists(repo, side_commit), "the packed unreachable commit was deleted"
    assert _object_exists(repo, loose_blob), "the loose unreachable blob was deleted"
    assert _loose_count(repo) == 1


def test_touches_no_refs(tmp_path, capsys):
    repo = _repo_with_loose_objects(tmp_path / "repo")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook_log = tmp_path / "reference-transaction.log"
    hook = hooks / "reference-transaction"
    hook.write_text(f'#!/bin/sh\ncat >/dev/null\necho "$1" >> "{hook_log}"\n', encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    _git(repo, "config", "core.hooksPath", str(hooks))

    _git(repo, "update-ref", "refs/heads/probe", "HEAD")
    assert hook_log.exists(), "premise: the reference-transaction hook must fire on a ref update"
    _git(repo, "update-ref", "-d", "refs/heads/probe")
    hook_log.unlink()

    packed_refs = Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / "packed-refs"
    refs_before = _git(repo, "for-each-ref")
    packed_before = packed_refs.read_bytes() if packed_refs.exists() else None

    _run(capsys, _hermetic_argv(tmp_path, repo))

    assert _git(repo, "for-each-ref") == refs_before
    assert (packed_refs.read_bytes() if packed_refs.exists() else None) == packed_before
    assert not hook_log.exists(), hook_log.read_text(encoding="utf-8")


def test_ambient_git_redirection_is_inert(tmp_path, capsys, monkeypatch):
    target = _repo_with_loose_objects(tmp_path / "target")
    decoy = _repo_with_loose_objects(tmp_path / "decoy")
    decoy_loose = _loose_count(decoy)
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(decoy / ".git" / "objects"))

    _rc, records, _err = _run(capsys, _hermetic_argv(tmp_path, target))

    monkeypatch.delenv("GIT_DIR")
    monkeypatch.delenv("GIT_OBJECT_DIRECTORY")
    assert _loose_count(decoy) == decoy_loose
    assert not _commit_graph(decoy).exists()
    assert [record["outcome"] for record in records] == ["maintained"]
    assert _loose_count(target) == 0


def test_runs_both_git_commands_niced_and_idle_io(tmp_path, capsys, monkeypatch):
    real_nice = shutil.which("nice")
    assert real_nice is not None
    calls = tmp_path / "nice-calls.jsonl"
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "nice"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(calls)!r}, 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"os.execv({real_nice!r}, [{real_nice!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    repo = _repo_with_loose_objects(tmp_path / "repo")
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")

    _run(capsys, _hermetic_argv(tmp_path, repo))

    recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(recorded) == 2
    assert all(argv[: len(NICED_GIT)] == NICED_GIT for argv in recorded)
    assert recorded[0][-4:] == ["repack", "-d", "-q", "-n"]
    assert recorded[1][-3:] == ["commit-graph", "write", "--reachable"]


def _unwritable_pack_dir(repo: Path) -> Path:
    pack_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / "objects" / "pack"
    pack_dir.chmod(0o555)
    return pack_dir


def test_repos_run_one_at_a_time_in_order_and_a_failure_does_not_stop_the_run(tmp_path, capsys):
    if os.geteuid() == 0:
        pytest.skip("root ignores the read-only pack dir that makes repack fail")
    repo_a = _repo_with_loose_objects(tmp_path / "a")
    repo_b = _repo_with_loose_objects(tmp_path / "b")
    missing = tmp_path / "missing"
    repo_c = _repo_with_loose_objects(tmp_path / "c")
    pack_dir = _unwritable_pack_dir(repo_b)
    try:
        rc, records, _err = _run(capsys, _hermetic_argv(tmp_path, repo_a, repo_b, missing, repo_c))
    finally:
        pack_dir.chmod(0o755)

    assert [record["root"] for record in records] == [
        str(root.resolve()) for root in (repo_a, repo_b, missing, repo_c)
    ]
    assert [record["outcome"] for record in records] == [
        "maintained", "failed", "not-a-repo", "maintained",
    ]
    assert records[1]["repack_rc"] != 0
    assert records[1]["commit_graph_rc"] is None
    assert rc == 1


def test_never_walks_up_into_an_enclosing_repo(tmp_path, capsys):
    enclosing = _repo_with_loose_objects(tmp_path / "enclosing")
    subdir = enclosing / "plain-subdir"
    subdir.mkdir()
    loose = _loose_count(enclosing)

    _rc, records, _err = _run(capsys, _hermetic_argv(tmp_path, subdir))

    assert [record["outcome"] for record in records] == ["not-a-repo"]
    assert _loose_count(enclosing) == loose
    assert not _commit_graph(enclosing).exists()


@pytest.mark.parametrize("marker", ["gc.pid", "objects/maintenance.lock"])
def test_a_busy_repo_is_left_alone(tmp_path, capsys, marker):
    repo = _repo_with_loose_objects(tmp_path / "repo")
    (Path(_git(repo, "rev-parse", "--absolute-git-dir").strip()) / marker).write_text("", encoding="utf-8")
    loose = _loose_count(repo)

    _rc, records, _err = _run(capsys, _hermetic_argv(tmp_path, repo))

    assert [record["outcome"] for record in records] == ["busy"]
    assert _loose_count(repo) == loose


def test_never_two_runs_at_once(tmp_path, capsys):
    repo = _repo_with_loose_objects(tmp_path / "repo")
    argv = _hermetic_argv(tmp_path, repo)
    lock_path = tmp_path / "maint.lock"
    loose = _loose_count(repo)

    with lock_path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        rc, records, err = _run(capsys, argv)

    assert rc == 0
    assert records == []
    assert _loose_count(repo) == loose
    assert str(lock_path) in err


def test_the_default_roster_is_every_root_an_orchestrator_unit_runs(tmp_path, capsys):
    repo = _repo_with_loose_objects(tmp_path / "repo")
    units = tmp_path / "units"
    units.mkdir()
    (units / "orchestrator-fake.service").write_text(
        f"[Service]\nExecStart=/usr/bin/true --config {repo}/dark-factory-orchestrator.yaml\n",
        encoding="utf-8",
    )

    rc, records, _err = _run(
        capsys, ["--units-dir", str(units), "--lock-file", str(tmp_path / "maint.lock")]
    )

    assert rc == 0
    assert [(record["root"], record["outcome"]) for record in records] == [
        (str(repo.resolve()), "maintained")
    ]


def test_an_empty_roster_is_loud(tmp_path, capsys):
    units = tmp_path / "units"
    units.mkdir()

    rc, records, _err = _run(
        capsys, ["--units-dir", str(units), "--lock-file", str(tmp_path / "maint.lock")]
    )

    assert rc == 2
    assert records == []
