"""Builders for the manifest-drift audit suites
(scripts/tests/test_audit_manifest_descriptor_drift*.py): synthetic project
roots, sidecar docs and tasks.db rows.

A plain module rather than a conftest fixture, the same convention as
``cli_subprocess_timeout``: conftest appends this directory to ``sys.path``, so
those test files import it by bare name.

The tasks-table schema and the tasks.db builder live in
scripts/tests/conftest.py behind the `make_tasks_db` fixture (task 3336); the
caller passes that fixture in. A synthetic project here needs one thing the
combine-audit fixtures do not: a real `git init` + `git add`, because this
sweep discovers manifests via `git ls-files` (TRACKED files only) rather than
by globbing. No commit is needed — ls-files reads the INDEX.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

GREP_CHECK = {"kind": "grep", "pattern": "def foo", "paths": ["a.py"], "expect": "present"}
SCRIPT_CHECK = {"kind": "script", "script": "scripts/x.sh", "args": ["--v"], "timeout_secs": 30}
MANUAL_CHECK = {"kind": "manual", "reason": "needs a human eye"}
PATH_CHECK = {"kind": "path", "paths": ["scripts/x.sh"], "expect": "present"}


def capability_doc(name: str, check: dict | None) -> dict:
    cap: dict[str, object] = {
        "name": name, "binding": "capability->producer (wired)", "verdict": "PASS"}
    if check is not None:
        cap["delivered_check"] = check
    return cap


def write_manifest(root: Path, relpath: str, doc) -> Path:
    """Write *doc* to ``<root>/<relpath>``; a str is written VERBATIM.

    Verbatim passthrough is what lets a test seed unparseable YAML.
    """
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc), encoding="utf-8")
    return path


def manifest_doc(task_id, label="δ", prd="plans/x-prd.md", checks=(("gate", GREP_CHECK),),
                  external_task_id=None):
    """One-task manifest doc. *external_task_id* binds a FOREIGN producer.

    The two id fields are mutually exclusive in the model, so a caller
    passing *external_task_id* passes ``task_id=None``. The key is emitted
    only when supplied, so every existing caller's doc is byte-identical.
    """
    task: dict[str, object] = {"label": label, "task_id": task_id,
                               "capabilities": [capability_doc(n, c) for n, c in checks]}
    if external_task_id is not None:
        task["external_task_id"] = external_task_id
    return {"prd": prd, "schema_version": 1, "tasks": [task]}


def git_init(root: Path) -> None:
    """``git init`` + ``git add -A`` so `git ls-files` finds the manifests.

    Local user config only — never --global — so the suite cannot mutate the
    developer's git configuration. No commit: ls-files reads the index.
    """
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(root), "config", key, value],
                       check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)


def make_project(tmp_path, make_tasks_db, *, name="proj", tasks=(), manifests=()):
    """Build a synthetic project root: tasks.db, manifest sidecars, a git index."""
    root = tmp_path / name
    (root / ".taskmaster" / "tasks").mkdir(parents=True, exist_ok=True)
    make_tasks_db(list(tasks), directory=root / ".taskmaster" / "tasks")
    for relpath, doc in manifests:
        write_manifest(root, relpath, doc)
    git_init(root)
    return root


def check_entry(name: str, check: dict, **overrides) -> dict:
    """One ``metadata.delivered_checks`` entry: a check dict plus its name."""
    return {"name": name, **check, **overrides}


def task_row(task_id: int, entries) -> dict:
    return {"id": task_id, "status": "done", "metadata": {"delivered_checks": list(entries)}}


def one_project(tmp_path, make_tasks_db, *, sidecar_check, task_entry,
                 cap="gate", task_id=100):
    """The minimal one-manifest/one-task shape most comparison tests need."""
    return make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(task_id, [task_entry])],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(task_id, checks=((cap, sidecar_check),)))],
    )


def finding_triples(audit) -> set[tuple[str, int, str]]:
    return {(d.manifest, d.task_id, d.capability) for d in audit.findings}


def labelled_row(task_id, *, status="pending", prd_path="plans/x-prd.md", label="α", **extra):
    """A task row whose metadata carries prd_path + prd_task_label (plus *extra*)."""
    return {"id": task_id, "status": status,
            "metadata": {"prd_path": prd_path, "prd_task_label": label, **extra}}


def declaring_sidecar(*labels, prd="plans/x-prd.md"):
    """A sidecar declaring exactly *labels*, in that order, every entry unstamped.

    No block carries a capability: the label-binding direction reads labels
    only.
    """
    return {"prd": prd, "schema_version": 1,
            "tasks": [{"label": label, "task_id": None, "capabilities": []}
                      for label in labels]}


_AUDIT_SCRIPT = str(Path(__file__).parent.parent / "audit_manifest_descriptor_drift.py")


def run_cli(*args):
    """Run the audit script by PATH in a child interpreter, capturing its output."""
    return subprocess.run(
        [sys.executable, _AUDIT_SCRIPT, *args], capture_output=True, text=True
    )
