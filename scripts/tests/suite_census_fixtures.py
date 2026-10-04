"""Builders for the suite census tests: real git trees, junit reports and runs.db files.

A plain module rather than a conftest fixture, the same convention as
``cli_subprocess_timeout``: conftest appends this directory to ``sys.path``, so
the census test files import it by bare name.
"""
from __future__ import annotations

import gzip
import os
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from xml.sax.saxutils import quoteattr

_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Suite Census Test",
    "GIT_AUTHOR_EMAIL": "suite-census@example.invalid",
    "GIT_COMMITTER_NAME": "Suite Census Test",
    "GIT_COMMITTER_EMAIL": "suite-census@example.invalid",
}

_CASE_CHILD = {
    "pass": "",
    "failure": '<failure message="assert False">AssertionError</failure>',
    "error": '<error message="fixture blew up">RuntimeError</error>',
    "skipped": '<skipped type="pytest.skip" message="not here" />',
}


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        env={**os.environ, **_GIT_IDENTITY},
    )


def git_tree(root: Path, files: Mapping[str, str | bytes]) -> Path:
    """Write *files* under *root*, commit them all to a fresh repo, return *root*."""
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def junit_xml(
    suite_timestamp: str, cases: Sequence[tuple[str, str, float | None, str]],
) -> str:
    """A pytest junit document; each case is (classname, name, time, kind)."""
    testcases = []
    for classname, name, seconds, kind in cases:
        timing = "" if seconds is None else f" time={quoteattr(str(seconds))}"
        attrs = f"classname={quoteattr(classname)} name={quoteattr(name)}{timing}"
        child = _CASE_CHILD[kind]
        testcases.append(
            f"<testcase {attrs}>{child}</testcase>" if child else f"<testcase {attrs} />"
        )
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{len(cases)}" '
        f"timestamp={quoteattr(suite_timestamp)}>"
        + "".join(testcases)
        + "</testsuite></testsuites>"
    )


def write_gz(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode()))
    return path


def flake_db(path: Path, rows: Sequence[tuple[str, str, str]]) -> Path:
    """A runs.db carrying a production-shaped flake_occurrence table of (observed_at, test_id, verdict)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE flake_occurrence ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " observed_at TEXT NOT NULL, test_id TEXT NOT NULL,"
            " project_id TEXT NOT NULL, verdict TEXT NOT NULL,"
            " call_site TEXT NOT NULL, runner TEXT, merge_sha TEXT,"
            " task_id TEXT, psi_cpu_some10 REAL, detail TEXT DEFAULT '{}')"
        )
        conn.executemany(
            "INSERT INTO flake_occurrence (observed_at, test_id, project_id, verdict, call_site)"
            " VALUES (?, ?, 'fixture', ?, 'merge_gate')",
            rows,
        )
    conn.close()
    return path
