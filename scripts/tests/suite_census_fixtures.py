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
from datetime import datetime
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


def _set_mtime(path: Path, iso_utc: str) -> None:
    stamp = datetime.fromisoformat(iso_utc).timestamp()
    os.utime(path, (stamp, stamp))


PYTEST_S1 = "2026-09-30T14:15:43.000000+01:00"


def pytest_project(root: Path) -> Path:
    """A dark-factory-shaped project whose verify artefacts cover every pytest source.

    Modules (dirs holding orchestrator.yaml): orchestrator, fused-memory, tests/scripts.
    ``tests/test_a.py`` exists under BOTH orchestrator and fused-memory.
    The root pyproject.toml declares those two as workspace members, so the
    Part 2 census (``workspace_domain``) can enumerate the tree.

    - archived junit, run S1 (orchestrator): TestX::test_p[1]/[2] pass 2.0/3.0,
      test_q@grp failure 0.5, test_e error 0.2, test_s skipped, and
      tests.test_gone::test_z pass 1.0 (no such tracked file);
    - archived junit (tests_scripts): tests.scripts.test_b::test_b1 pass 1.5,
      classname relative to the repo root;
    - live junit: w1 repeats run S1 (mtime 2026-10-02T12:00:00Z); w2 is a new run,
      test_live pass 0.7 (mtime 2026-10-03T12:00:00Z);
    - pytest log (orchestrator): FAILED TestX::test_p[1], ERROR test_r;
    - runs.db flake_occurrence: the ambiguous tests/test_a.py::test_q, '<unknown>'
      and the deleted tests/test_deleted.py::test_x.
    """
    git_tree(root, {
        "pyproject.toml": '[tool.uv.workspace]\nmembers = ["orchestrator", "fused-memory"]\n',
        "orchestrator/orchestrator.yaml": "",
        "orchestrator/tests/test_a.py": "def test_q():\n    pass\n",
        "fused-memory/orchestrator.yaml": "",
        "fused-memory/tests/test_a.py": "def test_q():\n    pass\n",
        "tests/scripts/orchestrator.yaml": "",
        "tests/scripts/test_b.py": "def test_b1():\n    pass\n",
    })
    s1 = junit_xml(PYTEST_S1, [
        ("tests.test_a.TestX", "test_p[1]", 2.0, "pass"),
        ("tests.test_a.TestX", "test_p[2]", 3.0, "pass"),
        ("tests.test_a", "test_q@grp", 0.5, "failure"),
        ("tests.test_a", "test_e", 0.2, "error"),
        ("tests.test_a", "test_s", 0.01, "skipped"),
        ("tests.test_gone", "test_z", 1.0, "pass"),
    ])
    logs = root / "data" / "verify-logs"
    write_gz(logs / "T1" / "attempt-1.orchestrator.junit-20260930T131543_256921Z.xml.gz", s1)
    write_gz(
        logs / "T2" / "attempt-1.tests_scripts.junit-20261001T000000_1Z.xml.gz",
        junit_xml("2026-10-01T01:00:00+01:00", [("tests.scripts.test_b", "test_b1", 1.5, "pass")]),
    )
    live = {
        "w1": (s1, "2026-10-02T12:00:00+00:00"),
        "w2": (
            junit_xml("2026-10-03T13:00:00+01:00", [("tests.test_a", "test_live", 0.7, "pass")]),
            "2026-10-03T12:00:00+00:00",
        ),
    }
    for worktree, (text, mtime) in live.items():
        report = root / ".worktrees" / worktree / ".df-verify-junit" / "report.orchestrator.xml"
        report.parent.mkdir(parents=True)
        report.write_text(text)
        _set_mtime(report, mtime)
    log = logs / "T3" / "attempt-2.orchestrator.test-20260920T101010_5Z.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "============ short test summary info ============\n"
        "FAILED tests/test_a.py::TestX::test_p[1] - AssertionError: boom\n"
        "ERROR tests/test_a.py::test_r\n"
        "1 failed, 1 error in 3.21s\n"
    )
    flake_db(root / "data" / "orchestrator" / "runs.db", [
        ("2026-09-01T00:00:00+00:00", "tests/test_a.py::test_q", "passes_in_isolation"),
        ("2026-09-02T00:00:00+00:00", "<unknown>", "unconfirmable"),
        ("2026-09-03T00:00:00+00:00", "tests/test_deleted.py::test_x", "fails_in_isolation"),
    ])
    return root


def nextest_project(root: Path) -> Path:
    """A reify-shaped project: a cargo workspace, tests/infra scripts and nextest-era artefacts.

    - log 9 (no infix): reify-compiler::harness_types mod_a::case_one PASS 1.348s and
      0.652s (a debug+release pair), a LEAK, a FAIL, a TIMEOUT plus its SLOW and
      TERMINATING progress lines, one unknown WEIRD status, the run_all line
      ``FAILED test_alpha.sh test_gamma.sh`` with its ``=== FAILED:`` banner, and a
      pytest-style ``PASS:`` line;
    - log 10 (remote-laptop segment): one PASS;
    - flaky-ledger.jsonl: two test_beta.sh entries and one malformed line;
    - runs.db flake_occurrence: one '<unknown>' row.
    """
    git_tree(root, {
        "Cargo.toml": '[workspace]\nmembers = ["crates/*"]\n',
        "crates/reify-eval/Cargo.toml": '[package]\nname = "reify-eval"\n',
        "crates/reify-eval/tests/e2e.rs": "#[test]\nfn t_fail() {\n    assert!(true);\n}\n",
        "tests/infra/test_alpha.sh": "#!/bin/sh\n",
        "tests/infra/test_beta.sh": "#!/bin/sh\n",
        "tests/infra/run_all.sh": "#!/bin/sh\n",
    })
    logs = root / "data" / "verify-logs"
    nine = logs / "9" / "attempt-1.test-20260920T101010_5Z.log"
    nine.parent.mkdir(parents=True)
    nine.write_text(
        "        PASS [   1.348s] ( 6259/23688) reify-compiler::harness_types mod_a::case_one\n"
        "         LEAK [   0.100s] ( 1/2) reify-eval mod::leaky\n"
        "        FAIL [   2.000s] ( 2/2) reify-eval::e2e t_fail\n"
        "        SLOW [>120.000s] (───────────) reify-eval::solve x\n"
        "     TIMEOUT [1200.048s] (22899/22899) reify-eval::solve x\n"
        "  TERMINATING [ ...\n"
        "        WEIRD [ 1.0s] a::b c\n"
        "        PASS [   0.652s] ( 6259/23688) reify-compiler::harness_types mod_a::case_one\n"
        "PASS: some text\n"
        "=== FAILED: test_alpha.sh test_gamma.sh ===\n"
        "FAILED test_alpha.sh test_gamma.sh\n"
    )
    ten = logs / "10" / "attempt-1.remote-laptop.test-20261001T000000_1Z.log"
    ten.parent.mkdir(parents=True)
    ten.write_text("        PASS [   0.500s] (1/1) reify-eval mod::other\n")
    (logs / "flaky-ledger.jsonl").write_text(
        '{"ts":"2026-09-05T01:02:03Z","test":"test_beta.sh","role":"merge",'
        '"task":"1","branch":"HEAD","run_id":"run-1"}\n'
        "not json at all\n"
        '{"ts":"2026-09-06T01:02:03Z","test":"test_beta.sh","role":"merge",'
        '"task":"2","branch":"HEAD","run_id":"run-2"}\n'
    )
    flake_db(root / "data" / "orchestrator" / "runs.db", [
        ("2026-09-07T00:00:00+00:00", "<unknown>", "unconfirmable"),
    ])
    return root
