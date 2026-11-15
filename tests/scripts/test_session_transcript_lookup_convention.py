"""Behavioral contract: every doc handing out the session-transcript lookup recipe has one that works.

Task 4637. Confusion-codebook entry ``entry-cand-20260809-16``, two sightings
on 2026-08-16 (census-verifier sessions 8dfbc96a and 13ed9bd7): each ran
``find /`` for a session id, outran the Bash tool's 2-minute default
``timeout``, and only then narrowed its way to a transcript that had been in
the project's agent-transcript archive all along.

WHY A MACHINE CHECK FOR A DOCUMENTATION FIX — the reasoning
``test_tasks_db_schema_convention.py`` (task 5330) records, which this file
follows structurally: prose only reaches agents who read it, and this guard
EXECUTES the command each doc in :data:`RECIPE_DOCS` hands them. It runs it
from a LINKED WORKTREE, because that is where a dispatched agent stands and
where ``data/`` does not exist, against an archive built by the real writer,
so the expected path is derived from
``shared.transcript_archive.durable_archive_path`` rather than remembered.

WHAT THIS FILE DELIBERATELY DOES NOT DO. It asserts nothing about the prose
around the marker: not wording, headings, ordering or the layout it states.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard runs under FULL_SUITE and merge-role ``merge_verify_breadth: full``.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

import pytest
import verify_command_invariants as vci
from shared.project_layout import agent_transcripts_dir
from shared.transcript_archive import archive_task_transcripts, durable_archive_path
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]

RECIPE_DOCS = (REPO_ROOT / "CLAUDE.md",)

MARKER = "session-transcript-lookup"
LABEL = "Session transcript"

_TOOL_RELATIVE_PATH = "scripts/locate_session_transcript.py"
_SESSION_ID_PLACEHOLDER = "<session-id>"

_PYTHON_ARGV0 = ("python", "python3")

_RUN_TIMEOUT_SECS = 60

_TASK_ID = "4637"
_SID = "4396db7a-2b3c-4d5e-8f90-a1b2c3d4e5f6"

_SELF_TEST_DOC_NAME = "self-test-doc.md"

_NO_MARKER_DOC = """\
- **Session transcript**: `python3 scripts/locate_session_transcript.py <session-id>`

Somewhere else entirely, an unmarked `find / -name '*<session-id>*'`.
"""

_DUPLICATE_MARKER_DOC = """\
<!-- session-transcript-lookup:begin -->
- **Session transcript**: `python3 scripts/locate_session_transcript.py <session-id>`
<!-- session-transcript-lookup:end -->

## Some later section

<!-- session-transcript-lookup:begin -->
- **Session transcript**: `ls data/orchestrator/agent-transcripts/*/*/<session-id>.jsonl`
<!-- session-transcript-lookup:end -->
"""


def _marked_command(text, doc_name):
    """The command on the :data:`LABEL` bullet between :data:`MARKER`'s begin/end comments."""
    return vci.marked_span(
        text,
        begin=f"{MARKER}:begin",
        end=f"{MARKER}:end",
        pattern=re.compile(r"- \*\*" + re.escape(LABEL) + r"\*\*: `([^`]+)`"),
        what=f"`- **{LABEL}**: <command>` bullet",
        source=doc_name,
        label=f"the {LABEL!r} command",
        task="4637",
    ).strip()


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
    ],
)
def test_marked_command_fails_loudly_on_a_broken_marker(markdown_text, case):
    """A missing or duplicated marker RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns the execution test green while executing nothing at all.
    """
    with pytest.raises(AssertionError) as excinfo:
        _marked_command(markdown_text, _SELF_TEST_DOC_NAME)

    message = str(excinfo.value)
    assert MARKER in message, case
    assert _SELF_TEST_DOC_NAME in message, case


def _git(*args):
    subprocess.run(
        ["git", *args], check=True, capture_output=True, timeout=_RUN_TIMEOUT_SECS
    )


def _checkout_with_archived_session(tmp_path):
    """A main checkout whose archive holds :data:`_SID`, and a linked worktree of it.

    Returns ``(worktree, expected_path)``; the expected path is what
    ``durable_archive_path`` answers for the session the real writer archived.
    """
    main = tmp_path / "checkout"
    _git("init", "-q", "-b", "main", str(main))
    _git(
        "-C", str(main), "-c", "user.email=t@e", "-c", "user.name=t",
        "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "seed",
    )
    main = main.resolve()
    worktree = main / ".worktrees" / _TASK_ID
    _git("-C", str(main), "worktree", "add", "-q", "-b", f"task/{_TASK_ID}", str(worktree))

    cfg = tmp_path / "cfg"
    transcript = cfg / "projects" / "-tmp-enc" / f"{_SID}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({"sessionId": _SID}) + "\n", encoding="utf-8")
    archive_root = agent_transcripts_dir(main)
    archive_task_transcripts(cfg, _TASK_ID, _SID, archive_root=archive_root)
    expected = durable_archive_path(archive_root, _TASK_ID, _SID)
    assert expected is not None, f"the real writer archived nothing under {archive_root}"
    return worktree, expected


@pytest.mark.parametrize(
    "doc_path",
    RECIPE_DOCS,
    ids=lambda doc_path: doc_path.relative_to(REPO_ROOT).as_posix(),
)
def test_documented_session_transcript_lookup_finds_an_archived_session_from_a_worktree(
    doc_path, tmp_path
):
    """Each doc's recipe must find an archived session from where an agent stands."""
    doc_name = doc_path.relative_to(REPO_ROOT).as_posix()
    command = _marked_command(doc_path.read_text(encoding="utf-8"), doc_name)
    argv = shlex.split(command)
    assert (
        len(argv) == 3
        and argv[0] in _PYTHON_ARGV0
        and argv[1:] == [_TOOL_RELATIVE_PATH, _SESSION_ID_PLACEHOLDER]
    ), (
        f"the command inside the {MARKER!r} marker in {doc_name} is not a bare run "
        f"of the lookup tool (task 4637): {command!r} tokenises to {argv!r}, "
        f"expected exactly [python|python3, {_TOOL_RELATIVE_PATH!r}, "
        f"{_SESSION_ID_PLACEHOLDER!r}]. A hand-written glob or `find` in its place "
        f"is the confusion this closes."
    )

    worktree, expected = _checkout_with_archived_session(tmp_path)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    run_argv = [sys.executable, str(REPO_ROOT / argv[1]), _SID]

    try:
        completed = subprocess.run(
            run_argv,
            cwd=str(worktree),
            env={**os.environ, "HOME": str(fake_home)},
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the {MARKER!r} command documented in {doc_name} did not finish within "
            f"{_RUN_TIMEOUT_SECS}s (task 4637); argv: {run_argv!r}, cwd: {worktree}"
        )

    context = (
        f" doc: {doc_name}\n argv: {run_argv!r}\n cwd: {worktree}\n"
        f" stdout: {completed.stdout.strip()!r}\n"
        f" stderr: {completed.stderr.strip()!r}"
    )
    assert completed.returncode == 0, (
        f"the {MARKER!r} command exited {completed.returncode} (task 4637): agents "
        f"are being handed a recipe that does not find an archived session.\n{context}"
    )
    assert str(expected) in completed.stdout.splitlines(), (
        f"the {MARKER!r} command exited 0 without printing {str(expected)!r}, the "
        f"path durable_archive_path gives for the session the real writer "
        f"archived (task 4637).\n{context}"
    )
