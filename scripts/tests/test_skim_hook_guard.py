"""Behavioural tests for scripts/skim_hook_guard.py, driven through its executable.

Every test pipes Claude Code PreToolUse hook JSON through the guard the way the
harness runs it, with a fake delegate standing where skim's generated hook
script would. The FLATTENING delegate models skim's hook re-serialisation —
whitespace-split tokens re-joined with one space — so the guard's invariant,
"the guarded pipeline never changes a command's whitespace", is checked without
skim installed. One test repeats the check against the real skim when present.
"""

import json
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import skim_hook_guard

GUARD = Path(__file__).resolve().parents[1] / "skim_hook_guard.py"

REPRODUCTION = 'git log --oneline -1 && python3 -c "\nimport json\nfor x in [1,2]:\n    print(x)\n"'

MULTI_LINE = [
    REPRODUCTION,
    'git status && git commit -m "$(cat <<\'EOF\'\nsubject\n\nbody\nEOF\n)"',
    "git status\r\necho hi",
    "git diff \\\n  --stat",
]

WHITESPACE_SENSITIVE_SINGLE_LINE = [
    'git log --oneline -1 && echo "x   y"',
    "git log --grep='a  b'",
    'git log --format="%h\t%s" -1',
    " git status",
    "git status ",
]

SINGLE_SPACED = [
    "git status",
    'git log --oneline -1 && echo "x y"',
    "cat foo.py",
]

CANNED_STDOUT = b'{"hookSpecificOutput": {"hookEventName": "PreToolUse"}}\n'
CANNED_STDERR = "recording delegate ran"


def _hook_input(command):
    return json.dumps(
        {
            "session_id": "t",
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": command},
        }
    ).encode()


def _write_executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _recording_delegate(tmp_path, *, path=None, exit_status=0, stderr_line=CANNED_STDERR):
    body = textwrap.dedent(
        f"""\
        import sys
        data = sys.stdin.buffer.read()
        with open({str(_record_path(tmp_path))!r}, "ab") as fh:
            fh.write(data)
        sys.stdout.buffer.write({CANNED_STDOUT!r})
        sys.stderr.write({stderr_line + chr(10)!r})
        sys.exit({exit_status})
        """
    )
    return _write_executable(
        path or tmp_path / "recording-delegate", f"#!{sys.executable}\n{body}"
    )


def _flattening_delegate(tmp_path):
    body = textwrap.dedent(
        """\
        import json
        import sys
        command = json.loads(sys.stdin.buffer.read())["tool_input"]["command"]
        rewritten = "skim " + " ".join(command.split())
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": {"command": rewritten},
        }}))
        """
    )
    return _write_executable(tmp_path / "flattening-delegate", f"#!{sys.executable}\n{body}")


def _record_path(tmp_path):
    return tmp_path / "delegate-stdin"


def _run_guard(args, stdin):
    return subprocess.run(
        [str(GUARD), *[str(a) for a in args]], input=stdin, capture_output=True
    )


@pytest.mark.parametrize("command", MULTI_LINE + WHITESPACE_SENSITIVE_SINGLE_LINE)
def test_whitespace_sensitive_commands_never_reach_the_delegate(tmp_path, command):
    delegate = _recording_delegate(tmp_path)

    proc = _run_guard([delegate], _hook_input(command))

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b""
    assert not _record_path(tmp_path).exists()


@pytest.mark.parametrize("command", SINGLE_SPACED)
def test_single_spaced_commands_are_handed_to_the_delegate_verbatim(tmp_path, command):
    delegate = _recording_delegate(tmp_path)
    stdin = _hook_input(command)

    proc = _run_guard([delegate], stdin)

    assert proc.returncode == 0, proc.stderr
    assert _record_path(tmp_path).read_bytes() == stdin
    assert proc.stdout == CANNED_STDOUT
    assert CANNED_STDERR in proc.stderr.decode()


def test_delegate_exit_status_is_forwarded_verbatim(tmp_path):
    delegate = _recording_delegate(
        tmp_path, exit_status=2, stderr_line="blocked by delegate"
    )

    proc = _run_guard([delegate], _hook_input("git status"))

    assert proc.returncode == 2
    assert "blocked by delegate" in proc.stderr.decode()


def test_guarded_pipeline_never_changes_a_commands_whitespace(tmp_path):
    delegate = _flattening_delegate(tmp_path)
    corpus = MULTI_LINE + WHITESPACE_SENSITIVE_SINGLE_LINE + SINGLE_SPACED

    outputs = {}
    for command in corpus:
        proc = _run_guard([delegate], _hook_input(command))
        assert proc.returncode == 0, proc.stderr
        outputs[command] = proc.stdout

    for command, stdout in outputs.items():
        if stdout == b"":
            continue
        rewritten = json.loads(stdout)["hookSpecificOutput"]["updatedInput"]["command"]
        assert rewritten.removeprefix("skim ") == command
    assert all(outputs[command] != b"" for command in SINGLE_SPACED)


@pytest.mark.parametrize(
    "stdin",
    [
        b"not json",
        json.dumps({"tool_name": "Bash"}).encode(),
        json.dumps({"tool_input": {"command": ["git", "status"]}}).encode(),
    ],
    ids=["invalid-json", "no-tool-input", "non-string-command"],
)
def test_unvetted_input_passes_through_without_the_delegate(tmp_path, stdin):
    delegate = _recording_delegate(tmp_path)

    proc = _run_guard([delegate], stdin)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b""
    assert not _record_path(tmp_path).exists()


def test_unrunnable_delegate_is_loud_but_never_blocking(tmp_path):
    missing = tmp_path / "missing"

    proc = _run_guard([missing], _hook_input("git status"))

    assert proc.returncode == 1
    assert proc.stdout == b""
    assert str(missing) in proc.stderr.decode()


@pytest.mark.parametrize("argc", [0, 2])
def test_bad_invocation_never_blocks(tmp_path, argc):
    delegate = _recording_delegate(tmp_path)

    proc = _run_guard([delegate] * argc, _hook_input("git status"))

    assert proc.returncode == 1
    assert proc.stdout == b""


def test_hook_command_composes_a_runnable_settings_command(tmp_path):
    delegate = _recording_delegate(tmp_path, path=tmp_path / "dir with space" / "delegate")
    stdin = _hook_input("git status")

    command = skim_hook_guard.hook_command(delegate)
    proc = subprocess.run(["sh", "-c", command], input=stdin, capture_output=True)

    assert proc.returncode == 0, proc.stderr
    assert _record_path(tmp_path).read_bytes() == stdin
    assert proc.stdout == CANNED_STDOUT
    assert shlex.split(command) == [str(GUARD), str(delegate)]


@pytest.mark.skipif(shutil.which("skim") is None, reason="skim is not installed")
def test_real_skim_behind_the_guard(tmp_path):
    delegate = _write_executable(
        tmp_path / "skim-delegate", "#!/bin/sh\nexec skim rewrite --hook\n"
    )

    assert _run_guard([delegate], _hook_input(REPRODUCTION)).stdout == b""

    stdin = _hook_input("git status")
    direct = subprocess.run([str(delegate)], input=stdin, capture_output=True)
    assert _run_guard([delegate], stdin).stdout == direct.stdout
