"""setup-host.sh registers jcodemunch in the user-level Claude config from the shared launch contract.

It does so on every run, replacing any existing entry, so a host carrying a
legacy registration converges on the contract. A failed registration is loud
but never aborts the bootstrap.

The block is sliced out of the shipped setup-host.sh and run with the REAL
REPO_ROOT, so `$REPO_ROOT/shared/src` resolves, against a `claude` stub that
records every invocation.
"""

from __future__ import annotations

import json
import os

from setup_host_sections import (
    REPO_ROOT,
    run_section,
    slice_section,
    stub_bin_dir,
    write_stub,
)
from shared.jcodemunch_launch import JCODEMUNCH_COMMAND, jcodemunch_server_config
from shell_sections import dispatch_stub_body

_START = '_jcodemunch_server_json="$('
_END = "\nfi\n"

_CALLS_LOG = "claude-calls.log"
_CALL_END = "<<END>>"

_REMOVAL = ["mcp", "remove", "--scope", "user", "jcodemunch"]
_REGISTRATION = ["mcp", "add-json", "--scope", "user", "jcodemunch"]


def _write_claude_stub(tmp_path, *, remove_rc, add_rc):
    """A `claude` that logs its argv one element per line, then exits as scripted."""
    log = tmp_path / _CALLS_LOG
    write_stub(
        stub_bin_dir(tmp_path),
        "claude",
        f"printf '%s\\n' \"$@\" >> '{log}'\n"
        f"printf '%s\\n' '{_CALL_END}' >> '{log}'\n"
        + dispatch_stub_body(
            (
                ('"mcp remove"*', f"    exit {remove_rc}\n"),
                ('"mcp add-json"*', f"    exit {add_rc}\n"),
            )
        ),
    )


def _claude_calls(tmp_path) -> list[list[str]]:
    log = tmp_path / _CALLS_LOG
    if not log.is_file():
        return []
    calls: list[list[str]] = []
    current: list[str] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line == _CALL_END:
            calls.append(current)
            current = []
        else:
            current.append(line)
    return calls


def _run_registration(tmp_path, *, section_suffix="", env_extra=None):
    return run_section(
        tmp_path,
        slice_section(_START, _END) + section_suffix,
        repo_root=REPO_ROOT,
        unit_dir=tmp_path / "units",
        env_extra=env_extra,
    )


def _lines_starting(stdout, prefix):
    return [line for line in stdout.splitlines() if line.startswith(prefix)]


def _path_without_claude(stub_bin):
    """The stub dir, plus every inherited PATH entry that carries no `claude`.

    `run_section` PREPENDS its stub dir to the inherited PATH, so simply
    writing no `claude` stub is not enough: on a developer host Claude Code is
    installed and the section would reach that real one. Dropping only the
    directories that actually hold a `claude` executable keeps `mkdir` and
    `python3` — which the preamble and this slice both need — while making
    `command -v claude` fail deterministically rather than per-host.
    """
    kept = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not os.access(os.path.join(entry, "claude"), os.X_OK)
    ]
    return os.pathsep.join([str(stub_bin), *kept])


def test_registration_carries_the_shared_launch_contract(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=1, add_rc=0)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    calls = _claude_calls(tmp_path)
    registrations = [call for call in calls if call[:5] == _REGISTRATION]
    assert len(registrations) == 1, calls
    registered = json.loads(registrations[0][5])
    assert registered == jcodemunch_server_config()
    assert registered["env"]["JCODEMUNCH_GIT_ROOT_IDENTITY"] == "0"
    assert registered["command"] == JCODEMUNCH_COMMAND
    assert _lines_starting(proc.stdout, "OK "), proc.stdout


def test_an_existing_registration_is_replaced_not_skipped(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=0, add_rc=0)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _claude_calls(tmp_path) == [
        _REMOVAL,
        [*_REGISTRATION, json.dumps(jcodemunch_server_config())],
    ]


def test_a_failed_registration_is_loud_but_does_not_abort_setup(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=0, add_rc=1)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _lines_starting(proc.stdout, "FAIL "), proc.stdout
    assert any("add-json" in line for line in _lines_starting(proc.stdout, "WARN ")), proc.stdout
    assert not _lines_starting(proc.stdout, "OK "), proc.stdout


def test_block_is_inert_on_a_host_with_no_claude(tmp_path):
    proc = _run_registration(
        tmp_path,
        section_suffix="printf 'SLICE-COMPLETED\\n'\n",
        env_extra={"PATH": _path_without_claude(stub_bin_dir(tmp_path))},
    )

    assert proc.returncode == 0, proc.stderr
    assert "SLICE-COMPLETED" in proc.stdout, proc.stdout
    assert not _lines_starting(proc.stdout, "OK "), proc.stdout
    assert not _lines_starting(proc.stdout, "FAIL "), proc.stdout
    assert _claude_calls(tmp_path) == []
