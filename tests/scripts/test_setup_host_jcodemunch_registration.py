"""setup-host.sh installs the jcodemunch launcher and registers it from the shared launch contract.

On every run it installs the prebuilt launcher at the contract's pin, then
registers jcodemunch in the user-level Claude config from the contract,
replacing any existing entry so a host carrying a legacy registration
converges on it. It never registers a command that does not resolve on PATH.
A contract that does not render, a failed install and a failed registration
are each loud but never abort the bootstrap.

The block is sliced out of the shipped setup-host.sh and run with the REAL
REPO_ROOT, so `$REPO_ROOT/shared/src` resolves, against `claude` and `uv`
stubs that record every invocation.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
from setup_host_sections import (
    REPO_ROOT,
    run_section,
    slice_section,
    stub_bin_dir,
    write_stub,
)
from shared.jcodemunch_launch import (
    JCODEMUNCH_COMMAND,
    jcodemunch_install_argv,
    jcodemunch_server_config,
)
from shell_sections import dispatch_stub_body

_START = '_jcodemunch_server_json="$('
_END = "\nfi\n"
_END_AFTER = 'claude mcp add-json --scope user jcodemunch "$_jcodemunch_server_json"'

_CLAUDE_LOG = "claude-calls.log"
_UV_LOG = "uv-calls.log"
_CALL_END = "<<END>>"

_REMOVAL = ["mcp", "remove", "--scope", "user", "jcodemunch"]
_REGISTRATION = ["mcp", "add-json", "--scope", "user", "jcodemunch"]


def _write_recording_stub(tmp_path, name, log_name, body):
    """A stub `name` that logs its argv one element per line, then runs *body*."""
    log = tmp_path / log_name
    write_stub(
        stub_bin_dir(tmp_path),
        name,
        f"printf '%s\\n' \"$@\" >> '{log}'\n"
        f"printf '%s\\n' '{_CALL_END}' >> '{log}'\n" + body,
    )


def _write_claude_stub(tmp_path, *, remove_rc, add_rc):
    _write_recording_stub(
        tmp_path,
        "claude",
        _CLAUDE_LOG,
        dispatch_stub_body(
            (
                ('"mcp remove"*', f"    exit {remove_rc}\n"),
                ('"mcp add-json"*', f"    exit {add_rc}\n"),
            )
        ),
    )


def _write_uv_stub(tmp_path, *, rc, installs_launcher):
    """A `uv` that, when *installs_launcher*, puts the launcher on PATH as a real install would."""
    launcher = stub_bin_dir(tmp_path) / JCODEMUNCH_COMMAND
    install = (
        f"printf '#!/usr/bin/env bash\\nexit 0\\n' > '{launcher}'\nchmod +x '{launcher}'\n"
        if installs_launcher
        else ""
    )
    _write_recording_stub(tmp_path, "uv", _UV_LOG, install + f"exit {rc}\n")


def _calls(tmp_path, log_name) -> list[list[str]]:
    log = tmp_path / log_name
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


def _uv_calls(tmp_path) -> list[list[str]]:
    return [["uv", *call] for call in _calls(tmp_path, _UV_LOG)]


def _path_without(stub_bin, *names):
    """The stub dir, plus every inherited PATH entry that carries none of *names*.

    `run_section` PREPENDS its stub dir to the inherited PATH, so simply
    writing no stub is not enough: on a developer host the section would reach
    the real binary. Dropping only the directories that actually hold one of
    *names* keeps `mkdir`, `python3` and `jq`, which the preamble and this
    slice need, while making `command -v` fail deterministically rather than
    per-host. On a developer host claude, uv and jcodemunch-mcp share
    ~/.local/bin, so hiding one hides all three; that is harmless because
    every test stubs claude and uv.
    """
    kept = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and not any(os.access(os.path.join(entry, name), os.X_OK) for name in names)
    ]
    return os.pathsep.join([str(stub_bin), *kept])


def _run_registration(
    tmp_path, *, uv_rc=0, uv_installs_launcher=False, hidden=(), section_suffix=""
):
    """Run the block with a `uv` stub always in place: no test may run a real `uv tool install`."""
    stub_bin = stub_bin_dir(tmp_path)
    _write_uv_stub(tmp_path, rc=uv_rc, installs_launcher=uv_installs_launcher)
    if JCODEMUNCH_COMMAND not in hidden:
        write_stub(stub_bin, JCODEMUNCH_COMMAND, "exit 0\n")
    return run_section(
        tmp_path,
        slice_section(_START, _END, end_after=_END_AFTER) + section_suffix,
        repo_root=REPO_ROOT,
        unit_dir=tmp_path / "units",
        env_extra={"PATH": _path_without(stub_bin, *hidden)} if hidden else None,
    )


def _lines_starting(stdout, prefix):
    return [line for line in stdout.splitlines() if line.startswith(prefix)]


def _registration_lines(stdout, prefix):
    """Lines reporting the REGISTRATION outcome, as opposed to the launcher install."""
    return [line for line in _lines_starting(stdout, prefix) if "user config" in line]


def test_registration_carries_the_shared_launch_contract(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=1, add_rc=0)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    calls = _calls(tmp_path, _CLAUDE_LOG)
    registrations = [call for call in calls if call[:5] == _REGISTRATION]
    assert len(registrations) == 1, calls
    registered = json.loads(registrations[0][5])
    assert registered == jcodemunch_server_config()
    assert registered["env"]["JCODEMUNCH_GIT_ROOT_IDENTITY"] == "0"
    assert registered["command"] == JCODEMUNCH_COMMAND
    assert _registration_lines(proc.stdout, "OK "), proc.stdout


def test_an_existing_registration_is_replaced_not_skipped(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=0, add_rc=0)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _calls(tmp_path, _CLAUDE_LOG) == [
        _REMOVAL,
        [*_REGISTRATION, json.dumps(jcodemunch_server_config())],
    ]


@pytest.mark.parametrize(
    ("remove_rc", "reports_the_removal"),
    [
        pytest.param(0, True, id="an-existing-entry-was-removed"),
        pytest.param(1, False, id="there-was-no-entry-to-remove"),
    ],
)
def test_a_failed_registration_is_loud_and_says_whether_it_removed_the_old_one(
    tmp_path, remove_rc, reports_the_removal
):
    """Remove-then-add is not atomic: an add that fails after a remove leaves no entry."""
    _write_claude_stub(tmp_path, remove_rc=remove_rc, add_rc=1)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _calls(tmp_path, _CLAUDE_LOG) == [
        _REMOVAL,
        [*_REGISTRATION, json.dumps(jcodemunch_server_config())],
    ]
    failures = _registration_lines(proc.stdout, "FAIL ")
    assert len(failures) == 1, proc.stdout
    assert ("removed" in failures[0]) is reports_the_removal, failures
    assert any("add-json" in line for line in _lines_starting(proc.stdout, "WARN ")), proc.stdout
    assert not _registration_lines(proc.stdout, "OK "), proc.stdout


@pytest.mark.parametrize(
    "failing_render_args",
    [
        pytest.param("-m shared.jcodemunch_launch", id="server-config"),
        pytest.param("-m shared.jcodemunch_launch install-argv", id="install-argv"),
    ],
)
def test_a_contract_that_does_not_render_is_loud_and_touches_nothing(
    tmp_path, failing_render_args
):
    real_python3 = shutil.which("python3")
    assert real_python3, "python3 must be on PATH to render the contract"
    write_stub(
        stub_bin_dir(tmp_path),
        "python3",
        f'if [ "$*" = "{failing_render_args}" ]; then exit 1; fi\n'
        f'exec "{real_python3}" "$@"\n',
    )
    _write_claude_stub(tmp_path, remove_rc=0, add_rc=0)

    proc = _run_registration(tmp_path, section_suffix="printf 'SLICE-COMPLETED\\n'\n")

    assert proc.returncode == 0, proc.stderr
    assert "SLICE-COMPLETED" in proc.stdout, proc.stdout
    assert _uv_calls(tmp_path) == []
    assert _calls(tmp_path, _CLAUDE_LOG) == []
    assert _registration_lines(proc.stdout, "FAIL "), proc.stdout
    assert any(
        "shared.jcodemunch_launch" in line for line in _lines_starting(proc.stdout, "WARN ")
    ), proc.stdout
    assert not _lines_starting(proc.stdout, "OK "), proc.stdout


def test_registration_is_skipped_on_a_host_with_no_claude(tmp_path):
    """The launcher is still provisioned: orchestrator agents and recon stages launch it."""
    proc = _run_registration(
        tmp_path,
        hidden=("claude",),
        section_suffix="printf 'SLICE-COMPLETED\\n'\n",
    )

    assert proc.returncode == 0, proc.stderr
    assert "SLICE-COMPLETED" in proc.stdout, proc.stdout
    assert _calls(tmp_path, _CLAUDE_LOG) == []
    assert not _registration_lines(proc.stdout, "OK "), proc.stdout
    assert not _registration_lines(proc.stdout, "FAIL "), proc.stdout
    assert _uv_calls(tmp_path) == [jcodemunch_install_argv()]


def test_the_launcher_is_installed_at_the_contract_pin(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=1, add_rc=0)

    proc = _run_registration(tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert _uv_calls(tmp_path) == [jcodemunch_install_argv()]


def test_a_fresh_host_gets_the_launcher_before_it_is_registered(tmp_path):
    _write_claude_stub(tmp_path, remove_rc=1, add_rc=0)

    proc = _run_registration(
        tmp_path, hidden=(JCODEMUNCH_COMMAND,), uv_installs_launcher=True
    )

    assert proc.returncode == 0, proc.stderr
    assert _uv_calls(tmp_path) == [jcodemunch_install_argv()]
    assert _calls(tmp_path, _CLAUDE_LOG) == [
        _REMOVAL,
        [*_REGISTRATION, json.dumps(jcodemunch_server_config())],
    ]
    assert _registration_lines(proc.stdout, "OK "), proc.stdout


@pytest.mark.parametrize(
    "uv_rc",
    [
        pytest.param(0, id="installed-to-a-bin-dir-not-on-PATH"),
        pytest.param(1, id="install-failed"),
    ],
)
def test_a_launcher_missing_from_path_is_loud_and_never_registered(tmp_path, uv_rc):
    """uv 0.11.6 exits 0 with only a warning when its bin dir is not on PATH.

    Not even the remove runs: an existing registration is left alone rather
    than swapped for one that cannot launch.
    """
    _write_claude_stub(tmp_path, remove_rc=0, add_rc=0)

    proc = _run_registration(
        tmp_path, uv_rc=uv_rc, hidden=(JCODEMUNCH_COMMAND,), uv_installs_launcher=False
    )

    assert proc.returncode == 0, proc.stderr
    assert _calls(tmp_path, _CLAUDE_LOG) == []
    assert _lines_starting(proc.stdout, "FAIL "), proc.stdout
    assert any("PATH" in line for line in _lines_starting(proc.stdout, "WARN ")), proc.stdout
    assert not _registration_lines(proc.stdout, "OK "), proc.stdout


def test_a_failed_install_is_loud_but_does_not_abort_setup(tmp_path):
    """An already-installed older launcher still resolves, so it is still registered."""
    _write_claude_stub(tmp_path, remove_rc=1, add_rc=0)

    proc = _run_registration(tmp_path, uv_rc=1)

    assert proc.returncode == 0, proc.stderr
    assert _lines_starting(proc.stdout, "FAIL "), proc.stdout
    assert any(
        "uv tool install" in line for line in _lines_starting(proc.stdout, "WARN ")
    ), proc.stdout
    calls = _calls(tmp_path, _CLAUDE_LOG)
    assert calls and calls[-1][:5] == _REGISTRATION, calls
