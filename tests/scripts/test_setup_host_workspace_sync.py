"""setup-host.sh section 3 syncs the whole uv workspace in one unscoped sync.

The slice runs with `uv` replaced by a recording stub, so nothing is installed.
"""

import pathlib

from setup_host_sections import (
    REPO_ROOT,
    run_section,
    slice_section,
    stub_bin_dir,
    write_stub,
)

_START = 'info "Syncing Python'
_END = 'info "Installing fused-memory systemd unit"'
_UV_LOG = "uv-calls.log"

_SCOPING_FLAGS = ("--project", "--package")


def _uv_calls(tmp_path: pathlib.Path) -> list[tuple[str, list[str]]]:
    log = tmp_path / _UV_LOG
    if not log.is_file():
        return []
    calls = []
    for line in log.read_text(encoding="utf-8").splitlines():
        cwd, _, argv = line.partition("|")
        calls.append((cwd, argv.split()))
    return calls


def test_setup_host_syncs_every_workspace_member_in_one_unscoped_sync(
    tmp_path: pathlib.Path,
) -> None:
    write_stub(
        stub_bin_dir(tmp_path),
        "uv",
        f"printf '%s|%s\\n' \"$PWD\" \"$*\" >> {tmp_path / _UV_LOG}\nexit 0\n",
    )

    result = run_section(
        tmp_path,
        slice_section(_START, _END),
        repo_root=REPO_ROOT,
        unit_dir=tmp_path / "units",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = _uv_calls(tmp_path)
    why = (
        "a member-scoped `uv sync` is exact and prunes the other members from "
        "the one shared .venv; see scripts/sync-orchestrator-env.sh and "
        "tests/scripts/test_sync_orchestrator_env.py::test_sync_uses_all_packages. "
        f"uv calls: {calls}"
    )
    assert len(calls) == 1, why
    cwd, argv = calls[0]
    assert cwd == str(REPO_ROOT), why
    assert argv[:1] == ["sync"], why
    assert "--all-packages" in argv, why
    assert not [
        tok
        for tok in argv
        if tok in _SCOPING_FLAGS or tok.startswith(tuple(f"{f}=" for f in _SCOPING_FLAGS))
    ], why
