"""The three bash call sites of scripts/recovery_archive.py (task 7153).

Each test slices one new section out of export-data.sh, import-data.sh or
verify-migration.sh and runs it against a stand-in helper that parses bash's
argv with the REAL parse_args, so an argument renamed on one side fails here.
The logging shims and verify's check_* helpers are sliced from the shipped
scripts, so every verdict below is the script's own.
"""

from __future__ import annotations

import pathlib

from shell_sections import REPO_ROOT, run_with_preamble, slice_section, stub_bin_dir, write_stub

EXPORT_DATA_PATH = REPO_ROOT / "scripts" / "export-data.sh"
IMPORT_DATA_PATH = REPO_ROOT / "scripts" / "import-data.sh"
VERIFY_MIGRATION_PATH = REPO_ROOT / "scripts" / "verify-migration.sh"

MARKER = "RECOVERY-ARCHIVE"
ARCHIVE_FAILED = "Recovery archives: a pinned dump is missing or altered"


def _stand_in(tmp_path: pathlib.Path, rc: int) -> None:
    """Install a stand-in scripts/recovery_archive.py under the slice's $REPO_ROOT.

    The real scripts dir goes in at sys.path[0], ahead of the stand-in's own
    directory, so `import recovery_archive` loads the real module. On a
    successful copy it creates the dest, as the real helper does.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "recovery_archive.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r})\n"
        "import recovery_archive\n"
        "args = recovery_archive.parse_args(sys.argv[1:])\n"
        f"if args.command == 'copy' and {rc} == 0:\n"
        "    args.dest.mkdir(parents=True, exist_ok=True)\n"
        "fields = ' '.join(f'{name}={getattr(args, name, None)}' "
        "for name in ('command', 'source', 'dest', 'root'))\n"
        f"print(f'{MARKER} {{fields}}')\n"
        f"sys.exit({rc})\n",
        encoding="utf-8",
    )


def _systemctl_log(tmp_path: pathlib.Path) -> pathlib.Path:
    """Stub `systemctl` to record its argv and succeed; return the record's path."""
    log = tmp_path / "systemctl.log"
    log.touch()
    write_stub(stub_bin_dir(tmp_path), "systemctl", f'echo "$*" >> "{log}"\nexit 0\n')
    return log


def _strict_preamble(tmp_path: pathlib.Path, script_path: pathlib.Path) -> str:
    return (
        "set -euo pipefail\n"
        + slice_section(script_path, "info()  {", "fail()  {")
        + f'REPO_ROOT="{tmp_path}"\n'
        + f'EXPORT_DIR="{tmp_path / "export"}"\n'
        + f'COMPOSE_FILE="{tmp_path / "docker-compose.yml"}"\n'
    )


# --- export-data.sh section 1b ----------------------------------------------

def _run_export(tmp_path: pathlib.Path, helper_rc: int):
    _stand_in(tmp_path, helper_rc)
    systemctl_log = _systemctl_log(tmp_path)
    section = slice_section(
        EXPORT_DATA_PATH,
        'info "Exporting recovery archives"',
        "\nfi\n",
        end_after="systemctl --user stop fused-memory",
    )
    result = run_with_preamble(tmp_path, _strict_preamble(tmp_path, EXPORT_DATA_PATH), section)
    return result, systemctl_log.read_text()


def test_export_copies_the_archives_into_the_export_then_stops_fused_memory(tmp_path):
    result, systemctl_calls = _run_export(tmp_path, helper_rc=0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"{MARKER} command=copy source={tmp_path} dest={tmp_path / 'export' / 'recovery-archive'}"
        in result.stdout
    )
    assert "stop fused-memory" in systemctl_calls


def test_a_failed_archive_export_fails_before_fused_memory_is_stopped(tmp_path):
    result, systemctl_calls = _run_export(tmp_path, helper_rc=1)

    assert result.returncode == 1
    assert ARCHIVE_FAILED in result.stdout, result.stdout + result.stderr
    assert "stop" not in systemctl_calls


# --- import-data.sh section 2b ----------------------------------------------

IMPORT_ARCHIVE_FAILED = "Recovery archives: the export's recovery-archive/ is missing"
RESTORING_FALKORDB = "Restoring FalkorDB data"


def _run_import(tmp_path: pathlib.Path, helper_rc: int):
    _stand_in(tmp_path, helper_rc)
    section = slice_section(
        IMPORT_DATA_PATH, 'info "Restoring recovery archives"', f'info "{RESTORING_FALKORDB}"'
    )
    return run_with_preamble(tmp_path, _strict_preamble(tmp_path, IMPORT_DATA_PATH), section)


def test_import_restores_the_archives_to_their_repo_paths_then_restores_live_data(tmp_path):
    result = _run_import(tmp_path, helper_rc=0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"{MARKER} command=copy source={tmp_path / 'export' / 'recovery-archive'} dest={tmp_path}"
        in result.stdout
    )
    assert RESTORING_FALKORDB in result.stdout


def test_a_failed_archive_restore_stops_the_import_before_any_live_data(tmp_path):
    result = _run_import(tmp_path, helper_rc=1)

    assert result.returncode == 1
    assert IMPORT_ARCHIVE_FAILED in result.stdout, result.stdout + result.stderr
    assert RESTORING_FALKORDB not in result.stdout


# --- verify-migration.sh section 4b -----------------------------------------
# This script runs `set -uo pipefail` WITHOUT -e and its fail only prints, so
# it gets its own preamble, with the shipped counters and check_* helpers.

def _run_verify(tmp_path: pathlib.Path, helper_rc: int):
    _stand_in(tmp_path, helper_rc)
    preamble = (
        "set -uo pipefail\n"
        + slice_section(VERIFY_MIGRATION_PATH, "info()  {", "fail()  {")
        + slice_section(VERIFY_MIGRATION_PATH, "CHECKS_PASSED=0", "check_warn() {")
        + f'REPO_ROOT="{tmp_path}"\n'
    )
    section = slice_section(VERIFY_MIGRATION_PATH, 'info "Recovery archive checks"', "\nfi\n")
    return run_with_preamble(
        tmp_path, preamble, section + 'echo "passed=$CHECKS_PASSED failed=$CHECKS_FAILED"\n'
    )


def test_verify_passes_when_every_pinned_dump_checks_out(tmp_path):
    result = _run_verify(tmp_path, helper_rc=0)

    assert f"{MARKER} command=check source=None dest=None root={tmp_path}" in result.stdout
    assert "passed=1 failed=0" in result.stdout, result.stdout + result.stderr


def test_a_missing_or_altered_dump_is_a_critical_verify_failure(tmp_path):
    result = _run_verify(tmp_path, helper_rc=1)

    assert ARCHIVE_FAILED in result.stdout
    assert "passed=0 failed=1" in result.stdout, result.stdout + result.stderr
