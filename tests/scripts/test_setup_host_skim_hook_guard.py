"""setup-host.sh section 7 re-applies the skim hook guard on every run.

The block is sliced out of the shipped setup-host.sh and executed with HOME in
tmp_path, driving the REAL installer from this checkout. A missing guard is
reported loudly but never aborts the host bootstrap.
"""

import json

import skim_hook_guard
from setup_host_sections import REPO_ROOT, run_section, slice_section

_START = '_skim_guard_installer="$REPO_ROOT/scripts/install_skim_hook_guard.py"'
_END = "\n  fi\n"


def _skim_hook(home):
    return home / ".claude" / "hooks" / "skim-rewrite.sh"


def _settings_path(home):
    return home / ".claude" / "settings.json"


def _write_settings(home, bash_command):
    path = _settings_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": bash_command, "timeout": 5}],
                }
            ]
        }
    }
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def _bash_command(home):
    settings = json.loads(_settings_path(home).read_text(encoding="utf-8"))
    return settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


def _run_guard_block(tmp_path, home):
    return run_section(
        tmp_path,
        slice_section(_START, _END),
        repo_root=REPO_ROOT,
        unit_dir=tmp_path / "units",
        env_extra={"HOME": str(home)},
    )


def _lines_starting(stdout, prefix):
    return [line for line in stdout.splitlines() if line.startswith(prefix)]


def test_bare_skim_hook_is_guarded(tmp_path):
    home = tmp_path / "home"
    _write_settings(home, str(_skim_hook(home)))

    proc = _run_guard_block(tmp_path, home)

    assert proc.returncode == 0, proc.stderr
    assert _bash_command(home) == skim_hook_guard.hook_command(_skim_hook(home))
    assert _lines_starting(proc.stdout, "OK ")


def test_unwired_settings_fail_loudly_without_aborting_setup(tmp_path):
    home = tmp_path / "home"
    _write_settings(home, "/x/some-other-hook.sh")

    proc = _run_guard_block(tmp_path, home)

    assert proc.returncode == 0, proc.stderr
    assert _lines_starting(proc.stdout, "FAIL ")
    assert any(
        "install_skim_hook_guard.py" in line
        for line in _lines_starting(proc.stdout, "WARN ")
    )


def test_rerun_on_a_guarded_file_is_a_no_op(tmp_path):
    home = tmp_path / "home"
    _write_settings(home, str(_skim_hook(home)))
    assert _run_guard_block(tmp_path, home).returncode == 0
    guarded = _settings_path(home).read_bytes()

    proc = _run_guard_block(tmp_path, home)

    assert proc.returncode == 0, proc.stderr
    assert _lines_starting(proc.stdout, "OK ")
    assert _settings_path(home).read_bytes() == guarded
