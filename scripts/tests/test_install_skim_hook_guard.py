"""Tests for scripts/install_skim_hook_guard.py: the pure settings transform and its CLI.

The transform must MERGE, never clobber (skills/spawn/hooks/README.md): only the
command of the bare skim hook entry changes, and re-applying it is a no-op. The
CLI is exercised on real tmp files, including once under a bare interpreter the
way scripts/setup-host.sh invokes it.
"""

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import install_skim_hook_guard
import pytest
import skim_hook_guard
from install_skim_hook_guard import Outcome, guard_skim_hook

INSTALLER = Path(__file__).resolve().parents[1] / "install_skim_hook_guard.py"

SKIM_HOOK = Path("/home/u/.claude/hooks/skim-rewrite.sh")


def _host_settings(skim_hook):
    return {
        "env": {"CLAUDE_CODE_EXAMPLE": "1"},
        "permissions": {"allow": ["Bash(git status)"]},
        "statusLine": {"type": "command", "command": "/x/statusline.sh"},
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": skim_hook, "timeout": 5}],
                },
                {
                    "matcher": "EnterWorktree",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "/x/worktree-hookspath-capture.sh",
                            "timeout": 10,
                        }
                    ],
                },
            ],
            "PostToolUse": [
                {
                    "matcher": "ExitWorktree",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "/x/worktree-hookspath-restore.sh",
                            "timeout": 10,
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "/x/stop.sh", "timeout": 10}],
                }
            ],
        },
    }


def _bash_hook(settings):
    return settings["hooks"]["PreToolUse"][0]["hooks"][0]


def _write_settings(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return path


def _backups(target):
    return sorted(target.parent.glob(f"{target.name}.*.bak"))


def _main(settings_path, skim_hook):
    return install_skim_hook_guard.main(
        ["--settings-path", str(settings_path), "--skim-hook", str(skim_hook)]
    )


def test_bare_skim_entry_is_rewired_and_nothing_else_changes():
    settings = _host_settings(str(SKIM_HOOK))
    before = copy.deepcopy(settings)

    result = guard_skim_hook(settings, SKIM_HOOK)

    assert result.outcome is Outcome.GUARDED
    hook = _bash_hook(result.settings)
    assert hook == {
        "type": "command",
        "command": skim_hook_guard.hook_command(SKIM_HOOK),
        "timeout": 5,
    }
    assert result.settings["hooks"]["PreToolUse"][0]["matcher"] == "Bash"
    reverted = copy.deepcopy(result.settings)
    _bash_hook(reverted)["command"] = str(SKIM_HOOK)
    assert reverted == before
    assert settings == before


def test_guarding_is_idempotent():
    first = guard_skim_hook(_host_settings(str(SKIM_HOOK)), SKIM_HOOK)

    second = guard_skim_hook(first.settings, SKIM_HOOK)

    assert second.outcome is Outcome.ALREADY_GUARDED
    assert second.settings == first.settings


@pytest.mark.parametrize(
    "settings",
    [
        {"env": {"A": "1"}},
        {"hooks": {"Stop": _host_settings(str(SKIM_HOOK))["hooks"]["Stop"]}},
        _host_settings("/x/some-other-hook.sh"),
    ],
    ids=["no-hooks", "no-pre-tool-use", "bash-entry-runs-another-command"],
)
def test_settings_without_the_skim_hook_are_not_wired(settings):
    before = copy.deepcopy(settings)

    result = guard_skim_hook(settings, SKIM_HOOK)

    assert result.outcome is Outcome.NOT_WIRED
    assert result.settings == before


def test_cli_guards_a_bare_entry_with_a_backup_and_the_same_mode(tmp_path):
    skim_hook = tmp_path / "hooks" / "skim-rewrite.sh"
    target = _write_settings(tmp_path / "settings.json", _host_settings(str(skim_hook)))
    target.chmod(0o640)
    original = target.read_bytes()

    assert _main(target, skim_hook) == 0

    assert _bash_hook(json.loads(target.read_text()))["command"] == (
        skim_hook_guard.hook_command(skim_hook)
    )
    backups = _backups(target)
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_cli_rerun_writes_nothing(tmp_path):
    skim_hook = tmp_path / "hooks" / "skim-rewrite.sh"
    target = _write_settings(tmp_path / "settings.json", _host_settings(str(skim_hook)))
    assert _main(target, skim_hook) == 0
    guarded = target.read_bytes()
    inode = target.stat().st_ino

    assert _main(target, skim_hook) == 0

    assert target.read_bytes() == guarded
    assert target.stat().st_ino == inode
    assert len(_backups(target)) == 1


def test_cli_edits_a_symlinked_settings_file_in_place(tmp_path):
    skim_hook = tmp_path / "hooks" / "skim-rewrite.sh"
    real = _write_settings(tmp_path / "real" / "settings.json", _host_settings(str(skim_hook)))
    link = tmp_path / "claude" / "settings.json"
    link.parent.mkdir()
    link.symlink_to(real)

    assert _main(link, skim_hook) == 0

    assert link.is_symlink()
    assert os.readlink(link) == str(real)
    assert _bash_hook(json.loads(real.read_text()))["command"] == (
        skim_hook_guard.hook_command(skim_hook)
    )


def test_cli_fails_when_the_skim_hook_is_not_wired(tmp_path, capsys):
    skim_hook = tmp_path / "hooks" / "skim-rewrite.sh"
    target = _write_settings(tmp_path / "settings.json", _host_settings("/x/other.sh"))
    original = target.read_bytes()

    assert _main(target, skim_hook) == 1

    assert target.read_bytes() == original
    assert _backups(target) == []
    assert str(skim_hook) in capsys.readouterr().err


def test_cli_fails_when_the_settings_file_is_missing(tmp_path, capsys):
    skim_hook = tmp_path / "hooks" / "skim-rewrite.sh"
    target = tmp_path / "settings.json"

    assert _main(target, skim_hook) == 1

    assert not target.exists()
    assert str(skim_hook) in capsys.readouterr().err


def test_cli_runs_under_a_bare_interpreter(tmp_path):
    skim_hook = tmp_path / "hooks" / "skim-rewrite.sh"
    target = _write_settings(tmp_path / "settings.json", _host_settings(str(skim_hook)))

    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            str(INSTALLER),
            "--settings-path",
            str(target),
            "--skim-hook",
            str(skim_hook),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert _bash_hook(json.loads(target.read_text()))["command"] == (
        skim_hook_guard.hook_command(skim_hook)
    )
