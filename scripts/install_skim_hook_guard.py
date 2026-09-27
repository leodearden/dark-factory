"""Put scripts/skim_hook_guard.py in front of skim's Claude Code Bash hook in settings.json.

`skim init` wires a PreToolUse entry whose command is its generated
~/.claude/hooks/skim-rewrite.sh. This rewires that one command to
`skim_hook_guard.hook_command(<that script>)`, touching nothing else. Re-running
`skim init` REPLACES a guarded entry with its bare hook, so this installer is
idempotent and scripts/setup-host.sh re-applies it on every run.

Run it from the PRIMARY checkout: hook_command bakes this checkout's absolute
path into settings.json, and a worktree path dangles once the worktree is
reaped. Agents cannot run it against the real file, because their sandbox denies
writes to ~/.claude/settings.json.

Exit 0 means only that the guard is in place.
"""

import argparse
import copy
import enum
import json
import stat
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import skim_hook_guard

_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.safe_io import atomic_write_text  # noqa: E402


class Outcome(enum.Enum):
    GUARDED = "guarded"
    ALREADY_GUARDED = "already-guarded"
    NOT_WIRED = "not-wired"


@dataclass(frozen=True)
class Rewire:
    settings: dict[str, Any]
    outcome: Outcome


@dataclass(frozen=True)
class Installed:
    outcome: Outcome
    backup: Path | None


def _pre_tool_use_hooks(settings: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for entry in settings.get("hooks", {}).get("PreToolUse", []):
        yield from entry.get("hooks", [])


def guard_skim_hook(settings: Mapping[str, Any], skim_hook: Path) -> Rewire:
    guarded = copy.deepcopy(dict(settings))
    bare = str(skim_hook)
    wrapped = skim_hook_guard.hook_command(skim_hook)
    hooks = list(_pre_tool_use_hooks(guarded))
    bare_hooks = [hook for hook in hooks if hook.get("command") == bare]
    for hook in bare_hooks:
        hook["command"] = wrapped
    if bare_hooks:
        outcome = Outcome.GUARDED
    elif any(hook.get("command") == wrapped for hook in hooks):
        outcome = Outcome.ALREADY_GUARDED
    else:
        outcome = Outcome.NOT_WIRED
    return Rewire(guarded, outcome)


def install(settings_path: Path, skim_hook: Path) -> Installed:
    target = settings_path.resolve()
    if not target.is_file():
        return Installed(Outcome.NOT_WIRED, None)
    text = target.read_text(encoding="utf-8")
    rewire = guard_skim_hook(json.loads(text), skim_hook)
    if rewire.outcome is not Outcome.GUARDED:
        return Installed(rewire.outcome, None)
    backup = target.with_name(f"{target.name}.{datetime.now(UTC):%Y%m%dT%H%M%SZ}.bak")
    backup.write_text(text, encoding="utf-8")
    atomic_write_text(
        target,
        json.dumps(rewire.settings, indent=2, ensure_ascii=False) + "\n",
        mode=stat.S_IMODE(target.stat().st_mode),
    )
    return Installed(Outcome.GUARDED, backup)


def main(argv: list[str] | None = None) -> int:
    claude_dir = Path.home() / ".claude"
    parser = argparse.ArgumentParser(
        description="Guard skim's Claude Code Bash hook in settings.json."
    )
    parser.add_argument("--settings-path", type=Path, default=claude_dir / "settings.json")
    parser.add_argument(
        "--skim-hook", type=Path, default=claude_dir / "hooks" / "skim-rewrite.sh"
    )
    args = parser.parse_args(argv)

    installed = install(args.settings_path, args.skim_hook)
    if installed.outcome is Outcome.GUARDED:
        print(f"{args.settings_path}: skim hook guarded (backup: {installed.backup})")
        return 0
    if installed.outcome is Outcome.ALREADY_GUARDED:
        print(f"{args.settings_path}: skim hook already guarded")
        return 0
    sys.stderr.write(
        f"no PreToolUse hook runs {args.skim_hook} in {args.settings_path}; "
        "run `skim init` first\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
