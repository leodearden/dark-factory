#!/usr/bin/env -S python3 -IS
"""Claude Code PreToolUse filter that stands in front of skim's Bash rewrite hook.

`skim rewrite --hook` (rskim 2.3.1 and 2.10.0) re-serialises a command by
joining its whitespace-split tokens with one space. Newlines inside a
`python -c` script or a heredoc are lost, as are runs of spaces inside quotes.
So this guard consults skim only for a command that is already in that
single-spaced form; any other command, and any input it cannot vet, passes
through with no output, and Claude Code runs it as written. For a command it
does hand over, the delegate's stdout, stderr and exit status are forwarded
verbatim.

Nothing here may originate exit status 2: a PreToolUse hook exiting 2 BLOCKS
the tool call, and this hook runs before every Bash call on the host. Hence the
hand-parsed argv (argparse exits 2 on a usage error), and hence settings.json
runs this file BY PATH through its shebang: a missing file then fails with the
shell's non-blocking 127, where `python3 <missing file>` would exit 2.

settings.json is rewired to run this guard by scripts/install_skim_hook_guard.py.
"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

_USAGE = "usage: skim_hook_guard.py <delegate-hook>\n"


def is_single_spaced(command: str) -> bool:
    return command == " ".join(command.split())


def hook_command(delegate: Path) -> str:
    return shlex.join([str(Path(__file__).resolve()), str(delegate)])


def is_hook_command(command: str, delegate: Path) -> bool:
    """True for hook_command(delegate) as composed by ANY checkout's copy of this file."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return (
        len(argv) == 2
        and Path(argv[0]).name == Path(__file__).name
        and argv[1] == str(delegate)
    )


def _command_of(raw: bytes) -> str | None:
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command")
    return command if isinstance(command, str) else None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write(_USAGE)
        return 1
    delegate = argv[0]
    raw = sys.stdin.buffer.read()
    command = _command_of(raw)
    if command is None or not is_single_spaced(command):
        return 0
    try:
        proc = subprocess.run([delegate], input=raw, capture_output=True)
    except OSError as err:
        sys.stderr.write(f"skim_hook_guard: cannot run delegate {delegate}: {err}\n")
        return 1
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
