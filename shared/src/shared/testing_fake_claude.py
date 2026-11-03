"""A fake ``claude`` executable that answers canned result lines, in order.

Tests that need the REAL spawn path (``build_claude_argv`` -> ``_run_subprocess``
-> ``_parse_claude_output``) point the public ``CLAUDE_BINARY`` override at the
executable this module writes:

    fake = install_fake_claude_cli(tmp_path, [stdout_1, stdout_2])
    monkeypatch.setenv('CLAUDE_BINARY', str(fake.executable))
    monkeypatch.delenv('DF_AGENT_CPU_GOVERN', raising=False)

``build_claude_argv`` reads ``CLAUDE_BINARY`` at call time on both the shared
and the orchestrator-sandboxed spawn paths, so nothing private is patched.
Pinning the env var rather than ``PATH`` matters: an ambient ``CLAUDE_BINARY``
would silently bypass a ``PATH`` patch.

Each run of the executable appends its argv to ``argv.jsonl``, writes the next
response verbatim to stdout (an empty string gives empty stdout) and exits 0.
Once the responses are spent it complains on stderr and exits 3. It never
reads stdin.

It lives in the shipped ``shared`` package, beside :mod:`shared.testing_stdin`
and :mod:`shared.testing_streams`, because ``shared.tests`` is not importable
from the orchestrator pytest rootdir and both suites drive it. Pure stdlib,
and registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ['FakeClaudeCli', 'install_fake_claude_cli']

_ARGV_LOG = 'argv.jsonl'
_RESPONSES = 'responses.json'
_COUNTER = 'counter'
_EXHAUSTED_EXIT_CODE = 3

_SCRIPT_TEMPLATE = '''#!{interpreter} -I
import json
import pathlib
import sys

state = pathlib.Path({state_dir!r})
with (state / {argv_log!r}).open('a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
responses = json.loads((state / {responses!r}).read_text())
counter = state / {counter!r}
index = int(counter.read_text()) if counter.exists() else 0
if index >= len(responses):
    sys.stderr.write(
        'fake claude: all %d canned responses were already used; run %d has none\\n'
        % (len(responses), index + 1)
    )
    sys.exit({exhausted})
counter.write_text(str(index + 1))
sys.stdout.write(responses[index])
'''


@dataclass(frozen=True)
class FakeClaudeCli:
    """The installed fake: the executable to pin, and where its runs are recorded."""

    executable: Path
    state_dir: Path

    def recorded_argv(self) -> list[list[str]]:
        """Each run's argv after argv[0], oldest first."""
        log = self.state_dir / _ARGV_LOG
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line]


def install_fake_claude_cli(root: Path, responses: Sequence[str]) -> FakeClaudeCli:
    """Write a fake ``claude`` under *root* that answers *responses* in order."""
    state_dir = root / 'fake-claude-state'
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / _RESPONSES).write_text(json.dumps(list(responses)))
    bin_dir = root / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    executable = bin_dir / 'claude'
    executable.write_text(_SCRIPT_TEMPLATE.format(
        interpreter=sys.executable,
        state_dir=str(state_dir),
        argv_log=_ARGV_LOG,
        responses=_RESPONSES,
        counter=_COUNTER,
        exhausted=_EXHAUSTED_EXIT_CODE,
    ))
    executable.chmod(0o755)
    return FakeClaudeCli(executable=executable, state_dir=state_dir)
