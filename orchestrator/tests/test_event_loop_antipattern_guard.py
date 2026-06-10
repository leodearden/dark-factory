"""Guard test: no asyncio.get_event_loop() antipattern in orchestrator/tests/*.py.

asyncio.get_event_loop() is a deprecated antipattern that causes runtime
failures in the test suite:

* In SYNC test bodies, asyncio.run() calls set_event_loop(None) in its
  finally block. Any subsequent get_event_loop() call finds the thread loop
  explicitly None and raises RuntimeError: There is no current event loop
  (task 1711, test_merge_queue_main_health.py was the first victim).

* In ASYNC test bodies the function currently returns the running loop, but
  this is deprecated and may change — the correct idiom is get_running_loop().

Correct replacements by context:

  | Call-site context                        | Correct idiom                               |
  |------------------------------------------|---------------------------------------------|
  | async def test body / async helper       | asyncio.get_running_loop().create_future()  |
  | sync driver / run_until_complete shape   | asyncio.run(coro)                           |
  | sync test building a placeholder future  | make_placeholder_future() (_orch_helpers)   |

This test scans every *.py under orchestrator/tests/ (excluding itself) and
fails immediately on any remaining get_event_loop() call, preventing the
landmine from being reintroduced.
"""
from __future__ import annotations

import re
from pathlib import Path

_PATTERN = re.compile(r'get_event_loop\s*\(\s*\)')
_THIS_FILE = Path(__file__).name


def test_no_get_event_loop_in_orchestrator_tests() -> None:
    """No orchestrator test file may call asyncio.get_event_loop()."""
    tests_dir = Path(__file__).parent
    offenders: list[str] = []

    for py_file in sorted(tests_dir.glob('*.py')):
        if py_file.name == _THIS_FILE:
            continue  # skip the guard itself
        source = py_file.read_text(encoding='utf-8')
        for lineno, line in enumerate(source.splitlines(), start=1):
            if _PATTERN.search(line):
                offenders.append(f'{py_file.name}:{lineno}: {line.strip()}')

    if offenders:
        offender_list = '\n  '.join(offenders)
        raise AssertionError(
            'asyncio.get_event_loop() antipattern found in orchestrator tests.\n'
            'Replace with the correct idiom for the call-site context:\n'
            '  * async def body / async helper → asyncio.get_running_loop().create_future()\n'
            '  * sync driver (run_until_complete shape) → asyncio.run(coro)\n'
            '  * sync test placeholder future → make_placeholder_future()  (from _orch_helpers)\n'
            f'\nOffending sites:\n  {offender_list}'
        )
