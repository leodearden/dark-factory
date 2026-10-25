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

This test uses AST parsing over every *.py under orchestrator/tests/
(recursively, so sub-packages are covered too), excluding itself, and fails
immediately on any actual get_event_loop() *call* node.  Comments and string
literals that mention the name are not flagged, avoiding false positives from
docstrings that explain why NOT to use get_event_loop().
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT
from _tree_scan import (
    ORCHESTRATOR_TESTS,
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
)

# The family mark and shared group: _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

_THIS_FILE = Path(__file__).name


def _find_get_event_loop_calls(source: str) -> list[int]:
    """Return line numbers of get_event_loop() call nodes in *source*.

    Uses the AST so comments and string literals are not matched.  Returns an
    empty list if the file cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return _find_get_event_loop_calls_in(tree)


def _find_get_event_loop_calls_in(tree: ast.Module) -> list[int]:
    """:func:`_find_get_event_loop_calls` over an already-parsed module."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'get_event_loop'
        )
    ]


def _get_event_loop_sites(module: ParsedModule) -> tuple[tuple[int, str], ...]:
    """``(lineno, line_text)`` per get_event_loop() call; none if unparseable."""
    try:
        tree = module.tree
    except SyntaxError:
        return ()
    source_lines = module.source.splitlines()
    return tuple(
        (lineno, source_lines[lineno - 1].strip() if lineno <= len(source_lines) else '')
        for lineno in _find_get_event_loop_calls_in(tree)
    )


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=ORCHESTRATOR_TESTS, extract=_get_event_loop_sites)
TREE_ANALYSES = (ANALYSIS,)


def test_no_get_event_loop_in_orchestrator_tests(tree_pass: TreePass) -> None:
    """No orchestrator test file may call asyncio.get_event_loop()."""
    offenders = [
        f'{py_file.relative_to(TESTS_DIR)}:{lineno}: {line_text}'
        for py_file, sites in tree_pass.facts(ANALYSIS).items()
        if py_file.name != _THIS_FILE  # skip the guard itself
        for lineno, line_text in sites
    ]

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
