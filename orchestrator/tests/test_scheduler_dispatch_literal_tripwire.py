"""B3 dispatch-literal tripwire: every ``dispatch_tool('<literal>', ...)`` in
scheduler.py must have a matching ``_StubMcpSession`` branch.

The eval ``Scheduler`` dispatches MCP tool calls through an in-process
``_StubMcpSession`` (evals/runner.py) instead of a real fused-memory HTTP
connection.  Any tool the real Scheduler dispatches by string literal that the
stub does not handle raises ``NotImplementedError`` at *eval runtime* — swallowed
and logged as heartbeat WARNING spam (the D2/B9 failure mode this task fixes).

This tripwire moves that failure to *CI*: it statically scans scheduler.py for
every ``dispatch_tool`` call site's string-literal first argument, then probes
the stub behaviourally to assert each one has a branch.  When a future feature
adds a new ``dispatch_tool('some_new_tool', ...)`` call without a stub branch,
this test fails and names the exact missing literal — exactly the silent drift
that let ``get_external_statuses`` land without a stub branch (it was dispatched
by the cross-project-deps feature, added AFTER the eval-revival PRD was written).

Design notes (see task 2470 plan design decisions):
- **AST, not regex/git-grep.** 6 of 7 ``dispatch_tool`` call sites put the
  string literal on a line *separate* from ``dispatch_tool(`` (multi-line
  calls), so a single-line regex would miss them.  An ``ast.parse`` robustly
  finds every ``ast.Call`` to ``dispatch_tool`` with a string-literal first arg.
- **Locate via ``__file__``, not a git checkout.** Unlike the git-grep guard
  convention (which ``pytest.skip``s outside a checkout), locating scheduler.py
  via ``Path(orchestrator.scheduler.__file__)`` is CWD-independent and needs no
  git, so this guard always runs.
- **Behavioural probe, not structural.** ``call_tool(name, {})`` must not raise
  ``NotImplementedError``; *any* other outcome (including a ``KeyError`` from the
  empty args dict) proves the branch exists.  This decouples the guard from the
  stub's internal ``if name == ...`` dispatch shape, so refactoring the stub does
  not break the guard.

``add_dependency`` is intentionally NOT asserted here: it is stubbed defensively
but is not dispatched anywhere in scheduler.py, so it is not among the enumerated
literals.  Its stub branch is covered by a direct unit test in
``test_eval_stub_mcp_session.py`` instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import orchestrator.scheduler
from orchestrator.evals.runner import _StubMcpSession


def _dispatch_tool_literals() -> set[str]:
    """Return the set of string-literal first args of every ``dispatch_tool``
    call in scheduler.py, found by AST scan.

    Matches both ``self.dispatch_tool('x', ...)`` (an ``ast.Attribute`` func
    whose ``attr == 'dispatch_tool'``) and a bare ``dispatch_tool('x', ...)``
    (an ``ast.Name`` func whose ``id == 'dispatch_tool'``).  Only call sites
    whose first positional argument is a string ``ast.Constant`` are collected;
    a dynamic (variable) first arg cannot be checked against the stub and is
    outside this tripwire's scope.  The ``async def dispatch_tool`` *definitions*
    are ``AsyncFunctionDef`` nodes, not ``ast.Call``, so they are never matched.
    """
    source_path = Path(orchestrator.scheduler.__file__)
    tree = ast.parse(source_path.read_text())
    literals: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_dispatch = (
            (isinstance(func, ast.Attribute) and func.attr == 'dispatch_tool')
            or (isinstance(func, ast.Name) and func.id == 'dispatch_tool')
        )
        if not is_dispatch or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            literals.add(first.value)
    return literals


def test_scanner_finds_known_dispatched_literals() -> None:
    """Self-test for the AST scanner: it must find known dispatched literals.

    Guards against a silently-broken scanner (e.g. an ``ast`` API change or a
    matcher typo) that returns an empty/degraded set — which would make the
    main tripwire below vacuously green.  ``set_task_claimant`` and
    ``get_external_statuses`` are the two literals this task added branches for.
    """
    literals = _dispatch_tool_literals()
    assert 'set_task_claimant' in literals
    assert 'get_external_statuses' in literals


@pytest.mark.asyncio
async def test_every_dispatched_literal_has_a_stub_branch() -> None:
    """Every ``dispatch_tool`` literal in scheduler.py has a ``_StubMcpSession``
    branch (probed behaviourally: ``call_tool`` must not raise NotImplementedError).

    A new ``dispatch_tool('new_tool', ...)`` added without a matching stub branch
    fails here, naming ``new_tool`` — caught at CI, not at eval runtime.
    """
    literals = _dispatch_tool_literals()
    # Guard against a broken scanner producing an empty set (vacuous pass).
    assert literals, (
        'AST scan found no dispatch_tool string literals in scheduler.py — the '
        'scanner is broken or dispatch_tool was renamed/refactored. Fix '
        '_dispatch_tool_literals before trusting this guard.'
    )

    missing: list[str] = []
    for name in sorted(literals):
        stub = _StubMcpSession()
        try:
            await stub.call_tool(name, {})
        except NotImplementedError:
            missing.append(name)
        except Exception:
            # Any other outcome (e.g. KeyError from the empty args dict) proves
            # the branch exists — this decouples the probe from arg-shape details.
            pass

    assert not missing, (
        'The following tool(s) are dispatched via dispatch_tool(...) in '
        'scheduler.py but have NO branch in _StubMcpSession '
        '(orchestrator/src/orchestrator/evals/runner.py). An eval run would '
        'raise NotImplementedError → swallowed heartbeat WARNING spam (D2/B9). '
        'Add a call_tool branch for each:\n  ' + '\n  '.join(missing)
    )
