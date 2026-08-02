"""Static detection of "every file-scoped pytest target is marker-deselected" (task 3494).

WHY STATIC, not a ``--collect-only`` probe.  This module's sole consumer is
``verify_plan._derive_module_runs``, and ``derive_verify_plan``'s docstring makes
purity an explicit invariant of that layer ("never executes that guard itself,
staying pure"): it is driven in every test through an injected
``worktree_reader`` with no filesystem and no subprocess.  ``_derive_module_runs``
runs once per ModuleConfig per verify plan, so a collection subprocess there
would put a real pytest collection (28.06s measured for the orchestrator module
alone) on the critical path of every verify, and would need a cache plus an
invalidation story keyed on content that changes inside the very diff being
verified.  The static parse instead reads facts that already exist in the repo —
``[tool.pytest.ini_options].addopts`` and a module-level ``pytestmark`` — at the
cost of one already-cached file read per module.

Every function here is PURE: no filesystem, no subprocess, no environment.  File
CONTENT arrives as a string (or None), and the composed entry point takes a
``read_source`` callable, mirroring ``verify_plan``'s existing ``worktree_reader``
seam exactly rather than introducing a second I/O seam.

FAIL-SAFE IN EXACTLY ONE DIRECTION.  Any unreadable file, TOML/AST/shlex failure,
unsupported expression node, or merely-unknown marker resolves to "no widening" —
i.e. precisely today's behaviour.  Widening is only ever chosen on positive
proof.  Nothing here raises: ``verify._safe_derive_verify_plan_dict`` swallows
exceptions and returns None, so a raise on a mid-edit ``pyproject.toml`` would
silently destroy the ENTIRE plan record.
"""
from __future__ import annotations

import shlex
import tomllib
from collections.abc import Sequence


def _marker_expr_from_tokens(tokens: Sequence[str]) -> str | None:
    """The LAST ``-m EXPR`` / ``-mEXPR`` value in *tokens*, else None.

    Last-wins mirrors pytest's own handling of a repeated ``-m``.
    """
    found: str | None = None
    for index, token in enumerate(tokens):
        if token == '-m':
            if index + 1 < len(tokens):
                found = tokens[index + 1]
        elif token.startswith('-m') and len(token) > 2:
            found = token[2:]
    return found


def _addopts_tokens(pyproject_text: str | None) -> list[str] | None:
    """``[tool.pytest.ini_options].addopts`` from *pyproject_text*, as a token list.

    A ``str`` addopts is split with ``shlex``; a list keeps only its ``str``
    elements.  Any other type, any malformed TOML, and any missing/non-dict
    intermediate table yields None.
    """
    if not pyproject_text:
        return None
    try:
        data = tomllib.loads(pyproject_text)
    except (tomllib.TOMLDecodeError, ValueError, TypeError):
        return None
    node: object = data
    for key in ('tool', 'pytest', 'ini_options', 'addopts'):
        if not isinstance(node, dict):
            return None
        if key not in node:
            return None
        node = node[key]
    if isinstance(node, str):
        try:
            return shlex.split(node)
        except ValueError:
            return None
    if isinstance(node, list):
        return [element for element in node if isinstance(element, str)]
    return None


def _cli_marker_expr(test_command: str | None) -> str | None:
    """The ``-m`` expression appearing AFTER the ``pytest`` keyword in *test_command*.

    The post-keyword restriction is LOAD-BEARING: it is what keeps
    ``python -m pytest tests/`` from being misread as the marker expression
    ``'pytest'``.  Only tokens following the last token that is ``pytest`` (or
    ends in ``/pytest``) are scanned; a command with no such token yields None,
    leaving the addopts expression untouched.
    """
    if not test_command:
        return None
    try:
        tokens = shlex.split(test_command)
    except ValueError:
        return None
    keyword_index: int | None = None
    for index, token in enumerate(tokens):
        if token == 'pytest' or token.endswith('/pytest'):
            keyword_index = index
    if keyword_index is None:
        return None
    return _marker_expr_from_tokens(tokens[keyword_index + 1:])


def resolve_marker_expression(
    pyproject_text: str | None,
    test_command: str | None,
) -> str | None:
    """The module's effective pytest ``-m`` marker expression, else None.

    Resolution order is pytest's documented last-wins rule, stated verbatim in
    ``orchestrator/pyproject.toml``'s ``warm_lane_bash`` marker text ("a CLI -m
    overrides the addopts -m, last wins"): the
    ``[tool.pytest.ini_options].addopts`` expression is the base, and a ``-m``
    appearing after the ``pytest`` keyword in *test_command* replaces it.

    Never raises — every failure path returns None.

    Caveat, recorded for the reader rather than handled here: ``verify_cmd``'s
    serial-retry recovery appends ``-o addopts=`` at EXECUTION time, which clears
    the addopts ``-m`` after planning.  A retry can therefore select MORE than
    the plan assumed, which is the safe direction (extra coverage, never less).
    """
    cli_expr = _cli_marker_expr(test_command)
    if cli_expr is not None:
        return cli_expr
    tokens = _addopts_tokens(pyproject_text)
    if tokens is None:
        return None
    return _marker_expr_from_tokens(tokens)
