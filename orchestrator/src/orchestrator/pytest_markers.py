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

import ast
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


def _is_pytestmark_target(node: ast.expr) -> bool:
    """True iff *node* is the bare name ``pytestmark``."""
    return isinstance(node, ast.Name) and node.id == 'pytestmark'


def _pytestmark_value(statement: ast.stmt) -> ast.expr | None:
    """The value *statement* binds to ``pytestmark``, else None.

    Covers both the plain ``pytestmark = ...`` and the annotated
    ``pytestmark: list = ...`` spellings; an annotation with no value binds
    nothing.
    """
    if isinstance(statement, ast.Assign):
        if any(_is_pytestmark_target(target) for target in statement.targets):
            return statement.value
        return None
    if isinstance(statement, ast.AnnAssign) and _is_pytestmark_target(statement.target):
        return statement.value
    return None


def _marker_name(element: ast.expr) -> str | None:
    """The marker name in a ``pytest.mark.NAME`` / ``pytest.mark.NAME(...)`` element.

    Anything else — a bare constant, a local name, an unrelated attribute chain —
    yields None and is skipped silently, without suppressing its siblings.
    """
    if isinstance(element, ast.Call):
        element = element.func
    if not isinstance(element, ast.Attribute):
        return None
    owner = element.value
    if (
        isinstance(owner, ast.Attribute)
        and owner.attr == 'mark'
        and isinstance(owner.value, ast.Name)
        and owner.value.id == 'pytest'
    ):
        return element.attr
    return None


def module_level_marker_names(source: str | None) -> frozenset[str]:
    """Marker names a module-level ``pytestmark`` applies to EVERY item in *source*.

    THE LOAD-BEARING CONTRACT: the return value is a **LOWER BOUND** on every
    collected item's marker set.  A module-level ``pytestmark`` is the only
    static form that provably applies to all of them, so per-function and
    per-class ``@pytest.mark.X`` decorators are deliberately NOT collected — a
    decorator sweep would have to reason about ``pytest.param(marks=...)``,
    parametrize, dynamically generated items, test classes imported from another
    module, and ``pytest_collection_modifyitems`` hooks, each a way for an
    unmarked (hence SELECTED) item to exist in a file that "looks" fully marked.

    A name ABSENT from this set is therefore UNKNOWN, not absent — which is
    precisely what makes :func:`expression_definitely_deselects`' Kleene
    treatment sound.  Excluding decorators makes the detector under-fire on some
    genuinely-deselected files, which is the safe direction.

    Accepted value shapes: a bare ``pytest.mark.NAME``, a ``pytest.mark.NAME(...)``
    call, or a list/tuple of either.  Only ``tree.body`` is walked (never
    ``ast.walk``), so a ``pytestmark`` bound inside a class or function body does
    not count.  If the module rebinds ``pytestmark`` more than once, the LAST
    assignment wins, mirroring Python's own semantics.

    ``source is None``, a ``SyntaxError``, or a ``ValueError`` yields an empty
    set.  Never raises.
    """
    if not source:
        return frozenset()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()

    value: ast.expr | None = None
    for statement in tree.body:
        bound = _pytestmark_value(statement)
        if bound is not None:
            value = bound
    if value is None:
        return frozenset()

    elements = list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
    return frozenset(
        name for name in (_marker_name(element) for element in elements) if name is not None
    )
