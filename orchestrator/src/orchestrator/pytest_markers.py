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
from collections.abc import Callable, Sequence


class _Unsupported(Exception):
    """A marker-expression node outside this module's deliberately tiny grammar."""


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


#: Shell chain operators that terminate one clause of a chained command.
#: Mirrors ``verify_cmd._CHAIN_OPERATOR_TOKENS`` by value, duplicated rather
#: than imported to keep this module dependency-free (see the module docstring).
_CHAIN_OPERATOR_TOKENS = frozenset({'&&', '||', ';', '|'})


def _cli_marker_expr(test_command: str | None) -> str | None:
    """The ``-m`` expression of the FIRST ``pytest`` clause in *test_command*.

    Two restrictions, both LOAD-BEARING:

    * only tokens AFTER the ``pytest`` keyword are scanned — this is what keeps
      ``python -m pytest tests/`` from being misread as the marker expression
      ``'pytest'``;
    * the scan STOPS at the first shell chain operator following that keyword,
      so a chained ``pytest tests/ && python -m mytool`` cannot contribute
      ``'mytool'``, which would otherwise override the real ``addopts``
      expression and silently suppress a legitimate widening.

    FIRST-occurrence-then-truncate-at-the-chain-delimiter is deliberately the
    same semantics ``verify_plan._scope_prefix_to_keyword`` applies when it
    scopes the very same string (``split_chain_tail`` + ``head.find(keyword)``),
    so the marker probe and the scoper can never describe DIFFERENT pytest
    invocations of one chained command.  ``split_chain_tail`` itself is not
    reusable here: ``'pytest'`` is deliberately off its
    ``_TAIL_PRESERVING_KEYWORDS`` allowlist (task 3218), so for this keyword it
    returns the whole string unsplit and would answer no question at all.

    A command with no ``pytest`` keyword yields None, leaving the addopts
    expression untouched.
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
            break
    if keyword_index is None:
        return None
    clause = tokens[keyword_index + 1:]
    for offset, token in enumerate(clause):
        if token in _CHAIN_OPERATOR_TOKENS:
            clause = clause[:offset]
            break
    return _marker_expr_from_tokens(clause)


def resolve_marker_expression(
    pyproject_text: str | None,
    test_command: str | None,
) -> str | None:
    """The module's effective pytest ``-m`` marker expression, else None.

    Resolution order is pytest's documented last-wins rule, stated verbatim in
    ``orchestrator/pyproject.toml``'s ``warm_lane_bash`` marker text ("a CLI -m
    overrides the addopts -m, last wins"): the
    ``[tool.pytest.ini_options].addopts`` expression is the base, and a ``-m``
    inside *test_command*'s FIRST ``pytest`` clause replaces it (see
    :func:`_cli_marker_expr` for why that clause, and only that clause).

    *pyproject_text* is the content of the ini file at pytest's ROOTDIR, which
    the caller locates from the command's effective cwd — see
    ``verify_plan._marker_deselecting_expression``.  This function does not and
    cannot check that the two describe the same invocation.

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


def _kleene(node: ast.expr, marker_names: frozenset[str]) -> bool | None:
    """Evaluate *node* under Kleene (strong 3-valued) logic; None is UNKNOWN.

    A name in *marker_names* is TRUE (guaranteed present on every item); a name
    outside it is UNKNOWN, because *marker_names* is only a LOWER BOUND — an
    individual item may carry additional markers.  Any node outside the tiny
    allowed grammar raises :class:`_Unsupported`.
    """
    if isinstance(node, ast.Name):
        return True if node.id in marker_names else None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return node.value
        raise _Unsupported(f'non-boolean constant: {node.value!r}')
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = _kleene(node.operand, marker_names)
        return None if operand is None else not operand
    if isinstance(node, ast.BoolOp):
        values = [_kleene(value, marker_names) for value in node.values]
        if isinstance(node.op, ast.And):
            if any(value is False for value in values):
                return False
            return None if any(value is None for value in values) else True
        if isinstance(node.op, ast.Or):
            if any(value is True for value in values):
                return True
            return None if any(value is None for value in values) else False
    raise _Unsupported(f'unsupported node: {type(node).__name__}')


def expression_definitely_deselects(expr: str, marker_names: frozenset[str]) -> bool:
    """True iff *expr* provably deselects EVERY collected item in the file.

    *marker_names* is the file's guaranteed marker set from
    :func:`module_level_marker_names` — a LOWER BOUND, so a name outside it is
    UNKNOWN rather than false.  Evaluating under Kleene logic on that reading
    means a definite FALSE is "False under EVERY assignment of the unknown
    names", which is exactly the property needed: no item in the file can be
    selected.  A naive two-valued eval would be unsound in the non-monotone
    direction — ``not a or b`` with ``a`` guaranteed reads False, yet an item
    that also carries ``b`` IS selected.

    Kleene is sound but INCOMPLETE: it evaluates each occurrence of a name
    independently, so the contradiction ``a and not a`` reads UNKNOWN rather
    than FALSE.  The incompleteness fails SAFE — no widening, today's
    FILE_SCOPED behaviour.  It was preferred over brute-forcing all assignments
    of the free names because it is linear, needs no arity bound, and decides
    every marker expression live in this repo (``not warm_lane_bash``,
    ``not integration``, ``not smoke``) exactly.

    The grammar is deliberately tiny: ``Name``, ``BoolOp(And/Or)``,
    ``UnaryOp(Not)`` and boolean ``Constant``.  Anything richer that pytest
    itself accepts (``Call``, ``Compare``, keyword-expression forms) bails to
    False.  A ``SyntaxError``, a ``ValueError`` and an empty *expr* likewise
    yield False.  Never raises.
    """
    if not expr or not expr.strip():
        return False
    try:
        tree = ast.parse(expr, mode='eval')
    except (SyntaxError, ValueError):
        return False
    try:
        return _kleene(tree.body, marker_names) is False
    except (_Unsupported, RecursionError):
        return False


def deselecting_expression_for_targets(
    targets: Sequence[str],
    pyproject_text: str | None,
    test_command: str | None,
    read_source: Callable[[str], str | None],
) -> str | None:
    """The module's effective ``-m`` expression iff it deselects EVERY target, else None.

    The composed entry point: resolve the module's marker expression
    (:func:`resolve_marker_expression`), then require every target to be
    provably fully deselected by it (:func:`module_level_marker_names` +
    :func:`expression_definitely_deselects`).  ALL, not ANY — a single target
    that still collects means the file-scoped run is not empty.  The EXPRESSION
    is returned rather than a bool so the caller can name it in the
    operator-facing ``PlannedRun.reason``.

    *read_source* mirrors ``verify_plan``'s injected ``worktree_reader``
    (``Callable[[str], str | None]``) exactly, so no new I/O seam is introduced
    and its content cache is shared: a touched test file already read for
    STRUCTURAL detection costs zero extra disk I/O.  A ``None`` answer (missing
    or unreadable) proves nothing and refuses.

    COST BOUND: with no ``-m`` expression resolved this performs ZERO target
    reads — it short-circuits BEFORE calling *read_source* at all.  The added
    cost of consulting this from a verify plan is therefore exactly one
    pyproject read per ModuleConfig, and nothing more for any module that
    declares no marker expression.

    DIRECTION OF SAFETY: a None return always means "keep today's FILE_SCOPED
    behaviour".  Widening is only ever chosen on positive proof, so this can
    turn a false RED into a real run but never a real run into a skip.  An empty
    *targets* is refused rather than treated as vacuously all-deselected.
    """
    if not targets:
        return None
    expr = resolve_marker_expression(pyproject_text, test_command)
    if expr is None:
        return None
    for target in targets:
        guaranteed = module_level_marker_names(read_source(target))
        if not expression_definitely_deselects(expr, guaranteed):
            return None
    return expr
