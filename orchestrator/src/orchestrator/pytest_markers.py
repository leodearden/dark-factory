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
unsupported expression node, merely-unknown marker, module shape neither the
per-item nor the class-level tier can exhaustively enumerate (see
:func:`per_item_marker_names` and :func:`guaranteed_marker_names`), or a
module with zero collected items resolves to "no widening" — i.e. precisely
today's behaviour.  Widening is only ever chosen on positive proof.  Nothing
here raises: ``verify._safe_derive_verify_plan_dict`` swallows exceptions and
returns None, so a raise on a mid-edit ``pyproject.toml`` would silently
destroy the ENTIRE plan record.
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
    ``verify_plan.deselecting_expression_for_command``.  This function does not
    and cannot check that the two describe the same invocation.

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


def _pytestmark_marker_names(body: Sequence[ast.stmt]) -> frozenset[str]:
    """Marker names the ``pytestmark`` binding in *body* names, else an empty set.

    THE ONE SHARED FOLD, covering both scopes at which pytest honours
    ``pytestmark``: a MODULE body (:func:`module_level_marker_names` and its
    tree-consuming twin :func:`_module_level_marker_names_from_tree`) and a
    CLASS body (:func:`_class_marker_names`).  Those three held
    character-for-character copies of this fold, differing only in the
    statement container, until they were collapsed here.  ONE copy is what
    stops the two readings of ``pytestmark`` syntax drifting apart — a drift
    would silently let the class tier accept a value shape the module tier
    rejects, breaking the "same shapes at both scopes" promise
    :func:`_class_marker_names` makes.

    Only DIRECT children of *body* are considered — never ``ast.walk`` — so a
    ``pytestmark`` bound inside an ``if`` is not the enclosing scope's marker.
    Accepted value shapes: a bare ``pytest.mark.NAME``, a
    ``pytest.mark.NAME(...)`` call, or a list/tuple of either.  A non-marker
    element yields None from :func:`_marker_name` and is skipped silently,
    without suppressing its siblings.  If the scope rebinds ``pytestmark`` more
    than once, the LAST binding wins, mirroring Python's own semantics.
    """
    value: ast.expr | None = None
    for statement in body:
        bound = _pytestmark_value(statement)
        if bound is not None:
            value = bound
    if value is None:
        return frozenset()

    elements = list(value.elts) if isinstance(value, ast.List | ast.Tuple) else [value]
    return frozenset(
        name for name in (_marker_name(element) for element in elements) if name is not None
    )


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
    genuinely-deselected files, which is the safe direction.  Those decorators are
    instead handled by two separate, enumeration-guarded tiers, neither of which
    affects this function's own contract: per-FUNCTION decorators by
    :func:`per_item_marker_names`, and per-CLASS markers by
    :func:`guaranteed_marker_names`.

    :func:`guaranteed_marker_names` is deliberately a SIBLING that widens this
    bound rather than an edit to it (task 4561, esc-3513-2).  Consumers reason
    about the set returned HERE as the strict module-only bound — it is what
    each of those tiers is defined as a superset OF — so widening it in place
    would move the baseline every one of them is stated against.

    The ``pytestmark`` read itself is :func:`_pytestmark_marker_names`, the ONE
    shared fold this module uses at every scope: accepted value shapes are a bare
    ``pytest.mark.NAME``, a ``pytest.mark.NAME(...)`` call, or a list/tuple of
    either; only ``tree.body`` is walked (never ``ast.walk``), so a ``pytestmark``
    bound inside a class or function body does not count; and if the module
    rebinds ``pytestmark`` more than once the LAST assignment wins, mirroring
    Python's own semantics.

    ``source is None``, a ``SyntaxError``, or a ``ValueError`` yields an empty
    set.  Never raises.
    """
    if not source:
        return frozenset()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()

    return _pytestmark_marker_names(tree.body)


def _class_marker_names(node: ast.ClassDef) -> frozenset[str]:
    """Marker names *node* applies to every item collected from its own body.

    Reads BOTH spellings pytest honours on a class, and unions them: the
    ``@pytest.mark.NAME`` DECORATORS on the class itself, and a ``pytestmark``
    bound in the class BODY.  The body form is read by the SAME
    :func:`_pytestmark_marker_names` fold :func:`module_level_marker_names` uses
    at module scope, so the two readings of ``pytestmark`` syntax cannot drift
    apart — that is now structural rather than a claim maintained by hand.  It
    inherits that fold's rules verbatim: the same accepted value shapes, the
    same LAST-binding-wins rule, and DIRECT children of ``node.body`` only, so a
    ``pytestmark`` bound inside an ``if`` in the class body is not the class's
    marker.

    A non-marker element (e.g. the ``qdrant_skipif()`` call heading the real
    shape at ``fused-memory/tests/test_mem0_client.py``) yields None from
    :func:`_marker_name` and is skipped silently, without suppressing its
    siblings.
    """
    markers = {
        name
        for name in (_marker_name(decorator) for decorator in node.decorator_list)
        if name is not None
    }

    return frozenset(markers) | _pytestmark_marker_names(node.body)


def _is_collectable_class(node: ast.ClassDef) -> bool:
    """True iff pytest may collect test items from *node* itself.

    Collectable iff the class is ``test``-PREFIXED (case-insensitively,
    matching the default ``python_classes = Test*``) **OR** its body directly
    defines a ``test*``-named function.

    THE SECOND DISJUNCT IS WHAT MAKES THE PREFIX ASSUMPTION SELF-CHECKING.
    :func:`per_item_marker_names`' docstring already records that the default
    ``python_classes`` is a premise this module cannot verify; a repo that
    overrides it to collect ``FooSuite`` would slip straight past a pure
    prefix rule.  Requiring that a non-``Test*``-named class holding ``test*``
    methods also be marked — or else force a refusal — moves that premise's
    failure into the SAFE direction.

    Deliberately NOT :func:`per_item_marker_names`' "refuse on class SHAPE,
    not class NAME": that tier can afford to refuse on any class because a
    class is outside the item shape it models at all, whereas
    :func:`guaranteed_marker_names` exists precisely to reason about classes,
    and refusing on every unmarked helper class (``class _Config: ...``) would
    kill it on most real modules.  The ``test`` prefix is not a new rule
    either — :func:`_bound_names_start_with_test_ci` already applies exactly
    this case-insensitive prefix to imported names, for the same reason.
    """
    if node.name.lower().startswith('test'):
        return True
    return any(
        isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        and statement.name.startswith('test')
        for statement in node.body
    )


def guaranteed_marker_names(source: str | None) -> frozenset[str]:
    """Marker names that provably apply to EVERY item collected from *source*.

    THE LOAD-BEARING CONTRACT, inherited verbatim from
    :func:`module_level_marker_names`: the return value is a **LOWER BOUND** on
    every collected item's marker set, so a name ABSENT from it is UNKNOWN
    rather than absent — which is exactly what keeps
    :func:`expression_definitely_deselects`' Kleene reading sound.  This tier
    widens that bound WITHOUT editing it: it is a sibling, not a replacement,
    because ``module_level_marker_names``' own set must stay the strict
    module-only bound its consumers reason about.

    The answer is ``module_level_marker_names(source)`` unioned with the
    INTERSECTION of the marker sets of every COLLECTABLE top-level class (see
    :func:`_is_collectable_class`), and only when the all-items-accounted-for
    guard below proves no collected item can exist outside those classes.
    Class markers are read from BOTH spellings pytest honours — the class
    decorators and a class-body ``pytestmark`` (:func:`_class_marker_names`).

    INTERSECTION, NEVER UNION.  If ``TestA`` carries ``slow`` and ``TestB``
    carries ``integration``, neither marker is a module-wide bound: a union
    would claim ``slow`` for ``TestB``'s items, ``not slow`` would read
    definitely-False, and a run that genuinely collects ``TestB`` would be
    widened away — the over-fire that reopens esc-3292-1 / task 1852.  The
    intersection is the only fold that preserves the bound.

    THE ALL-ITEMS-ACCOUNTED-FOR GUARD.  Falls back to the module-level answer
    alone on any of:

    * a ``test*``-named function (``def`` or ``async def``) that is NOT a
      direct child of a top-level class body — the ONE rule covering a
      module-level test function, a test hidden inside a top-level ``if``,
      and a test nested inside another function;
    * a ``ClassDef`` anywhere below ``tree.body`` — nested in another class,
      or inside a top-level ``if`` — whose body this fold never reaches;
    * a top-level ``pytest_*`` hook (``pytest_collection_modifyitems``,
      ``pytest_generate_tests``), which can add items this walk never sees;
    * a star import, whose bound names are statically unknowable;
    * any import binding a ``test``-prefixed name case-insensitively,
      honouring ``asname`` — an imported ``Test*`` CLASS or ``test_*``
      FUNCTION is collected in THIS module;
    * a top-level ``Assign``/``AnnAssign`` binding a ``test*``-prefixed name
      (``test_generated = _make_case()``) — a dynamically generated item;
    * ZERO collectable classes.  Guarded EXPLICITLY rather than left to the
      fold: ``frozenset.intersection()`` over an empty family is
      conventionally "every marker", which would prove any expression false.

    WHY THE RESULT IS STILL A LOWER BOUND.  Every item pytest collects from
    the module is a test function; by the first rule every such function is a
    direct child of a top-level class body; by the second there is no
    un-walked class body; by the import and dynamic-binding rules no item
    enters from elsewhere; by the hook rule nothing in this file adds one.
    Every owning class is collectable by definition (it holds a ``test*``
    method), hence contributes to the intersection, hence carries every marker
    in it.  Marks compose ADDITIVELY up the Module -> Class -> Function node
    chain, so ``parametrize`` and ``pytest.param(marks=...)`` can only ever
    ADD to an item's set: they multiply items but cannot break a
    universally-quantified bound.  Inherited test methods are collected under
    the marked SUBCLASS's node and carry its marks, so a base class defined in
    another module is not a hole either.

    DELIBERATE OVER-REFUSALS, all in the safe direction: any non-top-level
    class refuses even when it is plainly inert; a collectable-looking class
    that is simply unmarked drops the intersection to empty; and a helper
    class that happens to define a ``test*``-named method counts as
    collectable and so must be marked or refuse.

    ASSUMPTIONS THIS WALK CANNOT CHECK, identical to
    :func:`per_item_marker_names`': the caller's pytest configuration uses the
    DEFAULT collection prefixes (``python_functions = test*``,
    ``python_classes = Test*``), and no ancestor ``conftest.py`` implements an
    item-ADDING ``pytest_collection_modifyitems``/``pytest_generate_tests``
    hook — this walk only refuses on such a hook defined INSIDE *source*
    itself, a sibling ``conftest.py`` being outside a single module string's
    reach.  Both are pre-existing limits of a purely per-file static analysis,
    recorded here rather than fixed, since fixing them would need reading
    files this function is not given.

    STRICTLY ADDITIVE.  Every guard failure returns exactly
    ``module_level_marker_names``' answer — never a smaller set — so this
    function is a provable SUPERSET of that tier on every input and can never
    refuse a file the primary tier already proves.

    ``source is None``, a ``SyntaxError``, or a ``ValueError`` yields an empty
    set.  Never raises.
    """
    if not source:
        return frozenset()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()

    module_markers = _module_level_marker_names_from_tree(tree)

    body_ids = {id(statement) for statement in tree.body}
    class_body_ids = {
        id(statement)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for statement in node.body
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if id(node) not in body_ids:
                return module_markers
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith('test') and id(node) not in class_body_ids:
                return module_markers
            if node.name.startswith('pytest_') and id(node) in body_ids:
                return module_markers
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == '*' for alias in node.names):
                return module_markers
            if _bound_names_start_with_test_ci(node):
                return module_markers
        elif (
            (isinstance(node, ast.Import) and _bound_names_start_with_test_ci(node))
            or (
                isinstance(node, ast.Assign | ast.AnnAssign)
                and id(node) in body_ids
                and _assign_binds_test_prefixed_name(node)
            )
        ):
            return module_markers

    collectable = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _is_collectable_class(node)
    ]
    if not collectable:
        return module_markers
    shared = frozenset.intersection(*(_class_marker_names(node) for node in collectable))
    return module_markers | shared


def _bound_names_start_with_test_ci(node: ast.Import | ast.ImportFrom) -> bool:
    """True iff any alias in *node* binds a name starting with ``test`` (case-insensitive).

    Case-insensitive because pytest's collection is not limited to the
    ``Test*`` class-naming convention: the default ``python_functions = test*``
    collects any module-level callable named ``test*`` too, and an alias can
    import EITHER shape into this module — ``from helpers import TestBase``
    (a class) or ``from helpers import test_shared_case`` (a function) are
    both collected here.  ``asname`` wins, matching the name that actually
    lands in this module's namespace.
    """
    return any(
        (alias.asname or alias.name.split('.')[0]).lower().startswith('test')
        for alias in node.names
    )


def _assign_binds_test_prefixed_name(node: ast.Assign | ast.AnnAssign) -> bool:
    """True iff *node* binds a ``test*``-prefixed name (case-insensitive) to a ``Name`` target.

    Pytest's default ``python_functions = test*`` collects any module
    attribute so named that resolves to a callable — including one bound by
    a plain assignment (``test_generated = _make_case()``), not only a
    ``def``.  This walk cannot tell statically whether the bound value is
    actually callable, so any ``test*``-prefixed target refuses, in the safe
    direction.
    """
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(
        isinstance(target, ast.Name) and target.id.lower().startswith('test')
        for target in targets
    )


def _module_level_marker_names_from_tree(tree: ast.Module) -> frozenset[str]:
    """Same result as :func:`module_level_marker_names`, from an already-parsed *tree*.

    THE SHARED tree-consuming walk: both :func:`per_item_marker_names` and
    :func:`guaranteed_marker_names` call it off a tree they already hold,
    rather than calling ``module_level_marker_names(source)`` and paying a
    second, wholly redundant ``ast.parse`` of the same source purely to
    re-derive this set.

    It was introduced (task 3513 Gap 3) as a deliberate small duplicate of
    ``module_level_marker_names``' body, on the then-current understanding
    that task 4561 would edit that function IN PLACE, and it left a
    sequencing note asking whichever task landed second to fold its walk into
    the tier the other had added rather than adding a THIRD copy.  4561 has
    since landed and did NOT edit ``module_level_marker_names``: esc-3513-2
    re-specced Gap 2 as a SIBLING, :func:`guaranteed_marker_names`.

    The note's instruction is honoured in the strongest available form: there
    is now exactly ONE copy of the fold, :func:`_pytestmark_marker_names`, and
    every scope that reads ``pytestmark`` — this helper,
    ``module_level_marker_names`` and :func:`_class_marker_names` — calls it.
    What survives here is only the tree-consuming ENTRY POINT, kept because
    ``per_item_marker_names`` and ``guaranteed_marker_names`` hold a tree
    rather than a body and because this docstring is the record of the
    3513/4561 sequencing agreement.
    """
    return _pytestmark_marker_names(tree.body)


def per_item_marker_names(source: str | None) -> tuple[frozenset[str], ...] | None:
    """One guaranteed (lower-bound) marker set per top-level test item in *source*.

    THE LOAD-BEARING ENUMERATION GUARANTEE: the returned tuple enumerates EVERY
    item pytest can collect from this module, in source order — or the answer
    is None.  This is a SECOND, additive proof tier alongside
    :func:`module_level_marker_names` and does not weaken that function's own
    module-wide LOWER BOUND contract: each element here is still a lower bound
    on its item's actual marker set (``module_level_marker_names(source)``
    unioned with that item's own ``pytest.mark.NAME`` decorators), so the
    Kleene reading in :func:`expression_definitely_deselects` — a name outside
    the set is UNKNOWN, never False — carries over unchanged.

    Refuses (returns None) whenever the module contains a shape whose
    collected items this walk cannot exhaustively see:

    * any ``class`` anywhere in the module (``ast.walk``, not just the module
      body) — refused regardless of its name.  This is a DELIBERATE
      over-refusal for simplicity: only ``Test*``-prefixed classes are
      collected under the default ``python_classes``, so a name-prefix
      carve-out (mirroring the ``test*`` prefix already applied below to
      functions) would fire more often, but refusing on class SHAPE rather
      than class NAME keeps this tier's competence statable in one line —
      "a class means there may be items this walk does not model" — without
      a second, independently-driftable prefix rule;
    * a ``test*``-named function found anywhere that is NOT a direct child of
      the module body — e.g. one defined inside a top-level ``if`` — which
      would otherwise hide an undecorated, still-SELECTED sibling from this
      walk and let the module widen unsoundly;
    * a star import (``from x import *``), whose bound names are statically
      unknowable;
    * any import (plain or ``from``) that binds a name starting with ``test``
      case-insensitively (honouring ``asname``) — this covers BOTH an
      imported ``Test*`` class and an imported lowercase ``test_*`` function,
      either of which pytest collects in THIS module under the respective
      default;
    * a top-level assignment (``Assign``/``AnnAssign``) that binds a
      ``test*``-prefixed name to a plain ``Name`` target — e.g.
      ``test_generated = _make_case()`` — which the default
      ``python_functions = test*`` collects exactly as it would a ``def``,
      and which this walk cannot otherwise tell apart from an ordinary
      module constant;
    * a top-level ``pytest_*`` hook function (e.g.
      ``pytest_collection_modifyitems``, ``pytest_generate_tests``), which can
      add items this walk never sees.

    ASSUMPTIONS THIS WALK CANNOT CHECK: the caller's pytest configuration uses
    the DEFAULT collection prefixes (``python_functions = test*``,
    ``python_classes = Test*``) — a repo that overrides either in its ini
    options can collect items this walk's name-prefix reasoning does not
    model.  It also assumes no ancestor ``conftest.py`` implements an
    item-ADDING ``pytest_collection_modifyitems``/``pytest_generate_tests``
    hook: this walk only refuses on such a hook defined INSIDE *source*
    itself, because a sibling ``conftest.py``'s content is outside a single
    module string's reach.  Both are pre-existing limits of a purely
    per-file static analysis, recorded here rather than fixed, since fixing
    them would need reading files this function is not given.

    FAIL-SAFE IN EXACTLY ONE DIRECTION, matching the module docstring: every
    refusal above is a None, i.e. no proof, i.e. today's FILE_SCOPED
    behaviour.  None also covers ``source is None``, a ``SyntaxError``, and a
    ``ValueError``.  ``()`` is a distinct, still-refused answer: "enumerated,
    and there are zero top-level test functions" (see
    :func:`deselecting_expression_for_targets`, which treats both alike).
    Never raises.
    """
    if not source:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None

    body_ids = {id(statement) for statement in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return None
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            is_top_level = id(node) in body_ids
            if node.name.startswith('test') and not is_top_level:
                return None
            if node.name.startswith('pytest_') and is_top_level:
                return None
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == '*' for alias in node.names):
                return None
            if _bound_names_start_with_test_ci(node):
                return None
        elif (
            (isinstance(node, ast.Import) and _bound_names_start_with_test_ci(node))
            or (
                isinstance(node, ast.Assign | ast.AnnAssign)
                and id(node) in body_ids
                and _assign_binds_test_prefixed_name(node)
            )
        ):
            return None

    module_markers = _module_level_marker_names_from_tree(tree)
    items: list[frozenset[str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith('test'):
            continue
        decorator_markers = frozenset(
            name
            for name in (_marker_name(decorator) for decorator in node.decorator_list)
            if name is not None
        )
        items.append(module_markers | decorator_markers)
    return tuple(items)


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
    provably fully deselected by it under a TWO-TIER proof, tried in order per
    target:

    1. the PRIMARY tier — :func:`guaranteed_marker_names` +
       :func:`expression_definitely_deselects` — a MODULE-WIDE lower bound,
       covering a module-level ``pytestmark`` and, under that function's
       all-items-accounted-for guard, CLASS-level markers too (task 4561);
    2. only if that fails to prove deselection, the FALLBACK tier —
       :func:`per_item_marker_names` — which enumerates every collected item's
       own (module-level union per-decorator) marker set and requires EVERY
       one of them to be individually, definitely deselected.

    WHAT TIER 1 DOES WITH THE HARD SHAPES.  It covers test classes, imported
    test classes and dynamically generated items by REFUSING to widen on
    them, unless its guard can prove every collected item lives inside a
    marked class.  An imported test class and a dynamic top-level binding
    still refuse OUTRIGHT — no guard can see what they bring in — so only the
    all-classes-marked shape is newly provable.

    Both tiers are strictly ADDITIVE: each only ever turns a refusal into a
    proof, never the reverse, so this function can never refuse a target it
    already accepts today.  For tier 2 that holds because it is consulted only
    after tier 1 declines; for tier 1 it holds because
    :func:`guaranteed_marker_names` returns exactly
    :func:`module_level_marker_names`' answer on every guard failure and a
    SUPERSET of it otherwise, and :func:`expression_definitely_deselects` is
    monotone in its marker set.  ALL, not ANY — across both tiers and
    across every target — a single target (or a single item within a target)
    that still collects means the file-scoped run is not empty.  The
    EXPRESSION is returned rather than a bool so the caller can name it in the
    operator-facing ``PlannedRun.reason``.

    *read_source* mirrors ``verify_plan``'s injected ``worktree_reader``
    (``Callable[[str], str | None]``) exactly, so no new I/O seam is introduced
    and its content cache is shared: a touched test file already read for
    STRUCTURAL detection costs zero extra disk I/O.  Each target's source is
    read EXACTLY ONCE regardless of how many tiers consult it.  A ``None``
    answer (missing or unreadable) proves nothing and refuses.

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
        source = read_source(target)
        if expression_definitely_deselects(expr, guaranteed_marker_names(source)):
            continue
        item_markers = per_item_marker_names(source)
        if not item_markers:
            return None
        if not all(expression_definitely_deselects(expr, markers) for markers in item_markers):
            return None
    return expr
