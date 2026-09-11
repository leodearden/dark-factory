"""ONE canonical verify-budget derivation for the tests/scripts guard family, and a reader for it.

Task 4320. Three guards in this directory hold the same (measurement, derived
floor) contract for a module config's ``verify_command_timeout_secs``:

  * ``test_tests_scripts_module_config.py``  publishes the ``tests/scripts`` pair
  * ``test_scripts_module_config.py``        publishes the ``scripts`` pair
  * ``test_module_verify_budgets.py``        generalises the contract across
    every discovered module config, and is the family's designated ANTI-DRIFT
    member

Until this module existed the derivation lived in THREE spellings — two
``def _min_budget`` copies and one inlined ``(int(2 * MEASURED_SUITE_WORST_SECS)
// 100) * 100`` — and nothing in the repo could observe drift between them. The
owning guard said so itself: its derivation test carried a scope paragraph
conceding that "``_min_budget`` is a LOCAL copy ... nothing here can observe a
sibling's expression: if ``test_scripts_module_config.py`` changed its own
inline derivation, every assertion below would still pass", and that real
cross-file enforcement "would be a new mechanism rather than an amendment, and
is filed as a follow-up instead of being claimed here". This module is the
mechanism; ``test_module_verify_budgets.py::
test_the_budget_family_derives_every_floor_from_one_canonical_expression`` is
the guard that uses it.

THE NO-CROSS-IMPORT CONVENTION IS UNBROKEN, and it is worth stating precisely
because the deleted docstrings stated it too broadly. The convention is that a
test file must not import a SIBLING TEST FILE: that couples two guards which
have to be able to fail independently, and it lets a regression in one silence
the other. A shared NON-TEST helper module does not do that. Nothing here
imports a guard, no guard imports another guard, and every guard still fails on
its own — what changed is only that they can no longer each spell the derivation
differently. ``test_module_verify_budgets.py``'s header used to generalise the
convention to "HELPERS ARE COPIED, NOT IMPORTED, deliberately", which is a
strictly stronger claim than its own argument supports and which this module
falsifies; that header is corrected in place rather than left standing.

SIBLING-HELPER PRECEDENT: ``setup_host_sections.py``, ``setup_host_parsing.py``
and ``systemd_unit_invariants.py`` already live here and are imported by bare
name from the tests beside them. That resolves because ``tests/scripts/
conftest.py`` inserts THIS directory onto ``sys.path`` — pytest's
``--import-mode=importlib`` (set in the repo-root ``addopts``) deliberately does
NOT do that for you, and without the conftest line the failure surfaces at
COLLECTION as a bare ModuleNotFoundError rather than as anything resembling the
invariant under test.

Cited by SYMBOL, never by file:line — both publishers already record that every
line pin they once carried had rotted onto unrelated code, and a stale pin reads
as authoritative.
"""
from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

THIS_DIR = pathlib.Path(__file__).parent

# The family members that PUBLISH a (measurement, derived floor) pair for a
# module config prefix. Kept here rather than in the reading guard so that the
# guard cannot quietly narrow the set it checks: shrinking coverage then means
# editing the shared module every publisher already depends on.
FAMILY_PUBLISHER_PATHS: tuple[pathlib.Path, ...] = (
    THIS_DIR / 'test_tests_scripts_module_config.py',
    THIS_DIR / 'test_scripts_module_config.py',
)

# The family member that READS those pairs and enforces the contract across all
# of them. Recorded for the same reason the publishers are: a reader that is
# renamed or deleted takes the only cross-file enforcement with it, and naming
# it here is what makes that visible from the shared module rather than only
# from the file that disappeared.
#
# CHECKED, NOT MERELY DECLARED (task 4320 amendment). This constant is asserted
# against by the reader itself — ``test_the_budget_family_derives_every_floor_
# from_one_canonical_expression``'s assertion (1) requires it to name a real
# file AND to resolve to that guard's own ``__file__``. It carried the claim
# above while nothing read it at all, so a renamed or deleted reader would have
# left it pointing at nothing, silently: the same stale-pointer failure mode
# assertion (7) exists to close for ``MEASURED_BY_SIBLING_GUARD``, and an
# unused constant making a strong claim is precisely the authoritative-but-
# untrue text this family removes rather than writes.
FAMILY_READER_PATH = THIS_DIR / 'test_module_verify_budgets.py'

# The module-level names a publisher must bind. Spelled once, as constants, so
# the reader below and any future publisher agree on them by construction.
PREFIX_NAME = 'MODULE_PREFIX'
WORST_NAME = 'MEASURED_SUITE_WORST_SECS'
FLOOR_NAME = 'MIN_MODULE_BUDGET_SECS'

# THIS module and THE name a publisher must import from it. Spelled as
# constants for the same reason as the three above: the shadow check below
# compares a publisher's import against them, so a rename of this module or of
# its one exported function cannot leave the check silently looking for a name
# nothing binds any more.
HELPER_MODULE_NAME = 'module_budget_family'
HELPER_NAME = 'min_budget'


def min_budget(worst: float) -> int:
    """~2x the worst measured run, rounded DOWN to the nearest 100s.

    THE canonical derivation for this family. Every member's
    ``MIN_MODULE_BUDGET_SECS`` must be ``min_budget(MEASURED_SUITE_WORST_SECS)``
    — not a re-spelling that happens to agree, and not a literal.

    ORIGIN: the expression is task 3458's, written for
    ``test_scripts_module_config.MIN_MODULE_BUDGET_SECS``. The copies this
    module replaced each recorded that they reused it "verbatim so the guards
    cannot silently drift in SHAPE" — the right goal, pursued by the one method
    that cannot achieve it, since verbatim reuse by copying is exactly what
    nothing can verify stayed verbatim. The goal is unchanged; it is now met by
    there being one expression rather than three agreeing ones.

    DERIVED FROM THE MEASUREMENT RATHER THAN HAND-SET BESIDE IT, because that
    exact pair has already rotted once undetected. ``test_tests_scripts_module_
    config.py`` held a HAND-SET ``MIN_MODULE_BUDGET_SECS = 300`` against a
    ``MEASURED_SUITE_WORST_SECS = 127.0`` figure while
    ``tests/scripts/orchestrator.yaml`` had since recorded a 233.50s worst run
    of the VERBATIM command that module declares — so the yaml's own worst
    figure was gated by NOTHING. Both sibling guards had NAMED the staleness in
    prose, which is exactly what a comment can do and an assertion could not; a
    REVIEWER caught it, which is not a mechanism. Task 3703 refreshed the
    constant and made that one floor derived. The history is kept rather than
    deleted because it is the REASON for the derivation: a reader who finds only
    the rule learns nothing about why a hand-set floor is not an acceptable
    substitute.

    ONE IMPLEMENTATION IS THE POINT (task 4320). Task 3703 fixed the pair in ONE
    file, and nothing enforced the same property for the rest of the family —
    the two remaining members went on spelling the derivation for themselves,
    one as a copied ``def _min_budget`` and one inlined. The guard that consumes
    this function evaluates each publisher's floor expression in a namespace
    holding only ``min_budget`` and the published worst, WITHOUT
    ``__builtins__``, so a locally re-spelled derivation raises ``NameError``
    there even when it computes the right number today. That is what promotes
    "the family shares one expression" from a convention to a property.

    TWO SUB-CLAIMS OF THE MOVED TEXT ARE DELIBERATELY NOT CARRIED OVER, because
    they were already false when task 4320 moved it. ``test_module_verify_
    budgets._min_budget``'s docstring said the tests/scripts sibling "now holds
    ``MEASURED_SUITE_WORST_SECS = 233.50``" (task 4320 re-measured it to a
    larger figure, and quoting any figure here would recreate the lockstep copy
    ``MEASURED_BY_SIBLING_GUARD`` exists to refuse), and that its derivation
    test "pins the shape with ``_min_budget(930.59) == 1800`` against the
    sibling's published pair" (that test switched to SYNTHETIC cases during task
    3703's own amendment pass, for exactly the same second-copy reason). Both
    are recorded as corrected rather than silently dropped: a claim that reads
    as authoritative and is not true is the defect this family exists to remove,
    and text does not stop being subject to that rule by being relocated.

    ROUNDS DOWN, AND DEGENERATES TO ZERO for cheap suites: ``min_budget(22.49)
    == 0``, the sampler case ``test_module_verify_budgets.py`` records, where a
    ``budget >= floor`` assertion becomes VACUOUSLY true for any declared value.
    Publishers whose suites sit above that regime pin it explicitly rather than
    assuming it — see ``test_min_module_budget_is_derived_from_the_measured_
    worst_run``'s non-degeneracy assertion.
    """
    return (int(2 * worst) // 100) * 100


class PublishedPair(NamedTuple):
    """One family member's published (measurement, derived floor) pair, read from SOURCE.

    Read by parsing rather than by importing the publisher. Importing would
    hand back only the already-evaluated ``MIN_MODULE_BUDGET_SECS`` int, which
    is exactly what cannot distinguish a derivation from a literal that agrees
    with it today — the rot this family exists to catch. The EXPRESSION is the
    artefact under test, so the expression is what is captured.

    ``worst_source`` / ``floor_source`` are the verbatim source segments of the
    two right-hand sides; ``worst`` is the literal value if
    ``MEASURED_SUITE_WORST_SECS`` is a plain numeric literal and ``None``
    otherwise; ``floor`` is the value ``floor_source`` evaluates to against the
    canonical namespace, and ``error`` says why when it does not. Both a value
    and its failure reason are carried rather than raising, so the consuming
    guard can report every publisher's verdict in one run instead of stopping at
    the first.
    """

    prefix: str
    path: pathlib.Path
    worst: float | None
    worst_source: str | None
    floor_source: str | None
    floor: int | None
    error: str | None


def _module_level_bindings(tree: ast.Module) -> dict[str, ast.expr]:
    """Map each module-level assignment target name to the expression assigned to it.

    Handles both ``ast.Assign`` (``X = ...``, including chained targets) and
    ``ast.AnnAssign`` (``X: T = ...``): the family already spells its tables
    with annotations (``MEASURED_BY_SIBLING_GUARD: dict[str, str] = {...}``), so
    a publisher annotating its own pair must not silently drop out of the check.

    MODULE LEVEL ONLY — ``tree.body`` rather than ``ast.walk``. A same-named
    assignment inside a function or a class is a different binding, and picking
    one up would let a publisher satisfy the guard with a name that no importer
    of that module can see.
    """
    bindings: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            bindings[node.target.id] = node.value
    return bindings


# Statement types whose nested bodies execute in the ENCLOSING scope. A `def`
# or a `class` is deliberately absent: those introduce a scope of their own, so
# a `min_budget` bound inside one is a different binding and flagging it would
# be a false positive.
_SCOPE_TRANSPARENT = (
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Match,
)


def _module_scope_statements(body: Iterable[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement in *body* that executes in MODULE scope, nested ones included.

    Recurses through ``if`` / ``try`` / ``with`` / ``for`` / ``while`` /
    ``match`` bodies, which do NOT introduce a scope, and stops at ``def`` and
    ``class``, which do.

    DELIBERATELY ASYMMETRIC WITH ``_module_level_bindings``, which reads
    ``tree.body`` only, and the asymmetry is principled rather than an
    oversight. That function answers "what pair does this module
    UNCONDITIONALLY publish", so a conditionally-assigned ``MODULE_PREFIX``
    correctly yields NO pair and fails loudly at the consuming guard's floor-set
    assertion. This one answers "what could rebind ``min_budget`` here", where
    the conservative direction is to FLAG: a rebinding tucked inside a
    module-level ``if`` changes what the name means just as thoroughly as a
    top-level one.
    """
    for node in body:
        yield node
        if isinstance(node, _SCOPE_TRANSPARENT):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    yield from _module_scope_statements([child])
                elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                    yield from _module_scope_statements(child.body)


def _non_canonical_min_budget_binding(tree: ast.Module) -> str | None:
    """Describe how *tree* binds ``min_budget`` non-canonically, or ``None`` if it does not.

    THE HOLE THIS CLOSES (task 4320 amendment). ``published_pairs`` evaluates a
    publisher's floor expression in a SYNTHETIC namespace holding the CANONICAL
    ``min_budget``, not in that publisher's real module namespace. So a
    publisher that writes ``MIN_MODULE_BUDGET_SECS = min_budget(
    MEASURED_SUITE_WORST_SECS)`` while binding ``min_budget`` to something else
    — a local ``def min_budget``, or a re-assignment after the import — passed
    every check: the reference is there, it evaluates cleanly against the
    canonical helper, and the canonical result equals the canonical result.
    Meanwhile that module's OWN ``MIN_MODULE_BUDGET_SECS`` was whatever the
    shadow returned. The namespace check makes re-spelling the derivation under
    a DIFFERENT name impossible; without this one, re-spelling it under the
    SAME name was free.

    CANONICAL means ``from module_budget_family import min_budget`` — the
    module and the name spelled exactly, with no ``as`` clause (importing
    something else AS ``min_budget`` binds the name to a different object).
    Repeating that import is harmless and is not flagged; anything else that
    binds the name in module scope is, wherever it sits relative to the floor
    assignment. A rebinding BEFORE the assignment changes the published value
    outright, and one AFTER it leaves the name meaning two different things in
    one file, which is the ambiguity this family exists to remove.

    Returns a human-readable description of the FIRST offending binding — line
    numbers computed from the AST at read time, never transcribed, so they
    cannot rot the way a hard-coded file:line pin does — or ``None`` when every
    module-scope binding of the name is the canonical import.

    NOT A SANDBOX, and does not claim to be. This reads source structure, so it
    answers what the file says rather than what an adversary could contrive
    (``globals()`` mutation, an ``exec``, a rebinding inside a function that
    module-level code then calls). The property it establishes is that the
    family cannot DRIFT into several derivations by ordinary editing, which is
    the failure this guard family has actually observed twice.
    """
    imported = False
    for node in _module_scope_statements(tree.body):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != HELPER_NAME:
                    continue
                if node.module == HELPER_MODULE_NAME and alias.asname is None:
                    imported = True
                    continue
                return (
                    f'`from {node.module or "."} import '
                    f'{alias.name}{" as " + alias.asname if alias.asname else ""}` '
                    f'at line {node.lineno}'
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split('.')[0]) == HELPER_NAME:
                    return f'`import {alias.name}` at line {node.lineno}'
        elif (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == HELPER_NAME
        ):
            return (
                f'a module-level `{"class" if isinstance(node, ast.ClassDef) else "def"} '
                f'{HELPER_NAME}` at line {node.lineno}'
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == HELPER_NAME:
                    return f'a module-level assignment at line {node.lineno}'
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == HELPER_NAME
        ):
            return f'a module-level annotated assignment at line {node.lineno}'
    if not imported:
        return (
            f'no module-level `from {HELPER_MODULE_NAME} import {HELPER_NAME}` '
            f'at all, so the name its floor expression calls is bound by nothing '
            f'this reader can see'
        )
    return None


def published_pairs(
    paths: Iterable[pathlib.Path] = FAMILY_PUBLISHER_PATHS,
) -> dict[str, PublishedPair]:
    """Every family publisher's (measurement, floor) pair, keyed by module config prefix.

    ``ast.parse``s each path in *paths*, picks the module-level
    ``MODULE_PREFIX`` / ``MEASURED_SUITE_WORST_SECS`` /
    ``MIN_MODULE_BUDGET_SECS`` bindings, and EVALUATES the floor expression
    against the canonical namespace.

    *paths* DEFAULTS TO ``FAMILY_PUBLISHER_PATHS`` — the production call passes
    nothing, so the set this reader checks is still owned by this module rather
    than by its caller. It is a parameter only so the mechanism itself can be
    tested against SYNTHETIC publishers: until it was, every one of the
    consuming guard's assertions ran exclusively against two files that already
    comply, and the load-bearing claims below (an inlined expression raises
    ``NameError``, a literal is caught by the reference check, a shadowed
    ``min_budget`` is caught by the binding check) were demonstrated nowhere. An
    assertion that has never been observed to fail is the same gap this family
    calls out for hand-set floors.

    THE NAMESPACE IS THE MECHANISM. ``compile(ast.Expression(floor_node), ...,
    'eval')`` runs with globals ``{'__builtins__': {}}`` and locals holding ONLY
    ``min_budget`` and ``MEASURED_SUITE_WORST_SECS``. Deliberately no ``int``:
    an inlined ``(int(2 * W) // 100) * 100`` — the spelling
    ``test_scripts_module_config.py`` carried before task 4320 — raises
    ``NameError: name 'int' is not defined`` rather than quietly reproducing the
    right number, and a copied ``_min_budget(W)`` raises ``NameError: name
    '_min_budget' is not defined``. A publisher cannot satisfy this by
    re-implementing the derivation correctly; it has to CALL the canonical one.

    THE NAMESPACE IS ONLY HALF THE MECHANISM, and saying otherwise was the
    over-claim a reviewer caught (task 4320 amendment). Evaluating against the
    canonical helper in a synthetic namespace says nothing about what
    ``min_budget`` is bound to in the PUBLISHER's namespace: a local ``def
    min_budget`` — or a re-assignment after the import — satisfied every check
    while that module's own floor was whatever the shadow returned. The
    namespace closes re-spellings under a DIFFERENT name;
    ``_non_canonical_min_budget_binding`` closes re-spellings under the SAME
    name, by requiring the name to come from ``from module_budget_family import
    min_budget`` and to be rebound nowhere in module scope. Neither half
    subsumes the other, and the guard reports both through ``error``.

    A LITERAL FLOOR IS NOT CAUGHT HERE, and that is not an oversight: ``1800``
    evaluates fine in any namespace. It is caught by the consuming guard's
    separate check that the floor expression REFERENCES
    ``MEASURED_SUITE_WORST_SECS`` by name — the two checks close different
    halves, and neither subsumes the other.

    A publisher that is missing, or that no longer binds a module-level
    ``MODULE_PREFIX``, yields NO entry rather than a placeholder. The consuming
    guard asserts the returned set of prefixes is exactly the family's, so a
    dropped publisher fails loudly there instead of silently shrinking the set
    every other assertion iterates over.
    """
    pairs: dict[str, PublishedPair] = {}

    for path in paths:
        if not path.is_file():
            continue
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        bindings = _module_level_bindings(tree)

        prefix_node = bindings.get(PREFIX_NAME)
        if not isinstance(prefix_node, ast.Constant) or not isinstance(prefix_node.value, str):
            continue
        prefix = prefix_node.value

        worst_node = bindings.get(WORST_NAME)
        worst_source = (
            None if worst_node is None else ast.get_source_segment(source, worst_node)
        )
        worst: float | None = None
        if worst_node is not None:
            try:
                literal: Any = ast.literal_eval(worst_node)
            except (ValueError, SyntaxError, TypeError):
                literal = None
            if isinstance(literal, (int, float)) and not isinstance(literal, bool):
                worst = float(literal)

        floor_node = bindings.get(FLOOR_NAME)
        floor_source = (
            None if floor_node is None else ast.get_source_segment(source, floor_node)
        )

        floor: int | None = None
        error: str | None = None
        if floor_node is None:
            error = f'{path.name} binds no module-level {FLOOR_NAME}'
        elif worst is None:
            error = (
                f'{path.name} does not bind {WORST_NAME} to a plain numeric literal, so '
                f'its {FLOOR_NAME} expression cannot be evaluated against a measurement'
            )
        else:
            namespace: dict[str, Any] = {'min_budget': min_budget, WORST_NAME: worst}
            try:
                code = compile(ast.Expression(body=floor_node), str(path), 'eval')
                # eval of REPO SOURCE ALREADY ON DISK, in a namespace with no
                # __builtins__ and two bindings. Nothing here reads input the
                # test run did not already import; the restriction is the
                # mechanism, not a sandbox.
                value: Any = eval(code, {'__builtins__': {}}, namespace)
            # BARE `Exception` ON PURPOSE: the exception TYPE and MESSAGE are the
            # finding. NameError says the derivation was re-spelled locally,
            # TypeError says it was called wrongly, and both belong in the
            # guard's failure text rather than crashing the reader.
            except Exception as exc:
                error = (
                    f'floor uses a non-canonical derivation '
                    f'({type(exc).__name__}: {exc})'
                )
            else:
                if isinstance(value, int) and not isinstance(value, bool):
                    floor = value
                else:
                    error = (
                        f'{FLOOR_NAME} evaluated to {value!r} '
                        f'({type(value).__name__}), not an int'
                    )

        # THE SHADOW CHECK, applied only when the floor expression actually
        # CALLS the canonical name — a floor that never mentions `min_budget`
        # is already rejected by the eval above or by the consuming guard's
        # reference check, and flagging its import hygiene would report the
        # wrong defect. Checked AFTER the eval so a more specific failure (a
        # NameError naming the re-spelling) wins the `error` slot; this one
        # fires exactly when the expression looks canonical and the binding is
        # not, which is the case nothing else can see.
        if (
            error is None
            and floor_node is not None
            and any(
                isinstance(node, ast.Name) and node.id == HELPER_NAME
                for node in ast.walk(floor_node)
            )
        ):
            rebinding = _non_canonical_min_budget_binding(tree)
            if rebinding is not None:
                floor = None
                error = (
                    f'floor calls `{HELPER_NAME}`, but {path.name} binds that '
                    f'name via {rebinding} — so the expression evaluated here '
                    f'against the canonical derivation is NOT the one that runs '
                    f'in that module'
                )

        pairs[prefix] = PublishedPair(
            prefix=prefix,
            path=path,
            worst=worst,
            worst_source=worst_source,
            floor_source=floor_source,
            floor=floor,
            error=error,
        )

    return pairs
