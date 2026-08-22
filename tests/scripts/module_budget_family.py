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
FAMILY_READER_PATH = THIS_DIR / 'test_module_verify_budgets.py'

# The module-level names a publisher must bind. Spelled once, as constants, so
# the reader below and any future publisher agree on them by construction.
PREFIX_NAME = 'MODULE_PREFIX'
WORST_NAME = 'MEASURED_SUITE_WORST_SECS'
FLOOR_NAME = 'MIN_MODULE_BUDGET_SECS'


def min_budget(worst: float) -> int:
    """~2x the worst measured run, rounded DOWN to the nearest 100s.

    THE canonical derivation for this family. Every member's
    ``MIN_MODULE_BUDGET_SECS`` must be ``min_budget(MEASURED_SUITE_WORST_SECS)``
    — not a re-spelling that happens to agree, and not a literal.

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


def published_pairs() -> dict[str, PublishedPair]:
    """Every family publisher's (measurement, floor) pair, keyed by module config prefix.

    ``ast.parse``s each path in ``FAMILY_PUBLISHER_PATHS``, picks the
    module-level ``MODULE_PREFIX`` / ``MEASURED_SUITE_WORST_SECS`` /
    ``MIN_MODULE_BUDGET_SECS`` bindings, and EVALUATES the floor expression
    against the canonical namespace.

    THE NAMESPACE IS THE MECHANISM. ``compile(ast.Expression(floor_node), ...,
    'eval')`` runs with globals ``{'__builtins__': {}}`` and locals holding ONLY
    ``min_budget`` and ``MEASURED_SUITE_WORST_SECS``. Deliberately no ``int``:
    an inlined ``(int(2 * W) // 100) * 100`` — the spelling
    ``test_scripts_module_config.py`` carried before task 4320 — raises
    ``NameError: name 'int' is not defined`` rather than quietly reproducing the
    right number, and a copied ``_min_budget(W)`` raises ``NameError: name
    '_min_budget' is not defined``. A publisher cannot satisfy this by
    re-implementing the derivation correctly; it has to CALL the canonical one.

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

    for path in FAMILY_PUBLISHER_PATHS:
        if not path.is_file():
            continue
        source = path.read_text(encoding='utf-8')
        bindings = _module_level_bindings(ast.parse(source, filename=str(path)))

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
