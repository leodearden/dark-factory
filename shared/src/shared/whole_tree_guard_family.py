"""The whole-tree guard family: its per-test ceiling and its membership predicate.

A test module is a MEMBER iff its module-level ``pytestmark`` carries
``pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)``, bare or dotted.  Task-role
verify runs the touched tests plus this family
(``orchestrator/src/orchestrator/verify_plan.py::_whole_tree_guard_family``), so
the one mark both sizes a tree-sweeping test's ceiling and enrols it in the run.

Pure stdlib plus ``shared.pytest_mark_grammar``, which owns the reading of
``pytestmark`` syntax.  Nothing here raises.
"""

from __future__ import annotations

import ast
import math

from shared.pytest_mark_grammar import marker_name, pytestmark_elements

__all__ = [
    'WHOLE_TREE_SCAN_TEST_TIMEOUT',
    'WHOLE_TREE_SCAN_TIMEOUT_NAME',
    'is_whole_tree_guard_ast',
    'is_whole_tree_guard_module',
    'min_leg_budget_for_family',
    'module_level_timeout_arguments',
]

# task 4215: per-test ceiling for the family of guard tests that sweep the
# WHOLE tree -- `rglob('*.py')` over ~500 files, `ast.parse` on each -- and so
# cannot be sized by the ordinary 60s default.  This comment is the SINGLE
# home of that rationale: the modules that carry the mark, in every package's
# tests since task 8402, point HERE instead of repeating it, so switching
# `timeout_method`, retuning the ceiling, or revising the measurements is one
# edit rather than one per member.
#
# WHAT THE ORCHESTRATOR FAMILY IS (task 4600): the consumers of the one shared
# tree pass (orchestrator/tests/_tree_scan.py::TreePass, served by
# orchestrator/tests/conftest.py's session `tree_pass` fixture), plus the one
# permanent private sweeper and test_merge_lane_ratchet.py, both named with
# their reasons in orchestrator/tests/test_whole_tree_scan_timeout_guard.py.
# For a consumer, the per-item cost this ceiling must clear is no longer a
# sweep of its own but the SHARED PASS: one streaming read of the union of
# every requested analysis's file set, each file parsed at most once and
# walked only by the analyses whose `ParsedModule.may_name` prefilter cannot
# rule it out, billed to whichever consumer item requests `tree_pass` first on
# its worker, while every later consumer item projects facts already held.
# Any consumer may be the item billed, which is why each one carries this
# mark however cheap its own tests are.
#
# It is the pyproject's own sanctioned escape hatch -- "Slow tests opt out
# with `@pytest.mark.timeout(N)`", the comment on
# `[tool.pytest.ini_options].timeout` in orchestrator/pyproject.toml -- applied
# at MODULE level, which is the only form that is a sound LOWER bound on every
# collected item (a per-test decorator leaves any test added later at the
# default).  A tighter decorator still WINS where one is present, so the
# module mark narrows nothing.  Coverage of the family is ENFORCED, not
# sprinkled: orchestrator/tests/test_whole_tree_scan_timeout_guard.py
# recomputes the census (private sweepers and shared-pass consumers) from
# source on every run and fails a member that
# lacks the mark OR pins it below this ceiling, and
# shared/src/shared/testing_repo_wide_tests.py::judge_repo_wide_family fails
# every unmarked repo-wide test module in the other packages' tests.
#
# MECHANISM -- what a breach here costs:
#   * until task 6354, `[tool.pytest.ini_options].timeout_method = "thread"`
#     meant pytest-timeout answered a breach by `os._exit()`ing the whole xdist
#     worker ("node down: Not properly terminated"), NOT by failing the
#     offending test, and `--max-worker-restart=0` (in
#     `[tool.pytest.ini_options].addopts`, task 1907) then declined to replace
#     that worker, degrading the run to a TRUNCATED whole-suite session whose
#     surviving failure named some innocent guard that merely shared the dead
#     worker.  One slow tree-scan cost a whole verify run AND misattributed the
#     blame; the measured history below is from that era;
#   * since task 6354 the method is `signal`: a breach is one false red on the
#     scanner itself, and the worker lives on.  The ceiling still prevents that
#     false red.  A breach inside a C or Rust call (complexipy's analysis)
#     surfaces only once that call returns.
#
# MEASURED history of the family:
#   * unloaded and serial (`-n0`) on a 32-core box: 8.25s/call
#     (test_merge_queue_reachback_patch_guard), 6.70s
#     (test_event_loop_antipattern_guard), 6.46s
#     (the serial-worker import guard, deleted with its fixture by task 5034);
#   * the SAME serial-worker guard measured 17.85 / 21.32 / 30.75s per
#     call at loadavg 120-176 under `-n auto` -- ~4.8x load inflation;
#   * xdist worker deaths were then observed at loadavg 250-423, one further
#     inflation step past the 60s default THEN IN FORCE (esc-3980-1 on branch
#     task/3980, esc-3787-1 on branch task/3787).
#   THREE members crashed that way -- test_event_loop_antipattern_guard.py,
#   test_merge_queue_reachback_patch_guard.py and the serial-worker import
#   guard -- which is what makes this a
#   FAMILY defect rather than three accidents, and why the rest were marked
#   preemptively.  Marking is not free: since task 7253 a member joins every
#   task-role run of its leg, and a hang in it is caught only at this ceiling,
#   not at the ini default (docs/task-role-pytest-floor.md).
# The current anchor is
# orchestrator/tests/test_whole_tree_scan_timeout_guard.py::_MEASURED_UNDER_LOAD_WORST_CASE,
# which floors the family at
# orchestrator/tests/test_whole_tree_scan_timeout_guard.py::_ABSOLUTE_FLOOR_SECONDS.
# Derivation: plans/pytest-per-test-timeout-measurement-2026-09-17.md.
#
# WAS `5 * PYPROJECT_DEFAULT_TIMEOUT` until 2026-09-12, when that ini default
# was raised 60 -> 300 to stop CPU starvation on a loaded host false-redding a
# shifting victim (shared/pyproject.toml carries the rationale).  The multiple
# existed to TRACK the hazard: when the default was 60 it was the binding
# constraint on this family, so deriving from it kept the ceiling clear of it
# automatically.  The default has now overtaken the family's own measured
# requirement, and carrying the multiple forward would have set this to 1500s
# -- a 5x widening of the ceiling for the ~13 marked modules that no
# measurement asks for, and one that would silently let a genuinely hung tree
# scan burn 25 minutes.  So the two knobs are now what they always were
# SEMANTICALLY -- orthogonal, one sized by host contention and one by the cost
# of an AST sweep -- and this one is floored at the figure its own measurements
# justify.  orchestrator/tests/test_whole_tree_scan_timeout_guard.py::_ABSOLUTE_FLOOR_SECONDS
# encodes that independence; it has since been re-anchored, but its role is
# unchanged.
#
# It must clear BOTH the floor and
# orchestrator/tests/_orch_helpers.py::PYPROJECT_DEFAULT_TIMEOUT (pinned by
# orchestrator/tests/test_whole_tree_scan_timeout_guard.py); which of the two
# binds is value history, recorded in
# plans/pytest-per-test-timeout-measurement-2026-09-17.md.
#
# Deliberately NOT taken from
# orchestrator/tests/_merge_worker_drive.py::HEAVY_BARRIER_TEST_TIMEOUT: that
# one bounds a different hazard, sized by merge-barrier waits, and an AST sweep
# performs zero merge waits, so borrowing it would let a future merge-timing
# retune silently move this ceiling.  Never-narrow: it must also
# never fall below PYPROJECT_DEFAULT_TIMEOUT, or a module-level mark meant as
# a FLOOR would start narrowing its module below the global default (pinned by
# orchestrator/tests/test_whole_tree_scan_timeout_guard.py).
#
# PER-LEG BUDGET: a per-test ceiling is a backstop only inside a verify leg
# whose whole-run budget outlasts it; below that, verify kills the leg first
# and an attributable hang becomes a retried, unattributed infra_timeout.  So
# every registered verify leg whose tree holds a member must declare
# `verify_command_timeout_secs >= min_leg_budget_for_family(W)`, W being that
# leg's recorded worst green run, gated by
# tests/scripts/test_module_verify_budgets.py::test_every_leg_running_the_family_outlasts_its_ceiling.
# Raising this ceiling is therefore also a budget change for those legs.
WHOLE_TREE_SCAN_TEST_TIMEOUT = 1200

#: The spelling that admits a module to the family.  Membership is decided by
#: this NAME, never by the mark's value: a literal equal to the ceiling today
#: would silently part from it on the next retune, while every module spelled
#: through the name moves with it.
WHOLE_TREE_SCAN_TIMEOUT_NAME = 'WHOLE_TREE_SCAN_TEST_TIMEOUT'


def min_leg_budget_for_family(worst_green_run_secs: float) -> int:
    """The smallest whole-100s leg budget at which a hang starting when the leg's slowest green run ends still reaches the ceiling."""
    return math.ceil((worst_green_run_secs + WHOLE_TREE_SCAN_TEST_TIMEOUT) / 100) * 100


def _parse(source: str | None) -> ast.Module | None:
    if not source:
        return None
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _timeout_arguments(tree: ast.Module) -> tuple[ast.expr, ...]:
    arguments: list[ast.expr] = []
    for element in pytestmark_elements(tree.body):
        if not isinstance(element, ast.Call) or marker_name(element) != 'timeout':
            continue
        if element.args:
            arguments.append(element.args[0])
            continue
        keyword_value = next(
            (keyword.value for keyword in element.keywords if keyword.arg == 'timeout'), None,
        )
        if keyword_value is not None:
            arguments.append(keyword_value)
    return tuple(arguments)


def module_level_timeout_arguments(source: str | None) -> tuple[ast.expr, ...]:
    """The seconds expression of every ``pytest.mark.timeout(...)`` in *source*'s module ``pytestmark``.

    Read from the LAST module-level binding only (``tree.body``, never a class or
    function body), in element order, unevaluated.  Both spellings pytest-timeout
    accepts are read -- positional ``timeout(N)`` and keyword ``timeout(timeout=N)``;
    a call with neither pins nothing and is skipped.  ``None``, a ``SyntaxError``
    or a ``ValueError`` yields ``()``.
    """
    tree = _parse(source)
    return () if tree is None else _timeout_arguments(tree)


def is_whole_tree_guard_ast(tree: ast.Module) -> bool:
    """True iff the parsed module *tree*'s module-level timeout mark is spelled via the family name.

    The bare name and a dotted ``<module>.WHOLE_TREE_SCAN_TEST_TIMEOUT`` both
    count; a numeric literal never does, whatever its value.
    """
    for argument in _timeout_arguments(tree):
        if isinstance(argument, ast.Name) and argument.id == WHOLE_TREE_SCAN_TIMEOUT_NAME:
            return True
        if isinstance(argument, ast.Attribute) and argument.attr == WHOLE_TREE_SCAN_TIMEOUT_NAME:
            return True
    return False


def is_whole_tree_guard_module(source: str | None) -> bool:
    """:func:`is_whole_tree_guard_ast` over module *source*; unparseable or absent source is no member."""
    if not source or WHOLE_TREE_SCAN_TIMEOUT_NAME not in source:
        return False
    tree = _parse(source)
    return tree is not None and is_whole_tree_guard_ast(tree)
