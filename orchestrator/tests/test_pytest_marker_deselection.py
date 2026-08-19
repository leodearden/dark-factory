"""Static detection of a fully marker-DESELECTED file-scoped pytest run (task 3494).

The defect (escalation esc-3292-1): ``verify_plan.classify_file`` is purely
PATH-based — a ``test_*.py`` basename is COLLECTABLE_TEST — so a diff touching
only ``orchestrator/tests/test_warm_lane_bash_suite.py`` produces a FILE_SCOPED
pytest run targeting exactly that file.  But every item in that file carries a
module-level ``pytest.mark.warm_lane_bash``, and the owning module's own
``addopts`` say ``-m 'not warm_lane_bash'``.  The run therefore collects ZERO
items, pytest exits rc=5, and ``verify_classify._classify_opaque`` classifies
rc=5 as RED.  A false RED on a diff that touched a real, passing test file.

This is the SECOND instance of the task-1852 class ("the path says collectable,
pytest collects zero"), and it is fixed the same way the first one was: in the
SCOPING layer, by widening to the owning module's FULL_SUITE — never by
softening rc=5 in the classification layer (``verify_classify.py`` carries that
invariant verbatim and is untouched by this task).

BOTH ARMS (task 3513).  Task 3494 closed only ``_derive_module_runs`` arm 4.
The twin in ``_derive_fallback_runs`` — the branch that fires when a project
registers NO module_configs — has the identical failure mode, and dark-factory's
own root ``pyproject.toml`` carries ``-m 'not smoke'``, so the ingredient is
present; only this repo's registered module_configs keep the branch unreachable
here.  It is reachable in other projects dark-factory targets.

The two arms share ONE probe (``verify_plan.deselecting_expression_for_command``)
so they can never disagree about which commands are refused or where the ini
file is looked for.  The fallback arm needs one extra layer: that branch hands
``run_verification`` the ModuleConfig rather than the plan, so the widening is
applied to the ALREADY-EXECUTED config
(``verify_plan.widen_fallback_for_marker_deselection``) and the plan record is
reconciled to match (``verify._executed_fallback_plan``'s ``pytest_reason``).
Taking the executed config is also what makes the over-fire task 3494's
docstring feared structurally impossible: by then any subproject rescoping has
already happened, so the command's own shape decides.

This module unit-tests the pure static detector (``orchestrator.pytest_markers``)
against synthetic strings, then pins the wired ``derive_verify_plan`` behaviour —
including the real-config incident golden — then the shared probe, the fallback
widener's refusal and positive halves, the plan-record reconciliation, and
finally the end-to-end pin through ``run_scoped_verification``'s fallback branch.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from test_verify import _real_worktree_reader
from test_verify_plan import DATA_MODULE_DIFF, ROOT_CONFTEST_DIFF, SOURCE_ONLY_DIFF
from test_verify_scope_kappa import _executed_module_configs, _run_verification_spy

from orchestrator import verify, verify_plan
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.pytest_markers import (
    deselecting_expression_for_targets,
    expression_definitely_deselects,
    module_level_marker_names,
    per_item_marker_names,
    resolve_marker_expression,
)
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_cmd import parse_config_command
from orchestrator.verify_plan import (
    ScopeKind,
    VerifyPlan,
    derive_verify_plan,
    deselecting_expression_for_command,
    widen_fallback_for_marker_deselection,
)

# ---------------------------------------------------------------------------
# resolve_marker_expression (step-1: RED)
# ---------------------------------------------------------------------------

#: The real orchestrator addopts shape, verbatim from orchestrator/pyproject.toml:125.
#: Empirically validated: tomllib -> str -> shlex.split -> [..., '-m', 'not warm_lane_bash'].
_REAL_ADDOPTS = "-n auto --dist loadgroup --max-worker-restart=0 -m 'not warm_lane_bash'"


def _pyproject(addopts_literal: str) -> str:
    """A minimal ``pyproject.toml`` whose ini_options carry *addopts_literal* verbatim."""
    return f'[tool.pytest.ini_options]\naddopts = {addopts_literal}\n'


class TestResolveMarkerExpression:
    """``resolve_marker_expression(pyproject_text, test_command) -> str | None``.

    Resolves the effective ``-m`` marker expression for a module: the
    ``[tool.pytest.ini_options].addopts`` expression, overridden by a CLI ``-m``
    in the module's ``test_command`` (pytest's documented last-wins rule).
    Every failure path returns None; the function never raises.
    """

    # -- addopts as a string --------------------------------------------------

    def test_real_addopts_string_yields_the_warm_lane_expression(self):
        """The incident fixture's own addopts, verbatim."""
        assert resolve_marker_expression(_pyproject(f'"{_REAL_ADDOPTS}"'), None) == (
            'not warm_lane_bash'
        )

    def test_addopts_as_toml_list(self):
        text = '[tool.pytest.ini_options]\naddopts = ["-n", "auto", "-m", "not integration"]\n'
        assert resolve_marker_expression(text, None) == 'not integration'

    def test_attached_form_dash_m_expr(self):
        """``-mnot slow`` — the attached spelling pytest also accepts.

        Spelled so the attached form survives as ONE token: a ``str`` addopts is
        shlex-split (pytest splits it the same way), so a bare ``-mnot slow``
        would legitimately become ``['-mnot', 'slow']`` — argparse would then
        read the expression as ``'not'`` and treat ``slow`` as a positional.
        The inner quotes are what make it a single attached argument.
        """
        assert resolve_marker_expression(_pyproject('"-m\'not slow\'"'), None) == 'not slow'

    def test_attached_form_in_a_toml_list_is_one_token(self):
        """The list form needs no quoting — each element is already one token."""
        text = '[tool.pytest.ini_options]\naddopts = ["-n", "auto", "-mnot slow"]\n'
        assert resolve_marker_expression(text, None) == 'not slow'

    def test_later_dash_m_in_addopts_wins(self):
        text = _pyproject('"-m \'not slow\' -q -m \'not integration\'"')
        assert resolve_marker_expression(text, None) == 'not integration'

    # -- fail-safe: everything unparseable/absent resolves to None ------------

    def test_addopts_without_dash_m_is_none(self):
        assert resolve_marker_expression(_pyproject('"-n auto -q"'), None) is None

    def test_missing_ini_options_table_is_none(self):
        assert resolve_marker_expression('[tool.ruff]\nline-length = 100\n', None) is None

    def test_missing_addopts_key_is_none(self):
        text = '[tool.pytest.ini_options]\ntimeout = 300\n'
        assert resolve_marker_expression(text, None) is None

    def test_addopts_of_wrong_toml_type_is_none(self):
        assert resolve_marker_expression(_pyproject('42'), None) is None

    def test_malformed_toml_is_none_not_a_raise(self):
        assert resolve_marker_expression('[tool.pytest.ini_options\naddopts = ', None) is None

    def test_none_pyproject_text_is_none(self):
        assert resolve_marker_expression(None, None) is None

    # -- test_command last-wins ----------------------------------------------

    def test_cli_dash_m_overrides_the_addopts_expression(self):
        """A CLI ``-m`` overrides the addopts ``-m``, last wins (pytest's documented rule)."""
        resolved = resolve_marker_expression(
            _pyproject(f'"{_REAL_ADDOPTS}"'), 'uv run pytest tests/ -m warm_lane_bash',
        )
        assert resolved == 'warm_lane_bash'

    def test_test_command_without_dash_m_leaves_addopts_intact(self):
        resolved = resolve_marker_expression(
            _pyproject(f'"{_REAL_ADDOPTS}"'), 'uv run pytest tests/ --tb=short -q',
        )
        assert resolved == 'not warm_lane_bash'

    # -- the `python -m pytest` guard ----------------------------------------

    def test_python_dash_m_pytest_is_not_read_as_a_marker_expression(self):
        """``python -m pytest`` must never resolve to the expression ``'pytest'``.

        Only tokens AFTER the ``pytest`` keyword are scanned for a CLI ``-m``.
        With no pyproject expression to fall back on, the answer is None.
        """
        assert resolve_marker_expression(None, 'python -m pytest tests/') is None

    def test_python_dash_m_pytest_leaves_the_pyproject_expression_unchanged(self):
        resolved = resolve_marker_expression(
            _pyproject(f'"{_REAL_ADDOPTS}"'), 'python -m pytest tests/',
        )
        assert resolved == 'not warm_lane_bash'

    # -- the chained-command guard -------------------------------------------

    def test_a_later_chain_clause_cannot_supply_the_marker_expression(self):
        """``pytest tests/ && python -m mytool`` must not resolve to ``'mytool'``.

        The scan stops at the first chain operator after the ``pytest`` keyword.
        Without that stop, an unrelated trailing clause would OVERRIDE the real
        addopts expression and silently suppress a legitimate widening.
        """
        resolved = resolve_marker_expression(
            _pyproject('"-m \'not slow\'"'), 'pytest tests/ && python -m mytool',
        )
        assert resolved == 'not slow'

    def test_the_first_pytest_clause_wins_not_the_last(self):
        """FIRST occurrence — the same clause ``_scope_prefix_to_keyword`` scopes.

        ``head.find('pytest')`` truncates a chained ``test_command`` to its first
        pytest clause, so the marker probe must read that clause too; reading the
        last would let the two layers describe different invocations.
        """
        assert resolve_marker_expression(None, 'pytest a -m foo && pytest b -m bar') == 'foo'

    def test_command_with_no_pytest_token_leaves_the_expression_unchanged(self):
        resolved = resolve_marker_expression(
            _pyproject(f'"{_REAL_ADDOPTS}"'), 'cargo test -m something',
        )
        assert resolved == 'not warm_lane_bash'

    def test_unsplittable_test_command_leaves_the_expression_unchanged(self):
        """An unbalanced quote makes ``shlex.split`` raise — must never propagate."""
        resolved = resolve_marker_expression(
            _pyproject(f'"{_REAL_ADDOPTS}"'), "uv run pytest tests/ -m 'unbalanced",
        )
        assert resolved == 'not warm_lane_bash'


# ---------------------------------------------------------------------------
# module_level_marker_names (step-3: RED)
# ---------------------------------------------------------------------------

#: The real shape at orchestrator/tests/test_warm_lane_bash_suite.py:231-235,
#: validated by AST against the live file before any assertion here was written.
_REAL_PYTESTMARK_SOURCE = """\
import pytest

PYTEST_TIMEOUT = 960

pytestmark = [
    pytest.mark.warm_lane_bash,
    pytest.mark.timeout(PYTEST_TIMEOUT),
    pytest.mark.xdist_group('warm_lane_bash'),
]
"""


class TestModuleLevelMarkerNames:
    """``module_level_marker_names(source) -> frozenset[str]``.

    Returns a LOWER BOUND on every collected item's marker set: only a
    module-level ``pytestmark`` provably applies to every item in the file, so
    per-function/class decorators are deliberately excluded.  A name absent from
    the result is UNKNOWN, not absent — which is exactly what makes the Kleene
    treatment in ``expression_definitely_deselects`` sound.
    """

    def test_bare_attribute_assignment(self):
        assert module_level_marker_names('pytestmark = pytest.mark.slow') == frozenset({'slow'})

    def test_real_warm_lane_list_shape(self):
        """The incident fixture's own module-level pytestmark."""
        assert module_level_marker_names(_REAL_PYTESTMARK_SOURCE) == frozenset(
            {'warm_lane_bash', 'timeout', 'xdist_group'},
        )

    def test_tuple_form_collects_like_the_list_form(self):
        source = 'import pytest\npytestmark = (pytest.mark.slow, pytest.mark.timeout(5))\n'
        assert module_level_marker_names(source) == frozenset({'slow', 'timeout'})

    def test_annotated_assignment(self):
        source = 'import pytest\npytestmark: list = [pytest.mark.slow]\n'
        assert module_level_marker_names(source) == frozenset({'slow'})

    def test_module_without_pytestmark_is_empty(self):
        assert module_level_marker_names('import pytest\n\n\ndef test_a():\n    pass\n') == (
            frozenset()
        )

    def test_per_function_decorators_are_not_collected(self):
        """THE CONSERVATISM PIN.

        A decorator does not provably cover every collected item — the sibling
        ``test_b`` here is unmarked and would still be selected.  Collecting
        decorator markers would make the detector unsound in the widening
        direction; excluding them only makes it under-fire, which is safe.
        """
        source = (
            'import pytest\n\n\n'
            '@pytest.mark.slow\n'
            'def test_a():\n    pass\n\n\n'
            'def test_b():\n    pass\n'
        )
        assert module_level_marker_names(source) == frozenset()

    def test_pytestmark_inside_a_class_body_is_not_module_level(self):
        source = 'import pytest\n\n\nclass TestThing:\n    pytestmark = pytest.mark.slow\n'
        assert module_level_marker_names(source) == frozenset()

    def test_pytestmark_inside_a_function_body_is_not_module_level(self):
        source = 'import pytest\n\n\ndef configure():\n    pytestmark = pytest.mark.slow\n'
        assert module_level_marker_names(source) == frozenset()

    def test_an_unrelated_module_level_name_is_ignored(self):
        assert module_level_marker_names('import pytest\nothermark = pytest.mark.slow\n') == (
            frozenset()
        )

    def test_a_non_marker_element_does_not_suppress_its_siblings(self):
        source = 'import pytest\nSOME_CONST = 1\npytestmark = [SOME_CONST, pytest.mark.slow]\n'
        assert module_level_marker_names(source) == frozenset({'slow'})

    def test_syntax_error_is_empty_not_a_raise(self):
        assert module_level_marker_names('def broken(:\n') == frozenset()

    def test_none_source_is_empty(self):
        assert module_level_marker_names(None) == frozenset()


# ---------------------------------------------------------------------------
# per_item_marker_names (step-1: RED)
# ---------------------------------------------------------------------------

#: The real shape at tests/scripts/test_pump_web_ui_installed_unit_parity.py:
#: 187-223, condensed — validated by AST against the live file before any
#: assertion here was written.  Both top-level tests carry
#: ``@pytest.mark.integration``; the second also carries a ``skipif``.
_ALL_DECORATED_SOURCE = """\
import shutil

import pytest

SYSTEMCTL_SKIP_REASON = 'systemctl unavailable'


@pytest.mark.integration
def test_installed_unit_file_restart_backoff_effective() -> None:
    pass


@pytest.mark.integration
@pytest.mark.skipif(shutil.which('systemctl') is None, reason=SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_restart_steps_effective() -> None:
    pass
"""

#: The real shape at tests/scripts/test_know_live_installed_unit_parity.py:
#: 122-459, condensed to one representative of each kind found there: a
#: leading private helper (contributes nothing), an integration-decorated
#: test, a parametrize-only test, and a bare undecorated test — validated by
#: AST against the live file before any assertion here was written.
_MIXED_DECORATED_SOURCE = """\
import pytest


def _argv_from_exec_start_show(exec_start_value: str) -> str | None:
    return None


@pytest.mark.integration
def test_installed_unit_file_restart_backoff_effective() -> None:
    pass


@pytest.mark.parametrize('exec_start_value', ['a'])
def test_config_arg_from_exec_start_returns_value_or_none(exec_start_value) -> None:
    pass


def test_argv_from_exec_start_show_extracts_argv_segment() -> None:
    pass
"""


class TestPerItemMarkerNames:
    """``per_item_marker_names(source) -> tuple[frozenset[str], ...] | None``.

    THE SECOND, ADDITIVE PROOF TIER: one guaranteed (lower-bound) marker set
    per top-level test item — ``module_level_marker_names(source)`` unioned
    with that item's own ``pytest.mark.NAME`` decorators — but ONLY when this
    walk can see every item pytest would collect from *source*.  Refuses
    (returns None) whenever that is not provable; see the refusal cases in the
    ``-- refusals --`` sub-section below (step-3) for the enumeration
    guarantee this tier depends on.  A parsed module with zero top-level test
    functions yields ``()``, distinct from the None refusal, and still
    refused downstream by the caller.
    """

    def test_all_decorated_real_shape_yields_one_set_per_item_in_source_order(self):
        """tests/scripts/test_pump_web_ui_installed_unit_parity.py — the acceptance case.

        Both items are individually proven deselected by ``not integration``;
        this is the shape that returns rc=5 today (task 4459's defect).
        """
        assert per_item_marker_names(_ALL_DECORATED_SOURCE) == (
            frozenset({'integration'}),
            frozenset({'integration', 'skipif'}),
        )

    def test_mixed_real_shape_yields_one_set_per_test_function_only(self):
        """tests/scripts/test_know_live_installed_unit_parity.py — the control.

        The leading helper contributes nothing, and the bare test's set is
        EMPTY — so the tuple carries no all-quantified proof and widening
        must not fire on this shape.
        """
        assert per_item_marker_names(_MIXED_DECORATED_SOURCE) == (
            frozenset({'integration'}),
            frozenset({'parametrize'}),
            frozenset(),
        )

    def test_module_level_pytestmark_is_unioned_into_every_items_set(self):
        """The two tiers compose rather than compete."""
        source = (
            'import pytest\n\n'
            'pytestmark = pytest.mark.slow\n\n\n'
            '@pytest.mark.integration\n'
            'def test_a():\n    pass\n\n\n'
            'def test_b():\n    pass\n'
        )
        assert per_item_marker_names(source) == (
            frozenset({'slow', 'integration'}),
            frozenset({'slow'}),
        )

    def test_zero_top_level_test_functions_yields_empty_tuple_not_none(self):
        """Distinct from the None refusal; the caller still refuses this downstream."""
        assert per_item_marker_names('import pytest\n\n\ndef helper():\n    pass\n') == ()

    def test_item_hood_is_name_startswith_test_including_async(self):
        """A name is an item iff it starts with ``test``; ``async def`` counts too."""
        source = (
            'def test_a():\n    pass\n\n\n'
            'def testfoo():\n    pass\n\n\n'
            'def _test_helper():\n    pass\n\n\n'
            'def check_test():\n    pass\n\n\n'
            'def setup_module():\n    pass\n\n\n'
            'async def test_async_thing():\n    pass\n'
        )
        assert per_item_marker_names(source) == (frozenset(), frozenset(), frozenset())

    def test_non_marker_decorator_contributes_nothing_and_does_not_suppress_siblings(self):
        source = (
            'from unittest import mock\n\n\n'
            "@mock.patch('os.getenv')\n"
            '@pytest.mark.slow\n'
            '@some_alias\n'
            'def test_a(mock_getenv):\n    pass\n'
        )
        assert per_item_marker_names(source) == (frozenset({'slow'}),)

    # -- never raises ----------------------------------------------------------

    def test_none_source_is_none(self):
        assert per_item_marker_names(None) is None

    def test_syntax_error_is_none_not_a_raise(self):
        assert per_item_marker_names('def broken(:\n') is None


# ---------------------------------------------------------------------------
# expression_definitely_deselects (step-5: RED)
# ---------------------------------------------------------------------------


class TestExpressionDefinitelyDeselects:
    """``expression_definitely_deselects(expr, marker_names) -> bool``.

    True ONLY on positive proof that every collected item in the file is
    deselected: the ``-m`` expression evaluates to a definite FALSE under Kleene
    (strong 3-valued) logic, with names outside the guaranteed *marker_names*
    treated as UNKNOWN rather than False.  Sound but deliberately incomplete;
    the incompleteness fails safe (no widening, status quo FILE_SCOPED).
    """

    @pytest.mark.parametrize(
        ('expr', 'markers'),
        [
            ('not warm_lane_bash', frozenset({'warm_lane_bash'})),  # the esc-3292-1 case
            ('not integration', frozenset({'integration'})),        # shared/, fused-memory/
            ('not smoke', frozenset({'smoke'})),                    # cockpit/, root
        ],
    )
    def test_every_live_expression_in_this_repo_is_decided(self, expr: str, markers: frozenset):
        """The three ``-m`` expressions actually configured in this repo."""
        assert expression_definitely_deselects(expr, markers) is True

    # -- selected / unknown -> no widening ------------------------------------

    def test_unmarked_file_is_selected_not_deselected(self):
        """THE ANTI-OVER-WIDENING PIN.

        An unmarked file under ``-m 'not warm_lane_bash'`` is SELECTED, so a
        file-scoped run over it collects normally.  This is also the "a
        genuinely-vanished test target stays rc=5 RED" pin: nothing here may
        widen a target whose emptiness is a real defect.
        """
        assert expression_definitely_deselects('not warm_lane_bash', frozenset()) is False

    def test_a_positive_expression_selects_the_marked_file(self):
        assert expression_definitely_deselects(
            'warm_lane_bash', frozenset({'warm_lane_bash'}),
        ) is False

    def test_an_unrelated_guaranteed_marker_proves_nothing(self):
        assert expression_definitely_deselects('not a', frozenset({'b'})) is False

    # -- Kleene soundness / incompleteness ------------------------------------

    def test_false_and_unknown_is_false(self):
        """``not a and not b`` with only ``a`` guaranteed: FALSE and UNKNOWN == FALSE."""
        assert expression_definitely_deselects('not a and not b', frozenset({'a'})) is True

    def test_false_or_unknown_is_unknown(self):
        """``not a or b``: an item also carrying ``b`` would be SELECTED — not provable."""
        assert expression_definitely_deselects('not a or b', frozenset({'a'})) is False

    def test_contradiction_over_unknowns_is_not_proven(self):
        """``a and not a`` is False for every assignment, but Kleene reads UNKNOWN.

        Pins the documented, deliberately-safe incompleteness: Kleene evaluates
        each occurrence independently, so it cannot see the contradiction.  The
        cost is a missed widening, never a wrong one.
        """
        assert expression_definitely_deselects('a and not a', frozenset()) is False

    def test_tautology_over_unknowns_is_not_proven(self):
        assert expression_definitely_deselects('a or not a', frozenset()) is False

    # -- grammar bail-outs -> False, never a raise ----------------------------

    @pytest.mark.parametrize(
        'expr',
        [
            "device(type='cpu')",  # pytest DOES accept this; this detector does not
            'a == b',
            'a > 1',
            'not "s"',
            '',
            'not (',               # SyntaxError
            'lambda: 1',
        ],
    )
    def test_unsupported_grammar_bails_out(self, expr: str):
        assert expression_definitely_deselects(expr, frozenset({'a', 'b'})) is False

    # -- boolean literals are inside the grammar ------------------------------

    def test_literal_false_deselects_everything(self):
        assert expression_definitely_deselects('False', frozenset()) is True

    def test_literal_true_selects_everything(self):
        assert expression_definitely_deselects('True', frozenset()) is False


# ---------------------------------------------------------------------------
# deselecting_expression_for_targets (step-7: RED)
# ---------------------------------------------------------------------------

_WARM_LANE_PYPROJECT = _pyproject(f'"{_REAL_ADDOPTS}"')

_MARKED_SOURCE = 'import pytest\npytestmark = pytest.mark.warm_lane_bash\n\n\ndef test_x():\n    pass\n'  # noqa: E501
_UNMARKED_SOURCE = 'import pytest\n\n\ndef test_y():\n    pass\n'


class _RecordingReader:
    """A dict-backed ``read_source`` that RECORDS every path it is asked for.

    Mirrors ``verify_plan``'s injected ``worktree_reader`` seam
    (``Callable[[str], str | None]``) exactly, so no new I/O seam is introduced;
    the recording is what makes the short-circuit assertable.
    """

    def __init__(self, contents: dict[str, str]) -> None:
        self.contents = contents
        self.paths: list[str] = []

    def __call__(self, path: str) -> str | None:
        self.paths.append(path)
        return self.contents.get(path)


class TestDeselectingExpressionForTargets:
    """``deselecting_expression_for_targets(targets, pyproject_text, test_command, read)``.

    Returns the resolved ``-m`` expression when EVERY target is provably fully
    deselected by it, else None.  It returns the expression rather than a bool
    so the caller can name it in the operator-facing ``PlannedRun.reason``.
    """

    def test_all_marked_targets_return_the_expression(self):
        read = _RecordingReader({'a/test_a.py': _MARKED_SOURCE, 'a/test_b.py': _MARKED_SOURCE})
        assert deselecting_expression_for_targets(
            ['a/test_a.py', 'a/test_b.py'], _WARM_LANE_PYPROJECT, None, read,
        ) == 'not warm_lane_bash'

    def test_one_unmarked_target_is_enough_to_refuse(self):
        """ALL, not ANY — a single selectable target means the run collects."""
        read = _RecordingReader({'a/test_a.py': _MARKED_SOURCE, 'a/test_b.py': _UNMARKED_SOURCE})
        assert deselecting_expression_for_targets(
            ['a/test_a.py', 'a/test_b.py'], _WARM_LANE_PYPROJECT, None, read,
        ) is None

    def test_a_single_unmarked_target_is_refused(self):
        read = _RecordingReader({'a/test_b.py': _UNMARKED_SOURCE})
        assert deselecting_expression_for_targets(
            ['a/test_b.py'], _WARM_LANE_PYPROJECT, None, read,
        ) is None

    def test_empty_targets_never_widen(self):
        """An empty target list is NOT vacuously "all deselected"."""
        read = _RecordingReader({})
        assert deselecting_expression_for_targets([], _WARM_LANE_PYPROJECT, None, read) is None

    def test_unreadable_target_is_refused(self):
        """``read_source`` answering None (missing/unreadable) proves nothing."""
        read = _RecordingReader({})
        assert deselecting_expression_for_targets(
            ['a/test_a.py'], _WARM_LANE_PYPROJECT, None, read,
        ) is None

    def test_no_marker_expression_anywhere_is_refused(self):
        read = _RecordingReader({'a/test_a.py': _MARKED_SOURCE})
        assert deselecting_expression_for_targets(
            ['a/test_a.py'], _pyproject('"-n auto -q"'), 'uv run pytest tests/', read,
        ) is None

    def test_short_circuits_before_reading_any_target(self):
        """THE COST BOUND.

        With no ``-m`` expression to resolve, not a single target is read — so
        consulting this from a verify plan costs exactly one pyproject read per
        ModuleConfig, and zero target reads for every module that declares no
        marker expression at all.
        """
        read = _RecordingReader({'a/test_a.py': _MARKED_SOURCE})
        result = deselecting_expression_for_targets(
            ['a/test_a.py'], _pyproject('"-n auto -q"'), 'uv run pytest tests/', read,
        )
        assert result is None
        assert read.paths == []

    def test_a_cli_dash_m_that_reselects_the_bucket_is_refused(self):
        """Last-wins: the lane's own ``-m warm_lane_bash`` SELECTS the marked target."""
        read = _RecordingReader({'a/test_a.py': _MARKED_SOURCE})
        assert deselecting_expression_for_targets(
            ['a/test_a.py'],
            _WARM_LANE_PYPROJECT,
            'uv run pytest tests/ -m warm_lane_bash',
            read,
        ) is None


# ---------------------------------------------------------------------------
# derive_verify_plan wiring (step-9: RED until step-10)
# ---------------------------------------------------------------------------

ORCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

#: A module whose addopts deselect ``slow``, mirroring the real orchestrator shape.
_DESELECTING_PYPROJECT = _pyproject('"-n auto -m \'not slow\'"')
_PLAIN_PYPROJECT = _pyproject('"-n auto -q"')

_SYNTHETIC_CONTENTS: dict[str, str] = {
    'mod/pyproject.toml': _DESELECTING_PYPROJECT,
    'mod/tests/test_a.py': 'import pytest\npytestmark = pytest.mark.slow\n',
    'mod/tests/test_b.py': 'import pytest\n\n\ndef test_b():\n    pass\n',
    # 'other' declares no -m at all — the control for "module without a marker
    # expression is never widened", even with an identically-marked target.
    'other/pyproject.toml': _PLAIN_PYPROJECT,
    'other/tests/test_a.py': 'import pytest\npytestmark = pytest.mark.slow\n',
    # Arms 1-3 controls: each of these prefixes ALSO gets a deselecting
    # pyproject, so "unchanged" means unchanged in its presence, not its absence.
    'orchestrator/pyproject.toml': _DESELECTING_PYPROJECT,
    'shared/pyproject.toml': _DESELECTING_PYPROJECT,
}


def _synthetic_reader(path: str) -> str | None:
    """Dict-backed stand-in for real file I/O, same seam as ``worktree_reader``."""
    return _SYNTHETIC_CONTENTS.get(path)


def _mc(prefix: str) -> ModuleConfig:
    """A fully-configured ModuleConfig for *prefix* (all three tool commands set)."""
    return ModuleConfig(
        prefix=prefix,
        test_command=f'uv run --directory {prefix} pytest tests/',
        lint_command=f'uv run --directory {prefix} ruff check src/ tests/',
        type_check_command=f'uv run --directory {prefix} pyright src/ tests/',
    )


def _run_for(plan: VerifyPlan, prefix: str, tool_word: str):
    """The *prefix* PlannedRun whose reason names *tool_word* (e.g. ``'pytest:'``)."""
    return next(
        (r for r in plan.runs if r.module_prefix == prefix and r.reason.startswith(tool_word)),
        None,
    )


class TestDeriveModuleRunsWidensOnFullDeselection:
    """A fully marker-deselected FILE_SCOPED pytest run widens to the owning FULL_SUITE.

    Without the widening the planned run collects zero items, pytest exits rc=5,
    and ``verify_classify._classify_opaque`` reds it — a false RED on a diff that
    touched a real, passing test file.
    """

    @pytest.mark.parametrize('role', ['task', 'merge'])
    def test_fully_deselected_target_widens_at_both_roles(self, role):
        """Arm 4 is shared by both roles, so one fix covers both."""
        mc = _mc('mod')
        plan = derive_verify_plan(
            ['mod/tests/test_a.py'], [mc], None, _synthetic_reader, role=role,
        )
        run = _run_for(plan, 'mod', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)
        # PlannedRun's invariant: scoped_targets is non-empty iff FILE_SCOPED.
        assert run.scoped_targets == ()
        assert run.reason.startswith('pytest: ')
        assert 'not slow' in run.reason
        assert 'deselected' in run.reason

    def test_reason_names_the_files_the_widened_run_still_does_not_execute(self):
        """The widened run applies the SAME addopts — the trigger files stay unrun.

        FULL_SUITE forbids ``scoped_targets`` (PlannedRun's invariant), so the
        reason string is the only channel that can record WHICH files went
        unverified.  Without it the record reads as "the change was verified",
        which is the silent degradation the repo's design invariants forbid.
        """
        plan = derive_verify_plan(
            ['mod/tests/test_a.py'], [_mc('mod')], None, _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'mod', 'pytest:')
        assert run is not None
        assert 'mod/tests/test_a.py' in run.reason
        assert 'NOT executed' in run.reason

    def test_widening_does_not_touch_the_other_tool_slots(self):
        """Marker deselection is a pytest-only concern — lint and pyright stay scoped."""
        plan = derive_verify_plan(
            ['mod/tests/test_a.py'], [_mc('mod')], None, _synthetic_reader, role='task',
        )
        for tool_word in ('lint:', 'pyright:'):
            run = _run_for(plan, 'mod', tool_word)
            assert run is not None, tool_word
            assert run.scope_kind is ScopeKind.FILE_SCOPED, tool_word
            assert run.scoped_targets == ('mod/tests/test_a.py',), tool_word

    def test_unmarked_target_keeps_todays_file_scoped_run(self):
        """CONTROL: a genuinely-vanished test target must still rc=5 RED.

        An unmarked file is SELECTED, so its file-scoped run is not empty by
        marker deselection.  If it collects nothing anyway, that emptiness is a
        real defect and must stay visible — nothing here may widen it away.
        """
        plan = derive_verify_plan(
            ['mod/tests/test_b.py'], [_mc('mod')], None, _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'mod', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.reason == 'pytest: file-scoped to touched test file(s)'
        assert run.scoped_targets == ('mod/tests/test_b.py',)

    def test_mixed_targets_stay_file_scoped_to_both(self):
        """CONTROL: ALL, not ANY — one selectable target keeps the whole run scoped."""
        plan = derive_verify_plan(
            ['mod/tests/test_a.py', 'mod/tests/test_b.py'], [_mc('mod')], None,
            _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'mod', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.scoped_targets == ('mod/tests/test_a.py', 'mod/tests/test_b.py')

    def test_module_without_a_marker_expression_is_never_widened(self):
        """CONTROL: an identically-marked target under a module declaring no ``-m``."""
        plan = derive_verify_plan(
            ['other/tests/test_a.py'], [_mc('other')], None, _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'other', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.reason == 'pytest: file-scoped to touched test file(s)'

    def test_arm_1_conftest_is_unchanged(self):
        """CONTROL: arms 1-3 return above the probe, so they cannot be reached by it."""
        plan = derive_verify_plan(
            ROOT_CONFTEST_DIFF, [_mc('orchestrator')], None, _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.reason == (
            'pytest: conftest touched (orchestrator/tests/conftest.py) — full suite required'
        )

    def test_arm_2_test_data_is_unchanged(self):
        """CONTROL: the task-1852 golden — the first instance of this defect class."""
        plan = derive_verify_plan(
            DATA_MODULE_DIFF, [_mc('shared')], None, _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'shared', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.reason == (
            'pytest: test-data module touched (shared/tests/silent_fallthrough_allowlist.py)'
            ' — full suite required'
        )

    def test_arm_3_task_role_production_floor_is_unchanged(self):
        """CONTROL: the task-3294 floor keeps its own reason, not the new one."""
        plan = derive_verify_plan(
            SOURCE_ONLY_DIFF, [_mc('orchestrator')], None, _synthetic_reader, role='task',
        )
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.reason == (
            'pytest: source-only diff — owning-module full suite (task role); '
            'sibling modules NOT run'
        )


class TestProbeConsultsTheCommandsEffectiveRootdir:
    """WHERE the probe looks for ``addopts``, and which commands it refuses outright.

    pytest reads ``addopts`` from its ROOTDIR, which follows the command's
    effective cwd — NOT from ``mc.prefix``.  The two come apart in this repo:
    ``scripts`` and ``tests/scripts`` both run ``uv run --project shared pytest
    tests/scripts/ ...`` from the REPO ROOT, so ``<prefix>/pyproject.toml``
    would be a config pytest never applies.

    Every reader below serves a DESELECTING pyproject at EVERY ``*pyproject.toml``
    path it is asked for, so a refusal test can never pass vacuously through a
    missing file: if the guard under test stopped firing, the widening would.
    """

    #: A marked target under every prefix these tests use.
    _MARKED = 'import pytest\npytestmark = pytest.mark.slow\n'

    @classmethod
    def _reader_serving_every_pyproject(cls, reads: list[str]):
        """A recording reader for which EVERY pyproject path deselects ``slow``."""
        def read(path: str) -> str | None:
            reads.append(path)
            if path.endswith('pyproject.toml'):
                return _DESELECTING_PYPROJECT
            return cls._MARKED if path.endswith('.py') else None
        return read

    def test_repo_root_config_is_read_when_the_command_has_no_directory(self):
        """The real ``scripts`` shape: ``uv run --project shared pytest ...`` from root."""
        reads: list[str] = []
        def read(path: str) -> str | None:
            reads.append(path)
            if path == 'pyproject.toml':
                return _DESELECTING_PYPROJECT
            # The PREFIX config declares no -m at all: if the probe read this
            # one instead of the root's, no widening would happen.
            if path == 'sub/pyproject.toml':
                return _PLAIN_PYPROJECT
            return self._MARKED if path.endswith('.py') else None

        mc = ModuleConfig(
            prefix='sub',
            test_command='uv run --project shared pytest tests/sub/ --tb=short -q',
            lint_command='uv run --project shared ruff check sub/',
        )
        plan = derive_verify_plan(['sub/tests/test_a.py'], [mc], None, read, role='task')
        run = _run_for(plan, 'sub', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert 'sub/pyproject.toml' not in reads, 'prefix config is not pytest rootdir here'

    def test_directory_flag_selects_that_directorys_config(self):
        """``uv run --directory X pytest`` roots at X — the orchestrator/shared shape."""
        reads: list[str] = []
        read = self._reader_serving_every_pyproject(reads)
        plan = derive_verify_plan(
            ['mod/tests/test_a.py'], [_mc('mod')], None, read, role='task',
        )
        run = _run_for(plan, 'mod', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert 'mod/pyproject.toml' in reads
        assert 'pyproject.toml' not in reads

    #: ``npx jest`` parses STRUCTURED (ToolKind.NPX, ``raw is None``), so it
    #: isolates guard 1 — guard 2 cannot also be what refuses it.  ``npm test``
    #: parses OPAQUE and is refused by either; both are pinned because neither
    #: is a live config today, which is precisely why an unpinned regression
    #: here would go unnoticed.
    @pytest.mark.parametrize('test_command', ['npx jest tests/', 'npm test'])
    def test_a_non_pytest_test_command_is_never_widened(self, test_command):
        """GUARD 1: a non-pytest suite never applies a pyproject's addopts."""
        reads: list[str] = []
        read = self._reader_serving_every_pyproject(reads)
        mc = ModuleConfig(prefix='js', test_command=test_command)
        plan = derive_verify_plan(['js/tests/test_a.py'], [mc], None, read, role='task')
        run = _run_for(plan, 'js', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.reason == 'pytest: file-scoped to touched test file(s)'
        assert run.scoped_targets == ('js/tests/test_a.py',)
        assert not any(p.endswith('pyproject.toml') for p in reads), 'refused before any read'

    def test_a_chained_test_command_is_never_widened(self):
        """GUARD 2: a raw-retained chain hides both the rootdir and which clause runs.

        ``_scope_prefix_to_keyword`` truncates a chained ``test_command`` to its
        FIRST pytest clause; rather than risk the probe describing a different
        invocation than the scoper, the probe refuses the whole shape.
        """
        reads: list[str] = []
        read = self._reader_serving_every_pyproject(reads)
        mc = ModuleConfig(
            prefix='mod',
            test_command='cd mod && uv run pytest tests/ && python3 scripts/check.py mod/tests',
        )
        plan = derive_verify_plan(['mod/tests/test_a.py'], [mc], None, read, role='task')
        run = _run_for(plan, 'mod', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.reason == 'pytest: file-scoped to touched test file(s)'
        assert not any(p.endswith('pyproject.toml') for p in reads), 'refused before any read'


class TestWarmLaneBashRealConfigRegression:
    """The esc-3292-1 incident golden, driven against the REAL repo files.

    Never hand-seeded: the addopts, the suite file's ``pytestmark`` and the
    ModuleConfig's commands are all read from disk, so this pins the actual
    shipped configuration rather than a retyped copy of it.  Path resolution
    uses the ``Path(__file__).resolve().parents[N]`` idiom from
    test_warm_lane_bash_bucket_placement.py — correct under both the repo-root
    and the ``orchestrator/``-cwd pytest invocations.
    """

    #: Non-vacuity guards run FIRST: the golden below must never be able to pass
    #: because its premise evaporated (the deselection moved, or the marker was
    #: dropped from the suite file).
    def test_real_addopts_still_deselects_the_bucket(self):
        expr = resolve_marker_expression((ORCH_DIR / 'pyproject.toml').read_text(), None)
        assert expr is not None, 'orchestrator addopts no longer carry a -m expression'
        assert 'warm_lane_bash' in expr

    def test_real_suite_file_still_declares_the_bucket_at_module_level(self):
        source = (ORCH_DIR / 'tests' / 'test_warm_lane_bash_suite.py').read_text()
        assert 'warm_lane_bash' in module_level_marker_names(source)

    @staticmethod
    def _real_module_config() -> ModuleConfig:
        """A ModuleConfig whose commands come VERBATIM from orchestrator/orchestrator.yaml."""
        loaded = yaml.safe_load((ORCH_DIR / 'orchestrator.yaml').read_text())
        return ModuleConfig(
            prefix='orchestrator',
            test_command=loaded['test_command'],
            lint_command=loaded['lint_command'],
            type_check_command=loaded['type_check_command'],
        )

    @staticmethod
    def _read(path: str) -> str | None:
        try:
            return (REPO_ROOT / path).read_text()
        except OSError:
            return None

    @pytest.mark.parametrize('role', ['task', 'merge'])
    def test_touching_only_the_warm_lane_suite_widens_to_the_full_suite(self, role):
        """Before this task, BOTH roles planned a zero-collecting FILE_SCOPED run."""
        mc = self._real_module_config()
        plan = derive_verify_plan(
            ['orchestrator/tests/test_warm_lane_bash_suite.py'], [mc], None, self._read,
            role=role,
        )
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.scoped_targets == ()
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)


# ---------------------------------------------------------------------------
# The FALLBACK (no-module_configs) arm — task 3513, the 3494 twin
# ---------------------------------------------------------------------------
#
# Task 3494 closed the "path says COLLECTABLE_TEST, pytest collects zero -> rc=5
# -> false RED" defect in ``_derive_module_runs`` arm 4 only.  The twin arm in
# ``_derive_fallback_runs`` (fires when a project registers NO module_configs)
# has the identical failure mode: a repo root carrying
# ``addopts = "-m 'not X'"`` plus a diff touching only X-marked test files still
# plans — and, crucially, EXECUTES — a zero-collecting file-scoped run.
#
# The two arms share ONE probe (``deselecting_expression_for_command``) so they
# can never disagree about which commands are refused or where the ini file is
# looked for; a divergence there would reopen exactly the over-fire risk task
# 3494's docstring feared.

#: Every ``*.py`` these readers serve is module-level ``slow``-marked, and every
#: ``*pyproject.toml`` deselects ``slow`` — so a REFUSAL test can never pass
#: vacuously through a missing file.  If the guard under test stopped firing,
#: the widening would happen and the assertion would fail.
_SLOW_MARKED_SOURCE = 'import pytest\npytestmark = pytest.mark.slow\n\n\ndef test_x():\n    pass\n'
_UNMARKED_PLAIN_SOURCE = 'import pytest\n\n\ndef test_y():\n    pass\n'


def _permissive_reader(
    reads: list[str],
    overrides: dict[str, str | None] | None = None,
):
    """A recording reader that says "yes, widen" to everything it is not overridden on.

    Serves ``_DESELECTING_PYPROJECT`` at EVERY ``*pyproject.toml`` path and
    ``_SLOW_MARKED_SOURCE`` at every ``*.py`` path, recording each read into
    *reads*.  *overrides* wins where present (a ``None`` value models a file the
    reader cannot read — a directory, or a missing path).

    Same injected seam as ``verify_plan``'s ``worktree_reader``
    (``Callable[[str], str | None]``); no new I/O is introduced.
    """
    over = overrides or {}

    def read(path: str) -> str | None:
        reads.append(path)
        if path in over:
            return over[path]
        if path.endswith('pyproject.toml'):
            return _DESELECTING_PYPROJECT
        if path.endswith('.py'):
            return _SLOW_MARKED_SOURCE
        return None

    return read


class TestDeselectingExpressionForCommand:
    """``deselecting_expression_for_command(test_command, targets, worktree_reader)``.

    Task 3494's probe, promoted to a public pure function taking a COMMAND
    STRING instead of a ModuleConfig, so the module arm and the new fallback
    arm share one implementation of "which commands are refused" and "where the
    ini file is looked for".  Returns the effective ``-m`` expression iff it
    provably deselects EVERY target; None — "keep today's FILE_SCOPED
    behaviour" — in every other case.

    *targets* are worktree-ROOT-relative (the reader's frame), while the
    command's own targets may be cwd-relative; only the CONFIG path follows the
    command's ``cwd_rel``.
    """

    def test_marked_target_under_a_deselecting_root_returns_the_expression(self):
        """The positive case: the bare fallback shape, root pyproject, marked target."""
        reads: list[str] = []
        assert deselecting_expression_for_command(
            'pytest mod/tests/test_a.py',
            ['mod/tests/test_a.py'],
            _permissive_reader(reads),
        ) == 'not slow'

    def test_a_command_without_a_cwd_reads_the_repo_root_config(self):
        """WHERE: no ``cd``/``--directory`` means pytest's rootdir IS the repo root."""
        reads: list[str] = []
        deselecting_expression_for_command(
            'pytest mod/tests/test_a.py', ['mod/tests/test_a.py'], _permissive_reader(reads),
        )
        assert 'pyproject.toml' in reads
        assert 'mod/pyproject.toml' not in reads

    def test_a_cd_command_reads_that_subprojects_config_not_the_root(self):
        """The anti-over-fire pin: the probe follows the command's effective rootdir.

        ``cd sub && uv run pytest ...`` roots at ``sub``, so a root-only
        ``addopts`` must never be what proves the deselection.
        """
        reads: list[str] = []
        read = _permissive_reader(reads)
        assert deselecting_expression_for_command(
            'cd sub && uv run pytest tests/test_a.py', ['sub/tests/test_a.py'], read,
        ) == 'not slow'
        assert 'sub/pyproject.toml' in reads
        assert 'pyproject.toml' not in reads, 'root config is not this command\'s rootdir'

    #: ``npm test`` and ``./scripts/test.sh`` both parse OPAQUE; a pyproject's
    #: addopts describe a suite neither command ever invokes, so consulting them
    #: would widen on a false premise.
    @pytest.mark.parametrize('test_command', ['npm test', './scripts/test.sh'])
    def test_a_non_pytest_command_is_refused(self, test_command):
        """GUARD 1."""
        reads: list[str] = []
        assert deselecting_expression_for_command(
            test_command, ['mod/tests/test_a.py'], _permissive_reader(reads),
        ) is None
        assert not any(p.endswith('pyproject.toml') for p in reads), 'refused before any read'

    def test_a_raw_retained_chain_is_refused(self):
        """GUARD 2: a chained command hides both its rootdir and which clause is scoped."""
        reads: list[str] = []
        assert deselecting_expression_for_command(
            'cd sub && uv run pytest x && cd .. && pytest y',
            ['sub/tests/test_a.py'],
            _permissive_reader(reads),
        ) is None
        assert not any(p.endswith('pyproject.toml') for p in reads), 'refused before any read'

    def test_a_none_command_is_refused(self):
        """No command means no suite to reason about."""
        reads: list[str] = []
        assert deselecting_expression_for_command(
            None, ['mod/tests/test_a.py'], _permissive_reader(reads),
        ) is None
        assert reads == []

    def test_empty_targets_never_widen(self):
        """An empty target list is refused, never treated as vacuously all-deselected."""
        reads: list[str] = []
        assert deselecting_expression_for_command(
            'pytest', [], _permissive_reader(reads),
        ) is None

    def test_one_unmarked_target_is_enough_to_refuse(self):
        """ALL, not ANY — a single still-collecting target means the run is not empty."""
        reads: list[str] = []
        read = _permissive_reader(reads, {'mod/tests/test_b.py': _UNMARKED_PLAIN_SOURCE})
        assert deselecting_expression_for_command(
            'pytest mod/tests/test_a.py mod/tests/test_b.py',
            ['mod/tests/test_a.py', 'mod/tests/test_b.py'],
            read,
        ) is None

    def test_an_unreadable_target_is_refused(self):
        """A None answer proves nothing — the fail-safe direction."""
        reads: list[str] = []
        read = _permissive_reader(reads, {'mod/tests/test_a.py': None})
        assert deselecting_expression_for_command(
            'pytest mod/tests/test_a.py', ['mod/tests/test_a.py'], read,
        ) is None

    def test_a_cli_dash_m_that_reselects_the_bucket_is_refused(self):
        """Last-wins: a CLI ``-m slow`` SELECTS the marked target the addopts dropped."""
        reads: list[str] = []
        assert deselecting_expression_for_command(
            'pytest -m slow mod/tests/test_a.py',
            ['mod/tests/test_a.py'],
            _permissive_reader(reads),
        ) is None


def _fallback_mc(test_command: str | None) -> ModuleConfig:
    """The shape ``verify._build_fallback_config`` emits, with *test_command* substituted.

    ``prefix='__fallback__'`` and the lint/type-check commands are carried so
    every assertion below can check that widening touches the pytest slot ONLY —
    marker deselection is a pytest-only concern.
    """
    return ModuleConfig(
        prefix='__fallback__',
        test_command=test_command,
        lint_command='ruff check tests/test_a.py',
        type_check_command='pyright tests/test_a.py',
    )


class TestWidenFallbackRefuses:
    """``widen_fallback_for_marker_deselection(fallback, reader)`` — the REFUSAL half.

    The one-directional fail-safe: this function takes the ALREADY-EXECUTED
    fallback ModuleConfig (post ``_build_fallback_config`` + ``_apply_cargo_scope``),
    so the executed command's OWN SHAPE — never a guess about which branch
    produced it — decides.  Every shape ``_build_fallback_config`` can emit that
    must not widen is pinned here, and each returns the input ModuleConfig
    unchanged (identity, not merely equality) with a None reason.

    Every reader serves a DESELECTING pyproject at EVERY ``*pyproject.toml``
    path AND a ``slow``-marked module at EVERY ``*.py`` path, so no refusal can
    pass vacuously: if the guard under test stopped firing, the widening would.
    """

    @staticmethod
    def _refuses(test_command: str | None, overrides: dict[str, str | None] | None = None):
        """Assert *test_command* is refused, and return the reads for further checks."""
        reads: list[str] = []
        fallback = _fallback_mc(test_command)
        widened, reason = widen_fallback_for_marker_deselection(
            fallback, _permissive_reader(reads, overrides),
        )
        assert widened is fallback, 'a refusal must return the input config untouched'
        assert reason is None
        return reads

    def test_no_test_command_is_refused(self):
        """``_build_fallback_config``'s "no collectable tests" branch leaves it None."""
        assert self._refuses(None) == []

    def test_the_mixed_root_plus_subproject_chain_is_refused(self):
        """Task 2368's shape parses raw-retained: which clause gets scoped is unrecoverable."""
        reads = self._refuses(
            'cd sub && uv run pytest tests/test_a.py && cd .. && pytest tests/test_b.py',
        )
        assert not any(p.endswith('pyproject.toml') for p in reads), 'refused before any read'

    def test_the_uv_extras_shape_is_refused(self):
        """Task 2641's ``uv run --extra dev pytest`` also parses raw-retained."""
        reads = self._refuses('cd sub && uv run --extra dev pytest tests/test_a.py')
        assert not any(p.endswith('pyproject.toml') for p in reads), 'refused before any read'

    @pytest.mark.parametrize('test_command', ['npm test', './scripts/test.sh'])
    def test_a_non_pytest_command_is_refused(self, test_command):
        """A pyproject's addopts describe a suite these commands never invoke."""
        self._refuses(test_command)

    #: A configured suite is run VERBATIM by ``_build_fallback_config`` (never
    #: file-scoped), so it has no file targets to drop — nothing to widen.
    @pytest.mark.parametrize('test_command', ['pytest', "uv run pytest -m 'not slow'"])
    def test_an_unscoped_suite_has_nothing_to_widen(self, test_command):
        self._refuses(test_command)

    #: The conftest branch emits a DIRECTORY target (``_fallback_pytest_targets``),
    #: and a configured ``pytest tests/`` is directory-shaped too.  The
    #: ``is_file()``-guarded worktree reader answers None for a directory, so no
    #: proof of deselection can exist — the refusal is structural, not incidental.
    @pytest.mark.parametrize(
        'test_command', ['pytest . mod/tests/test_a.py', 'pytest mod/tests'],
    )
    def test_a_directory_target_is_unprovable_and_refused(self, test_command):
        self._refuses(test_command)

    def test_one_unmarked_target_is_enough_to_refuse(self):
        """ALL, not ANY — a single still-collecting file means the run is not empty."""
        self._refuses(
            'pytest mod/tests/test_a.py mod/tests/test_b.py',
            {'mod/tests/test_b.py': _UNMARKED_PLAIN_SOURCE},
        )

    def test_a_rootdir_declaring_no_marker_expression_is_refused(self):
        """No ``-m`` anywhere means nothing was deselected — today's FILE_SCOPED run stands."""
        self._refuses(
            'pytest mod/tests/test_a.py', {'pyproject.toml': _PLAIN_PYPROJECT},
        )


class TestWidenFallbackWidensOnFullDeselection:
    """``widen_fallback_for_marker_deselection`` — the POSITIVE half.

    The remedy mirrors task 3494's arm 4a exactly: drop the file targets and
    run the same command's full suite, rather than SKIP.  The task-1852 SKIP
    precedent applies only where there is NO suite at all to run; here the very
    existence of an ``-m`` expression at the command's rootdir is evidence of a
    real marker-partitioned suite.  Skipping would turn a false RED into a
    silent no-coverage GREEN.

    Widening is DEGRADATION, not verification: the widened run applies the SAME
    addopts, so the trigger files stay deselected.  The reason is the only
    channel that can say so (FULL_SUITE forbids ``scoped_targets``), so every
    assertion below checks it names the files as NOT executed.
    """

    def test_the_root_shape_widens_and_preserves_every_other_command(self):
        """THE defect: a bare ``pytest <file>`` under a deselecting repo root."""
        reads: list[str] = []
        fallback = _fallback_mc('pytest tests/test_smoke.py')
        read = _permissive_reader(reads, {
            'pyproject.toml': _pyproject('"-m \'not smoke\'"'),
            'tests/test_smoke.py': 'import pytest\npytestmark = pytest.mark.smoke\n',
        })
        widened, reason = widen_fallback_for_marker_deselection(fallback, read)

        assert widened.test_command == 'pytest', 'targets dropped, command otherwise intact'
        assert reason is not None
        # Marker deselection is a pytest-only concern: nothing else may move.
        assert widened.prefix == fallback.prefix
        assert widened.lint_command == fallback.lint_command
        assert widened.type_check_command == fallback.type_check_command

    def test_the_reason_names_the_files_this_run_still_does_not_execute(self):
        """Mirrors arm 4a's wording, so ONE operator-facing phrase covers the class."""
        reads: list[str] = []
        read = _permissive_reader(reads, {
            'pyproject.toml': _pyproject('"-m \'not smoke\'"'),
            'tests/test_smoke.py': 'import pytest\npytestmark = pytest.mark.smoke\n',
        })
        _, reason = widen_fallback_for_marker_deselection(
            _fallback_mc('pytest tests/test_smoke.py'), read,
        )
        assert reason is not None
        assert reason.startswith('pytest: ')
        assert 'tests/test_smoke.py' in reason
        assert 'not smoke' in reason
        assert 'NOT executed' in reason
        assert 'rc=5' in reason

    def test_multiple_marked_targets_widen_and_are_all_named(self):
        """ALL deselected — and the reason accounts for every file that goes unrun."""
        reads: list[str] = []
        widened, reason = widen_fallback_for_marker_deselection(
            _fallback_mc('pytest tests/test_a.py tests/test_b.py'), _permissive_reader(reads),
        )
        assert widened.test_command == 'pytest'
        assert reason is not None
        assert 'tests/test_a.py' in reason
        assert 'tests/test_b.py' in reason

    def test_the_subproject_shape_widens_against_its_own_rootdir(self):
        """``cd sub && uv run pytest <file>`` — task 2344's rescoping, unscoped in place.

        The executed command's targets are SUB-relative while the reader is
        worktree-ROOT-relative, so the widener must resolve them through
        ``cwd_rel`` before handing them to the probe.
        """
        reads: list[str] = []
        read = _permissive_reader(reads, {
            # ONLY the subproject deselects; the root declares no -m at all.
            'pyproject.toml': _PLAIN_PYPROJECT,
            'sub/tests/test_smoke.py': 'import pytest\npytestmark = pytest.mark.slow\n',
        })
        widened, reason = widen_fallback_for_marker_deselection(
            _fallback_mc('cd sub && uv run pytest tests/test_smoke.py'), read,
        )
        assert widened.test_command == 'cd sub && uv run pytest'
        assert reason is not None
        assert 'sub/tests/test_smoke.py' in reason, 'reason names the root-relative path'
        assert 'sub/tests/test_smoke.py' in reads, 'target resolved into the reader\'s frame'
        assert 'sub/pyproject.toml' in reads

    def test_a_root_only_addopts_never_widens_a_subproject_command(self):
        """THE anti-over-fire pin: the config a ``cd sub`` command's rootdir may not see.

        pytest would walk UP to the repo root when ``sub/pyproject.toml``
        declares no ini_options — a walk this probe deliberately does not model.
        Refusing is an UNDER-fire, the one direction task 3494 permits.
        """
        reads: list[str] = []
        read = _permissive_reader(reads, {
            'sub/pyproject.toml': _PLAIN_PYPROJECT,
            'sub/tests/test_smoke.py': 'import pytest\npytestmark = pytest.mark.slow\n',
        })
        fallback = _fallback_mc('cd sub && uv run pytest tests/test_smoke.py')
        widened, reason = widen_fallback_for_marker_deselection(fallback, read)
        assert widened is fallback
        assert reason is None


@pytest.mark.usefixtures('code_default_config')
class TestExecutedFallbackPlanRecordsTheWidening:
    """``verify._executed_fallback_plan(plan, fallback, pytest_reason=...)``.

    In the fallback branch the plan is a RECORD, not the execution driver:
    ``run_scoped_verification`` hands ``run_verification`` the ModuleConfig, not
    the plan.  The widening therefore happens to the EXECUTED config, and this
    reconciliation is what keeps the record honest about it.

    The fact is passed EXPLICITLY rather than re-derived here: after widening,
    a ``test_command == 'pytest'`` is byte-identical to an un-widened bare
    default suite, so this layer cannot tell the two apart and would have to
    re-run the probe — a second call, and a second chance to disagree with the
    executed layer.

    ``code_default_config`` is required: the autouse ``_isolate_orch_config``
    fixture pins ORCH_CONFIG_PATH at the real dark-factory-orchestrator.yaml,
    whose configured ``test_command`` would take the "configured suite runs
    verbatim" arm instead of the bare-default file-scoped one this task is about.
    """

    _WIDENED_REASON = (
        "pytest: touched test file(s) tests/test_smoke.py are ALL deselected by the "
        "effective -m 'not smoke' — fallback full suite instead of a zero-collecting "
        'file-scoped run (rc=5); those file(s) stay deselected in this run too and '
        'are NOT executed by it'
    )

    @staticmethod
    def _decision_plan(files: list[str], tmp_path: Path):
        """The fallback-branch DECISION plan, from the bare-default config."""
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.test_command == 'pytest', (
            'premise: only the bare default reaches the file-scoped fallback arm'
        )
        return verify_plan.derive_verify_plan(files, [], config, lambda _p: None)

    @staticmethod
    def _widened_fallback() -> ModuleConfig:
        """What ``widen_fallback_for_marker_deselection`` handed the executor."""
        return ModuleConfig(
            prefix='__fallback__',
            test_command='pytest',
            lint_command='ruff check tests/test_smoke.py',
            type_check_command='pyright tests/test_smoke.py',
        )

    def test_the_pytest_run_records_the_widening(self, tmp_path):
        plan = self._decision_plan(['tests/test_smoke.py'], tmp_path)
        decided = _run_for(plan, '__fallback__', 'pytest:')
        assert decided is not None
        assert decided.scope_kind is ScopeKind.FILE_SCOPED

        executed = verify._executed_fallback_plan(
            plan, self._widened_fallback(), pytest_reason=self._WIDENED_REASON,
        )
        run = _run_for(executed, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.reason == self._WIDENED_REASON
        # PlannedRun's invariant: scoped_targets is non-empty iff FILE_SCOPED.
        assert run.scoped_targets == ()
        # cmd is still the raw-wrapped EXECUTED command the reconciliation builds.
        assert run.cmd is not None
        assert run.cmd.raw == 'pytest'

    def test_the_other_tool_slots_are_untouched(self, tmp_path):
        """Task 3219's scoped_targets propagation must not regress on lint/pyright."""
        plan = self._decision_plan(['tests/test_smoke.py'], tmp_path)
        executed = verify._executed_fallback_plan(
            plan, self._widened_fallback(), pytest_reason=self._WIDENED_REASON,
        )
        for tool_word in ('lint:', 'pyright:'):
            before = _run_for(plan, '__fallback__', tool_word)
            after = _run_for(executed, '__fallback__', tool_word)
            assert after is not None and before is not None
            assert after.scope_kind is ScopeKind.FILE_SCOPED
            assert after.reason == before.reason
            assert after.scoped_targets == before.scoped_targets == ('tests/test_smoke.py',)

    def test_omitting_the_reason_is_byte_identical_to_today(self, tmp_path):
        """The default path must not move — the widening is strictly opt-in."""
        plan = self._decision_plan(['tests/test_smoke.py'], tmp_path)
        fallback = self._widened_fallback()
        assert (
            verify._executed_fallback_plan(plan, fallback).to_dict()
            == verify._executed_fallback_plan(plan, fallback, pytest_reason=None).to_dict()
        )

    def test_a_skipped_pytest_slot_does_not_crash(self, tmp_path):
        """A source-only diff skips pytest; only the 'pytest:'-prefixed run may move."""
        plan = self._decision_plan(['src/mod.py'], tmp_path)
        decided = _run_for(plan, '__fallback__', 'pytest:')
        assert decided is not None
        assert decided.scope_kind is ScopeKind.SKIPPED

        executed = verify._executed_fallback_plan(
            plan,
            ModuleConfig(prefix='__fallback__', lint_command='ruff check src/mod.py'),
            pytest_reason=self._WIDENED_REASON,
        )
        lint = _run_for(executed, '__fallback__', 'lint:')
        assert lint is not None
        assert lint.reason == 'lint: file-scoped to touched file(s)'
        assert lint.scope_kind is ScopeKind.FILE_SCOPED

    def test_the_no_py_files_shape_passes_through_unchanged(self, tmp_path):
        """A run with no recognised tool prefix is not keyed to any ModuleConfig slot."""
        plan = self._decision_plan(['src/lib.rs'], tmp_path)
        assert len(plan.runs) == 1

        executed = verify._executed_fallback_plan(
            plan,
            ModuleConfig(prefix='__fallback__', test_command='pytest'),
            pytest_reason=self._WIDENED_REASON,
        )
        assert executed.runs == plan.runs


#: A worktree-root pyproject whose addopts deselect ``smoke`` — the ingredient
#: dark-factory's own root pyproject.toml carries, and the reason this defect
#: class is reachable in any project that registers no module_configs.
_SMOKE_DESELECTING_PYPROJECT = _pyproject('"-m \'not smoke\'"')
_SMOKE_MARKED_SOURCE = 'import pytest\npytestmark = pytest.mark.smoke\n\n\ndef test_s():\n    pass\n'


@pytest.mark.usefixtures('code_default_config')
class TestRunScopedVerificationWidensTheFallback:
    """End-to-end through ``verify.run_scoped_verification``'s FALLBACK branch.

    The layer that matters: unlike the module-config branch (plan-authoritative
    via ``_executed_module_configs_from_plan``), this branch hands
    ``run_verification`` the ModuleConfig, NOT the plan.  A fix confined to the
    plan record would produce an honest record and the SAME rc=5 false RED, so
    these assertions are on what was actually handed to the executor.

    ``code_default_config`` is required — see
    ``TestExecutedFallbackPlanRecordsTheWidening``'s docstring for why the
    ambient config would otherwise take the "configured suite runs verbatim" arm.
    """

    @staticmethod
    async def _run(tmp_path: Path, sources: dict[str, str], pyproject_text: str | None):
        """Drive the fallback branch over a real worktree; return (executed_mc, result)."""
        if pyproject_text is not None:
            (tmp_path / 'pyproject.toml').write_text(pyproject_text)
        for rel, text in sources.items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        config = OrchestratorConfig(project_root=tmp_path)
        assert config.test_command == 'pytest', (
            'premise: only the bare default reaches the file-scoped fallback arm'
        )
        spy = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=spy):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=list(sources),
            )
        executed = _executed_module_configs(spy)
        assert len(executed) == 1, 'the fallback branch runs exactly one config'
        return executed[0], result

    @staticmethod
    def _plan_pytest_run(result) -> dict:
        """The ``'__fallback__'`` pytest run recorded on ``VerifyResult.plan``."""
        assert result.plan is not None
        return next(
            r for r in result.plan['runs']
            if r['module_prefix'] == '__fallback__' and r['reason'].startswith('pytest:')
        )

    @pytest.mark.asyncio
    async def test_the_executed_command_is_widened_not_zero_collecting(self, tmp_path: Path):
        """Without the fix this executes 'pytest tests/test_smoke.py' and rc=5 REDs."""
        executed, _ = await self._run(
            tmp_path,
            {'tests/test_smoke.py': _SMOKE_MARKED_SOURCE},
            _SMOKE_DESELECTING_PYPROJECT,
        )
        assert executed.test_command == 'pytest'
        assert executed.prefix == '__fallback__'
        # pytest-only concern: the other slots keep whatever
        # _build_fallback_config rendered for them.
        assert executed.lint_command is not None
        assert 'tests/test_smoke.py' in executed.lint_command
        assert executed.type_check_command is not None
        assert 'tests/test_smoke.py' in executed.type_check_command

    @pytest.mark.asyncio
    async def test_the_recorded_plan_matches_what_executed(self, tmp_path: Path):
        """Both layers must move together, or the plan describes a run that never happened."""
        _, result = await self._run(
            tmp_path,
            {'tests/test_smoke.py': _SMOKE_MARKED_SOURCE},
            _SMOKE_DESELECTING_PYPROJECT,
        )
        run = self._plan_pytest_run(result)
        assert run['scope_kind'] == str(ScopeKind.FULL_SUITE)
        assert run['scoped_targets'] == []
        assert 'tests/test_smoke.py' in run['reason']
        assert 'not smoke' in run['reason']
        assert 'NOT executed' in run['reason']

    @pytest.mark.asyncio
    async def test_an_unmarked_target_still_runs_file_scoped(self, tmp_path: Path):
        """CONTROL: a target that genuinely vanishes must still rc=5 RED.

        Widening only ever on positive proof — never a blanket "file-scoped
        pytest is risky, run everything".
        """
        executed, result = await self._run(
            tmp_path,
            {'tests/test_plain.py': _UNMARKED_PLAIN_SOURCE},
            _SMOKE_DESELECTING_PYPROJECT,
        )
        assert executed.test_command == 'pytest tests/test_plain.py'
        run = self._plan_pytest_run(result)
        assert run['scope_kind'] == str(ScopeKind.FILE_SCOPED)
        assert run['scoped_targets'] == ['tests/test_plain.py']

    @pytest.mark.asyncio
    async def test_a_marked_target_without_addopts_does_not_widen(self, tmp_path: Path):
        """CONTROL: the marker alone proves nothing — the -m expression is the proof."""
        executed, result = await self._run(
            tmp_path, {'tests/test_smoke.py': _SMOKE_MARKED_SOURCE}, _PLAIN_PYPROJECT,
        )
        assert executed.test_command == 'pytest tests/test_smoke.py'
        assert self._plan_pytest_run(result)['scope_kind'] == str(ScopeKind.FILE_SCOPED)

    def test_the_decision_function_itself_stays_pure(self, tmp_path: Path):
        """The fix lives in the EXECUTED layer: derive_verify_plan is unmoved.

        Its raw return value is still the idealized FILE_SCOPED decision — it
        cannot see the rescoping ``_build_fallback_config`` performs, which is
        exactly why the widening cannot live there.
        """
        (tmp_path / 'pyproject.toml').write_text(_SMOKE_DESELECTING_PYPROJECT)
        (tmp_path / 'tests').mkdir()
        (tmp_path / 'tests' / 'test_smoke.py').write_text(_SMOKE_MARKED_SOURCE)

        plan = derive_verify_plan(
            ['tests/test_smoke.py'], [], OrchestratorConfig(project_root=tmp_path),
            _real_worktree_reader(tmp_path),
        )
        run = _run_for(plan, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FILE_SCOPED
        assert run.scoped_targets == ('tests/test_smoke.py',)
