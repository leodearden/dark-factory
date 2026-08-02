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

This module unit-tests the pure static detector (``orchestrator.pytest_markers``)
against synthetic strings, then pins the wired ``derive_verify_plan`` behaviour —
including the real-config incident golden — at the end.
"""
from __future__ import annotations

import pytest

from orchestrator.pytest_markers import (
    deselecting_expression_for_targets,
    expression_definitely_deselects,
    module_level_marker_names,
    resolve_marker_expression,
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
