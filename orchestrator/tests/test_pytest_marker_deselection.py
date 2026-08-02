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

from orchestrator.pytest_markers import resolve_marker_expression

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
