"""Guard tests: no orchestrator test may assert a CODE DEFAULT off an
un-isolated bare ``OrchestratorConfig()``.

Incident (task 3886). ``test_verify.py::TestMergeConfigOnlyDiffForcesFullGate::
test_empty_globs_default_config_returns_false`` asserted
``config.git.merge_config_only_full_gate_globs == []`` against a bare
``OrchestratorConfig()``.  The autouse ``_isolate_orch_config`` fixture
(``conftest.py``) pins ``ORCH_CONFIG_PATH`` at the *operational*
``dark-factory-orchestrator.yaml``, so that assertion passed only because the
repo happens NOT to declare that key.  The moment an operator declares it the
test goes red — a landmine armed by an ops edit, in a file nobody thought to
pre-check, failing a test that was never about that operator's change.

The sanctioned defusal is the opt-in ``code_default_config`` fixture, which
re-points ``ORCH_CONFIG_PATH`` at a guaranteed-absent file so only the
package-bundled ``defaults.yaml`` loads.  Applied as
``@pytest.mark.usefixtures("code_default_config")``, it makes the assertion
mean what the test's own name already claims.

This module holds three things:

* unit tests for the AST detector in ``_config_default_scan``, driven over a
  SYNTHETIC corpus written to ``tmp_path`` — never over real suite files, so
  these unit tests do not churn when the suite changes;
* the sweep ratchet itself, parametrized per ``test_*.py`` file, so a future
  regression names the offending file directly;
* two contract guards pinning the premise every repair rests on (the hatch
  really does yield package defaults, and it is NOT autouse).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from _config_default_scan import iter_default_asserting_tests


def _write(tmp_path: Path, name: str, body: str) -> Path:
    """Write a synthetic test source to *tmp_path* and return its path."""
    p = tmp_path / name
    p.write_text(body)
    return p


def _names(results):
    """Reduce detector output to a comparable ``{(class, test)}`` set."""
    return {(cls, test) for cls, test, _lineno, _attr, _lit in results}


# ---------------------------------------------------------------------------
# MUST FLAG
# ---------------------------------------------------------------------------

class TestDetectorFlagsLandmines:
    """Shapes the detector is required to report."""

    def test_flags_int_default_equality(self, tmp_path):
        """``cfg.starvation_watchdog.skip_threshold == 50`` is the int shape."""
        src = _write(tmp_path, 'test_synth_int.py', '''
from orchestrator.config import OrchestratorConfig


def test_int_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_int_default')}, (
            f'expected the int-default landmine to be flagged; got {found!r}'
        )
        _cls, _test, lineno, attr, literal = found[0]
        assert attr == 'starvation_watchdog.skip_threshold', (
            f'expected the dotted attribute path; got {attr!r}'
        )
        assert literal == 50, f'expected the asserted literal; got {literal!r}'
        assert lineno > 0, f'expected a 1-indexed assertion line; got {lineno!r}'

    def test_flags_bool_default_via_is(self, tmp_path):
        """``is True`` is as much a default assertion as ``== True``."""
        src = _write(tmp_path, 'test_synth_bool.py', '''
from orchestrator.config import OrchestratorConfig


class TestBool:
    def test_bool_default(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.enabled is True
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {('TestBool', 'test_bool_default')}, (
            f'expected the bool-default landmine to be flagged; got {found!r}'
        )
        assert found[0][4] is True, f'expected literal True; got {found[0][4]!r}'

    def test_flags_empty_list_default(self, tmp_path):
        """The task-3886 exemplar shape: an empty-list default with a message."""
        src = _write(tmp_path, 'test_synth_list.py', '''
from orchestrator.config import OrchestratorConfig


def test_list_default():
    config = OrchestratorConfig()
    assert config.git.merge_config_only_full_gate_globs == [], (
        'globs default empty'
    )
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_list_default')}, (
            f'expected the exemplar list shape to be flagged; got {found!r}'
        )
        assert found[0][3] == 'git.merge_config_only_full_gate_globs', (
            f'expected the exemplar attribute path; got {found[0][3]!r}'
        )

    def test_flags_unhatched_sibling_of_a_hatched_test(self, tmp_path):
        """The unit is per-TEST, not per-FILE.

        A file that mentions ``code_default_config`` somewhere still harbours a
        landmine in any *other* test.  This is exactly the exclusion that made
        the task's own file-level candidate set drop ``test_config.py`` and
        ``test_verify.py`` — the two largest concentrations.
        """
        src = _write(tmp_path, 'test_synth_mixed.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


@pytest.mark.usefixtures("code_default_config")
def test_hatched():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50


def test_unhatched_sibling():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_unhatched_sibling')}, (
            'a hatched test elsewhere in the file must not shield an unhatched '
            f'sibling; got {found!r}'
        )


# ---------------------------------------------------------------------------
# MUST NOT FLAG
# ---------------------------------------------------------------------------

class TestDetectorHonoursIsolation:
    """Shapes the detector must stay silent about.

    A committed ratchet may admit false negatives; it may not admit false
    positives.
    """

    @pytest.mark.parametrize('mark', [
        '@pytest.mark.usefixtures("code_default_config")',
        "@pytest.mark.usefixtures('code_default_config')",
    ], ids=['double-quoted', 'single-quoted'])
    def test_per_test_usefixtures_mark(self, tmp_path, mark):
        """The sanctioned hatch, in either quote spelling."""
        src = _write(tmp_path, 'test_synth_mark.py', f'''
import pytest
from orchestrator.config import OrchestratorConfig


{mark}
def test_hatched():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'the sanctioned hatch spelled {mark!r} must be honoured'
        )

    def test_class_level_usefixtures_mark(self, tmp_path):
        """A class-level mark hatches every method in the class."""
        src = _write(tmp_path, 'test_synth_classmark.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


@pytest.mark.usefixtures("code_default_config")
class TestHatched:
    def test_one(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50

    def test_two(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.enabled is True
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a class-level usefixtures mark must hatch every method in the class'
        )

    def test_direct_fixture_argument(self, tmp_path):
        """Requesting the fixture by name is equivalent to the mark."""
        src = _write(tmp_path, 'test_synth_arg.py', '''
from orchestrator.config import OrchestratorConfig


def test_hatched(code_default_config):
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a direct `code_default_config` fixture argument must be honoured'
        )

    @pytest.mark.parametrize('stmt', [
        "monkeypatch.setenv('ORCH_CONFIG_PATH', '')",
        'monkeypatch.delenv("ORCH_CONFIG_PATH", raising=False)',
    ], ids=['setenv-empty', 'delenv'])
    def test_adhoc_env_idioms(self, tmp_path, stmt):
        """The two ad-hoc idioms already widespread in the suite count as
        isolation; this sweep deliberately does not migrate them."""
        src = _write(tmp_path, 'test_synth_env.py', f'''
from orchestrator.config import OrchestratorConfig


def test_adhoc(monkeypatch):
    {stmt}
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'the ad-hoc idiom {stmt!r} must be honoured as isolation'
        )

    def test_config_built_with_kwargs(self, tmp_path):
        """Only a BARE ``OrchestratorConfig()`` inherits the ambient config."""
        src = _write(tmp_path, 'test_synth_kwargs.py', '''
from orchestrator.config import GitConfig, OrchestratorConfig


def test_explicit():
    cfg = OrchestratorConfig(git=GitConfig(main_branch='main'))
    assert cfg.git.merge_config_only_full_gate_globs == []
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a config constructed with explicit kwargs is not a bare-default '
            'landmine'
        )

    def test_literal_differing_from_the_default(self, tmp_path):
        """An assertion that does not match the package default is not a
        default assertion at all."""
        src = _write(tmp_path, 'test_synth_other.py', '''
from orchestrator.config import OrchestratorConfig


def test_not_a_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 9999
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'only literals equal to the package default may be flagged'
        )


# ---------------------------------------------------------------------------
# THE RATCHET
# ---------------------------------------------------------------------------

#: One param per FILE, not one per suite, so a regression names the offending
#: file directly and each repair turns a specific subset green.  Collected by
#: glob so a newly added test module is covered automatically.
_SUITE_FILES = sorted(Path(__file__).parent.glob('test_*.py'))


@pytest.mark.parametrize('source', _SUITE_FILES, ids=lambda p: p.name)
def test_no_unhatched_code_default_assertions(source):
    """No test may assert a package default off an un-isolated bare config.

    See this module's docstring for the incident.  The remedy is one line.
    """
    findings = list(iter_default_asserting_tests(source))
    if not findings:
        return
    lines = '\n'.join(
        f'  {cls or "<module>"}::{test}  line {lineno}  '
        f'{attr} == {literal!r}'
        for cls, test, lineno, attr, literal in findings
    )
    tests = {(cls, test) for cls, test, _, _, _ in findings}
    pytest.fail(
        f'{source.name}: {len(tests)} test(s) / {len(findings)} assertion(s) '
        f'assert a CODE DEFAULT off a bare OrchestratorConfig() that is NOT '
        f'isolated from the operational dark-factory-orchestrator.yaml:\n'
        f'{lines}\n\n'
        f'Each of these passes only because the repo does not currently '
        f'declare that key. Declaring it — an ordinary ops edit — turns the '
        f'test red for a reason that has nothing to do with the change.\n'
        f'Remedy: add @pytest.mark.usefixtures("code_default_config") to the '
        f'test (or to its class, if every test in the class is defaults-only), '
        f'so the assertion means what the test name already claims.'
    )
