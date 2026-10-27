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

* unit tests for the two AST detectors in ``_config_default_scan``, driven over
  a SYNTHETIC corpus written to ``tmp_path`` — never over real suite files, so
  these unit tests do not churn when the suite changes;
* two sweep ratchets, each parametrized per ``test_*.py`` file so a future
  regression names the offending file directly: one bans un-isolated
  default assertions; the other bans blanking or deleting
  ``ORCH_CONFIG_PATH``, which isolates only by accident of cwd, except for the
  tests in ``_ABSENT_CONFIG_PATH_IS_THE_PRECONDITION``, whose every entry is
  itself checked to still name a live site;
* two contract guards pinning the premise every repair rests on (the hatch
  really does yield package defaults, and it is NOT autouse).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from _config_default_scan import (
    DEFAULT_CONFIG,
    ConfigPathWrite,
    iter_config_path_blanking_sites,
    iter_default_asserting_tests,
)
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

from orchestrator.config import OrchestratorConfig

# The ratchet below glob()s every test_*.py directly in tests/ and ast.parse()s each
# one, which makes this module the newest member of the whole-tree-scanner
# family.  Exceeding the 60s pyproject default does NOT merely fail the test:
# pytest-timeout's thread method os._exit()s the whole xdist worker and
# --max-worker-restart=0 then truncates the ENTIRE suite, reporting against
# some innocent guard that shared the dead worker (esc-3980-1, esc-3787-1).
# WHY 300s and the guard that ENFORCES this mark rather than trusting it to be
# sprinkled: see WHOLE_TREE_SCAN_TEST_TIMEOUT in _orch_helpers.py, and
# test_whole_tree_scan_timeout_guard.py (task 4215).
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    """Write a synthetic test source to *tmp_path* and return its path."""
    p = tmp_path / name
    p.write_text(body)
    return p


def _names(results):
    """Reduce detector output to a comparable ``{(class, test)}`` set."""
    return {(cls, test) for cls, test, _lineno, _attr, _lit in results}


#: A frozenset default's members, read live so no case spells them as literals.
_ALLOWLIST_MEMBERS = sorted(DEFAULT_CONFIG.escalation_revalidation_allowlist)


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

    def test_flags_inline_bare_construction(self, tmp_path):
        """``OrchestratorConfig().x.y == literal`` binds no name at all.

        The most common one-line defaults test asserts straight off the
        construction, so the scan must not bail out when no name is bound.
        """
        src = _write(tmp_path, 'test_synth_inline.py', '''
from orchestrator.config import OrchestratorConfig


def test_inline_default():
    assert OrchestratorConfig().starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_inline_default')}, (
            f'expected the inline bare-construction landmine to be flagged; '
            f'got {found!r}'
        )
        _cls, _test, _lineno, attr, literal = found[0]
        assert attr == 'starvation_watchdog.skip_threshold', (
            f'expected the dotted path below the inline call; got {attr!r}'
        )
        assert literal == 50, f'expected the asserted literal; got {literal!r}'

    def test_flags_despite_a_non_autouse_isolating_fixture(self, tmp_path):
        """An isolating fixture isolates only the tests that request it."""
        src = _write(tmp_path, 'test_synth_unrequested_fixture.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


@pytest.fixture
def isolated_config_env(monkeypatch, tmp_path):
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'project.yaml'))


def test_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_default')}, (
            'a non-autouse fixture the test does not request must not shield '
            f'it; got {found!r}'
        )

    def test_flags_despite_an_autouse_fixture_in_a_sibling_class(self, tmp_path):
        """A class-level autouse fixture isolates its own class only."""
        src = _write(tmp_path, 'test_synth_sibling_autouse.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


class A:
    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'project.yaml'))


class B:
    def test_default(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {('B', 'test_default')}, (
            'an autouse fixture in a sibling class must not shield this '
            f'class; got {found!r}'
        )

    @pytest.mark.parametrize('test_def', [
        '''
def test_adhoc(monkeypatch):
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''',
        '''
def test_adhoc(monkeypatch):
    monkeypatch.delenv("ORCH_CONFIG_PATH", raising=False)
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''',
        '''
def test_adhoc():
    with patch.dict(os.environ, {'ORCH_CONFIG_PATH': ''}):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''',
    ], ids=['setenv-empty', 'delenv', 'patch-dict-empty'])
    def test_flags_cwd_dependent_env_idioms(self, tmp_path, test_def):
        """Blanking or deleting the variable is not isolation, because
        orchestrator/src/orchestrator/config.py::OrchestratorConfig.settings_customise_sources
        maps '' and unset alike to the CWD-relative config.yaml."""
        src = _write(tmp_path, 'test_synth_env.py', f'''
import os
from unittest.mock import patch

from orchestrator.config import OrchestratorConfig

{test_def}''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_adhoc')}, (
            f'a blank or delete of ORCH_CONFIG_PATH must not count as '
            f'isolation; got {found!r}'
        )

    @pytest.mark.parametrize('scoped_tests, expected', [
        ('''
class TestScoped:
    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')

    def test_default(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''', ('TestScoped', 'test_default')),
        ('''
@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch):
    monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)


def test_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''', (None, 'test_default')),
    ], ids=['class-autouse-setenv-empty', 'module-autouse-delenv'])
    def test_flags_despite_an_autouse_fixture_that_only_blanks_the_variable(
        self, tmp_path, scoped_tests, expected,
    ):
        """An autouse fixture isolates only by the same rule a test body does."""
        src = _write(tmp_path, 'test_synth_blanking_autouse.py', f'''
import pytest
from orchestrator.config import OrchestratorConfig

{scoped_tests}''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {expected}, (
            'an autouse fixture that only blanks or deletes ORCH_CONFIG_PATH '
            f'must not shield its scope; got {found!r}'
        )

    def test_flags_a_mere_mention_of_the_variable(self, tmp_path):
        """Reading the variable writes nothing, so it isolates nothing."""
        src = _write(tmp_path, 'test_synth_mention.py', '''
import os

from orchestrator.config import OrchestratorConfig


def test_mentions():
    assert 'ORCH_CONFIG_PATH' in os.environ
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_mentions')}, (
            f'a mere mention of ORCH_CONFIG_PATH must not count as isolation; '
            f'got {found!r}'
        )

    def test_flags_despite_a_module_pytestmark_not_naming_the_hatch(
        self, tmp_path,
    ):
        """Only a ``pytestmark`` that names the hatch isolates the module."""
        src = _write(tmp_path, 'test_synth_other_pytestmark.py', '''
import pytest
from orchestrator.config import OrchestratorConfig

pytestmark = pytest.mark.timeout(300)


def test_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_default')}, (
            'a module pytestmark that does not name code_default_config must '
            f'not shield the module; got {found!r}'
        )


class TestDetectorFlagsComputedRightHandSides:
    """A right-hand side that EVALUATES to the default is as much a landmine as
    a literal spelling of it."""

    @pytest.mark.parametrize('rhs', ['5 * 10', '100 // 2', '2 ** 6 - 14'],
                             ids=['mult', 'floordiv', 'pow-sub'])
    def test_flags_constant_arithmetic(self, tmp_path, rhs):
        """Arithmetic over literals, which ``ast.literal_eval`` rejects."""
        src = _write(tmp_path, 'test_synth_arith.py', f'''
from orchestrator.config import OrchestratorConfig


def test_arith_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == {rhs}
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_arith_default')}, (
            f'expected `== {rhs}` to be flagged; got {found!r}'
        )
        value = found[0][4]
        assert value == 50 and type(value) is int, (
            f'expected the evaluated int value 50; got {value!r}'
        )

    def test_flags_local_bound_once_to_a_literal(self, tmp_path):
        """A name bound exactly once to a literal stands for that literal."""
        src = _write(tmp_path, 'test_synth_local.py', '''
from orchestrator.config import OrchestratorConfig


def test_local_default():
    cfg = OrchestratorConfig()
    expected = 50
    assert cfg.starvation_watchdog.skip_threshold == expected
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_local_default')}, (
            f'expected `== expected` bound to the default to be flagged; '
            f'got {found!r}'
        )

    def test_flags_rhs_computed_from_another_attribute_of_the_same_config(
        self, tmp_path,
    ):
        """The task-4481 exemplar: both sides read DIFFERENT undeclared keys."""
        src = _write(tmp_path, 'test_synth_derived.py', '''
from orchestrator.config import OrchestratorConfig


def test_derived_default():
    config = OrchestratorConfig()
    warm_lane_gib = config.git.warm_lane_min_free_gib
    expected = warm_lane_gib * 1024 ** 3
    assert config.no_landings_breaker_disk_free_floor_bytes == expected
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_derived_default')}, (
            f'expected the derived-RHS exemplar to be flagged; got {found!r}'
        )
        _cls, _test, _lineno, attr, value = found[0]
        assert attr == 'no_landings_breaker_disk_free_floor_bytes', (
            f'expected the asserted attribute path; got {attr!r}'
        )
        assert value == DEFAULT_CONFIG.no_landings_breaker_disk_free_floor_bytes, (
            f'expected the RHS valued under package defaults; got {value!r}'
        )

    def test_flags_rhs_rooted_at_an_inline_bare_config(self, tmp_path):
        """An inline bare construction on the RIGHT is a config root too."""
        src = _write(tmp_path, 'test_synth_derived_inline.py', '''
from orchestrator.config import OrchestratorConfig


def test_derived_inline_default():
    cfg = OrchestratorConfig()
    assert cfg.no_landings_breaker_disk_free_floor_bytes == OrchestratorConfig().git.warm_lane_min_free_gib * 1024 ** 3
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_derived_inline_default')}, (
            f'expected an RHS rooted at an inline bare config to be flagged; '
            f'got {found!r}'
        )

    @pytest.mark.parametrize('rhs', [
        'frozenset({' + ', '.join(map(repr, _ALLOWLIST_MEMBERS)) + '})',
        f'frozenset({_ALLOWLIST_MEMBERS!r})',
    ], ids=['set-literal-arg', 'list-arg'])
    def test_flags_collection_constructor_over_a_literal(self, tmp_path, rhs):
        """There is no frozenset literal, so ``frozenset({...})`` is how a
        frozenset default is spelled — and ``ast.literal_eval`` rejects it."""
        src = _write(tmp_path, 'test_synth_ctor.py', f'''
from orchestrator.config import OrchestratorConfig


def test_ctor_default():
    cfg = OrchestratorConfig()
    assert cfg.escalation_revalidation_allowlist == {rhs}
''')
        found = list(iter_default_asserting_tests(src))
        assert _names(found) == {(None, 'test_ctor_default')}, (
            f'expected `== {rhs}` to be flagged; got {found!r}'
        )
        _cls, _test, _lineno, attr, value = found[0]
        assert attr == 'escalation_revalidation_allowlist', (
            f'expected the asserted attribute path; got {attr!r}'
        )
        assert value == DEFAULT_CONFIG.escalation_revalidation_allowlist, (
            f'expected the constructed value; got {value!r}'
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

    @pytest.mark.parametrize('test_def', [
        '''
def test_repointed(monkeypatch, tmp_path):
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'project.yaml'))
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''',
        '''
def test_repointed(tmp_path):
    p = tmp_path / 'project.yaml'
    with patch.dict(os.environ, {'ORCH_CONFIG_PATH': str(p)}):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''',
    ], ids=['setenv-path', 'patch-dict-path'])
    def test_explicit_repoint_of_the_config(self, tmp_path, test_def):
        """A test reading its own project file is not reading the operational one."""
        src = _write(tmp_path, 'test_synth_repoint.py', f'''
import os
from unittest.mock import patch

from orchestrator.config import OrchestratorConfig

{test_def}''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'an explicit re-point of ORCH_CONFIG_PATH must be honoured as '
            'isolation'
        )

    @pytest.mark.parametrize('scoped_tests', [
        '''
pytestmark = pytest.mark.usefixtures("code_default_config")


def test_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''',
        '''
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('code_default_config')]


def test_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''',
        '''
class TestHatched:
    pytestmark = pytest.mark.usefixtures("code_default_config")

    def test_default(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''',
    ], ids=['module', 'module-list', 'class-body'])
    def test_pytestmark_naming_the_hatch(self, tmp_path, scoped_tests):
        """``pytestmark`` applies its marks to every test in its scope."""
        src = _write(tmp_path, 'test_synth_pytestmark.py', f'''
import pytest
from orchestrator.config import OrchestratorConfig

{scoped_tests}''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a pytestmark naming code_default_config must hatch every test in '
            'its scope'
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

    @pytest.mark.parametrize('write, asserted', [
        ('cfg.starvation_watchdog.skip_threshold = 50',
         'cfg.starvation_watchdog.skip_threshold == 50'),
        ('cfg.git.merge_config_only_full_gate_globs[:] = []',
         'cfg.git.merge_config_only_full_gate_globs == []'),
        ('cfg.git.warm_lane_min_free_gib = 7',
         'cfg.no_landings_breaker_disk_free_floor_bytes == '
         'cfg.git.warm_lane_min_free_gib * 1024 ** 3'),
    ], ids=['attribute', 'subscript', 'rhs-attribute'])
    def test_config_written_through_before_the_assertion(
        self, tmp_path, write, asserted,
    ):
        """A config the test itself wrote into no longer reads package defaults."""
        src = _write(tmp_path, 'test_synth_written_through.py', f'''
from orchestrator.config import OrchestratorConfig


def test_written_through():
    cfg = OrchestratorConfig()
    {write}
    assert {asserted}
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'a config written through by `{write}` must not be valued at '
            f'package defaults'
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

    def test_inline_construction_under_the_hatch(self, tmp_path):
        """The inline shape is honoured by the hatch like the bound shape."""
        src = _write(tmp_path, 'test_synth_inline_hatched.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


@pytest.mark.usefixtures("code_default_config")
def test_inline_default():
    assert OrchestratorConfig().starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'an inline bare construction under the sanctioned hatch must be '
            'honoured'
        )

    def test_inline_construction_with_kwargs(self, tmp_path):
        """Only a BARE inline call is a root; any other Call root is not."""
        src = _write(tmp_path, 'test_synth_inline_kwargs.py', '''
from orchestrator.config import GitConfig, OrchestratorConfig


def test_inline_explicit():
    assert OrchestratorConfig(git=GitConfig(main_branch='main')).git.merge_config_only_full_gate_globs == []
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'an inline config constructed with explicit kwargs is not a '
            'bare-default landmine'
        )

    def test_class_level_autouse_fixture_repointing_the_config(self, tmp_path):
        """An autouse fixture in the test's class isolates like the test body."""
        src = _write(tmp_path, 'test_synth_class_autouse.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


class TestIsolated:
    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch, tmp_path):
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'project.yaml'))

    def test_default(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a class-level autouse fixture re-pointing ORCH_CONFIG_PATH must '
            'be honoured as isolation'
        )

    def test_module_level_autouse_fixture_repointing_the_config(self, tmp_path):
        """An autouse fixture at module scope isolates every test in the file."""
        src = _write(tmp_path, 'test_synth_module_autouse.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch, tmp_path):
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'project.yaml'))


def test_default():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a module-level autouse fixture re-pointing ORCH_CONFIG_PATH must '
            'be honoured as isolation'
        )

    def test_autouse_fixture_requesting_the_hatch(self, tmp_path):
        """The ``test_verify_load_stamp.py`` shape: autouse requests the hatch."""
        src = _write(tmp_path, 'test_synth_autouse_hatch.py', '''
import pytest
from orchestrator.config import OrchestratorConfig


class TestPinned:
    @pytest.fixture(autouse=True)
    def _pin(self, code_default_config):
        pass

    def test_default(self):
        cfg = OrchestratorConfig()
        assert cfg.starvation_watchdog.skip_threshold == 50
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'an autouse fixture requesting code_default_config must be '
            'honoured as isolation'
        )


class TestDetectorRefusesUnsoundComputedRightHandSides:
    """A committed ratchet may admit false negatives; it may not admit false positives."""

    def test_rhs_reading_the_asserted_path_holds_under_any_declaration(self, tmp_path):
        """Both sides track a declaration of the same key, so neither is a landmine."""
        src = _write(tmp_path, 'test_synth_overlap.py', '''
from orchestrator.config import OrchestratorConfig


def test_same_key_both_sides():
    cfg = OrchestratorConfig()
    other = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == other.starvation_watchdog.skip_threshold
    assert cfg.git.warm_lane_min_free_gib == OrchestratorConfig().git.warm_lane_min_free_gib * 1
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'an RHS reading the asserted path itself must not be flagged'
        )

    @pytest.mark.parametrize('rhs', [
        '50 * 2 ** 100000 // 2 ** 100000',
        '50 * (10 ** 64) ** 64 // (10 ** 64) ** 64',
    ], ids=['huge-exponent', 'huge-base'])
    def test_oversized_power_is_not_evaluated(self, tmp_path, rhs):
        """A pathological power must not stall the whole-suite ratchet.

        Bounding the exponent alone is not enough: each nested power
        multiplies the width of an already-large base again.
        """
        src = _write(tmp_path, 'test_synth_pow.py', f'''
from orchestrator.config import OrchestratorConfig


def test_huge_power():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == {rhs}
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'`{rhs}` computes an oversized int and must not be evaluated'
        )

    @pytest.mark.parametrize('bindings, name', [
        ('x = x + 1', 'x'),
        ('a = b\n    b = a', 'a'),
    ], ids=['self-referential', 'mutually-referential'])
    def test_cyclic_local_is_not_resolved(self, tmp_path, bindings, name):
        """A local defined in terms of itself has no value to give, and must
        not recurse the scan to death trying."""
        src = _write(tmp_path, 'test_synth_cycle.py', f'''
from orchestrator.config import OrchestratorConfig


def test_cycle():
    cfg = OrchestratorConfig()
    {bindings}
    assert cfg.starvation_watchdog.skip_threshold == {name}
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'a cyclic local `{bindings!r}` must not be resolved'
        )

    @pytest.mark.parametrize('nested_scope, nested_binding, asserted', [
        ('def helper():', 'EXPECTED = 50',
         'OrchestratorConfig().starvation_watchdog.skip_threshold == EXPECTED'),
        ('class Helper:', 'EXPECTED = 50',
         'OrchestratorConfig().starvation_watchdog.skip_threshold == EXPECTED'),
        ('def helper():', 'cfg = OrchestratorConfig()',
         'cfg.starvation_watchdog.skip_threshold == 50'),
    ], ids=['local-in-def', 'local-in-class', 'bare-config-in-def'])
    def test_binding_in_a_nested_scope_is_not_the_tests_own(
        self, tmp_path, nested_scope, nested_binding, asserted,
    ):
        """The test reads the MODULE global; the nested binding is another
        variable that merely shares its name."""
        src = _write(tmp_path, 'test_synth_nested_scope.py', f'''
from orchestrator.config import OrchestratorConfig, StarvationWatchdogConfig

EXPECTED = OrchestratorConfig().starvation_watchdog.skip_threshold
cfg = OrchestratorConfig(
    starvation_watchdog=StarvationWatchdogConfig(skip_threshold=50),
)


def test_reads_module_globals():
    {nested_scope}
        {nested_binding}
    assert {asserted}
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'`{nested_binding}` inside `{nested_scope}` must not stand for '
            f'the module global the test reads'
        )

    def test_name_reached_mutable_value_is_not_resolved(self, tmp_path):
        """The scan cannot see a mutation between binding and assertion."""
        src = _write(tmp_path, 'test_synth_mutated.py', '''
from orchestrator.config import OrchestratorConfig


def test_mutated_local():
    cfg = OrchestratorConfig()
    expected = []
    expected.append('*.yaml')
    assert cfg.git.merge_config_only_full_gate_globs == expected
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a mutable value reached through a name must not be resolved'
        )

    def test_rebound_local_is_not_resolved(self, tmp_path):
        """The scan cannot know which of two bindings the assertion sees."""
        src = _write(tmp_path, 'test_synth_rebound.py', '''
from orchestrator.config import OrchestratorConfig, load_config


def test_rebound(config_path):
    cfg = OrchestratorConfig()
    expected = 50
    expected = load_config(config_path).starvation_watchdog.skip_threshold
    assert cfg.starvation_watchdog.skip_threshold == expected
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a local bound more than once must not be resolved'
        )

    def test_rhs_rooted_at_a_non_bare_config(self, tmp_path):
        """Only a bare config's attributes are valued under package defaults."""
        src = _write(tmp_path, 'test_synth_loaded_rhs.py', '''
from orchestrator.config import OrchestratorConfig, load_config


def test_loaded_rhs(config_path):
    cfg = OrchestratorConfig()
    other = load_config(config_path)
    assert cfg.no_landings_breaker_disk_free_floor_bytes == other.git.warm_lane_min_free_gib * 1024 ** 3
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'an RHS rooted at a non-bare config must not be resolved'
        )

    @pytest.mark.parametrize('rhs', ['25 * 2.0', '100 / 2', 'max(50, 0)'],
                             ids=['float-product', 'true-division', 'call'])
    def test_unevaluable_or_type_mismatched_rhs(self, tmp_path, rhs):
        """A float result never matches an int default; a call is not evaluated."""
        src = _write(tmp_path, 'test_synth_mismatch.py', f'''
from orchestrator.config import OrchestratorConfig


def test_mismatch():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == {rhs}
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            f'`== {rhs}` must not be flagged against an int default'
        )

    def test_constructor_over_an_unresolvable_argument(self, tmp_path):
        """A constructor is valued only when its argument is."""
        src = _write(tmp_path, 'test_synth_ctor_call.py', '''
from orchestrator.config import OrchestratorConfig


def test_ctor_over_call():
    cfg = OrchestratorConfig()
    assert cfg.escalation_revalidation_allowlist == frozenset(load_categories())
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a constructor over an unevaluable argument must not be resolved'
        )

    def test_shadowed_constructor_name(self, tmp_path):
        """A locally bound ``frozenset`` is not the builtin."""
        src = _write(tmp_path, 'test_synth_ctor_shadowed.py', f'''
from orchestrator.config import OrchestratorConfig


def test_shadowed_ctor(make_allowlist):
    frozenset = make_allowlist
    cfg = OrchestratorConfig()
    assert cfg.escalation_revalidation_allowlist == frozenset({_ALLOWLIST_MEMBERS!r})
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a shadowed constructor name must not be treated as the builtin'
        )


# ---------------------------------------------------------------------------
# BLANKING-SITE DETECTOR
# ---------------------------------------------------------------------------

#: One blank or delete in each kind of scope; each site's line is tagged
#: ``# site: <qualname>`` so the expected line numbers are read, not counted.
_BLANKING_SITES_MODULE = '''
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def module_fixture(monkeypatch):
    monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)  # site: module_fixture


def _helper(monkeypatch):
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')  # site: _helper


def test_module_level():
    with patch.dict(os.environ, {'ORCH_CONFIG_PATH': ''}):  # site: test_module_level
        pass


class TestX:
    @pytest.fixture(autouse=True)
    def _isolated(self, monkeypatch):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')  # site: TestX._isolated

    def test_y(self, monkeypatch):
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)  # site: TestX.test_y
'''


def _tagged_line(source: str, qualname: str) -> int:
    """1-indexed line of *source* tagged ``# site: <qualname>``."""
    tag = f'# site: {qualname}'
    return next(
        lineno for lineno, line in enumerate(source.splitlines(), start=1)
        if line.endswith(tag)
    )


class TestBlankingSiteDetector:
    """``iter_config_path_blanking_sites`` reports every blank or delete of
    ``ORCH_CONFIG_PATH``, in whatever scope it sits, and nothing else."""

    def test_yields_every_blank_and_delete_with_its_scope(self, tmp_path):
        src = _write(tmp_path, 'test_synth_sites.py', _BLANKING_SITES_MODULE)
        expected = [
            (qualname, _tagged_line(_BLANKING_SITES_MODULE, qualname), kind)
            for qualname, kind in [
                ('module_fixture', ConfigPathWrite.DELETE),
                ('_helper', ConfigPathWrite.BLANK),
                ('test_module_level', ConfigPathWrite.BLANK),
                ('TestX._isolated', ConfigPathWrite.BLANK),
                ('TestX.test_y', ConfigPathWrite.DELETE),
            ]
        ]
        found = sorted(iter_config_path_blanking_sites(src), key=lambda s: s[1])
        assert found == expected, (
            f'expected one (qualname, line, kind) per blanking site; got {found!r}'
        )

    @pytest.mark.parametrize('stmt', [
        "monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))",
        "with patch.dict(os.environ, {'ORCH_CONFIG_PATH': str(p)}):\n        pass",
        "fake_environ = {'ORCH_CONFIG_PATH': ''}\n"
        "    monkeypatch.setattr(verify.os, 'environ', fake_environ)",
        "assert 'ORCH_CONFIG_PATH' not in os.environ",
        "assert os.environ.get('ORCH_CONFIG_PATH', '') == ''",
    ], ids=['setenv-path', 'patch-dict-path', 'subprocess-env-dict',
            'membership-read', 'get-read'])
    def test_yields_nothing_for_a_non_blanking_mention(self, tmp_path, stmt):
        src = _write(tmp_path, 'test_synth_non_sites.py', f'''
import os
from unittest.mock import patch

from orchestrator import verify


def test_z(monkeypatch, tmp_path):
    p = tmp_path / 'project.yaml'
    {stmt}
''')
        found = list(iter_config_path_blanking_sites(src))
        assert found == [], (
            f'`{stmt}` neither blanks nor deletes ORCH_CONFIG_PATH; got {found!r}'
        )


# ---------------------------------------------------------------------------
# THE RATCHETS
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
        f"Blanking ORCH_CONFIG_PATH with setenv('') or removing it with delenv "
        f'does not count as isolation: both fall back to a cwd-relative '
        f'config.yaml.\n'
        f'Remedy: add @pytest.mark.usefixtures("code_default_config") to the '
        f'test (or to its class, if every test in the class is defaults-only), '
        f'so the assertion means what the test name already claims.'
    )


#: ``(file name, qualname)`` of the tests whose precondition IS an absent
#: ``ORCH_CONFIG_PATH`` — each checks what happens when no config is named at
#: all, which no re-point to any path could express.
_ABSENT_CONFIG_PATH_IS_THE_PRECONDITION = frozenset({
    ('test_config.py',
     'TestLayeredConfig.test_load_config_raises_when_no_config_and_no_env'),
    ('test_flake_report.py',
     'TestFlakeLedgerCommand.test_no_config_and_no_env_exits_one_with_an_error'),
    ('test_eval_task_sampler.py',
     'TestEvalListFixturesCLI.test_needs_no_orchestrator_config'),
})


@pytest.mark.parametrize('source', _SUITE_FILES, ids=lambda p: p.name)
def test_no_cwd_dependent_config_isolation(source):
    """No test may blank or delete ``ORCH_CONFIG_PATH`` to isolate itself."""
    sites = [
        (qualname, lineno, kind)
        for qualname, lineno, kind in iter_config_path_blanking_sites(source)
        if (source.name, qualname) not in _ABSENT_CONFIG_PATH_IS_THE_PRECONDITION
    ]
    if not sites:
        return
    lines = '\n'.join(
        f'  {qualname}  line {lineno}  {kind.name}'
        for qualname, lineno, kind in sites
    )
    pytest.fail(
        f'{source.name}: {len(sites)} site(s) blank or delete ORCH_CONFIG_PATH:\n'
        f'{lines}\n\n'
        f'orchestrator/src/orchestrator/config.py::OrchestratorConfig.'
        f'settings_customise_sources maps an empty and an unset ORCH_CONFIG_PATH '
        f'alike to a config.yaml relative to the process cwd, so these isolate '
        f'only while no such file sits there.\n'
        f'Remedy: add @pytest.mark.usefixtures("code_default_config") to the '
        f'test and delete the line; or, if the test only loads an explicit path '
        f'through load_config/census_config_keys, just delete the line; or, if '
        f'the absent variable IS the precondition under test, add an entry to '
        f'_ABSENT_CONFIG_PATH_IS_THE_PRECONDITION.'
    )


_EXEMPTIONS = sorted(_ABSENT_CONFIG_PATH_IS_THE_PRECONDITION)


@pytest.mark.parametrize('file_name, qualname', _EXEMPTIONS,
                         ids=[f'{f}::{q}' for f, q in _EXEMPTIONS])
def test_every_exemption_still_names_a_blanking_site(file_name, qualname):
    """A renamed or migrated exemption must not linger as a silent blind spot."""
    source = Path(__file__).parent / file_name
    assert source.is_file(), f'exempted file {file_name} no longer exists'
    qualnames = {site[0] for site in iter_config_path_blanking_sites(source)}
    assert qualname in qualnames, (
        f'{file_name}::{qualname} is exempted from '
        f'test_no_cwd_dependent_config_isolation but no longer blanks or '
        f'deletes ORCH_CONFIG_PATH; drop its entry from '
        f'_ABSENT_CONFIG_PATH_IS_THE_PRECONDITION'
    )


# ---------------------------------------------------------------------------
# CONTRACT GUARDS
#
# The two premises every repair in this sweep rests on.  Both are ratchets:
# green on arrival, red only if a future change breaks the contract.  Both are
# behavioural — nothing here asserts on docstring wording, comments or prose.
# ---------------------------------------------------------------------------

#: A probe leaf for the efficacy guard.  Its package default is read live from
#: DEFAULT_CONFIG rather than spelled as a literal, so this guard cannot itself
#: become the landmine it exists to prevent.
_PROBE_SECTION = 'delivered_checks'
_PROBE_LEAF = 'max_checks_per_tick'


def _probe_default() -> int:
    return getattr(getattr(DEFAULT_CONFIG, _PROBE_SECTION), _PROBE_LEAF)


def _probe_value() -> int:
    """Read the probe leaf off a freshly constructed bare config."""
    return getattr(getattr(OrchestratorConfig(), _PROBE_SECTION), _PROBE_LEAF)


@pytest.fixture
def declared_probe_config(monkeypatch, tmp_path) -> Path:
    """Point ORCH_CONFIG_PATH at a yaml DECLARING the probe leaf off-default.

    Returns the yaml's path.  Overrides the autouse ``_isolate_orch_config``
    (which pins the operational dark-factory-orchestrator.yaml); a test that
    also requests ``code_default_config`` must list THIS fixture first so the
    hatch is the last writer of the env var.
    """
    path = tmp_path / 'declared-orchestrator.yaml'
    path.write_text(
        f'{_PROBE_SECTION}:\n  {_PROBE_LEAF}: {_probe_default() + 7}\n'
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(path))
    return path


class TestHatchEfficacy:
    """The hatch really does swap a DECLARED value for the package default.

    This is the load-bearing premise of every repair in task 4481: the
    ``@pytest.mark.usefixtures("code_default_config")`` sites are a repair only
    if the fixture actually defeats the project config layer.  Nothing else in
    the suite pins it.  Measured during planning on three probe keys —
    operational, then declared-off-default, then hatched back to the default.
    """

    def test_a_declared_value_wins_without_the_hatch(self, declared_probe_config):
        """Control leg — proves the probe yaml is actually being read.

        Without it the hatched leg below would pass vacuously if the
        declaration were silently ignored.
        """
        declared = _probe_default() + 7
        assert declared != _probe_default()
        assert _probe_value() == declared, (
            f'a bare OrchestratorConfig() must read the DECLARED value '
            f'{declared} from the project config layer; got {_probe_value()}'
        )

    def test_the_hatch_defeats_a_declared_value(
        self, declared_probe_config, code_default_config,
    ):
        """The hatched leg: the same construction reads the PACKAGE default.

        ``declared_probe_config`` is requested first so ``code_default_config``
        is the last writer of ORCH_CONFIG_PATH.  That ordering is asserted
        outright rather than assumed, so a future change in fixture
        instantiation order fails loudly instead of making this guard vacuous.
        """
        in_effect = Path(os.environ['ORCH_CONFIG_PATH'])
        assert in_effect != declared_probe_config, (
            'fixture ordering regression: code_default_config must be the last '
            'writer of ORCH_CONFIG_PATH, but the declared probe yaml is the '
            'path in effect'
        )
        assert _probe_value() == _probe_default(), (
            f'code_default_config must restore the PACKAGE default '
            f'{_probe_default()}; got {_probe_value()} while '
            f'{declared_probe_config} declared {_probe_default() + 7} — the '
            f'hatch is not defeating the project config layer, so every '
            f'@pytest.mark.usefixtures("code_default_config") in this suite '
            f'is a no-op'
        )


class TestHatchIsNotAutouse:
    """``code_default_config`` must stay OPT-IN.

    Making it autouse is the inversion task 3886 rejected on the record: it
    would strip the operational config from the thousands of tests that
    legitimately read it, turning this sweep's premise into its opposite.
    Asserted off what pytest actually resolves for a test that does NOT ask
    for the hatch, not off prose or pytest internals.
    """

    def test_a_test_that_does_not_ask_for_the_hatch_does_not_get_it(self, request):
        assert 'code_default_config' not in request.fixturenames, (
            'code_default_config must never be autouse — it would re-point '
            'ORCH_CONFIG_PATH at an absent file for the whole suite, stripping '
            'the operational config from every test that legitimately reads it'
        )
        in_effect = Path(os.environ['ORCH_CONFIG_PATH'])
        assert in_effect.name == 'dark-factory-orchestrator.yaml' and in_effect.is_file(), (
            f'an un-hatched test must read the operational config; '
            f'ORCH_CONFIG_PATH is {in_effect}'
        )
