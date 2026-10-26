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

import os
from pathlib import Path

import pytest
from _config_default_scan import DEFAULT_CONFIG, iter_default_asserting_tests
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

    def test_oversized_exponent_is_not_evaluated(self, tmp_path):
        """A pathological power must not stall the whole-suite ratchet."""
        src = _write(tmp_path, 'test_synth_pow.py', '''
from orchestrator.config import OrchestratorConfig


def test_huge_power():
    cfg = OrchestratorConfig()
    assert cfg.starvation_watchdog.skip_threshold == 50 * 2 ** 100000 // 2 ** 100000
''')
        assert list(iter_default_asserting_tests(src)) == [], (
            'a power above the exponent bound must not be evaluated'
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
