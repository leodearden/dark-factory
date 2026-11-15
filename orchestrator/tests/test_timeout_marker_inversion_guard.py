"""Guard: no ``@pytest.mark.timeout(N)`` may INVERT into a tighter clamp under verify.

THE RULE, in one line: a marker at ``DELIBERATE_TIGHT_BOUND_CEILING < N <
VERIFY_CLI_PER_TEST_TIMEOUT`` is too large to read as a deliberate tight bound
and still sits below the budget verify passes, so under verify it grants LESS
than no marker at all -- a slow test's opt-out inverted into a clamp on the
run that gates the merge, because a pytest-timeout marker is a two-way
override and never a floor.

The derivation -- the ``pytest_timeout._get_item_settings`` precedence it
follows from and the three regimes the two budgets carve out -- has ONE home:
the module docstring of shared/src/shared/testing_timeout_markers.py.  Read it
there.  This module points at it rather than restating it, so a pytest-timeout
upgrade that moves that precedence invalidates one copy and not five.  The sole
deliberate exception is the failure text ``judge_inversion_band`` renders,
where the reader is looking at a traceback and not at the source.

A SWEEP WITH NO ALLOWLIST: every in-band marker under this directory fails.

SCOPE.  The extractor (``timeout_marker_sites``), the band (``inverts``) and
the judgement with its messages (``judge_inversion_band``) live in
``shared.testing_timeout_markers``.  This module instantiates them for
``orchestrator/tests`` with an empty allowlist and adds the orchestrator-only
pins: the sanctioned-name mirrors, the deep-gate and deep-landing spellings,
and the spawn budgets.  fused-memory, shared and escalation instantiate the
same guard in their own ``tests/test_timeout_marker_inversion_guard.py``, and
tests/scripts and scripts/tests do too, in their own directory (task 6338).

WHAT THIS DOES NOT DUPLICATE.  test_whole_tree_scan_timeout_guard.py polices a
per-FILE family invariant using MODULE-LEVEL marks only; test_marker_
registration_drift.py sweeps marker NAMES and never argument values;
tests/scripts/test_fallback_verify_config.py pins the CONFIG side (that every
pytest segment really does carry ``--timeout``).  Nothing before this checked a
per-test marker's VALUE against the verify budget.

Task 5147.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import _orch_helpers
import pytest
import yaml
from _orch_helpers import (
    ORCH_YAML,
    PYPROJECT_DEFAULT_TIMEOUT,
    VERIFY_CLI_PER_TEST_TIMEOUT,
)
from _sanctioned_timeout_names import (
    SANCTIONED_NAMES_DEFINED_ELSEWHERE,
    SANCTIONED_TIMEOUT_NAMES,
)
from _tree_scan import (
    ORCHESTRATOR_TESTS,
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
)
from shared.testing_timeout_markers import (
    DELIBERATE_TIGHT_BOUND_CEILING,
    InversionVerdict,
    SweepFloors,
    TimeoutSite,
    TreeScan,
    judge_inversion_band,
    timeout_marker_sites,
    verify_cli_timeout,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
# WHOLE_TREE_SCAN_TEST_TIMEOUT sits at or above the inversion band's OPEN upper
# edge, VERIFY_CLI_PER_TEST_TIMEOUT, so this module does not flag its own mark.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]


def _verify_test_command() -> str:
    """The ``test_command`` merge-verify runs for this package, from :data:`ORCH_YAML`."""
    return yaml.safe_load(ORCH_YAML.read_text(encoding='utf-8'))['test_command']

#: The module task 5333 sizes and ratchets: eight real-git classes that all
#: carried an identical bare ``300`` regardless of weight, which is what made
#: TestRow7KillSwitchByteIdentity's under-sizing invisible.
_DEEP_GATE_MODULE = 'test_merge_queue_deep_integration_gate.py'

#: The only spellings a timeout marker in :data:`_DEEP_GATE_MODULE` may use.
#: Two, not one, because the asymmetry is DELIBERATE: Row 7's marker is
#: derived from its own measured cost, and its neighbours stay at the verify
#: CLI budget because none of them has been measured.
_DEEP_GATE_SPELLINGS = frozenset({
    'VERIFY_CLI_PER_TEST_TIMEOUT',
    'DEEP_GATE_SCENE_TEST_TIMEOUT',
})

#: Non-vacuity floor for the sweep of that module: it had eight real-git
#: classes when the ratchet was written, and a sweep that found none would
#: pass by finding nothing rather than by finding nothing wrong.
_MIN_DEEP_GATE_MARKER_SITES = 8


# ---------------------------------------------------------------------------
# The REAL definitions behind SANCTIONED_TIMEOUT_NAMES.
#
# The map is a mirror; these read the originals, so the mirror can be checked
# rather than trusted.  See the comment on
# _sanctioned_timeout_names.py::SANCTIONED_TIMEOUT_NAMES for the two silent
# failures this closes.
# ---------------------------------------------------------------------------

#: Numbers a sanctioned constant's definition may be written in terms of.  All
#: four real definitions are literals or literal arithmetic over _orch_helpers'
#: own numeric globals, so ONE namespace resolves every case with no test module
#: imported.
#: Importing one for a constant would be a bad trade: test modules run code at
#: import, and test_warm_lane_bash_suite.py asserts at module scope right below
#: the PYTEST_TIMEOUT this map mirrors.
#:
#: STATED LIMIT: a definition written over a FILE-LOCAL name that shadows an
#: _orch_helpers global of the same spelling would resolve against the wrong
#: one.  No such shadow exists (all four definitions were read and resolved),
#: and the check is a floor, not a proof -- so this is recorded rather than
#: engineered around.
_MIRROR_NAMESPACE: Mapping[str, float] = {
    name: float(value)
    for name, value in vars(_orch_helpers).items()
    if isinstance(value, int | float) and not isinstance(value, bool)
}


class _Binding(NamedTuple):
    """One assignment of a :data:`SANCTIONED_TIMEOUT_NAMES` name found in the tree.

    ``seconds`` is None when the right-hand side is outside
    :func:`_binding_value`'s grammar -- "no longer statically verifiable",
    which :class:`TestSanctionedNameMirrors` reports rather than passes over.
    ``expression`` is the unparsed source, so a failure can show what it read.
    """

    module: str
    name: str
    lineno: int
    expression: str
    seconds: float | None


def _binding_value(node: ast.expr, namespace: Mapping[str, float]) -> float | None:
    """*node*'s value, if it is literal arithmetic over *namespace*'s numbers.

    Deliberately tiny: numeric literals, names from *namespace*, and ``+ - *``
    over them is the entire grammar the four real definitions use.  ``bool`` is
    excluded from the literal case for the same reason
    ``shared.testing_timeout_markers.timeout_marker_sites`` excludes it.  Anything else -- a call, an attribute, a name this namespace
    does not carry -- yields None.

    OPPOSITE POLARITY TO the marker census's resolution, deliberately.  There, None
    means "no opinion" and is waved through; here it means the mirror can no
    longer be checked against its original, which is a finding, not a pass.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value)
        return None
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.BinOp):
        left = _binding_value(node.left, namespace)
        right = _binding_value(node.right, namespace)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def _sanctioned_bindings_in(tree: ast.Module, module: str) -> list[_Binding]:
    """Every assignment in *tree* that binds a :data:`SANCTIONED_TIMEOUT_NAMES` name.

    ``ast.walk`` rather than module-level only, so the rule stays one sentence:
    ANY binding of a sanctioned ceiling name under this tree must carry the
    mirrored value.  A class-body binding really can reach a method decorator
    (decorators are evaluated in the class namespace), and a function-local one
    cannot -- but flagging that one anyway errs toward a LOUD false offender,
    and a local shadowing a tree-wide ceiling name at a different value is
    confusing enough to be worth the noise either way.
    """
    bindings: list[_Binding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        else:
            continue
        if value is None:
            continue
        bindings.extend(
            _Binding(
                module=module,
                name=target,
                lineno=node.lineno,
                expression=ast.unparse(value),
                seconds=_binding_value(value, _MIRROR_NAMESPACE),
            )
            for target in targets
            if target in SANCTIONED_TIMEOUT_NAMES
        )
    return bindings


class TestVerifyCliBudgetConstant:
    """``VERIFY_CLI_PER_TEST_TIMEOUT`` -- the budget the whole guard is built on.

    A MIRROR PIN, deliberately loud on a retune.  The sweep reads its band from
    the yaml, and :func:`test_the_verify_budget_is_configured` checks that band
    is non-empty; neither needs this constant.  What the pin protects is the
    CONSTANT's honesty, which other checks rest on: the
    ``faulthandler_timeout`` bound in TestTimeoutConstants, the deep-scene
    ``test_the_timeout_never_inverts_the_verify_budget`` floors, and every
    marker that spells the name.

    Its red is deliberate.  On an upward retune the sweep goes red anyway, over
    every site spelling this name, and its remedy names the spelling those
    sites already use; this is the one red test whose message names the fix.
    The ``--timeout`` lives in a tracked merge-verify config whose own verify
    runs this test, so a retune moves the constant in the same commit.
    """

    def test_the_inversion_band_is_non_empty(self) -> None:
        """The band ``(DELIBERATE_TIGHT_BOUND_CEILING, VERIFY_CLI_PER_TEST_TIMEOUT)`` must be real.

        The entire guard is the predicate "N sits strictly between the
        deliberate-tight-bound ceiling and the verify CLI budget".  If the two
        ever converge or invert, that predicate becomes unsatisfiable and every
        sweep below would pass VACUOUSLY -- green because nothing can offend,
        not because nothing does.  Asserted rather than assumed: the upper edge
        mirrors a config file that moves independently.

        The lower edge once mirrored the ini default and collapsed with it
        (64e24b547f); why it is now a literal:
        shared/src/shared/testing_timeout_markers.py::DELIBERATE_TIGHT_BOUND_CEILING.
        """
        assert DELIBERATE_TIGHT_BOUND_CEILING < VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'the inversion band ({DELIBERATE_TIGHT_BOUND_CEILING}, '
            f'{VERIFY_CLI_PER_TEST_TIMEOUT}) is empty -- the tight-bound '
            'ceiling has caught up with the verify CLI budget, so nothing can '
            'invert and this whole module would pass vacuously. Revisit it '
            'before changing either constant.'
        )

    def test_constant_mirrors_the_real_verify_test_command(self) -> None:
        """``VERIFY_CLI_PER_TEST_TIMEOUT`` must equal the ``--timeout`` verify really passes.

        THE DRIFT PIN.  The constant is deliberately a literal, which means
        nothing about the Python source keeps it honest.  THIS test is what
        does: it re-reads ``orchestrator/orchestrator.yaml``'s ``test_command``
        and extracts the ``--timeout`` token that pytest-timeout will actually
        see as ``config._env_timeout``.  If an operator retunes that flag, the
        sweep's band moves with it, because the sweep reads the yaml; what goes
        stale is the CONSTANT, and every marker and assertion built on it.
        This fails loudly so the constant moves too.
        """
        test_command = _verify_test_command()

        configured = verify_cli_timeout(test_command)
        assert configured is not None, (
            f'{ORCH_YAML} test_command carries no --timeout override '
            f'(got: {test_command!r}). VERIFY_CLI_PER_TEST_TIMEOUT models that '
            'flag, so without it the constant models nothing -- and every test '
            f'here silently falls back to the {PYPROJECT_DEFAULT_TIMEOUT}s '
            'pyproject default. This is also pinned from the other side by '
            'tests/scripts/test_fallback_verify_config.py::'
            'test_per_module_merge_verify_raises_per_test_timeout.'
        )
        assert configured == VERIFY_CLI_PER_TEST_TIMEOUT, (
            f'VERIFY_CLI_PER_TEST_TIMEOUT ({VERIFY_CLI_PER_TEST_TIMEOUT}) no '
            f'longer mirrors --timeout={configured} in {ORCH_YAML}. Update the '
            'constant in orchestrator/tests/_orch_helpers.py. Until then every '
            'marker spelled VERIFY_CLI_PER_TEST_TIMEOUT is judged at the stale '
            "value against the yaml's budget, so a raise reports them all as "
            'offenders, and the assertions built on the constant check a '
            'budget verify no longer passes. Why this red is deliberate: '
            "TestVerifyCliBudgetConstant's docstring."
        )


class TestSanctionedNameMirrors:
    """Every :data:`SANCTIONED_TIMEOUT_NAMES` value must match its REAL definition.

    THE EXECUTABLE LINK the mirror map would otherwise lack -- the same shape
    :class:`TestVerifyCliBudgetConstant` applies to the YAML, one anchor over.
    Without it the map is an unchecked claim about four constants defined
    elsewhere, and a wrong entry that reads too HIGH is silent: it resolves a
    marker to a safe number while the real constant sits inside the band, so
    the sweep stays green over a live inversion.

    THE CASE this closes:
    ``HEAVY_BARRIER_TEST_TIMEOUT`` is defined in the shared support module
    orchestrator/tests/_merge_worker_drive.py, not in ``_orch_helpers`` --
    lower it by hand from 540 to 225 and every site
    marked with it really pins 225s while this map still answers 540.  And
    because ``timeout_marker_sites`` matches on the TRAILING
    name only, a NEW file-local ``PYTEST_TIMEOUT = 120`` anywhere under this
    directory would resolve at 960.  Both are caught here, at their source.
    """

    def test_every_mirror_matches_its_real_definition(self, tree_pass: TreePass) -> None:
        """No binding of a sanctioned name may disagree with the mirrored value.

        Checked over EVERY binding in the tree, not just the one the map was
        written from, because the trailing-name match makes a second definition
        at a different value indistinguishable from the first.
        """
        wrong = [
            binding
            for binding in _marker_census(tree_pass).bindings
            if binding.seconds != SANCTIONED_TIMEOUT_NAMES[binding.name]
        ]

        assert not wrong, (
            f'{len(wrong)} binding(s) of a sanctioned timeout name disagree '
            'with SANCTIONED_TIMEOUT_NAMES in '
            '_sanctioned_timeout_names.py.\n\n'
            'That map is how a `@pytest.mark.timeout(NAME)` site is resolved, '
            'and it matches on the TRAILING name only. An entry reading HIGHER '
            'than the real constant fails SILENTLY: the site is waved through '
            'as safe while it really pins a value inside the inversion band '
            f'({DELIBERATE_TIGHT_BOUND_CEILING} < N < {VERIFY_CLI_PER_TEST_TIMEOUT}). '
            'Update the map to the real value -- and if the real value has '
            'moved INTO the band, fix the constant instead, not the mirror.\n\n'
            'A `seconds` of None means the definition left the literal '
            'arithmetic _binding_value understands, so the mirror can no '
            'longer be checked at all; give the constant a statically '
            'resolvable definition or drop it from the map.\n\n'
            + '\n'.join(
                f'  {b.module}:{b.lineno} {b.name} = {b.expression} -> '
                f'{b.seconds} (mirror says '
                f'{SANCTIONED_TIMEOUT_NAMES[b.name]})'
                for b in sorted(wrong, key=lambda b: (b.module, b.lineno))
            )
        )

    def test_every_mirrored_name_is_really_defined(self, tree_pass: TreePass) -> None:
        """The map may not outlive the constants it mirrors.

        An entry for a deleted constant is dead weight that silently
        re-sanctions the name if someone later reintroduces it at an arbitrary
        value.  It is also the
        anti-vacuity floor for the twin above, which passes trivially over an
        empty binding list.
        """
        defined = {binding.name for binding in _marker_census(tree_pass).bindings}
        undefined = sorted(
            set(SANCTIONED_TIMEOUT_NAMES) - SANCTIONED_NAMES_DEFINED_ELSEWHERE - defined
        )

        assert not undefined, (
            f'SANCTIONED_TIMEOUT_NAMES mirrors {undefined}, which no longer '
            f'name a constant defined anywhere under {TESTS_DIR}. Drop the '
            'entr(y/ies) -- a mirror for a deleted constant sanctions the bare '
            'NAME, so reintroducing it at any value at all would be resolved '
            'to the stale number instead of read as new.'
        )


# ---------------------------------------------------------------------------
# The tree-wide sweep.
# ---------------------------------------------------------------------------

#: Set below the 563 files and 148 marker sites MEASURED at authorship time,
#: so the guard survives the tree growing while still failing loudly if the
#: sweep itself ever breaks: a wrong TESTS_DIR, a read that silently yields
#: nothing, an extractor rotted to always-empty.  The house pattern for exactly
#: this risk: test_whole_tree_scan_timeout_guard.py::_MIN_EXPECTED_TEST_FILES,
#: test_marker_registration_drift.py::_MIN_EXPECTED_TEST_FILES.
_FLOORS = SweepFloors(test_files=400, marker_sites=100)


@dataclass(frozen=True)
class _ModuleMarkers:
    """One module's share of the marker census.

    ``readable`` is False when the module did not decode; a module that decoded
    but did not parse is examined with nothing extracted, exactly as
    ``shared.testing_timeout_markers.scan_python_tree`` treats it.
    """

    readable: bool
    sites: tuple[TimeoutSite, ...] = ()
    bindings: tuple[_Binding, ...] = ()


def _module_markers(module: ParsedModule) -> _ModuleMarkers:
    """A module naming neither the ``timeout`` marker nor a sanctioned constant has neither, unwalked."""
    try:
        if not module.may_name('timeout', *SANCTIONED_TIMEOUT_NAMES):
            return _ModuleMarkers(readable=True)
        tree = module.tree
    except (UnicodeDecodeError, OSError):
        return _ModuleMarkers(readable=False)
    except (SyntaxError, ValueError):
        return _ModuleMarkers(readable=True)
    return _ModuleMarkers(
        readable=True,
        sites=timeout_marker_sites(tree, SANCTIONED_TIMEOUT_NAMES),
        bindings=tuple(_sanctioned_bindings_in(tree, module.path.relative_to(TESTS_DIR).as_posix())),
    )


_MARKER_ANALYSIS = TreeAnalysis(
    name=f'{Path(__file__).stem}:marker_census', files=ORCHESTRATOR_TESTS, extract=_module_markers,
)
TREE_ANALYSES = (_MARKER_ANALYSIS,)


class _MarkerCensus(NamedTuple):
    """Every ``*.py`` under :data:`TESTS_DIR`, feeding two extractions.

    ``sites`` is the timeout-marker census, each site paired with its module --
    the path relative to :data:`TESTS_DIR` -- and carrying the sweep-health
    counters.  ``bindings`` is every assignment of a
    :data:`SANCTIONED_TIMEOUT_NAMES` name, for :class:`TestSanctionedNameMirrors`.
    """

    sites: TreeScan[tuple[str, TimeoutSite]]
    bindings: tuple[_Binding, ...]


def _marker_census(tree_pass: TreePass) -> _MarkerCensus:
    """This tests tree's marker census, projected from the shared tree pass (``_tree_scan.py``)."""
    by_module = {
        path.relative_to(TESTS_DIR).as_posix(): markers
        for path, markers in tree_pass.facts(_MARKER_ANALYSIS).items()
    }
    return _MarkerCensus(
        sites=TreeScan(
            root=TESTS_DIR,
            items=tuple(
                (module, site) for module, markers in by_module.items() for site in markers.sites
            ),
            examined=sum(markers.readable for markers in by_module.values()),
            unreadable=tuple(
                module for module, markers in by_module.items() if not markers.readable
            ),
        ),
        bindings=tuple(binding for markers in by_module.values() for binding in markers.bindings),
    )


def _verdict(tree_pass: TreePass) -> InversionVerdict:
    """:func:`_marker_census`'s census, judged against verify's budget with NO allowlist."""
    return judge_inversion_band(
        _marker_census(tree_pass).sites,
        verify_test_command=_verify_test_command(),
        grandfathered=frozenset(),
        floors=_FLOORS,
        slow_test_marker=(
            'from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT\n'
            '@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)   # slow test'
        ),
    )


def test_the_verify_budget_is_configured(tree_pass: TreePass) -> None:
    """The yaml's own ``--timeout`` must leave the band non-empty, read from the
    yaml and never from ``VERIFY_CLI_PER_TEST_TIMEOUT``.
    """
    failure = _verdict(tree_pass).budget_failure
    assert failure is None, failure


def test_no_timeout_marker_sits_in_the_inversion_band(tree_pass: TreePass) -> None:
    """No marker anywhere under this directory may sit in the inversion band.

    A marker at ``DELIBERATE_TIGHT_BOUND_CEILING < N <
    VERIFY_CLI_PER_TEST_TIMEOUT`` is too large to read as a deliberate tight
    bound, yet grants less than verify's budget would with no marker at all.
    Under
    ``timeout_method = "thread"`` a breach is not a red test: pytest-timeout
    ``os._exit()``s the xdist worker, ``--max-worker-restart=0`` declines to
    replace it, and the session is truncated with the blame landing on
    whatever innocent test shared the dead worker.  Three tasks (4176, 4384,
    4405) were failed that way by ONE such marker.
    """
    failure = _verdict(tree_pass).new_offender_failure
    assert failure is None, failure


def test_the_marker_census_is_not_vacuous(tree_pass: TreePass) -> None:
    """The sweep must find a substantial population of markers, in-band or not.

    Distinct from the file floor and load-bearing in a way it is not: a
    correct ``TESTS_DIR`` with an EXTRACTOR rotted to always-empty would
    examine 563 files, find zero sites, report zero offenders and pass. This is
    the floor that catches that. 148 sites measured at authorship time.
    """
    failure = _verdict(tree_pass).census_failure
    assert failure is None, failure


def test_the_deep_gate_module_pins_every_timeout_by_name(tree_pass: TreePass) -> None:
    """No bare number may pin a timeout in the deep merge-queue gate.

    WHY THIS MODULE GETS A RATCHET ITS NEIGHBOURS DO NOT.  All eight real-git
    classes in it carried an IDENTICAL bare ``300`` -- classes whose measured
    costs differ by more than 2x.  That uniformity is precisely what made
    TestRow7KillSwitchByteIdentity's under-sizing invisible: a reader had no
    way to see that one of the eight was roughly twice the weight of its
    neighbours, because the file said the same thing about all of them.

    A NAME fixes both halves of that.  It makes the next re-derivation ONE
    edit instead of eight, and it makes a divergent class conspicuous -- the
    file now states which classes are sized by the verify budget and which by
    their own measurement, in the marker itself rather than in a comment that
    can drift from it.

    A FLOOR, not a proof: this checks the SPELLING. The value behind a
    sanctioned name is held honest separately, by
    :class:`TestSanctionedNameMirrors`.
    """
    sites = [
        site for module, site in _marker_census(tree_pass).sites.items if module == _DEEP_GATE_MODULE
    ]

    assert len(sites) >= _MIN_DEEP_GATE_MARKER_SITES, (
        f'only {len(sites)} timeout marker site(s) found in '
        f'{_DEEP_GATE_MODULE} (expected at least '
        f'{_MIN_DEEP_GATE_MARKER_SITES}). Either the module was renamed -- '
        'update _DEEP_GATE_MODULE -- or its real-git classes lost their '
        'markers, which is the condition this sweep exists to prevent and '
        'would otherwise pass here VACUOUSLY, green because it found nothing '
        'rather than because it found nothing wrong.'
    )

    unnamed = sorted(
        (site for site in sites if site.spelling not in _DEEP_GATE_SPELLINGS),
        key=lambda site: site.lineno,
    )

    assert not unnamed, (
        f'{len(unnamed)} timeout marker(s) in {_DEEP_GATE_MODULE} pin a value '
        'that is not one of the sanctioned constants '
        f'{sorted(_DEEP_GATE_SPELLINGS)}.\n\n'
        'Every real-git class in that file once carried an identical bare '
        '300, and that uniformity is what hid TestRow7KillSwitchByteIdentity '
        'being roughly twice the weight of its neighbours until it had cost '
        'five recorded xdist worker crashes. Spelling the budget as a NAME '
        'makes a re-derivation one edit rather than eight, and makes a class '
        'whose budget genuinely differs conspicuous instead of invisible.\n\n'
        'Import the constant from _orch_helpers rather than writing the '
        'number: VERIFY_CLI_PER_TEST_TIMEOUT for a class sized by the verify '
        'budget, DEEP_GATE_SCENE_TEST_TIMEOUT for one sized by its own '
        'MEASURED spawn count. If a new class needs a third budget, measure '
        'it and add a named constant -- do not reach for a literal.\n\n'
        + '\n'.join(
            f'  {_DEEP_GATE_MODULE}:{site.lineno} {site.qualname} '
            f'({site.kind}) pins {site.spelling or "<no argument>"}'
            for site in unnamed
        )
    )
