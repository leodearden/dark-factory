"""The timeout-marker inversion guard, instantiated for fused-memory/tests.

The extractor, the band, the failure messages and their rationale live in
shared/src/shared/testing_timeout_markers.py.  This module wires in this
package's tests root, the verify budget its own orchestrator.yaml passes, its
grandfather allowlist and its anti-vacuity floors.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import yaml
from shared.testing_timeout_markers import (
    DELIBERATE_TIGHT_BOUND_CEILING,
    GrandfatherRatchet,
    TimeoutSite,
    TreeScan,
    grandfather_ratchet,
    inversion_failure_message,
    inverts,
    scan_python_tree,
    stale_grandfather_message,
    timeout_marker_sites,
    verify_cli_timeout,
)

# One xdist worker pays for the cached sweep once, under -n auto --dist loadgroup.
pytestmark = pytest.mark.xdist_group('timeout_marker_inversion_guard')

_TESTS_DIR = Path(__file__).resolve().parent
_ORCH_YAML = _TESTS_DIR.parent / 'orchestrator.yaml'

#: No named timeout constant is shared across this package's modules, so a
#: marker spelled as a NAME reads as "no opinion".  The guard is a floor, with
#: the same stated limit as orchestrator's.
_SANCTIONED: Mapping[str, float] = {}

_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset()

#: Floors, not equalities (487 files and 60 sites measured on 2026-10-07): a
#: broken sweep reports zero offenders, which is indistinguishable from a
#: clean tree without them.
_MIN_EXPECTED_TEST_FILES = 300
_MIN_EXPECTED_MARKER_SITES = 40


def _verify_budget() -> int:
    test_command = yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))['test_command']
    budget = verify_cli_timeout(test_command)
    assert budget is not None, (
        f'{_ORCH_YAML} test_command carries no --timeout (got: {test_command!r}), '
        'so there is no verify budget for a marker to invert.'
    )
    return budget


def _sites_by_module(module: str, tree: ast.Module) -> Iterator[tuple[str, TimeoutSite]]:
    for site in timeout_marker_sites(tree, _SANCTIONED):
        yield module, site


@functools.cache
def _scan() -> TreeScan[tuple[str, TimeoutSite]]:
    return scan_python_tree(_TESTS_DIR, _sites_by_module)


def _assert_the_sweep_read_the_tree(scan: TreeScan[tuple[str, TimeoutSite]]) -> None:
    assert scan.examined >= _MIN_EXPECTED_TEST_FILES, (
        f'only {scan.examined} .py files examined under {_TESTS_DIR} (expected '
        f'at least {_MIN_EXPECTED_TEST_FILES}; {len(scan.unreadable)} skipped '
        f'as unreadable: {sorted(scan.unreadable)}) -- the sweep itself is '
        'broken, so this guard would pass vacuously rather than because the '
        'tree is clean.'
    )


def _ratchet() -> GrandfatherRatchet:
    budget = _verify_budget()
    in_band = [pair for pair in _scan().items if inverts(pair[1].seconds, verify_cli_budget=budget)]
    return grandfather_ratchet(in_band, _GRANDFATHERED)


def test_the_verify_budget_is_configured() -> None:
    budget = _verify_budget()

    assert budget > DELIBERATE_TIGHT_BOUND_CEILING, (
        f'{_ORCH_YAML} passes --timeout={budget}, which leaves the inversion '
        f'band ({DELIBERATE_TIGHT_BOUND_CEILING}, {budget}) empty, so this '
        'guard would pass vacuously.'
    )


def test_no_new_timeout_marker_sits_in_the_inversion_band() -> None:
    _assert_the_sweep_read_the_tree(_scan())
    budget = _verify_budget()

    new_offenders = _ratchet().new_offenders

    assert not new_offenders, (
        inversion_failure_message(
            new_offenders,
            verify_cli_budget=budget,
            slow_test_marker=(
                f'@pytest.mark.timeout({budget})   # slow test -- or drop the '
                'marker and take the ambient budget'
            ),
        )
        + '\n\n_GRANDFATHERED only ever shrinks -- do not add yours to it.'
    )


def test_grandfather_allowlist_has_no_stale_entries() -> None:
    _assert_the_sweep_read_the_tree(_scan())

    stale = _ratchet().stale

    assert not stale, stale_grandfather_message(stale)


def test_the_marker_census_is_not_vacuous() -> None:
    scan = _scan()

    assert len(scan.items) >= _MIN_EXPECTED_MARKER_SITES, (
        f'only {len(scan.items)} timeout marker site(s) found across '
        f'{scan.examined} files (expected at least {_MIN_EXPECTED_MARKER_SITES}) '
        '-- timeout_marker_sites has probably stopped matching, so the sweep '
        'would pass vacuously.'
    )
