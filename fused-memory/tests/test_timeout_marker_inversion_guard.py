"""The timeout-marker inversion guard, instantiated for fused-memory/tests.

The extractor, the band, the judgement, its failure messages and their
rationale live in shared/src/shared/testing_timeout_markers.py.  This module
supplies only this package's inputs: its tests root, the verify test_command
its own orchestrator.yaml passes and its anti-vacuity floors.  It judges with
NO allowlist, so every in-band marker under this directory fails.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from shared.testing_timeout_markers import (
    InversionVerdict,
    SweepFloors,
    judge_inversion_band,
    scan_timeout_marker_sites,
)
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TIMEOUT_NAME,
)

# One xdist worker pays for the cached sweep once, under -n auto --dist loadgroup.
pytestmark = [
    pytest.mark.xdist_group('timeout_marker_inversion_guard'),
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
]

_TESTS_DIR = Path(__file__).resolve().parent
_ORCH_YAML = _TESTS_DIR.parent / 'orchestrator.yaml'

#: The one named timeout constant this package's modules share: the
#: whole-tree guard family's ceiling, resolved from its home.  Any other
#: marker spelled as a NAME reads as "no opinion".  The guard is a floor, with
#: the same stated limit as orchestrator's.
_SANCTIONED: Mapping[str, float] = {WHOLE_TREE_SCAN_TIMEOUT_NAME: WHOLE_TREE_SCAN_TEST_TIMEOUT}

#: Set below the 487 files and 60 sites measured on 2026-10-07.
_FLOORS = SweepFloors(test_files=300, marker_sites=40)


@functools.cache
def _verdict() -> InversionVerdict:
    return judge_inversion_band(
        scan_timeout_marker_sites(_TESTS_DIR, _SANCTIONED),
        verify_test_command=yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))['test_command'],
        grandfathered=frozenset(),
        floors=_FLOORS,
    )


def test_the_verify_budget_is_configured() -> None:
    failure = _verdict().budget_failure
    assert failure is None, failure


def test_no_new_timeout_marker_sits_in_the_inversion_band() -> None:
    failure = _verdict().new_offender_failure
    assert failure is None, failure


def test_the_marker_census_is_not_vacuous() -> None:
    failure = _verdict().census_failure
    assert failure is None, failure
