"""The timeout-marker inversion guard, instantiated for scripts/tests.

The extractor, the band, the judgement, its failure messages and their
rationale live in shared/src/shared/testing_timeout_markers.py.  This module
supplies only this tree's inputs: its tests root, the verify test_command
scripts/orchestrator.yaml passes, the only registered config that collects this
tree, and its anti-vacuity floors.  There is no allowlist.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path

import yaml
from shared.testing_timeout_markers import (
    InversionVerdict,
    SweepFloors,
    judge_inversion_band,
    scan_timeout_marker_sites,
)

_TESTS_DIR = Path(__file__).resolve().parent
_ORCH_YAML = _TESTS_DIR.parent / 'orchestrator.yaml'

#: No named timeout constant is shared across this tree's modules, so a
#: marker spelled as a NAME reads as "no opinion".  The guard is a floor, with
#: the same stated limit as orchestrator's.
_SANCTIONED: Mapping[str, float] = {}

#: Set below the 194 files and 5 sites measured on 2026-10-30.
_FLOORS = SweepFloors(test_files=140, marker_sites=3)


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


def test_grandfather_allowlist_has_no_stale_entries() -> None:
    failure = _verdict().stale_failure
    assert failure is None, failure


def test_the_marker_census_is_not_vacuous() -> None:
    failure = _verdict().census_failure
    assert failure is None, failure
