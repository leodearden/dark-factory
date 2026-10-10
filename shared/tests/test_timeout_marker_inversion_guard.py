"""The timeout-marker inversion guard, instantiated for shared/tests.

The extractor, the band, the judgement, its failure messages and their
rationale live in shared/src/shared/testing_timeout_markers.py.  This module
supplies only this package's inputs: its tests root, the verify test_command
its own orchestrator.yaml passes, its grandfather allowlist and its
anti-vacuity floors.
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

#: No named timeout constant is shared across this package's modules, so a
#: marker spelled as a NAME reads as "no opinion".  The guard is a floor, with
#: the same stated limit as orchestrator's.
_SANCTIONED: Mapping[str, float] = {}

#: In-band sites that predate this guard: 1, MEASURED on 2026-10-07 at main
#: 1c6a52dbfd.  Entries may only ever be REMOVED; migration is follow-up
#: tkt_0RVDQ479918XDV0C80NXAVT8HX.  Keyed per SITE on the path RELATIVE to
#: tests/.  The site's own comment still reasons against a 60s ini default that
#: is now 540, which is why 180 reads to its author as a loosening.
_GRANDFATHERED: frozenset[tuple[str, str]] = frozenset({
    # test_root_config_integration_deselection.py -- 1 site at 180s
    ('test_root_config_integration_deselection.py', 'TestRootConfigIntegrationDeselection'),
})

#: Set below the 138 files and 38 sites measured on 2026-10-07.
_FLOORS = SweepFloors(test_files=100, marker_sites=25)


@functools.cache
def _verdict() -> InversionVerdict:
    return judge_inversion_band(
        scan_timeout_marker_sites(_TESTS_DIR, _SANCTIONED),
        verify_test_command=yaml.safe_load(_ORCH_YAML.read_text(encoding='utf-8'))['test_command'],
        grandfathered=_GRANDFATHERED,
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
