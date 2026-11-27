"""The per-test timeout ceilings that rest on a measured worst case, and the worst case each rests on.

A CEILING ANCHOR pairs a per-test timeout ceiling with the worst per-test
wall clock (setup + call + teardown, what a timeout mark bounds) recorded for
the tests it bounds, under real load.  The ceiling is justified while
:func:`required_ceiling_secs` of that worst stays at or below it.

An anchor's POPULATION is the tests whose CLOSEST timeout mark (see
``shared.testing_timeout_markers.closest_timeout_site``) is spelled through a
named constant, or that carry no timeout argument at all and so run under the
ini default.  Membership is decided by the spelling, never by the value: a
literal equal to a ceiling today parts from it on the next retune.

This module is the code home of the recorded worst cases.  The guard tests
import them, and ``scripts/timeout_headroom_selfcheck.py`` re-measures them
against the archived verify junits every night.  Ceilings already homed
elsewhere are read, not restated: the family ceiling from
``shared.whole_tree_guard_family`` and the ini default from this package's
``pyproject.toml``.  Corpus and derivations:
``plans/pytest-per-test-timeout-measurement-2026-09-17.md``.

Pure stdlib plus ``shared``, registered in ``shared/tests/test_pure_stdlib_leaves.py``.
Importing it does no I/O.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.testing_timeout_headroom import UNDER_LOAD_HEADROOM_FACTOR
from shared.testing_timeout_markers import TimeoutSite
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TIMEOUT_NAME,
)

__all__ = [
    'INI_DEFAULT_MEASURED_WORST_SECS',
    'MAGICMOCK_REPO_SCAN_MEASURED_WORST_SECS',
    'MAGICMOCK_REPO_SCAN_TEST_TIMEOUT',
    'WHOLE_TREE_FAMILY_MEASURED_WORST_SECS',
    'CeilingAnchor',
    'Population',
    'PopulationKind',
    'registered_anchors',
    'required_ceiling_secs',
]

# Whole-tree guard family (every module marked with WHOLE_TREE_SCAN_TEST_TIMEOUT):
# test_merge_lane_ratchet.py::TestLanePatchTargets::test_real_tree_union_anchor,
# almost all of it module-fixture setup, in task 3165's orchestrator verify
# (2026-10-08, `-n 8`, loadavg 64 -> 79).  The worst run is kept, never the
# freshest; a figure measured on code since replaced retires with that code.
# Task 6556 addendum of the corpus; task 9396 owns family membership and sizing.
WHOLE_TREE_FAMILY_MEASURED_WORST_SECS = 148.65

# fused-memory/tests/test_check_bare_magicmock_config.py's two repo-wide scan
# classes, which sweep all seven packages' tests/: 61.52 x 8 = 492.16, rounded up
# to a whole minute.  Root-bound worst under real load, task 5442; task 5581
# addendum of the corpus.
MAGICMOCK_REPO_SCAN_TEST_TIMEOUT = 540
MAGICMOCK_REPO_SCAN_MEASURED_WORST_SECS = 61.52

# Unmarked tests, which run under [tool.pytest.ini_options].timeout:
# orchestrator/tests/test_hard_v2_fixture_pool.py::TestMintedPool::
# test_reference_unavailable_only_when_no_landing_merge_exists, root-bound,
# across every suite and both inifile-resolution modes.  Task 5581 addendum.
INI_DEFAULT_MEASURED_WORST_SECS = 45.48

_SHARED_PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'
_THIS_MODULE = 'shared/src/shared/testing_ceiling_anchors.py'


def required_ceiling_secs(worst_secs: float) -> int:
    """The ceiling *worst_secs* justifies: times the headroom factor, rounded up to a whole minute."""
    return math.ceil(worst_secs * UNDER_LOAD_HEADROOM_FACTOR / 60) * 60


class PopulationKind(StrEnum):
    """How an anchor names the tests it bounds."""

    SPELLED_THROUGH = 'spelled_through'
    UNMARKED = 'unmarked'


@dataclass(frozen=True)
class Population:
    """The tests an anchor bounds, decided by each test's closest timeout mark.

    SPELLED_THROUGH names the constant the closest mark's argument is spelled
    through; UNMARKED names none, and admits a test with no closest mark or one
    with no timeout argument, both of which fall back to the ini default.
    """

    kind: PopulationKind
    constant: str | None = None

    def __post_init__(self) -> None:
        if self.kind is PopulationKind.SPELLED_THROUGH and not self.constant:
            raise ValueError(
                f'a {self.kind} population names the constant its marks are spelled '
                f'through; got constant={self.constant!r}'
            )
        if self.kind is PopulationKind.UNMARKED and self.constant is not None:
            raise ValueError(
                f'an {self.kind} population names no constant; got constant={self.constant!r}'
            )

    def admits(self, closest: TimeoutSite | None) -> bool:
        """True iff a test whose closest timeout mark is *closest* belongs to this population."""
        if self.kind is PopulationKind.UNMARKED:
            return closest is None or closest.spelling == ''
        return closest is not None and closest.constant == self.constant


@dataclass(frozen=True)
class CeilingAnchor:
    """A per-test ceiling, its population, and the recorded worst case that justifies it.

    ``provenance`` is a ``path::symbol`` citation of where the recorded worst
    lives, for display only.
    """

    anchor_id: str
    ceiling_secs: float
    measured_worst_secs: float
    population: Population
    provenance: str

    def __post_init__(self) -> None:
        if self.ceiling_secs <= 0 or self.measured_worst_secs <= 0:
            raise ValueError(
                f'anchor {self.anchor_id!r}: a ceiling and its recorded worst are positive '
                f'seconds; got ceiling_secs={self.ceiling_secs}, '
                f'measured_worst_secs={self.measured_worst_secs}'
            )

    @property
    def admissible_worst_secs(self) -> float:
        """The largest per-test worst the ceiling still clears by the headroom factor."""
        return self.ceiling_secs / UNDER_LOAD_HEADROOM_FACTOR


def _ini_default_timeout() -> float:
    config = tomllib.loads(_SHARED_PYPROJECT.read_text(encoding='utf-8'))
    return float(config['tool']['pytest']['ini_options']['timeout'])


def registered_anchors() -> tuple[CeilingAnchor, ...]:
    """Every registered anchor.  Reads the ini default from ``shared/pyproject.toml`` on each call."""
    return (
        CeilingAnchor(
            anchor_id='whole_tree_family',
            ceiling_secs=WHOLE_TREE_SCAN_TEST_TIMEOUT,
            measured_worst_secs=WHOLE_TREE_FAMILY_MEASURED_WORST_SECS,
            population=Population(
                PopulationKind.SPELLED_THROUGH, constant=WHOLE_TREE_SCAN_TIMEOUT_NAME
            ),
            provenance=f'{_THIS_MODULE}::WHOLE_TREE_FAMILY_MEASURED_WORST_SECS',
        ),
        CeilingAnchor(
            anchor_id='bare_magicmock_repo_scan',
            ceiling_secs=MAGICMOCK_REPO_SCAN_TEST_TIMEOUT,
            measured_worst_secs=MAGICMOCK_REPO_SCAN_MEASURED_WORST_SECS,
            population=Population(
                PopulationKind.SPELLED_THROUGH, constant='MAGICMOCK_REPO_SCAN_TEST_TIMEOUT'
            ),
            provenance=f'{_THIS_MODULE}::MAGICMOCK_REPO_SCAN_MEASURED_WORST_SECS',
        ),
        CeilingAnchor(
            anchor_id='ini_default',
            ceiling_secs=_ini_default_timeout(),
            measured_worst_secs=INI_DEFAULT_MEASURED_WORST_SECS,
            population=Population(PopulationKind.UNMARKED),
            provenance=f'{_THIS_MODULE}::INI_DEFAULT_MEASURED_WORST_SECS',
        ),
    )
